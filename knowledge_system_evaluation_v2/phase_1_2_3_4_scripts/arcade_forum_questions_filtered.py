import argparse
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup


API_BASE = "https://api.stackexchange.com/2.3"


def html_to_text(raw_html: str) -> str:
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")

    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text(' ', strip=True)}`")

    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def unix_to_iso(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_champions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected the champions JSON to be a dictionary.")

    items = data.get("known_champions")

    if not isinstance(items, list):
        raise ValueError("Expected a 'known_champions' list in the JSON file.")

    champions = []

    for item in items:
        if not isinstance(item, dict):
            continue

        name = item.get("champion")

        if isinstance(name, str) and name.strip():
            champions.append(name.strip())

    if not champions:
        raise ValueError(f"No champion names found in {path}")

    return sorted(set(champions), key=len, reverse=True)

def build_champion_matcher(champions: list[str]):
    raw_patterns = [
        (
            name,
            re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            ),
        )
        for name in champions
    ]

    normalized_names = [
        (name, normalize_for_match(name))
        for name in champions
    ]

    def match(text: str) -> list[str]:
        matches = set()

        for name, pattern in raw_patterns:
            if pattern.search(text):
                matches.add(name)

        normalized_text = f" {normalize_for_match(text)} "
        for name, normalized_name in normalized_names:
            if normalized_name and f" {normalized_name} " in normalized_text:
                matches.add(name)

        return sorted(matches)

    return match


def api_get(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    delay: float = 0.15,
) -> dict[str, Any]:
    key = os.getenv("STACKEXCHANGE_KEY")
    if key:
        params["key"] = key

    url = f"{API_BASE}/{endpoint}"
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "error_id" in data:
        raise RuntimeError(
            f"Stack Exchange API error {data.get('error_id')}: "
            f"{data.get('error_name')} - {data.get('error_message')}"
        )

    if "backoff" in data:
        print(f"  [API] Backoff received: sleeping for {data['backoff']} seconds...")
        time.sleep(int(data["backoff"]) + 1)
    else:
        time.sleep(delay)

    if "quota_remaining" in data:
        print(f"  [API] Quota remaining: {data['quota_remaining']} / {data.get('quota_max', '???')}")

    return data


def fetch_questions(
    site: str,
    tag: str,
    sort: str,
    only_edited: bool,
    max_pages: int | None,
) -> list[dict[str, Any]]:
    session = requests.Session()
    page = 1
    questions = []

    while True:
        data = api_get(
            session,
            "questions",
            {
                "site": site,
                "tagged": tag,
                "sort": sort,
                "order": "desc",
                "page": page,
                "pagesize": 100,
                "filter": "withbody",
            },
        )

        for q in data.get("items", []):
            if "accepted_answer_id" not in q:
                continue

            if only_edited and "last_edit_date" not in q:
                continue

            questions.append(q)

        print(f"Fetched page {page}, total accepted-answer questions so far: {len(questions)}")

        if not data.get("has_more"):
            break

        page += 1

        if max_pages is not None and page > max_pages:
            break

    return questions


def chunks(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_answers(
    site: str,
    answer_ids: list[int],
) -> dict[int, dict[str, Any]]:
    session = requests.Session()
    answers_by_id = {}

    for batch in chunks(answer_ids, 100):
        ids = ";".join(str(x) for x in batch)

        data = api_get(
            session,
            f"answers/{ids}",
            {
                "site": site,
                "sort": "activity",
                "order": "desc",
                "pagesize": 100,
                "filter": "withbody",
            },
        )

        for answer in data.get("items", []):
            answers_by_id[answer["answer_id"]] = answer

        print(f"Fetched accepted answers: {len(answers_by_id)}/{len(answer_ids)}")

    return answers_by_id


def build_dataset(
    questions: list[dict[str, Any]],
    answers_by_id: dict[int, dict[str, Any]],
    match_champions,
) -> list[dict[str, Any]]:
    dataset = []

    for q in questions:
        title = html.unescape(q.get("title", ""))
        body_text = html_to_text(q.get("body", ""))

        # Champion matching is STRICTLY limited to question title, body, and tags.
        searchable_text = "\n".join([
            title,
            body_text,
            " ".join(q.get("tags", [])),
        ])

        champion_matches = match_champions(searchable_text)

        if not champion_matches:
            continue

        accepted_answer_id = q.get("accepted_answer_id")
        accepted_answer = answers_by_id.get(accepted_answer_id)

        # Only include if the answer was successfully fetched and exists.
        if not accepted_answer:
            continue

        answer_text = html_to_text(accepted_answer.get("body", ""))

        dataset.append({
            "source": "arqade",
            "question_id": q.get("question_id"),
            "accepted_answer_id": accepted_answer_id,
            "question_url": q.get("link"),
            "question_title": title,
            "question_body": body_text,
            "answer_gold": answer_text,
            "answer_gold_score": accepted_answer.get("score"),
            "champion_matches": champion_matches,
            "question_score": q.get("score"),
            "answer_count": q.get("answer_count"),
            "tags": q.get("tags", []),
            "creation_date": unix_to_iso(q.get("creation_date")),
            "last_activity_date": unix_to_iso(q.get("last_activity_date")),
        })

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--champions", default="system_known_champions_v2.json")
    parser.add_argument("--out", default="gaming_lol_questions_with_gold_answers.json")
    parser.add_argument("--site", default="gaming")
    parser.add_argument("--tag", default="league-of-legends")
    parser.add_argument("--sort", default="activity")
    parser.add_argument("--only-edited", action="store_true")
    parser.add_argument("--max-pages", type=int, default=None)

    args = parser.parse_args()

    champions = load_champions(args.champions)
    match_champions = build_champion_matcher(champions)

    print(f"Loaded {len(champions)} champions")

    questions = fetch_questions(
        site=args.site,
        tag=args.tag,
        sort=args.sort,
        only_edited=args.only_edited,
        max_pages=args.max_pages,
    )

    answer_ids = sorted({
        q["accepted_answer_id"]
        for q in questions
        if "accepted_answer_id" in q
    })

    answers_by_id = fetch_answers(args.site, answer_ids)

    dataset = build_dataset(
        questions=questions,
        answers_by_id=answers_by_id,
        match_champions=match_champions,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataset)} filtered Q&A pairs to {args.out}")


if __name__ == "__main__":
    main()