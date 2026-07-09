import argparse
import json
import re
import random
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from camoufox.sync_api import Camoufox

BASE_URL = "https://www.mobafire.com"

LIST_URL = (
    "https://www.mobafire.com/league-of-legends/questions"
    "?category=&sort_type=create_ts&sort_order=desc&page={page}"
)

MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)

ANSWER_AUTHOR_RE = re.compile(
    rf"\|\s+(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}",
    re.IGNORECASE,
)

class BrowserFetcher:
    def __init__(self, delay: float = 1.0, headless: bool = False):
        self.delay = delay
        self.headless = headless
        self.camoufox_context = None
        self.browser = None
        self.page = None

    def __enter__(self):
        # 1. Initialize the Camoufox object
        self.camoufox_context = Camoufox(
            headless=self.headless,
            geoip=True,    
            humanize=True  
        )
        
        # 2. Enter the context manager manually to get the actual Playwright Browser instance
        self.browser = self.camoufox_context.__enter__()
        
        # 3. Now you can safely call new_page()
        self.page = self.browser.new_page()
        return self

    def fetch(self, url: str) -> str:
        self.page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # Randomize the delay slightly to appear more human when looping
        actual_delay = self.delay * random.uniform(0.8, 1.5)
        self.page.wait_for_timeout(int(actual_delay * 1000))

        return self.page.content()

    def wait_for_manual_check(self, url: str):
        self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
        
        # Give Camoufox 5 seconds to automatically clear the Turnstile check
        print("\nLoading page. Allowing Camoufox to bypass any active Turnstile challenges...")
        self.page.wait_for_timeout(5000)
        
        html = self.page.content()
        if is_human_verification_page(html):
            print("Challenge did not clear automatically. If running headful, complete it manually.")
            input("Press ENTER when the real page is visible...")
            self.page.wait_for_timeout(3000)
        else:
            print("Turnstile bypassed successfully.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Forward the exit signal back to Camoufox to ensure a clean teardown of Playwright
        if self.camoufox_context:
            self.camoufox_context.__exit__(exc_type, exc_val, exc_tb)


def load_champions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected champions JSON to be a dictionary.")

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


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_question_id(url: str) -> str | None:
    path = urlparse(url).path.rstrip("/")
    match = re.search(r"/question/[^/]*-(\d+)$", path)
    return match.group(1) if match else None


def extract_question_links_from_list_page(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if re.search(r"/league-of-legends/question/[^?#]+-\d+/?$", href):
            full_url = urljoin(BASE_URL, href).rstrip("/")
            links.append(full_url)

    return sorted(set(links))


def detect_last_page(html: str) -> int | None:
    soup = BeautifulSoup(html, "html.parser")
    pages = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        match = re.search(r"[?&]page=(\d+)", href)

        if match:
            pages.append(int(match.group(1)))

    return max(pages) if pages else None


def text_lines_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    lines = [clean_text(line) for line in text.splitlines()]
    return [line for line in lines if line]


def extract_title(lines: list[str], html: str) -> str | None:
    for line in lines:
        if "League of Legends" in line and "Question:" in line:
            return line.split("Question:", 1)[1].strip()

    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")

    if h1:
        raw_title = clean_text(h1.get_text(" ", strip=True))

        if "Question:" in raw_title:
            return raw_title.split("Question:", 1)[1].strip()

        return raw_title

    return None


def find_question_body(lines: list[str], title: str) -> tuple[str, int | None]:
    body_title_idx = None

    for i, line in enumerate(lines):
        if line.strip().lower() == title.strip().lower():
            body_title_idx = i
            break

    search_start = body_title_idx if body_title_idx is not None else 0

    asked_idx = None
    for i in range(search_start, len(lines)):
        if line_looks_like_asked_metadata(lines[i]):
            asked_idx = i
            break

    answers_idx = None
    for i in range(search_start, len(lines)):
        if re.match(r"Answers\s*\((\d+)\)", lines[i]):
            answers_idx = i
            break

    if asked_idx is None or answers_idx is None or answers_idx <= asked_idx:
        return "", answers_idx

    body_lines = lines[asked_idx + 1:answers_idx]
    body = clean_text("\n".join(body_lines))

    return body, answers_idx


def line_looks_like_asked_metadata(line: str) -> bool:
    return (
        line.startswith("Asked by ")
        or line.startswith("Asked on ")
        or " asked " in line.lower()
    )


def parse_answer_count(line: str) -> int | None:
    match = re.match(r"Answers\s*\((\d+)\)", line)

    if not match:
        return None

    return int(match.group(1))


def should_stop_answer_section(line: str) -> bool:
    stop_phrases = [
        "Loading Comments",
        "Load More Comments",
        "Trending Build Guides",
        "Create Your Champion Build Guide",
        "New Guide Authors",
        "League of Legends Champions:",
        "#MOBAFire",
        "MOBAFire Network",
        "MFN",
        "About Us",
        "Contact Us",
    ]

    return any(phrase.lower() in line.lower() for phrase in stop_phrases)


def remove_answer_noise(lines: list[str]) -> list[str]:
    cleaned = []

    noise_exact = {
        "reply",
        "quote",
        "report",
        "delete",
        "edit",
        "upvote",
        "downvote",
        "vote",
        "comments",
        "comment",
        "show more",
        "show less",
    }

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.lower() in noise_exact:
            continue

        if re.fullmatch(r"[+-]?\d+", stripped):
            continue

        cleaned.append(stripped)

    return cleaned


def extract_score_before_author(answer_section: list[str], author_pos: int) -> int:
    for j in range(author_pos - 1, max(-1, author_pos - 5), -1):
        possible_score = answer_section[j].strip()

        if re.fullmatch(r"[+-]?\d+", possible_score):
            return int(possible_score)

    return 0


def parse_answers(lines: list[str], answers_idx: int) -> tuple[int | None, list[dict]]:
    answer_count = parse_answer_count(lines[answers_idx])

    raw_section = []

    for line in lines[answers_idx + 1:]:
        if should_stop_answer_section(line):
            break

        raw_section.append(line)

    author_positions = [
        i for i, line in enumerate(raw_section)
        if ANSWER_AUTHOR_RE.search(line)
    ]

    answers = []

    for pos_i, author_pos in enumerate(author_positions):
        start = author_pos + 1
        end = (
            author_positions[pos_i + 1]
            if pos_i + 1 < len(author_positions)
            else len(raw_section)
        )

        body_lines = raw_section[start:end]
        body_lines = remove_answer_noise(body_lines)
        answer_text = clean_text("\n".join(body_lines))

        if not answer_text:
            continue

        score = extract_score_before_author(raw_section, author_pos)

        answers.append({
            "rank_on_page": len(answers) + 1,
            "author_line": raw_section[author_pos],
            "score": score,
            "answer_text": answer_text,
            "word_count": len(answer_text.split()),
        })

    return answer_count, answers


def parse_question_page(html: str, url: str) -> dict | None:
    if is_human_verification_page(html):
        return None

    soup = BeautifulSoup(html, "html.parser")

    question_el = soup.select_one("li.question-list__item--main")

    if question_el is None:
        return None

    title_el = question_el.select_one("h4")
    title = soup_text(title_el)

    if not title:
        h1 = soup.select_one(".col-left h1")
        title = soup_text(h1)
        if "Question:" in title:
            title = clean_text(title.split("Question:", 1)[1])

    if not title:
        return None

    body_el = question_el.select_one(".copy")
    question_body = soup_text(body_el)

    answer_count_reported = None
    h2 = soup.find("h2", string=re.compile(r"Answers\s*\(\d+\)", re.I))

    if h2:
        match = re.search(r"Answers\s*\((\d+)\)", soup_text(h2), re.I)
        if match:
            answer_count_reported = int(match.group(1))

    page_tags = [
        soup_text(tag)
        for tag in soup.select("a.tag")
        if soup_text(tag)
    ]

    answers = parse_answers_from_soup(soup)

    return {
        "source": "mobafire",
        "question_id": extract_question_id(url),
        "question_url": url,
        "question_title": title,
        "question_body": question_body,
        "question_tags": page_tags,
        "answer_count_reported": answer_count_reported,
        "answers": answers,
    }


def choose_best_valid_answer(
    answers: list[dict],
    min_score: int,
    min_words: int,
) -> tuple[dict | None, list[dict]]:
    valid_answers = [
        answer for answer in answers
        if answer.get("score", 0) >= min_score
        and answer.get("word_count", 0) >= min_words
        and answer.get("external_link_count", 0) == 0
    ]

    if not valid_answers:
        return None, []

    best_answer = sorted(
        valid_answers,
        key=lambda a: (
            a.get("score", 0),
            a.get("word_count", 0),
            len(a.get("answer_text", "")),
        ),
        reverse=True,
    )[0]

    return best_answer, valid_answers


def write_json(path: str, data: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_human_verification_page(html: str) -> bool:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()

    markers = [
        "verify you are human",
        "verifying you are human",
        "checking your browser",
        "human verification",
        "just a moment",
        "cloudflare",
    ]

    return any(marker in text for marker in markers)


def fetch_real_page_or_pause(fetcher, url: str) -> str | None:
    html = fetcher.fetch(url)

    if is_human_verification_page(html):
        print(f"\nHuman verification detected for: {url}")
        print("Allowing Camoufox to bypass...")
        fetcher.page.wait_for_timeout(5000)
        
        html = fetcher.page.content()

        if is_human_verification_page(html):
            print("Still on human verification page. Skipping this URL.")
            return None

    return html


def soup_text(el) -> str:
    if el is None:
        return ""
    return clean_text(el.get_text(" ", strip=True))


def count_external_links(el) -> int:
    if el is None:
        return 0

    count = 0
    for a in el.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and "mobafire.com" not in href:
            count += 1

    return count


def parse_answers_from_soup(soup: BeautifulSoup) -> list[dict]:
    answers = []

    for comment in soup.select("div.comment"):
        vote_el = comment.select_one(".vote span")
        content_el = comment.select_one(".content.expand-toggle")
        author_el = comment.select_one(".info a.user-level")
        date_el = comment.select_one(".info .date")

        answer_text = soup_text(content_el)

        if not answer_text:
            continue

        try:
            score = int(soup_text(vote_el))
        except ValueError:
            score = 0

        author = soup_text(author_el)
        date = soup_text(date_el)

        answers.append({
            "rank_on_page": len(answers) + 1,
            "author": author,
            "date": date,
            "author_line": f"{author} | {date}".strip(" |"),
            "score": score,
            "answer_text": answer_text,
            "word_count": len(answer_text.split()),
            "external_link_count": count_external_links(content_el),
        })

    return answers


def extract_candidate_questions_from_list_page(html: str, match_champions) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    for item in soup.select("li.question-list__item"):
        link_el = item.select_one("a[href*='/league-of-legends/question/']")
        title_el = item.select_one(".question-list__item__info h4")
        info_el = item.select_one(".question-list__item__info")
        score_el = item.select_one(".question-list__item__meta__vote label")
        answers_el = item.select_one(".question-list__item__meta__vote span")

        if not link_el or not title_el or not info_el:
            continue

        href = link_el.get("href", "")
        question_url = urljoin(BASE_URL, href).rstrip("/")

        title = soup_text(title_el)

        preview_parts = []
        for p in info_el.find_all("p"):
            classes = p.get("class", [])
            if "byline" in classes:
                continue
            text = soup_text(p)
            if text:
                preview_parts.append(text)

        preview = clean_text("\n".join(preview_parts))

        try:
            question_score = int(soup_text(score_el))
        except ValueError:
            question_score = 0

        answers_text = soup_text(answers_el)
        match = re.search(r"(\d+)\s+answers?", answers_text, re.I)
        answer_count = int(match.group(1)) if match else 0

        if answer_count <= 0:
            continue

        champion_search_text = "\n".join([title, preview])
        champion_matches = match_champions(champion_search_text)

        if not champion_matches:
            continue

        candidates.append({
            "question_id": extract_question_id(question_url),
            "question_url": question_url,
            "list_title": title,
            "list_preview": preview,
            "list_question_score": question_score,
            "list_answer_count": answer_count,
            "list_champion_matches": champion_matches,
        })

    return candidates


def scrape_mobafire_questions(
    champions_path: str,
    output_path: str,
    max_pages: int | None,
    delay: float,
    min_answer_score: int,
    min_answer_words: int,
    match_answers_too: bool,
    headless: bool = False,
):
    champions = load_champions(champions_path)
    match_champions = build_champion_matcher(champions)

    print(f"Loaded {len(champions)} known champions")

    dataset = []
    skipped_no_parse = 0
    skipped_no_answers = 0
    skipped_no_valid_answer = 0
    skipped_no_champion_match = 0

    with BrowserFetcher(delay=delay, headless=headless) as fetcher:
        
        first_url = LIST_URL.format(page=1)

        fetcher.wait_for_manual_check(first_url)

        first_html = fetcher.page.content()

        Path("debug_mobafire_page1.html").write_text(first_html, encoding="utf-8")

        detected_last_page = detect_last_page(first_html)

        if max_pages is not None:
            last_page = max_pages
        elif detected_last_page is not None:
            last_page = detected_last_page
        else:
            last_page = 1

        print(f"Scraping list pages 1 to {last_page}")

        question_urls = []

        for page in range(1, last_page + 1):
            if page == 1:
                html = first_html
            else:
                html = fetch_real_page_or_pause(fetcher, LIST_URL.format(page=page))

                if html is None:
                    print(f"Page {page}: skipped because verification did not clear")
                    continue

            candidates = extract_candidate_questions_from_list_page(html, match_champions)
            question_urls.extend([c["question_url"] for c in candidates])

            print(
                f"Page {page}: found {len(candidates)} candidate question links "
                f"after answer/champion prefilter"
            )

        question_urls = sorted(set(question_urls))

        print(f"Total unique question URLs: {len(question_urls)}")
        print(f"Valid answer policy: score >= {min_answer_score}, words >= {min_answer_words}")

        for idx, url in enumerate(question_urls, start=1):
            try:
                html = fetch_real_page_or_pause(fetcher, url)

                if html is None:
                    skipped_no_parse += 1
                    continue

                parsed = parse_question_page(html, url)

                if not parsed:
                    skipped_no_parse += 1
                    debug_path = Path("debug_failed_pages") / f"{idx}_{extract_question_id(url) or 'unknown'}.html"
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    debug_path.write_text(html, encoding="utf-8")

                    print(f"[{idx}/{len(question_urls)}] Could not parse: {url}")
                    print(f"Saved debug HTML to {debug_path}")
                    continue

                if not parsed["answers"]:
                    skipped_no_answers += 1
                    continue

                best_answer, valid_answers = choose_best_valid_answer(
                    answers=parsed["answers"],
                    min_score=min_answer_score,
                    min_words=min_answer_words,
                )

                if not best_answer:
                    skipped_no_valid_answer += 1
                    continue

                if match_answers_too:
                    champion_search_text = "\n".join([
                        parsed["question_title"],
                        parsed["question_body"],
                        " ".join(parsed.get("question_tags", [])),
                        best_answer["answer_text"],
                    ])
                else:
                    champion_search_text = "\n".join([
                        parsed["question_title"],
                        parsed["question_body"],
                        " ".join(parsed.get("question_tags", [])),
                    ])

                champion_matches = match_champions(champion_search_text)

                if not champion_matches:
                    skipped_no_champion_match += 1
                    continue

                item = {
                    "source": "mobafire",
                    "question_id": parsed["question_id"],
                    "question_url": parsed["question_url"],
                    "question_title": parsed["question_title"],
                    "question_body": parsed["question_body"],
                    "answer_gold": best_answer["answer_text"],
                    "answer_gold_score": best_answer["score"],
                    "answer_gold_word_count": best_answer["word_count"],
                    "answer_gold_author_line": best_answer["author_line"],
                    "answer_count_reported": parsed["answer_count_reported"],
                    "answer_count_parsed": len(parsed["answers"]),
                    "valid_answer_count": len(valid_answers),
                    "champion_matches": champion_matches,
                    "all_answers": parsed["answers"],
                    "valid_answers": valid_answers,
                }

                dataset.append(item)

                print(
                    f"[{idx}/{len(question_urls)}] kept "
                    f"{parsed['question_id']} | score={best_answer['score']} | "
                    f"champions={champion_matches} | {parsed['question_title']}"
                )

                write_json(output_path, dataset)

            except Exception as e:
                print(f"[{idx}/{len(question_urls)}] ERROR {url}: {e}")

    write_json(output_path, dataset)

    print("\nDone.")
    print(f"Saved: {output_path}")
    print(f"Kept Q&A pairs: {len(dataset)}")
    print(f"Skipped no parse: {skipped_no_parse}")
    print(f"Skipped no parsed answers: {skipped_no_answers}")
    print(f"Skipped no valid answer: {skipped_no_valid_answer}")
    print(f"Skipped no champion match: {skipped_no_champion_match}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--champions",
        default="system_known_champions_v2.json",
        help="Path to system_known_champions_v2.json",
    )

    parser.add_argument(
        "--out",
        default="mobafire_forum_qa.json",
        help="Output JSON file",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit for testing. Example: --max-pages 2",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds",
    )

    parser.add_argument(
        "--min-answer-score",
        type=int,
        default=1,
        help="Minimum answer score required to use it as a valid gold answer",
    )

    parser.add_argument(
        "--min-answer-words",
        type=int,
        default=8,
        help="Minimum answer length required to use it as a valid gold answer",
    )

    parser.add_argument(
        "--match-answers-too",
        action="store_true",
        help=(
            "Also match champion names inside the answer text. "
            "By default, champion filtering only uses question title/body."
        ),
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode to prevent UI crashes.",
    )

    args = parser.parse_args()

    scrape_mobafire_questions(
        champions_path=args.champions,
        output_path=args.out,
        max_pages=args.max_pages,
        delay=args.delay,
        min_answer_score=args.min_answer_score,
        min_answer_words=args.min_answer_words,
        match_answers_too=args.match_answers_too,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()