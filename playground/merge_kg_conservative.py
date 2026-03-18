#!/usr/bin/env python3
import html
import re
import unicodedata
from collections import defaultdict

import networkx as nx


# Edit these paths before running.
INPUT_GRAPH_PATH_1 = "/home/gatv-projects/Desktop/project/knowledge_sanitization/cache2/cache/sanitized_global/graph_AetherNexus.graphml"
INPUT_GRAPH_PATH_2 = "/home/gatv-projects/Desktop/project/knowledge_sanitization/cache/sanitized_global/graph_AetherNexus.graphml"
OUTPUT_GRAPH_PATH = "/home/gatv-projects/Desktop/project/knowledge_sanitization/cache/big_merge_result.graphml"

SEP = "<SEP>"


def _html_unescape(s: str) -> str:
    return html.unescape(s) if isinstance(s, str) else s


def _strip_wrapping_quotes(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    while (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ('"', "'", "“", "”", "‘", "’")):
        s = s[1:-1].strip()
    return s


def normalize_entity_key(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = _html_unescape(text)
    s = _strip_wrapping_quotes(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-z]+", "", s)
    return s


def split_sep_field(value: str) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [v for v in value.split(SEP) if v.strip()]


def join_sep_unique(values: list[str]) -> str:
    seen = set()
    out = []
    for v in values:
        if not isinstance(v, str):
            v = str(v)
        vv = v.strip()
        if not vv:
            continue
        if vv not in seen:
            seen.add(vv)
            out.append(vv)
    return SEP.join(out)


_ALIAS_PATTERNS = [
    r"\baka\b[:\s]+(?P<name>[A-Za-z0-9][A-Za-z0-9’'\- ]{1,40})",
    r"\ba\.k\.a\.\b[:\s]+(?P<name>[A-Za-z0-9][A-Za-z0-9’'\- ]{1,40})",
    r"\balso called\b[:\s]+(?P<name>[A-Za-z0-9][A-Za-z0-9’'\- ]{1,40})",
    r"\bknown as\b[:\s]+(?P<name>[A-Za-z0-9][A-Za-z0-9’'\- ]{1,40})",
    r"\breferred to as\b[:\s]+(?P<name>[A-Za-z0-9][A-Za-z0-9’'\- ]{1,40})",
    r"\bin transcript\b.*?\b(?P<name>[A-Za-z0-9][A-Za-z0-9’'\-]{2,40})",
    r"\bspelled\b[:\s]+(?P<name>[A-Za-z0-9][A-Za-z0-9’'\-]{2,40})",
    r"\bmisspelled\b[:\s]+(?P<name>[A-Za-z0-9][A-Za-z0-9’'\-]{2,40})",
]

_ALIAS_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in _ALIAS_PATTERNS]


def extract_alias_candidates(description: str) -> set[str]:
    if not isinstance(description, str) or not description.strip():
        return set()

    desc = _html_unescape(description)
    chunks = split_sep_field(desc) or [desc]

    aliases = set()
    for ch in chunks:
        for rx in _ALIAS_REGEXES:
            for m in rx.finditer(ch):
                name = (m.groupdict().get("name") or "").strip()
                if name and len(name) >= 3:
                    aliases.add(name)
    return aliases


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def entity_type_of(G, node_id) -> str:
    t = G.nodes[node_id].get("entity_type", "")
    return t if isinstance(t, str) else str(t)


def types_compatible(t1: str, t2: str) -> bool:
    t1 = (t1 or "").strip()
    t2 = (t2 or "").strip()
    if not t1 or not t2:
        return True
    if t1 == t2:
        return True
    if t1 == "UNKNOWN" or t2 == "UNKNOWN":
        return True
    return False


def unify_entities_conservative(G: nx.Graph) -> nx.MultiGraph:
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        H_in = G.copy()
    else:
        H_in = nx.MultiGraph(G)

    uf = UnionFind()
    node_keys = defaultdict(set)
    key_to_nodes = defaultdict(list)

    for n in H_in.nodes():
        nid_key = normalize_entity_key(str(n))
        if nid_key:
            node_keys[n].add(nid_key)

        desc = H_in.nodes[n].get("description", "")
        for alias in extract_alias_candidates(desc):
            ak = normalize_entity_key(alias)
            if ak:
                node_keys[n].add(ak)

        for k in node_keys[n]:
            key_to_nodes[k].append(n)

    for _, nodes in key_to_nodes.items():
        if len(nodes) < 2:
            continue
        base = nodes[0]
        for other in nodes[1:]:
            t1 = entity_type_of(H_in, base)
            t2 = entity_type_of(H_in, other)
            if types_compatible(t1, t2):
                uf.union(base, other)

    groups = defaultdict(list)
    for n in H_in.nodes():
        groups[uf.find(n)].append(n)

    def rep_score(node_id: str) -> tuple:
        attrs = H_in.nodes[node_id]
        et = (attrs.get("entity_type") or "").strip()
        et_bonus = 1 if (et and et != "UNKNOWN") else 0
        raw = str(node_id)
        unescaped = _html_unescape(raw)
        stripped = _strip_wrapping_quotes(unescaped)
        quote_penalty = 1 if stripped != unescaped else 0
        return (et_bonus, -quote_penalty, -len(stripped), -len(raw))

    rep_of = {}
    members_of_rep = {}

    for root, members in groups.items():
        rep = sorted(members, key=rep_score, reverse=True)[0]
        rep_of[root] = rep
        members_of_rep[rep] = members

    H_out = nx.MultiGraph()

    def merge_node_attrs(nodes: list) -> dict:
        merged = {}
        all_keys = set()
        for nn in nodes:
            all_keys |= set(H_in.nodes[nn].keys())

        for k in all_keys:
            vals = []
            for nn in nodes:
                v = H_in.nodes[nn].get(k, None)
                if v is None:
                    continue
                if isinstance(v, str) and SEP in v:
                    vals.extend(split_sep_field(v))
                else:
                    vals.append(str(v))
            merged[k] = join_sep_unique(vals)
        return merged

    for rep, members in members_of_rep.items():
        H_out.add_node(rep, **merge_node_attrs(members))

    def map_node(n):
        return rep_of[uf.find(n)]

    for u, v, _, data in H_in.edges(keys=True, data=True):
        uu, vv = map_node(u), map_node(v)
        if uu == vv:
            continue

        new_data = {}
        for k, val in (data or {}).items():
            if isinstance(val, str) and SEP in val:
                new_data[k] = join_sep_unique(split_sep_field(val))
            else:
                new_data[k] = val
        H_out.add_edge(uu, vv, **new_data)

    return H_out


def load_graphml(path: str) -> nx.Graph:
    return nx.read_graphml(path)


def save_graphml(G: nx.Graph, path: str) -> None:
    nx.write_graphml(G, path)


def merge_graphs(graph_1: nx.Graph, graph_2: nx.Graph) -> nx.MultiGraph:
    merged = nx.compose(nx.MultiGraph(graph_1), nx.MultiGraph(graph_2))
    return unify_entities_conservative(merged)


def main() -> None:
    if not INPUT_GRAPH_PATH_1 or not INPUT_GRAPH_PATH_2 or not OUTPUT_GRAPH_PATH:
        raise ValueError("Set INPUT_GRAPH_PATH_1, INPUT_GRAPH_PATH_2, and OUTPUT_GRAPH_PATH before running.")

    print("Loading graphs...")
    graph_1 = load_graphml(INPUT_GRAPH_PATH_1)
    graph_2 = load_graphml(INPUT_GRAPH_PATH_2)

    print(f"Graph 1: {graph_1.number_of_nodes()} nodes | {graph_1.number_of_edges()} edges")
    print(f"Graph 2: {graph_2.number_of_nodes()} nodes | {graph_2.number_of_edges()} edges")

    print("Merging and unifying entities conservatively...")
    merged = merge_graphs(graph_1, graph_2)

    print(f"Merged: {merged.number_of_nodes()} nodes | {merged.number_of_edges()} edges")
    save_graphml(merged, OUTPUT_GRAPH_PATH)
    print(f"Saved merged graph to {OUTPUT_GRAPH_PATH}")


if __name__ == "__main__":
    main()
