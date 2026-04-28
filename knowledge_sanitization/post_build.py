import argparse
import asyncio
import html
import os
from collections import defaultdict

from .config import (
    PROJECT_ROOT,
    SANITIZATION_ROOT,
    REPORT_POST_ROOT,
    QUAR_POST_ROOT,
    SANITIZED_GLOBAL_ROOT,
    SPEC_ROOT,
    ALLOWED_ENTITY_TYPES,
)
from .utils import (
    ensure_dir,
    load_json,
    save_json,
    append_jsonl,
    clean_text,
    normalize_name,
    normalize_entity_type,
    canonicalize_source_ids,
    should_block_entity_name,
)


def _load_specs():
    alias_map = {}
    for fname in ("alias_champions.json", "alias_items.json", "alias_objectives.json"):
        alias_map.update(load_json(os.path.join(SPEC_ROOT, fname), default={}))
    blocked_placeholders = {
        str(x).strip().upper() for x in load_json(os.path.join(SPEC_ROOT, "blocked_placeholders.json"), default=[])
    }
    blocked_meta = load_json(os.path.join(SPEC_ROOT, "blocked_meta_patterns.json"), default=[])
    return alias_map, blocked_placeholders, blocked_meta


async def _embed_texts(texts: list[str], batch_size: int = 32):
    import numpy as np
    from knowledge_build._llm import local_llm_config

    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = await local_llm_config.embedding_func(batch)
        vectors.append(np.asarray(emb))
    if not vectors:
        return np.zeros((0, local_llm_config.embedding_dim), dtype=np.float32)
    return np.concatenate(vectors, axis=0)


def _dequote_node_id(node_id: str) -> str:
    n = html.unescape(str(node_id)).strip()
    if len(n) >= 2 and n[0] == '"' and n[-1] == '"':
        n = n[1:-1].strip()
    return n


def _sanitize_build_cache(build_dir: str, drop_llm_cache: bool = True) -> dict:
    import networkx as nx
    import numpy as np
    from nano_vectordb import NanoVectorDB
    from knowledge_build._llm import local_llm_config

    alias_map, blocked_placeholders, blocked_meta = _load_specs()

    bname = os.path.basename(build_dir)
    if not bname.startswith("knowledge_build_cache_"):
        raise ValueError(f"Unexpected build cache folder: {bname}")
    video_name = bname.replace("knowledge_build_cache_", "", 1)

    out_dir = os.path.join(SANITIZATION_ROOT, f"sanitized_build_cache_{video_name}")
    quar_dir = os.path.join(QUAR_POST_ROOT, video_name)
    ensure_dir(out_dir)
    ensure_dir(quar_dir)

    report = {
        "video": video_name,
        "status": "pass",
        "input_dir": build_dir,
        "output_dir": out_dir,
        "files": defaultdict(lambda: {"in": 0, "out": 0, "dropped": 0}),
        "entities": {"in": 0, "out": 0, "merged": 0, "dropped": 0},
        "edges": {"in": 0, "out": 0, "dropped": 0},
        "contamination_hits": 0,
        "warnings": [],
    }

    chunks = load_json(os.path.join(build_dir, "kv_store_text_chunks.json"), default={}) or {}
    segments = load_json(os.path.join(build_dir, "kv_store_video_segments.json"), default={}) or {}
    frames = load_json(os.path.join(build_dir, "kv_store_video_frames.json"), default={}) or {}
    video_path = load_json(os.path.join(build_dir, "kv_store_video_path.json"), default={}) or {}

    seg_map = segments.get(video_name, {})
    frame_map = frames.get(video_name, {})

    # sanitize chunks
    clean_chunks = {}
    report["files"]["kv_store_text_chunks.json"]["in"] = len(chunks)
    for ck, cv in chunks.items():
        content, cstats = clean_text(str(cv.get("content", "")), blocked_meta)
        report["contamination_hits"] += cstats.get("meta_tags_removed", 0) + cstats.get("meta_patterns_removed", 0)
        if not content:
            report["files"]["kv_store_text_chunks.json"]["dropped"] += 1
            append_jsonl(os.path.join(quar_dir, "text_chunks_invalid.jsonl"), {"chunk_id": ck, "reason": "empty_after_clean"})
            continue

        seg_ids = cv.get("video_segment_id", [])
        if not isinstance(seg_ids, list):
            seg_ids = []
        seg_ids = [sid for sid in seg_ids if isinstance(sid, str) and sid.startswith(f"{video_name}_")]
        if not seg_ids:
            report["files"]["kv_store_text_chunks.json"]["dropped"] += 1
            append_jsonl(os.path.join(quar_dir, "text_chunks_invalid.jsonl"), {"chunk_id": ck, "reason": "no_valid_segment_refs"})
            continue

        clean_chunks[ck] = {
            "tokens": max(1, len(content) // 4),
            "content": content,
            "chunk_order_index": int(cv.get("chunk_order_index", 0)),
            "video_segment_id": seg_ids,
            "sanitized": True,
        }
    report["files"]["kv_store_text_chunks.json"]["out"] = len(clean_chunks)

    valid_segment_keys = {sid.split("_")[-1] for c in clean_chunks.values() for sid in c["video_segment_id"]}

    # sanitize segments and frames cross-linked to chunks
    clean_segments = {video_name: {}}
    report["files"]["kv_store_video_segments.json"]["in"] = len(seg_map)
    for seg_idx, sv in seg_map.items():
        if str(seg_idx) not in valid_segment_keys:
            continue
        content, cstats = clean_text(str(sv.get("content", "")), blocked_meta)
        transcript, tstats = clean_text(str(sv.get("transcript", "")), blocked_meta)
        report["contamination_hits"] += cstats.get("meta_tags_removed", 0) + cstats.get("meta_patterns_removed", 0)
        report["contamination_hits"] += tstats.get("meta_tags_removed", 0) + tstats.get("meta_patterns_removed", 0)
        clean_segments[video_name][str(seg_idx)] = {
            **sv,
            "content": content,
            "transcript": transcript,
        }
    report["files"]["kv_store_video_segments.json"]["out"] = len(clean_segments[video_name])

    clean_frames = {video_name: {}}
    report["files"]["kv_store_video_frames.json"]["in"] = len(frame_map)
    for fk, fv in frame_map.items():
        seg_idx = str(fv.get("segment_idx", fk.split("_")[0]))
        if seg_idx not in clean_segments[video_name]:
            continue
        transcript, tstats = clean_text(str(fv.get("transcript", "")), blocked_meta)
        vlm_output, vstats = clean_text(str(fv.get("vlm_output", "")), blocked_meta)
        report["contamination_hits"] += tstats.get("meta_tags_removed", 0) + tstats.get("meta_patterns_removed", 0)
        report["contamination_hits"] += vstats.get("meta_tags_removed", 0) + vstats.get("meta_patterns_removed", 0)
        entities = fv.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        norm_entities = sorted(
            {
                normalize_name(str(entity), alias_map)
                for entity in entities
                if str(entity).strip()
            }
        )
        clean_frame = {
            **fv,
            "segment_idx": seg_idx,
            "entities": [entity for entity in norm_entities if entity and entity != "UNKNOWN"],
            "transcript": transcript,
            "vlm_output": vlm_output,
        }
        if "main_champ" in fv:
            clean_frame["main_champ"] = normalize_name(str(fv.get("main_champ", "Unknown")), alias_map)
        if "partners" in fv:
            partners = fv.get("partners", [])
            if not isinstance(partners, list):
                partners = []
            norm_partners = sorted(
                {
                    normalize_name(str(p), alias_map)
                    for p in partners
                    if str(p).strip()
                }
            )
            clean_frame["partners"] = [partner for partner in norm_partners if partner and partner != "UNKNOWN"]
        clean_frames[video_name][fk] = clean_frame
    report["files"]["kv_store_video_frames.json"]["out"] = len(clean_frames[video_name])

    clean_video_path = {}
    if video_name in video_path:
        clean_video_path[video_name] = video_path[video_name]
    report["files"]["kv_store_video_path.json"]["in"] = len(video_path)
    report["files"]["kv_store_video_path.json"]["out"] = len(clean_video_path)

    # graph sanitize
    graph_path = os.path.join(build_dir, "graph_chunk_entity_relation_clean.graphml")
    if not os.path.exists(graph_path):
        graph_path = os.path.join(build_dir, "graph_chunk_entity_relation.graphml")
    G = nx.read_graphml(graph_path) if os.path.exists(graph_path) else nx.Graph()

    report["entities"]["in"] = G.number_of_nodes()
    report["edges"]["in"] = G.number_of_edges()

    valid_chunk_ids = set(clean_chunks.keys())
    H = nx.Graph()
    name_to_node = {}

    for node_id, attrs in G.nodes(data=True):
        nid = _dequote_node_id(node_id)
        nid = normalize_name(nid, alias_map)
        if should_block_entity_name(nid, blocked_placeholders):
            report["entities"]["dropped"] += 1
            continue

        entity_type = normalize_entity_type(str(attrs.get("entity_type", "UNKNOWN")), ALLOWED_ENTITY_TYPES)
        desc, _ = clean_text(str(attrs.get("description", "")), blocked_meta)
        source_id = canonicalize_source_ids(attrs.get("source_id", ""), valid_chunk_ids)
        if not source_id:
            report["entities"]["dropped"] += 1
            continue

        if nid in name_to_node:
            prev = H.nodes[nid]
            merged_source = sorted(set((prev.get("source_id", "") + "<SEP>" + source_id).split("<SEP>")))
            merged_desc = sorted(set((prev.get("description", "") + "<SEP>" + desc).split("<SEP>")))
            prev["source_id"] = "<SEP>".join([x for x in merged_source if x])
            prev["description"] = "<SEP>".join([x for x in merged_desc if x])
            if prev.get("entity_type") == "UNKNOWN" and entity_type != "UNKNOWN":
                prev["entity_type"] = entity_type
            report["entities"]["merged"] += 1
        else:
            H.add_node(nid, entity_type=entity_type, description=desc, source_id=source_id)
            name_to_node[nid] = nid

    for s, t, attrs in G.edges(data=True):
        ns = normalize_name(_dequote_node_id(s), alias_map)
        nt = normalize_name(_dequote_node_id(t), alias_map)
        if ns == nt:
            report["edges"]["dropped"] += 1
            continue
        if ns not in H.nodes or nt not in H.nodes:
            report["edges"]["dropped"] += 1
            continue

        desc, _ = clean_text(str(attrs.get("description", "")), blocked_meta)
        sid = canonicalize_source_ids(attrs.get("source_id", ""), valid_chunk_ids)
        if not sid:
            report["edges"]["dropped"] += 1
            continue

        try:
            weight = float(attrs.get("weight", 1.0))
        except Exception:
            weight = 1.0
        weight = min(10.0, max(0.0, weight))
        order = int(attrs.get("order", 1)) if str(attrs.get("order", "1")).isdigit() else 1

        if H.has_edge(ns, nt):
            e = H.edges[(ns, nt)]
            e["weight"] = min(10.0, max(e.get("weight", 1.0), weight))
            e["description"] = "<SEP>".join(sorted(set((e.get("description", "") + "<SEP>" + desc).split("<SEP>"))))
            e["source_id"] = "<SEP>".join(sorted(set((e.get("source_id", "") + "<SEP>" + sid).split("<SEP>"))))
            e["order"] = min(e.get("order", 1), order)
        else:
            H.add_edge(ns, nt, weight=weight, description=desc, source_id=sid, order=order)

    report["entities"]["out"] = H.number_of_nodes()
    report["edges"]["out"] = H.number_of_edges()

    # save sanitized files
    save_json(clean_chunks, os.path.join(out_dir, "kv_store_text_chunks.json"))
    save_json(clean_segments, os.path.join(out_dir, "kv_store_video_segments.json"))
    save_json(clean_frames, os.path.join(out_dir, "kv_store_video_frames.json"))
    save_json(clean_video_path, os.path.join(out_dir, "kv_store_video_path.json"))
    nx.write_graphml(H, os.path.join(out_dir, "graph_chunk_entity_relation.graphml"))
    nx.write_graphml(H, os.path.join(out_dir, "graph_chunk_entity_relation_clean.graphml"))

    # optionally drop llm cache
    llm_cache_in = os.path.join(build_dir, "kv_store_llm_response_cache.json")
    if os.path.exists(llm_cache_in) and not drop_llm_cache:
        llm_cache = load_json(llm_cache_in, default={})
        save_json(llm_cache, os.path.join(out_dir, "kv_store_llm_response_cache.json"))

    # rebuild vdb chunks
    chunk_ids = list(clean_chunks.keys())
    chunk_texts = [clean_chunks[cid]["content"] for cid in chunk_ids]
    chunk_vectors = asyncio.run(_embed_texts(chunk_texts)) if chunk_ids else np.zeros((0, local_llm_config.embedding_dim), dtype=np.float32)
    vdb_chunks = NanoVectorDB(local_llm_config.embedding_dim, storage_file=os.path.join(out_dir, "vdb_chunks.json"))
    vdb_chunks.upsert([
        {"__id__": cid, "__vector__": chunk_vectors[i]} for i, cid in enumerate(chunk_ids)
    ])
    vdb_chunks.save()

    # rebuild vdb entities
    entity_ids = []
    entity_texts = []
    entity_names = []
    for n, attrs in H.nodes(data=True):
        ent_id = f"ent-{abs(hash(n)) % (10**32):032d}"[:36]
        entity_ids.append(ent_id)
        entity_names.append(n)
        entity_texts.append((n + " " + str(attrs.get("description", ""))).strip())
    entity_vectors = asyncio.run(_embed_texts(entity_texts)) if entity_ids else np.zeros((0, local_llm_config.embedding_dim), dtype=np.float32)
    vdb_entities = NanoVectorDB(local_llm_config.embedding_dim, storage_file=os.path.join(out_dir, "vdb_entities.json"))
    vdb_entities.upsert([
        {
            "__id__": entity_ids[i],
            "entity_name": entity_names[i],
            "__vector__": entity_vectors[i],
        }
        for i in range(len(entity_ids))
    ])
    vdb_entities.save()

    report["files"]["vdb_chunks.json"]["out"] = len(chunk_ids)
    report["files"]["vdb_entities.json"]["out"] = len(entity_ids)

    if report["entities"]["out"] == 0 or report["files"]["kv_store_text_chunks.json"]["out"] == 0:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "warn"

    save_json(report, os.path.join(REPORT_POST_ROOT, f"{video_name}.json"))
    return report


def _sanitize_global_graph() -> dict:
    import networkx as nx

    ensure_dir(SANITIZED_GLOBAL_ROOT)
    in_graph = os.path.join(PROJECT_ROOT, "knowledge_build_cache_global", "graph_AetherNexus.graphml")
    in_manifest = os.path.join(PROJECT_ROOT, "knowledge_build_cache_global", "aether_manifest.json")

    out_graph = os.path.join(SANITIZED_GLOBAL_ROOT, "graph_AetherNexus.graphml")
    out_manifest = os.path.join(SANITIZED_GLOBAL_ROOT, "aether_manifest.json")

    if not os.path.exists(in_graph):
        return {"status": "warn", "reason": "global_graph_missing"}

    G = nx.read_graphml(in_graph)
    # lightweight normalization only
    H = nx.Graph()
    for n, attrs in G.nodes(data=True):
        nid = _dequote_node_id(n)
        H.add_node(nid, **attrs)
    for s, t, attrs in G.edges(data=True):
        H.add_edge(_dequote_node_id(s), _dequote_node_id(t), **attrs)

    nx.write_graphml(H, out_graph)
    manifest = load_json(in_manifest, default={"processed_videos": []})
    if isinstance(manifest, list):
        manifest = {"processed_videos": sorted(set(map(str, manifest)))}
    elif isinstance(manifest, dict):
        videos = manifest.get("processed_videos", [])
        manifest = {"processed_videos": sorted(set(map(str, videos)))}
    else:
        manifest = {"processed_videos": []}
    save_json(manifest, out_manifest)

    return {"status": "pass", "nodes": H.number_of_nodes(), "edges": H.number_of_edges()}


def discover_build_dirs(project_root: str):
    out = []
    for e in os.scandir(project_root):
        if not e.is_dir():
            continue
        if e.name.startswith("knowledge_build_cache_") and e.name != "knowledge_build_cache_global":
            out.append(e.path)
    return sorted(out)


def main() -> int:
    try:
        import networkx  # noqa: F401
        import numpy  # noqa: F401
        import nano_vectordb  # noqa: F401
    except ModuleNotFoundError as exc:
        print(
            "Missing dependency for post-build sanitization: "
            f"{exc}. Install required packages in the active environment."
        )
        return 1

    parser = argparse.ArgumentParser(description="Sanitize build caches and rebuild retrieval artifacts.")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--video", default=None, help="Optional video name, not full folder")
    parser.add_argument("--keep-llm-cache", action="store_true")
    args = parser.parse_args()

    ensure_dir(REPORT_POST_ROOT)

    if args.video:
        dirs = [os.path.join(args.project_root, f"knowledge_build_cache_{args.video}")]
    else:
        dirs = discover_build_dirs(args.project_root)

    if not dirs:
        print("No build cache folders found.")
        return 1

    failed = 0
    for d in dirs:
        report = _sanitize_build_cache(d, drop_llm_cache=not args.keep_llm_cache)
        print(f"[post-build] {report['video']}: {report['status']}")
        if report["status"] == "fail":
            failed += 1

    g_report = _sanitize_global_graph()
    print(f"[global] {g_report.get('status')}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
