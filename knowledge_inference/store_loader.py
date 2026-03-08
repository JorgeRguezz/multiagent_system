from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
from nano_vectordb import NanoVectorDB

from knowledge_build._llm import local_llm_config

from . import config
from .types import VideoStore

logger = logging.getLogger(config.LOGGER_NAME)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_sanitized_path(path: Path) -> None:
    resolved = path.resolve()
    if config.SANITIZED_CACHE_ROOT.resolve() not in resolved.parents and resolved != config.SANITIZED_CACHE_ROOT.resolve():
        raise ValueError(f"Refusing to read non-sanitized path: {resolved}")


def discover_sanitized_video_dirs() -> list[Path]:
    root = config.SANITIZED_CACHE_ROOT
    _assert_sanitized_path(root)
    if not root.exists():
        raise FileNotFoundError(f"Sanitized cache root not found: {root}")
    dirs = [p for p in root.glob(config.SANITIZED_BUILD_GLOB) if p.is_dir()]
    return sorted(dirs)


def _extract_video_name(video_dir: Path) -> str:
    prefix = "sanitized_build_cache_"
    name = video_dir.name
    if not name.startswith(prefix):
        raise ValueError(f"Invalid sanitized cache directory name: {name}")
    return name[len(prefix) :]


def _load_vdb(vdb_path: Path) -> NanoVectorDB:
    _assert_sanitized_path(vdb_path)
    data = _read_json(vdb_path)
    embedding_dim = int(data.get("embedding_dim", local_llm_config.embedding_dim))
    return NanoVectorDB(embedding_dim, storage_file=str(vdb_path))


def _validate_store(store: VideoStore) -> None:
    chunk_count = len(store.chunks_kv)
    if chunk_count <= 0:
        logger.warning("Video '%s' has no chunks.", store.video_name)

    vdb_chunk_count = len(store.chunks_vdb)
    if vdb_chunk_count != chunk_count:
        logger.warning(
            "Video '%s' chunk mismatch: vdb_chunks=%s chunks_kv=%s",
            store.video_name,
            vdb_chunk_count,
            chunk_count,
        )

    segment_map = store.segments_kv.get(store.video_name, {}) if isinstance(store.segments_kv, dict) else {}
    for chunk_id, chunk in store.chunks_kv.items():
        seg_ids = chunk.get("video_segment_id", [])
        if isinstance(seg_ids, str):
            seg_ids = [seg_ids]
        for seg_id in seg_ids:
            if not isinstance(seg_id, str) or "_" not in seg_id:
                logger.warning("Chunk '%s' has malformed segment id: %r", chunk_id, seg_id)
                continue
            seg_idx = seg_id.rsplit("_", 1)[-1]
            if seg_idx not in segment_map:
                logger.warning(
                    "Chunk '%s' in video '%s' references missing segment idx '%s'",
                    chunk_id,
                    store.video_name,
                    seg_idx,
                )

    graph_nodes = store.graph.number_of_nodes()
    vdb_entities_count = len(store.entities_vdb)
    if graph_nodes == 0:
        logger.warning("Video '%s' graph has no nodes.", store.video_name)
    elif abs(graph_nodes - vdb_entities_count) > max(5, int(graph_nodes * 0.25)):
        logger.warning(
            "Video '%s' entity mismatch: graph_nodes=%s vdb_entities=%s",
            store.video_name,
            graph_nodes,
            vdb_entities_count,
        )


def load_video_store(video_dir: Path) -> VideoStore:
    _assert_sanitized_path(video_dir)
    video_name = _extract_video_name(video_dir)

    chunks_path = video_dir / "kv_store_text_chunks.json"
    segments_path = video_dir / "kv_store_video_segments.json"
    frames_path = video_dir / "kv_store_video_frames.json"
    vdb_chunks_path = video_dir / "vdb_chunks.json"
    vdb_entities_path = video_dir / "vdb_entities.json"
    graph_path = video_dir / "graph_chunk_entity_relation_clean.graphml"

    required = [
        chunks_path,
        segments_path,
        frames_path,
        vdb_chunks_path,
        vdb_entities_path,
        graph_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required sanitized files for '{video_name}': {missing}")

    chunks_kv = _read_json(chunks_path)
    segments_kv = _read_json(segments_path)
    frames_kv = _read_json(frames_path)
    chunks_vdb = _load_vdb(vdb_chunks_path)
    entities_vdb = _load_vdb(vdb_entities_path)
    graph = nx.read_graphml(graph_path)

    store = VideoStore(
        video_name=video_name,
        chunks_vdb=chunks_vdb,
        entities_vdb=entities_vdb,
        chunks_kv=chunks_kv,
        segments_kv=segments_kv,
        frames_kv=frames_kv,
        graph=graph,
    )
    _validate_store(store)
    return store


def load_all_video_stores() -> dict[str, VideoStore]:
    stores: dict[str, VideoStore] = {}
    for video_dir in discover_sanitized_video_dirs():
        try:
            store = load_video_store(video_dir)
            stores[store.video_name] = store
        except Exception as exc:
            logger.warning("Skipping sanitized video dir '%s' due to load error: %s", video_dir, exc)
    if not stores:
        raise RuntimeError("No valid sanitized video stores could be loaded.")
    return stores


def load_global_graph() -> nx.Graph:
    path = config.SANITIZED_GLOBAL_GRAPH
    _assert_sanitized_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sanitized global graph not found: {path}")
    return nx.read_graphml(path)


def warmup() -> tuple[dict[str, VideoStore], nx.Graph]:
    stores = load_all_video_stores()
    global_graph = load_global_graph()
    return stores, global_graph
