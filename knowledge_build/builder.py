from __future__ import annotations

import os
import json
import asyncio
import networkx as nx
from dataclasses import dataclass, field, asdict
from typing import Type, Dict, Optional, Callable, List, Union

if __package__ in (None, ""):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_build.config import (
    VIDEO_SEGMENTS_FILENAME,
    VIDEO_FRAMES_FILENAME,
    VIDEO_PATHS_FILENAME,
)
from knowledge_build._storage.kv_json import JsonKVStorage
from knowledge_build._storage.vdb_nanovectordb import NanoVectorDBStorage
from knowledge_build._storage.gdb_networkx import NetworkXStorage
from knowledge_build.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage
from knowledge_build._utils import limit_async_func_call, wrap_embedding_func_with_attrs, load_json, logger
from knowledge_build._op import chunking_by_video_segments, extract_entities, get_chunks
from knowledge_build._llm import LLMConfig, local_llm_config
from knowledge_build.clean_kg import load_graphml, save_graphml, unify_entities_conservative


@dataclass
class KnowledgeBuilder:
    """
    Build knowledge graph + vector DBs from extraction artifacts.
    Expects files produced by knowledge_extraction using the same
    naming convention as chatbot_system (kv_store_*.json).
    """
    extraction_dir: str

    # output config
    working_dir: str = field(init=False)

    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # LLM
    llm: LLMConfig = field(default_factory=lambda: local_llm_config)
    enable_llm_cache: bool = True

    # text chunking
    chunk_func: Callable[..., List[Dict[str, Union[str, int]]]] = field(
        default=chunking_by_video_segments
    )
    chunk_token_size: int = 1200

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    # entity extraction
    entity_extraction_func: callable = extract_entities
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # internal state
    artifact_dir: str = field(init=False)
    source_video_name: str = field(init=False)
    global_cache_dir: str = field(init=False)
    global_graph_name: str = field(default="graph_AetherNexus.graphml")
    global_manifest_name: str = field(default="aether_manifest.json")

    def __post_init__(self):
        self.artifact_dir = self._resolve_artifact_dir()
        self.source_video_name = os.path.basename(self.artifact_dir)
        self.working_dir = os.path.join(
            self._project_root(), f"knowledge_build_cache_{self.source_video_name}"
        )
        os.makedirs(self.working_dir, exist_ok=True)
        self.global_cache_dir = os.path.join(self._project_root(), "knowledge_build_cache_global")
        os.makedirs(self.global_cache_dir, exist_ok=True)

        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(
            wrap_embedding_func_with_attrs(
                embedding_dim=self.llm.embedding_dim,
                max_token_size=self.llm.embedding_max_token_size,
                model_name=self.llm.embedding_model_name,
            )(self.llm.embedding_func)
        )

        config_dict = asdict(self)

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=config_dict
        )
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=config_dict
        )
        self.video_frames = self.key_string_value_json_storage_cls(
            namespace="video_frames", global_config=config_dict
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=config_dict,
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=config_dict,
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=config_dict
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=config_dict
        )
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=config_dict
            )
            if self.enable_llm_cache
            else None
        )

    def _project_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve_artifact_dir(self) -> str:
        # Required layout: extraction_dir/sanitized_extracted_data/<video_name>/kv_store_*.json
        extracted_data_root = os.path.join(self.extraction_dir, "sanitized_extracted_data")
        if not os.path.isdir(extracted_data_root):
            raise FileNotFoundError(
                f"Could not find extraction artifacts directory: {extracted_data_root}"
            )

        video_dirs = [
            entry.path for entry in os.scandir(extracted_data_root) if entry.is_dir()
        ]
        if not video_dirs:
            raise FileNotFoundError(
                f"No video directories found in extracted_data: {extracted_data_root}"
            )

        candidate_dirs = [
            path
            for path in video_dirs
            if os.path.exists(os.path.join(path, VIDEO_SEGMENTS_FILENAME))
        ]
        if not candidate_dirs:
            raise FileNotFoundError(
                f"No artifact directory contains {VIDEO_SEGMENTS_FILENAME} under {extracted_data_root}"
            )

        # Build outputs are named as knowledge_build_cache_<video_folder_name>.
        project_root = self._project_root()
        unbuilt_dirs = [
            path
            for path in sorted(candidate_dirs)
            if not os.path.exists(
                os.path.join(project_root, f"knowledge_build_cache_{os.path.basename(path)}")
            )
        ]

        if not unbuilt_dirs:
            raise FileNotFoundError(
                "No unbuilt extraction folders found. All sanitized_extracted_data folders already have matching knowledge_build_cache_<video_name> outputs."
            )

        if len(candidate_dirs) > 1:
            logger.warning(
                f"Multiple sanitized_extracted_data folders found; selected next unbuilt folder: {os.path.basename(unbuilt_dirs[0])}"
            )
        return unbuilt_dirs[0]

    def _load_artifact(self, artifact_dir: str, filename: str) -> Optional[dict]:
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Artifact not found: {path}")
            return None
        return load_json(path)

    async def build(self):
        """Load extraction artifacts and build graph/vector DBs."""
        if not os.path.isdir(self.extraction_dir):
            raise FileNotFoundError(f"Extraction dir not found: {self.extraction_dir}")

        artifact_dir = self.artifact_dir
        logger.info(f"Using artifact dir: {artifact_dir}")
        logger.info(f"Using build output dir: {self.working_dir}")

        segments_data = self._load_artifact(artifact_dir, VIDEO_SEGMENTS_FILENAME)
        if not segments_data:
            raise FileNotFoundError(
                f"Missing required artifact: {VIDEO_SEGMENTS_FILENAME} in {artifact_dir}"
            )

        frames_data = self._load_artifact(artifact_dir, VIDEO_FRAMES_FILENAME)
        paths_data = self._load_artifact(artifact_dir, VIDEO_PATHS_FILENAME)

        await self.video_segments.upsert(segments_data)
        if frames_data:
            await self.video_frames.upsert(frames_data)
        if paths_data:
            await self.video_path_db.upsert(paths_data)

        await self.ainsert(self.video_segments._data)
        self._post_clean_graphs_and_update_global()

    async def ainsert(self, new_video_segment: Dict):
        await self._insert_start()
        try:
            inserting_chunks = get_chunks(
                new_videos=new_video_segment,
                chunk_func=self.chunk_func,
                max_token_size=self.chunk_token_size,
            )
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            config_dict = asdict(self)
            if self.llm_response_cache:
                config_dict["llm_response_cache"] = self.llm_response_cache

            maybe_new_kg, _, _ = await self.entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=config_dict,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [self.chunk_entity_relation_graph]:
            if storage_inst is None:
                continue
            tasks.append(
                ((storage_inst)).index_start_callback()
            )
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_frames,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(((storage_inst)).index_done_callback())
        await asyncio.gather(*tasks)

    def _post_clean_graphs_and_update_global(self) -> None:
        graph_path = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
        cleaned_graph_path = os.path.join(self.working_dir, "graph_chunk_entity_relation_clean.graphml")

        if not os.path.exists(graph_path):
            logger.warning(f"Skipping KG post-cleaning, graph not found: {graph_path}")
            return

        logger.info("Running post-cleaning on per-video knowledge graph...")
        graph = load_graphml(graph_path)
        cleaned = unify_entities_conservative(graph)
        save_graphml(cleaned, cleaned_graph_path)
        logger.info(
            f"Per-video graph cleaned: nodes {graph.number_of_nodes()} -> {cleaned.number_of_nodes()}, "
            f"edges {graph.number_of_edges()} -> {cleaned.number_of_edges()}"
        )

        try:
            self._update_global_knowledge_graph(cleaned_graph_path)
        except Exception as exc:
            logger.warning(f"Global knowledge graph update failed: {exc}")

    def _update_global_knowledge_graph(self, cleaned_graph_path: str) -> None:
        manifest_path = os.path.join(self.global_cache_dir, self.global_manifest_name)
        global_graph_path = os.path.join(self.global_cache_dir, self.global_graph_name)
        source_id = self.source_video_name

        processed = self._load_global_manifest(manifest_path)
        if source_id in processed:
            logger.info(f"Global graph already includes '{source_id}', skipping merge.")
            return

        incoming_graph = load_graphml(cleaned_graph_path)
        if not os.path.exists(global_graph_path):
            save_graphml(incoming_graph, global_graph_path)
            logger.info(f"Initialized global knowledge graph: {global_graph_path}")
        else:
            global_graph = load_graphml(global_graph_path)
            merged = nx.compose(nx.MultiGraph(global_graph), nx.MultiGraph(incoming_graph))
            merged_clean = unify_entities_conservative(merged)
            save_graphml(merged_clean, global_graph_path)
            logger.info(
                f"Updated global graph: nodes {global_graph.number_of_nodes()} -> {merged_clean.number_of_nodes()}, "
                f"edges {global_graph.number_of_edges()} -> {merged_clean.number_of_edges()}"
            )

        processed.append(source_id)
        self._save_global_manifest(manifest_path, processed)

    def _load_global_manifest(self, manifest_path: str) -> list[str]:
        if not os.path.exists(manifest_path):
            return []
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                items = data.get("processed_videos", [])
            elif isinstance(data, list):
                items = data
            else:
                items = []
            return [str(x) for x in items]
        except Exception:
            return []

    def _save_global_manifest(self, manifest_path: str, processed: list[str]) -> None:
        payload = {"processed_videos": sorted(set(processed))}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def _default_extraction_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_sanitization",
        "cache",
    )


def main():
    extraction_dir = _default_extraction_dir()
    print(f"Using extraction_dir: {extraction_dir}")
    builder = KnowledgeBuilder(extraction_dir=extraction_dir)
    print(f"Selected artifact_dir: {builder.artifact_dir}")
    print(f"Build output dir: {builder.working_dir}")
    asyncio.run(builder.build())


if __name__ == "__main__":
    main()
