from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Type, Dict, Optional

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
from knowledge_build.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage, StorageNameSpace
from knowledge_build._utils import limit_async_func_call, wrap_embedding_func_with_attrs, load_json, logger
from knowledge_build._op import chunking_by_video_segments, extract_entities, get_chunks
from knowledge_build._llm import LLMConfig, local_llm_config


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
    chunk_func = chunking_by_video_segments
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
    video_segment_feature_vdb: BaseVectorStorage = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.working_dir = f"./knowledge_build_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        os.makedirs(self.working_dir, exist_ok=True)

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

    def _load_artifact(self, filename: str) -> Optional[dict]:
        path = os.path.join(self.extraction_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Artifact not found: {path}")
            return None
        return load_json(path)

    async def build(self):
        """Load extraction artifacts and build graph/vector DBs."""
        if not os.path.isdir(self.extraction_dir):
            raise FileNotFoundError(f"Extraction dir not found: {self.extraction_dir}")

        segments_data = self._load_artifact(VIDEO_SEGMENTS_FILENAME)
        if not segments_data:
            raise FileNotFoundError(
                f"Missing required artifact: {VIDEO_SEGMENTS_FILENAME} in {self.extraction_dir}"
            )

        frames_data = self._load_artifact(VIDEO_FRAMES_FILENAME)
        paths_data = self._load_artifact(VIDEO_PATHS_FILENAME)

        await self.video_segments.upsert(segments_data)
        if frames_data:
            await self.video_frames.upsert(frames_data)
        if paths_data:
            await self.video_path_db.upsert(paths_data)

        await self.ainsert(self.video_segments._data)

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


def _default_extraction_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_extraction",
        "cache",
    )


def main():
    extraction_dir = _default_extraction_dir()
    print(f"Using extraction_dir: {extraction_dir}")
    builder = KnowledgeBuilder(extraction_dir=extraction_dir)
    asyncio.run(builder.build())


if __name__ == "__main__":
    main()
