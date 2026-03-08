from datetime import datetime
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Type, cast, Callable, List, Dict, Union, Optional
import asyncio

# MCP Imports (forward reference for type hint)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp import ClientSession

# Local project imports
from ._storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from .base import BaseVectorStorage, StorageNameSpace, BaseKVStorage, BaseGraphStorage
from ._utils import logger, limit_async_func_call, wrap_embedding_func_with_attrs
from ._op import chunking_by_video_segments, get_chunks
# Use the API version of extract_entities
from ._op_api import extract_entities as extract_entities_api
# Use the Gemini config
from ._llm_gemini_api import LLMConfig, gemini_config


@dataclass
class VideoKnowledgeExtractor:
    """
    A class to extract knowledge from a video file using Gemini API.
    This test version simulates the post-processing part after VLM/ASR.
    """
    # --- Configuration ---
    video_path: str
    mcp_sessions: Dict[str, 'ClientSession'] = field(repr=False)
    working_dir: str = field(init=False)

    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True
    
    # --- LLM (uses Gemini API) ---
    llm: LLMConfig = field(default_factory=lambda: gemini_config)
    enable_llm_cache: bool = True

    # --- Internal State ---
    video_segment_feature_vdb: "BaseVectorStorage" = field(init=False, repr=False, default=None)

    # text chunking
    chunk_func: Callable[..., List[Dict[str, Union[str, int]]]] = chunking_by_video_segments
    chunk_token_size: int = 1200

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    # entity extraction
    entity_extraction_func: callable = extract_entities_api
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    video_game_name: str = "League of Legends"
    relationship_strength_min: float = 1.0
    relationship_strength_max: float = 10.0
    extraction_use_domain_context: bool = True
    extraction_glean_mode: str = "split"

    def __post_init__(self):
        """Initializes working directory and storage for the client side."""
        self.working_dir = f"./test_data_build_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(wrap_embedding_func_with_attrs(
                embedding_dim = self.llm.embedding_dim,
                max_token_size = self.llm.embedding_max_token_size,
                model_name = self.llm.embedding_model_name)(self.llm.embedding_func))

        sessions = self.mcp_sessions
        self.mcp_sessions = None
        try:
            config_dict = asdict(self)
        finally:
            self.mcp_sessions = sessions

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=config_dict
        )
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=config_dict
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

    async def ainsert(self, new_video_segment):
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
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            
            sessions = self.mcp_sessions
            self.mcp_sessions = None
            try:
                config_dict = asdict(self)
            finally:
                self.mcp_sessions = sessions

            if self.llm_response_cache:
                config_dict['llm_response_cache'] = self.llm_response_cache

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
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)


async def main():
    """Main function to test the data building from a pre-defined JSON."""
    print("Starting data structure build test with Gemini API...")

    with open("/home/gatv-projects/Desktop/project/knowledge_build_cache_2025-11-25-10:03:18/kv_store_video_segments.json", "r") as f:
        full_data = json.load(f)
    sample_segments_data = full_data

    print("\n--- Input Data (first segment of first video) ---")
    first_video_key = next(iter(sample_segments_data))
    print(json.dumps(sample_segments_data[first_video_key]["0"], indent=2))
    print("--------------------------------------------------")

    extractor = VideoKnowledgeExtractor(video_path="/dummy/path/video.mp4", mcp_sessions={})

    print(f"Using working directory: {extractor.working_dir}")

    await extractor.video_segments.upsert(sample_segments_data)
    print(f"Loaded data for {len(extractor.video_segments._data)} video(s).")

    print("\nStep 2: Building graph and vector databases from segments using Gemini API...")
    await extractor.ainsert(extractor.video_segments._data)
    print("Data structure building complete.")


if __name__ == '__main__':
    asyncio.run(main())
