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
from ._op import chunking_by_video_segments, extract_entities, get_chunks
from ._llm import LLMConfig, local_llm_config, shutdown_local_llm


@dataclass
class VideoKnowledgeExtractor:
    """
    A class to extract knowledge from a video file.
    This test version simulates the post-processing part after VLM/ASR.
    """
    # --- Configuration ---
    video_path: str
    mcp_sessions: Dict[str, 'ClientSession'] = field(repr=False)
    working_dir: str = field(init=False)

    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True
    
    # --- LLM (runs locally) ---
    llm: LLMConfig = field(default_factory=lambda: local_llm_config)
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
    entity_extraction_func: callable = extract_entities
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    def __post_init__(self):
        """Initializes working directory and storage for the client side."""
        self.working_dir = f"./test_data_build_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(wrap_embedding_func_with_attrs(
                embedding_dim = self.llm.embedding_dim,
                max_token_size = self.llm.embedding_max_token_size,
                model_name = self.llm.embedding_model_name)(self.llm.embedding_func))

        # Create a serializable config dictionary by temporarily removing the un-serializable parts
        sessions = self.mcp_sessions
        self.mcp_sessions = None
        try:
            config_dict = asdict(self)
        finally:
            self.mcp_sessions = sessions

        # All storage is now on the client side, using the data processed by the server.
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
            # ---------- chunking
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

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            
            sessions = self.mcp_sessions
            self.mcp_sessions = None
            try:
                config_dict = asdict(self)
            finally:
                self.mcp_sessions = sessions

            # Pass the cache instance to the underlying functions
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
            # ---------- commit upsertings and indexing
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
    try:
        print("Starting data structure build test...")

        # Load the data that would normally come from the VLM/ASR server processing
        with open("knowledge_graph_build/kv_store_video_segments.json", "r") as f:
            full_data = json.load(f)
        # The extractor expects the full data structure, including the video name key.
        sample_segments_data = full_data

        print("\n--- Input Data (first segment of first video) ---")
        first_video_key = next(iter(sample_segments_data))
        print(json.dumps(sample_segments_data[first_video_key]["0"], indent=2))
        print("--------------------------------------------------")

        # Initialize the extractor. We don't need real mcp_sessions for this test.
        # A dummy video path is fine.
        extractor = VideoKnowledgeExtractor(video_path="/dummy/path/video.mp4", mcp_sessions={})

        print(f"Using working directory: {extractor.working_dir}")

        # Step 1: Load the data into the extractor's video_segments storage
        print("Step 1: Loading segments data into memory storage...")
        # Note: The `upsert` here is for the whole video collection
        await extractor.video_segments.upsert(sample_segments_data)
        print(f"Loaded data for {len(extractor.video_segments._data)} video(s).")

        # Step 2: Run the `ainsert` method to build the data structures
        print("\nStep 2: Building graph and vector databases from segments...")
        await extractor.ainsert(extractor.video_segments._data)
        print("Data structure building complete.")

        # Step 3: Verify the results
        # print("\n--- Verification ---")

        # # Check the graph
        # if extractor.chunk_entity_relation_graph:
        #     graph_data = extractor.chunk_entity_relation_graph.get_graph_data()
        #     print(f"\nGraph contains {len(graph_data['nodes'])} nodes and {len(graph_data['links'])} edges.")
        #     print("Graph Nodes:")
        #     for node in graph_data['nodes']:
        #         print(f"  - {node['id']} (type: {node.get('type', 'N/A')})")
        #     print("Graph Edges:")
        #     for link in graph_data['links']:
        #         print(f"  - {link['source']} -> {link['target']}")

        # # Check the entity VDB
        # if extractor.entities_vdb:
        #     try:
        #         num_entities = len(extractor.entities_vdb._db.index.get_uris())
        #         print(f"\nEntities VDB contains {num_entities} entities.")
        #         labels = extractor.entities_vdb._db.index.get_labels()
        #         print("Entities found:", labels)
        #     except Exception as e:
        #         print(f"Could not get entity count from VDB: {e}")


        # # Check the chunks VDB
        # if extractor.chunks_vdb:
        #     try:
        #         num_chunks = len(extractor.chunks_vdb._db.index.get_uris())
        #         print(f"\nChunks VDB contains {num_chunks} chunks.")
        #     except Exception as e:
        #         print(f"Could not get chunk count from VDB: {e}")

        # print("\n✅ Test finished successfully!")
    finally:
        # Ensure resources are released
        shutdown_local_llm()


if __name__ == '__main__':
    # To run this script, navigate to the `playground` directory and use:
    # python -m knowledge_graph_build.test_data_build
    asyncio.run(main())
