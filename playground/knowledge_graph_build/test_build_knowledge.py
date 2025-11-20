from pathlib import Path
import torch
from datetime import datetime
import os
import json
from tqdm import tqdm
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from dataclasses import dataclass, field, asdict
from typing import Type, cast, Callable, List, Optional, Dict, Union
import asyncio
import shutil

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

# Local project imports
from ._storage import NanoVectorDBVideoSegmentStorage, JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from .base import BaseVectorStorage, StorageNameSpace, BaseKVStorage, BaseGraphStorage
from ._utils import logger, always_get_an_event_loop, limit_async_func_call, wrap_embedding_func_with_attrs
from ._op import chunking_by_video_segments, extract_entities, get_chunks
from ._llm import LLMConfig, local_llm_config # Use local_llm_config


@dataclass
class VideoKnowledgeExtractor:
    """
    A class to extract knowledge from a video file using a remote MCP tool.
    The heavy processing (ASR, VLM) is done on the server side.
    """
    # --- Configuration ---
    video_path: str
    mcp_sessions: Dict[str, ClientSession] = field(repr=False)
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
        self.working_dir = f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
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

    async def run_extraction_pipeline(self):
        """
        Runs the video knowledge extraction pipeline by calling the remote MCP tool.
        """
        print("Starting video knowledge extraction process...")
        
        # Step 1: Call the high-level MCP tool to process the video on the server
        print("Step 1: Calling remote server to process video...")
        video_processor_session = self.mcp_sessions.get('extract_video_knowledge')
        if not video_processor_session:
            raise RuntimeError("MCP tool 'extract_video_knowledge' not found in connected servers.")

        video_abs_path = os.path.abspath(self.video_path)
        result = await video_processor_session.call_tool('extract_video_knowledge', arguments={'video_path': video_abs_path})
        
        if not result.content:
            raise RuntimeError("MCP tool returned no content.")
            
        response_data = json.loads(result.content[0].text)
        if "error" in response_data:
            raise RuntimeError(f"Server-side error: {response_data['error']}")

        video_segments_db_path = response_data.get("video_segments_db_path")
        if not video_segments_db_path or not os.path.exists(video_segments_db_path):
            raise FileNotFoundError(f"Server did not return a valid path for the video segments database. Got: {video_segments_db_path}")
        
        print(f"Step 2: Server processing complete. Loading data from '{video_segments_db_path}'")

        # Step 2: Load the processed data from the JSON file returned by the server
        with open(video_segments_db_path, 'r') as f:
            segments_data = json.load(f)
        
        # The data is now in self.video_segments, which can be used by the next step
        await self.video_segments.upsert(segments_data)

        print("\n" + "<>"*20)
        print("Final Extracted Information (from server):")
        print(json.dumps(segments_data, indent=4, ensure_ascii=False))
        print("<>"*20 + "\n")

        # Step 3: Build graph and vector databases locally
        print("Step 3: Building graph and vector databases...")
        await self.ainsert(self.video_segments._data)

        # Step 4: Clean up the server-side cache directory
        server_working_dir = response_data.get("working_dir")
        if server_working_dir and os.path.exists(server_working_dir):
            print(f"Step 4: Deleting server cache directory: {server_working_dir}")
            shutil.rmtree(server_working_dir)
            print("Server cache deleted.")

        print("\n✅ Pipeline finished successfully!")


async def main():
    """Main function to set up MCP client and run the extraction."""
    VIDEO_FILE = "/home/gatv-projects/Desktop/project/chatbot_system/downloads/My_Nintendo_Switch_2_Review.mp4"
    
    async with AsyncExitStack() as exit_stack:
        sessions = {}
        
        try:
            with open(Path(__file__).parent/"server_config.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
        except FileNotFoundError:
            print("Error: `server_config.json` not found. Please create it.")
            return
        except Exception as e:
            print(f"Error loading server_config.json: {e}")
            return

        vlm_server_config = servers.get('vlm_server')
        if not vlm_server_config:
            print("Error: 'vlm_server' not found in server_config.json. Please add it to connect to your media processing server.")
            return

        try:
            print("Connecting to Media Processing server via MCP...")
            server_params = StdioServerParameters(**vlm_server_config)
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            
            response = await session.list_tools()
            if not response.tools:
                raise RuntimeError("No tools found on the connected MCP server.")

            for tool in response.tools:
                print(f"[DIAGNOSTIC] Found tool: {tool.name}")
                sessions[tool.name] = session
            
            if 'extract_video_knowledge' not in sessions:
                raise RuntimeError("Required tool 'extract_video_knowledge' not found on the server.")

            print("Successfully connected to Media Processing server.")

        except Exception as e:
            print(f"Error connecting to Media Processing server: {e}")
            return

        # Initialize and run the extractor
        extractor = VideoKnowledgeExtractor(video_path=VIDEO_FILE, mcp_sessions=sessions)
        await extractor.run_extraction_pipeline()


if __name__ == '__main__':
    # To run this script, navigate to the `playground` directory and use:
    # python -m knowledge_graph_build.test_build_knowledge
    asyncio.run(main())