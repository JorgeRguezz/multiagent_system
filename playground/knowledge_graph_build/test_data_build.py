from datetime import datetime
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Type, cast, Callable, List, Dict, Union, Optional
import asyncio
import copy
import sys
import subprocess

# Monkeypatch for torchvision >= 0.17 compatibility with pytorchvideo
try:
    import torchvision
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F
except ImportError:
    pass

# Mock vllm if not installed (required by _llm.py but not used by GPTModel)
try:
    import vllm
except ImportError:
    from unittest.mock import MagicMock
    m = MagicMock()
    m.LLM = MagicMock
    m.SamplingParams = MagicMock
    sys.modules["vllm"] = m

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

# Imports for patching and vLLM
from vllm import LLM
from . import _llm as llm_module
from . import _op as op_module


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


async def run_extraction_test(model_label: str, llm_config: LLMConfig, sample_segments_data: dict):
    """Runs the extraction process for a specific model configuration."""
    print(f"\n{'='*50}")
    print(f"Running Extraction Test with: {model_label}")
    print(f"{ '='*50}")
    
    # Initialize the extractor with specific config
    # We create a new lambda for default factory to avoid closure issues if config changes
    extractor = VideoKnowledgeExtractor(
        video_path="/dummy/path/video.mp4", 
        mcp_sessions={},
        llm=llm_config,
        enable_llm_cache=False # Disable cache to ensure model runs
    )

    print(f"Using working directory: {extractor.working_dir}")

    # Step 1: Load the data into the extractor's video_segments storage
    print("Step 1: Loading segments data into memory storage...")
    await extractor.video_segments.upsert(sample_segments_data)
    print(f"Loaded data for {len(extractor.video_segments._data)} video(s).")

    # Step 2: Run the `ainsert` method to build the data structures
    print("\nStep 2: Building graph and vector databases from segments...")
    await extractor.ainsert(extractor.video_segments._data)
    print(f"Data structure building complete for {model_label}.")



# Add current directory to path to allow importing init_models if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import init_models
except ImportError:
    # If running as -m knowledge_graph_build.test_data_build
    from . import init_models

# Select the model to use for knowledge extraction
# Options: "llama", "deepseek", "apriel", "gpt"
MODEL_SELECTION = "gpt" 

# ... (Existing imports remain at top of file, this replaces the patching/main section)

def apply_model_patches(model_name):
    """
    Initializes the selected model using init_models and patches the _llm and _op modules
    to use this model for generation.
    """
    print(f"Initializing and patching for model: {model_name}")
    
    # 1. Load the model instance
    try:
        model_instance = init_models.load_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise e

    # 2. Define patched functions
    
    # Patch for batch generation (used by extract_entities in _op.py)
    async def patched_batch_generate(model_path, prompts):
        # We ignore model_path as we use the pre-loaded model_instance
        results = []
        # Simple loop for now. If models support batching, init_models could be improved.
        for p in tqdm(prompts, desc="LLM Batch Generation"):
            res = model_instance.generate(p)
            results.append(res)
        return results

    # Patch for single completion (used by summaries in _llm.py)
    async def patched_llm_complete(model_name, prompt, system_prompt=None, **kwargs):
        return model_instance.generate(prompt, system_prompt=system_prompt)

    # 3. Apply patches to modules
    op_module.local_llm_batch_generate = patched_batch_generate
    llm_module.local_llm_complete = patched_llm_complete
    
    # 4. Update the LLMConfig instance
    # We must update the raw functions in the config object because it holds references 
    # to the original functions.
    cfg = llm_module.local_llm_config
    
    cfg.best_model_func_raw = patched_llm_complete
    cfg.cheap_model_func_raw = patched_llm_complete
    
    # Re-bind the convenience lambdas
    cfg.best_model_func = lambda prompt, *args, **kwargs: cfg.best_model_func_raw(
        cfg.best_model_name, prompt, *args, **kwargs
    )
    cfg.cheap_model_func = lambda prompt, *args, **kwargs: cfg.cheap_model_func_raw(
        cfg.cheap_model_name, prompt, *args, **kwargs
    )
    
    print(f"Successfully patched system to use {model_name}")


async def main():
    """Main function to test the data building with the selected model."""
    try:
        print(f"Starting data structure build using model: {MODEL_SELECTION}")

        # Load the data
        cache_path = "/home/gatv-projects/Desktop/project/lol_test_cache/lol_history_cache/kv_store_video_segments.json"
        if not os.path.exists(cache_path):
             print(f"Error: Cache file not found at {cache_path}")
             return

        with open(cache_path, "r") as f:
            full_data = json.load(f)
        sample_segments_data = full_data

        print("\n--- Input Data (first segment of first video) ---")
        first_video_key = next(iter(sample_segments_data))
        print(json.dumps(sample_segments_data[first_video_key]["0"], indent=2))
        print("--------------------------------------------------")

        # Apply patches for the selected model
        apply_model_patches(MODEL_SELECTION)
        
        # Run extraction
        # We use the (now patched) local_llm_config
        await run_extraction_test(f"Model: {MODEL_SELECTION}", llm_module.local_llm_config, sample_segments_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # To run this script, navigate to the `playground` directory and use:
    # python -m knowledge_graph_build.test_data_build
    from tqdm import tqdm # ensure tqdm is available
    asyncio.run(main())
