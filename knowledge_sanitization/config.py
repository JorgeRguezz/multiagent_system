import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SANITIZATION_ROOT = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_SANITIZATION_ROOT",
            PROJECT_ROOT / "knowledge_sanitization" / "cache",
        )
    )
)

EXTRACTION_CACHE_ROOT = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_EXTRACTION_CACHE_ROOT",
            PROJECT_ROOT / "knowledge_extraction" / "cache",
        )
    )
)
EXTRACTED_DATA_ROOT = os.path.join(EXTRACTION_CACHE_ROOT, "extracted_data")
SANITIZED_EXTRACTED_DATA_ROOT = os.path.join(SANITIZATION_ROOT, "sanitized_extracted_data")

BUILD_CACHE_PREFIX = "knowledge_build_cache_"
SANITIZED_BUILD_CACHE_PREFIX = "sanitized_build_cache_"
SANITIZED_GLOBAL_ROOT = os.path.join(SANITIZATION_ROOT, "sanitized_global")

SPEC_ROOT = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_SANITIZATION_SPEC_ROOT",
            PROJECT_ROOT / "knowledge_sanitization" / "spec",
        )
    )
)
REPORT_PRE_ROOT = os.path.join(SANITIZATION_ROOT, "reports", "pre_build")
REPORT_POST_ROOT = os.path.join(SANITIZATION_ROOT, "reports", "post_build")
QUAR_PRE_ROOT = os.path.join(SANITIZATION_ROOT, "quarantine", "pre_build")
QUAR_POST_ROOT = os.path.join(SANITIZATION_ROOT, "quarantine", "post_build")

ALLOWED_ENTITY_TYPES = {"PERSON", "GEO", "EVENT", "ORGANIZATION", "UNKNOWN"}

REQUIRED_EXTRACTION_FILES = [
    "kv_store_video_segments.json",
    "kv_store_video_frames.json",
    "kv_store_video_path.json",
]

REQUIRED_BUILD_FILES = [
    "kv_store_text_chunks.json",
    "kv_store_video_segments.json",
    "kv_store_video_frames.json",
    "kv_store_video_path.json",
    "graph_chunk_entity_relation.graphml",
    "graph_chunk_entity_relation_clean.graphml",
]
