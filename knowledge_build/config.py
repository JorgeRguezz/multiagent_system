"""
Configuration for knowledge_build.
Keeps filenames consistent with chatbot_system integration.
"""
import os

PROJECT_ROOT = "/home/gatv-projects/Desktop/project"

# Extraction artifact filenames (same naming pattern as chatbot_system JsonKVStorage)
VIDEO_SEGMENTS_FILENAME = "kv_store_video_segments.json"
VIDEO_FRAMES_FILENAME = "kv_store_video_frames.json"
VIDEO_PATHS_FILENAME = "kv_store_video_path.json"

# Default output location for build artifacts
DEFAULT_BUILD_DIR = os.path.join(PROJECT_ROOT, "knowledge_build_cache")
