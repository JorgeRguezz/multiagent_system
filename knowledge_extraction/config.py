"""
Central configuration file for the knowledge extraction pipeline, defining paths, 
HUD regions, and processing parameters.
"""
import os

# Paths
PROJECT_ROOT = "/home/gatv-projects/Desktop/project"
VENV_SMOLVLM_PYTHON = os.path.join(PROJECT_ROOT, "venv_smolvlm/bin/python3")
# VENV_VLM_ASR_PYTHON = os.path.join(PROJECT_ROOT, "venv_qwen_vlm/bin/python3")
VENV_VLM_ASR_PYTHON = os.path.join(PROJECT_ROOT, "venv_internVL/bin/python3")

HF_HOME = "/media/gatv-projects/2C76BE277A0C0B8F/AI_models/huggingface_cache"

# Video Processing Config
VIDEO_PATH = os.path.join(PROJECT_ROOT, "downloads/The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends.mp4")
WORKING_DIR = os.path.join(PROJECT_ROOT, "knowledge_extraction/cache")
DB_PATH = os.path.join(PROJECT_ROOT, "knowledge_extraction/image_matching/lol_champions_square_224.nvdb")

SEGMENT_LENGTH = 30
FRAMES_PER_SEGMENT = 5
MAX_SEGMENTS = 6
NUM_ENTITY_WORKERS = 3

# HUD Regions [x0, y0, x1, y1]
MIDDLE_HUD = [500, 900, 1350, 1080]
BOTTOM_RIGHT_HUD = [1500, 600, 1923, 750]
