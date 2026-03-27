"""
Central configuration file for the knowledge extraction pipeline, defining paths,
HUD regions, and processing parameters.
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_SMOLVLM_PYTHON = str(
    Path(
        os.environ.get(
            "VENV_SMOLVLM_PYTHON",
            PROJECT_ROOT / "venv_smolvlm" / "bin" / "python3",
        )
    )
)
# VENV_VLM_ASR_PYTHON = os.path.join(PROJECT_ROOT, "venv_qwen_vlm/bin/python3")
VENV_VLM_ASR_PYTHON = str(
    Path(
        os.environ.get(
            "VENV_VLM_ASR_PYTHON",
            PROJECT_ROOT / "venv_internVL" / "bin" / "python3",
        )
    )
)

HF_HOME = os.environ.get("HF_HOME")

# Video Processing Config
VIDEO_PATH = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_EXTRACTION_VIDEO_PATH",
            PROJECT_ROOT / "downloads" / "The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends.mp4",
        )
    )
)
WORKING_DIR = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_EXTRACTION_WORKING_DIR",
            PROJECT_ROOT / "knowledge_extraction" / "cache",
        )
    )
)
DB_PATH = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_EXTRACTION_DB_PATH",
            PROJECT_ROOT / "knowledge_extraction" / "image_matching" / "lol_champions_square_224.nvdb",
        )
    )
)
CHAMPION_ASSETS_ROOT = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_EXTRACTION_CHAMPION_ASSETS_ROOT",
            PROJECT_ROOT / "knowledge_extraction" / "image_matching" / "assets" / "champions",
        )
    )
)

SEGMENT_LENGTH = 30
FRAMES_PER_SEGMENT = 5
# MAX_SEGMENTS = 6
NUM_ENTITY_WORKERS = 3

# HUD Regions [x0, y0, x1, y1]
MIDDLE_HUD = [500, 900, 1350, 1080]
BOTTOM_RIGHT_HUD = [1500, 600, 1923, 750]
