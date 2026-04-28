"""
Shared game-profile registry for game-specific pipeline behavior.

Phase 1 only defines the profile contract and initial profiles.
Runtime modules will switch to this registry in later phases.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DetectionMode = Literal["roi", "full_frame"]
EntityResultParser = Literal["lol", "generic"]


@dataclass(frozen=True)
class VlmPromptProfile:
    qwen_description_prompt: str
    internvl_description_prompt: str


@dataclass(frozen=True)
class GameProfile:
    id: str
    display_name: str
    detection_mode: DetectionMode
    regions_config: tuple[dict, ...]
    entity_db_path: str
    entity_result_parser: EntityResultParser
    vlm_prompt_profile: VlmPromptProfile
    grounded_qa_system_prompt: str


LOL_ENTITY_DB_PATH = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_EXTRACTION_DB_PATH",
            PROJECT_ROOT / "knowledge_extraction" / "image_matching" / "lol_champions_square_224.nvdb",
        )
    )
)

OTHER_ENTITY_DB_PATH = str(
    Path(
        os.environ.get(
            "KNOWLEDGE_EXTRACTION_OTHER_DB_PATH",
            PROJECT_ROOT / "knowledge_extraction" / "image_matching" / "test_vdb.nvdb",
        )
    )
)


# HUD regions [x0, y0, x1, y1]
MIDDLE_HUD = [500, 900, 1350, 1080]
BOTTOM_RIGHT_HUD = [1500, 600, 1923, 750]


LOL_QWEN_DESCRIPTION_PROMPT = """
You are a vision-language analyst for League of Legends gameplay.
Describe the current frame using the supplied gameplay context and transcript.
Focus on visible champions, combat state, landmarks, health bars, abilities,
items, and any gameplay-relevant information grounded in the frame.
Do not invent hidden information or unsupported interactions.
Return continuous descriptive text in paragraphs.
""".strip()

LOL_INTERNVL_DESCRIPTION_PROMPT = """
You are a vision-language analyst for League of Legends gameplay.
Describe the current frame using the supplied gameplay context and transcript.
Prioritize visible champions, map landmarks, health state, combat intensity,
abilities, and gameplay-relevant UI cues. Stay grounded in the image.
Return continuous descriptive text in paragraphs.
""".strip()

GENERIC_QWEN_DESCRIPTION_PROMPT = """
You are a vision-language analyst for gameplay video.
Describe the current frame using the supplied context and transcript.
Focus on visible characters, entities, actions, environment, UI cues, and any
gameplay-relevant events grounded in the frame.
Do not invent unsupported details or game-specific assumptions.
Return continuous descriptive text in paragraphs.
""".strip()

GENERIC_INTERNVL_DESCRIPTION_PROMPT = """
You are a vision-language analyst for gameplay video.
Describe the current frame using the supplied context and transcript.
Prioritize visible entities, actions, environment state, and UI cues while
remaining grounded in the frame.
Return continuous descriptive text in paragraphs.
""".strip()

LOL_SYSTEM_GROUNDED_QA = """You are a grounded QA League of Legends assistant.

Reasoning: High

Rules:
1. Use ONLY the provided evidence context.
2. Do not fabricate facts, times, or sources.
3. If evidence is insufficient or conflicting, say so explicitly.
4. Cite provenance inline using compact references like (video=<name>, time=<range>).
5. Prefer precise, concise answers.

<|channel|>analysis<|message|>[user request]. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>[your response]<|return|>
"""

GENERIC_SYSTEM_GROUNDED_QA = """You are a grounded gameplay QA assistant.

Reasoning: High

Rules:
1. Use ONLY the provided evidence context.
2. Do not fabricate facts, times, sources, or game-specific details.
3. If evidence is insufficient or conflicting, say so explicitly.
4. Cite provenance inline using compact references like (video=<name>, time=<range>).
5. Prefer precise, concise answers grounded in the evidence.

<|channel|>analysis<|message|>[user request]. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>[your response]<|return|>
"""


GAME_PROFILES: dict[str, GameProfile] = {
    "league_of_legends": GameProfile(
        id="league_of_legends",
        display_name="League of Legends",
        detection_mode="roi",
        regions_config=(
            {"name": "middle", "region": MIDDLE_HUD},
            {"name": "partners", "region": BOTTOM_RIGHT_HUD},
        ),
        entity_db_path=LOL_ENTITY_DB_PATH,
        entity_result_parser="lol",
        vlm_prompt_profile=VlmPromptProfile(
            qwen_description_prompt=LOL_QWEN_DESCRIPTION_PROMPT,
            internvl_description_prompt=LOL_INTERNVL_DESCRIPTION_PROMPT,
        ),
        grounded_qa_system_prompt=LOL_SYSTEM_GROUNDED_QA,
    ),
    "other": GameProfile(
        id="other",
        display_name="Other",
        detection_mode="full_frame",
        regions_config=(
            {"name": "entities", "region": None},
        ),
        entity_db_path=OTHER_ENTITY_DB_PATH,
        entity_result_parser="generic",
        vlm_prompt_profile=VlmPromptProfile(
            qwen_description_prompt=GENERIC_QWEN_DESCRIPTION_PROMPT,
            internvl_description_prompt=GENERIC_INTERNVL_DESCRIPTION_PROMPT,
        ),
        grounded_qa_system_prompt=GENERIC_SYSTEM_GROUNDED_QA,
    ),
}


VIDEO_GAME = os.environ.get("VIDEO_GAME", "league_of_legends")


def get_active_game_profile(video_game: str | None = None) -> GameProfile:
    profile_id = video_game or VIDEO_GAME
    try:
        return GAME_PROFILES[profile_id]
    except KeyError as exc:
        supported = ", ".join(sorted(GAME_PROFILES))
        raise ValueError(
            f"Unsupported VIDEO_GAME={profile_id!r}. Supported values: {supported}"
        ) from exc
