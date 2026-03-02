"""Schemas for knowledge_build input artifacts."""
from typing import TypedDict, List, Dict


class VideoSegmentSchema(TypedDict):
    time: str
    content: str
    transcript: str
    frame_times: List[float]


class VideoFrameSchema(TypedDict, total=False):
    frame_path: str
    segment_idx: str
    segment_name: str
    frame_idx: int
    main_champ: str
    partners: List[str]
    transcript: str
    vlm_output: str
    timestamp: float


VideoSegmentsFile = Dict[str, Dict[str, VideoSegmentSchema]]
VideoFramesFile = Dict[str, Dict[str, VideoFrameSchema]]
