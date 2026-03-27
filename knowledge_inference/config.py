from __future__ import annotations

from pathlib import Path

# Project root inferred from package location. Keeps all path handling absolute.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Sanitized cache root enforced for all inference reads.
SANITIZED_CACHE_ROOT = PROJECT_ROOT / "knowledge_sanitization" / "cache"

# Per-video cache naming convention used by sanitization stage.
SANITIZED_BUILD_GLOB = "sanitized_build_cache_*"

# Global graph emitted by sanitization/global merge.
SANITIZED_GLOBAL_GRAPH = SANITIZED_CACHE_ROOT / "sanitized_global" / "graph_AetherNexus.graphml"

# Global manifest for processed videos.
SANITIZED_GLOBAL_MANIFEST = SANITIZED_CACHE_ROOT / "sanitized_global" / "aether_manifest.json"

# Optional per-video metadata registry used for final answer formatting.
VIDEO_METADATA_REGISTRY = PROJECT_ROOT / "knowledge_inference" / "video_metadata.json"

# Dense retrieval depth over chunk embeddings.
TOP_K_CHUNKS_DENSE = 30

# Dense retrieval depth over entity embeddings.
TOP_K_ENTITIES_DENSE = 20

# Graph-derived chunk candidates to keep.
TOP_K_GRAPH_CHUNKS = 30

# Final number of evidence items passed to answer generation.
FINAL_EVIDENCE_K = 14

# Weighted ranker: semantic match from chunk embedding retrieval.
W_SEMANTIC = 0.50

# Weighted ranker: lexical entity/query overlap signal.
W_ENTITY = 0.20

# Weighted ranker: graph-support signal from local/global graph traversals.
W_GRAPH = 0.20

# Weighted ranker: diversity signal to avoid single-video collapse.
W_DIVERSITY = 0.10

# Context budget passed into the prompt (requested setting).
MAX_CONTEXT_TOKENS = 10000

# Max generated answer tokens (requested setting).
MAX_ANSWER_TOKENS = 2500

# Minimal score for evidence to be considered reliable enough for generation.
MIN_EVIDENCE_SCORE = 0.18

# Minimal supported-claims ratio expected after verification.
MIN_SUPPORTED_CLAIMS_RATIO = 0.45

# Latency threshold used by regression checks/eval warninging.
MAX_ACCEPTABLE_LATENCY_SECONDS = 45.0

# Retrieval policy defaults.
DEFAULT_MAX_PER_VIDEO = 4
CROSS_VIDEO_MAX_PER_VIDEO = 3

# Toggle post-generation verification. When disabled, the raw generated answer
# still flows through answer post-processing, but no verifier pruning or
# cautioning is applied.
ENABLE_VERIFIER = False

# GPT-OSS generation defaults mirroring knowledge_build/_llm.py usage.
GEN_TEMPERATURE = 0.1
GEN_TOP_P = 1.0
GEN_TOP_K = 0
GEN_REPEAT_PENALTY = 1.12

# Verifier generation defaults. Lower temperature for stable classifications.
VERIFIER_TEMPERATURE = 0.0
VERIFIER_TOP_P = 1.0
VERIFIER_TOP_K = 0
VERIFIER_REPEAT_PENALTY = 1.05
VERIFIER_MAX_TOKENS = 1200

# Logging namespace.
LOGGER_NAME = "knowledge-inference"
