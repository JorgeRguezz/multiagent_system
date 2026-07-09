import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from knowledge_inference import config
from knowledge_inference.context_builder import make_evidence_blocks
from knowledge_inference.query_analyzer import analyze_query
from knowledge_inference.reranker import rerank_hits
from knowledge_inference.service import InferenceService
from knowledge_inference.types import EvidenceBlock, RetrievalHit

BASE_DIR = Path("knowledge_system_evaluation_v2")
ABLATIONS_FILE = BASE_DIR / "community_qa_dataset_ablations.json"
FINAL_FILE = BASE_DIR / "community_qa_dataset_final.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evidence_sources(evidence: list[EvidenceBlock | RetrievalHit]):
    return [e.source for e in evidence]


def evidence_contexts(evidence: list[EvidenceBlock | RetrievalHit]):
    contexts = []
    for item in evidence:
        text = getattr(item, "text", None)
        if text is None:
            text = getattr(item, "chunk_text", "")
        if isinstance(text, str) and text.strip():
            contexts.append(text)
    return contexts


def save_outputs(data):
    for path in (ABLATIONS_FILE, FINAL_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


def build_query(item):
    return f"{item.get('question_title', '')} {item.get('question_body', '')}".strip()


def validate_phase5_ready(data):
    retrieval_ablations = {"bm25", "vector_only", "graph_rag", "asr_only", "vision_only"}
    no_context_ablations = {"vanilla_base", "parametric", "sota_base"}
    expected_ablations = retrieval_ablations | no_context_ablations
    errors = []

    for idx, item in enumerate(data):
        ablations = item.get("ablations", {})
        missing = expected_ablations - set(ablations)
        if missing:
            errors.append(f"q_idx={idx} missing ablations: {sorted(missing)}")
            continue

        for ab_name in expected_ablations:
            ab_data = ablations.get(ab_name, {})
            if not ab_data.get("answer"):
                errors.append(f"q_idx={idx}/{ab_name} missing answer")
            if "evidence_sources" not in ab_data:
                errors.append(f"q_idx={idx}/{ab_name} missing evidence_sources")
            if "evidence_contexts" not in ab_data:
                errors.append(f"q_idx={idx}/{ab_name} missing evidence_contexts")
                continue
            contexts = ab_data.get("evidence_contexts")
            if not isinstance(contexts, list):
                errors.append(f"q_idx={idx}/{ab_name} evidence_contexts is not a list")
            elif ab_name in retrieval_ablations and not contexts:
                errors.append(f"q_idx={idx}/{ab_name} has empty evidence_contexts")
            elif ab_name in no_context_ablations and contexts:
                errors.append(f"q_idx={idx}/{ab_name} should have empty evidence_contexts")

    if errors:
        sample = "\n".join(errors[:20])
        raise RuntimeError(
            f"Phase 5 readiness validation failed with {len(errors)} errors:\n{sample}"
        )


async def evidence_for_hits(service, query, hits):
    intent = analyze_query(query)
    ranked_hits = rerank_hits(
        hits=hits,
        query=query,
        intent=intent,
        available_videos=list(service.stores.keys()),
    )
    return make_evidence_blocks(
        hits=ranked_hits,
        stores=service.stores,
        budget_tokens=config.MAX_CONTEXT_TOKENS,
    )


def split_chunk_text(text: str) -> tuple[str, str]:
    caption_idx = text.find("Caption:")
    transcript_idx = text.find("Transcript:")
    
    caption_text = ""
    transcript_text = ""
    
    if caption_idx != -1 and transcript_idx != -1:
        if caption_idx < transcript_idx:
            caption_text = text[caption_idx:transcript_idx].replace("Caption:", "").strip()
            transcript_text = text[transcript_idx:].replace("Transcript:", "").strip()
        else:
            transcript_text = text[transcript_idx:caption_idx].replace("Transcript:", "").strip()
            caption_text = text[caption_idx:].replace("Caption:", "").strip()
    elif caption_idx != -1:
        caption_text = text.replace("Caption:", "").strip()
    elif transcript_idx != -1:
        transcript_text = text.replace("Transcript:", "").strip()
    else:
        transcript_text = text
        
    return transcript_text, caption_text

async def process_modality_ablations(contexts_only=False, test_mode=False):
    service = InferenceService()
    service.initialize()
    
    logger.info("Extracting modalities from chunks...")
    chunk_refs = []
    asr_texts = []
    vlm_texts = []
    
    for store_name, store in service.stores.items():
        for chunk_id, chunk in store.chunks_kv.items():
            text = str(chunk.get("content", ""))
            asr, vlm = split_chunk_text(text)
            
            chunk_refs.append((store, chunk_id, chunk.get("video_segment_id", [])))
            asr_texts.append(asr)
            vlm_texts.append(vlm)
            
    logger.info("Loading embedding model...")
    # Normalize embeddings to easily compute cosine similarity via dot product
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    
    logger.info("Embedding ASR texts...")
    asr_embeddings = model.encode(asr_texts, normalize_embeddings=True)
    
    logger.info("Embedding VLM texts...")
    vlm_embeddings = model.encode(vlm_texts, normalize_embeddings=True)
    
    def retrieve_top_k(query_emb, doc_embeddings, texts, k=15):
        scores = np.dot(doc_embeddings, query_emb)
        top_n = np.argsort(scores)[::-1][:k]
        
        hits = []
        for i in top_n:
            if not texts[i].strip():
                continue
            store, chunk_id, seg_ids = chunk_refs[i]
            if isinstance(seg_ids, str):
                seg_ids = [seg_ids]
            hits.append(RetrievalHit(
                chunk_id=chunk_id,
                video_name=store.video_name,
                source="vector_modality",
                chunk_text=texts[i],
                segment_ids=[str(x) for x in seg_ids],
                score_semantic=float(scores[i]),
                score_entity=0.0,
                score_graph=0.0,
            ))
        return hits[:k]
        
    input_file = FINAL_FILE if FINAL_FILE.exists() else ABLATIONS_FILE
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if test_mode:
        data = data[:2]
        
    for i, item in enumerate(data):
        logger.info(f"Processing {i+1}/{len(data)}: {item.get('question_title', 'Unknown')}")
        
        # Skip only if the modality answers and their real evidence contexts are already present.
        ablations = item.get("ablations", {})
        asr_done = "asr_only" in ablations and bool(ablations["asr_only"].get("evidence_contexts"))
        vision_done = "vision_only" in ablations and bool(ablations["vision_only"].get("evidence_contexts"))
        if not contexts_only and asr_done and vision_done:
            continue
            
        query = build_query(item)
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        
        # ASR Only
        asr_hits = retrieve_top_k(query_emb, asr_embeddings, asr_texts)
        asr_evidence = await evidence_for_hits(service, query, asr_hits)

        # Vision Only
        vlm_hits = retrieve_top_k(query_emb, vlm_embeddings, vlm_texts)
        vlm_evidence = await evidence_for_hits(service, query, vlm_hits)

        item.setdefault("ablations", {})
        if contexts_only:
            item["ablations"].setdefault("asr_only", {})
            item["ablations"].setdefault("vision_only", {})
            item["ablations"]["asr_only"]["evidence_sources"] = evidence_sources(asr_evidence)
            item["ablations"]["asr_only"]["evidence_contexts"] = evidence_contexts(asr_evidence)
            item["ablations"]["vision_only"]["evidence_sources"] = evidence_sources(vlm_evidence)
            item["ablations"]["vision_only"]["evidence_contexts"] = evidence_contexts(vlm_evidence)
        else:
            async def mock_asr(*args, **kwargs):
                return asr_hits
            with patch("knowledge_inference.service.retrieve_all", new=mock_asr):
                asr_res = await service._answer_async(query)

            async def mock_vlm(*args, **kwargs):
                return vlm_hits
            with patch("knowledge_inference.service.retrieve_all", new=mock_vlm):
                vlm_res = await service._answer_async(query)

            item["ablations"]["asr_only"] = {
                "answer": asr_res.answer,
                "evidence_sources": evidence_sources(asr_res.evidence),
                "evidence_contexts": evidence_contexts(asr_res.evidence)
            }
            item["ablations"]["vision_only"] = {
                "answer": vlm_res.answer,
                "evidence_sources": evidence_sources(vlm_res.evidence),
                "evidence_contexts": evidence_contexts(vlm_res.evidence)
            }
        
        # Save checkpoints safely
        if not test_mode and ((i + 1) % 5 == 0 or (i + 1) == len(data)):
            save_outputs(data)
                
    if not test_mode:
        validate_phase5_ready(data)
        save_outputs(data)
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run modality ablations or enrich saved modality evidence contexts.")
    parser.add_argument("--test", action="store_true", help="Run on the first 2 questions without saving.")
    parser.add_argument("--contexts-only", action="store_true", help="Only refresh ASR/Vision evidence_sources/evidence_contexts; keep existing answers.")
    args = parser.parse_args()

    asyncio.run(process_modality_ablations(contexts_only=args.contexts_only, test_mode=args.test))
