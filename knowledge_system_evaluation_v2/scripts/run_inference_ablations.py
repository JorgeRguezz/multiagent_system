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

import networkx as nx

from rank_bm25 import BM25Okapi

from knowledge_inference import config
from knowledge_inference.context_builder import make_evidence_blocks
from knowledge_inference.query_analyzer import analyze_query
from knowledge_inference.reranker import rerank_hits
from knowledge_inference.retrievers import retrieve_all
from knowledge_inference.service import InferenceService
from knowledge_inference.types import EvidenceBlock, RetrievalHit
from knowledge_build._llm import local_llm_config
from knowledge_inference.retrievers import retrieve_chunks_dense

BASE_DIR = Path("knowledge_system_evaluation_v2")
ANSWERABILITY_FILE = BASE_DIR / "community_qa_dataset_answerability.json"
ABLATIONS_FILE = BASE_DIR / "community_qa_dataset_ablations.json"
FINAL_FILE = BASE_DIR / "community_qa_dataset_final.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_bm25_index(stores):
    logger.info("Building BM25 Index across all stores...")
    corpus = []
    chunk_refs = []
    for store_name, store in stores.items():
        for chunk_id, chunk in store.chunks_kv.items():
            text = str(chunk.get("content", ""))
            corpus.append(text.lower().split())
            chunk_refs.append((store, chunk_id, text, chunk.get("video_segment_id", [])))
            
    bm25 = BM25Okapi(corpus)
    logger.info(f"BM25 Index built with {len(corpus)} chunks.")
    return bm25, chunk_refs

async def retrieve_bm25_mock(query=None, intent=None, stores=None, global_graph=None, bm25_index=None, chunk_refs=None, **kwargs):
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:15]
    
    hits = []
    for i in top_n:
        store, chunk_id, text, seg_ids = chunk_refs[i]
        if isinstance(seg_ids, str):
            seg_ids = [seg_ids]
        hits.append(RetrievalHit(
            chunk_id=chunk_id,
            video_name=store.video_name,
            source="bm25",
            chunk_text=text,
            segment_ids=[str(x) for x in seg_ids],
            score_semantic=scores[i],
            score_entity=0.0,
            score_graph=0.0,
        ))
    return hits

async def retrieve_vector_mock(query=None, intent=None, stores=None, global_graph=None, **kwargs):
    # Call only the dense retriever from our actual retrievers module
    return await retrieve_chunks_dense(query, stores, k=15)

async def retrieve_parametric_mock(query=None, intent=None, stores=None, global_graph=None, **kwargs):
    return []

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


def load_existing_sota_base():
    if not FINAL_FILE.exists():
        return {}
    with open(FINAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["question_id"]: item.get("ablations", {}).get("sota_base")
        for item in data
        if item.get("question_id") and item.get("ablations", {}).get("sota_base")
    }


def attach_sota_base(ablations_output, item, existing_sota):
    sota_base = existing_sota.get(item.get("question_id"))
    if not sota_base:
        return
    sota_base.setdefault("evidence_sources", [])
    sota_base.setdefault("evidence_contexts", [])
    ablations_output["sota_base"] = sota_base


def build_query(item):
    return f"{item.get('question_title', '')} {item.get('question_body', '')}".strip()


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


async def graph_evidence(service, query):
    intent = analyze_query(query)
    hits = await retrieve_all(
        query=query,
        intent=intent,
        stores=service.stores,
        global_graph=service.global_graph if service.global_graph is not None else nx.Graph(),
    )
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


async def run_vanilla_base(query):
    # Vanilla chatbot experience, absolutely no RAG system prompt or context,
    # but we DO provide the formatting rules so it outputs correctly instead of rambling.
    vanilla_system_prompt = """You are a helpful AI assistant.

Reasoning: High

Rules:
1. Answer the user's question directly and concisely.

<|channel|>analysis<|message|>[user request]. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>[your response]<|return|>
"""
    res = await local_llm_config.best_model_func(
        query, 
        system_prompt=vanilla_system_prompt,
        return_metadata=True
    )
    return str(res.get("answer", "")).strip()

async def populate_contexts_only(test_mode=False):
    if not FINAL_FILE.exists():
        raise FileNotFoundError(f"Context-only mode requires existing {FINAL_FILE}")

    service = InferenceService()
    service.initialize()
    bm25_index, chunk_refs = build_bm25_index(service.stores)

    with open(FINAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if test_mode:
        data = data[:2]

    for i, item in enumerate(data):
        logger.info(f"Context-only {i+1}/{len(data)}: {item.get('question_title', 'Unknown')}")
        query = build_query(item)
        ablations = item.setdefault("ablations", {})

        for ab_name in ("vanilla_base", "parametric", "sota_base"):
            if ab_name in ablations:
                ablations[ab_name].setdefault("evidence_sources", [])
                ablations[ab_name]["evidence_contexts"] = []

        bm25_hits = await retrieve_bm25_mock(
            query=query,
            stores=service.stores,
            global_graph=service.global_graph,
            bm25_index=bm25_index,
            chunk_refs=chunk_refs,
        )
        bm25_evidence = await evidence_for_hits(service, query, bm25_hits)
        if "bm25" in ablations:
            ablations["bm25"]["evidence_sources"] = evidence_sources(bm25_evidence)
            ablations["bm25"]["evidence_contexts"] = evidence_contexts(bm25_evidence)

        vector_hits = await retrieve_vector_mock(query=query, stores=service.stores, global_graph=service.global_graph)
        vector_evidence = await evidence_for_hits(service, query, vector_hits)
        if "vector_only" in ablations:
            ablations["vector_only"]["evidence_sources"] = evidence_sources(vector_evidence)
            ablations["vector_only"]["evidence_contexts"] = evidence_contexts(vector_evidence)

        graph_blocks = await graph_evidence(service, query)
        if "graph_rag" in ablations:
            ablations["graph_rag"]["evidence_sources"] = evidence_sources(graph_blocks)
            ablations["graph_rag"]["evidence_contexts"] = evidence_contexts(graph_blocks)

        if not test_mode and ((i + 1) % 5 == 0 or (i + 1) == len(data)):
            save_outputs(data)

    if not test_mode:
        save_outputs(data)
    logger.info("Context-only enrichment done")


async def process_dataset(test_mode=False):
    service = InferenceService()
    service.initialize()
    
    bm25_index, chunk_refs = build_bm25_index(service.stores)
    existing_sota = load_existing_sota_base()
    
    with open(ANSWERABILITY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if test_mode:
        data = data[:2]
        
    results = []
    
    for i, item in enumerate(data):
        logger.info(f"Processing {i+1}/{len(data)}: {item.get('question_title', 'Unknown')}")
        query = build_query(item)
        
        # 1. Vanilla Base
        logger.info("  -> Running vanilla_base")
        vanilla_ans = await run_vanilla_base(query)
        
        # 2. Parametric
        logger.info("  -> Running parametric")
        with patch("knowledge_inference.service.retrieve_all", new=retrieve_parametric_mock):
            param_res = await service._answer_async(query)
            
        # 3. BM25
        logger.info("  -> Running bm25")
        async def bm25_wrapper(query=None, intent=None, stores=None, global_graph=None, **kwargs):
            return await retrieve_bm25_mock(query=query, intent=intent, stores=stores, global_graph=global_graph, bm25_index=bm25_index, chunk_refs=chunk_refs)
        with patch("knowledge_inference.service.retrieve_all", new=bm25_wrapper):
            bm25_res = await service._answer_async(query)
            
        # 4. Vector-Only
        logger.info("  -> Running vector_only")
        with patch("knowledge_inference.service.retrieve_all", new=retrieve_vector_mock):
            vector_res = await service._answer_async(query)
            
        # 5. Graph-RAG (Full)
        logger.info("  -> Running graph_rag")
        graph_res = await service._answer_async(query)
        
        ablations_output = {
            "vanilla_base": {
                "answer": vanilla_ans,
                "evidence_sources": [],
                "evidence_contexts": []
            },
            "parametric": {
                "answer": param_res.answer,
                "evidence_sources": evidence_sources(param_res.evidence),
                "evidence_contexts": evidence_contexts(param_res.evidence)
            },
            "bm25": {
                "answer": bm25_res.answer,
                "evidence_sources": evidence_sources(bm25_res.evidence),
                "evidence_contexts": evidence_contexts(bm25_res.evidence)
            },
            "vector_only": {
                "answer": vector_res.answer,
                "evidence_sources": evidence_sources(vector_res.evidence),
                "evidence_contexts": evidence_contexts(vector_res.evidence)
            },
            "graph_rag": {
                "answer": graph_res.answer,
                "evidence_sources": evidence_sources(graph_res.evidence),
                "evidence_contexts": evidence_contexts(graph_res.evidence)
            }
        }
        attach_sota_base(ablations_output, item, existing_sota)
        
        item["ablations"] = ablations_output
        results.append(item)
        
        # Save checkpoints
        if not test_mode and ((i + 1) % 5 == 0 or (i + 1) == len(data)):
            save_outputs(results + data[i + 1:])
                
    if not test_mode:
        save_outputs(results + data[len(results):])
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference ablations or enrich saved evidence contexts.")
    parser.add_argument("--test", action="store_true", help="Run on the first 2 questions without saving.")
    parser.add_argument("--contexts-only", action="store_true", help="Only refresh evidence_sources/evidence_contexts; keep existing answers.")
    args = parser.parse_args()

    if args.contexts_only:
        asyncio.run(populate_contexts_only(test_mode=args.test))
    else:
        asyncio.run(process_dataset(test_mode=args.test))
