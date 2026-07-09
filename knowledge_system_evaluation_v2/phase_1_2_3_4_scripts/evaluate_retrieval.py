import asyncio
import json
import logging
import math

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from knowledge_inference.service import InferenceService
from knowledge_inference.query_analyzer import analyze_query, _extract_entity_terms
from knowledge_inference.retrievers import retrieve_chunks_dense, retrieve_all
from knowledge_inference.reranker import rerank_hits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_bm25_index(stores):
    logger.info("Building BM25 index...")
    corpus = []
    chunk_refs = []
    for store_name, store in stores.items():
        for chunk_id, chunk in store.chunks_kv.items():
            text = str(chunk.get("content", ""))
            corpus.append(text.lower().split())
            chunk_refs.append((store, chunk_id, text))
    bm25 = BM25Okapi(corpus)
    return bm25, chunk_refs

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

def get_hit_rank(hits_texts, targets):
    for rank, text in enumerate(hits_texts, start=1):
        text_lower = text.lower()
        for target in targets:
            if target.lower() in text_lower:
                return rank
    return 0

async def main():
    service = InferenceService()
    service.initialize()
    
    bm25_index, bm25_refs = build_bm25_index(service.stores)
    
    logger.info("Extracting modalities...")
    asr_texts = []
    vlm_texts = []
    for store_name, store in service.stores.items():
        for chunk_id, chunk in store.chunks_kv.items():
            text = str(chunk.get("content", ""))
            asr, vlm = split_chunk_text(text)
            asr_texts.append(asr)
            vlm_texts.append(vlm)
            
    logger.info("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    asr_embeddings = model.encode(asr_texts, normalize_embeddings=True)
    vlm_embeddings = model.encode(vlm_texts, normalize_embeddings=True)
    
    def retrieve_modality(query_emb, doc_embeddings, texts, k=15):
        scores = np.dot(doc_embeddings, query_emb)
        top_n = np.argsort(scores)[::-1][:k]
        return [texts[i] for i in top_n if texts[i].strip()]

    def get_entity_recall(hits_texts, gold_entities):
        if not gold_entities:
            return 0.0
        
        found_entities = set()
        for text in hits_texts:
            text_lower = text.lower()
            for entity in gold_entities:
                if entity in text_lower:
                    found_entities.add(entity)
                    
        return len(found_entities) / len(gold_entities)

    with open("knowledge_system_evaluation_v2/community_qa_dataset_answerability.json", "r") as f:
        data = json.load(f)
        
    metrics = {
        "bm25": {"entity_recall": 0.0},
        "vector_only": {"entity_recall": 0.0},
        "graph_rag": {"entity_recall": 0.0},
        "asr_only": {"entity_recall": 0.0},
        "vision_only": {"entity_recall": 0.0},
    }
    
    valid_count = 0
    
    for i, item in enumerate(data):
        gold_answer = item.get("answer_gold", "")
        if not gold_answer:
            continue
            
        # Extract gold entities using the built-in query analyzer function
        gold_entities = _extract_entity_terms(gold_answer)
        if not gold_entities:
            continue
            
        valid_count += 1
        query = item.get("question_title", "") + " " + item.get("question_body", "")
        intent = analyze_query(query)
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        
        # 1. BM25
        tokenized_query = query.lower().split()
        scores = bm25_index.get_scores(tokenized_query)
        top_n = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:15]
        bm25_texts = [bm25_refs[idx][2] for idx in top_n]
        
        # 2. Vector Only
        vec_hits = await retrieve_chunks_dense(query, service.stores, k=30)
        vec_hits.sort(key=lambda h: h.score_semantic, reverse=True)
        vec_texts = [hit.chunk_text for hit in vec_hits][:15]
        
        # 3. Graph RAG
        all_hits = await retrieve_all(query, intent, service.stores, service.global_graph)
        graph_hits = rerank_hits(all_hits, query, intent, list(service.stores.keys()))
        graph_texts = [hit.chunk_text for hit in graph_hits][:15]
        
        # 4. ASR Only
        asr_texts_res = retrieve_modality(query_emb, asr_embeddings, asr_texts, k=15)
        
        # 5. Vision Only
        vlm_texts_res = retrieve_modality(query_emb, vlm_embeddings, vlm_texts, k=15)
        
        configs = [
            ("bm25", bm25_texts),
            ("vector_only", vec_texts),
            ("graph_rag", graph_texts),
            ("asr_only", asr_texts_res),
            ("vision_only", vlm_texts_res)
        ]
        
        for name, texts in configs:
            recall = get_entity_recall(texts, gold_entities)
            metrics[name]["entity_recall"] += recall
                
        if (i+1) % 50 == 0:
            logger.info(f"Evaluated {i+1} questions...")

    logger.info("Evaluation complete. Compiling results...")
    
    results = {}
    for name, m in metrics.items():
        results[name] = {
            "mean_entity_recall_15": round(m["entity_recall"] / valid_count, 4),
        }
        
    results["_meta"] = {"valid_questions": valid_count}
        
    print("\n=== RETRIEVAL RESULTS ===")
    print(json.dumps(results, indent=2))
    
    with open("knowledge_system_evaluation_v2/retrieval_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    asyncio.run(main())
