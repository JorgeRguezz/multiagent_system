import json
import math
import torch
import asyncio
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
import os

# Add the parent directory to sys.path so we can import the project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from knowledge_inference.service import InferenceService
from knowledge_inference.query_analyzer import analyze_query
from knowledge_inference.retrievers import retrieve_all
from knowledge_inference.reranker import rerank_hits

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

async def process_dataset():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "community_qa_dataset_temporal_classified.json"
    output_path = base_dir / "community_qa_dataset_answerability.json"
    
    print(f"Loading dataset from {input_path.name}...")
    dataset = load_data(input_path)
    
    print("Initializing Qwen3-Reranker-4B...")
    model_name = "Qwen/Qwen3-Reranker-4B"
    # trust_remote_code=True is standard for newer custom architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # Fix for "Cannot handle batch sizes > 1 if no padding token is defined."
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.eval()
    
    print("Initializing InferenceService for retrieval...")
    service = InferenceService()
    service.initialize()
    stores = service.stores
    global_graph = service.global_graph
    available_videos = list(stores.keys())
    
    instruction = (
        "Determine whether the Candidate Chunk alone contains sufficient factual evidence "
        "to answer the Question with the stated Gold Answer.\n\n"
        "Return YES only when the chunk explicitly states or unambiguously entails every "
        "essential claim needed by the Gold Answer. Return NO for merely related gameplay "
        "discussion, partial support, contradictory information, outdated mechanics, "
        "generic advice, or unsupported inference."
    )
    
    print(f"Processing {len(dataset)} questions...")
    for idx, entry in enumerate(tqdm(dataset, desc="Evaluating Answerability")):
        # Skip if already processed (useful if script crashes and is restarted)
        if "answerability_status" in entry:
            continue
            
        q_title = entry.get("question_title", "")
        q_body = entry.get("question_body", "")
        gold_answer = entry.get("answer_gold", "")
        
        full_query = f"{q_title}\n{q_body}".strip()
        
        # 1. Retrieve the top candidates using the system's live retrievers
        intent = analyze_query(full_query)
        hits = await retrieve_all(
            query=full_query,
            intent=intent,
            stores=stores,
            global_graph=global_graph
        )
        
        ranked_hits = rerank_hits(
            hits=hits,
            query=full_query,
            intent=intent,
            available_videos=available_videos,
        )
        
        # Take Top 15 to ensure high recall
        top_15_hits = ranked_hits[:15]
        
        query_formatted = f"Instruction: {instruction}\n\nQuestion: {full_query}\nGold Answer: {gold_answer}"
        
        max_score = -999.0
        
        if top_15_hits:
            for hit in top_15_hits:
                pair = [[query_formatted, hit.chunk_text]]
                inputs = tokenizer(pair, padding=True, truncation=True, return_tensors='pt', max_length=2048).to(model.device)
                
                with torch.no_grad():
                    score = model(**inputs, return_dict=True).logits.view(-1,)[0].float().item()
                    
                if score > max_score:
                    max_score = score
            
        # Standard sigmoid normalization from logits
        normalized_score = 1 / (1 + math.exp(-max_score)) if max_score != -999.0 else 0.0
        
        # Thresholds (We start with 0.6 and 0.1 for BGE/Qwen Sigmoid distribution)
        if normalized_score >= 0.60:
            status = "Full"
        elif normalized_score >= 0.10:
            status = "Partial"
        else:
            status = "None"
            
        entry["answerability_status"] = status
        entry["answerability_score"] = round(normalized_score, 4)
        
        # Checkpoint every 20 queries to prevent data loss
        if (idx + 1) % 20 == 0:
            save_data(dataset, output_path)
            
    # Final save
    save_data(dataset, output_path)
    print(f"\nDone! Saved fully evaluated dataset to {output_path.name}")

if __name__ == "__main__":
    asyncio.run(process_dataset())
