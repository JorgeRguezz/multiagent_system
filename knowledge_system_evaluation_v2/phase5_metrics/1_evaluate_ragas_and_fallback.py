import os
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
import concurrent.futures
import argparse
import pandas as pd

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, AnswerRelevancy
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["RAGAS_DO_NOT_TRACK"] = "true" # Disable RAGAS telemetry to prevent DNS crashes

llm_judge = llm_factory("gpt-5.4-nano") 
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# We replace text-embedding-3-small with BAAI/bge-small-en-v1.5 to speed up local math processing
langchain_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 128,
    },
)

embeddings_model = LangchainEmbeddingsWrapper(
    langchain_embeddings
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

NO_CONTEXT_ABLATIONS = {"vanilla_base", "parametric", "sota_base"}
RETRIEVAL_ABLATIONS = {"bm25", "vector_only", "graph_rag", "asr_only", "vision_only"}
SOURCE_LABEL_TOKENS = {"bm25", "dense_chunk", "entity_graph", "global_graph", "visual_support", "vector_modality"}
MAX_FALLBACK_CONTEXTS = 5
FALLBACK_MAX_WORKERS = 8
FALLBACK_CHECKPOINT_EVERY = 50
OPENAI_RETRY_ATTEMPTS = 5


def get_question_text(item):
    return f"{item.get('question_title', '')}\n\n{item.get('question_body', '')}".strip()


def normalize_contexts(value):
    if not isinstance(value, list):
        value = [value] if value else []
    return [context for context in value if isinstance(context, str) and context.strip()]


def contexts_look_like_source_labels(contexts):
    if not contexts:
        return False

    for context in contexts:
        parts = [part.strip() for part in context.split("|") if part.strip()]
        if not parts or any(part not in SOURCE_LABEL_TOKENS for part in parts):
            return False
    return True


def get_contexts(ab_name, ab_data):
    contexts = normalize_contexts(ab_data.get("evidence_contexts", []))
    if contexts_look_like_source_labels(contexts):
        raise ValueError(
            f"{ab_name} evidence_contexts appears to contain source labels instead of retrieved text"
        )
    return contexts


def call_openai_json_with_retries(prompt, required_keys):
    last_error = None
    for attempt in range(OPENAI_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[
                    {"role": "system", "content": "You are a strict JSON-outputting evaluator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            missing = [key for key in required_keys if key not in result]
            if missing:
                raise ValueError(f"Missing required JSON keys: {missing}; content={content}")
            return result
        except Exception as exc:
            last_error = exc
            if attempt == OPENAI_RETRY_ATTEMPTS - 1:
                break
            sleep_s = min(30.0, (2 ** attempt) + random.uniform(0.0, 0.75))
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI JSON call failed after {OPENAI_RETRY_ATTEMPTS} attempts: {last_error}")


def judge_refusal(question, generated_answer):
    prompt = f"""You are an expert AI evaluator.
Your task is to determine whether an AI system refused to answer a user's question.

[Inputs]
Question: {question}
Generated Answer: {generated_answer}

[Instructions]
1. Set "is_refusal" to true only if the Generated Answer explicitly states that it cannot answer, cannot determine, lacks enough information, lacks context/evidence, or that the provided clips/context do not contain the answer.
2. Do not mark cautious answers as refusals if they still provide a substantive answer.
3. Provide a brief 1-sentence reason.

Respond ONLY with a JSON object in this exact format:
{{
    "is_refusal": true/false,
    "reason": "..."
}}
"""
    res = call_openai_json_with_retries(prompt, required_keys=["is_refusal", "reason"])
    return bool(res.get("is_refusal", False)), res.get("reason", "")


def judge_overrefusal(question, generated_answer, contexts):
    context_text = "\n\n".join(contexts[:MAX_FALLBACK_CONTEXTS])
    prompt = f"""You are an expert AI evaluator.
The AI system's answer refused to answer the user's question. Your task is to determine whether this refusal was an over-refusal.

[Inputs]
Question: {question}
Generated Answer: {generated_answer}
Retrieved Context:
{context_text}

[Instructions]
1. Read the Retrieved Context.
2. If the context contains sufficient factual information to answer the Question, set "is_overrefusal" to true.
3. If the context does not contain sufficient factual information to answer the Question, set "is_overrefusal" to false.
4. Provide a brief 1-sentence reason.

Respond ONLY with a JSON object in this exact format:
{{
    "is_overrefusal": true/false,
    "reason": "..."
}}
"""
    res = call_openai_json_with_retries(prompt, required_keys=["is_overrefusal", "reason"])
    return bool(res.get("is_overrefusal", False)), res.get("reason", "")


def evaluate_fallback_task(task):
    q_idx, ab_name, question, generated_answer, contexts = task
    try:
        is_refusal, refusal_reason = judge_refusal(question, generated_answer)
        if not is_refusal:
            return (q_idx, ab_name, False, False, refusal_reason)
        if not contexts:
            reason = f"Refusal: {refusal_reason} Overrefusal: no retrieved context was available."
            return (q_idx, ab_name, True, False, reason)

        is_overrefusal, overrefusal_reason = judge_overrefusal(
            question,
            generated_answer,
            contexts,
        )
        reason = f"Refusal: {refusal_reason} Overrefusal: {overrefusal_reason}".strip()
        return (q_idx, ab_name, True, is_overrefusal, reason)
    except Exception as e:
        return (q_idx, ab_name, None, None, f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAGAS and Fallback")
    parser.add_argument("--test", action="store_true", help="Run a quick test on the first 3 questions")
    args = parser.parse_args()

    input_base_file = Path(__file__).resolve().parent.parent / "community_qa_dataset_final.json"
    output_file = Path(__file__).resolve().parent.parent / "community_qa_dataset_evaluated.json"
    
    # Smart Resume: Load from the output file if it exists so we pick up from checkpoints
    target_load_file = output_file if output_file.exists() else input_base_file
    
    print(f"Loading data from {target_load_file}")
    with open(target_load_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_questions = len(data)
    limit = 3 if args.test else total_questions
    if args.test:
        print(f"TEST MODE ENABLED: Only evaluating the first {limit} questions.")
    
    # 1. Gather all tasks instead of evaluating sequentially
    fallback_tasks = []
    ragas_with_context = {"question": [], "answer": [], "contexts": [], "q_idx": [], "ab_name": []}
    ragas_no_context = {"question": [], "answer": [], "contexts": [], "q_idx": [], "ab_name": []}
    
    missing_contexts = []

    print("Scanning dataset for missing evaluations...")
    for q_idx, item in enumerate(data[:limit]):
        question = get_question_text(item)
        ablations = item.get("ablations", {})
        
        for ab_name, ab_data in ablations.items():
            generated_answer = ab_data.get("answer", "")
            if not generated_answer:
                continue
                
            contexts = get_contexts(ab_name, ab_data)
            if ab_name in RETRIEVAL_ABLATIONS and not contexts:
                missing_contexts.append((q_idx, ab_name))
                continue

            # If missing fallback
            if "is_refusal" not in ab_data or ab_data.get("is_refusal") is None:
                fallback_tasks.append((q_idx, ab_name, question, generated_answer, contexts))
                
            needs_relevance = ab_data.get("relevance") is None
            needs_faithfulness = bool(contexts) and ab_data.get("faithfulness") is None

            # If missing RAGAS
            if needs_relevance or needs_faithfulness:
                if not contexts:
                    ragas_no_context["question"].append(question)
                    ragas_no_context["answer"].append(generated_answer)
                    ragas_no_context["contexts"].append([""]) # Ragas needs at least empty string
                    ragas_no_context["q_idx"].append(q_idx)
                    ragas_no_context["ab_name"].append(ab_name)
                else:
                    ragas_with_context["question"].append(question)
                    ragas_with_context["answer"].append(generated_answer)
                    ragas_with_context["contexts"].append(contexts)
                    ragas_with_context["q_idx"].append(q_idx)
                    ragas_with_context["ab_name"].append(ab_name)

    if missing_contexts:
        sample = ", ".join(f"q_idx={q_idx}/{ab_name}" for q_idx, ab_name in missing_contexts[:10])
        raise RuntimeError(
            "Missing evidence_contexts for retrieval ablations. "
            "Rerun knowledge_system_evaluation_v2/scripts/run_inference_ablations.py "
            "and knowledge_system_evaluation_v2/scripts/run_modality_ablations.py before Phase 5. "
            f"Missing count={len(missing_contexts)}; sample: {sample}"
        )
                    
    # 2. Execute Fallback Concurrently (Massive Speedup)
    print(f"Pending Custom Fallback API calls: {len(fallback_tasks)}")
    if len(fallback_tasks) > 0:
        print(f"Running Fallback Evaluation with {FALLBACK_MAX_WORKERS} parallel threads...")
        fallback_success = 0
        fallback_error = 0
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=FALLBACK_MAX_WORKERS) as executor:
            futures = [executor.submit(evaluate_fallback_task, task) for task in fallback_tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fallback API"):
                q_idx, ab_name, is_ref, is_overref, reason = future.result()
                ab_data = data[q_idx]["ablations"][ab_name]
                if is_ref is not None:
                    ab_data["is_refusal"] = is_ref
                    ab_data["is_overrefusal"] = is_overref
                    ab_data["refusal_reason"] = reason
                    ab_data.pop("fallback_error", None)
                    fallback_success += 1
                else:
                    ab_data["fallback_error"] = reason
                    fallback_error += 1

                completed += 1
                if completed % FALLBACK_CHECKPOINT_EVERY == 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)

        print(f"Fallback completed: {fallback_success} successful, {fallback_error} failed")
        print(f"Saving Fallback metrics to {output_file.name}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # 3. Execute RAGAS natively on the whole dataset
    if len(ragas_with_context['question']) > 0:
        print(f"Evaluating: {len(ragas_with_context['question'])} items with context")
        ds_ctx = Dataset.from_dict(ragas_with_context)
        res_ctx = evaluate(
            ds_ctx,
            metrics=[Faithfulness(llm=llm_judge), AnswerRelevancy(llm=llm_judge, strictness=1)],
            embeddings=embeddings_model,
            run_config=RunConfig(timeout=600, max_workers=3), # Lowered from 10 to 3 to prevent OpenAI 200k TPM rate limits
            raise_exceptions=False,
        )
        
        df_ctx = res_ctx.to_pandas()
        for idx, row in df_ctx.iterrows():
            q_idx = int(ds_ctx["q_idx"][idx])
            ab_name = ds_ctx["ab_name"][idx]
            
            # Convert NaNs to None to avoid JSON serialization issues if rate limits occur
            faith = float(row["faithfulness"]) if pd.notna(row.get("faithfulness")) else None
            rel = float(row["answer_relevancy"]) if pd.notna(row.get("answer_relevancy")) else None
            
            if faith is not None:
                data[q_idx]["ablations"][ab_name]["faithfulness"] = faith
            if rel is not None:
                data[q_idx]["ablations"][ab_name]["relevance"] = rel

        print(f"Saving With Context RAGAS metrics to {output_file.name}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    if len(ragas_no_context["question"]) > 0:
        print(f"\nPending RAGAS (No Context - Relevancy Only): {len(ragas_no_context['question'])}")
        ds_no_ctx = Dataset.from_dict(ragas_no_context)
        res_no_ctx = evaluate(
            ds_no_ctx,
            metrics=[AnswerRelevancy(llm=llm_judge, strictness=1)],
            embeddings=embeddings_model,
            run_config=RunConfig(timeout=600, max_workers=3), # Lowered to 3
            raise_exceptions=False,
        )
        
        df_no_ctx = res_no_ctx.to_pandas()
        for idx, row in df_no_ctx.iterrows():
            q_idx = int(ds_no_ctx["q_idx"][idx])
            ab_name = ds_no_ctx["ab_name"][idx]
            
            rel = float(row["answer_relevancy"]) if pd.notna(row.get("answer_relevancy")) else None
            if rel is not None:
                data[q_idx]["ablations"][ab_name]["relevance"] = rel
                data[q_idx]["ablations"][ab_name]["faithfulness"] = None

        print(f"Saving No Context RAGAS metrics to {output_file.name}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    print(f"\nEvaluation complete. Dataset fully saved to {output_file.name}")
    
    # 5. Reproducibility metadata
    metadata = {
        "answer_relevancy_embedding_model": "BAAI/bge-small-en-v1.5",
        "answer_relevancy_embedding_device": "cuda",
        "answer_relevancy_embedding_normalized": True,
        "answer_relevancy_embedding_batch_size": 128,
        "answer_relevancy_embedding_dimension": 384,
        "answer_relevancy_strictness": 1,
        "question_generation_llm": "gpt-5.4-nano",
        "ragas_context_field": "evidence_contexts"
    }
    print("\n=== REPRODUCIBILITY METADATA ===")
    print(json.dumps(metadata, indent=2))
    print("================================")

if __name__ == "__main__":
    main()
