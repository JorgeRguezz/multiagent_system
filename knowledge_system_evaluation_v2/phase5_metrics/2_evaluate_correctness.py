import os
import json
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import argparse
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

def get_question_text(item):
    return f"{item.get('question_title', '')}\n\n{item.get('question_body', '')}".strip()

def evaluate_correctness_task(task):
    q_idx, ab_name, question, answer_gold, generated_answer = task
    prompt = f"""You are an impartial expert judge evaluating an AI-generated answer for a League of Legends knowledge base.

[Inputs]
Question: {question}
Gold Answer (Community Accepted): {answer_gold}
Generated Answer (System Output): {generated_answer}

[Task]
Evaluate the Reference Answer Correctness of the Generated Answer compared to the Gold Answer. 
Do NOT penalize the Generated Answer if it includes extra *correct* context, as long as it solves the problem equally well.

[Scoring Rubric - Strict 0 to 2 Scale]
0 (Incorrect or non-answer): The response is materially wrong, contradicts the reference answer, invents important facts, or does not answer the question.
1 (Partially correct): The response captures some of the relevant answer but has a material omission, ambiguity, or factual error compared to the Gold Answer.
2 (Substantially correct): The response directly and correctly answers the question and is consistent with the Gold Answer.

Respond ONLY with a JSON object in this exact format:
{{
    "correctness_score": <int 0, 1, or 2>,
    "correctness_reason": "<1-sentence justification>"
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": "You are a strict JSON-outputting academic evaluator."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        res = json.loads(response.choices[0].message.content)
        return (q_idx, ab_name, res.get("correctness_score", 0), res.get("correctness_reason", ""))
    except Exception as e:
        # Return None on error so we don't permanently save '0' for a rate limit
        return (q_idx, ab_name, None, f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Reference Correctness")
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
    
    tasks = []
    
    print("Scanning dataset for missing Correctness evaluations...")
    for q_idx, item in enumerate(data[:limit]):
        question = get_question_text(item)
        answer_gold = item.get("answer_gold", "")
        ablations = item.get("ablations", {})
        
        for ab_name, ab_data in ablations.items():
            generated_answer = ab_data.get("answer", "")
            
            if not generated_answer:
                continue
                
            if "correctness_score" not in ab_data or ab_data["correctness_score"] is None:
                tasks.append((q_idx, ab_name, question, answer_gold, generated_answer))
                
    print(f"Pending Correctness API calls: {len(tasks)}")
    
    if len(tasks) > 0:
        print("Running Correctness Evaluation with 20 parallel threads...")
        # Save every 50 tasks incrementally so progress is never lost on Ctrl+C
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            for q_idx, ab_name, score, reason in tqdm(executor.map(evaluate_correctness_task, tasks), total=len(tasks), desc="Correctness API"):
                if score is not None:
                    data[q_idx]["ablations"][ab_name]["correctness_score"] = score
                    data[q_idx]["ablations"][ab_name]["correctness_reason"] = reason
                
                completed += 1
                if completed % 50 == 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)
                
        print(f"Saving final updated dataset to {output_file.name}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
    print("Successfully finished evaluating Reference Correctness.")

if __name__ == "__main__":
    main()
