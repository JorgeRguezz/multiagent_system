import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Retaining original hardcoded key as found in the script, though env var is preferred
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

MODEL = "gpt-5.4"
INPUT_PATH = "knowledge_system_evaluation_v2/community_qa_dataset_ablations.json"
OUTPUT_PATH = "knowledge_system_evaluation_v2/community_qa_dataset_final.json"

SYSTEM_PROMPT = (
    "Answer the user's question using only your internal parametric knowledge. "
    "Do not use the internet or external search tools. "
    "Provide the answer directly without conversational filler."
)

def complete_llm_answers(is_test=False):
    # 1. Determine which file to load (resume if output exists)
    load_path = OUTPUT_PATH if os.path.exists(OUTPUT_PATH) else INPUT_PATH
    
    if not os.path.exists(load_path):
        print(f"Error: Could not find input file {load_path}")
        return

    with open(load_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"--- Starting evaluation process with {MODEL} (Test Mode: {is_test}) ---")

    # 2. Limit data if testing
    if is_test:
        data = data[:3]
        
    valid_count = 0

    # 3. Process questions
    for i, entry in enumerate(tqdm(data, desc="Processing questions")):
        # Ensure ablations dict exists
        if "ablations" not in entry:
            entry["ablations"] = {}
            
        # Check if already processed
        if "sota_base" in entry["ablations"] and not is_test:
            continue

        # Extract question
        question_title = entry.get("question_title", "")
        question_body = entry.get("question_body", "")
        question = f"{question_title}\n{question_body}".strip()
        
        if not question:
            continue
            
        valid_count += 1

        try:
            # 4. Call OpenAI API
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )

            answer = response.choices[0].message.content

            if is_test:
                print(f"\n[TEST - Question {i+1}]")
                print(f"Prompt:\n{question}\n")
                print(f"Response:\n{answer}\n")
                print("-" * 50)
            
            # 5. Store answer in the ablations object
            entry["ablations"]["sota_base"] = {
                "answer": answer,
                "evidence_sources": []
            }

            # 6. Incremental save (skip if testing)
            if not is_test:
                with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\n[!] Error on question {i}: {e}")
            if not is_test:
                print("Progress saved incrementally. Exiting...")
            break

    if not is_test:
        print(f"\n--- Process complete. Updated file saved to: {OUTPUT_PATH} ---")
    else:
        print(f"\n--- Test complete. No files were modified. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA model ablation inference.")
    parser.add_argument("--test", action="store_true", help="Run in test mode (processes first 3 items and doesn't save).")
    args = parser.parse_args()
    
    complete_llm_answers(is_test=args.test)