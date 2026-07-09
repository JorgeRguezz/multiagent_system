import json
from pathlib import Path
from tqdm import tqdm
import evaluate as hf_evaluate

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate BERTScore")
    parser.add_argument("--test", action="store_true", help="Run a quick test on the first 3 questions")
    args = parser.parse_args()

    input_base_file = Path(__file__).resolve().parent.parent / "community_qa_dataset_final.json"
    output_file = Path(__file__).resolve().parent.parent / "community_qa_dataset_evaluated.json"

    # Smart Resume: Load evaluated metrics if present, otherwise start from the final inference dataset.
    target_load_file = output_file if output_file.exists() else input_base_file

    print(f"Loading data from {target_load_file}")
    with open(target_load_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_questions = len(data)
    print(f"Loaded {total_questions} questions.")
    
    limit = 3 if args.test else total_questions
    if args.test:
        print(f"TEST MODE ENABLED: Only evaluating the first {limit} questions.")
    
    # Load HuggingFace BERTScore
    print("Loading BERTScore module (distilbert-base-uncased)...")
    bertscore = hf_evaluate.load("bertscore")
    
    # We will gather all pairs that need evaluating to batch process them efficiently
    predictions = []
    references = []
    metadata_map = [] # To map the result back to the correct dictionary location
    
    print("Scanning dataset for missing BERTScores...")
    for q_idx, item in enumerate(data[:limit]):
        answer_gold = item.get("answer_gold", "")
        ablations = item.get("ablations", {})
        
        for ablation_name, ablation_data in ablations.items():
            # Skip if already evaluated
            if "bertscore_f1" in ablation_data:
                continue
                
            generated_answer = ablation_data.get("answer", "")
            
            # If generation failed or is missing entirely, just give it 0.0 or skip
            if not generated_answer:
                ablation_data["bertscore_f1"] = 0.0
                continue
                
            predictions.append(generated_answer)
            references.append(answer_gold)
            metadata_map.append((q_idx, ablation_name))
            
    if len(predictions) == 0:
        print("No new answers to evaluate for BERTScore. Exiting.")
        return
        
    print(f"Computing BERTScore for {len(predictions)} generation pairs...")
    
    # Run the batch evaluation
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="distilbert-base-uncased",
    )
    
    f1_scores = results["f1"]
    
    # Map the scores back to the JSON object
    for score, (q_idx, ablation_name) in zip(f1_scores, metadata_map):
        data[q_idx]["ablations"][ablation_name]["bertscore_f1"] = float(score)
        
    # Save the updated JSON
    print("Evaluation complete. Saving updated dataset...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully added {len(predictions)} BERTScores to {output_file.name}.")

if __name__ == "__main__":
    main()
