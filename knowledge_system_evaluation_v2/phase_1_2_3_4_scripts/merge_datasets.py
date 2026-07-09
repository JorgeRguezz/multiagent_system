import json
import os
from pathlib import Path

def normalize_entry(entry, source_name):
    # Ensure standard keys exist and types match
    entry["source"] = entry.get("source", source_name)
    entry["question_id"] = str(entry.get("question_id", ""))
    
    # Ensure champion_matches is always a list
    if "champion_matches" not in entry:
        entry["champion_matches"] = []
        
    return entry

def main():
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"
    
    arqade_path = scripts_dir / "arcade_forum_qa_test.json"
    mobafire_path = scripts_dir / "mobafire_forum_qa_test.json"
    output_path = base_dir / "community_qa_dataset.json"
    
    combined_data = []
    
    # Load Arqade
    print(f"Loading Arqade dataset from {arqade_path}...")
    if arqade_path.exists():
        with open(arqade_path, 'r', encoding='utf-8') as f:
            arqade_data = json.load(f)
            for entry in arqade_data:
                combined_data.append(normalize_entry(entry, "arqade"))
        print(f"Loaded {len(arqade_data)} Arqade questions.")
    else:
        print(f"Warning: {arqade_path} not found.")

    # Load MobaFire
    print(f"Loading MobaFire dataset from {mobafire_path}...")
    if mobafire_path.exists():
        with open(mobafire_path, 'r', encoding='utf-8') as f:
            mobafire_data = json.load(f)
            for entry in mobafire_data:
                combined_data.append(normalize_entry(entry, "mobafire"))
        print(f"Loaded {len(mobafire_data)} MobaFire questions.")
    else:
        print(f"Warning: {mobafire_path} not found.")
        
    # Save combined
    print(f"Saving combined dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
        
    print(f"Success! {len(combined_data)} total questions saved to {output_path.name}")

if __name__ == "__main__":
    main()
