import json
import logging
from concurrent.futures import ThreadPoolExecutor
from knowledge_build._llm import local_llm_config
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a League of Legends evaluation judge. 
Analyze the following question and classify its query type into EXACTLY ONE of the following three categories:
1. "Entity" - Questions about factual recall of specific champion stats, abilities, or items.
2. "Event" - Questions grounded in temporal or action-based visual occurrences (e.g., first dragon, early game actions).
3. "Relationship" - Strategic questions about matchups, counter-picks, item synergies, or team compositions.

Reply with ONLY the exact category name: "Entity", "Event", or "Relationship".

Question Title: {title}
Question Body: {body}
"""

async def classify_question(item):
    if "query_type" in item:
        return item

    prompt = CLASSIFICATION_PROMPT.format(
        title=item.get("question_title", ""),
        body=item.get("question_body", "")
    )
    
    try:
        # Since local_llm_config is async
        response = await local_llm_config.best_model_func(prompt, system_prompt="You are a classifier.")
        text = str(response.get("answer", response) if isinstance(response, dict) else response).strip().strip('"').strip("'")
        
        if "Relationship" in text or "relationship" in text:
            item["query_type"] = "Relationship"
        elif "Event" in text or "event" in text:
            item["query_type"] = "Event"
        else:
            item["query_type"] = "Entity"
            
    except Exception as e:
        logger.error(f"Error classifying: {e}")
        item["query_type"] = "Entity"
        
    return item

async def main():
    input_file = "knowledge_system_evaluation_v2/community_qa_dataset_final.json"
    
    with open(input_file, "r") as f:
        data = json.load(f)
        
    logger.info(f"Classifying {len(data)} questions...")
    
    # Process sequentially or in batches
    for i, item in enumerate(data):
        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(data)}")
            # checkpoint
            with open(input_file, "w") as f:
                json.dump(data, f, indent=4)
                
        await classify_question(item)
        
    with open(input_file, "w") as f:
        json.dump(data, f, indent=4)
        
    logger.info("Done classifying query types.")

if __name__ == "__main__":
    asyncio.run(main())
