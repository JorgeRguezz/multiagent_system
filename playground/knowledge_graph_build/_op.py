import re
import json
import openai
import asyncio
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from ._videoutil import (
    retrieved_segment_caption,
)

# TODO: The following two chunking functions are not used in the default pipeline
# and have been commented out as they relied on the tiktoken library.
# They need to be refactored to work with a supplied tokenizer or string-based approximations
# if they are to be used in the future.
# def chunking_by_token_size(...):
# def chunking_by_seperators(...):

def chunking_by_video_segments(
    docs_list: list[str],
    doc_keys,
    max_token_size=1024,
):
    # This function now works with strings and character counts instead of tokens.
    # The `max_token_size` is approximated as 4 characters per token.
    max_char_size = max_token_size * 4

    # make sure each segment is not larger than max_char_size
    processed_docs = []
    for doc in docs_list:
        if len(doc) > max_char_size:
            processed_docs.append(doc[:max_char_size])
        else:
            processed_docs.append(doc)

    results = []
    chunk_content = ""
    chunk_segment_ids = []
    chunk_order_index = 0
    for index, doc in enumerate(processed_docs):
        
        if len(chunk_content) + len(doc) <= max_char_size:
            # add new segment
            chunk_content += " " + doc
            chunk_segment_ids.append(doc_keys[index])
        else:
            # save the current chunk
            results.append(
                {
                    "tokens": len(chunk_content) // 4, # Approximation
                    "content": chunk_content.strip(),
                    "chunk_order_index": chunk_order_index,
                    "video_segment_id": chunk_segment_ids,
                }
            )
            # new chunk with current segment as begin
            chunk_content = doc
            chunk_segment_ids = [doc_keys[index]]
            chunk_order_index += 1
    
    # save the last chunk
    if len(chunk_content) > 0:
        results.append(
            {
                "tokens": len(chunk_content) // 4,
                "content": chunk_content.strip(),
                "chunk_order_index": chunk_order_index,
                "video_segment_id": chunk_segment_ids,
            }
        )
    
    return results


def get_chunks(new_videos, chunk_func=chunking_by_video_segments, **chunk_func_params):
    inserting_chunks = {}

    new_videos_list = list(new_videos.keys())
    for video_name in new_videos_list:
        segment_id_list = list(new_videos[video_name].keys())
        docs = [new_videos[video_name][index]["content"] for index in segment_id_list]
        doc_keys = [f'{video_name}_{index}' for index in segment_id_list]

        # REPLACED TIKTOKEN with direct string processing.
        # The chunking function now receives raw documents instead of tokens.
        chunks = chunk_func(
            docs, doc_keys=doc_keys, **chunk_func_params
        )

        for chunk in chunks:
            inserting_chunks.update(
                {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
            )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    llm_max_tokens = global_config["llm"]["cheap_model_max_token_size"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    llm_response_cache = global_config.get("llm_response_cache")

    # Use character count as an approximation for token count (1 token ~ 4 chars)
    # This removes the dependency on tiktoken for this function.
    if len(description) < summary_max_tokens * 4:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    
    # Truncate based on character count approximation
    use_description = description[:llm_max_tokens * 4]
    
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens, hashing_kv=llm_response_cache)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    # Ensure it's a relationship record and has at least the first 4 essential fields
    if len(record_attributes) < 4 or record_attributes[0] != '"relationship"':
        return None

    # Extract required fields
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    
    # relationship_strength is optional, default to 1.0 if not provided or not a float
    weight = 1.0
    if len(record_attributes) >= 5 and is_float_regex(record_attributes[4]):
        weight = float(record_attributes[4])
        
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )
    return_edge_data = dict(
        src_tgt=(src_id, tgt_id),
        description=description,
        weight=weight
    )
    return return_edge_data


from ._llm import local_llm_batch_generate


async def extract_entities(chunks: dict[str, TextChunkSchema], knowledge_graph_inst: BaseGraphStorage, entity_vdb: BaseVectorStorage, global_config: dict, video_game_name: str= "League of Legends") -> Union[BaseGraphStorage, None]:
    """
    Extracts entities and relationships using a 3-stage Context-Aware Pipeline:
    1. Global Game Context Analysis (Summary of the game).
    2. Entity Extraction (conditioned on Game Context).
    3. Relationship Extraction (conditioned on Game Context + Found Entities).
    """
    model_name = global_config["llm"]["best_model_name"]
    ordered_chunks = list(chunks.items())
    
    # --- Step 1: Generate Global Game Context ---
    logger.info("--- Step 1: Analyzing Game Context ---")
    
    
    game_context_prompt = f"""### INSTRUCTION
    You are a Context Generator for an Entity Extraction AI.
    Your goal is to provide a **concise, keyword-focused summary** of the video game "{video_game_name}".

    The output will be used to help an AI understand proper nouns and specific terminology in text.
    You must IGNORE release dates, developers, sales history, reviews, or graphical fidelity.

    ### CONTENT REQUIREMENTS
    Focus ONLY on these three categories:
    1.  **Setting & Premise:** The fictional world, time period, and main conflict.
    2.  **Key Factions & Figures:** Major protagonists, antagonists, and groups (e.g., "The Empire", "Mario", "Ganon").
    3.  **Unique Vocabulary:** Specific names for currency, magic, items, or mechanics (e.g., "Rupees", "Mana", "Drifting", "Fatality").

    ### FORMATTING RULES
    - Keep the total output under 150 words.
    - Use dense, informative sentences.
    - Do not use bullet points; write as a cohesive summary paragraph.

    ### EXAMPLE (Pac-Man)
    Pac-Man is an arcade maze game set in a neon labyrinth. The protagonist, Pac-Man, must navigate the maze to eat dots and "Power Pellets" while avoiding four ghosts: Blinky, Pinky, Inky, and Clyde. Consuming a Power Pellet turns the ghosts blue, allowing Pac-Man to eat them for points. Fruit items occasionally appear as bonuses.

    ### INPUT
    Target Game: {video_game_name}

    ### OUTPUT
    """
    
    # Single call for context
    game_context = await local_llm_batch_generate(model_name, [game_context_prompt])
    game_context = game_context[0]
    logger.info(f"Game Context Generated:\n{game_context[:200]}...")

    # --- DEBUG PRINT 1: Game Context ---
    print("\n" + "="*40)
    print("DEBUG: LLM Response - Game Context")
    print("="*40)
    print(game_context)
    print("="*40 + "\n")

    # --- Step 2: Context-Aware Entity Extraction ---
    logger.info("--- Step 2: Extracting Entities with Context ---")
    
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )

    entity_prompt_template = """
    ### INSTRUCTION
    You are an expert entity extraction system for video games.
    Your task is to identify entities in the input text that match specific types and the provided game context.

    ### GAME CONTEXT
    {game_context}

    ### ENTITY TYPES
    Only extract entities that fit into these categories:
    [{entity_types}]

    ### FORMATTING RULES
    1. **Extraction:** For each entity, determine the Name, Type, and a brief Description based on the text and Game Context.
    2. **Capitalization:** Always capitalize the Entity Name.
    3. **Strict Constraints:**
    - Do NOT output lists, bullet points, or numbering.
    - Do NOT output the same entity twice.
    - Output ONLY the formatted string.
    4. **Structure:** Format every entity exactly like this:
    ("entity"{tuple_delimiter}<Name>{tuple_delimiter}<Type>{tuple_delimiter}<Description>)
    5. **Separation:** Separate every entity with: {record_delimiter}
    6. **Completion:** When finished, output: {completion_delimiter}

    ### EXAMPLES

    --- Example 1: Mario Kart 8 Deluxe ---

    **Game Context:**
    Mario Kart 8 Deluxe is a kart racing game set in the whimsical Mushroom Kingdom. Key characters include Mario (balanced), Bowser (heavyweight), and Toad (lightweight). The gameplay focuses on chaotic racing on gravity-defying tracks. Mechanics include "Drifting" to gain speed boosts, using "Items" (like Shells and Bananas) found in Item Boxes to hinder opponents, and "Gliding" sections.

    **Input Text:**
    Yoshi was drifting tight around the corner of Toad Harbor, sparks flying from his tires. He was in second place, just behind Donkey Kong. Suddenly, Yoshi picked up a Red Shell from an item box. He threw it forward, hitting Donkey Kong and spinning him out, allowing Yoshi to take the lead just before the glide ramp.

    **Output:**
    ("entity"{tuple_delimiter}"Yoshi"{tuple_delimiter}"character"{tuple_delimiter}"Yoshi is a playable racer who utilizes drifting mechanics to maintain speed."){record_delimiter}
    ("entity"{tuple_delimiter}"Toad Harbor"{tuple_delimiter}"location"{tuple_delimiter}"Toad Harbor is a race track setting within the Mushroom Kingdom."){record_delimiter}
    ("entity"{tuple_delimiter}"Donkey Kong"{tuple_delimiter}"character"{tuple_delimiter}"Donkey Kong is a rival racer who is hit by an item and loses first place."){record_delimiter}
    ("entity"{tuple_delimiter}"Red Shell"{tuple_delimiter}"item"{tuple_delimiter}"A Red Shell is an offensive item used to target and hit the racer in front."){record_delimiter}
    ("entity"{tuple_delimiter}"Drifting"{tuple_delimiter}"mechanic"{tuple_delimiter}"Drifting is a driving technique used to navigate corners and generate sparks for speed."){record_delimiter}
    ("entity"{tuple_delimiter}"Glide Ramp"{tuple_delimiter}"mechanic"{tuple_delimiter}"A Glide Ramp is a track feature that allows racers to fly through the air."){completion_delimiter}

    --- Example 2: The Legend of Zelda: Breath of the Wild ---

    **Game Context:**
    An open-world action-adventure game set in a post-apocalyptic Hyrule. The protagonist, Link, wakes after 100 years to defeat Calamity Ganon. The world is vast, containing "Shrines" (puzzle rooms), "Towers," and dangerous wilderness. Key mechanics include climbing surfaces, cooking food for buffs, and using the "Sheikah Slate," a tablet that provides magical runes like Magnesis and Stasis.

    **Input Text:**
    Link crouched in the tall grass outside the Shrine of Resurrection, watching the Bokoblin camp below. He checked his Sheikah Slate and selected the Magnesis rune. Aiming carefully, he lifted a large metal crate and dropped it on the unsuspecting enemy. Afterwards, he gathered some Spicy Peppers to cook a meal for the cold journey ahead.

    **Output:**
    ("entity"{tuple_delimiter}"Link"{tuple_delimiter}"character"{tuple_delimiter}"Link is the main protagonist who uses stealth and technology to defeat enemies."){record_delimiter}
    ("entity"{tuple_delimiter}"Shrine Of Resurrection"{tuple_delimiter}"location"{tuple_delimiter}"The Shrine of Resurrection is a specific location where Link begins his journey."){record_delimiter}
    ("entity"{tuple_delimiter}"Bokoblin"{tuple_delimiter}"enemy"{tuple_delimiter}"A Bokoblin is a common enemy type found in camps across Hyrule."){record_delimiter}
    ("entity"{tuple_delimiter}"Sheikah Slate"{tuple_delimiter}"technology"{tuple_delimiter}"The Sheikah Slate is a multi-purpose tablet device used to access runes."){record_delimiter}
    ("entity"{tuple_delimiter}"Magnesis"{tuple_delimiter}"mechanic"{tuple_delimiter}"Magnesis is a rune ability that allows the manipulation of metal objects."){record_delimiter}
    ("entity"{tuple_delimiter}"Spicy Peppers"{tuple_delimiter}"item"{tuple_delimiter}"Spicy Peppers are a resource ingredient used in cooking to provide cold resistance."){completion_delimiter}

    ### TASK DATA

    **Game Context:**
    {game_context}

    **Input Text:**
    {input_text}

    **Output:**
    """
    
    entity_prompts = [
        entity_prompt_template.format(
            game_context=game_context,
            input_text=chunk_dp["content"],
            **context_base
        )
        for _, chunk_dp in ordered_chunks
    ]
    
    # Batch Call 1: Entities
    raw_entity_results = await local_llm_batch_generate(model_name, entity_prompts)

    # --- DEBUG PRINT 2: Entities ---
    print("\n" + "="*40)
    print("DEBUG: LLM Response - Entity Extraction (First 3)")
    print("="*40)
    for i, res in enumerate(raw_entity_results[:3]):
        print(f"--- Entity Response {i+1} ---")
        print(res)
        print("---------------------------")
    print("="*40 + "\n")
    
    # Parse Entities per chunk
    chunk_entities_map = {} # chunk_key -> list of dict
    maybe_nodes = defaultdict(list)
    
    for (chunk_key, _), result in zip(ordered_chunks, raw_entity_results):
        chunk_entities = []
        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            record_match = re.search(r"\((.*)\)", record)
            if record_match is None: continue
            
            attributes = split_string_by_multi_markers(
                record_match.group(1), [context_base["tuple_delimiter"]]
            )
            
            entity_data = await _handle_single_entity_extraction(attributes, chunk_key)
            if entity_data:
                chunk_entities.append(entity_data)
                maybe_nodes[entity_data["entity_name"]].append(entity_data)
        
        chunk_entities_map[chunk_key] = chunk_entities

    # --- Step 3: Context-Aware Relationship Extraction ---
    logger.info("--- Step 3: Extracting Relationships with Context & Entities ---")
    
    relation_prompt_template = """
    ### INSTRUCTION
    You are an expert relationship extraction system for video games.
    Your task is to identify relationships between valid entities within the input text based on the game context.

    ### GAME CONTEXT
    {game_context}

    ### VALID ENTITIES
    You must ONLY use entities from this list as Source or Target:
    [{entity_list_str}]

    ### FORMATTING RULES
    1. **Identification:** Find pairs of entities from the list that interact or are logically connected in the text.
    2. **Direction:** Determine the Source (actor/owner) and Target (receiver/object).
    3. **Scoring:** Assign a Relationship Strength (1-10):
    - 1-4: Indirect or weak connection (e.g., in the same room).
    - 5-7: Direct interaction (e.g., talking, observing).
    - 8-10: Critical interaction (e.g., attacking, using item, distinct plot point).
    4. **Constraints:**
    - Do NOT create new entities. Use the exact spelling from the "Valid Entities" list.
    - Do NOT output relationships if there is no evidence in the text.
    - Output ONLY the formatted string.
    5. **Structure:** Format every relationship exactly like this:
    ("relationship"{tuple_delimiter}<Source_Entity>{tuple_delimiter}<Target_Entity>{tuple_delimiter}<Description>{tuple_delimiter}<Strength>)
    6. **Separation:** Separate every relationship with: {record_delimiter}
    7. **Completion:** When finished, output: {completion_delimiter}

    ### EXAMPLES

    --- Example 1: Mario Kart 8 Deluxe ---

    **Valid Entities:**
    ['Yoshi', 'Donkey Kong', 'Red Shell', 'Toad Harbor']

    **Input Text:**
    Yoshi was drifting tight around the corner of Toad Harbor. He picked up a Red Shell and threw it forward, hitting Donkey Kong and spinning him out.

    **Output:**
    ("relationship"{tuple_delimiter}"Yoshi"{tuple_delimiter}"Toad Harbor"{tuple_delimiter}"Yoshi is racing on the track Toad Harbor."{tuple_delimiter}5){record_delimiter}
    ("relationship"{tuple_delimiter}"Yoshi"{tuple_delimiter}"Red Shell"{tuple_delimiter}"Yoshi picks up and uses the Red Shell as a weapon."{tuple_delimiter}9){record_delimiter}
    ("relationship"{tuple_delimiter}"Red Shell"{tuple_delimiter}"Donkey Kong"{tuple_delimiter}"The Red Shell physically hits Donkey Kong, causing him to spin out."{tuple_delimiter}10){record_delimiter}
    ("relationship"{tuple_delimiter}"Yoshi"{tuple_delimiter}"Donkey Kong"{tuple_delimiter}"Yoshi attacks Donkey Kong to overtake him in the race."{tuple_delimiter}8){completion_delimiter}

    --- Example 2: The Legend of Zelda: Breath of the Wild ---

    **Valid Entities:**
    ['Link', 'Magnesis', 'Metal Crate', 'Bokoblin', 'Sheikah Slate']

    **Input Text:**
    Link checked his Sheikah Slate and selected the Magnesis rune. Aiming carefully, he lifted a large Metal Crate and dropped it on the unsuspecting Bokoblin.

    **Output:**
    ("relationship"{tuple_delimiter}"Link"{tuple_delimiter}"Sheikah Slate"{tuple_delimiter}"Link uses the Sheikah Slate to access his abilities."{tuple_delimiter}7){record_delimiter}
    ("relationship"{tuple_delimiter}"Sheikah Slate"{tuple_delimiter}"Magnesis"{tuple_delimiter}"The Sheikah Slate provides the Magnesis rune ability."{tuple_delimiter}9){record_delimiter}
    ("relationship"{tuple_delimiter}"Link"{tuple_delimiter}"Magnesis"{tuple_delimiter}"Link activates Magnesis to manipulate the environment."{tuple_delimiter}8){record_delimiter}
    ("relationship"{tuple_delimiter}"Magnesis"{tuple_delimiter}"Metal Crate"{tuple_delimiter}"Magnesis is used to lift and move the Metal Crate."{tuple_delimiter}9){record_delimiter}
    ("relationship"{tuple_delimiter}"Metal Crate"{tuple_delimiter}"Bokoblin"{tuple_delimiter}"The Metal Crate is used as a weapon to crush the Bokoblin."{tuple_delimiter}10){record_delimiter}
    ("relationship"{tuple_delimiter}"Link"{tuple_delimiter}"Bokoblin"{tuple_delimiter}"Link attacks the Bokoblin using the environment."{tuple_delimiter}8){completion_delimiter}

    ### TASK DATA

    **Valid Entities:**
    [{entity_list_str}]

    **Input Text:**
    {input_text}

    **Output:**
    """
    
    relation_prompts = []
    for chunk_key, chunk_dp in ordered_chunks:
        found_entities = chunk_entities_map.get(chunk_key, [])
        if not found_entities:
            # If no entities, skip or send empty list prompt (LLM might find new ones or fail, better to skip relation extraction if no nodes)
            # But to keep batch alignment, we send a dummy prompt or a valid one with empty list.
            entity_list_str = "None identified."
        else:
            entity_list_str = ", ".join([e['entity_name'] for e in found_entities])
            
        relation_prompts.append(
            relation_prompt_template.format(
                game_context=game_context,
                entity_list_str=entity_list_str,
                input_text=chunk_dp["content"],
                **context_base
            )
        )

    # Batch Call 2: Relationships
    raw_relation_results = await local_llm_batch_generate(model_name, relation_prompts)

    # --- DEBUG PRINT 3: Relationships ---
    print("\n" + "="*40)
    print("DEBUG: LLM Response - Relationship Extraction (First 3)")
    print("="*40)
    for i, res in enumerate(raw_relation_results[:3]):
        print(f"--- Relation Response {i+1} ---")
        print(res)
        print("---------------------------")
    print("="*40 + "\n")
    
    maybe_edges = defaultdict(list)
    
    for (chunk_key, _), result in zip(ordered_chunks, raw_relation_results):
        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            record_match = re.search(r"\((.*)\)", record)
            if record_match is None: continue
            
            attributes = split_string_by_multi_markers(
                record_match.group(1), [context_base["tuple_delimiter"]]
            )
            
            relation_data = await _handle_single_relationship_extraction(attributes, chunk_key)
            if relation_data:
                 maybe_edges[tuple(sorted((relation_data["src_id"], relation_data["tgt_id"])))].append(
                    relation_data
                )

    # --- Step 4: Merge & Upsert (Existing Logic) ---
    logger.info(f"Extracted {len(maybe_nodes)} unique entities and {len(maybe_edges)} unique relationships.")
    
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_edges_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working as expected.")
        return None, None, None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
        
    return knowledge_graph_inst, all_entities_data, all_edges_data


async def _find_most_related_segments_from_entities(
    topk_chunks: int,
    node_datas: list[dict],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    sorted_text_units = sorted(
        all_text_units, key=lambda x: -x["relation_counts"]
    )[:topk_chunks]
    
    chunk_related_segments = set()
    for _chunk_data in sorted_text_units:
        for s_id in _chunk_data['data']['video_segment_id']:
            chunk_related_segments.add(s_id)
    
    return chunk_related_segments

async def _refine_entity_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    llm_response_cache = global_config.get("llm_response_cache")
    query_rewrite_prompt = PROMPTS["query_rewrite_for_entity_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt, hashing_kv=llm_response_cache)
    return final_result

async def _refine_visual_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    llm_response_cache = global_config.get("llm_response_cache")
    query_rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt, hashing_kv=llm_response_cache)
    return final_result

async def _extract_keywords_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    llm_response_cache = global_config.get("llm_response_cache")
    keywords_prompt = PROMPTS["keywords_extraction"]
    keywords_prompt = keywords_prompt.format(input_text=query)
    final_result = await use_llm_func(keywords_prompt, hashing_kv=llm_response_cache)
    return final_result


async def videorag_query(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    caption_model,
    caption_tokenizer,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["llm"]["best_model_func"]
    query = query
    
    # naive chunks
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
    retreived_chunk_context = section
    
    # visual retrieval
    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query,
        query_param,
        global_config,
    )
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)
    entity_retrieved_segments = set()
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(entity_results, node_datas, node_degrees)
            if n is not None
        ]
        entity_retrieved_segments = entity_retrieved_segments.union(await _find_most_related_segments_from_entities(
            global_config["retrieval_topk_chunks"], node_datas, text_chunks_db, knowledge_graph_inst
        ))
    
    # visual retrieval
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query,
        query_param,
        global_config,
    )
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])
    
    # caption
    retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    )
    print(query_for_entity_retrieval)
    print(f"Retrieved Text Segments {entity_retrieved_segments}")
    print(query_for_visual_retrieval)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    
    already_processed = 0
    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        nonlocal use_model_func, already_processed
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = PROMPTS["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Checked {already_processed} segments\r",
            end="",
            flush=True,
        )
        return (segment_key, result)
    
    rough_captions = {}
    for s_id in retrieved_segments:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        rough_captions[s_id] = video_segments._data[video_name][index]["content"]
    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    )
    remain_segments = [x[0] for x in results if 'yes' in x[1].lower()]
    print(f"{len(remain_segments)} Video Segments remain after filtering")
    if len(remain_segments) == 0:
        print("Since no segments remain after filtering, we utilized all the retrieved segments.")
        remain_segments = retrieved_segments
    print(f"Remain segments {remain_segments}")
    
    # visual retrieval
    keywords_for_caption = await _extract_keywords_query(
        query,
        query_param,
        global_config,
    )
    print(f"Keywords: {keywords_for_caption}")
    caption_results = retrieved_segment_caption(
        caption_model,
        caption_tokenizer,
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment']
    )

    ## data table
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0])
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1])
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    if query_param.wo_reference:
        sys_prompt_temp = PROMPTS["videorag_response_wo_reference"]
    else:
        sys_prompt_temp = PROMPTS["videorag_response"]
        
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

async def videorag_query_multiple_choice(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    caption_model,
    caption_tokenizer,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """_summary_
    A copy of the videorag_query function with several updates for handling multiple-choice queries.
    """
    use_model_func = global_config["llm"]["best_model_func"]
    query = query
    
    # naive chunks
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    # NOTE: I update here, not len results can also process
    if len(results):
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.naive_max_token_for_text_unit,
        )
        logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
        section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
        retreived_chunk_context = section
    else:
        retreived_chunk_context = "No Content"
        
    # visual retrieval
    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query,
        query_param,
        global_config,
    )
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)
    entity_retrieved_segments = set()
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(entity_results, node_datas, node_degrees)
            if n is not None
        ]
        entity_retrieved_segments = entity_retrieved_segments.union(await _find_most_related_segments_from_entities(
            global_config["retrieval_topk_chunks"], node_datas, text_chunks_db, knowledge_graph_inst
        ))
    
    # visual retrieval
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query,
        query_param,
        global_config,
    )
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])
    
    # caption
    retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    )
    print(query_for_entity_retrieval)
    print(f"Retrieved Text Segments {entity_retrieved_segments}")
    print(query_for_visual_retrieval)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    
    already_processed = 0
    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        nonlocal use_model_func, already_processed
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = PROMPTS["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Checked {already_processed} segments\r",
            end="",
            flush=True,
        )
        return (segment_key, result)
    
    rough_captions = {}
    for s_id in retrieved_segments:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        rough_captions[s_id] = video_segments._data[video_name][index]["content"]
    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    )
    remain_segments = [x[0] for x in results if 'yes' in x[1].lower()]
    print(f"{len(remain_segments)} Video Segments remain after filtering")
    if len(remain_segments) == 0:
        print("Since no segments remain after filtering, we utilized all the retrieved segments.")
        remain_segments = retrieved_segments
    print(f"Remain segments {remain_segments}")
    
    # visual retrieval
    keywords_for_caption = await _extract_keywords_query(
        query,
        query_param,
        global_config,
    )
    print(f"Keywords: {keywords_for_caption}")
    caption_results = retrieved_segment_caption(
        caption_model,
        caption_tokenizer,
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment']
    )

    ## data table
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0])
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1])
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    # NOTE: I update here to use a different prompt
    sys_prompt_temp = PROMPTS["videorag_response_for_multiple_choice_question"]
        
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        use_cache=False,
    )
    while True:
        try:
            json_response = json.loads(response)
            assert "Answer" in json_response and "Explanation" in json_response
            return json_response
        except Exception as e:
            logger.info(f"Response is not valid JSON for query {query}. Found {e}. Retrying...")
            response = await use_model_func(
                query,
                system_prompt=sys_prompt,
                use_cache=False,
            )
    