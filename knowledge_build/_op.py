import re
import json
# import openai
import asyncio
from typing import Union
from collections import Counter, defaultdict
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
    TextChunkSchema,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
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
    summary = await use_llm_func(
        use_prompt,
        system_prompt=PROMPTS["system_prompt_kg_summary"],
        max_tokens=summary_max_tokens,
        hashing_kv=llm_response_cache,
    )
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


from ._llm import oss_llm_batch_generate


def _split_extraction_records(text: str, context_base: dict) -> list[list[str]]:
    records = split_string_by_multi_markers(
        text,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )
    parsed_records = []
    for record in records:
        record_match = re.search(r"\((.*)\)", record)
        if record_match is None:
            continue
        parsed_records.append(
            split_string_by_multi_markers(
                record_match.group(1),
                [context_base["tuple_delimiter"]],
            )
        )
    return parsed_records


def _normalize_entity_type(entity_type: str, allowed_types: set[str]) -> str:
    normalized = clean_str(entity_type.upper())
    return normalized if normalized in allowed_types else "UNKNOWN"


def _sanitize_relationship_weight(weight: float, minimum: float = 1.0, maximum: float = 10.0) -> float:
    if weight < minimum:
        return minimum
    if weight > maximum:
        return maximum
    return weight


async def _extract_entities_from_text(
    chunk_key: str,
    raw_text: str,
    context_base: dict,
    allowed_types: set[str],
) -> list[dict]:
    parsed = _split_extraction_records(raw_text, context_base)
    chunk_entities = []
    seen_names = set()
    for attributes in parsed:
        entity_data = await _handle_single_entity_extraction(attributes, chunk_key)
        if not entity_data:
            continue
        entity_data["entity_type"] = _normalize_entity_type(entity_data["entity_type"], allowed_types)
        if entity_data["entity_name"] in seen_names:
            continue
        if not entity_data["description"].strip():
            continue
        seen_names.add(entity_data["entity_name"])
        chunk_entities.append(entity_data)
    return chunk_entities


async def _extract_relationships_from_text(
    chunk_key: str,
    raw_text: str,
    context_base: dict,
    valid_entities: set[str],
    min_weight: float = 1.0,
    max_weight: float = 10.0,
) -> list[dict]:
    parsed = _split_extraction_records(raw_text, context_base)
    chunk_relationships = []
    seen_pairs = set()
    for attributes in parsed:
        relation_data = await _handle_single_relationship_extraction(attributes, chunk_key)
        if not relation_data:
            continue
        src = relation_data["src_id"]
        tgt = relation_data["tgt_id"]
        if src == tgt:
            continue
        if src not in valid_entities or tgt not in valid_entities:
            continue
        relation_data["weight"] = _sanitize_relationship_weight(
            relation_data["weight"], minimum=min_weight, maximum=max_weight
        )
        if not relation_data["description"].strip():
            continue
        undirected_key = tuple(sorted((src, tgt)))
        if undirected_key in seen_pairs:
            continue
        seen_pairs.add(undirected_key)
        chunk_relationships.append(relation_data)
    return chunk_relationships


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    Simple-only KG extraction:
    1) Base extraction of entities + relationships per chunk.
    2) Unified glean passes for missing tuples.
    """
    use_llm_func: callable = global_config["llm"]["best_model_func"]
    llm_response_cache = global_config.get("llm_response_cache")
    max_gleaning = int(global_config.get("entity_extract_max_gleaning", 1))
    min_weight = 1.0
    max_weight = 10.0

    ordered_chunks = list(chunks.items())
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    allowed_types = {t.upper() for t in PROMPTS["DEFAULT_ENTITY_TYPES"]}

    logger.info("--- Simple Mode: Base Graph Extraction + Unified Glean ---")
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    simple_prompt_template = PROMPTS["kg_simple_graph_extraction_template"]
    simple_prompts = [
        simple_prompt_template.format(input_text=chunk_dp["content"], **context_base)
        for _, chunk_dp in ordered_chunks
    ]
    raw_simple_results = await oss_llm_batch_generate(
        simple_prompts,
        system_prompt=PROMPTS["system_prompt_kg_glean_unified"],
        max_tokens=3000,
    )

    for (chunk_key, _), prompt, result in zip(ordered_chunks, simple_prompts, raw_simple_results):
        logger.info(f"Simple extraction output for chunk {chunk_key}:\n{result}\n---")
        chunk_entities = await _extract_entities_from_text(
            chunk_key=chunk_key,
            raw_text=result,
            context_base=context_base,
            allowed_types=allowed_types,
        )
        existing_names = {e["entity_name"] for e in chunk_entities}
        valid_entities = set(existing_names)

        chunk_relationships = await _extract_relationships_from_text(
            chunk_key=chunk_key,
            raw_text=result,
            context_base=context_base,
            valid_entities=valid_entities,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        existing_pairs = {tuple(sorted((r["src_id"], r["tgt_id"]))) for r in chunk_relationships}

        history = pack_user_ass_to_openai_messages(prompt, result)
        chunk_text = chunks[chunk_key]["content"]
        for _ in range(max_gleaning):
            entity_snapshot = "\n".join(
                [f'- "{e["entity_name"]}" ({e["entity_type"]}): {e["description"]}' for e in chunk_entities]
            )
            relation_snapshot = "\n".join(
                [f'- "{r["src_id"]}" -> "{r["tgt_id"]}" (w={r["weight"]}): {r["description"]}' for r in chunk_relationships]
            )
            unified_prompt = PROMPTS["kg_unified_glean_template"].format(
                entity_types=context_base["entity_types"],
                entity_snapshot=(entity_snapshot if entity_snapshot else "None"),
                relation_snapshot=(relation_snapshot if relation_snapshot else "None"),
                chunk_text=chunk_text,
                tuple_delimiter=context_base["tuple_delimiter"],
                record_delimiter=context_base["record_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
            )
            unified_result = await use_llm_func(
                unified_prompt,
                system_prompt=PROMPTS["system_prompt_kg_glean_unified"],
                history_messages=history,
                max_tokens=3000,
                hashing_kv=llm_response_cache,
            )
            history += pack_user_ass_to_openai_messages(unified_prompt, unified_result)

            gleaned_entities = await _extract_entities_from_text(
                chunk_key=chunk_key,
                raw_text=unified_result,
                context_base=context_base,
                allowed_types=allowed_types,
            )
            net_new_entities = []
            for ent in gleaned_entities:
                if ent["entity_name"] in existing_names:
                    continue
                existing_names.add(ent["entity_name"])
                net_new_entities.append(ent)
            if net_new_entities:
                chunk_entities.extend(net_new_entities)

            valid_entities = {e["entity_name"] for e in chunk_entities}
            gleaned_relations = await _extract_relationships_from_text(
                chunk_key=chunk_key,
                raw_text=unified_result,
                context_base=context_base,
                valid_entities=valid_entities,
                min_weight=min_weight,
                max_weight=max_weight,
            )
            net_new_relations = []
            for rel in gleaned_relations:
                pair = tuple(sorted((rel["src_id"], rel["tgt_id"])))
                if pair in existing_pairs:
                    continue
                existing_pairs.add(pair)
                net_new_relations.append(rel)
            if net_new_relations:
                chunk_relationships.extend(net_new_relations)

            if not net_new_entities and not net_new_relations:
                break

        for ent in chunk_entities:
            maybe_nodes[ent["entity_name"]].append(ent)
        for rel in chunk_relationships:
            maybe_edges[tuple(sorted((rel["src_id"], rel["tgt_id"])))].append(rel)

    logger.info(
        f"Extracted {len(maybe_nodes)} unique entities and {len(maybe_edges)} unique relationships."
    )

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

