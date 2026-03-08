import asyncio
import json
import os
import sys
from dataclasses import asdict

# Allow running this file directly via:
# python3 playground/knowledge_graph_build/run_build_from_memory.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from playground.knowledge_graph_build.test_data_build import VideoKnowledgeExtractor
from playground.knowledge_graph_build.clean_kg import (
    load_graphml,
    save_graphml,
    unify_entities_conservative,
)
from playground.knowledge_graph_build._utils import compute_mdhash_id


# Edit these manually before running.
INPUT_SEGMENTS_JSON = "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/extracted_data/How_To_Dominate_with_Zaahen_League_of_Legends/kv_store_video_segments.json"
RUN_POST_CLEANING = True
SYNC_ENTITY_VDB_AFTER_CLEANING = True


async def main():
    with open(INPUT_SEGMENTS_JSON, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    extractor = VideoKnowledgeExtractor(video_path="/dummy/path/video.mp4", mcp_sessions={})

    # Optional extraction knobs.
    extractor.entity_extract_max_gleaning = 1
    extractor.extraction_use_domain_context = False
    extractor.extraction_glean_mode = "simple"  # options: "split" | "unified" | "simple"
    extractor.relationship_strength_min = 1.0
    extractor.relationship_strength_max = 10.0

    print(f"Using working directory: {extractor.working_dir}")
    print(f"Loaded videos: {len(segments_data)}")

    await extractor.video_segments.upsert(segments_data)
    await extractor.ainsert(extractor.video_segments._data)

    config = asdict(extractor)
    graph_path = os.path.join(config["working_dir"], "graph_chunk_entity_relation.graphml")
    cleaned_graph_path = os.path.join(config["working_dir"], "graph_chunk_entity_relation_clean.graphml")

    if RUN_POST_CLEANING:
        print("\nRunning post-cleaning with clean_kg.py logic...")
        G = load_graphml(graph_path)
        H = unify_entities_conservative(G)
        save_graphml(H, cleaned_graph_path)
        print(
            f"Graph cleaned: nodes {G.number_of_nodes()} -> {H.number_of_nodes()}, "
            f"edges {G.number_of_edges()} -> {H.number_of_edges()}"
        )
        print(f"Cleaned graph file: {cleaned_graph_path}")

        if SYNC_ENTITY_VDB_AFTER_CLEANING and extractor.entities_vdb is not None:
            print("Rebuilding entities VDB from cleaned graph nodes...")
            vdb_payload = {}
            for node_id, node_data in H.nodes(data=True):
                entity_name = str(node_id)
                description = str(node_data.get("description", ""))
                vdb_payload[compute_mdhash_id(entity_name, prefix="ent-")] = {
                    "content": entity_name + description,
                    "entity_name": entity_name,
                }
            await extractor.entities_vdb.upsert(vdb_payload)
            await extractor.entities_vdb.index_done_callback()
            print(f"Entities VDB synced from cleaned graph ({len(vdb_payload)} nodes).")

    print("\nBuild complete.")
    print(f"Graph file: {graph_path}")
    if RUN_POST_CLEANING:
        print(f"Graph file (cleaned): {cleaned_graph_path}")
    print(f"Entity VDB: {config['working_dir']}/vdb_entities.json")


if __name__ == "__main__":
    asyncio.run(main())
