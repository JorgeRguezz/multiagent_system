
import os
import asyncio
import shutil
from dataclasses import asdict

# Assuming the script is run from the root of the VideoRAG-algorithm project
from videorag.videorag import VideoRAG
from videorag._videoutil import (
    split_video,
    speech_to_text,
    segment_caption,
    merge_segment_information,
)
from videorag._op import get_chunks
from videorag._utils import logger

# --- Configuration ---
# IMPORTANT: Change this to the actual path of the video you want to test
VIDEO_PATH = "examples/hot-dog.mp4" 
WORKING_DIR = "test_bg_cache"


async def main():
    """
    A simplified script to test the core inference pipeline of VideoRAG:
    1. Video processing (splitting, ASR, VLM captioning).
    2. Text processing (chunking, entity extraction).
    3. Knowledge graph construction and saving.
    """
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"Video file not found at: {VIDEO_PATH}")
        logger.error("Please update the VIDEO_PATH variable in this script.")
        return

    # --- 1. Initialization ---
    # We instantiate the VideoRAG class to easily set up all necessary
    # configurations, storage backends, and models.
    logger.info("Initializing VideoRAG and loading models...")
    videorag = VideoRAG(working_dir=WORKING_DIR)
    videorag.load_caption_model()  # Loads the MiniCPM-V-2 model

    video_name = os.path.basename(VIDEO_PATH).split('.')[0]
    video_cache_dir = os.path.join(WORKING_DIR, '_cache', video_name)

    # --- 2. Video Processing ---
    logger.info(f"Processing video: {VIDEO_PATH}")

    # Step 2.1: Split the video into segments
    logger.info("Splitting video into segments...")
    segment_index2name, segment_times_info = split_video(
        VIDEO_PATH,
        WORKING_DIR,
        videorag.video_segment_length,
        videorag.rough_num_frames_per_segment,
        videorag.audio_output_format,
    )
    logger.info(f"Video split into {len(segment_index2name)} segments.")

    # Step 2.2: Perform ASR (Speech-to-Text) on audio segments
    logger.info("Extracting transcripts with ASR...")
    transcripts = speech_to_text(
        video_name,
        WORKING_DIR,
        segment_index2name,
        videorag.audio_output_format
    )
    logger.info("Finished ASR.")
    # print("Transcripts:", transcripts)

    # Step 2.3: Perform VLM captioning on video segments
    logger.info("Generating captions with VLM...")
    captions = {}  # Use a simple dict for sequential processing
    segment_caption(
        video_name=video_name,
        video_path=VIDEO_PATH,
        segment_index2name=segment_index2name,
        transcripts=transcripts,
        segment_times_info=segment_times_info,
        captions=captions,
        error_queue=None,  # Not needed for sequential execution
        model=videorag.caption_model,
        tokenizer=videorag.caption_tokenizer
    )
    logger.info("Finished VLM captioning.")
    # print("Captions:", captions)

    # Step 2.4: Merge all extracted information
    segments_information = merge_segment_information(
        segment_index2name,
        segment_times_info,
        transcripts,
        captions,
    )
    await videorag.video_segments.upsert({video_name: segments_information})
    logger.info("Merged and saved segment information.")

    # --- 3. Text Processing and Knowledge Graph Construction ---
    logger.info("Starting text processing and KG construction...")

    # Step 3.1: Create text chunks from the merged information
    inserting_chunks = get_chunks(
        new_videos={video_name: segments_information},
        chunk_func=videorag.chunk_func,
        max_token_size=videorag.chunk_token_size,
    )
    await videorag.text_chunks.upsert(inserting_chunks)
    logger.info(f"Created {len(inserting_chunks)} text chunks.")

    # Step 3.2: Extract entities and build the knowledge graph
    logger.info("Extracting entities and building knowledge graph...")
    maybe_new_kg, _, _ = await videorag.entity_extraction_func(
        inserting_chunks,
        knowledge_graph_inst=videorag.chunk_entity_relation_graph,
        entity_vdb=videorag.entities_vdb,
        global_config=asdict(videorag),
    )

    if maybe_new_kg is None:
        logger.warning("No new entities found, KG not updated.")
    else:
        videorag.chunk_entity_relation_graph = maybe_new_kg
        # Step 3.3: Save the graph to a file
        await videorag.chunk_entity_relation_graph.index_done_callback()
        graph_path = videorag.chunk_entity_relation_graph.db_path
        logger.info(f"Knowledge graph construction complete. Graph saved to: {graph_path}")

    # --- 4. Cleanup ---
    if os.path.exists(video_cache_dir):
        shutil.rmtree(video_cache_dir)
        logger.info(f"Cleaned up cache directory: {video_cache_dir}")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    # Ensure you have your LLM provider (e.g., OpenAI) API keys set up
    # in your environment for entity extraction to work.
    asyncio.run(main())
