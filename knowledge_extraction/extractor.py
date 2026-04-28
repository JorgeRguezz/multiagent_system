"""
The main orchestration script that manages the video processing pipeline, 
coordinating video splitting and tool calls to the MCP servers.
"""
import asyncio
import argparse
import os
import sys
import json
import shutil
import time
from contextlib import AsyncExitStack
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip

# Add project root and playground to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "playground"))

from knowledge_extraction.config import *
from knowledge_build._videoutil.split import split_video
from knowledge_pipeline.game_profiles import get_active_game_profile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _prepare_workspace_for_video(video_path: str) -> None:
    os.makedirs(WORKING_DIR, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Keep extracted_data across runs; clean only transient artifacts.
    transient_cache_dir = os.path.join(WORKING_DIR, "_cache", video_basename)
    if os.path.isdir(transient_cache_dir):
        shutil.rmtree(transient_cache_dir)

    for entry in os.scandir(WORKING_DIR):
        if entry.is_file() and entry.name.startswith("frame_s") and entry.name.endswith(".png"):
            os.remove(entry.path)


def _dedupe_entity_names(matches: list, limit: int | None = None) -> list[str]:
    ordered_names = []
    seen = set()
    for match in matches:
        name = match.get("name")
        if not name or name in seen:
            continue
        ordered_names.append(name)
        seen.add(name)
        if limit is not None and len(ordered_names) >= limit:
            break
    return ordered_names


def _parse_lol_entity_results(batch_results: dict) -> dict:
    main_champ = "Unknown"
    partners = []

    parsed_main = batch_results.get("middle", [])
    if parsed_main:
        parsed_main.sort(key=lambda x: x["score"], reverse=True)
        main_champ = parsed_main[0]["name"]

    parsed_partners = batch_results.get("partners", [])
    if parsed_partners:
        parsed_partners.sort(key=lambda x: x["score"], reverse=True)
        partners = _dedupe_entity_names(parsed_partners, limit=4)

    entities = []
    if main_champ != "Unknown":
        entities.append(main_champ)
    for partner in partners:
        if partner not in entities:
            entities.append(partner)

    return {
        "main_champ": main_champ,
        "partners": partners,
        "entities": entities,
    }


def _parse_generic_entity_results(batch_results: dict) -> dict:
    parsed_entities = batch_results.get("entities", [])
    if parsed_entities:
        parsed_entities.sort(key=lambda x: x["score"], reverse=True)
    entities = _dedupe_entity_names(parsed_entities)
    return {
        "entities": entities,
    }


def _parse_entity_results(batch_results: dict, parser_mode: str) -> dict:
    if parser_mode == "lol":
        return _parse_lol_entity_results(batch_results)
    if parser_mode == "generic":
        return _parse_generic_entity_results(batch_results)
    raise ValueError(f"Unsupported entity_result_parser={parser_mode!r}")


async def run_pipeline(video_path: str = VIDEO_PATH):
    start_total = time.time()
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    active_profile = get_active_game_profile()
    if not os.path.exists(active_profile.entity_db_path):
        raise FileNotFoundError(
            f"Entity DB for game profile {active_profile.id!r} was not found: "
            f"{active_profile.entity_db_path}"
        )
    print("\n[Config] Active Game Profile")
    print(f" >> Game: {active_profile.display_name} ({active_profile.id})")
    print(f" >> Detection mode: {active_profile.detection_mode}")
    print(f" >> Entity DB: {active_profile.entity_db_path}")

    # 1. Setup workspace for this video only
    _prepare_workspace_for_video(video_path)

    # 2. Split Video
    print("\n[Step 1] Splitting Video...")
    print(f" >> Video: {video_path}")
    start_split = time.time()
    segment_index2name, segment_times_info = split_video(
        video_path=video_path,
        working_dir=WORKING_DIR,
        segment_length=SEGMENT_LENGTH,
        num_frames_per_segment=FRAMES_PER_SEGMENT,
        audio_output_format='mp3'
    )
    end_split = time.time()
    print(f" >> Split complete in {end_split - start_split:.2f}s")

    # Server parameters
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    # -----------------------

    entity_params = StdioServerParameters(
        command=VENV_SMOLVLM_PYTHON,
        args=["-m", "knowledge_extraction.entity_server"],
        env=env
    )
    
    vlm_params = StdioServerParameters(
        command=VENV_VLM_ASR_PYTHON,
        args=["-m", "knowledge_extraction.vlm_asr_server"],
        env=env
    )

    context_data = []

    # 3. Entity Extraction Phase
    print(f"\n[Step 2] Phase 1: Entity Extraction (SAM3 + DINOv2) using {NUM_ENTITY_WORKERS} workers")
    start_entities = time.time()
    
    # 3.1 Pre-extract frames to avoid moviepy contention
    tasks = []
    with VideoFileClip(video_path) as video:
        sorted_indices = sorted(segment_index2name.keys(), key=lambda x: int(x))
        for index in sorted_indices:
            segment_name = segment_index2name[index]
            timestamps = segment_times_info[index]["frame_times"]
            
            for i, t in enumerate(timestamps):
                frame_array = video.get_frame(t)
                frame_path = os.path.join(WORKING_DIR, f"frame_s{index}_i{i}.png")
                Image.fromarray(frame_array.astype('uint8')).save(frame_path)
                
                tasks.append({
                    "frame_path": frame_path,
                    "index": index,
                    "segment_name": segment_name,
                    "frame_idx": i
                })
    
    results = [None] * len(tasks)
    task_queue = asyncio.Queue()
    for i, task in enumerate(tasks):
        task_queue.put_nowait((i, task))

    async def entity_worker(worker_id, session):
        while not task_queue.empty():
            try:
                task_idx, task = await task_queue.get()
            except asyncio.QueueEmpty:
                break
            
            frame_path = task["frame_path"]
            
            try:
                regions_config = [
                    {"name": region["name"], "region": region["region"]}
                    for region in active_profile.regions_config
                ]

                res = await session.call_tool("detect_and_match_regions", arguments={
                    "image_path": frame_path,
                    "regions_config": regions_config,
                    "db_path": active_profile.entity_db_path,
                    "threshold": 0.0
                })

                # Helper to parse dictionary response
                def parse_content(content_list):
                    if not content_list: return {}
                    try:
                        return json.loads(content_list[0].text)
                    except: return {}


                # print(f"    [Worker {worker_id}] Frame s{task['index']}i{task['frame_idx']} - Raw MCP Output: {res.content}")
                batch_results = parse_content(res.content)
                parsed_entities = _parse_entity_results(
                    batch_results,
                    active_profile.entity_result_parser,
                )

                results[task_idx] = {
                    "frame_path": frame_path,
                    "segment_idx": task["index"],
                    "segment_name": task["segment_name"],
                    "frame_idx": task["frame_idx"],
                    "game": active_profile.id,
                    **parsed_entities,
                }
                print(
                    f"    [Worker {worker_id}] Frame s{task['index']}i{task['frame_idx']} "
                    f"Done: entities -> {parsed_entities['entities']}"
                )
                
            except Exception as e:
                print(f"    [Worker {worker_id}] ERROR on {frame_path}: {e}")
            finally:
                task_queue.task_done()

    # Launch workers
    async with AsyncExitStack() as stack:
        sessions = []
        print(f" >> Launching {NUM_ENTITY_WORKERS} entity server instances...")
        for i in range(NUM_ENTITY_WORKERS):
            if i > 0:
                print(f"    Waiting 3s to stagger startup...")
                await asyncio.sleep(3)
                
            client = await stack.enter_async_context(stdio_client(entity_params))
            read, write = client
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            print(f"    [Worker {i}] Initializing models...")
            await session.call_tool("warmup", arguments={"db_path": active_profile.entity_db_path})
            sessions.append(session)
            print(f"    [Worker {i}] Ready! ✅")
        
        worker_tasks = [entity_worker(i, sessions[i]) for i in range(NUM_ENTITY_WORKERS)]
        await asyncio.gather(*worker_tasks)

    context_data = [r for r in results if r is not None]
    end_entities = time.time()

    # 4. VLM & ASR Phase
    print("\n[Step 3] Phase 2: VLM Inference & ASR")
    start_vlm_asr = time.time()
    asr_times = []
    vlm_times = []
    frames_data = {}
    segments_captions = {}
    async with stdio_client(vlm_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Transcribe segments
            transcripts = {}
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            cache_path = os.path.join(WORKING_DIR, '_cache', video_basename)
            
            print(" >> Running ASR...")
            for index in segment_index2name:
                audio_path = os.path.join(cache_path, f"{segment_index2name[index]}.mp3")
                if os.path.exists(audio_path):
                    s_asr = time.time()
                    res = await session.call_tool("transcribe_audio", arguments={"audio_path": audio_path})
                    transcripts[index] = res.content[0].text
                    asr_times.append(time.time() - s_asr)

            # Run VLM
            last_description = "This is the first frame of the video."
            print(" >> Running VLM Inference...")
            for entry in context_data:
                s_vlm = time.time()
                transcript = transcripts.get(entry['segment_idx'], "")

                context = {
                    "game": active_profile.id,
                    "transcript": transcript,
                    "entities": entry.get("entities", []),
                }
                if active_profile.entity_result_parser == "lol":
                    context["champion"] = entry["main_champ"]
                    context["teammates"] = entry["partners"]
                
                # # 2x faster; lower quality
                # description = await session.call_tool("run_qwen_description", arguments={
                #     "image_path": entry['frame_path'],
                #     "context": context,
                #     "last_description": last_description
                # })

                # Slower, higher quality
                description = await session.call_tool("run_internvl_description", arguments={
                    "image_path": entry['frame_path'],
                    "context": context,
                    "last_description": last_description
                })

                vlm_times.append(time.time() - s_vlm)
                print(f" >> VLM Output for {os.path.basename(entry['frame_path'])} (took {vlm_times[-1]:.2f}s)")
                print(f"    {description.content[0].text}")
                last_description = description.content[0].text
                
                # Collect per-frame outputs
                if video_basename not in frames_data:
                    frames_data[video_basename] = {}
                frame_key = f"{entry['segment_idx']}_{entry['frame_idx']}"
                frames_data[video_basename][frame_key] = {
                    "frame_path": entry["frame_path"],
                    "segment_idx": str(entry["segment_idx"]),
                    "segment_name": entry["segment_name"],
                    "frame_idx": entry["frame_idx"],
                    "game": entry["game"],
                    "entities": entry["entities"],
                    "transcript": transcript,
                    "vlm_output": description.content[0].text,
                }
                if active_profile.entity_result_parser == "lol":
                    frames_data[video_basename][frame_key]["main_champ"] = entry["main_champ"]
                    frames_data[video_basename][frame_key]["partners"] = entry["partners"]
                segments_captions.setdefault(entry["segment_idx"], []).append(description.content[0].text)
            # Unload VLM/ASR before starting GPT-OSS to reduce GPU memory pressure
            try:
                await session.call_tool("unload_vlm_asr", arguments={})
            except Exception as e:
                print(f" >> Warning: failed to unload VLM/ASR models: {e}")
    end_vlm_asr = time.time()

    # 5. Merge and persist artifacts for knowledge_build
    print("\n[Step 4] Saving extraction artifacts...")
    gpt_params = StdioServerParameters(
        command=os.path.join(project_root, "venv_gpt/bin/python3"),
        args=["-m", "knowledge_extraction.segment_summarization_server"],
    )
    gpt_session = None
    gpt_times = []
    gpt_total_start = time.time()
    async with stdio_client(gpt_params) as (gpt_read, gpt_write):
        async with ClientSession(gpt_read, gpt_write) as gpt_session:
            await gpt_session.initialize()
            segments_information = {}
            for index in segment_index2name:
                segment_name = segment_index2name[index]
                captions = segments_captions.get(index, [])
                llm_start_time = time.time()
                res = await gpt_session.call_tool(
                    "summarize_segment_captions", arguments={"captions": captions}
                )
                llm_end_time = time.time()
                gpt_times.append(llm_end_time - llm_start_time)
                print(
                    f" >> Segment {index} LLM summary time: {llm_end_time - llm_start_time:.2f}s"
                )
                segment_caption = res.content[0].text if res.content else ""
                segments_information[index] = {
                    "time": "-".join(segment_name.split('-')[-2:]),
                    "content": f"Caption:\n{segment_caption}\nTranscript:\n{transcripts.get(index, '')}\n\n",
                    "transcript": transcripts.get(index, ''),
                    "frame_times": segment_times_info[index]["frame_times"].tolist(),
                }
    gpt_total_end = time.time()
    gpt_total_time = gpt_total_end - gpt_total_start

    segments_data = {video_basename: segments_information}
    paths_data = {video_basename: video_path}

    extracted_data_dir = os.path.join(WORKING_DIR, "extracted_data", video_basename)
    os.makedirs(extracted_data_dir, exist_ok=True)

    with open(os.path.join(extracted_data_dir, "kv_store_video_segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(extracted_data_dir, "kv_store_video_frames.json"), "w", encoding="utf-8") as f:
        json.dump(frames_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(extracted_data_dir, "kv_store_video_path.json"), "w", encoding="utf-8") as f:
        json.dump(paths_data, f, indent=2, ensure_ascii=False)

    # Summary
    end_total = time.time()
    print("\n" + "="*50)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"Total Time:           {end_total - start_total:.2f}s")
    print(f"1. Video Splitting:   {end_split - start_split:.2f}s")
    print(f"2. Entity Extraction: {end_entities - start_entities:.2f}s")
    print(f"3. ASR & VLM Phase:   {end_vlm_asr - start_vlm_asr:.2f}s")
    print(f"4. GPT-OSS Summary:   {gpt_total_time:.2f}s")
    if asr_times:
        print(f"   - Avg ASR/segment: {sum(asr_times)/len(asr_times):.2f}s")
    if vlm_times:
        print(f"   - Avg VLM/frame:   {sum(vlm_times)/len(vlm_times):.2f}s")
    if gpt_times:
        print(f"   - Avg GPT/segment: {sum(gpt_times)/len(gpt_times):.2f}s")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run knowledge extraction pipeline for one video.")
    parser.add_argument(
        "--video-path",
        default=VIDEO_PATH,
        help="Absolute or relative path to the input video file.",
    )
    args = parser.parse_args()
    asyncio.run(run_pipeline(video_path=args.video_path))
