# Knowledge Extraction Phase

## Purpose of the module

`knowledge_extraction/` is the first operational phase of the end-to-end knowledge pipeline. Its job is to convert an input gameplay video into structured intermediate artifacts that later phases can sanitize, chunk, index, and transform into a knowledge graph.

At a high level, this phase does five things:

1. Splits a source video into fixed-duration temporal segments.
2. Samples a fixed number of frames from each segment.
3. Detects champion identities from HUD regions in each sampled frame.
4. Transcribes each segment's audio and produces a dense visual-language description for each sampled frame.
5. Aggregates per-frame outputs into per-segment summaries and saves the final extraction artifacts as JSON.

The phase is implemented as an orchestrated multi-process pipeline. The main controller is [`knowledge_extraction/extractor.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/extractor.py), but three separate MCP-compatible subprocess servers perform the heavy model inference:

- [`knowledge_extraction/entity_server.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/entity_server.py): champion detection and matching.
- [`knowledge_extraction/vlm_asr_server.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/vlm_asr_server.py): ASR and per-frame VLM descriptions.
- [`knowledge_extraction/segment_summarization_server.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/segment_summarization_server.py): per-segment caption summarization.

## File-level responsibilities

### `config.py`

[`knowledge_extraction/config.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/config.py) centralizes hard-coded paths and extraction parameters:

- `PROJECT_ROOT`: absolute project path.
- `VENV_SMOLVLM_PYTHON`: Python interpreter used to start `entity_server.py`.
- `VENV_VLM_ASR_PYTHON`: Python interpreter used to start `vlm_asr_server.py`.
- `HF_HOME`: Hugging Face cache root.
- `VIDEO_PATH`: default input video.
- `WORKING_DIR`: extraction cache directory, currently `knowledge_extraction/cache`.
- `DB_PATH`: NanoVectorDB file containing champion reference embeddings.
- `SEGMENT_LENGTH = 30`: segment duration in seconds.
- `FRAMES_PER_SEGMENT = 5`: number of frames sampled per segment.
- `MAX_SEGMENTS = 6`: defined but not used by the current extractor.
- `NUM_ENTITY_WORKERS = 3`: number of parallel entity server workers.
- `MIDDLE_HUD = [500, 900, 1350, 1080]`: ROI used to identify the main player champion.
- `BOTTOM_RIGHT_HUD = [1500, 600, 1923, 750]`: ROI used to identify teammate portraits.

These constants are not just configuration in the abstract; they directly determine the temporal resolution, visual sampling density, concurrency, and semantics of the extracted metadata.

### `extractor.py`

[`knowledge_extraction/extractor.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/extractor.py) is the orchestration entry point. It:

- validates the input video path,
- cleans transient cache state,
- calls the video splitter,
- launches and warms multiple entity extraction workers,
- pre-extracts sampled frames to PNG files,
- runs ASR on per-segment audio,
- runs VLM descriptions on every sampled frame,
- launches a summarization LLM for segment-level aggregation,
- writes JSON artifacts to `knowledge_extraction/cache/extracted_data/<video_name>/`.

### `entity_server.py`

[`knowledge_extraction/entity_server.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/entity_server.py) exposes MCP tools that detect and identify champions visible in fixed HUD regions. It combines:

- SAM3 image grounding for region-level box detection,
- DINOv2-derived global image embeddings for initial nearest-neighbor retrieval,
- patch-token MaxSim rescoring for the final identity decision,
- NanoVectorDB as the persistent reference index.

### `vlm_asr_server.py`

[`knowledge_extraction/vlm_asr_server.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/vlm_asr_server.py) exposes MCP tools for:

- Whisper ASR transcription,
- InternVL3-14B per-frame gameplay descriptions,
- unloading models to reduce GPU pressure.

It also contains a Qwen2.5-VL path, but the active extractor currently calls the InternVL path, not the Qwen path.

### `segment_summarization_server.py`

[`knowledge_extraction/segment_summarization_server.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/segment_summarization_server.py) loads a local GGUF version of GPT-OSS-20B through `llama_cpp` and merges the five frame-level descriptions of each 30-second segment into a single segment caption.

### `run_extraction_queue.py`

[`knowledge_extraction/run_extraction_queue.py`](/home/gatv-projects/Desktop/project/knowledge_extraction/run_extraction_queue.py) is a sequential batch runner. It scans `downloads/queue`, skips videos that already have the three required extraction outputs, and invokes `python -m knowledge_extraction.extractor --video-path <video>`.

This script is operational glue for dataset generation, not part of the per-video inference logic itself.

## End-to-end control flow

## 1. Workspace preparation

The first concrete action inside `run_pipeline()` is `_prepare_workspace_for_video(video_path)`.

This function:

- ensures `WORKING_DIR` exists,
- derives the current video's basename,
- deletes only the transient cache folder `knowledge_extraction/cache/_cache/<video_basename>`,
- deletes root-level frame files matching `frame_s*.png`,
- intentionally preserves `knowledge_extraction/cache/extracted_data/`.

This means the module is designed to be rerunnable without deleting prior finished outputs. It cleans temporary artifacts, not the historical extraction corpus.

## 2. Video splitting

The pipeline then calls `split_video()` from [`knowledge_build/_videoutil/split.py`](/home/gatv-projects/Desktop/project/knowledge_build/_videoutil/split.py). Although the splitter lives under `knowledge_build`, it is operationally part of extraction because extraction uses it directly.

`split_video()` performs the following logic:

- Opens the full video with `moviepy.VideoFileClip`.
- Computes segment start times as `range(0, total_video_length, segment_length)`.
- If the final segment would be shorter than 5 seconds, it drops that last start time and merges the tail into the previous segment.
- Creates a per-video transient cache directory at `knowledge_extraction/cache/_cache/<video_name>/`.
- For each segment:
  - computes `end`,
  - creates a subclip,
  - samples `num_frames_per_segment` timestamps using `numpy.linspace(..., endpoint=False)`,
  - offsets those timestamps into absolute video time,
  - stores a synthetic segment name of the form `<millisecond_timestamp>-<segment_index>-<start>-<end>`,
  - writes the subclip audio to MP3.

The splitter returns two structures:

- `segment_index2name`: maps segment index strings like `"0"` to the synthetic segment name.
- `segment_times_info`: maps the same index to:
  - `frame_times`: absolute timestamps for sampled frames,
  - `timestamp`: `(start, end)`.

Important implication: extraction artifacts are segment-based, but the frame timestamps recorded later are absolute timestamps in the original full video, not offsets relative to the segment.

## 3. Runtime environment for subprocess servers

Before launching MCP subprocesses, `extractor.py` prepares an environment dictionary:

- `PYTORCH_ALLOC_CONF = "expandable_segments:True"`
- `OMP_NUM_THREADS = "1"`
- `MKL_NUM_THREADS = "1"`

The commented-out `HF_HOME` line shows the system was prepared to pin Hugging Face cache location, but the active code does not export it into the subprocess environment.

The extractor then defines two `StdioServerParameters` blocks:

- `entity_params`: starts `knowledge_extraction.entity_server` under `VENV_SMOLVLM_PYTHON`.
- `vlm_params`: starts `knowledge_extraction.vlm_asr_server` under `VENV_VLM_ASR_PYTHON`.

Later, a third server is launched:

- `gpt_params`: starts `knowledge_extraction.segment_summarization_server` under `venv_gpt/bin/python3`.

So the phase is intentionally split across separate Python environments to accommodate incompatible model stacks.

## 4. Frame extraction strategy

Entity extraction is parallelized, but raw frame decoding is not. The code explicitly pre-extracts all sampled frames before worker inference because the comment states this avoids MoviePy contention.

For every segment and every sampled timestamp:

- `video.get_frame(t)` reads the frame from the original full video,
- the frame is saved to `knowledge_extraction/cache/frame_s<segment>_i<frame>.png`,
- a task record is created with:
  - `frame_path`,
  - `index` (segment index),
  - `segment_name`,
  - `frame_idx`.

This produces a flat task list covering all sampled frames across the entire input video.

## 5. Entity extraction phase

### 5.1 Worker topology

The extractor creates:

- a shared `asyncio.Queue` of frame tasks,
- a `results` list aligned by task index,
- `NUM_ENTITY_WORKERS` separate MCP client sessions,
- one `entity_worker()` coroutine per session.

Workers are started with a 3-second stagger after the first one. This is a practical GPU/RAM stabilization measure during model initialization.

Each session is explicitly warmed by calling:

```json
{"db_path": "<DB_PATH>"}
```

on the `warmup` MCP tool. That loads SAM3 and the vector database before real requests begin.

### 5.2 Region semantics

For every frame, the extractor requests entity matches for two named ROIs in a single tool call:

- `middle`: `MIDDLE_HUD`
- `partners`: `BOTTOM_RIGHT_HUD`

The naming is semantically important:

- `middle` is interpreted as the main player champion.
- `partners` is interpreted as teammate portraits, from which up to four unique names are kept.

### 5.3 Entity server internals

`entity_server.py` maintains several global singletons:

- `sam3_model`
- `sam3_processor`
- `db`
- `db_token_cache`

This makes each MCP server stateful. Models and database contents are loaded once per subprocess and reused for subsequent frames.

#### SAM3 loading

`load_sam3()` builds a SAM3 image model using `build_sam3_image_model()` and wraps it in `Sam3Processor`.

#### Vector DB loading

`load_db(db_path)` opens a `NanoVectorDB` instance with:

- embedding dimension `1024`,
- storage file `DB_PATH`,
- reference vectors representing known champion images.

#### ROI preprocessing

Each detected crop is normalized through `preprocess_image()`:

- forced to RGB,
- isotropically resized to fit inside `224 x 224`,
- padded with black background.

This guarantees size compatibility for the DINOv2 embedding functions.

#### Detection and matching algorithm

For each named region:

1. Crop the ROI from the full frame.
2. Run `sam3_processor.set_image(roi)`.
3. Run `sam3_processor.set_text_prompt(..., prompt="Character.")`.
4. Read `output["boxes"]`.
5. For each box:
   - convert coordinates to integers,
   - crop the detected subimage,
   - reject detections with height `0`,
   - reject detections whose aspect ratio is outside `[0.7, 1.4]`.
6. Preprocess the accepted crop to `224 x 224`.
7. Compute a global embedding with `embed_image(processed_cutout)`.
8. Query `db.query(query_embedding, top_k=30)`.
9. Compute patch-token embeddings for the query crop with `embed_patch_tokens`.
10. Remove mostly black padded patches using `select_tokens_by_padding_mask`.
11. For each retrieved candidate:
    - load or reuse its patch tokens from `db_token_cache`,
    - compute `maxsim_score(q_tokens, d_tokens)`,
    - keep the candidate with the highest local score.
12. If `best_score > threshold`, emit `{"name": <champion>, "score": <score>}`.

The effective matching strategy is therefore two-stage:

- global nearest-neighbor retrieval to narrow candidates,
- local patch-token rescoring to choose the final identity.

### 5.4 Post-processing in the extractor

After `detect_and_match_regions()` returns:

- `middle` matches are sorted descending by score,
- the top result becomes `main_champ`,
- default is `"Unknown"` if nothing is found.

For `partners`:

- matches are sorted descending by score,
- duplicates are removed while preserving score order,
- at most four names are kept.

Each completed frame result is stored as:

```json
{
  "frame_path": "...",
  "main_champ": "Smolder",
  "partners": ["Lux", "Irelia", "Fizz", "Kayn"],
  "segment_idx": "1",
  "segment_name": "1773133195703-1-30-60",
  "frame_idx": 3
}
```

Only non-`None` results survive into `context_data`.

## 6. ASR phase

The extractor then opens a single session to `vlm_asr_server.py`.

For every segment:

- it builds `audio_path = knowledge_extraction/cache/_cache/<video_basename>/<segment_name>.mp3`,
- if the MP3 exists, it calls `transcribe_audio(audio_path)`.

`transcribe_audio()`:

- lazily loads Whisper `base`,
- runs `asr_model.transcribe(audio_path)`,
- serializes the result as one concatenated string,
- formats each chunk as:
  - `[<start>s -> <end>s] <text>`

The output is not a structured list of segments; it is a single string containing timestamped utterance spans. This string is later copied both into frame records and segment records.

## 7. VLM description phase

### 7.1 Frame-level context package

For every frame result in `context_data`, the extractor creates:

```json
{
  "champion": "<main_champ>",
  "teammates": ["..."],
  "transcript": "<segment transcript>"
}
```

This context is sent along with the image path and the previous frame description.

### 7.2 Temporal chaining

The variable `last_description` starts as:

`"This is the first frame of the video."`

After each VLM call, it is replaced with the new VLM output. Therefore each frame description is conditioned on the description of the immediately previous sampled frame, even if the next frame belongs to a new segment. Temporal continuity is global across the entire video, not reset per segment.

### 7.3 Active model path: InternVL

The active extractor uses:

- `run_internvl_description(...)`

not `run_qwen_description(...)`.

Inside `vlm_asr_server.py`, `run_internvl_description()`:

1. Loads InternVL3-14B in 8-bit quantized mode using `BitsAndBytesConfig(load_in_8bit=True)`.
2. Loads the tokenizer with `trust_remote_code=True`.
3. Converts the image into a tiled tensor representation:
   - computes the closest aspect-ratio grid,
   - resizes the image accordingly,
   - crops it into multiple `448 x 448` tiles,
   - optionally appends a thumbnail tile,
   - stacks them into one tensor.
4. Builds a dense prompt containing:
   - player champion,
   - teammate names,
   - transcript context,
   - explicit instructions to prioritize what changed since the last frame,
   - the previous description as delta context.
5. Calls `internvl_model.chat(...)` with:
   - `max_new_tokens=512`,
   - `do_sample=False`,
   - `temperature=1.0`.

### 7.4 Prompt design

The InternVL prompt is explicitly optimized for gameplay-state extraction rather than generic captioning. It prioritizes:

- new banners, gold numbers, and kill notifications,
- action intensity,
- player state from HUD,
- map logic and relative positioning,
- tying visible actions to transcript context.

It also instructs the model not to repeat static background details from the prior frame. This is an attempt to push the VLM toward delta-style state tracking rather than isolated image descriptions.

### 7.5 Stored frame outputs

For each processed frame, the extractor writes a record under key `<segment_idx>_<frame_idx>`.

The stored schema is:

```json
{
  "frame_path": "/abs/path/to/frame.png",
  "segment_idx": "1",
  "segment_name": "1773133195703-1-30-60",
  "frame_idx": 3,
  "main_champ": "Smolder",
  "partners": ["Lux", "Irelia", "Fizz", "Kayn"],
  "transcript": "[0.00s -> 3.96s] ...",
  "vlm_output": "Dense textual scene description ..."
}
```

This is the canonical per-frame artifact consumed downstream.

The extractor also builds `segments_captions`, which is a map:

- key: segment index
- value: ordered list of the five VLM outputs belonging to that segment

### 7.6 Model unloading

After all ASR and VLM calls finish, the extractor invokes `unload_vlm_asr()`, which:

- drops Qwen objects if loaded,
- drops InternVL objects if loaded,
- drops Whisper if loaded,
- empties CUDA cache,
- synchronizes CUDA.

This exists specifically to reduce GPU pressure before starting the summarization LLM.

## 8. Segment summarization phase

The final inference stage uses `segment_summarization_server.py`.

### 8.1 Model loading

`get_llm()` lazily downloads or resolves:

- repo: `unsloth/gpt-oss-20b-GGUF`
- file: `gpt-oss-20b-F16.gguf`

It then creates a `llama_cpp.Llama` instance with:

- `n_gpu_layers = -1`
- `n_ctx = 20000`
- `n_batch = 512`
- `f16_kv = True`

### 8.2 Input construction

For each segment:

- the five per-frame descriptions are merged using the separator:
  - `--- Next Segment ---`
- the merged text is inserted into a user prompt asking for a detailed description of what is happening in the gameplay segment.

The system prompt frames the model as an expert summarizer of League of Legends gameplay descriptions and asks it to focus on champions, actions, interactions, and inferred gameplay events.

### 8.3 Output cleanup

The code expects GPT-OSS style channel markup. It strips the response after the regex:

- `final<\|message\|>`

If the pattern exists, only the final answer portion is returned. Otherwise the raw output is returned as malformed-but-usable text.

### 8.4 Segment record assembly

For each segment index, the extractor writes:

```json
{
  "time": "30-60",
  "content": "Caption:\n<segment summary>\nTranscript:\n<segment transcript>\n\n",
  "transcript": "<segment transcript>",
  "frame_times": [30.0, 36.0, 42.0, 48.0, 54.0]
}
```

Important details:

- `time` is derived from the last two dash-separated fields in `segment_name`.
- `content` is a concatenated text block that already combines segment caption and transcript.
- `transcript` is also stored separately as its own field.
- `frame_times` is serialized using `tolist()`, so it becomes standard JSON numeric arrays.

## 9. Persisted outputs

The final outputs are written under:

`knowledge_extraction/cache/extracted_data/<video_basename>/`

Three JSON files are required:

### `kv_store_video_segments.json`

Top-level shape:

```json
{
  "<video_name>": {
    "<segment_idx>": {
      "time": "start-end",
      "content": "Caption:\n...\nTranscript:\n...\n\n",
      "transcript": "...",
      "frame_times": [ ... ]
    }
  }
}
```

This is the segment-level artifact used later for chunking and entity extraction in `knowledge_build`.

### `kv_store_video_frames.json`

Top-level shape:

```json
{
  "<video_name>": {
    "<segment_idx>_<frame_idx>": {
      "frame_path": "...",
      "segment_idx": "...",
      "segment_name": "...",
      "frame_idx": 0,
      "main_champ": "...",
      "partners": [ ... ],
      "transcript": "...",
      "vlm_output": "..."
    }
  }
}
```

This is the fine-grained frame-level artifact.

### `kv_store_video_path.json`

Top-level shape:

```json
{
  "<video_name>": "/absolute/path/to/source/video.mp4"
}
```

This preserves provenance back to the original source media.

## 10. Contract with downstream modules

This phase is not an isolated research prototype. Its outputs are the formal handoff into the next modules.

`knowledge_sanitization/config.py` defines the extraction root as:

- `knowledge_extraction/cache/extracted_data`

and expects exactly these three files:

- `kv_store_video_segments.json`
- `kv_store_video_frames.json`
- `kv_store_video_path.json`

`knowledge_build/builder.py` then loads these artifacts and inserts them into:

- `video_segments`
- `video_frames`
- `video_path`

stores before performing chunking, entity extraction, graph construction, and vector indexing.

So the extraction phase is best understood as the multimodal evidence generation and normalization layer for the rest of the system.

## 11. Important implementation characteristics

### Fixed temporal granularity

The current implementation samples:

- one segment every 30 seconds,
- five frames per segment,
- therefore one frame every 6 seconds within each full 30-second segment.

This is a deliberate compression strategy. The system does not operate on every frame or even dense video snippets. It creates sparse but information-rich multimodal checkpoints.

### HUD-centric champion grounding

Champion identity is inferred from HUD regions, not from the world scene. This is a strong architectural choice:

- `middle` HUD identifies the player champion,
- `bottom-right` portraits identify teammates,
- the system avoids the harder problem of full-scene champion re-identification.

### Cross-modal fusion happens through prompting, not explicit symbolic alignment

The VLM receives:

- image pixels,
- champion guesses from entity extraction,
- teammate guesses,
- transcript text,
- previous frame description.

These modalities are fused in the prompt layer. There is no explicit learned multimodal fusion network at the application level; the fusion is delegated to the VLM.

### Global temporal memory is shallow and textual

Temporal continuity is represented only through `last_description`, meaning:

- memory depth is one step,
- the memory is textual, not latent,
- the memory persists across segment boundaries.

### Stateful server design

Each subprocess caches heavy resources:

- SAM3,
- NanoVectorDB,
- Whisper,
- InternVL,
- reference patch tokens,
- GPT-OSS model.

This avoids reloading costs on every call but makes each server a long-lived stateful component rather than a stateless function.

## 12. Practical execution path

For a single video, the default entry point is:

```bash
python -m knowledge_extraction.extractor --video-path <path-to-video>
```

For a directory queue, the operational entry point is:

```bash
python -m knowledge_extraction.run_extraction_queue
```

The queue runner skips videos that already have all three required output JSON files unless `--force` is passed.

## 13. Concise conceptual summary

`knowledge_extraction/` converts a raw gameplay video into a sparse, multimodal, segment-aligned evidence store.

Its internal logic is:

1. segment the video,
2. sample frames,
3. identify champions from HUD crops,
4. transcribe audio,
5. describe each sampled frame with transcript-aware VLM prompting,
6. summarize all frame descriptions into segment captions,
7. save segment-level, frame-level, and provenance-level JSON files.

Those JSON files are the formal interface between raw video and the later graph-building stages of the system.
