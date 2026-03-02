# Multi-Modal MCP Chatbot System

This project is a sophisticated, multi-component chatbot system designed to integrate local Large Language Models (LLMs) with Vision-Language Models (VLMs) and external tools using the **Model Context Protocol (MCP)**.

The system features a modular architecture where the main chatbot acts as a client connecting to various "servers" (tools) to perform complex tasks like video knowledge extraction, internet searches, and more. It includes a custom implementation of a Knowledge Graph RAG (Retrieval-Augmented Generation) pipeline for video content.

## 📂 Project Structure

The project is divided into two main directories to distinguish between the core production system and experimental code.

### 1. `chatbot_system/` (Core System)
This directory contains the stable, production-ready code for the chatbot and its tools.

*   **`mcp_chatbot.py`**: The main entry point. It initializes the LLM, manages conversation history, and orchestrates connections to MCP servers defined in `server_config.json`.
*   **`server_config.json`**: Configuration file defining the available MCP servers and their execution commands.
*   **`knowledge_graph/`**: A comprehensive package for building and querying knowledge graphs from video content.
    *   **`extractor.py`**: Coordinates the extraction pipeline, sending video processing requests to the media tool and building the local graph/vector databases.
    *   **`_storage/`**: Drivers for different storage backends (NetworkX, Neo4j, NanoVectorDB, JSON KV).
    *   **`_videoutil/`**: Client-side utilities for video chunking and feature extraction.
    *   **`_op.py`** & **`_llm.py`**: Operations for entity extraction and LLM interactions.
*   **`media_processing_tool.py`**: A dedicated MCP server responsible for heavy media processing (ASR via Whisper, Captioning via SmolVLM2). It runs in a separate environment.
*   **`videogame_search_tool.py`**: An MCP server that provides video game data via the RAWG API.
*   **`gpu_manager.py`**: Utility for managing GPU memory and model loading.

### 2. `playground/` (Labs & Tests)
This directory is for research, prototyping, and testing. **Do not use code here for production.**

#### General Tests & Diagnostics
Scripts to verify hardware, libraries, and basic model connections.

*   **`ars_test.py`**: Tests Automatic Speech Recognition (ASR). Runs OpenAI's `whisper` and `faster-whisper` on a specific video file to verify that audio transcription works.
*   **`check_device.py`**: A simple diagnostic to verify if `llama_cpp` can successfully load a GGUF model on the current hardware.
*   **`diagnose_mcp.py`**: Debugs the `mcp` library installation, checking for key classes to ensure the correct version is installed.
*   **`diagnose_torch.py`**: System health check. Prints Python/PyTorch versions, CUDA availability, and detailed GPU memory statistics.
*   **`mcp_chatbot_improved.py`**: An experimental variation of the chatbot using `llama_cpp` (GGUF models) instead of `vllm`, useful for testing on lower VRAM hardware.
*   **`test_api.py`**: Tests the standalone Flask VLM server (`smolvlm2_api.py`) by sending a Python HTTP POST request.
*   **`test_api.sh`**: Shell script containing `curl` commands to test the Flask VLM server endpoints.
*   **`test_smolvlm2.py`**: Direct, interactive test of the `SmolVLM2` model locally (no API), allowing chat with a video file.
*   **`test_smolvlm2_desc.py`**: Single-shot inference script for `SmolVLM2` to generate a detailed video description.
*   **`prueba.py`**: Basic sanity check for the `vllm` library, generating text from simple prompts.
*   **`run_qwen3_gguf.py`**: Downloads and runs a quantized Qwen model using `llama_cpp`.

#### `playground/knowledge_graph_build/` (The "GraphRAG" Engine)
Contains the logic for converting video content into a queryable Knowledge Graph. It mirrors the production structure but includes experimental backends.

**Core Logic (Variants):**
*   **`_llm.py`**: Standard implementation using local `vllm` and `sentence-transformers`.
*   **`_llm_gemini_api.py`**: Variant using Google's **Gemini API** for embeddings/generation.
*   **`_llm_openai_api.py`**: Variant using **OpenAI/Azure/DeepSeek APIs**.
*   **`_op.py` / `_op_api.py`**: Operations for graph construction (chunking, entity extraction). `_op_api.py` is optimized for API-based calls.

**Pipelines & Integration Tests:**
*   **`test_end_to_end.py`**: **The Main End-to-End Test.** Takes a raw video, calls the local MCP tool for processing, and builds the full Knowledge Graph locally.
*   **`test_end_to_end.py`**: End-to-end test utilizing the Gemini API.
*   **`test_data_build.py`**: Faster test that skips video processing by loading pre-computed JSON data (`kv_store_video_segments_*.json`) to focus on graph construction.
*   **`test_data_build_gemini.py`**: Same as above, but using Gemini API.
*   **`test_knowledge_extraction.py`**: Runs ASR and VLM logic locally in a standalone script (bypassing MCP) to debug inference issues.
*   **`test_mcp_knowledge.py`**: Tests the MCP connection, verifying the `extract_video_knowledge` tool returns correct JSON paths.

**Visualization & Data:**
*   **`visualize_graph.ipynb`**: Jupyter Notebook that reads `graph_*.graphml` files and renders interactive PyVis networks.
*   **`kv_store_video_segments_*.json`**: Sample output data used for testing without running heavy models.

#### `playground/rag_implementation/` (Retrieval Logic)
Scripts testing how to retrieve answers from the constructed graph.

*   **`video_content_embedding.py`**: Summarizes and embeds video content to facilitate finding the relevant video for a query.
*   **`query_video_match.py`**: Core retrieval logic:
    1.  Matches query to the best video.
    2.  Extracts keywords/entities.
    3.  Performs **Vector Search** (Standard RAG).
    4.  Performs **Graph Search** (Graph RAG).
*   **`full_qa.py`**: Simulates a full Q&A flow ("Who is Ahri...?") by running the pipeline and generating an answer.

#### `playground/pruebas_qwen_model/`
*   **`download_qwen.py`**: Utility to download the Qwen model snapshot.
*   **`prueba_qwen.py`**: Inference test for the Qwen GPTQ model.

---

## 🛠️ Installation & Setup

Due to conflicting dependencies between `vllm` (used by the main chatbot) and `transformers`/`torch` versions required by the VLM, this project **requires two separate Python virtual environments**.

### Environment 1: LLM & Main Chatbot (`venv_llm`)
This environment hosts the main application, the Knowledge Graph logic, and the Game Search tool.

1.  **Create and activate:**
    ```bash
    python3 -m venv venv_llm
    source venv_llm/bin/activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r chatbot_system/requirements.txt
    ```
    *Note: Ensure `vllm` and `mcp` are properly installed.*

### Environment 2: Media Processing (`venv_smolvlm`)
This environment hosts the `media_processing_tool.py` which runs the Vision-Language Model and ASR.

1.  **Create and activate:**
    ```bash
    python3 -m venv venv_smolvlm
    source venv_smolvlm/bin/activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r chatbot_system/smolvlm2_api_requirements.txt
    # You also need 'mcp' and 'moviepy' here if not in the txt
    pip install mcp moviepy openai-whisper
    ```

---

## 🚀 Usage

1.  **Configure API Keys:**
    Create a `.env` file in the root directory:
    ```env
    RAWG_API_KEY="your_rawg_api_key"
    # Any other required keys
    ```

2.  **Configure Servers:**
    Ensure `chatbot_system/server_config.json` points to the correct python interpreters for your environments.
    ```json
    {
      "mcpServers": {
        "videogame": {
          "command": "python3",
          "args": ["-m", "chatbot_system.videogame_search_tool"]
        },
        "vlm_server": {
          "command": "/absolute/path/to/project/venv_smolvlm/bin/python3",
          "args": ["-m", "chatbot_system.media_processing_tool"],
          "cwd": "/absolute/path/to/project"
        }
      }
    }
    ```

3.  **Run the System:**
    Activate the **LLM environment** and run the main script:
    ```bash
    source venv_llm/bin/activate
    python -m chatbot_system.mcp_chatbot
    ```

## 🧠 Features

*   **Interactive CLI**: A command-line interface that supports natural language queries.
*   **Tool Use**: The LLM can autonomously decide to call tools (e.g., `@search_video_games`) based on user input.
*   **Video Analysis**:
    *   Paste a YouTube URL to download and process it.
    *   The system splits the video, transcribes audio (Whisper), and captions frames (SmolVLM2).
    *   It builds a Knowledge Graph and Vector Index locally to answer complex questions about the video content.
*   **Modular Design**: Easily add new tools by creating a script and adding it to `server_config.json`.
