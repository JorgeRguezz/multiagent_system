# GEMINI.md - Project Context & Agent Guide

This file provides the comprehensive context required for an AI agent to understand, maintain, and extend this specific codebase. **Read this first.**

## 1. Project Identity: Multi-Modal MCP Chatbot

This is a sophisticated chatbot system designed to integrate local Large Language Models (LLMs) with specialized tools (Vision-Language Models, Web Search, Knowledge Graphs) using the **Model Context Protocol (MCP)**.

**Core Mission:** To enable a central LLM to "see" video content, "hear" audio, and "remember" complex relationships by delegating heavy processing tasks to specialized, isolated server processes.

## 2. System Architecture

The project utilizes a distributed, multi-process architecture to solve the "Dependency Hell" problem common in AI development (specifically, conflicting `torch` and `transformers` versions between LLMs and VLMs).

### The MCP Pattern (Model Context Protocol)
*   **Client (`mcp_chatbot.py`):** The main brain. It runs the conversation loop, manages history, and decides when to call tools. It connects to servers via standard I/O (`stdin`/`stdout`).
*   **Servers (`media_processing_tool.py`, etc.):** Standalone scripts running in their own processes (and often their own virtual environments). They expose functions via the `@mcp.tool()` decorator.
*   **Communication:** The Client launches Servers as subprocesses. JSON-RPC messages are exchanged over stdio.

### The Dual-Environment Strategy
**CRITICAL:** The system *cannot* run in a single Python environment.
1.  **`venv_llm` (The "Brain" Env):**
    *   **Role:** Runs the Main Chatbot Client, Knowledge Graph logic, and lightweight tools.
    *   **Key Libs:** `vllm` (for LLM inference), `mcp`, `networkx`, `sentence-transformers`.
    *   **Path:** `venv_llm/`
2.  **`venv_smolvlm` (The "Eyes & Ears" Env):**
    *   **Role:** Runs the `media_processing_tool.py` server. Handles heavy GPU tasks like ASR and Image Captioning.
    *   **Key Libs:** `transformers` (specific version for SmolVLM2), `torch`, `openai-whisper`, `moviepy`.
    *   **Path:** `venv_smolvlm/`

## 3. Key File Map

### `chatbot_system/` (Production Core)
*   **`mcp_chatbot.py`**: **Entry Point.** The specific implementation of the MCP Client.
*   **`server_config.json`**: **Registry.** Defines available tools and, crucially, the *specific python interpreter path* to use for each tool (mapping tools to environments).
*   **`media_processing_tool.py`**: **The VLM Server.** An MCP server that accepts video URLs/files, performs Whisper ASR and SmolVLM2 captioning, saves the result to JSON, and returns the file path.
*   **`videogame_search_tool.py`**: An example of a lightweight MCP tool (RAWG API) running in the main env.

### `playground/` (Labs & Research)
*   **`knowledge_graph_build/`**: The engine room for the RAG pipeline.
    *   **`test_build_knowledge.py`**: The "Main" script for testing extraction. It acts as a client to the media tool and builds the graph locally.
*   **`rag_implementation/`**: The retrieval logic.
    *   **`query_video_match.py`**: Implements the Hybrid Search (Vector + Graph) logic to answer questions.

## 4. The Data Flow: Video to Answer

1.  **Ingestion:** User provides a YouTube URL.
2.  **Processing (Remote MCP Call):**
    *   `mcp_chatbot` calls `extract_video_knowledge` on `media_processing_tool`.
    *   Server (in `venv_smolvlm`) downloads video -> Splits -> ASR (Whisper) -> Caption (SmolVLM2).
    *   Server writes raw data to `downloads/<id>_data.json`.
3.  **Graph Construction (Local):**
    *   Chatbot reads the JSON.
    *   Uses `knowledge_graph/extractor.py` to chunk text and extract entities (People, Locations, Concepts) using a local LLM.
    *   Builds:
        *   **Vector DB:** Embeddings of text chunks.
        *   **NetworkX Graph:** Links entities to chunks.
4.  **Retrieval (Hybrid RAG):**
    *   User asks a question.
    *   System performs **Vector Search** (finding semantic matches).
    *   System performs **Graph Traversal** (finding chunks linked to key entities in the query).
    *   Context is combined and fed to the LLM for the final answer.

## 5. Developer Guidelines

*   **Never mix dependencies:** When adding a new library, verify which environment it belongs to. If it conflicts with `vllm`, it goes in `venv_smolvlm` (or a new env) and must be wrapped as an MCP tool.
*   **Test in Playground:** Use `playground/` scripts to test model inference or graph logic *before* integrating into `mcp_chatbot.py`.
*   **Configuration:** Always check `server_config.json` when debugging "tool not found" or environment errors. Ensure paths are absolute or correctly relative.