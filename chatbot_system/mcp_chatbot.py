import json 
import os
import asyncio
import nest_asyncio
import re
import torch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from gpu_manager import GPUModelManager


nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:
    # 1 -> Initializes the MCP_ChatBot, setting up the LLM, sampling parameters, and initial state.
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.history = []
        self.system_prompt = "You are a helpful assistant."
        self.llm = None
        self.vlm_processor = None

        llm_config = {
            "model_name": "/home/gatv-projects/Desktop/project/llama-3.2-3B-Instruct",
            "dtype": "float16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.35, # 0.35 -> lowest it can go
            "load_format": "safetensors"
        }

        vlm_config = {
            "model_id": "HuggingFaceTB/SmolVLM2-256M-Instruct"
        }

        self.gpu_manager = GPUModelManager(
            llm_config=llm_config,
            vlm_config=vlm_config
        )
        
        try:
            self.gpu_manager.load_models_to_gpu()
            self.llm = self.gpu_manager.get_llm()
            self.vlm_processor = self.gpu_manager.get_vlm_processor()
        except Exception as e:
            print(f"Failed to load models: {e}")
            raise

        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9, # measures the randomness/creativity of words used by the model (Higher = more creative)
            max_tokens=2024,
            stop=["User:", "Assistant:"]
        )

        # Tools list required for Anthropic API
        self.available_tools = []
        # Prompts list for quick display 
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}

    # 2 -> Connects to a single MCP server, lists its tools, prompts, and resources, and stores them.
    async def connect_to_server(self, server_name, server_config):
        try:
            print(f"[DIAGNOSTIC] Attempting to connect to server: {server_name}...")



            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            print(f"[DIAGNOSTIC] Successfully connected to {server_name}.")
            
            
            try:
                # List available tools
                print(f"[DIAGNOSTIC] Listing tools for {server_name}...")
                response = await session.list_tools()
                for tool in response.tools:
                    print(f"[DIAGNOSTIC] Found tool: {tool.name}")
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
            
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })

                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
            
            except Exception as e:
                print(f"[DIAGNOSTIC] Error listing tools/resources for {server_name}: {e}")
                
        except Exception as e:
            print(f"[DIAGNOSTIC] Error connecting to {server_name}: {e}")

    def _build_system_prompt(self):
        if not self.available_tools:
            return

        tool_descriptions = []
        for tool in self.available_tools:
            tool_descriptions.append(
                f'- @{tool["name"]}(query): {tool["description"]}'
            )

        self.system_prompt = (
            "You are a helpful assistant. You have access to the following tools to answer user questions. "
            "When a tool is needed, you should call it using the format: `@tool_name(argument)`. "
            "For example, to search for 'The Last of Us', you would say: `@search_video_games(The Last of Us)`\n\n"
            "Available tools:\n"
            + "\n".join(tool_descriptions)
        )

    # 3 -> Reads server configurations from 'server_config.json' and connects to each server.
    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
            self._build_system_prompt()
        except Exception as e:
            print(f"Error loading server config: {e}")
            raise

    # 4 -> Processes a user query, generates a response from the LLM, handles tool calls if detected, and prints the final output.
    async def process_query(self, query):
        if not self.llm:
            return "Models are not loaded. Please restart the application."

        # First, check for a direct tool call from the user or youtube video
        direct_tool_match = re.search(r"@(\w+)\(([^)]*)\)", query)
 
        YOUTUBE_PATTERN = r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://(?:www\.)?youtu\.be/[\w-]+"

        youtube_match = re.search(YOUTUBE_PATTERN, query)
 
        if youtube_match:
            youtube_url = youtube_match.group(0)
            print(f"[DEBUG] Youtube video detected: '{youtube_url}'")

            try:
                print("[DEBUG] Downloading video...")
                mp4_path = self.vlm_processor.download_youtube_video(youtube_url)
                print(f"[DEBUG] Video downloaded to: {mp4_path}")
                
                print("[DEBUG] Analyzing video with VLM...")
                vlm_response = self.vlm_processor.analyze_video(mp4_path, query)
                print(f"[DEBUG] VLM analysis complete.")
                return vlm_response
            except torch.cuda.OutOfMemoryError:
                print("GPU out of memory during VLM analysis. Cleaning up.")
                self.gpu_manager.cleanup()
                self.llm = None
                self.vlm_processor = None
                return "A critical memory error occurred. The models have been unloaded. Please restart the application."
            except Exception as e:
                print(f"An error occurred during video analysis: {e}")
                return f"Sorry, an error occurred while analyzing the video: {e}"
        else:
            print(f"[DEBUG] No youtube video detected")

        if direct_tool_match:
            tool_name = direct_tool_match.group(1)  
            tool_arg = direct_tool_match.group(2)

            print(f"[DEBUG] Direct tool call detected: '{tool_name}' with argument: '{tool_arg}'")
            
            self.history.append(f"User: {query}")
            session = self.sessions.get(tool_name)
            if not session:
                return f"Error: Tool '{tool_name}' not found."

            try:
                result = await session.call_tool(tool_name, arguments={"query": tool_arg})
                self.history.append(f"Tool result ({tool_name}): {result.content}")

                new_prompt = "\n".join(self.history) + "\nAssistant:"
                new_response = self.llm.generate([new_prompt], sampling_params=self.sampling_params)
                final_output = new_response[0].outputs[0].text.strip()

                self.history.append(f"Assistant: {final_output}")
                return final_output
            except Exception as e:
                return f"Error calling tool '{tool_name}': {e}"

        # If no direct call, proceed with the original LLM-based logic
        self.history.append(f"User: {query}")
        prompt = self.system_prompt + "\n\n" + "\n".join(self.history) + "\nAssistant:"
        response = self.llm.generate([prompt], sampling_params=self.sampling_params)
        output = response[0].outputs[0].text.strip()

        llm_tool_match = re.search(r"@(\w+)\(([^)]*)\)", output)

        if llm_tool_match:
            tool_name = llm_tool_match.group(1)
            tool_arg = llm_tool_match.group(2)

            print(f"[DEBUG] LLM tool call detected: '{tool_name}' with argument: '{tool_arg}'")

            session = self.sessions.get(tool_name)
            if not session:
                self.history.append(f"Assistant: {output}")
                return output

            try:
                result = await session.call_tool(tool_name, arguments={"query": tool_arg})
                self.history.append(f"Assistant: {output}") # Save the LLM output that contained the call
                self.history.append(f"Tool result ({tool_name}): {result.content}")

                new_prompt = "\n".join(self.history) + "\nAssistant:"
                new_response = self.llm.generate([new_prompt], sampling_params=self.sampling_params)
                final_output = new_response[0].outputs[0].text.strip()

                self.history.append(f"Assistant: {final_output}")
                return final_output
            except Exception as e:
                self.history.append(f"Assistant: {output}")
                return f"Error calling tool '{tool_name}': {e}"
        else:
            self.history.append(f"Assistant: {output}")
            return output


    # 5 -> Retrieves and displays the content of a specified resource URI.
    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)
            
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")
    
    # 6 -> Lists all available prompts that can be executed.
    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return
        
        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")
    
    # 7 -> Executes a specified prompt with given arguments and processes the resulting text.
    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return
        
        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                
                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) 
                                  for item in prompt_content)
                
                print(f"\nExecuting prompt '{prompt_name}'...")
                response = await self.process_query(text)
                if response:
                    print(f"Assistant: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    # 8 -> The main loop for user interaction, handling user input, commands, and queries.
    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
        
                if query.lower() == 'quit':
                    break
                
                # Check for /command syntax
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()
                    
                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        prompt_name = parts[1]
                        args = {}
                        
                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value
                        
                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue
                
                response = await self.process_query(query)
                if response:
                    print(f"Assistant: {response}")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    # 9 -> Cleans up resources, such as closing the exit stack.
    async def cleanup(self):
        if self.gpu_manager:
            self.gpu_manager.cleanup()
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())