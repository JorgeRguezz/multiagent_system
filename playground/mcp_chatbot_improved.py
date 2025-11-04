import json 
import os
import asyncio  # Provides the foundation for running asynchronous code for non-blocking operations.
import nest_asyncio  # Allows the asyncio event loop to be nested, useful in environments where an event loop is already running.
import re  # Used for regular expressions, used for detecting tool calls in the model's output.
from mcp import ClientSession, StdioServerParameters  # Core components from the mcp library for session management and server connection parameters.
from mcp.client.stdio import stdio_client  # Provides the function to establish a connection to a server using standard I/O.
from contextlib import AsyncExitStack  # Manages multiple asynchronous context managers for proper cleanup.

from dotenv import load_dotenv  # Used to load environment variables from a .env file for configuration.
from llama_cpp import Llama

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:
    # 1 -> Initializes the MCP_ChatBot, setting up the LLM, sampling parameters, and initial state.
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.history = []
        self.system_prompt = "You are a helpful assistant." 

        model_name = "/home/gatv-projects2/Desktop/project/Qwen3-GGUF/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
        self.llm = Llama(
            model_path=model_name,
            n_gpu_layers=-1,        
            n_ctx=4096,             
            verbose=True
        )

        self.sampling_params = {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 2024,
            "stop": ["\nUser:"]
        }

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
        # First, check for a direct tool call from the user
        direct_tool_match = re.search(r"@(\w+)\(([^)]*)\)", query)

        if direct_tool_match:
            tool_name = direct_tool_match.group(1)
            tool_arg = direct_tool_match.group(2)

            print(f"[DEBUG] Direct tool call detected: '{tool_name}' with argument: '{tool_arg}'")
            
            self.history.append(f"User: {query}")
            session = self.sessions.get(tool_name)
            if not session:
                print(f"Error: Tool '{tool_name}' not found.")
                return

            try:
                new_prompt = "\n".join(self.history) + "\nAssistant:"
                new_response = self.llm(
                    new_prompt,
                    max_tokens=self.sampling_params["max_tokens"],
                    temperature=self.sampling_params["temperature"],
                    top_p=self.sampling_params["top_p"],
                    stop=self.sampling_params["stop"]
                )
                final_output = new_response['choices'][0]['text'].strip()

                self.history.append(f"Assistant: {final_output}")
                print(final_output)
            except Exception as e:
                print(f"Error calling tool '{tool_name}': {e}")
            return # End processing here for direct call

        # If no direct call, proceed with the original LLM-based logic
        self.history.append(f"User: {query}")
        prompt = self.system_prompt + "\n\n" + "\n".join(self.history) + "\nAssistant:"
        response = self.llm(
            prompt,
            max_tokens=self.sampling_params["max_tokens"],
            temperature=self.sampling_params["temperature"],
            top_p=self.sampling_params["top_p"],
            stop=self.sampling_params["stop"]
        )
        output = response['choices'][0]['text'].strip()

        llm_tool_match = re.search(r"@(\w+)\(([^)]*)\)", output)

        if llm_tool_match:
            tool_name = llm_tool_match.group(1)
            tool_arg = llm_tool_match.group(2)

            print(f"[DEBUG] LLM tool call detected: '{tool_name}' with argument: '{tool_arg}'")

            session = self.sessions.get(tool_name)
            if not session:
                print(f"Error: Tool '{tool_name}' not found.")
                self.history.append(f"Assistant: {output}")
                print(output)
                return

            try:
                result = await session.call_tool(tool_name, arguments={"query": tool_arg})
                self.history.append(f"Assistant: {output}") # Save the LLM output that contained the call
                self.history.append(f"Tool result ({tool_name}): {result.content}")

                new_prompt = "\n".join(self.history) + "\nAssistant:"
                new_response = self.llm(
                    new_prompt,
                    max_tokens=self.sampling_params["max_tokens"],
                    temperature=self.sampling_params["temperature"],
                    top_p=self.sampling_params["top_p"],
                    stop=self.sampling_params["stop"]
                )
                final_output = new_response['choices'][0]['text'].strip()

                self.history.append(f"Assistant: {final_output}")
                print(final_output)
            except Exception as e:
                print(f"Error calling tool '{tool_name}': {e}")
                self.history.append(f"Assistant: {output}")
                print(output)
        else:
            self.history.append(f"Assistant: {output}")
            print(output)


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
                await self.process_query(text)
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
                
                await self.process_query(query)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    # 9 -> Cleans up resources, such as closing the exit stack.
    async def cleanup(self):
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