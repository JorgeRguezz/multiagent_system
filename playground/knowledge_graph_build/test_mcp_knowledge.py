import asyncio
import os
import json
from contextlib import AsyncExitStack
from pathlib import Path
import shutil

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPToolTester:
    """
    A basic test class to verify the functionality of the high-level
    `extract_video_knowledge` MCP tool.
    """
    def __init__(self, video_path, mcp_sessions):
        self.video_path = video_path
        self.mcp_sessions = mcp_sessions

    async def run_test(self):
        """Runs the full test pipeline."""
        print("--- Starting MCP Knowledge Extraction Test ---")
        
        extraction_session = self.mcp_sessions.get('extract_video_knowledge')
        if not extraction_session:
            raise RuntimeError("Tool 'extract_video_knowledge' not found in connected MCP servers.")

        server_working_dir = None
        try:
            # Step 1: Call the high-level tool
            print(f"\n--> Client: Calling 'extract_video_knowledge' for video: {self.video_path}")
            result = await extraction_session.call_tool(
                'extract_video_knowledge', 
                arguments={'video_path': self.video_path}
            )
            
            if not result.content:
                raise Exception("MCP tool returned no content.")

            # Step 2: Parse the response from the server
            response_data = json.loads(result.content[0].text)
            if "error" in response_data:
                raise Exception(f"Server returned an error: {response_data['error']}")

            print("\n--> Client: Received response from server.")
            video_segments_db_path = response_data.get("video_segments_db_path")
            server_working_dir = response_data.get("working_dir")
            
            if not video_segments_db_path or not os.path.exists(video_segments_db_path):
                raise FileNotFoundError(f"Server did not return a valid path for the video segments DB. Path: {video_segments_db_path}")

            print(f"  - Server-side working directory: {server_working_dir}")
            print(f"  - Video Segments DB path: {video_segments_db_path}")

            # Step 3: Load and print the data created by the server
            print("\n--> Client: Reading the database file created by the server...")
            with open(video_segments_db_path, "r") as f:
                segments_data = json.load(f)

            print("\n--- Test Result: Content of video_segments.json ---")
            print(json.dumps(segments_data, indent=4, ensure_ascii=False))
            print("----------------------------------------------------")

        except Exception as e:
            print(f"\n--- ❌ Test Failed ---")
            print(f"An error occurred: {e}")
            return
        finally:
            # Clean up the directory created by the server
            if server_working_dir and os.path.exists(server_working_dir):
                print(f"\n--> Client: Cleaning up server-side directory: {server_working_dir}")
                shutil.rmtree(server_working_dir)


        print("\n--- ✅ MCP Knowledge Extraction Test Finished Successfully! ---")


async def main():
    """Main function to set up MCP client and run the test."""
    VIDEO_FILE = "/home/gatv-projects/Desktop/project/chatbot_system/downloads/My_Nintendo_Switch_2_Review.mp4"
    
    if not os.path.exists(VIDEO_FILE):
        print(f"Error: Video file not found at '{VIDEO_FILE}'.")
        return

    async with AsyncExitStack() as exit_stack:
        sessions = {}
        
        try:
            with open(Path(__file__).parent/"server_config.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
        except FileNotFoundError:
            print("Error: `server_config.json` not found.")
            return
        except Exception as e:
            print(f"Error loading server_config.json: {e}")
            return

        vlm_server_config = servers.get('vlm_server')
        if not vlm_server_config:
            print("Error: 'vlm_server' not found in server_config.json.")
            return

        try:
            print("Connecting to VLM/ASR server via MCP...")
            server_params = StdioServerParameters(**vlm_server_config)
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            
            response = await session.list_tools()
            for tool in response.tools:
                sessions[tool.name] = session
            print("Successfully connected to VLM/ASR server.")

        except Exception as e:
            print(f"Error connecting to VLM/ASR server: {e}")
            return

        tester = MCPToolTester(video_path=os.path.abspath(VIDEO_FILE), mcp_sessions=sessions)
        await tester.run_test()


if __name__ == '__main__':
    # To run this script, navigate to the `playground` directory and use:
    # python -m knowledge_graph_build.test_mcp_knowledge
    asyncio.run(main())
