import json
import httpx
import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("videogame_server")

RAWG_API_KEY = os.environ.get("RAWG_API_KEY")

@mcp.tool()
async def search_video_games(query: str) -> str:
    """
    Searches for a video game on the RAWG.io database and returns its details in a JSON string.

    Args:
        query (str): The name of the video game to search for.

    Returns:
        str: A JSON string containing the game's details, or an error message.
    """
    if not RAWG_API_KEY or "your_rawg_api_key_here" in RAWG_API_KEY:
        return "Error: RAWG_API_KEY is not configured. Please set it in your .env file."

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.rawg.io/api/games",
                params={"key": RAWG_API_KEY, "search": query, "page_size": 1},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            if data.get("results"):
                game = data["results"][0]
                # Return a structured JSON string for the LLM to interpret
                return json.dumps({
                    "name": game.get("name"),
                    "released": game.get("released"),
                    "rating": f'{game.get("rating")} / {game.get("rating_top")}',
                    "platforms": [p.get("platform", {}).get("name") for p in game.get("platforms", [])],
                    "genres": [g.get("name") for g in game.get("genres", [])],
                })
            else:
                return f"No results found for '{query}'."
    except httpx.RequestError as e:
        return f"Error making request to RAWG API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    # Initialize and run the server using stdio transport
    mcp.run(transport='stdio')