import inspect
import pydantic

print(f"pydantic version: {pydantic.__version__}\n")

try:
    from mcp import StdioServerParameters
    print("--- Inspecting mcp.StdioServerParameters ---")

    print("\n>>> Signature:")
    try:
        sig = inspect.signature(StdioServerParameters)
        print(sig)
    except Exception as e:
        print(f"Could not get signature: {e}")

    print("\n>>> Docstring:")
    try:
        doc = inspect.getdoc(StdioServerParameters)
        print(doc)
    except Exception as e:
        print(f"Could not get docstring: {e}")

    print("\n>>> __init__ Docstring:")
    try:
        init_doc = inspect.getdoc(StdioServerParameters.__init__)
        print(init_doc)
    except Exception as e:
        print(f"Could not get __init__ docstring: {e}")

except ImportError:
    print("Error: Could not import StdioServerParameters from mcp.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
