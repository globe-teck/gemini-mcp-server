"""MCP stdio server exposing one tool: gemini_ask."""
import os

from mcp.server.fastmcp import FastMCP

from .client import GeminiClient, GeminiError, load_api_key, model_chain

mcp = FastMCP("gemini")


def gemini_ask(prompt: str, model: str = "", system: str = "", max_output_tokens: int = 2048) -> str:
    """Ask Google Gemini a question and return its text answer.

    model: optional model name (default from GEMINI_MODEL or gemini-flash-latest); on
    rate limits or outages the server retries with backoff, then falls back to
    GEMINI_FALLBACK_MODELS. system: optional system instruction.
    """
    try:
        client = GeminiClient(load_api_key(), retries=int(os.environ.get("GEMINI_RETRIES", "3")))
        used, text = client.ask(prompt, model_chain(model or None), system, max_output_tokens)
    except GeminiError as e:
        return f"Gemini error: {e}"
    return text if used == (model or model_chain()[0]) else f"[answered by {used}]\n{text}"


mcp.tool()(gemini_ask)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
