import pytest

pytest.importorskip("mcp")
from gemini_mcp import client as c  # noqa: E402
from gemini_mcp import server  # noqa: E402


def test_tool_reports_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    assert server.gemini_ask("hi").startswith("Gemini error: no API key")


def test_tool_labels_fallback_answers(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "backup")
    monkeypatch.setattr(c.GeminiClient, "ask", lambda self, p, models, *a: ("backup", "text"))
    assert server.gemini_ask("hi", model="primary") == "[answered by backup]\ntext"
    monkeypatch.setattr(c.GeminiClient, "ask", lambda self, p, models, *a: (models[0], "text"))
    assert server.gemini_ask("hi") == "text"


def test_tool_is_registered():
    import asyncio
    names = [t.name for t in asyncio.run(server.mcp.list_tools())]
    assert names == ["gemini_ask"]
