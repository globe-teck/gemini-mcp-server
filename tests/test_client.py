import io
import urllib.error
import urllib.request

import pytest

from gemini_mcp import client as c


def _block_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted in tests")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


_no_network = pytest.fixture(autouse=True)(_block_network)


def ok(text="hi"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("u", code, "err", headers, io.BytesIO(b'{"error":"x"}'))


class FakePost:
    """Scripted responses keyed by model; records calls."""
    def __init__(self, script):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls = []

    def __call__(self, url, body, key, timeout):
        model = url.split("/models/")[1].split(":")[0]
        self.calls.append((model, body, key))
        item = self.script[model].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(script, **kw):
    post, sleeps = FakePost(script), []
    return c.GeminiClient("test-key", post=post, sleep=sleeps.append, **kw), post, sleeps


def test_key_from_env_then_file(tmp_path):
    assert c.load_api_key({"GEMINI_API_KEY": " abc "}) == "abc"
    f = tmp_path / "k"
    f.write_text("fromfile\n")
    assert c.load_api_key({"GEMINI_API_KEY_FILE": str(f)}) == "fromfile"
    assert c.load_api_key({"GEMINI_API_KEY": "env", "GEMINI_API_KEY_FILE": str(f)}) == "env"
    with pytest.raises(c.GeminiError):
        c.load_api_key({"GEMINI_API_KEY_FILE": str(tmp_path / "missing")})


def test_model_chain():
    assert c.model_chain(env={}) == ["gemini-flash-latest", "gemini-flash-lite-latest"]
    assert c.model_chain("a", {"GEMINI_FALLBACK_MODELS": "b, a ,c"}) == ["a", "b", "c"]
    assert c.model_chain(env={"GEMINI_MODEL": "x", "GEMINI_FALLBACK_MODELS": ""}) == ["x"]


def test_body_and_extract():
    b = c.build_body("p", system="s", max_output_tokens=10, temperature=0)
    assert b["systemInstruction"] == {"parts": [{"text": "s"}]}
    assert b["generationConfig"] == {"maxOutputTokens": 10, "temperature": 0}
    assert c.extract_text(ok("yo")) == "yo"
    with pytest.raises(c.GeminiError, match="blocked: SAFETY"):
        c.extract_text({"promptFeedback": {"blockReason": "SAFETY"}})


def test_success_sends_key_header():
    cl, post, sleeps = make({"m": [ok("answer")]})
    assert cl.ask("q", ["m"]) == ("m", "answer")
    assert post.calls[0][2] == "test-key" and sleeps == []


def test_retry_on_429_and_503_with_backoff():
    cl, post, sleeps = make({"m": [http_error(429), http_error(503), ok()]}, backoff=1, max_backoff=8)
    assert cl.ask("q", ["m"]) == ("m", "hi")
    assert len(post.calls) == 3 and len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1 and 1 <= sleeps[1] <= 2


def test_retry_after_header_respected():
    cl, _, sleeps = make({"m": [http_error(429, "7"), ok()]}, max_backoff=20)
    cl.ask("q", ["m"])
    assert sleeps == [7.0]


def test_fallback_after_retries_exhausted():
    cl, post, _ = make({"a": [http_error(503)] * 3, "b": [ok("from b")]}, retries=2)
    assert cl.ask("q", ["a", "b"]) == ("b", "from b")
    assert [m for m, *_ in post.calls] == ["a", "a", "a", "b"]


def test_fallback_on_404_without_retry():
    cl, post, sleeps = make({"a": [http_error(404)], "b": [ok()]})
    assert cl.ask("q", ["a", "b"])[0] == "b" and sleeps == []


def test_auth_error_is_not_fallen_back():
    cl, post, _ = make({"a": [http_error(403)], "b": [ok()]})
    with pytest.raises(c.GeminiError) as ei:
        cl.ask("q", ["a", "b"])
    assert ei.value.status == 403 and len(post.calls) == 1


def test_all_models_fail():
    cl, _, _ = make({"a": [http_error(500)] * 2, "b": [urllib.error.URLError("down")] * 2}, retries=1)
    with pytest.raises(c.GeminiError, match="all models failed"):
        cl.ask("q", ["a", "b"])


def test_default_post_uses_urlopen_and_is_blocked():
    with pytest.raises(AssertionError, match="network"):
        c._post("https://example.com", {}, "k", 1)
