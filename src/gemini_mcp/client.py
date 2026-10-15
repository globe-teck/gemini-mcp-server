"""Gemini REST client with key loading, retry/backoff and model fallback.

No MCP dependency here, so the logic is easy to unit test.
"""
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_FALLBACKS = ("gemini-flash-lite-latest",)
RETRY_STATUSES = {429, 500, 503}


class GeminiError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def load_api_key(env=None):
    """GEMINI_API_KEY wins; otherwise read the file named by GEMINI_API_KEY_FILE."""
    env = os.environ if env is None else env
    key = env.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    path = env.get("GEMINI_API_KEY_FILE", "").strip()
    if path:
        p = Path(path).expanduser()
        if p.is_file():
            key = p.read_text().strip()
            if key:
                return key
    raise GeminiError("no API key: set GEMINI_API_KEY (or GEMINI_API_KEY_FILE pointing to a key file)")


def model_chain(model=None, env=None):
    """Primary model followed by fallbacks (GEMINI_FALLBACK_MODELS, comma-separated), de-duplicated."""
    env = os.environ if env is None else env
    primary = model or env.get("GEMINI_MODEL") or DEFAULT_MODEL
    raw = env.get("GEMINI_FALLBACK_MODELS")
    fallbacks = [m.strip() for m in raw.split(",") if m.strip()] if raw is not None else list(DEFAULT_FALLBACKS)
    chain = []
    for m in [primary, *fallbacks]:
        if m not in chain:
            chain.append(m)
    return chain


def build_body(prompt, system="", max_output_tokens=2048, temperature=None):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output_tokens}}
    if temperature is not None:
        body["generationConfig"]["temperature"] = temperature
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def extract_text(data):
    try:
        cand = data["candidates"][0]
        text = "".join(p.get("text", "") for p in cand["content"]["parts"])
    except (KeyError, IndexError, TypeError):
        reason = (data.get("promptFeedback") or {}).get("blockReason") if isinstance(data, dict) else None
        raise GeminiError(f"no text in response{f' (blocked: {reason})' if reason else ''}")
    if not text:
        raise GeminiError(f"empty response (finishReason={cand.get('finishReason')})")
    return text


def _post(url, body, key, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


class GeminiClient:
    def __init__(self, api_key, retries=3, backoff=1.0, max_backoff=20.0, timeout=120,
                 sleep=time.sleep, post=_post):
        self.api_key = api_key
        self.retries = retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.timeout = timeout
        self._sleep = sleep
        self._post = post

    def _delay(self, attempt, retry_after=None):
        if retry_after:
            try:
                return min(float(retry_after), self.max_backoff)
            except ValueError:
                pass
        return min(self.backoff * (2 ** attempt), self.max_backoff) * (0.5 + random.random() / 2)

    def _call_model(self, model, body):
        url = f"{BASE_URL}{model}:generateContent"
        for attempt in range(self.retries + 1):
            try:
                return extract_text(self._post(url, body, self.api_key, self.timeout))
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300] if e.fp else ""
                if e.code in RETRY_STATUSES and attempt < self.retries:
                    self._sleep(self._delay(attempt, e.headers.get("Retry-After") if e.headers else None))
                    continue
                raise GeminiError(f"{model}: HTTP {e.code} {detail}".strip(), status=e.code)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    self._sleep(self._delay(attempt))
                    continue
                raise GeminiError(f"{model}: network error {e}")

    def ask(self, prompt, models, system="", max_output_tokens=2048, temperature=None):
        """Try each model in order; return (model_used, text). Auth errors are not retried or fallen back."""
        body = build_body(prompt, system, max_output_tokens, temperature)
        errors = []
        for model in models:
            try:
                return model, self._call_model(model, body)
            except GeminiError as e:
                if e.status in (400, 401, 403):
                    raise
                errors.append(str(e))
        raise GeminiError("all models failed: " + " | ".join(errors))
