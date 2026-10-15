# gemini-mcp-server

[![CI](https://github.com/leighsrodney-prog/gemini-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/leighsrodney-prog/gemini-mcp-server/actions/workflows/ci.yml)

A tiny [Model Context Protocol](https://modelcontextprotocol.io) **stdio server** that gives any MCP
client one tool, `gemini_ask`, which sends a prompt to Google Gemini and returns the text answer.
It retries with backoff on rate limits and outages, and falls back to another model when one fails.

## Why a tiny MCP server?
- **A second opinion inside your agent.** Let one assistant consult another model for cross-checks, reviews or drafts without leaving the workflow.
- **Small surface, easy to audit.** One tool and about 150 lines of logic. The key comes from env, nothing else is exposed, and you can read all of it in a few minutes.
- **A good MCP learning project.** Stdio transport, a typed tool signature and clean error strings are the core of most MCP integrations.

## Setup (uv)
```bash
git clone <this-repo-url> gemini-mcp-server
cd gemini-mcp-server
uv sync                      # creates .venv and installs mcp (+ pytest in the dev group)
read -rs GEMINI_API_KEY && export GEMINI_API_KEY   # from Google AI Studio
uv run gemini-mcp-server     # speaks MCP over stdio; normally your client launches it
uv run pytest -q             # all HTTP is mocked
```

### Configuration (environment variables)
| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | API key (preferred) | none |
| `GEMINI_API_KEY_FILE` | Path to a file holding the key, used only if `GEMINI_API_KEY` is unset | none |
| `GEMINI_MODEL` | Primary model | `gemini-flash-latest` |
| `GEMINI_FALLBACK_MODELS` | Comma-separated fallbacks (empty = none) | `gemini-flash-lite-latest` |
| `GEMINI_RETRIES` | Retries per model on 429/500/503/network errors | `3` |

Error handling: 429/500/503 and network errors are retried with exponential backoff and jitter,
and a `Retry-After` header is honored. Once retries run out, or on a 404 (unknown model), the next
model in the chain is tried. 400/401/403 (bad request or bad key) fail fast, because a fallback
wouldn't help. Errors come back to the client as a readable `Gemini error: ...` string instead of a crash.
If a fallback model answers, its reply is prefixed with `[answered by <model>]`.

## Add it to an MCP client
Most clients (Claude Desktop, Cursor and others) accept a JSON entry like this:
```json
{
  "mcpServers": {
    "gemini": {
      "command": "uv",
      "args": ["--directory", "/path/to/gemini-mcp-server", "run", "gemini-mcp-server"],
      "env": { "GEMINI_API_KEY_FILE": "/path/outside/the/repo/gemini.key" }
    }
  }
}
```
It's better to point `GEMINI_API_KEY_FILE` at a file with `chmod 600` permissions than to paste the key into
client config. Restart the client and the `gemini_ask` tool will appear.

## Free-tier caveats
- The Gemini API free tier has **low per-minute and per-day request limits** that vary by model and change over time, so expect 429s under load. That's why the retry and fallback logic exists.
- Free-tier prompts and responses **may be used by Google to improve its products**. Don't send confidential or personal data.
- `*-latest` aliases can change underneath you. Pin a specific model version if you need reproducibility.
- Availability, quotas and terms are set by Google. Check the current Gemini API pricing and terms pages before relying on it.

## Roadmap
- Streaming responses (`streamGenerateContent`)
- Token-usage and latency reporting in tool output
- Optional per-process rate limiter to stay under free-tier quotas proactively
- Multi-turn context and file/image inputs
- Structured (JSON-schema) output mode

## How this was built
Built with AI coding assistance as part of my AI-agent operations work. I defined the requirements and error-handling behavior, ran the tests, and reviewed the changes.

## License
MIT. See [LICENSE](LICENSE).
