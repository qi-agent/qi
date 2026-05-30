<div align="center">
  <img src="img/qi.svg" alt="Qi logo" width="120">
  <h1>Qi</h1>
  <p><em>a minimalist coding agent</em></p>
</div>

<p align="center">
  <a href="https://pypi.org/project/qi-agent/"><img src="https://img.shields.io/pypi/v/qi-agent" alt="PyPI"></a>
  <a href="https://github.com/qi-agent/qi/actions"><img src="https://github.com/qi-agent/qi/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/pypi/pyversions/qi-agent" alt="Python versions">
  <img src="https://img.shields.io/pypi/l/qi-agent" alt="ISC">
</p>

Qi connects your local files to an LLM of your choice, executes tools natively,
and loops until the job is done. Designed for pipes, sessions, and zero ceremony.

## Quickstart

```sh
pip install qi-agent
export QI_API_KEY="sk-..."
qi run -p "fix the bug in" main.py
```

Can also run with `uv`:

```sh
uv tool run qi-agent run -p "..." main.py
```

## Philosophy

**Unix-philosophy agent.** Qi works in pipelines. Pipe files in, get structured
output out. Composable with `jq`, `glow`, `grep`, or whatever else is in your
toolchain.

**Bring your own LLM.** OpenAI, Anthropic, Google, local (vLLM, Ollama) — any
OpenAI-compatible API. No vendor lock-in, no telemetry.

**Minimal.** The source is a few hundred lines. You can read it in an afternoon
and hack on it by morning.


## Reference

### `qi run`

```
qi run [-p <prompt>] <file...>
```

Process files with an LLM. The default command.

| Flag | Description |
|---|---|
| `-p`, `--prompt` | Instruction for the LLM (default: "Analyse the following.") |

`file` is one or more file paths to read into context. If none provided, a
prompt is required.

### `qi ping`

```
qi ping
```

Test LLM connectivity. Sends "Say pong" and prints the response.

### Global flags

| Flag | Description |
|---|---|
| `--help`, `-h` | Show help text |
| `--version`   | Show version |

## Configuration

Precedence (lowest to highest):

1. `~/.config/qi/config.toml`
3. `.qi/config.toml` (project-local)
4. Environment variables (`QI_*`)

| Env var | Config key | Default |
|---|---|---|
| `QI_API_KEY` | `api_key` | — |
| `QI_BASE_URL` | `base_url` | `https://api.openai.com/v1` |
| `QI_MODEL` | `model` | `gpt-4o` |
| `QI_TEMPERATURE` | `temperature` | `0.0` |
| `QI_MAX_TOKENS` | `max_tokens` | `4096` |

Example `.qi/config.toml` for a Gemma4 model using Ollama:

```toml
api_key = ""
model = "gemma4:26b-mlx"
base_url = "http://localhost:11434/v1"
```

Using Google Generative Language API:

```toml
api_key = "YOUR_API_KEY_HERE"
model = "gemini-flash-latest"
base_url = "https://generativelanguage.googleapis.com"
```


## Examples

```sh
# Basic analysis
qi path/to/module.py

# Pipe to other tools
qi -p "Summarize the changes" src/qi/cli.py | glow

# Multiple files
qi -p "Review all for security issues" src/qi/*.py
```

## Roadmap

| Milestone | Status | What |
|---|---|---|
| **M1 — Edit code** | ✅ Done | Core loop: LLM reads files, makes tool calls, outputs results. |
| **M2 — Sessions** | 🔜 Next | Every run logged to JSONL; resume with `qi resume <id>`. |
| **M3 — MCP tools** | 🔜 Planned | Load MCP servers from config, expose tools to LLM. |
| **M4 — Pipeline mode** | 🔜 Planned | Streaming output, stdin integration, true pipe composition. |
| **M5 — Multi-agent** | 🔜 Future | Delegation and sub-agents for parallel tasks. |

## Development

```sh
uv sync
uv run ruff check src/
uv run mypy src/
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

ISC &mdash; see [LICENSE](LICENSE).
