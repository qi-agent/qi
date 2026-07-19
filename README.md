<div align="center">
    <pre>
   ____    _ 
  / __ \  (_)
 / /_/ /  / /
 \__  /  /_/ 
   /_/       
    </pre>
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

## Getting Started

### 1. Install

```sh
pip install qi-agent
```

Or with `uv`:

```sh
uv tool run qi-agent
```

### 2. Init a project

```sh
qi init
```

This creates a `.qi/` directory with a `config.toml` and a `sessions/` folder.
Edit `.qi/config.toml` to point at your LLM.

For example, with [Ollama](https://ollama.com):

```toml
api_key = ""
model = "qwen2.5:7b"
base_url = "http://localhost:11434/v1"
```

### 3. Run

```sh
qi run -p "fix the bug in" main.py
```

## Philosophy

**Unix-philosophy agent.** Qi works in pipelines. Pipe files in, get structured
output out. Composable with `jq`, `glow`, `grep`, or whatever else is in your
toolchain.

**Bring your own LLM.** OpenAI, Anthropic, Google, local (vLLM, Ollama) — any
OpenAI-compatible API. No vendor lock-in, no telemetry.

**Minimal.** The source is a few hundred lines. You can read it in an afternoon
and hack on it by morning.

## Multi-agent

Qi can spin up subagents — independent `qi run` processes wired together with
pipes — as it sees fit, or when you ask it to delegate:

```sh
qi -p "Split this review across one subagent per file, then merge their findings" src/*.py
```

- The `Agent` tool spawns a subagent; its final reply comes back to the parent
  as the tool result.
- `reads_from` pipes the replies of earlier agents into a new one, so fan-out
  and fan-in compose into arbitrary graphs. An agent can only read from agents
  spawned before it, which makes the graph a DAG by construction — no cycles.
- `background=true` runs agents in parallel; the `AgentWait` tool collects
  their replies.

While a subagent runs, its stdin is a named pipe at `.qi/agents/<run>/<name>.in`
(POSIX), speaking the same JSON message protocol as piped mode — so other
processes can inject messages into a running agent.

At the end of a run qi depicts the graph, and `qi graph` re-renders it from the
session log:

```
qi ──▶ researcher  [done]
qi ──▶ tester      [done]
researcher, tester ──▶ summary  [done]
```

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

### `qi init`

```
qi init [--dir <path>] [--force]
```

Scaffold a `.qi/` project directory with a default `config.toml` and
`sessions/` folder.

| Flag | Description |
|---|---|
| `--dir <path>` | Target directory (default: current directory) |
| `-f`, `--force` | Overwrite existing `config.toml` |

### `qi graph`

```
qi graph [<session_id>] [--format ascii|mermaid|dot]
```

Depict the subagent graph recorded in a session (default: the most recent
session in `.qi/sessions/`). `--format mermaid` and `--format dot` emit
diagram sources for embedding or rendering elsewhere.

| Flag | Description |
|---|---|
| `--format` | Output format: `ascii` (default), `mermaid`, or `dot` |

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
| **M5 — Multi-agent** | ✅ Done | Delegation and sub-agents for parallel tasks, wired with pipes into arbitrary graphs; `qi graph` depicts them. |

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
