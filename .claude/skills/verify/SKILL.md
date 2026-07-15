---
name: verify
description: Drive qi end-to-end against a stub OpenAI-compatible server to verify agent-loop changes at the CLI surface.
---

# Verifying qi at the CLI surface

qi's surface is the `qi run` CLI talking to an OpenAI-compatible endpoint.
Verify changes by running the real CLI against a scripted stub server — no
API key needed.

## Recipe

1. Stub server: a ~30-line `http.server` script that answers
   `POST /v1/chat/completions` with scripted `{"choices": [...]}` bodies in
   order and appends each request body to a log file. The script file is a
   JSON list of `{"finish_reason": ..., "message": {...}}` choice objects.
2. Temp project dir with:
   - `.qi/config.toml`: `api_key = "stub-key"`, `model = "stub-model"`,
     `base_url = "http://127.0.0.1:<port>/v1"`
   - any fixture files tools will read.
3. Piped mode (the default under a pipe):
   `printf '{"role": "user", "content": "..."}\n' | uv run --project <repo> qi run`
4. Interactive mode needs a tty — wrap with `script`:
   `printf 'answer\n' | script -qec "uv run --project <repo> qi run -p 'task'" /dev/null`
5. Assert on (a) CLI output, (b) the request log — message sequence, roles,
   tool results — and (c) exit code.

## Gotchas

- The turn protocol is structural: tool calls → loop; no tool calls → done.
  Questions are `AskUser` tool calls (piped: deferred tool result +
  next stdin line is the answer; interactive: console prompt).
- The stub's response counter advances on ANY request — don't curl-probe it
  before the run, or restart it between scenarios.
- Session files land in `<workdir>/.qi/sessions/` — useful evidence of the
  logged message history.
