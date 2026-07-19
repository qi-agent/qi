"""Multi-agent runtime: subagents as child qi processes wired with pipes.

A subagent is an independent ``qi run`` process. Its task travels on the
command line; input from upstream agents arrives on stdin as a user-role JSON
message (the existing piped-mode protocol); its JSONL event stream on stdout
is watched for the final assistant reply.

Topology: an agent may only read from agents spawned before it, so the wiring
is a DAG by construction -- arbitrary fan-out and fan-in, never a cycle. On
POSIX each agent's stdin is a named pipe (``.qi/agents/<run>/<name>.in``)
while the agent runs, so outside processes can inject extra user messages
with the same protocol; on Windows a plain pipe is used instead.
"""

import atexit
import contextlib
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qi.lib.constants import LogKey, RecordType, Role

logger = logging.getLogger(__name__)

DEPTH_ENV = "QI_AGENT_DEPTH"
MAX_DEPTH = 3
MAX_AGENTS = 32
DEFAULT_TIMEOUT = 300
MAX_TIMEOUT = 3600
MAX_REPLY_CHARS = 50_000
STDERR_TAIL_LINES = 20
ROOT_NAME = "qi"

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_KILLED = "killed"

# Keys and values of agent-event records (session log meta and event sink).
EVENT_KEY = "event"
NAME_KEY = "name"
PROMPT_KEY = "prompt"
READS_FROM_KEY = "reads_from"
BACKGROUND_KEY = "background"
DETAIL_KEY = "detail"
EVENT_SPAWN = "spawn"

SUBAGENT_PROMPT = (
    "You are subagent '{name}' working for a parent qi agent. Work autonomously: "
    "never call AskUser; make reasonable assumptions and note them. Your final "
    "reply (a message with no tool calls) is returned verbatim to the parent, so "
    "make it a complete, self-contained result.\n\nTASK: {prompt}"
)

EventSink = Callable[[dict[str, Any]], None]


@dataclass
class AgentNode:
    """Pure graph data for one subagent; safe to rebuild from a session log."""

    name: str
    prompt: str = ""
    reads_from: tuple[str, ...] = ()
    background: bool = False
    status: str = STATUS_RUNNING
    detail: str = ""


class AgentGraph:
    """The subagent wiring of one run.

    Nodes are added in spawn order and may only read from existing nodes, so
    the graph is a DAG by construction.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, AgentNode] = {}

    def add(self, node: AgentNode) -> None:
        if node.name == ROOT_NAME:
            raise ValueError(f"'{ROOT_NAME}' is the root agent's name")
        if node.name in self.nodes:
            raise ValueError(f"an agent named '{node.name}' already exists")
        for upstream in node.reads_from:
            if upstream not in self.nodes:
                raise ValueError(f"unknown upstream agent '{upstream}' (agents can only read from agents spawned before them)")
        self.nodes[node.name] = node

    def render_ascii(self) -> str:
        if not self.nodes:
            return "(no agents)"
        rows: list[tuple[str, str, str]] = []
        for node in self.nodes.values():
            sources = ", ".join(node.reads_from) or ROOT_NAME
            label = node.status + (f": {node.detail}" if node.detail else "")
            rows.append((sources, node.name, label))
        width = max(len(sources) for sources, _, _ in rows)
        return "\n".join(f"{sources.rjust(width)} ──▶ {name}  [{label}]" for sources, name, label in rows)

    def render_mermaid(self) -> str:
        lines = ["flowchart LR", f"    {ROOT_NAME}(({ROOT_NAME}))"]
        for node in self.nodes.values():
            lines.append(f'    {_mermaid_id(node.name)}["{node.name}<br>({node.status})"]')
        for node in self.nodes.values():
            for source in node.reads_from or (ROOT_NAME,):
                lines.append(f"    {_mermaid_id(source)} --> {_mermaid_id(node.name)}")
        return "\n".join(lines)

    def render_dot(self) -> str:
        lines = ["digraph agents {", "  rankdir=LR;", f'  "{ROOT_NAME}" [shape=doublecircle];']
        for node in self.nodes.values():
            lines.append(f'  "{node.name}" [label="{node.name}\\n{node.status}"];')
        for node in self.nodes.values():
            for source in node.reads_from or (ROOT_NAME,):
                lines.append(f'  "{source}" -> "{node.name}";')
        lines.append("}")
        return "\n".join(lines)


def _mermaid_id(name: str) -> str:
    return re.sub(r"\W", "_", name)


def _sanitize_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip())[:40].strip("-")


def _current_depth() -> int:
    try:
        return int(os.environ.get(DEPTH_ENV, "0"))
    except ValueError:
        return 0


def _default_child_argv(prompt: str, files: list[str]) -> list[str]:
    return [sys.executable, "-m", "qi", "run", "--output-format", "jsonl", "-p", prompt, *files]


def _stderr_tail() -> deque[str]:
    return deque(maxlen=STDERR_TAIL_LINES)


class _StdinWriter(Protocol):
    def write_line(self, line: str) -> None: ...

    def close(self) -> None: ...


class _FdWriter:
    """Write end of a named pipe; closing it delivers EOF to the child."""

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    def write_line(self, line: str) -> None:
        if self._fd is not None:
            os.write(self._fd, line.encode())

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class _PipeWriter:
    """Plain-pipe fallback where named pipes are unavailable (Windows)."""

    def __init__(self, proc: "subprocess.Popen[str]") -> None:
        self._stdin = proc.stdin

    def write_line(self, line: str) -> None:
        if self._stdin is not None:
            self._stdin.write(line)
            self._stdin.flush()

    def close(self) -> None:
        if self._stdin is not None:
            self._stdin.close()
            self._stdin = None


@dataclass
class _Child:
    """Runtime state of one spawned subagent process."""

    node: AgentNode
    proc: "subprocess.Popen[str]"
    timeout: int = DEFAULT_TIMEOUT
    fifo_path: Path | None = None
    done: threading.Event = field(default_factory=threading.Event)
    reply: str = ""  # last assistant content with no tool calls
    last_content: str = ""  # last assistant content of any kind (fallback)
    stderr_tail: deque[str] = field(default_factory=_stderr_tail)
    timed_out: bool = False
    killed: bool = False
    completion_logged: bool = False
    timer: threading.Timer | None = None


class AgentRunner:
    """Spawns and tracks the subagent processes of one qi run."""

    def __init__(
        self,
        base_dir: Path | None = None,
        child_argv: Callable[[str, list[str]], list[str]] | None = None,
    ) -> None:
        self.graph = AgentGraph()
        self.on_event: EventSink | None = None
        self._children: dict[str, _Child] = {}
        self._base_dir = base_dir or Path(".").resolve() / ".qi" / "agents"
        self._run_dir: Path | None = None
        self._child_argv = child_argv or _default_child_argv
        self._shut_down = False

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    def spawn(
        self,
        prompt: str,
        name: str | None = None,
        files: list[str] | None = None,
        reads_from: list[str] | None = None,
        background: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        self._flush_events()
        files = files or []
        reads_from = list(reads_from or [])
        timeout = max(1, min(timeout, MAX_TIMEOUT))

        if _current_depth() >= MAX_DEPTH:
            return f"ERROR: subagent depth limit ({MAX_DEPTH}) reached; do this task yourself."
        if len(self.graph.nodes) >= MAX_AGENTS:
            return f"ERROR: agent limit ({MAX_AGENTS}) reached for this run."

        name = _sanitize_name(name) or f"agent{len(self.graph.nodes) + 1}"
        node = AgentNode(name=name, prompt=prompt, reads_from=tuple(reads_from), background=background)
        try:
            self.graph.add(node)
        except ValueError as e:
            known = ", ".join(self.graph.nodes) or "(none)"
            return f"ERROR: {e}. Existing agents: {known}"
        self._emit({
            EVENT_KEY: EVENT_SPAWN,
            NAME_KEY: name,
            PROMPT_KEY: prompt[:200],
            READS_FROM_KEY: reads_from,
            BACKGROUND_KEY: background,
        })

        wrapped = SUBAGENT_PROMPT.format(name=name, prompt=prompt)
        if reads_from:
            upstream_list = ", ".join(f"'{u}'" for u in reads_from)
            wrapped += f"\n\nOutput from upstream agent(s) {upstream_list} arrives as the next message."

        try:
            self._start_child(node, wrapped, files, reads_from, timeout)
        except OSError as e:
            node.status = STATUS_FAILED
            node.detail = str(e)
            self._emit({EVENT_KEY: STATUS_FAILED, NAME_KEY: name, DETAIL_KEY: node.detail})
            return f"ERROR: failed to spawn agent '{name}': {e}"

        if background:
            wiring = f" (reads from {', '.join(reads_from)})" if reads_from else ""
            return f"Spawned background agent '{name}'{wiring}. Collect its reply with AgentWait."
        return self.wait([name])

    def wait(self, names: list[str] | None = None) -> str:
        targets = list(names) if names else list(self._children)
        unknown = [n for n in targets if n not in self._children]
        if unknown:
            known = ", ".join(self._children) or "(none)"
            return f"ERROR: unknown agent(s) {', '.join(unknown)}. Known agents: {known}"
        if not targets:
            return "No agents have been spawned."

        parts: list[str] = []
        for target in targets:
            child = self._children[target]
            if not child.done.wait(child.timeout + 30):
                with contextlib.suppress(OSError):
                    child.proc.kill()
                child.done.wait(10)
            parts.append(self._result_text(child))
        self._flush_events()
        return "\n\n".join(parts)

    def shutdown(self) -> None:
        """Kill stragglers and settle final statuses; idempotent."""
        if self._shut_down:
            return
        self._shut_down = True
        for child in self._children.values():
            if child.timer is not None:
                child.timer.cancel()
            if not child.done.is_set():
                child.killed = True
                with contextlib.suppress(OSError):
                    child.proc.kill()
        for child in self._children.values():
            child.done.wait(5)
        self._flush_events()
        if self._run_dir is not None:
            with contextlib.suppress(OSError):
                self._run_dir.rmdir()

    def _start_child(
        self,
        node: AgentNode,
        wrapped_prompt: str,
        files: list[str],
        reads_from: list[str],
        timeout: int,
    ) -> None:
        argv = self._child_argv(wrapped_prompt, files)
        env = os.environ.copy()
        env[DEPTH_ENV] = str(_current_depth() + 1)

        writer: _StdinWriter
        if hasattr(os, "mkfifo"):
            fifo_path = self._ensure_run_dir() / f"{node.name}.in"
            os.mkfifo(fifo_path)
            # Rendezvous without blocking: the non-blocking read end lets the
            # write end open immediately; the child then inherits a normal
            # blocking stdin.
            read_fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            write_fd = os.open(fifo_path, os.O_WRONLY)
            os.set_blocking(read_fd, True)
            try:
                proc = subprocess.Popen(
                    argv, stdin=read_fd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                )
            except OSError:
                os.close(write_fd)
                with contextlib.suppress(OSError):
                    fifo_path.unlink()
                raise
            finally:
                os.close(read_fd)
            writer = _FdWriter(write_fd)
        else:
            fifo_path = None
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            )
            writer = _PipeWriter(proc)

        child = _Child(node=node, proc=proc, timeout=timeout, fifo_path=fifo_path)
        self._children[node.name] = child
        child.timer = threading.Timer(timeout, self._kill_on_timeout, args=(child,))
        child.timer.daemon = True
        child.timer.start()
        for target, args in (
            (self._pump_stdin, (child, reads_from, writer)),
            (self._read_stdout, (child,)),
            (self._read_stderr, (child,)),
        ):
            thread = threading.Thread(target=target, args=args, daemon=True)
            thread.start()

    def _ensure_run_dir(self) -> Path:
        if self._run_dir is None:
            run_id = f"{datetime.datetime.now():%Y%m%dT%H%M%S}-{os.getpid()}"
            self._run_dir = self._base_dir / run_id
            self._run_dir.mkdir(parents=True, exist_ok=True)
        return self._run_dir

    def _kill_on_timeout(self, child: _Child) -> None:
        if child.done.is_set():
            return
        child.timed_out = True
        with contextlib.suppress(OSError):
            child.proc.kill()

    def _pump_stdin(self, child: _Child, reads_from: list[str], writer: _StdinWriter) -> None:
        """Deliver upstream replies as one user message, then EOF."""
        try:
            if reads_from:
                sections: list[str] = []
                for upstream_name in reads_from:
                    upstream = self._children.get(upstream_name)
                    if upstream is None:
                        sections.append(f'<AGENT_OUTPUT name="{upstream_name}">\n[agent failed to spawn]\n</AGENT_OUTPUT>')
                        continue
                    upstream.done.wait()
                    body = upstream.reply or upstream.last_content
                    if upstream.node.status != STATUS_DONE:
                        note = f"[agent ended with status '{upstream.node.status}'"
                        note += f": {upstream.node.detail}]" if upstream.node.detail else "]"
                        body = f"{note}\n{body}" if body else note
                    sections.append(f'<AGENT_OUTPUT name="{upstream_name}">\n{body}\n</AGENT_OUTPUT>')
                message = {LogKey.ROLE.value: Role.USER.value, LogKey.CONTENT.value: "\n\n".join(sections)}
                writer.write_line(json.dumps(message) + "\n")
        except OSError as e:
            # Broken pipe: the child exited before reading; its status tells the story.
            logger.info(f"Could not deliver input to agent '{child.node.name}': {e}")
        finally:
            with contextlib.suppress(OSError):
                writer.close()

    def _read_stdout(self, child: _Child) -> None:
        """Follow the child's JSONL event stream and keep its final reply."""
        assert child.proc.stdout is not None
        for line in child.proc.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get(LogKey.TYPE) != RecordType.MESSAGE:
                continue
            if record.get(LogKey.ROLE) != Role.ASSISTANT:
                continue
            content = record.get(LogKey.CONTENT)
            if not isinstance(content, str) or not content:
                continue
            child.last_content = content[:MAX_REPLY_CHARS]
            if not record.get(LogKey.TOOL_CALLS):
                child.reply = child.last_content
        child.proc.stdout.close()
        self._finish_child(child, child.proc.wait())

    def _read_stderr(self, child: _Child) -> None:
        assert child.proc.stderr is not None
        for line in child.proc.stderr:
            stripped = line.rstrip("\n")
            if stripped:
                child.stderr_tail.append(stripped)
        child.proc.stderr.close()

    def _finish_child(self, child: _Child, exit_code: int) -> None:
        if child.timer is not None:
            child.timer.cancel()
        if not child.reply:
            child.reply = child.last_content
        node = child.node
        if child.killed:
            node.status = STATUS_KILLED
            node.detail = "killed at end of run"
        elif child.timed_out:
            node.status = STATUS_FAILED
            node.detail = f"timed out after {child.timeout}s"
        elif exit_code == 0:
            node.status = STATUS_DONE
        else:
            node.status = STATUS_FAILED
            node.detail = f"exit {exit_code}"
        if child.fifo_path is not None:
            with contextlib.suppress(OSError):
                child.fifo_path.unlink()
        child.done.set()

    def _result_text(self, child: _Child) -> str:
        node = child.node
        status = node.status if child.done.is_set() else "unresponsive"
        header = f"=== Agent '{node.name}': {status}" + (f" ({node.detail})" if node.detail else "") + " ==="
        body = child.reply or "(no reply)"
        if status != STATUS_DONE and child.stderr_tail:
            body += "\n--- stderr tail ---\n" + "\n".join(child.stderr_tail)
        return f"{header}\n{body}"

    def _flush_events(self) -> None:
        """Emit completion events for settled children.

        Called only from the spawning thread so the session log is written
        from one thread; reader threads just flip in-memory state.
        """
        for child in self._children.values():
            if child.done.is_set() and not child.completion_logged:
                child.completion_logged = True
                event: dict[str, Any] = {EVENT_KEY: child.node.status, NAME_KEY: child.node.name}
                if child.node.detail:
                    event[DETAIL_KEY] = child.node.detail
                self._emit(event)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            logger.exception("Agent event sink failed")


_runner: AgentRunner | None = None
_event_sink: EventSink | None = None


def set_event_sink(sink: EventSink | None) -> None:
    """Route agent events (spawn/done/failed) to e.g. the session log."""
    global _event_sink
    _event_sink = sink
    if _runner is not None:
        _runner.on_event = sink


def get_runner() -> AgentRunner:
    global _runner
    if _runner is None:
        _runner = AgentRunner()
        _runner.on_event = _event_sink
        atexit.register(_runner.shutdown)
    return _runner


def peek_runner() -> AgentRunner | None:
    return _runner


def reset_runner() -> None:
    global _runner
    if _runner is not None:
        _runner.shutdown()
        atexit.unregister(_runner.shutdown)
    _runner = None
