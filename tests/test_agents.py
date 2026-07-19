"""Tests for the multi-agent runtime (graph, runner, tools)."""

import json
import os
import stat
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from qi.lib.agents import (
    DEPTH_ENV,
    EVENT_KEY,
    EVENT_SPAWN,
    MAX_DEPTH,
    NAME_KEY,
    STATUS_DONE,
    STATUS_FAILED,
    AgentGraph,
    AgentNode,
    AgentRunner,
    _default_child_argv,
)
from qi.tools import TOOL_MAP
from qi.tools.agent import AgentTool, AgentWaitTool

# A stand-in for a child `qi run` process: reads the piped-mode stdin, honours
# @sleep/@exit/@err directives embedded in its prompt, and emits one assistant
# record on the JSONL event stream.
STUB_CHILD = """\
import json, os, re, sys, time

prompt = sys.argv[1] if len(sys.argv) > 1 else ""
directives = dict(re.findall(r"@(\\w+)=(\\S+)", prompt))
data = sys.stdin.read().strip()
time.sleep(float(directives.get("sleep", 0)))
if "err" in directives:
    print(directives["err"], file=sys.stderr)
reply = "|".join([
    "reply",
    "task:" + prompt,
    "stdin:" + data,
    "depth:" + os.environ.get("QI_AGENT_DEPTH", ""),
])
print(json.dumps({"type": "message", "role": "assistant", "content": reply}))
sys.exit(int(directives.get("exit", 0)))
"""


@pytest.fixture
def runner(tmp_path: Path) -> AgentRunner:
    stub = tmp_path / "stub_child.py"
    stub.write_text(STUB_CHILD)
    return AgentRunner(
        base_dir=tmp_path / "agents",
        child_argv=lambda prompt, files: [sys.executable, str(stub), prompt, *files],
    )


class TestAgentGraph:
    def test_add_and_order(self) -> None:
        graph = AgentGraph()
        graph.add(AgentNode(name="a"))
        graph.add(AgentNode(name="b", reads_from=("a",)))
        assert list(graph.nodes) == ["a", "b"]

    def test_duplicate_name_rejected(self) -> None:
        graph = AgentGraph()
        graph.add(AgentNode(name="a"))
        with pytest.raises(ValueError, match="already exists"):
            graph.add(AgentNode(name="a"))

    def test_unknown_upstream_rejected(self) -> None:
        graph = AgentGraph()
        with pytest.raises(ValueError, match="unknown upstream"):
            graph.add(AgentNode(name="b", reads_from=("nope",)))

    def test_root_name_reserved(self) -> None:
        graph = AgentGraph()
        with pytest.raises(ValueError, match="root"):
            graph.add(AgentNode(name="qi"))

    def test_render_ascii(self) -> None:
        graph = AgentGraph()
        graph.add(AgentNode(name="a", status=STATUS_DONE))
        graph.add(AgentNode(name="b", status=STATUS_FAILED, detail="exit 1"))
        graph.add(AgentNode(name="c", reads_from=("a", "b")))
        out = graph.render_ascii()
        assert "qi ──▶ a  [done]" in out
        assert "qi ──▶ b  [failed: exit 1]" in out
        assert "a, b ──▶ c  [running]" in out

    def test_render_ascii_empty(self) -> None:
        assert AgentGraph().render_ascii() == "(no agents)"

    def test_render_mermaid(self) -> None:
        graph = AgentGraph()
        graph.add(AgentNode(name="a", status=STATUS_DONE))
        graph.add(AgentNode(name="b", reads_from=("a",)))
        out = graph.render_mermaid()
        assert out.startswith("flowchart LR")
        assert "qi --> a" in out
        assert "a --> b" in out
        assert '(done)' in out

    def test_render_dot(self) -> None:
        graph = AgentGraph()
        graph.add(AgentNode(name="a"))
        graph.add(AgentNode(name="b", reads_from=("a",)))
        out = graph.render_dot()
        assert out.startswith("digraph agents {")
        assert '"qi" -> "a";' in out
        assert '"a" -> "b";' in out


class TestAgentRunner:
    def test_foreground_spawn_returns_reply(self, runner: AgentRunner) -> None:
        result = runner.spawn("say hello", name="greeter")
        assert "=== Agent 'greeter': done ===" in result
        assert "task:" in result and "say hello" in result
        assert runner.graph.nodes["greeter"].status == STATUS_DONE

    def test_child_depth_is_incremented(self, runner: AgentRunner) -> None:
        result = runner.spawn("check depth")
        assert "depth:1" in result

    def test_depth_limit_blocks_spawn(self, runner: AgentRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DEPTH_ENV, str(MAX_DEPTH))
        result = runner.spawn("too deep")
        assert result.startswith("ERROR")
        assert not runner.graph.nodes

    def test_unknown_upstream_is_error(self, runner: AgentRunner) -> None:
        result = runner.spawn("b task", name="b", reads_from=["ghost"])
        assert result.startswith("ERROR")
        assert "ghost" in result

    def test_duplicate_name_is_error(self, runner: AgentRunner) -> None:
        runner.spawn("first", name="dup")
        result = runner.spawn("second", name="dup")
        assert result.startswith("ERROR")
        assert "dup" in result

    def test_name_is_sanitized(self, runner: AgentRunner) -> None:
        result = runner.spawn("task", name="My Agent!")
        assert "=== Agent 'My-Agent'" in result

    def test_fan_in_pipes_upstream_replies(self, runner: AgentRunner) -> None:
        runner.spawn("produce alpha", name="a")
        runner.spawn("produce beta", name="b")
        result = runner.spawn("combine inputs", name="c", reads_from=["a", "b"])
        # c received one piped-protocol user message containing both replies
        assert '\\"role\\": \\"user\\"' in result or '"role": "user"' in result
        assert 'AGENT_OUTPUT name=\\"a\\"' in result or 'AGENT_OUTPUT name="a"' in result
        assert "produce alpha" in result
        assert "produce beta" in result

    def test_background_agents_run_in_parallel(self, runner: AgentRunner) -> None:
        start = time.monotonic()
        assert "Spawned background agent 'p1'" in runner.spawn("@sleep=1.0 one", name="p1", background=True)
        assert "Spawned background agent 'p2'" in runner.spawn("@sleep=1.0 two", name="p2", background=True)
        result = runner.wait()
        elapsed = time.monotonic() - start
        assert "=== Agent 'p1': done ===" in result
        assert "=== Agent 'p2': done ===" in result
        assert elapsed < 1.9, f"agents did not run in parallel ({elapsed:.2f}s)"

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes need POSIX")
    def test_background_agent_stdin_is_named_pipe(self, runner: AgentRunner) -> None:
        runner.spawn("@sleep=1.0 wait around", name="fifo-kid", background=True)
        assert runner.run_dir is not None
        fifo = runner.run_dir / "fifo-kid.in"
        assert fifo.exists()
        assert stat.S_ISFIFO(fifo.stat().st_mode)
        runner.wait(["fifo-kid"])
        assert not fifo.exists()  # cleaned up after completion

    def test_failure_reports_exit_code_and_stderr(self, runner: AgentRunner) -> None:
        result = runner.spawn("@exit=3 @err=boom fail please", name="crash")
        assert "=== Agent 'crash': failed (exit 3) ===" in result
        assert "boom" in result
        assert runner.graph.nodes["crash"].status == STATUS_FAILED

    def test_timeout_kills_child(self, runner: AgentRunner) -> None:
        result = runner.spawn("@sleep=30 never finish", name="slow", timeout=1)
        assert "failed" in result
        assert "timed out after 1s" in result

    def test_wait_unknown_name_is_error(self, runner: AgentRunner) -> None:
        assert runner.wait(["ghost"]).startswith("ERROR")

    def test_wait_without_agents(self, runner: AgentRunner) -> None:
        assert runner.wait() == "No agents have been spawned."

    def test_events_are_emitted(self, runner: AgentRunner) -> None:
        events: list[dict[str, object]] = []
        runner.on_event = events.append
        runner.spawn("emit events", name="ev")
        assert [e[EVENT_KEY] for e in events] == [EVENT_SPAWN, STATUS_DONE]
        assert all(e[NAME_KEY] == "ev" for e in events)

    def test_default_child_argv_is_piped_qi_run(self) -> None:
        argv = _default_child_argv("do it", ["a.py"])
        assert argv[1:4] == ["-m", "qi", "run"]
        assert "--output-format" in argv and "jsonl" in argv
        assert argv[-1] == "a.py"


class TestAgentTools:
    def test_registered_in_tool_map(self) -> None:
        assert "Agent" in TOOL_MAP
        assert "AgentWait" in TOOL_MAP

    def test_agent_schema_structure(self) -> None:
        schema = AgentTool().schema
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "Agent"
        assert "prompt" in func["parameters"]["required"]
        for param in ("name", "files", "reads_from", "background", "timeout"):
            assert param in func["parameters"]["properties"]

    def test_agent_wait_schema_structure(self) -> None:
        schema = AgentWaitTool().schema
        assert schema["function"]["name"] == "AgentWait"

    def test_agent_tool_delegates_to_runner(self) -> None:
        mock_runner = Mock()
        mock_runner.spawn.return_value = "ok"
        with patch("qi.tools.agent.get_runner", return_value=mock_runner):
            result = AgentTool()(prompt="do it", name="x", background=True)
        assert result == "ok"
        mock_runner.spawn.assert_called_once_with(
            prompt="do it", name="x", files=[], reads_from=[], background=True, timeout=300,
        )

    def test_agent_wait_tool_delegates_to_runner(self) -> None:
        mock_runner = Mock()
        mock_runner.wait.return_value = "done"
        with patch("qi.tools.agent.get_runner", return_value=mock_runner):
            assert AgentWaitTool()(names=["a"]) == "done"
            assert AgentWaitTool()() == "done"
        assert mock_runner.wait.call_args_list[0].args == (["a"],)
        assert mock_runner.wait.call_args_list[1].args == (None,)


def test_stub_child_emits_valid_record(tmp_path: Path) -> None:
    """Meta-test: the stub used above speaks the same JSONL protocol as qi run."""
    import subprocess

    stub = tmp_path / "stub.py"
    stub.write_text(STUB_CHILD)
    proc = subprocess.run(
        [sys.executable, str(stub), "hello"], input="", capture_output=True, text=True,
    )
    record = json.loads(proc.stdout)
    assert record["type"] == "message"
    assert record["role"] == "assistant"
