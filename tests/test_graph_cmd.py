"""Tests for the `qi graph` subcommand."""

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from qi.commands.graph import build_graph, run


def _write_session(session_dir: Path, session_id: str, records: list[dict[str, Any]]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / f"{session_id}.jsonl"
    start = {"type": "session_start", "meta": {"model": "stub"}}
    file_path.write_text("\n".join(json.dumps(r) for r in [start, *records]) + "\n")
    return file_path


AGENT_RECORDS: list[dict[str, Any]] = [
    {"type": "agent", "meta": {"event": "spawn", "name": "a", "prompt": "one", "reads_from": [], "background": False}},
    {"type": "agent", "meta": {"event": "done", "name": "a"}},
    {"type": "agent", "meta": {"event": "spawn", "name": "b", "prompt": "two", "reads_from": ["a"], "background": True}},
    {"type": "agent", "meta": {"event": "failed", "name": "b", "detail": "exit 1"}},
]


def test_build_graph_from_session(tmp_path: Path) -> None:
    file_path = _write_session(tmp_path, "s1", AGENT_RECORDS)
    graph = build_graph(file_path)
    assert list(graph.nodes) == ["a", "b"]
    assert graph.nodes["a"].status == "done"
    assert graph.nodes["b"].status == "failed"
    assert graph.nodes["b"].detail == "exit 1"
    assert graph.nodes["b"].reads_from == ("a",)


def test_build_graph_skips_junk_lines(tmp_path: Path) -> None:
    file_path = tmp_path / "s.jsonl"
    file_path.write_text('not json\n{"type": "message", "role": "user", "content": "hi"}\n[1, 2]\n')
    assert not build_graph(file_path).nodes


def test_graph_by_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_session(tmp_path / ".qi" / "sessions", "mysession", AGENT_RECORDS)
    rc = run(["mysession"])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert "Agent graph for session 'mysession':" in out
    assert "qi ──▶ a  [done]" in out
    assert "a ──▶ b  [failed: exit 1]" in out


def test_graph_defaults_to_latest_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / ".qi" / "sessions"
    old = _write_session(session_dir, "older", [])
    _write_session(session_dir, "newer", AGENT_RECORDS)
    past = time.time() - 60
    os.utime(old, (past, past))
    rc = run([])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert "newer" in out
    assert "qi ──▶ a  [done]" in out


def test_graph_mermaid_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_session(tmp_path / ".qi" / "sessions", "s", AGENT_RECORDS)
    rc = run(["s", "--format", "mermaid"])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert out.startswith("flowchart LR")
    assert "qi --> a" in out
    assert "a --> b" in out


def test_graph_dot_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_session(tmp_path / ".qi" / "sessions", "s", AGENT_RECORDS)
    rc = run(["s", "--format", "dot"])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert out.startswith("digraph agents {")
    assert '"a" -> "b";' in out


def test_graph_session_without_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_session(tmp_path / ".qi" / "sessions", "quiet", [])
    rc = run(["quiet"])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert "No subagent activity" in out


def test_graph_unknown_session_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".qi" / "sessions").mkdir(parents=True)
    assert run(["nope"]) == 1


def test_graph_no_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert run([]) == 1


def test_graph_rejects_path_separators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert run(["../evil"]) == 1
