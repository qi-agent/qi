"""Tests for native tools (Bash, ReadFile)."""

import json
from pathlib import Path

from qi.tools.bash import BashTool
from qi.tools.read_file import ReadFileTool


class TestBashTool:
    tool = BashTool()

    def test_basic_command(self) -> None:
        result = self.tool("echo hello")
        assert "Exit code: 0" in result
        assert "<stdout>\nhello<stdout>" in result
        assert "<stderr>" not in result

    def test_non_zero_exit(self) -> None:
        result = self.tool("exit 42")
        assert "Exit code: 42" in result

    def test_stderr_capture(self) -> None:
        result = self.tool("echo out; echo err >&2")
        assert "Exit code: 0" in result
        assert "<stdout>\nout<stdout>" in result
        assert "<stderr>\nerr\n<stderr>" in result

    def test_workdir(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker.txt"
        marker.write_text("here")
        result = self.tool("ls", workdir=str(tmp_path))
        assert "Exit code: 0" in result
        assert "marker.txt" in result

    def test_timeout(self) -> None:
        result = self.tool("sleep 10", timeout=1)
        assert "Exit code: -1" in result
        assert "timed out" in result.lower()

    def test_multi_line_script(self) -> None:
        result = self.tool("for i in a b c; do echo line $i; done")
        assert "Exit code: 0" in result
        assert "<stdout>\nline a\nline b\nline c<stdout>" in result

    def test_schema_structure(self) -> None:
        schema = self.tool.schema
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "Bash"
        assert "command" in func["parameters"]["required"]


class TestReadFileTool:
    tool = ReadFileTool()

    def test_read_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello\nworld\n")
        result = self.tool(str(f))
        assert result == "hello\nworld\n"

    def test_read_with_range(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = self.tool(str(f), start=1, end=2)
        assert result == "line2\n"
