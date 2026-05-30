"""Tests for native tools (Bash, ReadFile)."""

import json
from pathlib import Path

from qi.tools.bash import BashTool
from qi.tools.read_file import ReadFileTool


class TestBashTool:
    tool = BashTool()

    def test_basic_command(self) -> None:
        result = json.loads(self.tool("echo hello"))
        assert result["exit_code"] == 0
        assert result["stdout"] == "hello\n"
        assert result["stderr"] == ""

    def test_non_zero_exit(self) -> None:
        result = json.loads(self.tool("exit 42"))
        assert result["exit_code"] == 42

    def test_stderr_capture(self) -> None:
        result = json.loads(self.tool("echo out; echo err >&2"))
        assert result["exit_code"] == 0
        assert result["stdout"] == "out\n"
        assert result["stderr"] == "err\n"

    def test_workdir(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker.txt"
        marker.write_text("here")
        result = json.loads(self.tool("ls", workdir=str(tmp_path)))
        assert result["exit_code"] == 0
        assert "marker.txt" in result["stdout"]

    def test_timeout(self) -> None:
        result = json.loads(self.tool("sleep 10", timeout=1))
        assert result["exit_code"] == -1
        assert "timed out" in result["stderr"].lower()

    def test_multi_line_script(self) -> None:
        result = json.loads(self.tool("for i in a b c; do echo line $i; done"))
        assert result["exit_code"] == 0
        assert result["stdout"] == "line a\nline b\nline c\n"

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
