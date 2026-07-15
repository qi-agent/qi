"""Tests for native tools (Bash, ReadFile, Skill)."""

from pathlib import Path

import pytest

from qi.tools.bash import BashTool
from qi.tools.read_file import ReadFileTool
from qi.tools.skill import SkillTool


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


class TestSkillTool:
    tool = SkillTool()

    def _write_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / ".qi" / "skills" / "greet"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: greet\ndescription: Say hello\n---\nAlways greet warmly.\n"
        )

    def test_loads_skill_body(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._write_skill(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = self.tool("greet")
        assert result.startswith("Skill: greet\n")
        assert f"Directory: {tmp_path / '.qi' / 'skills' / 'greet'}" in result
        assert "Always greet warmly." in result

    def test_unknown_skill(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._write_skill(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = self.tool("nope")
        assert result.startswith("ERROR: Unknown skill 'nope'")
        assert "greet" in result

    def test_registered(self) -> None:
        from qi.tools import TOOL_MAP, TOOL_SCHEMAS

        assert "Skill" in TOOL_MAP
        assert any(s["function"]["name"] == "Skill" for s in TOOL_SCHEMAS)

    def test_schema_structure(self) -> None:
        schema = self.tool.schema
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "Skill"
        assert "name" in func["parameters"]["required"]
