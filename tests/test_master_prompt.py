"""Tests for the master prompt."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from qi.lib.context import get_system_prompt
from qi.lib.llm_client._types import LLMResponse
from qi.prompts.master import SYSTEM_PROMPT


def test_returns_base_prompt_when_no_agents_md(tmp_path: Path) -> None:
    result = get_system_prompt(cwd=tmp_path)
    assert result == SYSTEM_PROMPT


def test_returns_base_prompt_when_agents_md_empty(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("")
    result = get_system_prompt(cwd=tmp_path)
    assert result == SYSTEM_PROMPT


def test_returns_base_prompt_when_agents_md_blank(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("   \n\n  ")
    result = get_system_prompt(cwd=tmp_path)
    assert result == SYSTEM_PROMPT


def test_appends_agents_md_content(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Always use tabs for indentation.")
    result = get_system_prompt(cwd=tmp_path)

    assert result.startswith(SYSTEM_PROMPT)
    assert "<PROJECT_INSTRUCTIONS>" in result
    assert "Always use tabs for indentation." in result
    assert "</PROJECT_INSTRUCTIONS>" in result


def test_appends_wrapped_in_tags(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Use type hints everywhere.")
    result = get_system_prompt(cwd=tmp_path)

    expected_suffix = "<PROJECT_INSTRUCTIONS>\nUse type hints everywhere.\n</PROJECT_INSTRUCTIONS>"
    assert result.endswith(expected_suffix)


def test_defaults_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("pin test deps")
    result = get_system_prompt()

    assert "pin test deps" in result
    assert "<PROJECT_INSTRUCTIONS>" in result


def _write_skill(tmp_path: Path, name: str = "greet", description: str = "Say hello nicely") -> None:
    skill_dir = tmp_path / ".qi" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nbody\n")


def test_no_skills_block_when_no_skills(tmp_path: Path) -> None:
    result = get_system_prompt(cwd=tmp_path)
    assert "<AVAILABLE_SKILLS>" not in result


def test_lists_skills(tmp_path: Path) -> None:
    _write_skill(tmp_path)
    result = get_system_prompt(cwd=tmp_path)

    assert result.startswith(SYSTEM_PROMPT)
    assert "<AVAILABLE_SKILLS>" in result
    assert "- greet: Say hello nicely" in result
    assert "</AVAILABLE_SKILLS>" in result


def test_skills_coexist_with_project_instructions(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Use tabs.")
    _write_skill(tmp_path)
    result = get_system_prompt(cwd=tmp_path)

    assert "<PROJECT_INSTRUCTIONS>" in result
    assert "<AVAILABLE_SKILLS>" in result
    assert result.index("<PROJECT_INSTRUCTIONS>") < result.index("<AVAILABLE_SKILLS>")


def test_appended_via_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Always write tests first.")

    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(content="{}")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("builtins.open", mock_open(read_data="x")),
    ):
        from qi.commands.run import run

        mock_load.return_value = Mock(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["f.py"])

    assert rc == 0
    messages = mock_client.chat.call_args[0][0]
    assert "<PROJECT_INSTRUCTIONS>" in messages[0]["content"]
    assert "Always write tests first." in messages[0]["content"]
    assert "</PROJECT_INSTRUCTIONS>" in messages[0]["content"]
