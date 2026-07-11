"""Tests for skill discovery and loading."""

from pathlib import Path

import pytest

from qi.lib.skills import (
    Skill,
    _parse_frontmatter,
    discover_skills,
    load_skill_body,
    user_skills_dir,
)

SKILL_MD = """\
---
name: greet
description: Say hello nicely
---
# Greeting

Always greet the user warmly.
"""


def _write_skill(base: Path, dirname: str, content: str = SKILL_MD) -> Path:
    skill_dir = base / dirname
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


class TestParseFrontmatter:
    def test_valid(self) -> None:
        meta, body = _parse_frontmatter(SKILL_MD)
        assert meta == {"name": "greet", "description": "Say hello nicely"}
        assert body.startswith("# Greeting")
        assert "---" not in body

    def test_quoted_values(self) -> None:
        meta, _ = _parse_frontmatter('---\nname: "greet"\ndescription: \'hi: there\'\n---\nbody')
        assert meta == {"name": "greet", "description": "hi: there"}

    def test_no_frontmatter(self) -> None:
        meta, body = _parse_frontmatter("# Just markdown\n")
        assert meta == {}
        assert body == "# Just markdown\n"

    def test_unterminated_frontmatter(self) -> None:
        text = "---\nname: greet\nno closing delimiter"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_ignores_comments_and_blank_lines(self) -> None:
        meta, _ = _parse_frontmatter("---\n# a comment\n\nname: x\ndescription: y\n---\nbody")
        assert meta == {"name": "x", "description": "y"}


class TestDiscoverSkills:
    def test_finds_project_skill(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / ".qi" / "skills", "greet")
        skills = discover_skills(cwd=tmp_path, user_dir=tmp_path / "nouser")
        assert list(skills) == ["greet"]
        assert skills["greet"].description == "Say hello nicely"

    def test_finds_user_skill(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user-skills"
        _write_skill(user_dir, "greet")
        skills = discover_skills(cwd=tmp_path, user_dir=user_dir)
        assert list(skills) == ["greet"]

    def test_project_overrides_user(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user-skills"
        _write_skill(user_dir, "greet", SKILL_MD.replace("nicely", "from user"))
        _write_skill(tmp_path / ".qi" / "skills", "greet")
        skills = discover_skills(cwd=tmp_path, user_dir=user_dir)
        assert skills["greet"].description == "Say hello nicely"

    def test_name_falls_back_to_dir_name(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / ".qi" / "skills", "mydir", "---\ndescription: d\n---\nbody")
        skills = discover_skills(cwd=tmp_path, user_dir=tmp_path / "nouser")
        assert list(skills) == ["mydir"]

    def test_skips_missing_description(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / ".qi" / "skills", "bad", "---\nname: bad\n---\nbody")
        skills = discover_skills(cwd=tmp_path, user_dir=tmp_path / "nouser")
        assert skills == {}

    def test_skips_missing_frontmatter(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / ".qi" / "skills", "bad", "just markdown")
        skills = discover_skills(cwd=tmp_path, user_dir=tmp_path / "nouser")
        assert skills == {}

    def test_ignores_stray_files_and_empty_dirs(self, tmp_path: Path) -> None:
        skills_root = tmp_path / ".qi" / "skills"
        skills_root.mkdir(parents=True)
        (skills_root / "README.md").write_text("not a skill")
        (skills_root / "empty").mkdir()
        skills = discover_skills(cwd=tmp_path, user_dir=tmp_path / "nouser")
        assert skills == {}

    def test_nonexistent_dirs(self, tmp_path: Path) -> None:
        skills = discover_skills(cwd=tmp_path, user_dir=tmp_path / "nouser")
        assert skills == {}


def test_load_skill_body(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "greet")
    body = load_skill_body(Skill(name="greet", description="d", path=skill_dir))
    assert body == "# Greeting\n\nAlways greet the user warmly."


def test_user_skills_dir_respects_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert user_skills_dir() == tmp_path / "qi" / "skills"
