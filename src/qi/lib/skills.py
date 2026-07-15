"""Skill discovery and loading.

A skill is a directory containing a SKILL.md file with YAML frontmatter
(name, description) followed by a markdown body of instructions. Skills are
discovered from the user config dir (~/.config/qi/skills) and the project
(.qi/skills); project skills override user skills on name collision.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from qi.lib.config import _user_config_path

logger = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"


@dataclass
class Skill:
    name: str
    description: str
    path: Path  # the skill directory


def user_skills_dir() -> Path:
    return _user_config_path().parent / "skills"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split text into (frontmatter dict, body).

    Frontmatter is a leading YAML block delimited by `---` lines. Returns
    ({}, text) when no valid block is present, or when it doesn't parse to a
    flat mapping.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            frontmatter = "\n".join(lines[1:i]) + "\n"
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            try:
                parsed = yaml.safe_load(frontmatter)
            except yaml.YAMLError:
                return {}, text
            if not isinstance(parsed, dict):
                return {}, text
            meta = {str(k): str(v) for k, v in parsed.items()}
            return meta, body

    return {}, text  # unterminated frontmatter


def _load_skill(skill_dir: Path) -> Skill | None:
    skill_file = skill_dir / SKILL_FILE
    try:
        text = skill_file.read_text()
    except OSError as e:
        logger.warning(f"Skipping skill {skill_dir}: cannot read {SKILL_FILE}: {e}")
        return None

    meta, _ = _parse_frontmatter(text)
    if not meta:
        logger.warning(f"Skipping skill {skill_dir}: no frontmatter in {SKILL_FILE}")
        return None
    description = meta.get("description", "").strip()
    if not description:
        logger.warning(f"Skipping skill {skill_dir}: missing description in {SKILL_FILE}")
        return None

    return Skill(
        name=meta.get("name") or skill_dir.name,
        description=description,
        path=skill_dir,
    )


def discover_skills(cwd: Path | None = None, user_dir: Path | None = None) -> dict[str, Skill]:
    """Discover skills from the user dir then the project; project wins collisions."""
    if cwd is None:
        cwd = Path.cwd()
    dirs = [user_dir or user_skills_dir(), cwd / ".qi" / "skills"]

    skills: dict[str, Skill] = {}
    for base in dirs:
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or not (entry / SKILL_FILE).is_file():
                continue
            skill = _load_skill(entry)
            if skill is not None:
                skills[skill.name] = skill
    return skills


def load_skill_body(skill: Skill) -> str:
    """Return the SKILL.md body without its frontmatter."""
    text = (skill.path / SKILL_FILE).read_text()
    _, body = _parse_frontmatter(text)
    return body.strip()
