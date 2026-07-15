"""Context Management functions"""

from pathlib import Path

from qi.lib.skills import discover_skills
from qi.prompts.master import SYSTEM_PROMPT


def _skills_block(cwd: Path) -> str:
    skills = discover_skills(cwd)
    if not skills:
        return ""
    listing = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
    return (
        "<AVAILABLE_SKILLS>\n"
        "Skills are named instruction sets. When one is relevant to the task, "
        "load its full instructions with the Skill tool before proceeding.\n"
        + listing +
        "\n</AVAILABLE_SKILLS>"
    )


def get_system_prompt(cwd: Path | None = None) -> str:
    if cwd is None:
        cwd = Path.cwd()
    prompt = SYSTEM_PROMPT
    agents_path = cwd / "AGENTS.md"
    if agents_path.is_file():
        content = agents_path.read_text().strip()
        if content:
            prompt += "\n\n<PROJECT_INSTRUCTIONS>\n" + content + "\n</PROJECT_INSTRUCTIONS>"
    skills_block = _skills_block(cwd)
    if skills_block:
        prompt += "\n\n" + skills_block
    return prompt
