"""Context Management functions"""

from pathlib import Path

from qi.prompts.master import SYSTEM_PROMPT


def get_system_prompt(cwd: Path | None = None) -> str:
    if cwd is None:
        cwd = Path.cwd()
    agents_path = cwd / "AGENTS.md"
    if agents_path.is_file():
        content = agents_path.read_text().strip()
        if content:
            return SYSTEM_PROMPT + "\n\n<PROJECT_INSTRUCTIONS>\n" + content + "\n</PROJECT_INSTRUCTIONS>"
    return SYSTEM_PROMPT
