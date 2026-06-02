"""Init subcommand — scaffold a .qi project directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.text import Text

from qi.lib.logging import output

DEFAULT_CONFIG = """\
# Qi configuration
# Edit this file to configure your LLM provider.
# See https://github.com/qi-agent/qi for documentation.

api_key = ""
model = "gemma4:26b-mlx"
base_url = "http://localhost:11434/v1"  # Ollama
# base_url = "https://generativelanguage.googleapis.com"  # Google
# base_url = "https://api.openai.com/v1"  # OpenAI

"""


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qi init",
        description="Scaffold a .qi project directory",
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing config.toml",
    )
    args = parser.parse_args(argv)

    root = Path(args.dir).resolve()
    qi_dir = root / ".qi"
    config_path = qi_dir / "config.toml"
    session_dir = qi_dir / "sessions"

    if config_path.exists() and not args.force:
        output(Text.styled(f"Already initialized ({config_path}). Use --force to overwrite.", "yellow"))
        return 1

    qi_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_CONFIG)

    output(Text.styled(f"Created {qi_dir}/"))
    output(Text.styled(f"Created {config_path}"))
    output(Text.styled(f"Created {session_dir}/"))
    output(Text.styled("Edit .qi/config.toml to set your model and API key.", "bold"))
    return 0
