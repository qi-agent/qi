"""Run subcommand: process a file."""

from __future__ import annotations

import argparse


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qi",
        description="Process a file",
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Prompt text",
    )
    parser.add_argument(
        "file",
        help="File to process",
    )
    parsed = parser.parse_args(argv)
    print(f"Running with file={parsed.file}, prompt={parsed.prompt}")
    return 0
