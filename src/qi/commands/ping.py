"""Ping subcommand."""

from __future__ import annotations

import argparse


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qi ping",
        description="Ping the server",
    )
    parser.parse_args(argv)
    print("pong")
    return 0
