"""CLI entry point for Qi."""

from __future__ import annotations

import importlib
import sys

from qi import __version__

SUBCOMMANDS: dict[str, str] = {
    "run": "qi.commands.run",
    "ping": "qi.commands.ping",
}

HELP = """Usage: qi [<options>] <file>
       qi <command> [<args>]

Commands:
  run    Process a file (default)
  ping   Ping the server

Run 'qi <command> --help' for more information on a command.
"""


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]

    if not args:
        from qi.commands.run import run as run_cmd

        return run_cmd(["--help"])

    if args[0] in ("--help", "-h"):
        print(HELP)
        return 0

    if args[0] == "--version":
        print(f"qi {__version__}")
        return 0

    if args[0] in SUBCOMMANDS:
        mod = importlib.import_module(SUBCOMMANDS[args[0]])
        return mod.run(args[1:])  # type: ignore[no-any-return]

    from qi.commands.run import run as run_cmd

    return run_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
