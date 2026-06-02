"""CLI entry point for Qi."""

import argparse
import importlib
import logging
import sys

from qi import __version__
from qi.lib.logging import QiLogHandler

SUBCOMMANDS: dict[str, str] = {
    "run": "qi.commands.run",
    "ping": "qi.commands.ping",
    "init": "qi.commands.init",
}

HELP = """Usage: qi [<options>] <file>
       qi <command> [<args>]

Commands:
  init   Scaffold a .qi project directory
  run    Process a prompt or analyse given files (default)
  ping   Ping the server

Global options:
  --version   Display the uv version

  Run 'qi <command> --help' for more information on a command.
"""

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, str, list[str]]:
    args = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="qi", add_help=False)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("subcommand", nargs="?")
    parsed, remaining = parser.parse_known_args(args)

    if parsed.version or parsed.help:
        return parsed, "", remaining

    subcommand = parsed.subcommand
    if subcommand:
        if subcommand in SUBCOMMANDS:
            return parsed, subcommand, remaining
        return parsed, "run", args

    if not args:
        return parsed, "", remaining
    return parsed, "run", args


def main(argv: list[str] | None = None) -> int:
    args, subcommand, remaining = parse_args(argv)

    if not subcommand:
        if args.version:
            print(f"qi {__version__}")
            return 0
        else:
            print(HELP)
            return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s.:%(lineno)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[QiLogHandler()],
    )

    if subcommand in SUBCOMMANDS:
        mod = importlib.import_module(SUBCOMMANDS[subcommand])
        return mod.run(remaining)  # type: ignore[no-any-return]

    from qi.commands.run import run as run_cmd

    return run_cmd(remaining)


if __name__ == "__main__":
    sys.exit(main())
