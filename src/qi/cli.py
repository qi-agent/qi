"""CLI entry point for Qi."""

import argparse
import contextlib
import errno
import importlib
import logging
import os
import sys
from datetime import datetime

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

_WINDOWS = os.name == "nt"


def _is_broken_pipe(e: OSError) -> bool:
    """POSIX raises BrokenPipeError (EPIPE) when the reader goes away; Windows
    surfaces the same condition as a plain OSError with EINVAL (or EPIPE)."""
    if isinstance(e, BrokenPipeError):
        return True
    return _WINDOWS and e.errno in (errno.EPIPE, errno.EINVAL)


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
        # No subcommand and no args: if stdin is piped, default to run (piped mode);
        # otherwise show help.
        if not sys.stdin.isatty():
            return parsed, "run", remaining
        return parsed, "", remaining
    return parsed, "run", args


def setup_logging() -> None:
    """Configure logging for the application."""
    # Console handler: only WARNING and above
    console_handler = QiLogHandler()
    console_handler.setLevel(logging.WARNING)
    # The QiLogHandler will set its own formatter in the format method

    # File handler: INFO and above to a file in .qi/logs/YYYYmmddTHHMMSS.log
    log_dir = os.path.join(".qi", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s.%(funcName)s:%(lineno)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]"
    )
    file_handler.setFormatter(file_formatter)

    # Configure root logger with both handlers, forcing any existing handlers to be removed
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    args, subcommand, remaining = parse_args(argv)

    if not subcommand:
        if args.version:
            print(f"qi {__version__}")
            return 0
        else:
            print(HELP)
            return 0

    setup_logging()

    try:
        if subcommand in SUBCOMMANDS:
            mod = importlib.import_module(SUBCOMMANDS[subcommand])
            return mod.run(remaining)  # type: ignore[no-any-return]

        from qi.commands.run import run as run_cmd

        return run_cmd(remaining)
    except OSError as e:
        if not _is_broken_pipe(e):
            raise
        # The reader went away (e.g. `qi ... | head`). Standard Unix tools die
        # quietly on SIGPIPE; mirror that with the conventional 128+SIGPIPE code.
        # Point stdout at devnull (os.devnull is 'nul' on Windows) so the
        # interpreter's exit-time flush of the broken stream can't print
        # "Exception ignored" noise.
        with contextlib.suppress(OSError, ValueError):
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        return 141


if __name__ == "__main__":
    sys.exit(main())
