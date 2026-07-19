"""Graph subcommand: depict the subagent graph recorded in a session."""

import argparse
import json
import logging
from pathlib import Path

from qi.lib.agents import (
    BACKGROUND_KEY,
    DETAIL_KEY,
    EVENT_KEY,
    EVENT_SPAWN,
    NAME_KEY,
    PROMPT_KEY,
    READS_FROM_KEY,
    AgentGraph,
    AgentNode,
)
from qi.lib.constants import LogKey, RecordType

FORMAT_ASCII = "ascii"
FORMAT_MERMAID = "mermaid"
FORMAT_DOT = "dot"

logger = logging.getLogger(__name__)


def _get_session_dir() -> Path:
    return Path('.').resolve() / ".qi" / "sessions"


def _latest_session(session_dir: Path) -> Path | None:
    files = sorted(session_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def build_graph(file_path: Path) -> AgentGraph:
    """Rebuild the agent graph from a session log's agent-event records."""
    graph = AgentGraph()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get(LogKey.TYPE) != RecordType.AGENT:
                continue
            meta = record.get(LogKey.META)
            if not isinstance(meta, dict):
                continue
            event = str(meta.get(EVENT_KEY, ""))
            name = str(meta.get(NAME_KEY, ""))
            if not event or not name:
                continue
            if event == EVENT_SPAWN:
                reads_from = meta.get(READS_FROM_KEY) or []
                if not isinstance(reads_from, list):
                    reads_from = []
                try:
                    graph.add(AgentNode(
                        name=name,
                        prompt=str(meta.get(PROMPT_KEY, "")),
                        reads_from=tuple(str(u) for u in reads_from),
                        background=bool(meta.get(BACKGROUND_KEY, False)),
                    ))
                except ValueError as e:
                    logger.warning(f"Skipping malformed agent record in {file_path.name}: {e}")
            elif name in graph.nodes:
                graph.nodes[name].status = event
                graph.nodes[name].detail = str(meta.get(DETAIL_KEY, ""))
    return graph


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qi graph",
        description="Depict the subagent graph recorded in a session",
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        help="Session id under .qi/sessions (defaults to the most recent session)",
    )
    parser.add_argument(
        "--format",
        choices=[FORMAT_ASCII, FORMAT_MERMAID, FORMAT_DOT],
        default=FORMAT_ASCII,
        help="Output format",
    )
    parsed = parser.parse_args(argv)

    session_dir = _get_session_dir()
    if parsed.session_id:
        # Same guard as --resume: a session id is a bare file stem.
        if "/" in parsed.session_id or "\\" in parsed.session_id or ".." in parsed.session_id:
            logger.error(f"Invalid session id '{parsed.session_id}': path separators are not allowed.")
            return 1
        file_path = session_dir / f"{parsed.session_id}.jsonl"
        if not file_path.exists():
            logger.error(f"No session '{parsed.session_id}' found in {session_dir}")
            return 1
    else:
        latest = _latest_session(session_dir)
        if latest is None:
            logger.error(f"No sessions found in {session_dir}")
            return 1
        file_path = latest

    graph = build_graph(file_path)
    if not graph.nodes:
        print(f"No subagent activity recorded in session '{file_path.stem}'.")
        return 0

    if parsed.format == FORMAT_MERMAID:
        print(graph.render_mermaid())
    elif parsed.format == FORMAT_DOT:
        print(graph.render_dot())
    else:
        print(f"Agent graph for session '{file_path.stem}':")
        print(graph.render_ascii())
    return 0
