"""Response handler for LLM JSONL responses and tool execution."""

import json
import logging
import re
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

logger = logging.getLogger(__name__)
console = Console()

ToolMap = dict[str, Any]


def _truncate(obj: object, max_len: int = 500) -> str:
    s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"... (truncated, {len(s)} total chars)"
    return s


def _read_file(path: str, start: int = 0, end: int | None = None) -> str:
    with open(path) as f:
        lines = f.readlines()
    if end is not None:
        lines = lines[start:end]
    elif start > 0:
        lines = lines[start:]
    return "".join(lines)


DEFAULT_TOOLS: ToolMap = {
    "ReadFile": _read_file,
}

_TOOL_PARAMS: dict[str, tuple[str, ...]] = {
    "ReadFile": ("path", "start", "end"),
}


def _normalize_args(tool_name: str, args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, list):
        params = _TOOL_PARAMS.get(tool_name)
        if params:
            return dict(zip(params, args))
    return {}


def _strip_code_fence(content: str) -> str:
    content = re.sub(r'\A```\w*\n?', '', content)
    content = re.sub(r'\n?```\s*\Z', '', content)
    return content.strip()


def handle_response(
    content: str,
    tool_map: ToolMap | None = None,
) -> tuple[list[dict[str, str]] | None, bool]:
    if tool_map is None:
        tool_map = DEFAULT_TOOLS

    content = _strip_code_fence(content)
    reply_messages: list[dict[str, str]] = []
    done = False
    try:
        body = json.loads(content)
        if isinstance(body, dict):
            items = body.get("messages", [body])
        else:
            items = body if isinstance(body, list) else [body]
        for item in items:
            match item.get("type"):

                case "thought":
                    content = item.get("content", "")
                    logger.debug("Thought: %s", content)
                    console.print(content, style="dim", soft_wrap=True)

                case "reply":
                    console.print(Markdown(item["content"]), style="bold", soft_wrap=True)

                case "ask":
                    console.print(Markdown(item["content"]), style="bold", soft_wrap=True)
                    answer = console.input("[bold cyan]> [/bold cyan]")
                    reply_messages.append({"role": "user", "content": answer})

                case "conclusion":
                    console.print(Markdown(item["content"]), style="bold", soft_wrap=True)
                    done = True

                case "call":
                    tool_name = item.get("tool", "")
                    args = _normalize_args(tool_name, item.get("args", {}))
                    console.print(f"[blue]{tool_name}[/blue] [dim]{json.dumps(args)}[/dim]", soft_wrap=True)
                    tool_fn = tool_map.get(tool_name)
                    if tool_fn is None:
                        logger.error(f"Unknown tool: {tool_name}")
                        done = True
                    else:
                        result = tool_fn(**args)
                        logger.info("Tool result:\n%s\n=============", _truncate(result))
                        reply_messages.append({"role": "tool", "content": result})

                case _:
                    done = True
                    logger.warning("Unknown type: %s", item.get("type", "unknown"))

    except json.JSONDecodeError as e:
        logger.error(f"Unable to parse JSON: {e}")
        logger.error(f"Full response:\n{content}")

    return reply_messages, done
