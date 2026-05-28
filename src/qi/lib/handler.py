"""Response handler for LLM structured JSON responses and native tool execution."""

import json
import logging
import re
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from qi.lib.llm_client._types import ToolCall
from qi.tools import TOOL_MAP

logger = logging.getLogger(__name__)
console = Console()

ToolMap = dict[str, Any]
FnTool = Any


def _truncate(obj: object, max_len: int = 5000) -> str:
    s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"... (truncated, {len(s)} total chars)"
    return s


def _strip_code_fence(content: str) -> str:
    content = re.sub(r'\A```\w*\n?', '', content)
    content = re.sub(r'\n?```\s*\Z', '', content)
    return content.strip()


def _assistant_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.args),
            },
        }
        for tc in tool_calls
    ]


def handle_response(
    content: str,
    tool_calls: list[ToolCall],
) -> tuple[list[dict[str, str]] | None, bool]:
    content = _strip_code_fence(content)
    reply_messages: list[dict[str, Any]] = []
    done = False
    try:
        body = json.loads(content) if content else []
        if isinstance(body, dict):
            items = body.get("messages", [body])
        else:
            items = body if isinstance(body, list) else [body]

        for item in items:
            match item.get("type"):

                case "thought":
                    content = item.get("content", "")
                    logger.debug("Thought: %s", content)
                    console.print(content, style="dim", soft_wrap=True, crop=False, overflow="fold")

                case "reply":
                    console.print(Markdown(item["content"]), style="bold", soft_wrap=True, crop=False, overflow="fold")

                case "ask":
                    console.print(Markdown(item["content"]), style="bold", soft_wrap=True, crop=False, overflow="fold")
                    answer = console.input("[bold cyan]> [/bold cyan]")
                    reply_messages.append({"role": "user", "content": answer})

                case "conclusion":
                    console.print(Markdown(item["content"]), style="bold", soft_wrap=True, crop=False, overflow="fold")
                    done = True

                case "call":
                    # inline assistant message tool call - Google API does this
                    # {"type": "call", "api": "default_api:ReadFile", "parameters": ["olaf.txt"]}
                    reply_messages.append({"role": "assistant", "content": "", "tool_calls": [item]})
                    call_res = handle_tool_calls([
                        ToolCall(name=item["api"].removeprefix("default_api:"), args=item["parameters"])
                    ])
                    reply_messages.append(call_res[0])
                case _:
                    done = True
                    logger.warning("Unknown type: %s", item.get("type", "unknown"))

    except json.JSONDecodeError as e:
        logger.error(f"Unable to parse JSON: {e}")
        logger.error(f"Full response:\n{content}")

    # top-level tool calls
    if tool_calls:
        tool_msgs = handle_tool_calls(tool_calls)
        reply_messages.extend(tool_msgs)

    return reply_messages, done


def handle_tool_calls(
    tool_calls: list[ToolCall],
    tool_map: ToolMap | None = None,
) -> list[dict[str, str]]:
    if tool_map is None:
        tool_map = TOOL_MAP

    messages: list[dict[str, str]] = []
    for tc in tool_calls:
        console.print(f"[blue]{tc.name}[/blue] [dim]{json.dumps(tc.args)}[/dim]", soft_wrap=True)
        tool_fn = tool_map.get(tc.name)
        if tool_fn is None:
            logger.error(f"Unknown tool: {tc.name}")
            messages.append({
                "role": "tool",
                "name": tc.name,
                "content": f"ERROR: Unkonwn tool {tc.name}",
            })
            continue

        if isinstance(tc.args, (list, tuple)):
            result = tool_fn(*tc.args)  # type: ignore[arg-type]
        else:
            result = tool_fn(**tc.args)  # type: ignore[arg-type]

        logger.info("Tool result:\n%s\n=============", _truncate(result))

        # https://developers.openai.com/api/docs/guides/function-calling#handling-function-calls
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            # "type": "tool_output",
            "content": result,
        })
    return messages
