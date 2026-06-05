"""Response handler for LLM structured JSON responses and native tool execution."""

import json
import logging
import re
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from qi.lib.constants import (
    LogKey,
    MessageKey,
    MessageType,
    Role,
)
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


def handle_response(
    content: str | None,  # OpenRouter would give content = None when combined with tool calling
    tool_calls: list[ToolCall],
) -> tuple[list[dict[str, str]] | None, bool]:
    content = _strip_code_fence(content or "")
    reply_messages: list[dict[str, Any]] = []
    done = False
    try:
        body = json.loads(content) if content else []
        items: list[dict[str, Any]] 
        if isinstance(body, dict):
            items = body.get("messages", [body])
        else:
            items: list[dict[str, Any]] = body if isinstance(body, list) else [body]

        for item in items:
            match item.get(MessageKey.TYPE):

                case MessageType.THOUGHT:
                    content = item.get(MessageKey.CONTENT, "")
                    logger.debug("Thought: %s", content)
                    console.print(content, style="dim")

                case MessageType.REPLY:
                    console.print(Markdown(item[MessageKey.CONTENT]), style="bold")

                case MessageType.ASK:
                    console.print(Markdown(item[MessageKey.CONTENT]), style="bold")
                    answer = console.input("[bold cyan]> [/bold cyan]")
                    reply_messages.append({LogKey.ROLE.value: Role.USER.value, LogKey.CONTENT.value: answer})

                case MessageType.CONCLUSION:
                    console.print(Markdown(item[MessageKey.CONTENT]), style="bold")
                    done = True

                case MessageType.CALL:
                    # inline assistant message tool call - Google API does this
                    # {"type": "call", "api": "default_api:ReadFile", "parameters": ["olaf.txt"]}
                    # reply_messages.append({"role": "assistant", "content": "", "tool_calls": [item]})
                    call_res = handle_tool_calls([
                        ToolCall(name=item[MessageKey.API].removeprefix("default_api:"), args=item[MessageKey.PARAMETERS])
                    ])
                    reply_messages.append(call_res[0])
                case _:
                    done = True
                    logger.warning("Unknown type: %s", item.get(MessageKey.TYPE, "unknown"))

    except json.JSONDecodeError as e:
        logger.error(f"Unable to parse JSON: {e}")
        logger.error(f"Full response:\n{content}")

    # top-level tool calls
    if tool_calls:
        if done:
            logger.warning(f"Ignoring tool call requests from LLM since it signalled done: {tool_calls}")
        else:
            tool_msgs = handle_tool_calls(tool_calls)
            reply_messages.extend(tool_msgs)
    else:
        # done if no tool calls and no asks
        done = done or not {MessageType.ASK, MessageType.CALL}.union({item[MessageKey.TYPE] for item in items})

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
                LogKey.ROLE.value: Role.TOOL.value,
                LogKey.NAME.value: tc.name,
                LogKey.CONTENT.value: f"ERROR: Unknown tool {tc.name}",
            })
            continue

        if isinstance(tc.args, (list, tuple)):
            result = tool_fn(*tc.args)
        else:
            result = tool_fn(**tc.args)

        logger.info("Tool result:\n%s\n=============", _truncate(result))

        # https://developers.openai.com/api/docs/guides/function-calling#handling-function-calls
        messages.append({
            LogKey.ROLE.value: Role.TOOL.value,
            LogKey.TOOL_CALL_ID.value: tc.id,
            LogKey.NAME.value: tc.name,
            LogKey.CONTENT.value: result,
        })
    return messages
