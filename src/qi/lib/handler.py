"""Response handler: structural done-ness signaling and native tool execution.

The turn-termination protocol is structural, mirroring what the provider APIs
already report (OpenAI finish_reason, Anthropic stop_reason, Gemini
functionCall parts):

- tool calls present  -> execute them, keep looping
- no tool calls       -> the model ended its turn; content is the final reply

Questions to the user are an AskUser tool call, so content is plain markdown
and is never parsed.
"""

import json
import logging
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from qi.lib.constants import LogKey, Role
from qi.lib.llm_client._types import ToolCall
from qi.tools import TOOL_MAP
from qi.tools.ask_user import AskUserTool

logger = logging.getLogger(__name__)
console = Console()

ASK_USER_TOOL_NAME = AskUserTool.name

# Piped mode: stdin carries the message protocol, so a question can never read
# a raw answer. This note closes the tool call (keeping the providers' message
# protocol valid) while the turn ends and the next user message is the answer.
DEFERRED_ANSWER_NOTE = "[Question delivered to the user. Their next message is the answer.]"


def route_console_output(to_stderr: bool) -> None:
    """Route human-readable output to stderr so stdout can carry a machine protocol."""
    global console
    console = Console(stderr=to_stderr)

ToolMap = dict[str, Any]


def _truncate(obj: object, max_len: int = 5000) -> str:
    s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"... (truncated, {len(s)} total chars)"
    return s


def _question_from_args(args: dict[str, object] | list[object]) -> str:
    if isinstance(args, dict):
        return str(args.get("question", ""))
    return str(args[0]) if args else ""


def handle_response(
    content: str | None,  # OpenRouter would give content = None when combined with tool calling
    tool_calls: list[ToolCall],
    interactive: bool = True,
    finish_reason: str = "",
) -> tuple[list[dict[str, Any]] | None, bool]:
    if finish_reason == "length":
        logger.warning("LLM response was truncated (finish_reason=length); consider raising max_tokens.")

    if content:
        console.print(Markdown(content), style="bold")

    reply_messages: list[dict[str, Any]] = []
    awaiting_user = False
    for tc in tool_calls:
        if tc.name == ASK_USER_TOOL_NAME:
            question = _question_from_args(tc.args)
            console.print(Markdown(question), style="bold")
            if interactive:
                answer = console.input("[bold cyan]> [/bold cyan]")
            else:
                answer = DEFERRED_ANSWER_NOTE
                awaiting_user = True
            reply_messages.append({
                LogKey.ROLE.value: Role.TOOL.value,
                LogKey.TOOL_CALL_ID.value: tc.id,
                LogKey.NAME.value: tc.name,
                LogKey.CONTENT.value: answer,
            })
        else:
            reply_messages.extend(handle_tool_calls([tc]))

    # Structural done-ness: no tool calls means the model ended its turn. A
    # deferred question also ends the turn — the next stdin message answers it.
    done = not tool_calls or awaiting_user
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
