"""Tests for handle_response turn-termination and tool-call dispatch.

Done-ness is structural: the turn ends when the model stops requesting tool
calls. Content is plain markdown, never a JSON protocol. Questions to the user
are an AskUser tool call, not a content type.
"""

import logging
from unittest.mock import patch

import pytest

from qi.lib.constants import LogKey, Role
from qi.lib.handler import DEFERRED_ANSWER_NOTE, handle_response
from qi.lib.llm_client._types import ToolCall


def test_plain_text_without_tool_calls_ends_turn() -> None:
    outputs, done = handle_response("All fixed. The bug was in `foo()`.", [])

    assert done is True
    assert outputs == []


def test_empty_response_ends_turn() -> None:
    outputs, done = handle_response(None, [])

    assert done is True
    assert outputs == []


def test_json_looking_content_is_not_parsed() -> None:
    # Legacy protocol payloads are now just markdown; no parsing, no crash.
    outputs, done = handle_response('{"messages":[{"type":"reply","done":false}]}', [])

    assert done is True
    assert outputs == []


def test_tool_calls_execute_and_continue() -> None:
    tool_calls = [ToolCall(id="call_1", name="Echo", args={"text": "hi"})]

    with patch("qi.lib.handler.TOOL_MAP", {"Echo": lambda text: f"echoed {text}"}):
        outputs, done = handle_response(None, tool_calls)

    assert done is False
    assert outputs == [{
        LogKey.ROLE.value: Role.TOOL.value,
        LogKey.TOOL_CALL_ID.value: "call_1",
        LogKey.NAME.value: "Echo",
        LogKey.CONTENT.value: "echoed hi",
    }]


def test_content_alongside_tool_calls_does_not_end_turn() -> None:
    tool_calls = [ToolCall(id="call_1", name="Echo", args={"text": "hi"})]

    with patch("qi.lib.handler.TOOL_MAP", {"Echo": lambda text: f"echoed {text}"}):
        outputs, done = handle_response("Let me check that file.", tool_calls)

    assert done is False
    assert len(outputs) == 1


def test_ask_user_interactive_returns_answer_as_tool_result() -> None:
    tool_calls = [ToolCall(id="call_ask", name="AskUser", args={"question": "proceed?"})]

    with patch("qi.lib.handler.console.input", return_value="yes"):
        outputs, done = handle_response(None, tool_calls, interactive=True)

    assert done is False
    assert outputs == [{
        LogKey.ROLE.value: Role.TOOL.value,
        LogKey.TOOL_CALL_ID.value: "call_ask",
        LogKey.NAME.value: "AskUser",
        LogKey.CONTENT.value: "yes",
    }]


def test_ask_user_piped_defers_answer_and_ends_turn() -> None:
    tool_calls = [ToolCall(id="call_ask", name="AskUser", args={"question": "proceed?"})]

    outputs, done = handle_response(None, tool_calls, interactive=False)

    assert done is True
    assert outputs == [{
        LogKey.ROLE.value: Role.TOOL.value,
        LogKey.TOOL_CALL_ID.value: "call_ask",
        LogKey.NAME.value: "AskUser",
        LogKey.CONTENT.value: DEFERRED_ANSWER_NOTE,
    }]


def test_ask_user_piped_never_reads_stdin() -> None:
    tool_calls = [ToolCall(id="call_ask", name="AskUser", args={"question": "proceed?"})]

    with patch(
        "qi.lib.handler.console.input",
        side_effect=AssertionError("console.input must not be called in piped mode"),
    ):
        _, done = handle_response(None, tool_calls, interactive=False)

    assert done is True


def test_ask_user_accepts_positional_args() -> None:
    # Gemini occasionally sends positional parameters instead of a dict.
    tool_calls = [ToolCall(id="call_ask", name="AskUser", args=["which file?"])]

    with patch("qi.lib.handler.console.input", return_value="main.py"):
        outputs, done = handle_response(None, tool_calls, interactive=True)

    assert done is False
    assert outputs[0][LogKey.CONTENT.value] == "main.py"


def test_ask_user_mixed_with_other_tools_piped_answers_all_calls() -> None:
    # Every tool_call must get a result even when the turn ends on a deferred
    # question, or the next request violates the providers' message protocol.
    tool_calls = [
        ToolCall(id="call_1", name="Echo", args={"text": "hi"}),
        ToolCall(id="call_ask", name="AskUser", args={"question": "proceed?"}),
    ]

    with patch("qi.lib.handler.TOOL_MAP", {"Echo": lambda text: f"echoed {text}"}):
        outputs, done = handle_response(None, tool_calls, interactive=False)

    assert done is True
    assert [o[LogKey.TOOL_CALL_ID.value] for o in outputs] == ["call_1", "call_ask"]


def test_truncated_response_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _, done = handle_response("partial answ", [], finish_reason="length")

    assert done is True
    assert any("truncated" in r.getMessage().lower() for r in caplog.records)
