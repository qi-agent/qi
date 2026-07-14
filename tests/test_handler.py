"""Tests for handle_response turn-termination and tool-call dispatch."""

from unittest.mock import patch

from qi.lib.constants import LogKey, Role
from qi.lib.handler import handle_response
from qi.lib.llm_client._types import ToolCall


def test_empty_messages_array_ends_turn() -> None:
    outputs, done = handle_response('{"messages":[]}', [])

    assert done is True
    assert outputs == []


def test_reply_without_done_flag_ends_turn() -> None:
    content = '{"messages":[{"type":"reply","content":"hi","done":false}]}'

    outputs, done = handle_response(content, [])

    assert done is True
    assert outputs == []


def test_reply_with_done_flag_ends_turn() -> None:
    content = '{"messages":[{"type":"reply","content":"all done","done":true}]}'

    _, done = handle_response(content, [])

    assert done is True


def test_question_interactive_collects_answer_and_continues() -> None:
    content = '{"messages":[{"type":"question","content":"proceed?","done":false}]}'

    with patch("qi.lib.handler.console.input", return_value="yes"):
        outputs, done = handle_response(content, [], interactive=True)

    assert done is False
    assert outputs == [{LogKey.ROLE.value: Role.USER.value, LogKey.CONTENT.value: "yes"}]


def test_question_piped_ends_turn_without_reading_stdin() -> None:
    content = '{"messages":[{"type":"question","content":"proceed?","done":false}]}'

    outputs, done = handle_response(content, [], interactive=False)

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


def test_tool_calls_ignored_when_done() -> None:
    content = '{"messages":[{"type":"reply","content":"finished","done":true}]}'
    tool_calls = [ToolCall(id="call_1", name="Echo", args={"text": "hi"})]

    with patch("qi.lib.handler.TOOL_MAP", {"Echo": lambda text: f"echoed {text}"}):
        outputs, done = handle_response(content, tool_calls)

    assert done is True
    assert outputs == []


def test_unparseable_content_is_final_reply() -> None:
    outputs, done = handle_response("not json {", [])

    assert done is True
    assert outputs == []
