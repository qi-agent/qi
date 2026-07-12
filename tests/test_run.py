"""Tests for the run command."""

import io
import json
import re
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from qi.commands.run import run
from qi.lib.config import Settings
from qi.lib.llm_client._types import LLMResponse, ToolCall


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def test_files_sent_as_user_messages() -> None:
    """Files are read and sent as user messages to the LLM."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(content='{"result": "ok"}')


    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", mock_open(read_data="file content")),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["test.py"])

    assert rc == 0
    mock_client.chat.assert_called_once()
    messages = mock_client.chat.call_args[0][0]

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] is not None
    assert len(messages[0]["content"]) > 0

    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "file content"


def test_files_sent_as_user_messages_capsys(capsys: pytest.CaptureFixture[str]) -> None:
    """LLM response is printed to stdout."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content=(
            '{"messages": [{"type": "reply", "content": "analysis complete"},'
            ' {"type": "conclusion", "content": ""}]}'
        )
    )

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", mock_open(read_data="x")),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["f.py"])

    assert rc == 0
    out, _ = capsys.readouterr()
    lines = _strip_ansi(out).strip().splitlines()
    assert lines[-1] == "analysis complete"


def test_prompt_adds_instruction_message() -> None:
    """-p adds an instruction message before file contents."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(content="{}")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("builtins.open", mock_open(read_data="code")),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["-p", "review this code", "test.py"])

    assert rc == 0
    messages = mock_client.chat.call_args[0][0]

    assert messages[0]["role"] == "system"

    assert messages[1]["role"] == "user"
    assert "review this code" in str(messages[1]["content"])

    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "code"


def test_multiple_files() -> None:
    """Multiple files are each sent as separate user messages."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(content="{}")

    handles = [
        mock_open(read_data="content a").return_value,
        mock_open(read_data="content b").return_value,
    ]

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", side_effect=handles),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["a.py", "b.py"])

    assert rc == 0
    messages = mock_client.chat.call_args[0][0]
    assert messages[-2]["content"] == "content a"
    assert messages[-1]["content"] == "content b"


def test_missing_file_returns_error() -> None:
    """If a file doesn't exist, return non-zero exit code."""
    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.lib.session.Session._write"),
        patch("qi.commands.run.LLMClient.create"),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["nonexistent.py"])

    assert rc != 0


def test_llm_error_returns_error() -> None:
    """If the LLM call fails, return non-zero exit code."""
    mock_client = Mock()
    mock_client.chat.side_effect = Exception("API error")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", mock_open(read_data="x")),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["f.py"])

    assert rc == 1


def test_tool_calls_are_executed() -> None:
    """Native tool calls from the LLM are executed and results added to messages."""
    mock_client = Mock()
    mock_client.chat.side_effect = [
        LLMResponse(
            content='{"messages": [{"type": "thought", "content": "Need to read the file"}]}',
            tool_calls=[
                ToolCall(
                    style="openai",
                    id="call_1",
                    name="ReadFile",
                    args={"path": "test.py"},
                )
            ],
        ),
        LLMResponse(
            content='{"messages": [{"type": "reply", "content": "done"}, {"type": "conclusion", "content": ""}]}',
        ),
    ]

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", mock_open(read_data="file content")),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["test.py"])

    assert rc == 0
    assert mock_client.chat.call_count == 2


def test_passes_tools_and_response_format() -> None:
    """Tools and response_format are passed to the LLM client."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(content="{}")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", mock_open(read_data="x")),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )

        rc = run(["f.py"])
        assert rc == 0

    kwargs = mock_client.chat.call_args.kwargs
    assert "tools" in kwargs
    assert "response_format" in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"


class _PipedStdin(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_piped_mode_does_not_require_prompt_or_files(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "done"}]}'
    )
    stdin = _PipedStdin(json.dumps({"role": "user", "content": "from-pipe"}) + "\n")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run([])

    assert rc == 0
    mock_client.chat.assert_called_once()


def test_piped_mode_does_not_inject_default_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "done"}]}'
    )
    stdin = _PipedStdin(json.dumps({"role": "user", "content": "literal-user-message"}) + "\n")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run([])

    assert rc == 0
    messages = mock_client.chat.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "literal-user-message"}


def test_piped_mode_processes_each_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = Mock()
    mock_client.chat.side_effect = [
        LLMResponse(content='{"messages": [{"type": "conclusion", "content": "one"}]}'),
        LLMResponse(content='{"messages": [{"type": "conclusion", "content": "two"}]}'),
    ]
    stdin = _PipedStdin(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "first"}),
                json.dumps({"role": "assistant", "content": "ignore"}),
                json.dumps({"role": "user", "content": "second"}),
            ]
        )
        + "\n"
    )

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
    ):
        mock_load.return_value = Settings(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
            temperature=0.0,
        )
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run([])

    assert rc == 0
    assert mock_client.chat.call_count == 2
    first_messages = mock_client.chat.call_args_list[0][0][0]
    second_messages = mock_client.chat.call_args_list[1][0][0]
    assert first_messages[1]["content"] == "first"
    # Continuous session: the second call carries the first turn's history, so the
    # first user message is still "first", the assistant's first reply follows, and
    # "second" is the latest (last) message.
    assert second_messages[1]["content"] == "first"
    assert second_messages[2]["role"] == "assistant"
    assert second_messages[-1]["content"] == "second"


def _piped_settings() -> Settings:
    return Settings(
        api_key="sk-test",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        max_tokens=4096,
        temperature=0.0,
    )


def test_piped_mode_prompt_and_stdin_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    """--prompt is logged as a user turn; the first stdin line drives the first call,
    so that call sees prompt + line as consecutive user messages with no assistant
    turn in between."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "done"}]}'
    )
    stdin = _PipedStdin(json.dumps({"role": "user", "content": "stdin-line"}) + "\n")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run(["-p", "prompt-instruction"])

    assert rc == 0
    # The prompt alone did not trigger a round-trip; the single stdin line drove the
    # single call.
    mock_client.chat.assert_called_once()
    messages = mock_client.chat.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "prompt-instruction"}
    assert messages[2] == {"role": "user", "content": "stdin-line"}
    assert all(m["role"] != "assistant" for m in messages)


def test_piped_mode_prompt_only_empty_stdin_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt-only invocation with an empty/closed pipe still runs one loop."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "done"}]}'
    )
    stdin = _PipedStdin("")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run(["-p", "just-a-prompt"])

    assert rc == 0
    mock_client.chat.assert_called_once()
    messages = mock_client.chat.call_args[0][0]
    assert messages[-1] == {"role": "user", "content": "just-a-prompt"}


def test_piped_mode_invalid_json_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed JSON line aborts with a non-zero code and does not skip ahead."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "done"}]}'
    )
    stdin = _PipedStdin(
        "this is not json\n" + json.dumps({"role": "user", "content": "after"}) + "\n"
    )

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run([])

    assert rc != 0
    mock_client.chat.assert_not_called()


def test_piped_mode_non_dict_line_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid JSON that isn't an object aborts cleanly instead of raising AttributeError."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "done"}]}'
    )
    stdin = _PipedStdin(
        "[1, 2]\n" + json.dumps({"role": "user", "content": "after"}) + "\n"
    )

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run([])

    assert rc != 0
    mock_client.chat.assert_not_called()


def test_output_format_accepts_jsonl_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """--output-format accepts 'jsonl' and 'text' without changing the run outcome."""
    for output_format in ("jsonl", "text"):
        mock_client = Mock()
        mock_client.chat.return_value = LLMResponse(
            content='{"messages": [{"type": "conclusion", "content": "done"}]}'
        )
        stdin = _PipedStdin(json.dumps({"role": "user", "content": "hi"}) + "\n")

        with (
            patch("qi.commands.run.load") as mock_load,
            patch("qi.commands.run.LLMClient.create", return_value=mock_client),
            patch("qi.lib.session.Session._write"),
        ):
            mock_load.return_value = _piped_settings()
            monkeypatch.setattr("sys.stdin", stdin)

            rc = run(["--output-format", output_format])

        assert rc == 0
        mock_client.chat.assert_called_once()


def test_output_format_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown --output-format value is rejected by argument parsing."""
    monkeypatch.setattr("sys.stdin", _PipedStdin(""))
    with pytest.raises(SystemExit):
        run(["--output-format", "xml"])


def _run_piped_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    mock_client: Mock,
    stdin_lines: list[str],
    argv: list[str] | None = None,
) -> int:
    stdin = _PipedStdin("\n".join(stdin_lines) + ("\n" if stdin_lines else ""))
    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)
        return run(argv if argv is not None else ["--output-format", "jsonl"])


def test_jsonl_mode_stdout_is_pure_jsonl(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --output-format jsonl every stdout line is a JSON object; the
    human-readable rendering must not pollute the stream."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "all done"}]}'
    )

    rc = _run_piped_jsonl(
        monkeypatch, mock_client, [json.dumps({"role": "user", "content": "hi"})]
    )

    assert rc == 0
    out, _ = capsys.readouterr()
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines, "expected JSONL events on stdout"
    for line in lines:
        obj = json.loads(line)
        assert isinstance(obj, dict)


def test_jsonl_mode_emits_session_events(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The jsonl stream mirrors the session log: session_start first, then message
    events carrying role/content, including the user turn and the assistant reply."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "all done"}]}'
    )

    rc = _run_piped_jsonl(
        monkeypatch, mock_client, [json.dumps({"role": "user", "content": "hi"})]
    )

    assert rc == 0
    out, _ = capsys.readouterr()
    events = [json.loads(line) for line in out.splitlines() if line.strip()]

    assert events[0]["type"] == "session_start"
    messages = [e for e in events if e["type"] == "message"]
    assert any(m["role"] == "user" and m["content"] == "hi" for m in messages)
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert assistant, "expected an assistant message event"
    assert "all done" in assistant[-1]["content"]


def test_jsonl_mode_emits_tool_call_and_result_events(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mock_client = Mock()
    mock_client.chat.side_effect = [
        LLMResponse(
            content='{"messages": [{"type": "thought", "content": "reading"}]}',
            tool_calls=[
                ToolCall(style="openai", id="call_1", name="ReadFile", args={"path": "t.py"})
            ],
        ),
        LLMResponse(
            content='{"messages": [{"type": "conclusion", "content": "done"}]}'
        ),
    ]

    with patch("builtins.open", mock_open(read_data="file content")):
        rc = _run_piped_jsonl(
            monkeypatch, mock_client, [json.dumps({"role": "user", "content": "go"})]
        )

    assert rc == 0
    out, _ = capsys.readouterr()
    events = [json.loads(line) for line in out.splitlines() if line.strip()]
    with_tool_calls = [e for e in events if e.get("tool_calls")]
    assert with_tool_calls and with_tool_calls[0]["role"] == "assistant"
    tool_results = [e for e in events if e.get("role") == "tool"]
    assert tool_results and tool_results[0]["name"] == "ReadFile"
    assert tool_results[0]["tool_call_id"] == "call_1"


def test_one_shot_mode_supports_jsonl_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--output-format jsonl also works in one-shot (non-piped) mode: stdout is a
    pure JSONL event stream, human rendering goes to stderr."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "one-shot done"}]}'
    )

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", mock_open(read_data="file content")),
    ):
        mock_load.return_value = _piped_settings()

        rc = run(["-p", "analyze", "f.py", "--output-format", "jsonl"])

    assert rc == 0
    out, _ = capsys.readouterr()
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines, "expected JSONL events on stdout"
    events = [json.loads(line) for line in lines]
    assert all(isinstance(e, dict) for e in events)
    assert events[0]["type"] == "session_start"
    assert any(e.get("role") == "assistant" for e in events)


def test_text_mode_emits_no_jsonl_events(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default text mode keeps human-readable output and no JSON event lines."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "plain text done"}]}'
    )

    rc = _run_piped_jsonl(
        monkeypatch,
        mock_client,
        [json.dumps({"role": "user", "content": "hi"})],
        argv=[],
    )

    assert rc == 0
    out, _ = capsys.readouterr()
    for line in _strip_ansi(out).splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        assert not isinstance(parsed, dict), f"unexpected JSON event in text mode: {line}"


def test_piped_mode_skips_unknown_command_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid JSON object that isn't a user message is skipped, and processing
    continues with the next line. EOF — not any in-band command — ends the run."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "done"}]}'
    )

    rc = _run_piped_jsonl(
        monkeypatch,
        mock_client,
        [
            json.dumps({"type": "future-command", "payload": 1}),
            json.dumps({"type": "interrupt"}),
            json.dumps({"role": "user", "content": "after-unknown"}),
        ],
        argv=[],
    )

    assert rc == 0
    mock_client.chat.assert_called_once()
    messages = mock_client.chat.call_args[0][0]
    assert messages[-1]["content"] == "after-unknown"


def test_piped_mode_question_is_answered_by_next_stdin_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In piped mode a question must never read raw stdin (that would swallow
    protocol lines); it ends the turn and the next user message is the answer."""
    mock_client = Mock()
    mock_client.chat.side_effect = [
        LLMResponse(content='{"messages": [{"type": "question", "content": "which one?"}]}'),
        LLMResponse(content='{"messages": [{"type": "conclusion", "content": "done"}]}'),
    ]

    with patch(
        "rich.console.Console.input",
        side_effect=AssertionError("console.input must not be called in piped mode"),
    ):
        rc = _run_piped_jsonl(
            monkeypatch,
            mock_client,
            [
                json.dumps({"role": "user", "content": "start"}),
                json.dumps({"role": "user", "content": "my answer"}),
            ],
            argv=[],
        )

    assert rc == 0
    assert mock_client.chat.call_count == 2
    second_messages = mock_client.chat.call_args_list[1][0][0]
    assert second_messages[-1] == {"role": "user", "content": "my answer"}


def test_piped_mode_question_at_eof_ends_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A question when stdin is exhausted must not raise EOFError; the run ends."""
    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "question", "content": "anyone there?"}]}'
    )

    with patch(
        "rich.console.Console.input",
        side_effect=AssertionError("console.input must not be called in piped mode"),
    ):
        rc = _run_piped_jsonl(
            monkeypatch,
            mock_client,
            [json.dumps({"role": "user", "content": "start"})],
            argv=[],
        )

    assert rc == 0
    mock_client.chat.assert_called_once()


def test_interactive_question_still_prompts_on_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside piped mode the question flow keeps reading the answer interactively."""
    mock_client = Mock()
    mock_client.chat.side_effect = [
        LLMResponse(content='{"messages": [{"type": "question", "content": "which one?"}]}'),
        LLMResponse(content='{"messages": [{"type": "conclusion", "content": "done"}]}'),
    ]

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", mock_open(read_data="x")),
        patch("rich.console.Console.input", return_value="typed-answer"),
    ):
        mock_load.return_value = _piped_settings()

        rc = run(["f.py"])

    assert rc == 0
    assert mock_client.chat.call_count == 2
    second_messages = mock_client.chat.call_args_list[1][0][0]
    assert second_messages[-1]["role"] == "user"
    assert second_messages[-1]["content"] == "typed-answer"


def _write_session_fixture(session_dir: Path, session_id: str) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / f"{session_id}.jsonl"
    records = [
        {"type": "session_start", "timestamp": "2026-07-12T00:00:00", "role": "", "meta": {"model": "gpt-4o"}},
        {"type": "message", "timestamp": "2026-07-12T00:00:01", "role": "system", "content": "sys-prompt"},
        {"type": "message", "timestamp": "2026-07-12T00:00:02", "role": "user", "content": "earlier question"},
        {"type": "message", "timestamp": "2026-07-12T00:00:03", "role": "assistant", "content": "earlier answer"},
    ]
    file_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return file_path


def test_resume_continues_existing_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--resume rebuilds history from the session file, appends the new turn to the
    same file, and emits the new events on the jsonl stream."""
    monkeypatch.chdir(tmp_path)
    session_id = "20260712T000000-deadbeef-earlier_question"
    file_path = _write_session_fixture(tmp_path / ".qi" / "sessions", session_id)

    mock_client = Mock()
    mock_client.chat.return_value = LLMResponse(
        content='{"messages": [{"type": "conclusion", "content": "resumed done"}]}'
    )
    stdin = _PipedStdin(json.dumps({"role": "user", "content": "follow-up"}) + "\n")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run(["--resume", session_id, "--output-format", "jsonl"])

    assert rc == 0
    mock_client.chat.assert_called_once()
    messages = mock_client.chat.call_args[0][0]
    assert messages[0] == {"role": "system", "content": "sys-prompt"}
    assert {"role": "user", "content": "earlier question"} in messages
    assert {"role": "assistant", "content": "earlier answer"} in messages
    assert messages[-1] == {"role": "user", "content": "follow-up"}

    # The same session file gained the new records.
    lines = [json.loads(line) for line in file_path.read_text().splitlines()]
    assert any(r.get("content") == "follow-up" for r in lines)
    assert any(r.get("role") == "assistant" and "resumed done" in str(r.get("content")) for r in lines)

    # Only the NEW records are streamed; replayed history is not re-emitted.
    out, _ = capsys.readouterr()
    events = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert any(e.get("content") == "follow-up" for e in events)
    assert not any(e.get("content") == "earlier question" for e in events)


def test_resume_unknown_session_id_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    mock_client = Mock()
    stdin = _PipedStdin(json.dumps({"role": "user", "content": "hi"}) + "\n")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run(["--resume", "no-such-session"])

    assert rc != 0
    mock_client.chat.assert_not_called()


def test_resume_rejects_input_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Files would be re-injected into an already-established history; reject the
    combination instead of guessing."""
    monkeypatch.chdir(tmp_path)
    session_id = "20260712T000000-deadbeef-earlier_question"
    _write_session_fixture(tmp_path / ".qi" / "sessions", session_id)

    mock_client = Mock()
    stdin = _PipedStdin("")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run(["--resume", session_id, "some_file.py"])

    assert rc != 0
    mock_client.chat.assert_not_called()


def test_resume_requires_piped_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    session_id = "20260712T000000-deadbeef-earlier_question"
    _write_session_fixture(tmp_path / ".qi" / "sessions", session_id)

    mock_client = Mock()

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.commands.run._is_piped_mode", return_value=False),
    ):
        mock_load.return_value = _piped_settings()

        rc = run(["--resume", session_id])

    assert rc != 0
    mock_client.chat.assert_not_called()


def test_piped_mode_files_only_empty_stdin_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Files-only with an empty pipe and no prompt is an intentional no-op."""
    mock_client = Mock()
    stdin = _PipedStdin("")

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
        patch("qi.lib.session.Session._write"),
        patch("builtins.open", mock_open(read_data="file content")),
    ):
        mock_load.return_value = _piped_settings()
        monkeypatch.setattr("sys.stdin", stdin)

        rc = run(["somefile.py"])

    assert rc == 0
    mock_client.chat.assert_not_called()
