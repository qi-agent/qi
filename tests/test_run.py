"""Tests for the run command."""

import io
import json
import re
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
