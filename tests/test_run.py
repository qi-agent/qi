"""Tests for the run command."""

import re
from unittest.mock import Mock, mock_open, patch

import pytest

from qi.commands.run import run
from qi.lib.config import Settings


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def test_files_sent_as_user_messages() -> None:
    """Files are read and sent as user messages to the LLM."""
    mock_client = Mock()
    mock_client.chat.return_value = '{"result": "ok"}'

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
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

    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "file content"


def test_files_sent_as_user_messages_capsys(capsys: pytest.CaptureFixture[str]) -> None:
    """LLM response is printed to stdout."""
    mock_client = Mock()
    mock_client.chat.return_value = (
        '{"messages": [{"type": "reply", "content": "analysis complete"},'
        ' {"type": "conclusion", "content": ""}]}'
    )

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
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
    assert _strip_ansi(out).strip() == "analysis complete"


def test_prompt_adds_instruction_message() -> None:
    """-p adds an instruction message before file contents."""
    mock_client = Mock()
    mock_client.chat.return_value = "{}"

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
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
    mock_client.chat.return_value = "{}"

    handles = [
        mock_open(read_data="content a").return_value,
        mock_open(read_data="content b").return_value,
    ]

    with (
        patch("qi.commands.run.load") as mock_load,
        patch("qi.commands.run.LLMClient.create", return_value=mock_client),
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
