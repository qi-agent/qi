"""Tests for CLI subcommand routing."""

from unittest.mock import Mock

import pytest

from qi import __version__
from qi.cli import HELP, main


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert out == HELP + "\n"


def test_help_flag_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--help"])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert out == HELP + "\n"


def test_h_flag_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["-h"])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert out == HELP + "\n"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--version"])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert out == f"qi {__version__}\n"


def test_ping_routes_to_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.ping.run", mock_run)
    rc = main(["ping"])
    assert rc == 0
    mock_run.assert_called_once_with([])


def test_ping_passes_args(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.ping.run", mock_run)
    rc = main(["ping", "--some-flag"])
    assert rc == 0
    mock_run.assert_called_once_with(["--some-flag"])


def test_run_explicit_routes_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    rc = main(["run", "foo.py"])
    assert rc == 0
    mock_run.assert_called_once_with(["foo.py"])


def test_run_explicit_passes_all_args(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    rc = main(["run", "-p", "hello", "foo.py"])
    assert rc == 0
    mock_run.assert_called_once_with(["-p", "hello", "foo.py"])


def test_run_implicit_file_routes_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    rc = main(["bar.py"])
    assert rc == 0
    mock_run.assert_called_once_with(["bar.py"])


def test_run_implicit_with_prompt_routes_to_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    rc = main(["-p", "hello", "foo.py"])
    assert rc == 0
    mock_run.assert_called_once_with(["-p", "hello", "foo.py"])


def test_run_implicit_passes_unknown_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    rc = main(["--bogus", "foo.py"])
    assert rc == 0
    mock_run.assert_called_once_with(["--bogus", "foo.py"])


def test_return_value_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock(return_value=42)
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    rc = main(["foo.py"])
    assert rc == 42


def test_ping_return_value_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock(return_value=99)
    monkeypatch.setattr("qi.commands.ping.run", mock_run)
    rc = main(["ping"])
    assert rc == 99
