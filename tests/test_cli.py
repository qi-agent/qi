"""Tests for CLI subcommand routing."""

import errno
from unittest.mock import Mock

import pytest

from qi import __version__
from qi.cli import HELP, main


class _FakeStdin:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_no_args_interactive_prints_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", _FakeStdin(tty=True))
    rc = main([])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert out == HELP + "\n"


def test_no_args_piped_routes_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    monkeypatch.setattr("sys.stdin", _FakeStdin(tty=False))
    rc = main([])
    assert rc == 0
    mock_run.assert_called_once_with([])


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


def test_broken_pipe_exits_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the reader closes stdout (e.g. `qi ... | head`), die like a standard
    Unix tool: no traceback, conventional 128+SIGPIPE exit code."""
    mock_run = Mock(side_effect=BrokenPipeError)
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    rc = main(["foo.py"])
    assert rc == 141
    _, err = capsys.readouterr()
    assert "Traceback" not in err


def test_windows_broken_pipe_exits_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On Windows a closed pipe surfaces as OSError(EINVAL), not BrokenPipeError;
    it must get the same quiet exit."""
    mock_run = Mock(side_effect=OSError(errno.EINVAL, "Invalid argument"))
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    monkeypatch.setattr("qi.cli._WINDOWS", True)
    rc = main(["foo.py"])
    assert rc == 141
    _, err = capsys.readouterr()
    assert "Traceback" not in err


def test_einval_still_raises_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside Windows, OSError(EINVAL) is a real error and must not be swallowed."""
    mock_run = Mock(side_effect=OSError(errno.EINVAL, "Invalid argument"))
    monkeypatch.setattr("qi.commands.run.run", mock_run)
    monkeypatch.setattr("qi.cli._WINDOWS", False)
    with pytest.raises(OSError):
        main(["foo.py"])


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
