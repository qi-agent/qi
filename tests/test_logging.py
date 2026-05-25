import logging
from unittest.mock import patch

import pytest
from rich.text import Text

from qi.lib.logging import QiLogFormatter, QiLogHandler, _StyledLogValue, console, output


class TestOutput:
    def test_str(self) -> None:
        with patch("qi.lib.logging.console.print") as mock_print:
            output("hello")
        mock_print.assert_called_once_with("hello", markup=False)

    def test_text(self) -> None:
        with patch("qi.lib.logging.console.print") as mock_print:
            output(Text("hello"))
        mock_print.assert_called_once_with(Text("hello"), markup=False)

    def test_markup(self) -> None:
        with patch("qi.lib.logging.console.print") as mock_print:
            output("[bold]hi[/]", markup=True)
        mock_print.assert_called_once_with("[bold]hi[/]", markup=True)

    def test_empty_string(self) -> None:
        with patch("qi.lib.logging.console.print") as mock_print:
            output("")
        mock_print.assert_called_once_with("", markup=False)


class TestStyledLogValue:
    def test_valid_style_renders_styled_text(self) -> None:
        v = _StyledLogValue("test", "logging.message")
        result = str(v)
        assert "test" in result

    def test_missing_style_returns_raw_value(self) -> None:
        v = _StyledLogValue("test", "nonexistent.style")
        assert str(v) == "test"

    def test_empty_value(self) -> None:
        v = _StyledLogValue("", "logging.message")
        assert str(v) == ""


class TestQiLogFormatter:
    def test_format_with_message(self) -> None:
        fmt = QiLogFormatter("%(message)s")
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "hello", (), None)
        result = fmt.format(record)
        assert "hello" in result

    def test_format_includes_levelname(self) -> None:
        fmt = QiLogFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "msg", (), None)
        result = fmt.format(record)
        assert "INFO" in result
        assert "msg" in result


class TestQiLogHandlerInit:
    def test_default_level(self) -> None:
        handler = QiLogHandler()
        assert handler.level == logging.NOTSET

    def test_custom_level(self) -> None:
        handler = QiLogHandler(level=logging.INFO)
        assert handler.level == logging.INFO

    def test_attr_styles_defaults_to_none(self) -> None:
        handler = QiLogHandler()
        assert handler.attr_styles is None

    def test_rich_formatter_starts_none(self) -> None:
        handler = QiLogHandler()
        assert handler.rich_formatter is None


class TestQiLogHandlerFormat:
    def test_non_terminal_uses_standard_formatter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(console, "_force_terminal", False)
        handler = QiLogHandler()
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "plain msg", (), None)
        result = handler.format(record)
        assert "\x1b[" not in result
        assert "plain msg" in result

    def test_terminal_uses_rich_formatter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(console, "_force_terminal", True)
        handler = QiLogHandler()
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "rich msg", (), None)
        result = handler.format(record)
        assert "rich msg" in result

    def test_lazily_creates_formatters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(console, "_force_terminal", False)
        handler = QiLogHandler()
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "msg", (), None)
        handler.format(record)
        assert handler.formatter is not None

    def test_reuses_cached_formatters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(console, "_force_terminal", False)
        handler = QiLogHandler()
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "msg", (), None)
        fmt1 = handler.formatter
        handler.format(record)
        fmt2 = handler.formatter
        handler.format(record)
        fmt3 = handler.formatter
        assert fmt1 is None
        assert fmt2 is not None
        assert fmt2 is fmt3

    def test_reuses_cached_rich_formatter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(console, "_force_terminal", True)
        handler = QiLogHandler()
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "msg", (), None)
        handler.format(record)
        rf1 = handler.rich_formatter
        handler.format(record)
        rf2 = handler.rich_formatter
        assert rf1 is not None
        assert rf1 is rf2

    def test_set_formatter_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(console, "_force_terminal", False)
        handler = QiLogHandler()
        custom_fmt = logging.Formatter("%(message)s CUSTOM")
        handler.setFormatter(custom_fmt)
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "msg", (), None)
        result = handler.format(record)
        assert result == "msg CUSTOM"


class TestQiLogHandlerEmit:
    def test_emit_calls_output_with_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(console, "_force_terminal", False)
        handler = QiLogHandler()
        record = logging.LogRecord("test", logging.INFO, "file.py", 42, "emit test", (), None)
        with patch("qi.lib.logging.output") as mock_output:
            handler.emit(record)
        mock_output.assert_called_once()
        args, _ = mock_output.call_args
        assert isinstance(args[0], Text)
        assert "emit test" in str(args[0])
