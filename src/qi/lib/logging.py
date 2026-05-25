import logging
from typing import cast, override

from rich.console import Console
from rich.errors import MissingStyle
from rich.text import Text
from rich.theme import Theme

DEFAULT_THEME = {
    "logging.asctime": "bright_cyan",
    "logging.message": "none",
    "logging.funcName": "dim",
    "logging.name": "dim",
    "logging.lineno": "dim",
}
console = Console(soft_wrap=True, theme=Theme(DEFAULT_THEME))


def output(s: str | Text, markup: bool = False) -> None:
    console.print(s, markup=markup)


class _StyledLogValue:
    def __init__(self, value: str, style_name: str):
        self._value = value
        self._style_name = style_name

    def __str__(self) -> str:
        try:
            style = console.get_style(self._style_name)
        except MissingStyle:
            return self._value
        return style.render(self._value)


class QiLogFormatter(logging.Formatter):
    def _rich_format(self, style: logging.PercentStyle, record: logging.LogRecord) -> str:
        defaults = cast(dict[str, str], style._defaults or {})
        values = defaults | record.__dict__
        styled_values = {k: _StyledLogValue(str(v), f"logging.{k}") for k, v in values.items()}
        styled_values["levelname"] = _StyledLogValue(
            record.levelname.ljust(8), f"logging.level.{record.levelname.lower()}"
        )
        return style._fmt % styled_values

    @override
    def formatMessage(self, record: logging.LogRecord) -> str:
        return self._rich_format(self._style, record)


class QiLogHandler(logging.Handler):
    DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s.%(funcName)s:%(lineno)s %(message)s"

    def __init__(
        self,
        level: int = logging.NOTSET,
        formatter: logging.Formatter | None = None,
        attr_styles: dict[str, str] | None = None,
        level_styles: dict[str, str] | None = None,
    ):
        super().__init__(level=level)
        self.attr_styles = attr_styles
        self.rich_formatter: QiLogFormatter | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """Output the record"""
        msg = self.format(record)
        output(Text.from_ansi(msg))

    def format(self, record: logging.LogRecord) -> str:
        # mimick what super().format() does
        if self.formatter:
            fmt = self.formatter
        else:
            fmt = self.formatter = logging.Formatter(fmt=self.DEFAULT_FORMAT)

        if self.rich_formatter:
            rich_fmt = self.rich_formatter
        else:
            rich_fmt = self.rich_formatter = QiLogFormatter(fmt=fmt._fmt)

        if console.is_terminal:
            return rich_fmt.format(record)
        else:
            return fmt.format(record)
