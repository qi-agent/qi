import datetime
import json
import logging
import re
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

from qi.lib.constants import (
    LogKey,
    LogMetaKey,
    LogRecord,
    RecordType,
    Role,
)
from qi.lib.exceptions import CorruptedFileError

ARTICLES = {
    "a",
    "an",
    "the",
}
TURN_ASSISTANT = "assistant"
TURN_USER = "user"

logger = logging.getLogger(__name__)


def make_slug(prompt: str) -> str:
    # Use the first few words of the prompt as a slug
    words = re.findall(r'\w+', prompt.lower())
    # Filter for stop words
    words = [w for w in words if w not in ARTICLES]
    if not words:
        return "session"
    slug = "_".join(words[:5])
    return slug


class Session:
    def __init__(self, session_id: str, model: str, session_dir: Path):
        self.model = model
        self.session_id = session_id
        self.file_path = session_dir / f"{self.session_id}.jsonl"
        self.turn = TURN_ASSISTANT
        self._messages: list[LogRecord] = []

    @classmethod
    def ensure(cls, session_dir: Path) -> None:
        # Ensure directory exists
        session_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def generate_session_id(cls, prompt: str, timestamp: datetime.datetime | None = None) -> str:
        timestamp = timestamp or datetime.datetime.now()
        slug = make_slug(prompt)
        session_id = f"{timestamp:%Y%m%d%H%M%S}-{uuid.uuid4().hex}-{slug}"
        return session_id

    @classmethod
    def from_prompt(cls, prompt: str, model: str, session_dir: Path) -> "Session":
        session_id = cls.generate_session_id(prompt)
        session = cls(session_id, model=model, session_dir=session_dir)
        return session

    @classmethod
    def from_session_file(cls, file_path: Path) -> "Session":
        log_lines = cls._read_log_lines(file_path)
        log_session_start = next(log_lines)
        if ev_type := log_session_start[LogKey.TYPE] != RecordType.SESSION_START:
            raise CorruptedFileError(f"Session log '{file_path}' does not begin with a session start event, instead I got {ev_type}; full line:\n"
                                     f"{log_session_start}")

        session_id = file_path.stem
        meta = cast(dict[str, Any], log_session_start[LogKey.META])
        session = Session(session_id, meta[LogMetaKey.MODEL], session_dir=file_path.parent)
        for _, log_entry in enumerate(log_lines, 1):
            session.log_record(
                type_=cast(str, log_entry[LogKey.TYPE]),
                role=cast(str, log_entry[LogKey.ROLE]),
                content=cast(str | None, log_entry.get(LogKey.CONTENT)),
                tool_name=cast(str | None, log_entry.get(LogKey.TOOL)),
                tool_call_id=cast(str | None, log_entry.get(LogKey.TOOL_CALL_ID)),
                tool_calls=cast(list[dict[str, Any]] | None, log_entry.get(LogKey.TOOL_CALLS)),
                meta=cast(dict[str, Any] | None, log_entry.get(LogKey.META)),
                write=False,  # update state only
            )

        return session

    @property
    def messages(self) -> list[LogRecord]:
        return [r for r in self._messages if r[LogKey.TYPE] == RecordType.MESSAGE]

    def _update_state(
            self,
            type_: str,
            role: str = "",
            content: str | None = None,
            tool_name: str | None = None,
            meta: dict[str, Any] | None = None,
    ) -> None:
        meta = meta or {}
        if model := meta.get(LogMetaKey.MODEL, ""):
            self.model = model

        if type_ == RecordType.MESSAGE:
            if role == Role.ASSISTANT:
                self.turn = TURN_USER
            else:
                self.turn = TURN_ASSISTANT

        return

    @staticmethod
    def _read_log_lines(file_path: Path) -> Generator[LogRecord]:
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = cast(LogRecord, json.loads(line))
                    yield data
                except json.JSONDecodeError as e:
                    logger.exception(f"Error loading session file (line {i}): {e}")

    def _write(self, data: dict[str, Any]) -> None:
        with self.file_path.open("a") as f:
            f.write(json.dumps(data) + "\n")

    def log_record(
            self,
            type_: str,
            role: str = "",
            content: str | None = None,
            tool_calls: list[dict[str, Any]] | None = None,
            tool_call_id: str | None = None,
            tool_name: str | None = None,
            meta: dict[str, Any] | None = None,
            write: bool = True,
    ) -> None:
        """Logs a record with a given role and content/tool name."""
        record: LogRecord = {
            LogKey.TYPE.value: type_,
            LogKey.TIMESTAMP.value: datetime.datetime.now().isoformat(),
        }
        for (k, v) in [
            (LogKey.ROLE.value, role),
            (LogKey.CONTENT.value, content),
            (LogKey.TOOL_CALL_ID.value, tool_call_id),
            (LogKey.TOOL_CALLS.value, tool_calls),
            (LogKey.META.value, meta),
            (LogKey.NAME.value, tool_name),
        ]:
            if v is not None:
                record[k] = v

        self._messages.append(record)
        if write:
            self._write(record)

        self._update_state(
            type_=type_,
            role=role,
            content=content,
            tool_name=tool_name,
            meta=meta,
        )

    def log_message(self, role: str, content: str | None = None, tool_calls: list[dict[str, Any]] | None = None) -> None:
        return self.log_record(
            type_=RecordType.MESSAGE.value,
            role=role,
            content=content,
            tool_calls=tool_calls,
        )

    def log_tool_result(self, content: str, tool_name: str, tool_call_id: str = "") -> None:
        return self.log_record(
            type_=RecordType.MESSAGE.value,
            role=Role.TOOL,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

    def log_start(self, model: str) -> None:
        return self.log_record(
            type_=RecordType.SESSION_START.value,
            meta={
                LogMetaKey.MODEL.value: model,
            },
        )
