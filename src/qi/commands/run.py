"""Run subcommand: process files with the LLM."""

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from qi.lib.config import Settings, load
from qi.lib.constants import LogKey, LogRecord, Role
from qi.lib.context import get_system_prompt
from qi.lib.handler import handle_response, route_console_output
from qi.lib.llm_client import LLMClient
from qi.lib.schema import RESPONSE_FORMAT
from qi.lib.session import Session
from qi.tools import TOOL_SCHEMAS

CHARS_PER_TOKEN = 4
FILE_READ_HEAD_CHARS = 1024
OUTPUT_FORMAT_TEXT = "text"
OUTPUT_FORMAT_JSONL = "jsonl"


logger = logging.getLogger(__name__)


def _get_session_dir() -> Path:
    return Path('.').resolve() / ".qi" / "sessions"


def _is_piped_mode() -> bool:
    return not sys.stdin.isatty()


def _read_files(file_paths: list[str]) -> list[str]:
    file_messages: list[str] = []
    for file_path in file_paths:
        try:
            with open(file_path) as f:
                content = f.read(FILE_READ_HEAD_CHARS)
        except OSError as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise
        file_messages.append(content)
    return file_messages


def _create_initial_prompt(prompt_message: str, file_paths: list[str]) -> tuple[str, str]:
    files_instruction = ""
    if len(file_paths) == 1:
        files_instruction = (
            f"The following message contains the content (truncated to {FILE_READ_HEAD_CHARS} chars) of the input file '{file_paths[0]}' "
            "relating to this instruction."
        )
    elif len(file_paths) > 1:
        files_instruction = (
            f"The following {len(file_paths)} messages contain the content of the input files relating to this instruction:\n" +
            "\n- ".join(f"- {p}" for p in file_paths)
        )

    if not prompt_message:
        prompt_message = "Analyze the following file(s) then exit."

    prompt_instruction = f"INSTRUCTION: {prompt_message}\n\n" + files_instruction
    slug_hint = f"analyze {Path(file_paths[0]).name}" if file_paths else prompt_instruction
    return prompt_instruction, slug_hint


def _emit_record(record: LogRecord) -> None:
    print(json.dumps(record), flush=True)


def _create_session_from_messages(
    session_dir: Path,
    model: str,
    messages: list[dict[str, Any]],
    slug_hint: str,
    output_format: str = OUTPUT_FORMAT_TEXT,
) -> Session:
    Session.ensure(session_dir)
    session = Session.from_prompt(slug_hint, model, session_dir)
    if output_format == OUTPUT_FORMAT_JSONL:
        session.on_record = _emit_record
    session.log_start(model)
    for message in messages:
        session.log_message(message[LogKey.ROLE], message[LogKey.CONTENT])
    return session


def _create_session(session_dir: Path, model: str, user_prompt: str, file_paths: list[str], file_messages: list[str]) -> Session:
    prompt, slug_hint = _create_initial_prompt(user_prompt, file_paths)

    Session.ensure(session_dir)
    session = Session.from_prompt(slug_hint, model, session_dir)
    session.log_start(model)
    session.log_message(Role.SYSTEM.value, get_system_prompt())
    session.log_message(Role.USER.value, prompt)
    for content in file_messages:
        session.log_message(Role.USER.value, content)
    return session


def _iter_stdin_user_messages(stdin: TextIO) -> Iterator[str]:
    """Yield the content of each user-role JSON message on stdin.

    Non-user roles are skipped (routing). Malformed input — invalid JSON, or valid
    JSON that isn't an object — raises ValueError so the caller can abort cleanly
    rather than crashing with a traceback or silently dropping input.
    """
    for line in stdin:
        raw = line.strip()
        if not raw:
            continue

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON on stdin: {e}: {raw!r}") from e

        if not isinstance(obj, dict):
            raise ValueError(f"Stdin message is not a JSON object: {raw!r}")

        if obj.get(LogKey.ROLE) != Role.USER.value:
            continue

        content = obj.get(LogKey.CONTENT)
        if content is None:
            continue

        yield content


def _files_context_messages(file_paths: list[str], file_messages: list[str]) -> list[dict[str, Any]]:
    """Build the file-context user messages for piped mode.

    A summary message describing the file(s) followed by one message per file's
    content. Unlike _create_initial_prompt, there is no single instruction to tie the
    files to, so the summary omits the "relating to this instruction" tail.
    """
    messages: list[dict[str, Any]] = []
    if not file_paths:
        return messages

    if len(file_paths) == 1:
        summary = (
            f"The following message contains the content (truncated to {FILE_READ_HEAD_CHARS} chars) "
            f"of the input file '{file_paths[0]}'."
        )
    else:
        summary = (
            f"The following {len(file_paths)} messages contain the content of the input files:\n" +
            "\n- ".join(f"- {p}" for p in file_paths)
        )

    messages.append({LogKey.ROLE.value: Role.USER.value, LogKey.CONTENT.value: summary})
    for content in file_messages:
        messages.append({LogKey.ROLE.value: Role.USER.value, LogKey.CONTENT.value: content})
    return messages


def _run_loop(
    session: Session,
    client: Any,
    settings: Any,
    max_iterations: int = 100,
    output_format: str = OUTPUT_FORMAT_TEXT,
) -> int:
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Prompt:\n{session.messages[-1]}")
        try:
            response = client.chat(
                session.messages,
                tools=TOOL_SCHEMAS,
                response_format=RESPONSE_FORMAT,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            logger.error(f"More details may be found in {session.file_path}")
            logger.info(f"[ERR] LLM call failed: {e}", exc_info=True)
            return 1

        if response.content or response.tool_calls:
            if response.content:
                logger.info(f"Response:\n{response.content}")
            if response.tool_calls:
                logger.info(f"Tool calls:\n{response.tool_calls}")
            session.log_message(
                Role.ASSISTANT.value,
                response.content or None,
                tool_calls=[tc.as_dict() for tc in response.tool_calls],
                extra=response.extra,
            )

        outputs, done = handle_response(response.content, response.tool_calls)
        for res in outputs or []:
            if res[LogKey.ROLE] == Role.TOOL:
                session.log_tool_result(res[LogKey.CONTENT], res[LogKey.NAME], res[LogKey.TOOL_CALL_ID])
            else:
                session.log_message(res[LogKey.ROLE], res[LogKey.CONTENT], res.get(LogKey.TOOL_CALLS))
        if done:
            break

    return 0


def _run_piped(
    client: Any,
    settings: Settings,
    prompt: str,
    file_paths: list[str],
    file_messages: list[str],
    output_format: str = OUTPUT_FORMAT_TEXT,
) -> int:
    """Piped stdin mode: one continuing session, one agent-loop iteration per line.

    The system prompt and any file context are logged once as the session's base.
    A --prompt (if given) is logged as a user message but does not trigger an LLM
    round-trip on its own; the first stdin line drives the first iteration, so that
    call sees the prompt and the first line as consecutive user messages. Each
    subsequent line is appended to the same session, preserving history.
    """
    messages: list[dict[str, Any]] = [
        {LogKey.ROLE.value: Role.SYSTEM.value, LogKey.CONTENT.value: get_system_prompt()},
    ]
    messages.extend(_files_context_messages(file_paths, file_messages))

    if prompt:
        slug_hint = prompt
    elif file_paths:
        slug_hint = f"analyze {Path(file_paths[0]).name}"
    else:
        slug_hint = ""

    session = _create_session_from_messages(
        _get_session_dir(), settings.model, messages, slug_hint, output_format=output_format,
    )
    logger.info(f"Session file: {session.file_path}")

    if prompt:
        session.log_message(Role.USER.value, prompt)

    ran_any = False
    try:
        for user_content in _iter_stdin_user_messages(sys.stdin):
            session.log_message(Role.USER.value, user_content)
            ran_any = True
            rc = _run_loop(session, client, settings, output_format=output_format)
            if rc != 0:
                return rc
    except ValueError as e:
        logger.error(f"Aborting piped run: {e}")
        return 1

    # A prompt-only invocation with an empty/closed pipe still does the work.
    # Files-only with empty stdin and no prompt is an intentional no-op.
    if not ran_any and prompt:
        return _run_loop(session, client, settings, output_format=output_format)

    return 0


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qi",
        description="Process files with the LLM",
    )
    parser.add_argument(
        "-p", "--prompt",
        help="User instruction",
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="file",
        help="Files to process",
    )
    parser.add_argument(
        "--output-format",
        choices=[OUTPUT_FORMAT_TEXT, OUTPUT_FORMAT_JSONL],
        default=OUTPUT_FORMAT_TEXT,
        help="Output format: human-readable text or one JSON event per line",
    )
    parsed = parser.parse_args(argv)

    settings = load()
    piped_mode = _is_piped_mode()
    # jsonl reserves stdout for the event stream; humans read stderr. Set explicitly
    # either way so one invocation can't inherit the routing of a previous one.
    route_console_output(to_stderr=piped_mode and parsed.output_format == OUTPUT_FORMAT_JSONL)

    if not parsed.files and not parsed.prompt and not piped_mode:
        logger.error("No input files or prompt provided.")
        return 1

    try:
        file_messages = _read_files(parsed.files)
    except Exception as e:
        logger.error(f"Failed to read files: {e}")
        return 1

    client = LLMClient.create(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
    )

    if piped_mode:
        return _run_piped(
            client, settings, parsed.prompt or "", parsed.files, file_messages,
            output_format=parsed.output_format,
        )

    session = _create_session(_get_session_dir(), settings.model, parsed.prompt, parsed.files, file_messages)
    logger.info(f"Session file: {session.file_path}")
    return _run_loop(session, client, settings)
