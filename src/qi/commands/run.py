"""Run subcommand: process files with the LLM."""

import argparse
import logging
from pathlib import Path
from typing import Any

from qi.lib.config import load
from qi.lib.constants import LogKey, Role
from qi.lib.context import get_system_prompt
from qi.lib.handler import handle_response
from qi.lib.llm_client import LLMClient
from qi.lib.schema import RESPONSE_SCHEMA
from qi.lib.session import Session
from qi.tools import TOOL_SCHEMAS

CHARS_PER_TOKEN = 4
FILE_READ_HEAD_CHARS = 1024


logger = logging.getLogger(__name__)


def _read_files(file_paths: list[str]) -> list[str]:
    file_messages: list[str] = []
    for file_path in file_paths:
        try:
            with open(file_path) as f:
                content = f.read(FILE_READ_HEAD_CHARS)
        except OSError as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise
        file_messages.append(content[:FILE_READ_HEAD_CHARS])
    return file_messages


def _build_messages(prompt_message: str, file_paths: list[str], file_messages: list[str]) -> list[dict[str, Any]]:
    files_instruction = ""
    if len(file_paths) == 1:
        files_instruction = (
            f"The following message contains the contents (truncated to {FILE_READ_HEAD_CHARS} chars) of the input file '{file_paths[0]}' "
            "relating to this instruction."
        )
    elif len(file_paths) > 1:
        files_instruction = (
            f"The following {len(file_paths)} messages contain the contents of the input files relating to this instruction:\n" +
            "\n- ".join(f"- {p}" for p in file_paths)
        )

    if not prompt_message:
        prompt_message = "Analyze the following file(s) then exit."

    prompt_instruction = f"INSTRUCTION: {prompt_message}\n\n" + files_instruction

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt_instruction},
    ]
    for content in file_messages:
        messages.append({"role": "user", "content": content})

    return messages


def _create_initial_prompt(prompt_message: str, file_paths: list[str]) -> tuple[str, str]:
    files_instruction = ""
    if len(file_paths) == 1:
        files_instruction = (
            f"The following message contains the contents (truncated to {FILE_READ_HEAD_CHARS} chars) of the input file '{file_paths[0]}' "
            "relating to this instruction."
        )
    elif len(file_paths) > 1:
        files_instruction = (
            f"The following {len(file_paths)} messages contain the contents of the input files relating to this instruction:\n" +
            "\n- ".join(f"- {p}" for p in file_paths)
        )

    if not prompt_message:
        prompt_message = "Analyze the following file(s) then exit."

    prompt_instruction = f"INSTRUCTION: {prompt_message}\n\n" + files_instruction
    slug_hint = f"analyze {Path(file_paths[0]).name}" if file_paths else prompt_instruction
    return prompt_instruction, slug_hint


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


def _get_session_dir() -> Path:
    return Path('.').resolve() / ".qi"/ "sessions"


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
    parsed = parser.parse_args(argv)

    settings = load()

    if not parsed.files and not parsed.prompt:
        logger.error("No input files or prompt provided.")
        return 1

    try:
        file_messages = _read_files(parsed.files)
    except Exception as e:
        logger.error(f"Failed to read files: {e}")
        return 1

    session = _create_session(_get_session_dir(), settings.model, parsed.prompt, parsed.files, file_messages)
    print(session.file_path)
    client = LLMClient.create(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
    )

    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "qi_response",
            "strict": True,
            "schema": RESPONSE_SCHEMA,
        },
    }

    while True:
        logger.info(">>>>>>>>>>>>\n" + "\n".join([str(x) for x in session.messages]))
        try:
            response = client.chat(
                session.messages,
                tools=TOOL_SCHEMAS,
                response_format=response_format,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return 1

        if response.content or response.tool_calls:
            session.log_message(
                Role.ASSISTANT.value,
                response.content or "",
                tool_calls=[tc.as_dict() for tc in response.tool_calls],
            )

        outputs, done = handle_response(response.content, response.tool_calls)
        for tool_res in outputs or []:
            session.log_tool_result(tool_res[LogKey.CONTENT], tool_res[LogKey.NAME], tool_res[LogKey.TOOL_CALL_ID])

        if done:
            break

    return 0
