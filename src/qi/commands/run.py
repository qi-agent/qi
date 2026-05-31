"""Run subcommand: process files with the LLM."""

import argparse
import logging
import time
from typing import Any

from qi.lib.config import load
from qi.lib.context import get_system_prompt
from qi.lib.handler import handle_response
from qi.lib.llm_client import LLMClient
from qi.lib.schema import RESPONSE_SCHEMA
from qi.tools import TOOL_SCHEMAS

CHARS_PER_TOKEN = 4
FILE_READ_HEAD_CHARS = 1024


logger = logging.getLogger(__name__)


def _build_messages(prompt_message: str, file_paths: list[str]) -> list[dict[str, Any]]:
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
        prompt_message = "Analyse the following file(s) then exit."

    prompt_instruction = f"INSTRUCTION: {prompt_message}\n\n" + files_instruction

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt_instruction},
    ]
    for file_path in file_paths:
        try:
            with open(file_path) as f:
                content = f.read(FILE_READ_HEAD_CHARS)
        except OSError as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise

        messages.append({"role": "user", "content": content[:FILE_READ_HEAD_CHARS]})

    return messages


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
        messages = _build_messages(parsed.prompt, parsed.files)
    except Exception as e:
        logger.error(f"Failed to construct messages: {e}")
        return 1

    logger.info(messages)

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
        logger.info(">>>>>>>>>>>>\n" + "\n".join([str(x) for x in messages]))
        try:
            response = client.chat(
                messages,
                tools=TOOL_SCHEMAS,
                response_format=response_format,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return 1

        outputs, done = handle_response(response.content, response.tool_calls)

        if response.content or response.tool_calls:
            assistant_msg = {"role": "assistant", "content": response.content or "", "tool_calls": [tc.as_dict() for tc in response.tool_calls]}
            messages.append(assistant_msg)
        if outputs:
            messages.extend(outputs)

        if done:
            break
        logger.info("Sleeping...")
        time.sleep(1)

    return 0
