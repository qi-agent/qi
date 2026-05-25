"""Ping subcommand."""

from __future__ import annotations

import argparse
import logging

from qi.lib.config import load
from qi.lib.llm_client import LLMClient
from qi.lib.logging import output

logger = logging.getLogger(__name__)


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qi ping",
        description="Ping the server",
    )
    parser.parse_args(argv)

    settings = load()

    client = LLMClient.create(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
    )

    logger.info("Sending ping to %s with model %s", settings.base_url, settings.model)

    try:
        response = client.chat(
            [{"role": "user", "content": "Say pong"}],
            temperature=0,
            max_tokens=settings.max_tokens,
        )
    except Exception:
        logger.exception("Ping failed")
        return 1

    output(response)
    return 0
