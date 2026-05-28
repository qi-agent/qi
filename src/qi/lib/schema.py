"""JSON schema for structured LLM response."""

from typing import Any

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "description": "Sequence of messages to execute in order",
            "items": {
                "oneOf": [
                    {
                        "description": "Internal reasoning step",
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["thought"]},
                            "content": {
                                "type": "string",
                                "description": "The agent's internal reasoning",
                            },
                        },
                        "required": ["type", "content"],
                    },
                    {
                        "description": "Text output to the user",
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["reply"]},
                            "content": {
                                "type": "string",
                                "description": "Text to output to the user",
                            },
                        },
                        "required": ["type", "content"],
                    },
                    {
                        "description": "Question directed at the user",
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["ask"]},
                            "content": {
                                "type": "string",
                                "description": "Question to ask the user",
                            },
                        },
                        "required": ["type", "content"],
                    },
                    {
                        "description": "Final summary concluding the task",
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["conclusion"]},
                            "content": {
                                "type": "string",
                                "description": "Final summary or result",
                            },
                        },
                        "required": ["type", "content"],
                    },
                ],
            },
        },
    },
    "required": ["messages"],
}

__all__ = ["RESPONSE_SCHEMA"]
