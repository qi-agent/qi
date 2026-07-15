"""JSON schema for structured LLM response."""

from typing import Any

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "description": "Sequence of messages to execute in order",
            "items": {
                "description": "Reply or question to the user",
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["reply", "question"]},
                    "content": {"type": "string"},
                    "done": {
                        "type": "boolean",
                        "description": "Set to true if the task is concluded, or unable to proceed"
                    }
                },
                "required": ["type", "content", "done"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["messages"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "qi_response",
        "strict": True,
        "schema": RESPONSE_SCHEMA,
    },
}


__all__ = [
    "RESPONSE_FORMAT",
    "RESPONSE_SCHEMA",
]
