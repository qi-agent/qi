"""JSON schema for structured LLM response and tool definitions."""

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

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ReadFile",
            "description": "Read contents of a file from the local filesystem by line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "start": {
                        "type": "integer",
                        "description": "Character index to start reading from (0-based); defaults to 0",
                    },
                    "end": {
                        "type": "integer",
                        "description": "Character index to stop at (exclusive); omit to read to end of file",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


def _openai_tool_to_google(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool["function"]
    return {"functionDeclarations": [{"name": fn["name"], "description": fn.get("description", ""), "parameters": fn["parameters"]}]}


GOOGLE_TOOLS: list[dict[str, Any]] = [_openai_tool_to_google(t) for t in OPENAI_TOOLS]

__all__ = ["RESPONSE_SCHEMA", "OPENAI_TOOLS", "GOOGLE_TOOLS"]
