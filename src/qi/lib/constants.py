from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"  # OpenAI: tool-generated messagse
    USER = "user"


class RecordType(StrEnum):
    MESSAGE = "message"
    SESSION_END = "session_end"
    SESSION_START = "session_start"


class LogKey(StrEnum):
    API = "api"  # LLM indicates which tool to call
    ID = "id"
    CONTENT = "content"
    META = "meta"
    NAME = "name"  # name of function in OpenAI
    PARAMETERS = "parameters"
    ROLE = "role"
    TIMESTAMP = "timestamp"
    TOOL  = "tool"
    TOOL_CALLS = "tool_calls"  # OpenAI: LLM-generated https://developers.openai.com/api/reference/resources/chat#(resource)%20chat.completions%20%3E%20(model)%20chat_completion_assistant_message_param%20%3E%20(schema)%20%3E%20(property)%20tool_calls
    TOOL_CALL_ID = "tool_call_id"
    TYPE = "type"


class LogMetaKey(StrEnum):
    MODEL = "model"



class MessageKey(StrEnum):
    """Keys in the response message we require from LLM"""
    API = "api"  # Google returns this: '{"type": "call", "api": "default_api:Bash", "parameters": ["git branch --show-current"]}'
    TYPE = "type"
    CONTENT = "content"
    PARAMETERS = "parameters"


class MessageType(StrEnum):
    ASK = "ask"
    CALL = "call"
    CONCLUSION = "conclusion"
    REPLY = "reply"
    THOUGHT = "thought"


# LogRecord: TypeAlias = dict[str, str | dict[str, Any]]
type LogRecord = dict[str, str | dict[str, Any] | list[Any]]
