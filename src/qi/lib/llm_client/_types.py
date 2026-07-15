import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    style: str = "openai"  # or "google"
    index: int = 0
    id: str = ""
    name: str = ""
    args: dict[str, object] | list[object] = field(default_factory=dict[str, object])
    extra: dict[str, object] = field(default_factory=dict[str, object])

    def as_dict(self) -> dict[str, object]:
        if self.style == "openai":
            return self.as_dict_openai()
        if self.style == "google":
            return self.as_dict_google()
        else:
            raise NotImplementedError(f"Unsupported style: {self.style}")

    def as_dict_openai(self) -> dict[str, object]:
        return {
            "index": self.index,
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.args),
            },
        }

    def as_dict_google(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "name": self.name,
            "args": self.args,
        }
        if self.extra.get("thoughtSignature"):
            result["thoughtSignature"] = self.extra["thoughtSignature"]
        return result

@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    extra: dict[str, Any] = field(default_factory=dict[str, Any])
    # Normalized reason the model stopped generating: "stop", "tool_calls",
    # "length" (truncated by max_tokens), or "" when the provider omitted it.
    finish_reason: str = ""


__all__ = ["LLMResponse", "ToolCall"]
