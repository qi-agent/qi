import json
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    style: str = "openai"  # or "google"
    index: int = 0
    id: str = ""
    name: str = ""
    args: dict[str, object] | list[object] = field(default_factory=dict[str, object])

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
        # may not be needed, as this is inline only
        return {
            "type": "call",
            "api": f"default_api:{self.name}",
            "parametres": json.dumps(self.args),
        }

@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])


__all__ = ["LLMResponse", "ToolCall"]
