from typing import Any

from pydantic import BaseModel


class AskUserParams(BaseModel):
    question: str


class AskUserTool:
    """Ask the user a clarifying question; the tool result is their answer.

    The response handler intercepts this tool by name so it can prompt on the
    console (interactive) or defer to the next stdin message (piped). Calling
    it directly is a fallback that must never block on stdin.
    """

    name = "AskUser"
    description = (
        "Ask the user a clarifying question and receive their answer as the tool result. "
        "Use this whenever you need input or a decision from the user; never end a reply "
        "with a question in plain text."
    )
    params = AskUserParams

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params.model_json_schema(),
            },
        }

    def __call__(self, question: str) -> str:
        params = self.params(question=question)
        return f"[Question relayed to the user: {params.question}]"
