import logging
from typing import Any

import requests

from qi.lib.llm_client._types import LLMResponse, ToolCall
from qi.lib.schema import RESPONSE_SCHEMA

logger = logging.getLogger(__name__)


def _truncate(obj: object, max_len: int = 10000) -> str:
    s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"... (truncated, {len(s)} total chars)"
    return s


DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


class GoogleLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        tools: list[dict[str, object]] | None = None,
        *,
        api_key: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tools = tools or []
        self.api_key = api_key

    @staticmethod
    def _format_tool_declarations(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, object]] = []
        for t in tools:
            fn: dict[str, Any] = t.get("function", {})
            assert isinstance(fn, dict)
            result.append({
                "functionDeclarations": [
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                ],
            })
        return result

    def chat(
        self,
        messages: list[dict[str, str | dict[str, Any] | list[Any]]],
        response_format: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 0,
        tools: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        if tools or self.tools:
            tools = self._format_tool_declarations(tools or self.tools)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-goog-api-key"] = self.api_key

        contents: list[dict[str, Any]] = []
        system_instruction: dict[str, Any] | None = None

        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_instruction = {"parts": [{"text": msg["content"]}]}
            elif role == "tool":
                tool_response: dict[str, Any] = {
                    "role": "function",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.get("name", ""),
                                "response": {"result": msg["content"]},
                            }
                        }
                    ],
                }
                contents.append(tool_response)
            else:
                contents.append({
                    "role": "model" if role == "assistant" else "user",  # For google, rename "assistant" to "model"
                    "parts": [{"text": msg["content"]}],
                })

        body: dict[str, Any] = {"contents": contents}
        if system_instruction is not None:
            body["system_instruction"] = system_instruction

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        }
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens
        body["generationConfig"] = generation_config

        if tools is not None:
            body["tools"] = tools

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        logger.info(">>>>>>>>>>>> Request: POST %s\n%s", url, _truncate(body))

        resp = requests.post(url, headers=headers, json=body, timeout=300)
        if not resp.ok:
            logger.error("<<<<<<<<<<<< Response: %s %s\n%s", resp.status_code, resp.reason, _truncate(resp.text))
            resp.raise_for_status()
        data: Any = resp.json()
        logger.info("<<<<<<<<<<<< Response:\n%s", _truncate(data))

        candidate = data["candidates"][0]
        parts: list[Any] = candidate["content"]["parts"]

        content: str | None = None
        tool_calls: list[ToolCall] = []
        for part in parts:
            if "text" in part:
                content = part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        style="google",
                        index=0,
                        id=fc.get("id", ""),
                        name=fc["name"],
                        args=dict(fc.get("args", {})),
                    )
                )

        return LLMResponse(content=content or "", tool_calls=tool_calls)
