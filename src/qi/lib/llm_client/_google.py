import logging
from typing import Any, cast

import requests

from qi.lib.constants import LogKey, Role
from qi.lib.llm_client._types import LLMResponse, ToolCall
from qi.lib.schema import RESPONSE_SCHEMA
from qi.lib.utils import make_dict_optional_keys

logger = logging.getLogger(__name__)


def _truncate(obj: object, max_len: int = 10000) -> str:
    s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"... (truncated, {len(s)} total chars)"
    return s


DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
API_URL_TEMPLATE = DEFAULT_BASE_URL + "/v1beta/models/{model}:generateContent"

# Reference: https://ai.google.dev/api

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

    def _build_system_prompt(self, system_messages: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
        if len(system_messages) > 1:
            logger.warning(f"Multiple {len(system_messages)} system prompts found. If they are not contiguous it may confuse the model.")
        return {
            "parts": [
                {"text": m[LogKey.CONTENT].strip()} for m in system_messages
            ]
        }

    def _build_contents(self, messages: list[dict[str, str | dict[str, Any] | list[Any]]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            if role == Role.TOOL:
                tool_response: dict[str, Any] = {
                    "role": "function",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.get(LogKey.NAME, ""),
                                "response": {"result": msg[LogKey.CONTENT]},
                            }
                        }
                    ],
                }
                contents.append(tool_response)
            else:
                extra = cast(dict[str, str], msg.get(LogKey.EXTRA, {}))
                thought_sig_param_dict = make_dict_optional_keys({
                    "thoughtSignature": extra.get("thoughtSignature")
                })
                tool_calls = cast(list[Any], msg.get(LogKey.TOOL_CALLS, []))
                parts: list[dict[str, Any] | None] = [
                    {"text": msg["content"], **thought_sig_param_dict} if msg.get("content") else None,
                    {"functionCall": tool_calls[0], **thought_sig_param_dict} if tool_calls else None,
                ]
                contents.append({
                    "role": "model" if role == "assistant" else "user",  # For google, rename "assistant" to "model"
                    "parts": [part for part in parts if part],
                })
        return contents

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

        generation_config: dict[str, Any] = make_dict_optional_keys({
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "maxOutputTokens": max_tokens or None,
        })
        body: dict[str, Any] = make_dict_optional_keys({
            "system_instruction": self._build_system_prompt(cast(list[dict[str, str]], [m for m in messages if m[LogKey.ROLE] == Role.SYSTEM])),
            "contents": self._build_contents([m for m in messages if m[LogKey.ROLE] != Role.SYSTEM]),
            "generation_config": generation_config,
            "tools": tools,
        })

        url = API_URL_TEMPLATE.format(model=self.model)
        logger.info(">>>>>>>>>>>> Request: POST %s\n%s", url, body)

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
        extra: dict[str, Any] = {}

        for part in parts:
            extra = make_dict_optional_keys({'thoughtSignature': part.get('thoughtSignature')})
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

        return LLMResponse(content=content or "", tool_calls=tool_calls, extra=extra)
