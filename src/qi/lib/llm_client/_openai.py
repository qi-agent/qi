import json
import logging
from typing import Any

import requests

from qi.lib.llm_client._types import LLMResponse, ToolCall

logger = logging.getLogger(__name__)


def _truncate(obj: object, max_len: int = 5000) -> str:
    s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"... (truncated, {len(s)} total chars)"
    return s


DEFAULT_MODEL = "gpt-4o"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAILLMClient:
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
        self.tools = tools
        self.api_key = api_key

    def chat(
        self,
        messages: list[dict[str, str | dict[str, Any] | list[Any]]],
        response_format: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 0,
        tools: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if tools or self.tools:
            # Never combine tools with response_format: many OpenRouter providers
            # implement strict structured output via grammar-constrained decoding,
            # which makes emitting a tool call impossible.
            body["tools"] = tools or self.tools
            body["tool_choice"] = "auto"
        elif response_format is not None:
            body["response_format"] = response_format

        url = f"{self.base_url}/chat/completions"

        logger.info("[INF] >>>>>>>>>>>> Request: POST %s\n%s", url, body)


        resp = requests.post(url, headers=headers, json=body)
        if not resp.ok:
            logger.info("[ERR] <<<<<<<<<<<< Response: %s %s\n%s", resp.status_code, resp.reason, resp.text)
            resp.raise_for_status()

        choice: dict[str, Any]
        try:
            data: Any = resp.json()
            logger.info(f"[INF] <<< Response:\n{data}")
            choice = data["choices"][0]["message"]
        except Exception as e:
            logger.warning(f"Error decoding API response: {e}")
            logger.info(f"[ERR] Error decoding API response: {e}\nRaw content:\n{resp.text}")
            raise



        content = choice.get("content", "")
        raw_calls: list[Any] = choice.get("tool_calls") or []
        # https://developers.openai.com/api/reference/resources/chat#(resource)%20chat.completions%20%3E%20(model)%20chat_completion_message_tool_call%20%3E%20(schema)
        tool_calls = [
            ToolCall(
                style="openai",
                index=tc.get("index", 0),
                id=tc["id"],
                name=tc["function"]["name"],
                args=json.loads(tc["function"]["arguments"]),  # packaged as JSON - a dictionary
            )
            for tc in raw_calls
        ]
        if tool_calls:
            logger.info(f"Calling {len(tool_calls)} tools: {[x.name for x in tool_calls]}")
        return LLMResponse(content=content, tool_calls=tool_calls)

    def responses_chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        response_format: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 0,
        text_format: str | None = None,
    ) -> LLMResponse:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        input_list: list[dict[str, Any]] = []
        instructions: str | None = None
        for msg in messages:
            if msg["role"] == "system":
                instructions = msg["content"]
            elif msg["role"] == "assistant":
                input_list.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "tool":
                input_list.append({"role": "assistant", "content": msg["content"]})
            else:
                input_list.append({"role": "user", "content": msg["content"]})

        body: dict[str, Any] = {"model": self.model, "input": input_list}
        if instructions is not None:
            body["instructions"] = instructions
        if tools is not None:
            body["tools"] = tools
        text: dict[str, Any] = {}
        if text_format is not None:
            text["format"] = text_format
        if text:
            body["text"] = text

        resp = requests.post(
            f"{self.base_url}/responses",
            headers=headers,
            json=body,
        )
        resp.raise_for_status()
        data: Any = resp.json()
        logger.info("<<< Response:\n%s", _truncate(data))

        content: str | None = None
        tool_calls: list[ToolCall] = []
        for item in data.get("output", []):
            if item["type"] == "message":
                for part in item.get("content", []):
                    if part["type"] == "output_text":
                        content = part["text"]
            elif item["type"] == "function_call":
                tool_calls.append(
                    ToolCall(
                        style="openai",
                        index=item.get("index", 0),
                        id=item["id"],
                        name=item["name"],
                        args=json.loads(item["arguments"]),
                    )
                )
        return LLMResponse(content=content or "", tool_calls=tool_calls)
