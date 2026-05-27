from qi.lib.llm_client._google import GoogleLLMClient
from qi.lib.llm_client._openai import OpenAILLMClient
from qi.lib.llm_client._types import LLMResponse, ToolCall


class LLMClient:
    @staticmethod
    def create(
        base_url: str, model: str, *, api_key: str | None = None
    ) -> OpenAILLMClient | GoogleLLMClient:
        clean = base_url.strip().removeprefix("https://").removeprefix("http://").lstrip("/")
        if clean.startswith("generativelanguage.googleapis.com"):
            return GoogleLLMClient(base_url, model, api_key=api_key)
        return OpenAILLMClient(base_url, model, api_key=api_key)


__all__ = ["LLMClient", "LLMResponse", "ToolCall", "GoogleLLMClient", "OpenAILLMClient"]
