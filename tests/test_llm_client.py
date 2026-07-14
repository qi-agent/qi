"""Tests for LLM client implementations."""

from unittest.mock import Mock, patch

import requests

from qi.lib.llm_client import LLMClient
from qi.lib.llm_client._google import GoogleLLMClient
from qi.lib.llm_client._openai import OpenAILLMClient


def test_openai_chat_returns_content() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello!", "tool_calls": None}}]
    }

    with patch("qi.lib.llm_client._openai.requests.post", return_value=mock_resp):
        client = LLMClient.create(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        result = client.chat([{"role": "user", "content": "Hi"}])

    assert result.content == "Hello!"
    assert result.tool_calls == []


def test_openai_chat_passes_temperature_and_max_tokens() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello!", "tool_calls": None}}]
    }

    with patch("qi.lib.llm_client._openai.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-test",
        )
        client.chat(
            [{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
        )

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["temperature"] == 0.7
    assert call_kwargs["json"]["max_tokens"] == 100


def test_openai_chat_passes_tools() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": None, "tool_calls": None}}]
    }

    tools = [{"type": "function", "function": {"name": "ReadFile", "parameters": {"type": "object", "properties": {}}}}]

    with patch("qi.lib.llm_client._openai.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-test",
        )
        client.chat(
            [{"role": "user", "content": "Hi"}],
            tools=tools,
        )

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["tools"] == tools


def test_openai_chat_passes_response_format() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "{}", "tool_calls": None}}]
    }

    response_format = {"type": "json_schema", "json_schema": {"name": "test", "schema": {"type": "object"}}}

    with patch("qi.lib.llm_client._openai.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        client.chat(
            [{"role": "user", "content": "Hi"}],
            response_format=response_format,
        )

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["response_format"] == response_format


def test_openai_chat_drops_response_format_when_tools_present() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": None, "tool_calls": None}}]
    }

    tools = [{"type": "function", "function": {"name": "ReadFile", "parameters": {"type": "object", "properties": {}}}}]
    response_format = {"type": "json_schema", "json_schema": {"name": "test", "schema": {"type": "object"}}}

    with patch("qi.lib.llm_client._openai.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        client.chat(
            [{"role": "user", "content": "Hi"}],
            tools=tools,
            response_format=response_format,
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["tools"] == tools
    assert "response_format" not in body


def test_response_schema_is_valid_for_openai_strict_mode() -> None:
    # OpenAI strict structured outputs require additionalProperties: false
    # on every object in the schema, including the root.
    from qi.lib.schema import RESPONSE_SCHEMA

    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert RESPONSE_SCHEMA["properties"]["messages"]["items"]["additionalProperties"] is False


def test_openai_parses_tool_calls() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "function": {
                            "name": "ReadFile",
                            "arguments": '{"path": "test.py", "start": 0}',
                        },
                    }
                ],
            }
        }]
    }

    with patch("qi.lib.llm_client._openai.requests.post", return_value=mock_resp):
        client = LLMClient.create(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        result = client.chat([{"role": "user", "content": "Read test.py"}])

    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_abc"
    assert result.tool_calls[0].name == "ReadFile"
    assert result.tool_calls[0].args == {"path": "test.py", "start": 0}


def test_openai_parses_both_content_and_tool_calls() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"messages": [{"type": "thought", "content": "Let me read"}]}',
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "ReadFile",
                            "arguments": '{"path": "test.py"}',
                        },
                    }
                ],
            }
        }]
    }

    with patch("qi.lib.llm_client._openai.requests.post", return_value=mock_resp):
        client = LLMClient.create(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        result = client.chat([{"role": "user", "content": "Read test.py"}])

    assert result.content is not None
    assert len(result.tool_calls) == 1


def test_google_chat_returns_content() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "AI learns patterns"}]}}],
    }

    with patch("qi.lib.llm_client._google.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://generativelanguage.googleapis.com",
            model="gemini-flash-latest",
            api_key="goog-key",
        )
        result = client.chat([{"role": "user", "content": "Explain AI"}])

    assert result.content == "AI learns patterns"
    assert result.tool_calls == []
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["X-goog-api-key"] == "goog-key"
    assert call_kwargs["json"]["contents"] == [
        {"role": "user", "parts": [{"text": "Explain AI"}]}
    ]


def test_google_chat_system_instruction() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "OK"}]}}],
    }

    with patch("qi.lib.llm_client._google.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://generativelanguage.googleapis.com",
            model="gemini-flash-latest",
            api_key="goog-key",
        )
        client.chat([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ])

    body = mock_post.call_args.kwargs["json"]
    assert body["system_instruction"] == {"parts": [{"text": "Be concise."}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]}
    ]


def test_google_chat_maps_generation_config() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "OK"}]}}],
    }

    with patch("qi.lib.llm_client._google.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://generativelanguage.googleapis.com",
            model="gemini-flash-latest",
            api_key="goog-key",
        )
        client.chat(
            [{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=200,
        )

    body = mock_post.call_args.kwargs["json"]
    gc = body["generation_config"]
    assert gc["temperature"] == 0.5
    assert gc["maxOutputTokens"] == 200
    assert gc["responseMimeType"] == "application/json"
    assert "responseSchema" in gc


def test_google_chat_tools_in_body() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "OK"}]}}],
    }

    tools = [{
        "type": "function",
        "function": {
            "name": "ReadFile",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }]

    with patch("qi.lib.llm_client._google.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://generativelanguage.googleapis.com",
            model="gemini-flash-latest",
            api_key="goog-key",
        )
        client.chat(
            [{"role": "user", "content": "Hi"}],
            tools=tools,
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "ReadFile",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                }
            ],
        },
    ]


def test_google_parses_function_call() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Reading file..."},
                    {"functionCall": {"name": "ReadFile", "args": {"path": "test.py"}}},
                ]
            }
        }],
    }

    with patch("qi.lib.llm_client._google.requests.post", return_value=mock_resp):
        client = LLMClient.create(
            base_url="https://generativelanguage.googleapis.com",
            model="gemini-flash-latest",
            api_key="goog-key",
        )
        result = client.chat([{"role": "user", "content": "Read test.py"}])

    assert result.content == "Reading file..."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "ReadFile"
    assert result.tool_calls[0].args == {"path": "test.py"}


def test_google_chat_tool_result_message() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Result: hello"}]}}],
    }

    with patch("qi.lib.llm_client._google.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient.create(
            base_url="https://generativelanguage.googleapis.com",
            model="gemini-flash-latest",
            api_key="goog-key",
        )
        client.chat([
            {"role": "user", "content": "Read file"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "name": "ReadFile", "content": "file content"},
        ])

    contents = mock_post.call_args.kwargs["json"]["contents"]
    assert contents[-1] == {
        "role": "function",
        "parts": [
            {"functionResponse": {"name": "ReadFile", "response": {"result": "file content"}}}
        ],
    }


def test_factory_returns_openai_client() -> None:
    client = LLMClient.create(
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
    )
    assert isinstance(client, OpenAILLMClient)


def test_factory_returns_google_client() -> None:
    client = LLMClient.create(
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-flash-latest",
        api_key="goog-key",
    )
    assert isinstance(client, GoogleLLMClient)


def test_factory_detects_google_without_scheme() -> None:
    client = LLMClient.create(
        base_url="generativelanguage.googleapis.com",
        model="gemini-flash-latest",
    )
    assert isinstance(client, GoogleLLMClient)
