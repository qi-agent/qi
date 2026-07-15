"""Tests for LLM client implementations."""

from unittest.mock import Mock, patch

import requests

from qi.lib.llm_client import LLMClient
from qi.lib.llm_client._google import GoogleLLMClient, _sanitize_schema
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


def test_sanitize_schema_strips_additional_properties_recursively() -> None:
    schema = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["reply", "question"]},
                    },
                    "required": ["type"],
                    "additionalProperties": False,
                },
                "description": "Sequence of messages",
            },
        },
        "required": ["messages"],
        "additionalProperties": False,
    }

    result = _sanitize_schema(schema)

    assert "additionalProperties" not in result
    items = result["properties"]["messages"]["items"]
    assert "additionalProperties" not in items
    assert items["required"] == ["type"]
    assert items["properties"]["type"]["enum"] == ["reply", "question"]
    assert result["properties"]["messages"]["description"] == "Sequence of messages"
    # the input (a shared module constant in production) must not be mutated
    assert schema["additionalProperties"] is False
    assert schema["properties"]["messages"]["items"]["additionalProperties"] is False  # type: ignore[index]


def test_google_chat_response_schema_has_no_additional_properties() -> None:
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
        client.chat([{"role": "user", "content": "Hi"}])

    def assert_clean(node: object) -> None:
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for value in node.values():
                assert_clean(value)
        elif isinstance(node, list):
            for item in node:
                assert_clean(item)

    schema = mock_post.call_args.kwargs["json"]["generation_config"]["responseSchema"]
    assert_clean(schema)


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


def test_google_parses_thought_signatures_per_tool_call() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "functionCall": {"name": "Skill", "args": {"name": "caveman"}, "id": "a"},
                        "thoughtSignature": "sig-abc",
                    },
                    {"functionCall": {"name": "Bash", "args": {"command": "git status"}, "id": "b"}},
                    {"functionCall": {"name": "Bash", "args": {"command": "git diff"}, "id": "c"}},
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
        result = client.chat([{"role": "user", "content": "explain dirty files"}])

    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].extra == {"thoughtSignature": "sig-abc"}
    assert result.tool_calls[1].extra == {}
    assert result.tool_calls[2].extra == {}
    # signature survives serialization to the session format
    assert result.tool_calls[0].as_dict()["thoughtSignature"] == "sig-abc"
    assert "thoughtSignature" not in result.tool_calls[1].as_dict()


def test_google_parses_text_part_thought_signature() -> None:
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "thinking done", "thoughtSignature": "text-sig"},
                    {"functionCall": {"name": "Bash", "args": {"command": "ls"}, "id": "x"}},
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
        result = client.chat([{"role": "user", "content": "list files"}])

    # a signature-less later part must not clobber the text part's signature
    assert result.extra == {"thoughtSignature": "text-sig"}
    assert result.tool_calls[0].extra == {}


def test_google_replays_all_tool_calls_with_signatures() -> None:
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
            {"role": "user", "content": "explain dirty files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "name": "Skill", "args": {"name": "caveman"}, "thoughtSignature": "sig-abc"},
                    {"id": "b", "name": "Bash", "args": {"command": "git status"}},
                    {"id": "c", "name": "Bash", "args": {"command": "git diff"}},
                ],
            },
            {"role": "tool", "name": "Skill", "content": "CAVEMAN MODE"},
            {"role": "tool", "name": "Bash", "content": "M file.py"},
            {"role": "tool", "name": "Bash", "content": "diff output"},
        ])

    contents = mock_post.call_args.kwargs["json"]["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            {
                "functionCall": {"id": "a", "name": "Skill", "args": {"name": "caveman"}},
                "thoughtSignature": "sig-abc",
            },
            {"functionCall": {"id": "b", "name": "Bash", "args": {"command": "git status"}}},
            {"functionCall": {"id": "c", "name": "Bash", "args": {"command": "git diff"}}},
        ],
    }


def test_google_replays_text_message_level_signature() -> None:
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
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "extra": {"thoughtSignature": "text-sig"}},
            {"role": "user", "content": "continue"},
        ])

    contents = mock_post.call_args.kwargs["json"]["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [{"text": "hello", "thoughtSignature": "text-sig"}],
    }


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
