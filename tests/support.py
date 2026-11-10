"""Shared test helpers: real provider SDKs over mocked HTTP, and canned payloads.

No sockets are opened. ``make_llm`` builds a provider LLM whose SDK client
sends every request to ``respond(request)`` and records it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import httpx2

from hiyal.models import AnthropicLLM, DeepSeekLLM, GeminiLLM, OllamaLLM, OpenAILLM

run = asyncio.run

# The OpenAI-family and Anthropic SDKs use httpx2; Google GenAI and Ollama use httpx.
HTTP = {OpenAILLM: httpx2, AnthropicLLM: httpx2, DeepSeekLLM: httpx2, GeminiLLM: httpx, OllamaLLM: httpx}
DEFAULT_MODEL = {
    OpenAILLM: "gpt-5.5",
    AnthropicLLM: "claude-sonnet-5-5",
    DeepSeekLLM: "deepseek-chat",
    GeminiLLM: "gemini-2.5-flash",
    OllamaLLM: "qwen3",
}
PROVIDERS = (OpenAILLM, AnthropicLLM, DeepSeekLLM, GeminiLLM, OllamaLLM)


def make_llm(
    cls: type,
    respond: Callable[[Any], Any],
    *,
    asynchronous: bool = False,
    client_options: dict[str, Any] | None = None,
    **settings: Any,
) -> tuple[Any, list[Any]]:
    """A provider LLM over a mock HTTP transport; returns ``(llm, requests)``."""
    http = HTTP[cls]
    requests: list[Any] = []

    def handler(request: Any) -> Any:
        requests.append(request)
        return respond(request)

    transport = http.MockTransport(handler)
    options = dict(client_options or {})
    if cls is OllamaLLM:
        options.setdefault("transport", transport)
    else:
        if cls is GeminiLLM:
            # One genai client serves both modes, so both HTTP clients are mocked.
            http_options = dict(options.pop("http_options", {}))
            http_options["httpx_client"] = http.Client(transport=transport)
            http_options["httpx_async_client"] = http.AsyncClient(transport=transport)
            http_options.setdefault("retry_options", {"initial_delay": 0.001, "max_delay": 0.001})
            options["http_options"] = http_options
        else:
            options["http_client"] = (http.AsyncClient if asynchronous else http.Client)(transport=transport)
    values = {"model": DEFAULT_MODEL[cls], "api_key": "test-key", "max_retries": 0, **settings}
    llm = cls(asynchronous=asynchronous, client_options=options, **values)
    return llm, requests


def build_request(llm: Any, config: Any = None, **parameters: Any) -> dict[str, Any]:
    """The SDK arguments a query would send, without sending anything.

    A default user prompt is added unless one is given, by any spelling.
    """
    from hiyal.models.schemas.fields import config_aliases

    aliases = config_aliases(llm.info.generation_adapter)
    given = any(aliases.get(name, name) == "user_prompt" for name in parameters)
    if not given and (config is None or config.user_prompt is None):
        parameters["user_prompt"] = "Hi"
    call = llm.resolver.call_config(llm.config, config, parameters)
    llm.translator.validate(call)
    return llm.translator.build_request(call)


def body(request: Any) -> Any:
    """The JSON body of a recorded request."""
    return json.loads(request.content) if request.content else None


def auth(request: Any) -> str | None:
    """The credential a recorded request carried, whatever header the provider uses."""
    for name in ("x-goog-api-key", "x-api-key"):
        if name in request.headers:
            return request.headers[name]
    value = request.headers.get("authorization")
    return value.removeprefix("Bearer ") if value else None


def sse(events: list[dict[str, Any]], *, named: bool = True, done: bool = False) -> str:
    """A server-sent-events body; ``named`` adds ``event:`` lines from each ``type``."""
    text = "".join(
        (f"event: {event['type']}\n" if named and "type" in event else "")
        + "data: " + json.dumps(event) + "\n\n"
        for event in events
    )
    return text + ("data: [DONE]\n\n" if done else "")


def ndjson(chunks: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(chunk) + "\n" for chunk in chunks)


def respond_json(http: Any, payload: Any, status: int = 200) -> Callable[[Any], Any]:
    return lambda request: http.Response(status, json=payload)


# ---------------------------------------------------------------- OpenAI --

def openai_usage(**overrides: Any) -> dict[str, Any]:
    return {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 1, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 2},
        **overrides,
    }


def openai_response(output: list[dict[str, Any]] | None = None, *, status: str = "completed",
                    incomplete_reason: str | None = None, **overrides: Any) -> dict[str, Any]:
    payload = {
        "id": "resp_1", "object": "response", "created_at": 1792499155, "completed_at": 1792499157,
        "model": "gpt-5.5", "status": status, "output": output or [],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "usage": openai_usage(),
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
    }
    payload.update(overrides)
    return payload


def openai_message(text: str = "Hello.", item_id: str = "msg_1") -> dict[str, Any]:
    return {"type": "message", "id": item_id, "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def openai_function_call(index: int = 0, *, name: str | None = "get_weather",
                         arguments: str = '{"city":"Istanbul"}', status: str | None = "completed") -> dict[str, Any]:
    item = {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
            "name": name, "arguments": arguments}
    if status is not None:
        item["status"] = status
    return item


# ------------------------------------------------------ Chat Completions --

def chat_completion(message: dict[str, Any] | None = None, *, finish_reason: str | None = "stop",
                    usage: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    payload = {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1792499155, "model": "chat-model",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": "Hello.", **(message or {})}}],
        "usage": usage if usage is not None else {"prompt_tokens": 30, "completion_tokens": 9, "total_tokens": 39},
    }
    payload.update(overrides)
    return payload


def chat_chunk(delta: dict[str, Any] | None = None, *, finish_reason: str | None = None,
               usage: dict[str, Any] | None = None, index: int = 0, choices: bool = True) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1792499155, "model": "chat-model",
        "choices": [{"index": index, "delta": delta or {}, "finish_reason": finish_reason}] if choices else [],
        "usage": usage,
    }


# ------------------------------------------------------------- Anthropic --

def anthropic_message(content: list[dict[str, Any]] | None = None, *, stop_reason: str | None = "end_turn",
                      usage: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    payload = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": content if content is not None else [{"type": "text", "text": "Hello."}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": usage or {"input_tokens": 5, "output_tokens": 7},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------- Gemini --

def gemini_response(parts: list[dict[str, Any]] | None = None, *, finish_reason: str | None = "STOP",
                    usage: dict[str, Any] | None = None, candidates: list[dict[str, Any]] | None = None,
                    **overrides: Any) -> dict[str, Any]:
    if candidates is None:
        candidate: dict[str, Any] = {"index": 0, "content": {"role": "model", "parts": parts or [{"text": "Hello."}]}}
        if finish_reason is not None:
            candidate["finishReason"] = finish_reason
        candidates = [candidate]
    payload = {
        "responseId": "g1", "modelVersion": "gemini-2.5-flash", "candidates": candidates,
        "usageMetadata": usage or {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------- Ollama --

def ollama_chat(message: dict[str, Any] | None = None, *, done: bool = True, done_reason: str | None = "stop",
                **overrides: Any) -> dict[str, Any]:
    payload = {
        "model": "qwen3", "created_at": "2026-01-01T00:00:00Z", "done": done,
        "message": {"role": "assistant", "content": "Hello.", **(message or {})},
        "prompt_eval_count": 3, "eval_count": 4,
        "total_duration": 1_000_000_000, "load_duration": 10, "prompt_eval_duration": 20, "eval_duration": 30,
    }
    if done_reason is not None:
        payload["done_reason"] = done_reason
    payload.update(overrides)
    return payload
