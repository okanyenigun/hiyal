<p align="center">
  <img src="https://raw.githubusercontent.com/okanyenigun/hiyal/master/docs/assets/logo.png" alt="Hiyal" width="160">
</p>

# Hiyal

One Python interface for OpenAI, Anthropic, Gemini, Ollama and DeepSeek.

Write your request once, then switch provider by changing one word. Responses,
streaming events, tool calls, token usage and errors all come back in the same shape.

> **Early release (0.0.x).** The API may still change.

## Install

```bash
pip install hiyal
```

Python 3.12 or newer. Each provider reads its usual environment variable
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`), or pass `api_key=...`.
Ollama needs no key; it talks to a local server.

## Quickstart

```python
from hiyal.models import LLM

llm = LLM(provider="openai", model="gpt-5.5")
response = llm.query(system_prompt="You are concise.", user_prompt="Say hello.")

print(response.text)
print(response.usage.input_tokens, response.usage.output_tokens)
```

Same code, different provider:

```python
llm = LLM(provider="anthropic", model="claude-sonnet-5-5")
```

## Streaming

```python
for event in llm.query_stream(user_prompt="Tell me a short story."):
    if event.event_type == "content.delta":
        print(event.delta.text, end="")
    elif event.event_type == "response.completed":
        print("\n", event.response.usage)
```

## Tools

Plain functions work as tools; Hiyal builds the schema from the type hints and docstring.
It never runs your function: it returns the calls the model asked for.

```python
from hiyal.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city.

    Args:
        city: The city to look up.
    """
    return f"Sunny in {city}"

response = llm.query(user_prompt="Weather in Istanbul?", tools=[get_weather])
for call in response.tool_calls:
    print(call.name, call.arguments)   # get_weather {'city': 'Istanbul'}
```

## Async

```python
llm = LLM(provider="openai", model="gpt-5.5", asynchronous=True)
response = await llm.query_async(user_prompt="Hi")
```

## Thinking, images, audio, files

```python
llm.query(user_prompt="Solve this.", thinking_effort="low")   # translated per provider

from hiyal.prompts import UserPrompt
llm.query(user_prompt=UserPrompt(text="What is in this picture?", images="chart.png"))
```

| Provider  | Images | Documents | Audio in | Files API |
| --------- | :----: | :-------: | :------: | :-------: |
| OpenAI    |   ✓    |     ✓     |    ✓     |     ✓     |
| Anthropic |   ✓    |     ✓     |          |     ✓     |
| Gemini    |   ✓    |     ✓     |    ✓     |     ✓     |
| Ollama    |   ✓    |           |          |           |
| DeepSeek  |   ✓    |           |          |           |

Anything a provider does not support raises a clear error instead of being dropped silently.

## Development

```bash
uv sync
uv run pytest
uvx ruff check .
```

The tests use the real provider SDKs against mocked HTTP, so they need no API keys and no network.

## License

[MIT](LICENSE) © Okan Yenigün
