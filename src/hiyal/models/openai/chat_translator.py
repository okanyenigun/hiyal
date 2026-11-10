"""OpenAI Chat Completions translation (DeepSeek reuses it)."""

from datetime import datetime, timezone
from typing import Any
from .chat_contents import (
    CHAT_AUDIO_FORMATS,
    CHAT_FINISH_REASONS,
    CHAT_INCOMPLETE_FINISH_REASONS,
    ChatContents,
)
from ..bases import StreamAccumulator, Translator
from ..schemas import (
    LLMConfig,
    LLMResponse,
    LLMStreamEvent,
    ResponseTerminalEvent,
    ResponseTiming,
    TokenUsage,
)
from ...prompts.messages import UserPrompt


class ChatCompletionsTranslator(Translator):
    """Parses OpenAI Chat Completions responses and streams, and builds their user content.

    ``OpenAITranslator`` holds one and hands it the requests the Responses API
    cannot carry (audio in or out). ``DeepSeekTranslator`` inherits from it,
    because DeepSeek's API is the same Chat Completions format.
    """

    def parse_response(
        self,
        response: Any,
        *,
        model: str | None = None,
        created_at: datetime | None = None,
        elapsed_seconds: float | None = None,
    ) -> LLMResponse:
        contents = ChatContents()
        choice = response.choices[0] if response.choices else None
        if choice is not None:
            contents.apply(choice.message)
        return self.build_chat_response(
            response_id=response.id,
            model=response.model,
            created=response.created,
            contents=contents,
            finish_reason=None if choice is None else choice.finish_reason,
            usage=response.usage,
            # Chat Completions reports no timing: this is the client-side wall time.
            timing=self.calculated_timing(elapsed_seconds),
            raw=response,
        )

    def stream_accumulator(self, config: LLMConfig) -> "ChatStreamAccumulator":
        return ChatStreamAccumulator(translator=self, model=config.model)

    def build_chat_response(
        self,
        *,
        response_id: str,
        model: str,
        created: int | None,
        contents: ChatContents,
        finish_reason: str | None,
        usage: Any,
        raw: Any,
        interrupted: bool = False,
        timing: ResponseTiming | None = None,
    ) -> LLMResponse:
        """An ``LLMResponse`` from merged message parts; used for full and streamed responses."""
        complete = (
            not interrupted
            and finish_reason is not None
            and finish_reason not in CHAT_INCOMPLETE_FINISH_REASONS
        )
        final_contents = contents.final_contents(complete=complete)
        return LLMResponse(
            text=self.text(final_contents),
            messages=self.messages(
                contents.ids,
                final_contents,
                finish_reason=self.finish_reason(finish_reason, CHAT_FINISH_REASONS),
            ),
            provider=self.info.provider,
            id_=response_id,
            model=model,
            created_at=datetime.fromtimestamp(created, tz=timezone.utc)
            if created
            else datetime.now(timezone.utc),
            status=self.status(
                finish_reason, CHAT_INCOMPLETE_FINISH_REASONS, interrupted=interrupted
            ),
            usage=self.parse_chat_usage(usage),
            timing=timing,
            raw=raw,
        )

    @staticmethod
    def parse_chat_usage(usage: Any) -> TokenUsage | None:
        if usage is None:
            return None
        completion_details = usage.completion_tokens_details
        prompt_details = usage.prompt_tokens_details
        cache_read = getattr(usage, "prompt_cache_hit_tokens", None)
        if cache_read is None and prompt_details is not None:
            cache_read = prompt_details.cached_tokens
        return TokenUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            reasoning_tokens=None
            if completion_details is None
            else completion_details.reasoning_tokens,
            total_tokens=usage.total_tokens,
            cache_read_tokens=cache_read,
        )

    @staticmethod
    def chat_user_content(prompt: UserPrompt) -> list[dict[str, Any]]:
        """Chat Completions content parts: text, images, files, then audio."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt.text}]
        for image in prompt.images:
            if image.file_id is not None:
                raise ValueError(
                    "Chat Completions cannot take an image by file_id; pass it inline or by URL"
                )
            content.append(
                {"type": "image_url", "image_url": {"url": image.url or image.data_url}}
            )
        for document in prompt.documents:
            if document.file_id is not None:
                content.append({"type": "file", "file": {"file_id": document.file_id}})
            elif document.url is not None:
                raise ValueError(
                    "Chat Completions cannot take a document by URL; pass it inline or upload it"
                )
            else:
                content.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": document.name,
                            "file_data": document.data_url,
                        },
                    }
                )
        for audio in prompt.audio:
            audio_format = CHAT_AUDIO_FORMATS.get(audio.mime_type)
            if audio_format is None:
                raise ValueError(
                    f"Chat Completions audio input must be WAV or MP3, got {audio.mime_type!r}"
                )
            if audio.data is None:
                raise ValueError(
                    "Chat Completions audio input must be inline data, not a URL or file ID"
                )
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio.base64, "format": audio_format},
                }
            )
        return content


class ChatStreamAccumulator(StreamAccumulator[ChatCompletionsTranslator]):
    """Per-request state; the last chunks carry the finish reason and usage."""

    def __init__(
        self, *, translator: ChatCompletionsTranslator, model: str | None
    ) -> None:
        super().__init__(translator=translator, model=model)
        self.created: int | None = None
        self.contents = ChatContents()
        self.finish_reason: str | None = None
        self.usage: Any = None

    def _adapt(self, event: Any) -> list[LLMStreamEvent]:
        chunk = event
        self.response_id = chunk.id or self.response_id
        self.model = chunk.model or self.model
        self.created = self.created or chunk.created
        self.usage = chunk.usage or self.usage

        events: list[LLMStreamEvent] = []
        for choice in chunk.choices:
            # Alternative choices remain on each chunk's raw object.
            if choice.index != 0:
                continue
            self.finish_reason = choice.finish_reason or self.finish_reason
            events.extend(
                self._delta(
                    message_index=index,
                    message_id=self.contents.ids[index],
                    content=content,
                )
                for index, content in self.contents.apply(choice.delta)
            )
        return events

    def finalize(self) -> tuple[ResponseTerminalEvent, ...]:
        if self.raw is None:
            return ()
        response = self.translator.build_chat_response(
            response_id=self.response_id or "",
            model=self.model or "",
            created=self.created,
            contents=self.contents,
            # A stream that ends without a finish reason was cut short.
            finish_reason=self.finish_reason,
            interrupted=self.finish_reason is None,
            usage=self.usage,
            raw=self.raw,
        )
        return self._terminal(response)
