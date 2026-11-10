"""OpenAI Chat Completions message parts and constants (DeepSeek reuses them)."""

import base64
from typing import Any, Literal, cast
from ..schemas import (
    FinishReason,
    ResponseContent,
    ToolCallContent,
    ToolCallDeltaContent,
)
from ...prompts.media import sniff_audio_mime_type
from ...tools.calls.responses import ToolCall, ToolCallDelta
from ...utility import Identifiers


CHAT_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_call",
    # DeepSeek: the server ran out of capacity and cut generation short.
    "insufficient_system_resource": "other",
}

# Chat Completions input_audio formats, by MIME type.
CHAT_AUDIO_FORMATS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}

CHAT_INCOMPLETE_FINISH_REASONS = {"length", "insufficient_system_resource"}


class ChatContents:
    """Builds the ordered response parts of one Chat Completions result.

    A Chat Completions message packs reasoning, text, spoken audio and tool
    calls into a single object, and a stream delivers that object in many
    small deltas. Hiyal wants each of those as its own response message with a
    stable ID and position. This class does the splitting and keeps the running
    state, and it behaves the same for one full message or for many deltas, so
    a streamed answer and a non-streamed answer come out identical.

    How the parts are formed:

    - Reasoning, text and audio each become one part, created the first time
      that kind appears. Later deltas are appended to the same part.
    - Each tool call becomes one part. Its first fragment carries the call ID
      and name, and later fragments add argument text to a buffer, matched by
      the delta's call index.
    - Parts are ordered by first appearance and given generated IDs, since the
      API assigns none.

    Use ``apply`` for every message or delta, then ``final_contents`` once the
    result is complete. The instance is created per response, or per stream.

    Attributes:
        contents: The parts so far, in order.
        ids: One generated ID per part, in the same order as ``contents``.
    """

    def __init__(self) -> None:
        """Start with no parts; tracking is filled in by ``apply``."""
        self.contents: list[ResponseContent] = []
        # Chat Completions assigns no IDs to message parts; one is generated per part.
        self.ids: list[str] = []
        self._text_index: dict[str, int] = {}
        self._tool_index: dict[int, int] = {}
        # Per tool-call part: the argument text so far, and the first fragment's call item.
        self._arguments: dict[int, str] = {}
        self._raw_calls: dict[int, Any] = {}

    def _add(self, content: ResponseContent) -> int:
        """Append a new part with a generated ID and return its index."""
        self.contents.append(content)
        self.ids.append(Identifiers.generate_id("item"))
        return len(self.contents) - 1

    def apply(self, message: Any) -> list[tuple[int, ResponseContent]]:
        """Merge a message or stream delta into the parts and return what is new.

        Each returned ``(index, content)`` pair is the fragment added to the
        part at ``index``; the stream accumulator turns these into delta events.
        """
        fragments: list[tuple[int, ResponseContent]] = []
        parts: tuple[tuple[Literal["reasoning", "text"], str | None], ...] = (
            ("reasoning", getattr(message, "reasoning_content", None)),
            ("text", message.content),
        )
        for content_type, text in parts:
            if not text:
                continue
            index = self._text_index.get(content_type)
            if index is None:
                index = self._text_index[content_type] = self._add(
                    ResponseContent(content_type=content_type, text="")
                )
            current = self.contents[index]
            self.contents[index] = current.model_copy(
                update={"text": (current.text or "") + text}
            )
            fragments.append(
                (index, ResponseContent(content_type=content_type, text=text))
            )

        fragments.extend(self._apply_audio(getattr(message, "audio", None)))

        # Stream deltas number their tool calls; a full message lists them.
        for position, call in enumerate(message.tool_calls or []):
            key = getattr(call, "index", None)
            key = position if key is None else key
            arguments = (call.function.arguments or "") if call.function else ""
            index = self._tool_index.get(key)
            if index is None:
                # The first fragment carries the call's id and name.
                name = call.function.name if call.function else None
                index = self._tool_index[key] = self._add(
                    self._tool_call(name, arguments, call, complete=False)
                )
                self._arguments[index] = arguments
                self._raw_calls[index] = call
                fragments.append((index, self._fragment(name, arguments)))
            elif arguments:
                self._arguments[index] += arguments
                fragments.append((index, self._fragment(None, arguments)))
        return fragments

    @staticmethod
    def _tool_call(
        name: str | None, arguments: str, raw: Any, *, complete: bool
    ) -> ToolCallContent:
        """A typed call from a name and JSON argument text."""
        return ToolCallContent(
            data=ToolCall.normalize(
                name=name,
                arguments=arguments,
                raw=raw,
                complete=complete,
                json_encoded=True,
            )
        )

    @staticmethod
    def _fragment(name: str | None, arguments: str) -> ToolCallDeltaContent:
        """A stream fragment: the call's name and/or more argument text."""
        return ToolCallDeltaContent(
            data=ToolCallDelta(name=name, arguments_delta=arguments or None)
        )

    def _apply_audio(self, audio: Any) -> list[tuple[int, ResponseContent]]:
        """Merge spoken output into one audio part and return the new fragment.

        Base64 chunks are decoded and joined into bytes, and transcript pieces
        are concatenated; a delta with neither adds nothing.
        """
        if audio is None:
            return []
        read = (
            audio.get
            if isinstance(audio, dict)
            else (lambda name: getattr(audio, name, None))
        )
        chunk = base64.b64decode(read("data") or "")
        transcript = read("transcript") or ""
        if not chunk and not transcript:
            return []
        index = self._text_index.get("audio")
        if index is None:
            index = self._text_index["audio"] = self._add(
                ResponseContent(
                    content_type="audio",
                    data={"mime_type": "audio/pcm", "data": b"", "transcript": ""},
                )
            )
        current = cast(dict[str, Any], self.contents[index].data)
        data = current["data"] + chunk
        self.contents[index] = ResponseContent(
            content_type="audio",
            data={
                # The format is whatever the request asked for; streamed PCM has no header.
                "mime_type": sniff_audio_mime_type(data) or "audio/pcm",
                "data": data,
                "transcript": current["transcript"] + transcript,
            },
        )
        return [
            (
                index,
                ResponseContent(
                    content_type="audio", data={"data": chunk, "transcript": transcript}
                ),
            )
        ]

    def final_contents(self, *, complete: bool) -> list[ResponseContent]:
        """The finished parts, with each tool call built from its buffered argument text.

        Call this once the message or stream is over. ``complete`` says whether
        generation ended normally: tool calls are ``incomplete`` otherwise, and
        ``invalid`` when their name or JSON arguments cannot be used. Other
        parts are returned unchanged.
        """
        return [
            self._tool_call(
                content.data.name,
                self._arguments[index],
                self._raw_calls[index],
                complete=complete,
            )
            if index in self._arguments and isinstance(content, ToolCallContent)
            else content
            for index, content in enumerate(self.contents)
        ]
