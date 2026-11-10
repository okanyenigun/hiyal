from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar
from pydantic import ValidationError
from ..schemas.fields import NOT_GENERATION_FIELDS, config_aliases
from ..schemas import (
    MessageRole,
    UnsupportedThinkingError,
    ProviderInfo,
    FinishReason,
    LLMConfig,
    LLMResponse,
    ResponseContent,
    ResponseMessage,
    ResponseStatus,
    ResponseTiming,
    ThinkingConfig,
)
from ...tools.calls.spec import Tools, ToolSpec
from ...utility import Dicts


if TYPE_CHECKING:
    from .stream_accumulator import StreamAccumulator

# UserPrompt attachment fields, in the order they are reported.
MEDIA_FIELDS = ("images", "audio", "documents")


class Translator(ABC):
    """Converts between Hiyal's common types and one provider SDK's formats.

    Every provider subclasses it once. It is the only place that knows how a
    provider spells its request and how it shapes its answer, so ``BaseLLM``
    can treat all providers the same. It works in two directions:

    **Request: ``LLMConfig`` → SDK keyword arguments**

    - ``validate`` rejects a call the provider cannot serve before anything is
      sent (a missing user prompt, or attachments it does not accept).
    - ``build_request`` produces the SDK arguments: it renames fields to the
      provider's spelling, adds tools, and applies thinking settings. Providers
      extend it with ``super().build_request(config)`` and add their own
      message and media format.
    - ``adapt_thinking`` / ``apply_thinking``, ``tool_parameters`` and
      ``tool_definitions`` are the hooks and helpers used for thinking and
      tools.

    **Response: SDK output → Hiyal types**

    - ``parse_response`` (abstract) turns a complete SDK response into an
      ``LLMResponse``.
    - ``stream_accumulator`` (abstract) returns the per-request object that
      turns streamed SDK events into normalized stream events.
    - ``finish_reason``, ``status``, ``calculated_timing``, ``messages`` and
      ``text`` are shared helpers for building that response.

    A translator holds no connection or request state, so it can be tested
    without an SDK client. Sending requests belongs to ``Transport``, and
    merging settings into an ``LLMConfig`` belongs to ``ConfigResolver``.

    Args:
        info: The provider's static facts (name, accepted media, adapters).
    """

    # Whether the provider accepts a thinking token budget (``thinking_budget``).
    supports_thinking_budget: ClassVar[bool] = True

    def __init__(self, info: ProviderInfo) -> None:
        """Keep the provider facts and prepare the common-to-provider field renames."""
        self.info = info
        self._provider_names = {
            common: provider
            for provider, common in config_aliases(info.generation_adapter).items()
        }

    # Request ---------------------------------------------------------------

    def validate(self, config: LLMConfig) -> None:
        """Check that the call can be expressed for this provider before anything is sent.

        The user prompt is required. Attachments outside ``prompt_media``
        raise instead of being dropped silently.
        """
        user = config.user_prompt
        if user is None:
            raise ValidationError.from_exception_data(
                "LLMConfig", [{"type": "missing", "loc": ("user_prompt",), "input": {}}]
            )
        unsupported = [
            name
            for name in MEDIA_FIELDS
            if getattr(user, name) and name not in self.info.prompt_media
        ]
        if unsupported:
            raise NotImplementedError(
                f"{self.info.display_name} does not support user_prompt {', '.join(unsupported)}"
            )

    def build_request(self, config: LLMConfig) -> dict[str, Any]:
        """Convert a resolved config into SDK keyword arguments.

        Providers extend this by calling ``super().build_request(config)`` and
        adjusting the result.
        """
        values = config.model_dump(
            exclude_none=True, exclude=set(NOT_GENERATION_FIELDS)
        )
        others = values.pop("others", {})
        renamed = {
            self._provider_names.get(name, name): value
            for name, value in values.items()
        }
        params = {**others, **renamed}
        if config.tools is not None:
            params.update(self.tool_parameters(config.tools))
        if config.thinking is not None or any(
            value is not None
            for value in (
                config.thinking_mode,
                config.thinking_effort,
                config.thinking_budget,
            )
        ):
            thinking = ThinkingConfig(
                mode=config.thinking_mode,
                effort=config.thinking_effort,
                budget=config.thinking_budget,
                others=config.thinking.others if config.thinking is not None else {},
            )
            params = self.adapt_thinking(params, thinking, config.model)
        return params

    def adapt_thinking(
        self, params: dict[str, Any], thinking: ThinkingConfig, model: str | None
    ) -> dict[str, Any]:
        """Return fresh SDK parameters with the thinking controls applied.

        Runs the checks every provider shares, then ``apply_thinking``.
        Unsupported controls raise; they are never replaced by weaker ones.
        """
        result = dict(params)
        if thinking.mode == "enabled" and thinking.effort == "none":
            raise ValueError("thinking.mode='enabled' conflicts with effort='none'")
        if (
            thinking.budget is not None
            and thinking.mode != "disabled"
            and not self.supports_thinking_budget
        ):
            raise UnsupportedThinkingError(
                self.info.provider,
                model or "",
                "thinking_budget is not supported by this provider",
            )
        # Model names are matched without a "vendor/" prefix, case-insensitively.
        self.apply_thinking(result, thinking, (model or "").rsplit("/", 1)[-1].lower())
        return result

    def apply_thinking(
        self, params: dict[str, Any], thinking: ThinkingConfig, model: str
    ) -> None:
        """Write the provider's native thinking settings into ``params`` in place."""
        raise ValueError(f"No thinking adapter for provider {self.info.provider!r}")

    def tool_parameters(self, tools: Tools) -> dict[str, Any]:
        """Translate typed tool settings; providers that support tools override this."""
        raise NotImplementedError(
            f"{self.info.display_name} does not support normalized tool configuration yet"
        )

    def tool_definitions(
        self, tools: Tools, spec_type: type[ToolSpec]
    ) -> list[dict[str, Any]]:
        """Convert tool specs into SDK tool declarations, rejecting tools written for another provider."""
        declarations = []
        for tool in tools.tool_list:
            if tool._provider not in {None, self.info.provider}:
                raise ValueError(
                    f"{self.info.display_name} tools must use ToolSpec or {spec_type.__name__}"
                )
            spec = (
                tool
                if isinstance(tool, spec_type)
                else spec_type(**tool.model_dump(exclude_unset=True))
            )
            declarations.append(spec.to_sdk_tool())
        return declarations

    def prompt_texts(self, config: LLMConfig) -> tuple[str | None, str]:
        """System text (``None`` when not given) and user text, after ``validate``."""
        self.validate(config)
        assert config.user_prompt is not None  # guaranteed by validate
        system = config.system_prompt
        return (None if system is None else system.text), config.user_prompt.text

    def chat_messages(self, config: LLMConfig) -> list[dict[str, Any]]:
        """System (when given) and user prompts as a chat-style message list."""
        system, user = self.prompt_texts(config)
        messages: list[dict[str, Any]] = (
            [] if system is None else [{"role": "system", "content": system}]
        )
        return [*messages, {"role": "user", "content": user}]

    def require_text_only(self, params: dict[str, Any]) -> None:
        """Drop ``response_modalities``, raising unless it asks for text only."""
        modalities = params.pop("response_modalities", None)
        if modalities not in (None, ["TEXT"]):
            raise ValueError(
                f"{self.info.display_name} only supports the TEXT response modality"
            )

    def dict_param(
        self,
        params: dict[str, Any],
        key: str,
        *,
        pop: bool = False,
        label: str | None = None,
    ) -> dict[str, Any]:
        """``params[key]`` as a dict (default ``{}``), with this provider named in errors."""
        return Dicts.get(
            params, key, owner=self.info.display_name, pop=pop, label=label
        )

    # Response --------------------------------------------------------------

    @abstractmethod
    def parse_response(
        self,
        response: Any,
        *,
        model: str | None = None,
        created_at: datetime | None = None,
        elapsed_seconds: float | None = None,
    ) -> LLMResponse:
        """Convert an SDK response into an ``LLMResponse``.

        ``model`` is the requested model, ``created_at`` the request time and
        ``elapsed_seconds`` the client-side wall time; use them only where the
        response lacks its own values.
        """

    @abstractmethod
    def stream_accumulator(self, config: LLMConfig) -> "StreamAccumulator[Any]":
        """Return fresh state for one streamed request."""

    @staticmethod
    def finish_reason(
        reason: str | None,
        finish_reasons: dict[str, FinishReason],
        *,
        default: FinishReason = "unknown",
        contents: Iterable[ResponseContent] = (),
    ) -> FinishReason | None:
        """Map a provider finish reason; ``None`` means the provider gave none.

        Providers that stop the same way for answers and tool calls get
        ``tool_call`` when ``contents`` holds one.
        """
        if reason is None:
            return None
        finish_reason = finish_reasons.get(reason, default)
        if finish_reason == "stop" and any(
            content.content_type == "tool_call" for content in contents
        ):
            return "tool_call"
        return finish_reason

    @staticmethod
    def status(
        reason: str | None, incomplete_reasons: set[str], *, interrupted: bool = False
    ) -> ResponseStatus:
        """``incomplete`` when cut short or stopped for an incomplete reason."""
        if interrupted or reason in incomplete_reasons:
            return "incomplete"
        return "completed"

    @staticmethod
    def calculated_timing(elapsed_seconds: float | None) -> ResponseTiming | None:
        """Client-side wall time, for providers that report no timing."""
        if elapsed_seconds is None:
            return None
        return ResponseTiming(source="calculated", total_seconds=elapsed_seconds)

    @staticmethod
    def messages(
        ids: list[str],
        contents: list[ResponseContent],
        *,
        finish_reason: FinishReason | None,
        role: MessageRole = "assistant",
    ) -> list[ResponseMessage]:
        """One message per content part; reasoning parts get no finish reason."""
        return [
            ResponseMessage(id=message_id, role="other", content=content)
            if content.content_type == "reasoning"
            else ResponseMessage(
                id=message_id,
                role=role,
                content=content,
                finish_reason=finish_reason,
            )
            for message_id, content in zip(ids, contents, strict=True)
        ]

    @staticmethod
    def text(contents: list[ResponseContent]) -> str:
        """Visible output: the text parts joined in order."""
        return "".join(
            content.text or "" for content in contents if content.content_type == "text"
        )
