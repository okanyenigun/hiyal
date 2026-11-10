from copy import deepcopy
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from typing import Any, Literal, Self
from .thinking import ThinkingConfig, flatten_thinking
from ...prompts.messages import SystemPrompt, UserPrompt
from ...tools.calls.spec import Tools


class LLMConfig(BaseModel):
    """Reusable client defaults or explicitly selected per-call overrides."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str | None = Field(
        default=None,
        description="Model identifier; set at initialization as a default, or per call.",
        examples=["gpt-5.5"],
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key; a per-call value does not replace the stored default.",
        examples=["sk-..."],
    )
    timeout: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Maximum time an HTTP request may wait before failing, in seconds.",
        examples=[30.0],
    )
    max_retries: int | None = Field(
        default=2,
        ge=0,
        description="Maximum automatic retries.",
        examples=[2],
    )
    asynchronous: bool = Field(default=False, description="Async usage")
    client_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific connection settings; fixed at initialization.",
    )
    system_prompt: SystemPrompt | None = Field(
        default=None,
        description="Optional system instructions; accepts a string or SystemPrompt. Omitted from the request when None.",
    )
    user_prompt: UserPrompt | None = Field(
        default=None,
        description="User query; accepts a string or UserPrompt. Required when making a call.",
    )
    tools: Tools | None = Field(
        default=None,
        description=(
            "Tool definitions and tool-choice/parallel-call configuration: a Tools "
            "object, or a plain list of tools (specs or functions) using default "
            "settings. Provider adapters convert these normalized settings to SDK parameters."
        ),
    )
    tool_choice: Literal["auto", "required", "none"] | None = Field(
        default=None,
        description=(
            "Shorthand for Tools.tool_choice. Folded into tools, overriding its value; "
            "unset Tools fields keep their defaults. Needs tools on the config or client."
        ),
        examples=["required"],
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        strict=True,
        description=(
            "Shorthand for Tools.parallel_tool_calls. Folded into tools, overriding its "
            "value. Needs tools on the config or client."
        ),
        examples=[True],
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum visible plus reasoning output tokens.",
        examples=[4096],
    )
    thinking_mode: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^\S+$",
        description="Thinking mode: enabled or disabled, or a supported provider-specific mode such as adaptive.",
        examples=["enabled", "disabled", "adaptive"],
    )
    thinking_effort: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^\S+$",
        description="Thinking effort: low, medium or high, or a supported native level. Ignored when thinking is disabled.",
        examples=["low", "medium", "high"],
    )
    thinking_budget: int | None = Field(
        default=None,
        ge=0,
        strict=True,
        description="Thinking-token budget on supported models; ignored when thinking is disabled.",
        examples=[2048],
    )
    thinking: ThinkingConfig | None = Field(
        default=None,
        description=(
            "Dictionary shorthand for mode, effort and budget, plus provider-specific others. "
            "Common controls are flattened into thinking_mode/effort/budget; direct fields win. "
            "Only native options remain here after validation."
        ),
        examples=[
            {"effort": "medium", "mode": "enabled", "others": {"summary": "auto"}}
        ],
    )
    temperature: float | None = Field(
        default=None, ge=0, description="Sampling temperature", examples=[0.2]
    )
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Nucleus-sampling probability mass",
        examples=[0.2],
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Top-k sampling limit",
        examples=[40],
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Strings that stop text generation.",
        examples=[["\nEND"]],
    )
    seed: int | None = Field(
        default=None,
        description="Best-effort deterministic seed; omitted means random.",
        examples=[42],
    )
    response_modalities: list[Literal["TEXT", "IMAGE", "AUDIO"]] | None = Field(
        default=None,
        description="Requested output modalities supported by the model.",
        examples=[["TEXT"]],
    )
    others: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific generation settings, reusable as defaults or supplied per call.",
    )

    @model_validator(mode="before")
    @classmethod
    def _flatten_thinking(cls, values: Any) -> Any:
        return flatten_thinking(values)

    @field_validator("tools", mode="before")
    @classmethod
    def _parse_tools(cls, value: Any) -> Any:
        """A plain list or tuple of tools becomes Tools with default settings."""
        if isinstance(value, (list, tuple)):
            return Tools(tool_list=list(value))
        return value

    @model_validator(mode="after")
    def _fold_tool_settings(self) -> "LLMConfig":
        """Move tool_choice / parallel_tool_calls into tools once tools are present.

        Only explicitly set Tools fields are carried over, so provider adapters can
        still tell an explicit parallel_tool_calls=False from the default. The
        shorthand fields are then cleared, so they cannot override a later Tools.
        """
        settings = {
            name: getattr(self, name)
            for name in ("tool_choice", "parallel_tool_calls")
            if getattr(self, name) is not None
        }
        if self.tools is None or not settings:
            return self
        kept = {
            name: getattr(self.tools, name)
            for name in self.tools.model_fields_set - {"tool_list"}
        }
        self.tools = Tools(tool_list=self.tools.tool_list, **{**kept, **settings})
        self.tool_choice = None
        self.parallel_tool_calls = None
        return self

    @field_validator(
        "system_prompt", mode="before", json_schema_input_type=str | SystemPrompt | None
    )
    @classmethod
    def _parse_system_prompt(cls, value: Any) -> Any:
        return SystemPrompt(text=value) if isinstance(value, str) else value

    @field_validator(
        "user_prompt", mode="before", json_schema_input_type=str | UserPrompt | None
    )
    @classmethod
    def _parse_user_prompt(cls, value: Any) -> Any:
        return UserPrompt(text=value) if isinstance(value, str) else value

    def isolated_copy(self) -> Self:
        """A deep copy that shares no mutable config data with this one.

        Credentials, callbacks and HTTP clients or transports supplied in
        ``others``, ``client_options`` or ``thinking`` are provider-owned
        resources, not config structures, so the copy shares them instead of
        reconstructing them.
        """
        shared: dict[int, Any] = {}
        pending = [self.others, self.client_options, self.thinking]
        visited: set[int] = set()
        while pending:
            value = pending.pop()
            if id(value) in visited:
                continue
            visited.add(id(value))
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
            elif isinstance(value, BaseModel):
                pending.extend(
                    getattr(value, name) for name in type(value).model_fields
                )
            else:
                shared[id(value)] = value
        return deepcopy(self, shared)
