from pydantic import BaseModel, Field
from typing import Literal


class TokenUsage(BaseModel):
    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Prompt tokens consumed (Ollama: prompt_eval_count).",
        examples=[25],
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Generated tokens as reported by the provider. OpenAI includes "
            "reasoning tokens here; Gemini reports them separately."
        ),
        examples=[758],
    )
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Tokens spent on reasoning/thinking, when reported.",
        examples=[640],
    )
    total_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Total tokens; may be derived as input + output when absent.",
        examples=[783],
    )
    cache_read_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Input tokens served from the provider's prompt cache.",
        examples=[0],
    )
    cache_write_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Input tokens written to the provider's prompt cache.",
        examples=[0],
    )


class ResponseTiming(BaseModel):
    source: Literal["provider", "calculated"] = Field(
        description=(
            "'provider' when the provider reported the durations itself; "
            "'calculated' when this library derived them, either from provider "
            "timestamps or by measuring the call on the client."
        ),
        examples=["calculated"],
    )
    total_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Total time taken for the request.",
        examples=[1.23],
    )
    load_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Time taken to load the model.",
        examples=[0.1],
    )
    input_evaluation_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Time taken to evaluate the input.",
        examples=[0.5],
    )
    output_evaluation_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Time taken to evaluate the output.",
        examples=[0.63],
    )
