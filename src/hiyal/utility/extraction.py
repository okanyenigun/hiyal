import json
from typing import TYPE_CHECKING, TypeVar
from pydantic import BaseModel, ValidationError
from .parsing import Parsing
from ..prompts import SystemPrompt, UserPrompt

if (
    TYPE_CHECKING
):  # only for the annotations: the model layer itself imports this package
    from ..models.bases import BaseLLM
    from ..models.schemas import LLMResponse

Model = TypeVar("Model", bound=BaseModel)


class Extraction:
    """Helpers for extracting structured data from a text with an LLM."""

    @staticmethod
    def extract(
        llm: "BaseLLM", system_prompt: str, text: str, schema: type[Model]
    ) -> Model:
        """``schema`` filled in from ``text`` by one ``llm`` call that follows ``system_prompt``.

        The user prompt holds ``text`` and the JSON schema of ``schema``
        (field names, types and descriptions), and asks for one JSON object
        with every field. A reply wrapped in a Markdown code fence is
        accepted.

        **Every field must be in the answer**, an empty value when the text
        has nothing for it. A field left out would take its default
        silently; for notes being updated, that would wipe what it held.

        Raises ``TypeError`` when ``schema`` is not a Pydantic model class,
        and ``ValueError`` when ``text`` is blank, or the answer is cut off
        (not ``completed``), is not a JSON object, leaves out a field, or
        does not fit ``schema``.
        """
        system, user = Extraction._prompts(system_prompt, text, schema)
        response = llm.query(system_prompt=system, user_prompt=user)
        return Extraction._answer(response, schema)

    @staticmethod
    async def extract_async(
        llm: "BaseLLM", system_prompt: str, text: str, schema: type[Model]
    ) -> Model:
        """``extract``, on the LLM's async client."""
        system, user = Extraction._prompts(system_prompt, text, schema)
        response = await llm.query_async(system_prompt=system, user_prompt=user)
        return Extraction._answer(response, schema)

    @staticmethod
    def _prompts(
        system_prompt: str, text: str, schema: type[BaseModel]
    ) -> tuple[SystemPrompt, UserPrompt]:
        """The system and user prompts of the call; raises when ``schema`` or ``text`` cannot be used (see ``extract``)."""
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(f"schema must be a Pydantic model class, got {schema!r}")
        if not text.strip():
            raise ValueError("there is no text to extract from")
        return SystemPrompt(text=system_prompt), UserPrompt(
            text=Extraction._prompt(text, schema)
        )

    @staticmethod
    def _answer(response: "LLMResponse", schema: type[Model]) -> Model:
        """``schema`` filled in from the LLM's ``response``; raises ``ValueError`` when it cannot be (see ``extract``)."""
        if response.status != "completed":
            raise ValueError(
                f"the answer is {response.status!r}, not 'completed'; a cut-off "
                "answer is not valid JSON (is max_output_tokens too low?)"
            )
        try:
            answer = json.loads(Parsing.unfenced(response.text))
        except json.JSONDecodeError as error:
            raise ValueError(f"the answer is not JSON: {error}") from error
        if not isinstance(answer, dict):
            raise ValueError(
                f"the answer is not a JSON object, but {type(answer).__name__}"
            )
        missing = [
            name
            for name, field in schema.model_fields.items()
            if (field.alias or name) not in answer
        ]
        if missing:
            raise ValueError(f"the answer leaves out fields: {', '.join(missing)}")
        try:
            return schema.model_validate(answer)
        except ValidationError as error:
            raise ValueError(
                f"the answer does not fit {schema.__name__}: {error}"
            ) from error

    @staticmethod
    def _prompt(text: str, schema: type[BaseModel]) -> str:
        """The user prompt: the text, the JSON schema, and what the answer must be."""
        return (
            f"Text:\n{text}\n\n"
            f"JSON schema:\n{json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=False)}\n\n"
            "Answer with one JSON object that fits the schema, with every one of its "
            "fields (an empty value when the text has nothing for a field), and nothing else."
        )
