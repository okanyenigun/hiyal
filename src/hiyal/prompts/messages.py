from typing import Any
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from .media import Audio, Document, Image, Media


class Prompt(BaseModel):
    text: str = Field(description="Text")
    kind: str = Field(default="base", description="type of prompt")


class SystemPrompt(Prompt):
    kind: str = Field(default="system", description="type of prompt")


class UserPrompt(Prompt):
    """User turn: text plus optional images, audio and documents.

    Media fields accept media objects, file paths or http(s) URLs, singly or
    in a list: ``UserPrompt(text="Describe it", images="chart.png")``.
    """

    kind: str = Field(default="user", description="type of prompt")
    images: list[Image] = Field(
        default_factory=list, description="Images sent with the text."
    )
    audio: list[Audio] = Field(
        default_factory=list, description="Audio clips sent with the text."
    )
    documents: list[Document] = Field(
        default_factory=list,
        description="Document files (PDF, DOCX, CSV, ...) sent with the text.",
    )

    @field_validator("images", "audio", "documents", mode="before")
    @classmethod
    def _coerce_media(cls, value: Any, info: ValidationInfo) -> Any:
        media_class: type[Media] = {
            "images": Image,
            "audio": Audio,
            "documents": Document,
        }[info.field_name or ""]
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            value = [value]
        return [media_class.coerce(item) for item in value]

    @property
    def has_media(self) -> bool:
        return bool(self.images or self.audio or self.documents)
