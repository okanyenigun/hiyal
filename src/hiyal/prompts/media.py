"""Media items: images, audio and documents sent to or returned by a model."""

import base64
import io
import mimetypes
import wave
from pathlib import Path
from typing import ClassVar, Self
from pydantic import BaseModel, Field, model_validator

# Python's guesses that providers name differently.
_MIME_OVERRIDES = {
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
}


# The reverse of _MIME_OVERRIDES, for naming media that has no filename.
_OVERRIDE_EXTENSIONS = {mime: suffix for suffix, mime in _MIME_OVERRIDES.items()}


def guess_mime_type(name: str) -> str | None:
    """MIME type from a file name or URL, using the names providers expect."""
    suffix = Path(name.split("?", 1)[0]).suffix.lower()
    return _MIME_OVERRIDES.get(suffix) or mimetypes.guess_type(f"file{suffix}")[0]


class Media(BaseModel):
    """One media item, given by exactly one source: bytes, a URL, or a file ID."""

    mime_type: str = Field(
        description="MIME type, e.g. 'image/png'.", examples=["image/png"]
    )
    data: bytes | None = Field(default=None, repr=False, description="Raw file bytes.")
    url: str | None = Field(
        default=None,
        description="An http(s) URL, or a provider file URI such as a Gemini Files API URI.",
    )
    file_id: str | None = Field(
        default=None, description="ID of a file uploaded to the provider's Files API."
    )
    filename: str | None = Field(
        default=None,
        description="Original file name; some providers require one for inline files.",
    )

    # Required MIME top-level type for subclasses ('image', 'audio'); None accepts any.
    media_kind: ClassVar[str | None] = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        sources = [
            name
            for name in ("data", "url", "file_id")
            if getattr(self, name) is not None
        ]
        if len(sources) != 1:
            raise ValueError(
                f"give exactly one of data, url or file_id (got {sources or 'none'})"
            )
        if self.media_kind and not self.mime_type.startswith(f"{self.media_kind}/"):
            raise ValueError(
                f"{type(self).__name__} needs a '{self.media_kind}/*' MIME type, got {self.mime_type!r}"
            )
        return self

    @classmethod
    def from_path(cls, path: str | Path, mime_type: str | None = None) -> Self:
        path = Path(path)
        mime_type = mime_type or guess_mime_type(path.name)
        if mime_type is None:
            raise ValueError(
                f"cannot guess the MIME type of {path.name!r}; pass mime_type"
            )
        return cls(data=path.read_bytes(), mime_type=mime_type, filename=path.name)

    @classmethod
    def from_bytes(
        cls, data: bytes, mime_type: str, filename: str | None = None
    ) -> Self:
        return cls(data=data, mime_type=mime_type, filename=filename)

    @property
    def name(self) -> str:
        """``filename``, or a generic name with an extension matching the MIME type."""
        if self.filename:
            return self.filename
        extension = (
            _OVERRIDE_EXTENSIONS.get(self.mime_type.lower())
            or mimetypes.guess_extension(self.mime_type)
            or ""
        )
        return f"{type(self).__name__.lower()}{extension}"

    @classmethod
    def from_url(cls, url: str, mime_type: str | None = None) -> Self:
        mime_type = mime_type or guess_mime_type(url)
        if mime_type is None:
            raise ValueError(f"cannot guess the MIME type of {url!r}; pass mime_type")
        return cls(url=url, mime_type=mime_type)

    @classmethod
    def from_file_id(cls, file_id: str, mime_type: str) -> Self:
        return cls(file_id=file_id, mime_type=mime_type)

    @classmethod
    def coerce(cls, value: object) -> object:
        """Accept a path or http(s) URL in place of a media object."""
        if isinstance(value, Path):
            return cls.from_path(value)
        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                return cls.from_url(value)
            return cls.from_path(value)
        return value

    @property
    def base64(self) -> str:
        """Base64 text of the bytes, as most APIs expect for inline media."""
        if self.data is None:
            raise ValueError("this media has no inline data (it is a URL or file ID)")
        return base64.b64encode(self.data).decode()

    @property
    def data_url(self) -> str:
        """``data:<mime>;base64,...`` form of the bytes."""
        return f"data:{self.mime_type};base64,{self.base64}"

    def save(self, path: str | Path) -> Path:
        """Write the bytes to ``path`` and return it."""
        if self.data is None:
            raise ValueError(
                "this media has no inline data to save (it is a URL or file ID)"
            )
        path = Path(path)
        path.write_bytes(self.data)
        return path


class Image(Media):
    """An image, e.g. PNG, JPEG, WebP or GIF."""

    media_kind: ClassVar[str | None] = "image"


class Audio(Media):
    """An audio clip, e.g. WAV or MP3."""

    media_kind: ClassVar[str | None] = "audio"
    transcript: str | None = Field(
        default=None, description="What the audio says, when the provider returns it."
    )

    @property
    def is_raw_pcm(self) -> bool:
        """Headerless 16-bit PCM (``audio/L16``, ``audio/pcm``), which players can't open."""
        return self.mime_type.lower().startswith(("audio/l16", "audio/pcm"))

    def to_wav(self) -> "Audio":
        """A playable WAV copy of raw 16-bit PCM; other audio is returned unchanged.

        Rate and channels come from MIME parameters (``audio/L16;rate=24000``),
        defaulting to 24 kHz mono, as Gemini and OpenAI speech output use.
        """
        if not self.is_raw_pcm:
            return self
        if self.data is None:
            raise ValueError("this audio has no inline data to convert")
        options = dict(
            part.strip().split("=", 1)
            for part in self.mime_type.split(";")[1:]
            if "=" in part
        )
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(int(options.get("channels", 1)))
            wav.setsampwidth(2)
            wav.setframerate(int(options.get("rate", 24000)))
            wav.writeframes(
                self.data
            )  # providers send little-endian samples, as WAV stores them
        filename = f"{Path(self.filename).stem}.wav" if self.filename else None
        return Audio(
            data=buffer.getvalue(),
            mime_type="audio/wav",
            transcript=self.transcript,
            filename=filename,
        )


def sniff_audio_mime_type(data: bytes) -> str | None:
    """MIME type from an audio file's leading bytes; None for headerless PCM."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:3] == b"ID3" or (
        len(data) > 1 and data[0] == 0xFF and data[1] & 0xF6 == 0xF2
    ):
        return "audio/mpeg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if len(data) > 1 and data[0] == 0xFF and data[1] & 0xF6 == 0xF0:
        return "audio/aac"
    return None


class Document(Media):
    """A document file, e.g. PDF, DOCX, CSV or plain text."""


def media_class_for(mime_type: str) -> type[Media]:
    """Image for image/*, Audio for audio/*, Document for anything else."""
    top_level = mime_type.split("/", 1)[0]
    return {"image": Image, "audio": Audio}.get(top_level, Document)


def as_media(
    file: "str | Path | bytes | Media",
    mime_type: str | None = None,
    filename: str | None = None,
) -> Media:
    """A path, raw bytes (with ``mime_type``) or inline media, as typed media with data.

    ``filename`` names the upload; for a path it replaces the file's own name,
    and for media it replaces the media's filename. ``mime_type`` is needed for
    bytes, can fix a path whose type cannot be guessed, and is ignored for media.
    """
    if isinstance(file, Media):
        if file.data is None:
            raise ValueError(
                "media to upload must have inline data, not a URL or file ID"
            )
        return (
            file if filename is None else file.model_copy(update={"filename": filename})
        )
    if isinstance(file, bytes):
        if mime_type is None:
            raise ValueError("pass mime_type when uploading raw bytes")
        return media_class_for(mime_type).from_bytes(file, mime_type, filename)
    path = Path(file)
    mime_type = mime_type or guess_mime_type(path.name)
    if mime_type is None:
        raise ValueError(f"cannot guess the MIME type of {path.name!r}; pass mime_type")
    media = media_class_for(mime_type).from_path(path, mime_type)
    return (
        media if filename is None else media.model_copy(update={"filename": filename})
    )
