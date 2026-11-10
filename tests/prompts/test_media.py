import base64
import io
import wave

import pytest
from pydantic import ValidationError

from hiyal.prompts.media import (
    Audio,
    Document,
    Image,
    Media,
    as_media,
    guess_mime_type,
    media_class_for,
    sniff_audio_mime_type,
)

PNG = b"\x89PNG\r\n\x1a\n0000"
WAV_HEADER = b"RIFF\x24\x00\x00\x00WAVEfmt "


class TestGuessMimeType:
    @pytest.mark.parametrize("name, expected", [
        ("chart.png", "image/png"),
        ("photo.JPG", "image/jpeg"),
        ("speech.wav", "audio/wav"),
        ("speech.m4a", "audio/mp4"),
        ("speech.flac", "audio/flac"),
        ("speech.aac", "audio/aac"),
        ("speech.mp3", "audio/mpeg"),
        ("report.pdf", "application/pdf"),
        ("https://example.com/a/chart.png?size=large&x=1", "image/png"),
        ("dir.with.dots/chart.webp", "image/webp"),
    ])
    def test_known_types(self, name, expected):
        assert guess_mime_type(name) == expected

    @pytest.mark.parametrize("name", ["README", "archive.unknownext", "https://example.com/download"])
    def test_unknown_types_give_none(self, name):
        assert guess_mime_type(name) is None


class TestMediaSources:
    def test_exactly_one_source_is_required(self):
        with pytest.raises(ValidationError, match="exactly one of data, url or file_id"):
            Media(mime_type="image/png")
        with pytest.raises(ValidationError, match="exactly one"):
            Media(mime_type="image/png", data=b"x", url="https://e.com/a.png")
        with pytest.raises(ValidationError, match="exactly one"):
            Media(mime_type="image/png", data=b"x", url="https://e.com/a.png", file_id="f")

    def test_empty_bytes_count_as_a_source(self):
        assert Media(mime_type="image/png", data=b"").data == b""

    @pytest.mark.parametrize("source", [{"data": b"x"}, {"url": "https://e.com/x"}, {"file_id": "file-1"}])
    def test_each_single_source_is_accepted(self, source):
        assert Media(mime_type="application/octet-stream", **source)

    def test_typed_media_checks_the_mime_kind(self):
        with pytest.raises(ValidationError, match=r"Image needs a 'image/\*' MIME type"):
            Image(mime_type="audio/wav", data=b"x")
        with pytest.raises(ValidationError, match=r"Audio needs a 'audio/\*' MIME type"):
            Audio(mime_type="image/png", data=b"x")

    def test_document_accepts_any_mime_type(self):
        assert Document(mime_type="image/png", data=b"x")
        assert Document(mime_type="text/csv", data=b"x")

    def test_audio_keeps_a_transcript(self):
        assert Audio(mime_type="audio/wav", data=b"x", transcript="hi").transcript == "hi"


class TestMediaConstructors:
    def test_from_path_reads_bytes_name_and_guesses_type(self, tmp_path):
        path = tmp_path / "chart.png"
        path.write_bytes(PNG)
        image = Image.from_path(path)
        assert (image.data, image.mime_type, image.filename) == (PNG, "image/png", "chart.png")

    def test_from_path_accepts_a_string_and_an_explicit_type(self, tmp_path):
        path = tmp_path / "blob.bin"
        path.write_bytes(b"x")
        assert Document.from_path(str(path), "application/x-custom").mime_type == "application/x-custom"

    def test_from_path_without_a_guessable_type_raises(self, tmp_path):
        path = tmp_path / "blob.unknownext"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="cannot guess the MIME type of 'blob.unknownext'"):
            Document.from_path(path)

    def test_from_path_of_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Image.from_path(tmp_path / "missing.png")

    def test_from_path_validates_the_kind(self, tmp_path):
        path = tmp_path / "speech.wav"
        path.write_bytes(WAV_HEADER)
        with pytest.raises(ValidationError):
            Image.from_path(path)

    def test_from_bytes(self):
        audio = Audio.from_bytes(b"abc", "audio/mpeg", "a.mp3")
        assert (audio.data, audio.mime_type, audio.filename) == (b"abc", "audio/mpeg", "a.mp3")

    def test_from_url_guesses_or_takes_the_type(self):
        assert Image.from_url("https://e.com/a.webp?x=1").mime_type == "image/webp"
        assert Image.from_url("https://e.com/render", "image/png").url == "https://e.com/render"

    def test_from_url_without_a_guessable_type_raises(self):
        with pytest.raises(ValueError, match="cannot guess the MIME type of 'https://e.com/render'"):
            Image.from_url("https://e.com/render")

    def test_from_file_id(self):
        document = Document.from_file_id("file-1", "application/pdf")
        assert (document.file_id, document.data, document.url) == ("file-1", None, None)


class TestGeneratedNames:
    @pytest.mark.parametrize("cls, mime, name", [
        (Audio, "audio/wav", "audio.wav"), 
        (Audio, "audio/mp4", "audio.m4a"), (Audio, "audio/flac", "audio.flac"), (Audio, "audio/aac", "audio.aac"),
        (Audio, "audio/mpeg", "audio.mp3"), (Image, "image/jpeg", "image.jpg"), (Document, "application/pdf", "document.pdf"),
    ])
    def test_extension_follows_the_mime_type(self, cls, mime, name):
        assert cls(mime_type=mime, data=b"x").name == name

    def test_every_override_round_trips(self):
        for suffix, mime in {".wav": "audio/wav", ".m4a": "audio/mp4", ".flac": "audio/flac", ".aac": "audio/aac"}.items():
            assert Audio(mime_type=mime, data=b"x").name.endswith(suffix)
            assert guess_mime_type(f"file{suffix}") == mime


class TestMediaHelpers:
    @pytest.mark.parametrize("media, name", [
        (Image(mime_type="image/png", data=b"x"), "image.png"),
        (Audio(mime_type="audio/wav", data=b"x"), "audio.wav"),
        (Document(mime_type="application/pdf", data=b"x"), "document.pdf"),
        (Document(mime_type="application/x-unknown-thing", data=b"x"), "document"),
        (Image(mime_type="image/png", data=b"x", filename="chart.png"), "chart.png"),
    ])
    def test_name(self, media, name):
        assert media.name == name

    def test_base64_and_data_url(self):
        image = Image(mime_type="image/png", data=PNG)
        assert base64.b64decode(image.base64) == PNG
        assert image.data_url == f"data:image/png;base64,{image.base64}"

    @pytest.mark.parametrize("media", [
        Image(mime_type="image/png", url="https://e.com/a.png"),
        Image(mime_type="image/png", file_id="file-1"),
    ])
    def test_inline_data_helpers_raise_without_data(self, media, tmp_path):
        with pytest.raises(ValueError, match="no inline data"):
            media.base64
        with pytest.raises(ValueError, match="no inline data"):
            media.data_url
        with pytest.raises(ValueError, match="no inline data to save"):
            media.save(tmp_path / "out.png")

    def test_save_writes_bytes_and_returns_the_path(self, tmp_path):
        target = tmp_path / "out.png"
        assert Image(mime_type="image/png", data=PNG).save(str(target)) == target
        assert target.read_bytes() == PNG


class TestCoerce:
    def test_media_objects_pass_through(self):
        image = Image(mime_type="image/png", data=b"x")
        assert Image.coerce(image) is image

    def test_http_and_https_strings_become_urls(self):
        assert Image.coerce("http://e.com/a.png").url == "http://e.com/a.png"
        assert Image.coerce("https://e.com/a.png").url == "https://e.com/a.png"

    def test_strings_and_paths_become_files(self, tmp_path):
        path = tmp_path / "chart.png"
        path.write_bytes(PNG)
        assert Image.coerce(str(path)).data == PNG
        assert Image.coerce(path).data == PNG

    def test_other_objects_are_left_for_validation(self):
        value = {"mime_type": "image/png", "data": b"x"}
        assert Image.coerce(value) is value


class TestAudioToWav:
    def test_raw_pcm_detection(self):
        assert Audio(mime_type="audio/L16;rate=24000", data=b"").is_raw_pcm
        assert Audio(mime_type="audio/l16", data=b"").is_raw_pcm
        assert Audio(mime_type="audio/pcm", data=b"").is_raw_pcm
        assert not Audio(mime_type="audio/wav", data=b"").is_raw_pcm

    def test_kind_check_is_case_sensitive(self):
        # Pinned current behavior (reported as a decision): MIME types are
        # case-insensitive by RFC, but the kind check compares exactly.
        with pytest.raises(ValidationError):
            Audio(mime_type="AUDIO/PCM", data=b"")

    def test_pcm_becomes_wav_with_rate_and_channels_from_the_mime_type(self):
        pcm = b"\x01\x00\x02\x00" * 10
        audio = Audio(mime_type="audio/L16; rate=16000; channels=2", data=pcm,
                      transcript="hi", filename="speech.pcm")
        wav_audio = audio.to_wav()
        assert (wav_audio.mime_type, wav_audio.transcript, wav_audio.filename) == ("audio/wav", "hi", "speech.wav")
        with wave.open(io.BytesIO(wav_audio.data)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 2, 16000)
            assert wav.readframes(wav.getnframes()) == pcm

    def test_pcm_defaults_to_24khz_mono(self):
        wav_audio = Audio(mime_type="audio/pcm", data=b"\x00\x00" * 4).to_wav()
        assert wav_audio.filename is None
        with wave.open(io.BytesIO(wav_audio.data)) as wav:
            assert (wav.getnchannels(), wav.getframerate()) == (1, 24000)

    def test_non_pcm_audio_is_returned_unchanged(self):
        audio = Audio(mime_type="audio/mpeg", data=b"ID3")
        assert audio.to_wav() is audio

    def test_pcm_without_inline_data_raises(self):
        with pytest.raises(ValueError, match="no inline data to convert"):
            Audio(mime_type="audio/pcm", url="https://e.com/a.pcm").to_wav()


class TestSniffAudioMimeType:
    @pytest.mark.parametrize("data, expected", [
        (WAV_HEADER + b"\x00" * 8, "audio/wav"),
        (b"ID3\x03\x00", "audio/mpeg"),
        (b"\xff\xfb\x90\x00", "audio/mpeg"),
        (b"\xff\xf3\x90\x00", "audio/mpeg"),
        (b"fLaC\x00", "audio/flac"),
        (b"OggS\x00", "audio/ogg"),
        (b"\xff\xf1\x50\x80", "audio/aac"),
        (b"\xff\xf9\x50\x80", "audio/aac"),
    ])
    def test_known_headers(self, data, expected):
        assert sniff_audio_mime_type(data) == expected

    @pytest.mark.parametrize("data", [b"", b"\xff", b"\x00\x01\x02\x03", b"RIFF\x00\x00\x00\x00AVI "])
    def test_headerless_or_unknown_data_gives_none(self, data):
        assert sniff_audio_mime_type(data) is None


class TestMediaClassFor:
    @pytest.mark.parametrize("mime_type, cls", [
        ("image/png", Image), ("audio/wav", Audio), ("application/pdf", Document),
        ("text/plain", Document), ("video/mp4", Document), ("garbage", Document),
    ])
    def test_class_by_top_level_type(self, mime_type, cls):
        assert media_class_for(mime_type) is cls


class TestAsMedia:
    def test_inline_media_is_returned_as_is(self):
        image = Image(mime_type="image/png", data=b"x")
        assert as_media(image) is image

    @pytest.mark.parametrize("media", [
        Image(mime_type="image/png", url="https://e.com/a.png"),
        Image(mime_type="image/png", file_id="file-1"),
    ])
    def test_media_without_inline_data_is_rejected(self, media):
        with pytest.raises(ValueError, match="must have inline data"):
            as_media(media)

    def test_bytes_need_a_mime_type(self):
        with pytest.raises(ValueError, match="pass mime_type when uploading raw bytes"):
            as_media(b"x")

    @pytest.mark.parametrize("mime_type, cls", [("image/png", Image), ("audio/wav", Audio), ("application/pdf", Document)])
    def test_bytes_become_typed_media(self, mime_type, cls):
        media = as_media(b"x", mime_type, filename="f")
        assert type(media) is cls and (media.data, media.filename) == (b"x", "f")

    def test_paths_become_typed_media(self, tmp_path):
        path = tmp_path / "speech.wav"
        path.write_bytes(WAV_HEADER)
        media = as_media(path)
        assert type(media) is Audio and media.filename == "speech.wav"
        assert type(as_media(str(path))) is Audio

    def test_path_with_an_explicit_type(self, tmp_path):
        path = tmp_path / "blob.unknownext"
        path.write_bytes(b"x")
        assert type(as_media(path, "image/png")) is Image

    def test_path_without_a_guessable_type_raises(self, tmp_path):
        path = tmp_path / "blob.unknownext"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="cannot guess the MIME type"):
            as_media(path)

    def test_path_with_an_explicit_filename_uses_it(self, tmp_path):
        # BaseLLM.upload_file(path, filename=...) passes the filename through here.
        path = tmp_path / "tmp123.pdf"
        path.write_bytes(b"%PDF")
        media = as_media(path, filename="report.pdf")
        assert (media.filename, media.data) == ("report.pdf", b"%PDF")
        assert as_media(path).filename == "tmp123.pdf"

    def test_media_with_an_explicit_filename_is_renamed_on_a_copy(self):
        original = Image(mime_type="image/png", data=b"x", filename="old.png")
        renamed = as_media(original, filename="new.png")
        assert (renamed.filename, original.filename) == ("new.png", "old.png")
        assert renamed.data == b"x" and as_media(original) is original

    def test_mime_type_is_ignored_for_media(self):
        image = Image(mime_type="image/png", data=b"x")
        assert as_media(image, "image/jpeg").mime_type == "image/png"

    def test_a_path_mime_type_overrides_the_guess(self, tmp_path):
        path = tmp_path / "chart.png"
        path.write_bytes(b"x")
        assert type(as_media(path, "application/pdf")) is Document
