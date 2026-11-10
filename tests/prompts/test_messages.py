import pytest
from pydantic import ValidationError

from hiyal.prompts import Audio, Document, Image, Prompt, SystemPrompt, UserPrompt

PNG = b"\x89PNG\r\n\x1a\n0000"


class TestPrompts:
    def test_kinds(self):
        assert Prompt(text="x").kind == "base"
        assert SystemPrompt(text="x").kind == "system"
        assert UserPrompt(text="x").kind == "user"

    def test_text_is_required_and_must_be_a_string(self):
        with pytest.raises(ValidationError):
            UserPrompt()
        with pytest.raises(ValidationError):
            UserPrompt(text=None)

    def test_text_only_prompt_has_no_media(self):
        prompt = UserPrompt(text="hi")
        assert (prompt.images, prompt.audio, prompt.documents) == ([], [], [])
        assert prompt.has_media is False


class TestUserPromptMedia:
    def test_single_items_lists_tuples_and_none_are_accepted(self):
        image = Image(mime_type="image/png", data=PNG)
        assert UserPrompt(text="x", images=image).images == [image]
        assert UserPrompt(text="x", images=[image, image]).images == [image, image]
        assert UserPrompt(text="x", images=(image,)).images == [image]
        assert UserPrompt(text="x", images=None).images == []

    def test_paths_and_urls_are_coerced_per_field(self, tmp_path):
        path = tmp_path / "chart.png"
        path.write_bytes(PNG)
        prompt = UserPrompt(text="x", images=[path, "https://e.com/b.jpg"],
                            audio="https://e.com/a.mp3", documents="https://e.com/r.pdf")
        assert [type(item) for item in prompt.images] == [Image, Image]
        assert prompt.images[0].data == PNG and prompt.images[1].url == "https://e.com/b.jpg"
        assert type(prompt.audio[0]) is Audio and type(prompt.documents[0]) is Document
        assert prompt.has_media

    @pytest.mark.parametrize("field", ["images", "audio", "documents"])
    def test_any_media_field_sets_has_media(self, field):
        media = {"images": Image, "audio": Audio, "documents": Document}[field]
        mime = {"images": "image/png", "audio": "audio/wav", "documents": "application/pdf"}[field]
        assert UserPrompt(text="x", **{field: media(mime_type=mime, data=b"x")}).has_media

    def test_media_of_the_wrong_class_is_rejected(self):
        with pytest.raises(ValidationError):
            UserPrompt(text="x", images=Audio(mime_type="audio/wav", data=b"x"))

    def test_url_with_the_wrong_kind_is_rejected(self):
        with pytest.raises(ValidationError, match="Image needs a 'image/\\*' MIME type"):
            UserPrompt(text="x", images="https://e.com/speech.mp3")

    def test_unguessable_url_is_rejected(self):
        with pytest.raises(ValidationError, match="cannot guess the MIME type"):
            UserPrompt(text="x", images="https://e.com/render")

    def test_missing_file_path_raises_file_not_found(self, tmp_path):
        # Pinned current behavior: the OSError is not wrapped in ValidationError.
        with pytest.raises(FileNotFoundError):
            UserPrompt(text="x", images=str(tmp_path / "missing.png"))

    def test_dicts_are_validated_as_media(self):
        prompt = UserPrompt(text="x", images={"mime_type": "image/png", "url": "https://e.com/a.png"})
        assert prompt.images[0].url == "https://e.com/a.png"
