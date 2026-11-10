"""Prompt media inputs and quick access to media in responses."""

import base64
import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase

from pydantic import ValidationError

import json

from hiyal.models import AnthropicLLM, DeepSeekLLM, GeminiLLM, LLMConfig, OllamaLLM, OpenAILLM
from hiyal.models.bases import BaseLLM
from test_connection_overrides import _close_sync, _make_model
from hiyal.models.schemas.response import LLMResponse, ResponseContent, ResponseMessage
from hiyal.prompts import Audio, Document, Image, UserPrompt



class MediaTests(TestCase):

    def test_exactly_one_source(self):
        with self.assertRaises(ValidationError):
            Image(mime_type="image/png")
        with self.assertRaises(ValidationError):
            Image(mime_type="image/png", data=b"x", url="https://example.com/a.png")


    def test_base64_data_url_and_save(self):
        image = Image.from_bytes(b"png-bytes", "image/png")
        self.assertEqual(image.data_url, "data:image/png;base64," + base64.b64encode(b"png-bytes").decode())
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(image.save(Path(folder) / "out.png").read_bytes(), b"png-bytes")
        with self.assertRaises(ValueError):
            Image.from_url("https://example.com/a.png").base64


class UserPromptMediaTests(TestCase):

    def test_text_only_prompt_still_works(self):
        config = LLMConfig(system_prompt="s", user_prompt="u")
        self.assertFalse(config.user_prompt.has_media)
        self.assertEqual(BaseLLM._prompt_texts(config), ("s", "u"))



class OpenAIImageInputTests(TestCase):
    """OpenAILLM sends UserPrompt images as Responses ``input_image`` parts."""

    def query(self, prompt):
        llm, http, requests = _make_model(OpenAILLM)
        try:
            llm.query(system_prompt="Be brief.", user_prompt=prompt)
            return json.loads(requests[-1].content)
        finally:
            _close_sync(llm, http)

    def test_text_only_prompt_stays_a_string(self):
        self.assertEqual(self.query("Hello.")["input"], "Hello.")





class ResponseMediaTests(TestCase):
    @staticmethod
    def make(*contents):
        return LLMResponse(
            text="",
            messages=[ResponseMessage(role="assistant", content=content) for content in contents],
            provider="gemini", id_="r", model="m",
            created_at=datetime.now(timezone.utc), status="completed", raw=None,
        )

    def test_images_and_audio_from_adapter_dicts(self):
        response = self.make(
            ResponseContent(content_type="text", text="Here you go"),
            ResponseContent(content_type="image", data={"mime_type": "image/png", "data": b"img"}),
            ResponseContent(content_type="audio", data={"mime_type": "audio/wav", "data": base64.b64encode(b"wav").decode()}),
            ResponseContent(content_type="image", data={"mime_type": "image/jpeg", "uri": "gs://bucket/a.jpg"}),
        )
        self.assertEqual([(i.mime_type, i.data, i.url) for i in response.images],
                         [("image/png", b"img", None), ("image/jpeg", None, "gs://bucket/a.jpg")])
        self.assertEqual([(a.mime_type, a.data) for a in response.audio], [("audio/wav", b"wav")])

    def test_media_objects_are_returned_as_is(self):
        image = Image.from_bytes(b"img", "image/png")
        self.assertIs(self.make(ResponseContent(content_type="image", data=image)).images[0], image)

    def test_empty_when_text_only(self):
        response = self.make(ResponseContent(content_type="text", text="hi"))
        self.assertEqual((response.images, response.audio), ([], []))


def _files_client(kind):
    """An LLM whose HTTP calls are recorded and answered with a file object."""
    import httpx2

    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "DELETE":
            return httpx2.Response(200, json={"id": "file-1", "object": "file", "deleted": True, "type": "file_deleted"})
        return httpx2.Response(200, json={
            "id": "file-1", "object": "file", "type": "file", "bytes": 3, "created_at": 1792499155,
            "filename": "x", "purpose": "user_data", "mime_type": "application/pdf",
        })

    http = httpx2.Client(transport=httpx2.MockTransport(respond))
    llm = kind(config=LLMConfig(api_key="k", model="m", max_retries=0, client_options={"http_client": http}))
    return llm, http, requests


class FilesApiTests(TestCase):
    """upload_file / delete_file on every provider: real calls or a clear error."""


    def test_anthropic_upload(self):
        llm, http, requests = _files_client(AnthropicLLM)
        try:
            audio = llm.upload_file(b"raw-bytes", "audio/wav", filename="note.wav")
            self.assertEqual(requests[-1].url.path, "/v1/files")
            self.assertIsInstance(audio, Audio)
            self.assertEqual((audio.file_id, audio.filename), ("file-1", "note.wav"))
        finally:
            _close_sync(llm, http)





class OpenAIAudioInputTests(TestCase):
    """Audio prompts go to Chat Completions, since Responses has no audio input."""

    def setUp(self):
        import httpx2

        self.requests = []

        def respond(request):
            self.requests.append(request)
            return httpx2.Response(200, json={
                "id": "chatcmpl-1", "object": "chat.completion", "created": 1792499155, "model": "gpt-audio-mini",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Thursday at 3 PM in room 204."}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 9, "total_tokens": 39},
            })

        self.http = httpx2.Client(transport=httpx2.MockTransport(respond))
        self.llm = OpenAILLM(config=LLMConfig(api_key="k", model="gpt-audio-mini", max_retries=0,
                                              client_options={"http_client": self.http}))

    def tearDown(self):
        self.llm.client.close()
        self.http.close()



    def test_unsupported_audio_inputs_raise(self):
        for audio, message in (
            (Audio.from_bytes(b"x", "audio/ogg"), "WAV or MP3"),
            (Audio.from_url("https://example.com/a.wav"), "inline data"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.llm.query(system_prompt="s", user_prompt=UserPrompt(text="u", audio=audio))
        self.assertEqual(self.requests, [])






class GeminiMediaTests(TestCase):
    """GeminiLLM sends prompt media as inline_data / file_data parts."""


    def test_text_only_prompt_unchanged(self):
        llm, http, requests = _make_model(GeminiLLM, model="gemini-3.8-flash")
        try:
            llm.query(system_prompt="s", user_prompt="Hello.")
            payload = json.loads(requests[-1].content)
        finally:
            _close_sync(llm, http)
        self.assertEqual(payload["contents"], [{"role": "user", "parts": [{"text": "Hello."}]}])

    def test_file_id_media_is_rejected(self):
        llm, http, requests = _make_model(GeminiLLM, model="gemini-3.8-flash")
        try:
            with self.assertRaisesRegex(ValueError, "URI"):
                llm.query(system_prompt="s",
                          user_prompt=UserPrompt(text="u", documents=Document.from_file_id("file-1", "application/pdf")))
        finally:
            _close_sync(llm, http)
        self.assertEqual(requests, [])

    def test_response_modalities_reach_the_config(self):
        llm, http, requests = _make_model(GeminiLLM, model="gemini-nano-banana-2.1")
        try:
            llm.query(system_prompt="s", user_prompt="Draw.", response_modalities=["TEXT", "IMAGE"])
            payload = json.loads(requests[-1].content)
        finally:
            _close_sync(llm, http)
        self.assertEqual(payload["generationConfig"]["responseModalities"], ["TEXT", "IMAGE"])


class PcmToWavTests(TestCase):
    def test_raw_pcm_becomes_playable_wav(self):
        import wave

        pcm = Audio.from_bytes(b"\x00\x01" * 2400, "audio/L16;codec=pcm;rate=24000")
        wav = pcm.to_wav()
        self.assertEqual(wav.mime_type, "audio/wav")
        with wave.open(io.BytesIO(wav.data)) as reader:
            self.assertEqual((reader.getframerate(), reader.getnchannels(), reader.getnframes()), (24000, 1, 2400))



class GeminiSpeechConfigTests(TestCase):
    def test_speech_config_reaches_generation_config(self):
        llm, http, requests = _make_model(GeminiLLM, model="gemini-3.8-flash-tts")
        try:
            llm.query(system_prompt="s", user_prompt="Say hi.", response_modalities=["AUDIO"],
                      speech_config={"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}})
            config = json.loads(requests[-1].content)["generationConfig"]
        finally:
            _close_sync(llm, http)
        self.assertEqual(config["responseModalities"], ["AUDIO"])
        # Nested fields keep snake_case on the wire, as google-genai sends them.
        self.assertEqual(config["speechConfig"], {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}})


class OptionalSystemPromptTests(TestCase):
    """Without a system prompt, no system instruction is sent (Gemini TTS rejects one)."""

    def test_no_system_instruction_in_any_provider_request(self):
        for kind in (OpenAILLM, AnthropicLLM, GeminiLLM, DeepSeekLLM, OllamaLLM):
            with self.subTest(provider=kind.provider):
                llm, http, requests = _make_model(kind)
                try:
                    llm.query(user_prompt="Hello.")
                    payload = json.loads(requests[-1].content)
                finally:
                    _close_sync(llm, http)
                self.assertFalse({"instructions", "system", "systemInstruction"} & payload.keys())
                for message in payload.get("messages", []):
                    self.assertNotEqual(message["role"], "system")

    def test_system_instruction_still_sent_when_given(self):
        llm, http, requests = _make_model(GeminiLLM)
        try:
            llm.query(system_prompt="Be brief.", user_prompt="Hello.")
            payload = json.loads(requests[-1].content)
        finally:
            _close_sync(llm, http)
        self.assertEqual(payload["systemInstruction"]["parts"], [{"text": "Be brief."}])



class AnthropicMediaTests(TestCase):
    """AnthropicLLM sends images and documents as content blocks before the text."""

    def query(self, prompt, **parameters):
        llm, http, requests = _make_model(AnthropicLLM)
        try:
            llm.query(system_prompt="Be brief.", user_prompt=prompt, **parameters)
            return json.loads(requests[-1].content), requests
        finally:
            _close_sync(llm, http)



    def test_text_only_prompt_stays_a_string(self):
        payload, _ = self.query("Hello.")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello."}])




