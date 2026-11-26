"""Vision and speech providers.

Text completion and embeddings stay on ``LLMClient`` (see ``app.core.llm``). Vision and speech
take different payloads, so they are sibling protocols rather than extra methods the text
client has to grow. Adapters are selected from settings; nothing here reads a secret from code.
"""

from pathlib import Path
from typing import Protocol

from app.core.llm import BedrockLLMClient
from app.models.schemas import Transcript


class VisionClient(Protocol):
    def describe(self, *, image: bytes, mime_type: str, prompt: str) -> str: ...


class SpeechToTextClient(Protocol):
    def transcribe(self, audio_path: Path, *, language: str | None = None) -> Transcript: ...


class NullVision:
    """Used when ``PALPATE_VISION_MODEL_ID`` is empty. Ingest still records the figure region."""

    def describe(self, *, image: bytes, mime_type: str, prompt: str) -> str:
        return ""


class BedrockVisionClient:
    def __init__(self, bedrock: BedrockLLMClient, model_id: str) -> None:
        self._bedrock = bedrock
        self._model_id = model_id

    def describe(self, *, image: bytes, mime_type: str, prompt: str) -> str:
        return self._bedrock.describe_image(
            image=image,
            mime_type=mime_type,
            prompt=prompt,
            model_id=self._model_id,
        )


class UnconfiguredSpeechToText:
    def transcribe(self, audio_path: Path, *, language: str | None = None) -> Transcript:
        raise RuntimeError(
            "No speech-to-text provider configured. Set PALPATE_STT_PROVIDER "
            "(for example 'transcribe') or inject a SpeechToTextClient. "
            f"Audio path was {audio_path}."
        )


class AmazonTranscribeClient:
    """Placeholder for Amazon Transcribe.

    Batch transcription needs an S3 object and a polling loop. That workflow is not
    wired yet; the protocol is, so the graph can call it once the bucket exists.
    Tests inject a fake client and never reach this class.
    """

    def __init__(self, region: str, model_id: str) -> None:
        self._region = region
        self._model_id = model_id

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> Transcript:
        raise NotImplementedError(
            "Amazon Transcribe adapter is not implemented yet. "
            f"region={self._region} model_id={self._model_id or '(default)'} "
            f"language={language or '(auto)'} file={audio_path.name}. "
            "Inject a SpeechToTextClient, or wait for the S3 transcription workflow."
        )


def build_speech_client(provider: str, *, region: str, model_id: str) -> SpeechToTextClient:
    name = provider.strip().lower()
    if name in ("", "none"):
        return UnconfiguredSpeechToText()
    if name == "transcribe":
        return AmazonTranscribeClient(region=region, model_id=model_id)
    raise ValueError(
        f"Unknown PALPATE_STT_PROVIDER {provider!r}. Expected '', 'none', or 'transcribe'."
    )
