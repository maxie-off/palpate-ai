"""One thin interface for every model call, so the provider can be swapped by config."""

import json
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, model_id: str, max_tokens: int = 1024) -> str: ...

    def embed(
        self, texts: list[str], *, input_type: str = "search_document"
    ) -> list[list[float]]: ...


class BedrockLLMClient:
    """Claude chat via the Bedrock Converse API, embeddings via Cohere on Bedrock."""

    def __init__(self, region: str, embedding_model_id: str) -> None:
        import boto3

        self._runtime = boto3.client("bedrock-runtime", region_name=region)
        self._embedding_model_id = embedding_model_id

    def complete(self, *, system: str, user: str, model_id: str, max_tokens: int = 1024) -> str:
        if not model_id:
            raise ValueError(
                "model_id is empty: set PALPATE_ANSWER_MODEL_ID / PALPATE_BULK_MODEL_ID"
            )
        resp = self._runtime.converse(
            modelId=model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        text: str = resp["output"]["message"]["content"][0]["text"]
        return text

    def embed(self, texts: list[str], *, input_type: str = "search_document") -> list[list[float]]:
        # Cohere v3: search_document when indexing, search_query when retrieving.
        body = json.dumps({"texts": texts, "input_type": input_type})
        resp = self._runtime.invoke_model(modelId=self._embedding_model_id, body=body)
        embeddings: list[list[float]] = json.loads(resp["body"].read())["embeddings"]
        return embeddings

    def describe_image(
        self,
        *,
        image: bytes,
        mime_type: str,
        prompt: str,
        model_id: str,
        max_tokens: int = 512,
    ) -> str:
        """Caption a figure with a multimodal Claude model on Bedrock."""
        if not model_id:
            raise ValueError("model_id is empty: set PALPATE_VISION_MODEL_ID")
        image_format = _bedrock_image_format(mime_type)
        if image_format is None:
            raise ValueError(f"unsupported image type for Bedrock vision: {mime_type}")
        resp = self._runtime.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": image_format, "source": {"bytes": image}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": max_tokens},
        )
        text: str = resp["output"]["message"]["content"][0]["text"]
        return text


def _bedrock_image_format(mime_type: str) -> str | None:
    mime = mime_type.lower()
    if "png" in mime:
        return "png"
    if "jpeg" in mime or "jpg" in mime:
        return "jpeg"
    if "gif" in mime:
        return "gif"
    if "webp" in mime:
        return "webp"
    return None
