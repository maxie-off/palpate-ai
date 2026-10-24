"""One thin interface for every model call, so the provider can be swapped by config."""

import json
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, model_id: str, max_tokens: int = 1024) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


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

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"texts": texts, "input_type": "search_document"})
        resp = self._runtime.invoke_model(modelId=self._embedding_model_id, body=body)
        embeddings: list[list[float]] = json.loads(resp["body"].read())["embeddings"]
        return embeddings
