"""Smoke test: one chat call and one embedding call through LLMClient (needs AWS creds + .env)."""

from app.core.config import get_settings
from app.core.llm import BedrockLLMClient


def main() -> None:
    s = get_settings()
    client = BedrockLLMClient(region=s.aws_region, embedding_model_id=s.embedding_model_id)
    for label, model_id in (("answer", s.answer_model_id), ("bulk", s.bulk_model_id)):
        reply = client.complete(
            system="Reply in one short sentence.",
            user="What does 'palpate' mean?",
            model_id=model_id,
            max_tokens=60,
        )
        print(f"[{label}] {model_id}\n  {reply}")
    vecs = client.embed(["牙齒", "tooth"])
    print(f"[embed] {s.embedding_model_id}: {len(vecs)} vectors x {len(vecs[0])} dims")


if __name__ == "__main__":
    main()
