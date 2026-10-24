# Eval harness (stage 2)

Goal: publishable numbers, not vibes.

1. Hand-label, for each worksheet question, the source page(s) it should cite (`gold_questions.example.jsonl` shows the format).
2. Metrics: retrieval recall@k, citation accuracy (verifier pass rate and correctness), abstention precision.
3. Ablations: glossary expansion on/off, highlight boost on/off, embedding model, reranker on/off.
4. Store results per `config/dentistry.yaml` version so every improvement is traceable.

Real gold sets built from copyrighted material stay in `data/private/` (gitignored); publish aggregate metrics only.
