# 0003: All model calls go through one `LLMClient` interface

**Status:** accepted

**Context.** We want Claude on Bedrock for the AWS story, but quotas or provider changes should not force a rewrite.

**Decision.** `app/core/llm.py` defines a small `LLMClient` protocol (`complete`, `embed`). Bedrock is the first implementation. Frameworks such as LangChain sit behind our own interfaces; prompts and domain logic stay in our code.

**Consequences.** Tests use a trivial fake client. Swapping to the Anthropic API or a local embedding model is a config and adapter change.
