# 0002: Citations are verified deterministically

**Status:** accepted

**Context.** LLMs can invent a quote or a source ID. For a study tool, a fabricated citation is worse than no citation.

**Decision.** The model returns `{chunk_id, quote}` pairs. Plain code checks that each quote exists in the cited chunk (normalised for whitespace, width, and case) and drops any that fail. If no citation survives, the answer becomes "insufficient evidence".

**Consequences.** Users can trust every citation shown. Abstention rate and verifier pass rate become measurable metrics for the eval harness.
