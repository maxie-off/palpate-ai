# 0001: A fixed pipeline, not a free-roaming agent

**Status:** accepted

**Context.** Quiz help follows one predictable flow: parse the question, retrieve evidence, answer, verify.

**Decision.** Implement it as an explicit pipeline (later a LangGraph graph with a bounded retry loop). The model decides what to say, not which steps to run.

**Consequences.** Cheaper, faster, and easier to evaluate and debug than an open-ended agent loop. Every stage can be unit-tested and measured on its own. We give up flexibility for tasks outside the quiz flow, which is acceptable for stage 1 to 3.
