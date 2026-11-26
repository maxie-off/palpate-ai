# 0004: LangGraph routes tasks; it does not roam

**Status:** accepted

**Context.** Decision 0001 kept quiz help as a fixed pipeline and said the model decides what to say, not which steps run. It also named LangGraph as the later shape of that pipeline, once there was a bounded retry. The product now has more than one task: ingest a PDF (text and figures, with positions), draft class notes from a PDF plus a recording, and answer a question from the same index. Quiz and summarize are expected later. The owner asked whether LangChain or LangGraph should route those tasks.

**Decision.** One LangGraph `StateGraph` is the entry point. The branch is the request's `task`, or a keyword fallback when the caller did not set one. Ingest, notes, and Q&A share indexing, hybrid retrieval, and the existing deterministic citation check. Verify may send a draft back once when every quote failed. Quiz and summarize are nodes that return `not_implemented` until they are built. LangChain is not used as an agent: prompts and provider calls stay behind `LLMClient`, `VisionClient`, and `SpeechToTextClient`, as decision 0003 requires. LangGraph is a dependency because it is the graph runtime; we do not call LangChain chains or tool loops.

**Consequences.** Adding a task means adding a node and an edge, which is visible in `docs/agent-pipeline.md`. The model still cannot invent a tool call. We take on the LangGraph dependency and a small amount of state-passing boilerplate. A free-form "agent that figures it out" remains out of scope, including for notes and Q&A.
