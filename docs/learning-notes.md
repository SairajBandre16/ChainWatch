# Learning notes

Plain-English notes on the concepts each phase uses: what it is, why it is used here, and the trade-offs.

## Phase 1: LLM extraction

### One LLM interface (`src/chainwatch/llm/`)
- **What:** every module calls `get_llm_client()` and gets an object with `generate()` (text) and
  `generate_structured(prompt, Schema)` (validated Pydantic object). Ollama, Groq, Gemini and a fake
  client all implement one method, `_complete()`.
- **Why:** swapping models becomes a config change, which makes "compare two models" experiments cheap,
  and the fake client lets tests run offline in milliseconds.
- **Structured output + retry:** LLMs sometimes return broken or schema-violating JSON. We parse, validate
  with Pydantic, and on failure re-ask with the validation error pasted in. This "self-repair" loop fixes
  most small mistakes. Trade-off: extra calls and latency when the model is weak.
- **Disk cache:** the response is stored under a SHA-256 hash of (provider, model, prompts, options).
  Same inputs give the same answer instantly, saving free-tier quota and making experiments reproducible.
  Trade-off: if you change something that affects output but is not in the key, you get stale answers,
  so anything that matters must go in the key.
