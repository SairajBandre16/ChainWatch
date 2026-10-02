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

### News ingestion (`src/chainwatch/ingest/`)
- **Normalize early:** RSS, GDELT DOC and GDELT event files all become one `NewsItem` model. Downstream code
  never cares where an item came from. The id is a hash of the URL, so duplicates across feeds collapse.
- **Cache every HTTP call:** `CachedHttp` stores raw responses in `data/raw/http/`. With `--offline` the
  pipeline replays from disk only, which is how tests and demos run without network.
- **Rate limits and backoff:** GDELT allows one request per 5 seconds. On HTTP 429 or 5xx we wait, double the
  wait, and retry (exponential backoff). Trade-off: a slow source can make a run slow, so failures are logged
  and skipped rather than fatal.
- **Recall-first pre-filter:** the GDELT event filter keeps any URL slug with a logistics word. It lets some
  noise through on purpose; the LLM step decides what is a real disruption. Whole-word matching stops
  "report" from matching "port".

### Event schema and extraction (`src/chainwatch/extraction/`)
- **Two-layer schema:** the LLM fills a small `ExtractedEvent` (type, place, codes, severity, dates,
  confidence). The pipeline then adds provenance (source URL, model, prompt version) to make a
  `DisruptionEvent`. Small models do better when asked for fewer fields, and provenance must never come
  from the model anyway.
- **Enums + aliases:** event types are a fixed list, so results are countable. Near-misses ("strike",
  "weather") are mapped to the enum instead of being rejected, which saves retries.
- **Validators as guard rails:** country codes must be 2 letters, port codes 5 characters, severity 1-5.
  A model reply that breaks a rule triggers the retry loop with the exact error message.
- **Versioned prompts:** prompts live in `extraction/prompts/extract_v1.md`. Every event records the
  prompt version, so a later `extract_v2` can be compared fairly on the same labeled items.
- **Failures are data:** an item that still fails after retries becomes a `status="failed"` record.
  The failure rate is a metric, and one bad article never kills a batch.
- **Keyword baseline:** a rules-only extractor. It doubles as the fake LLM (offline runs) and as the
  floor the real models must beat in the evaluation.
