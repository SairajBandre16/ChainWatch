# Decisions log

Each entry: choice, alternatives, reason.

## D1. Python 3.12 via uv
- Choice: pin Python 3.12 with `uv python pin`.
- Alternatives: system Python 3.10 (too old for spec), 3.13/3.14 (some ML wheels lag).
- Reason: satisfies 3.11+ and has wheels for LightGBM, SHAP and friends.

## D2. Default local model qwen2.5:3b
- Choice: default Ollama model `qwen2.5:3b`; `llama3.2:3b` as the second model for comparisons.
- Alternatives: `qwen2.5:7b` (spec suggestion, 4.7 GB download, slower on a laptop CPU).
- Reason: both 3B models are already installed; swap via `CHAINWATCH_LLM_MODEL` at any time.

## D3. News ingestion lives in `src/chainwatch/ingest/`
- Choice: separate `ingest` package (models, cached HTTP, sources, CLI) instead of putting fetchers in `extraction/`.
- Alternatives: one `extraction` package for everything.
- Reason: fetching and LLM extraction change for different reasons; tests stay focused.

## D4. Three GDELT paths, RSS as the reliable live source
- Choice: RSS (6 feeds) for live news; GDELT DOC API for live search; GDELT 1.0 daily event files for history.
- Alternatives: GDELT DOC only; GDELT GKG files (much larger).
- Reason: the DOC API returned HTTP 429 to every request from this machine (IP-level throttle) and only covers
  ~3 months; daily event files go back to 2013, are free, cache well, and are what the backtest needs.
  Event files carry no headline, so the title is the publisher's own URL slug (nothing invented).

## D5. Keyword baseline doubles as the fake LLM
- Choice: the fake LLM used by the extraction CLI answers with a rule-based keyword extractor.
- Alternatives: a fake that returns fixed canned JSON; no baseline.
- Reason: the offline pipeline produces realistic output, and the eval gets a non-LLM baseline for free.

## D6. Ruff ignores E501
- Choice: `ruff format` enforces line length for code; `E501` is ignored for comments and docstrings.
- Alternatives: hand-wrap every docstring at 100 chars.
- Reason: standard practice with an auto-formatter; avoids churn without hurting readability.

## D7. Schematic sea network with haversine leg lengths
- Choice: hand-curated ports/waypoints/legs; leg length = great-circle distance between nodes.
- Alternatives: real sea-routing library (e.g. searoute) or AIS-derived lanes.
- Reason: free, offline, small and explainable; accurate enough for exposure and detour reasoning.

## D8. Fuzzy matching with stdlib difflib
- Choice: `difflib.SequenceMatcher` with a 0.85 threshold, after exact and whole-word alias rules.
- Alternatives: rapidfuzz (faster), embeddings.
- Reason: no extra dependency; alias lists are small (~300 strings), so speed does not matter.
