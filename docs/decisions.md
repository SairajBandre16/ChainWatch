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

## D9. DataCo from Mendeley Data instead of Kaggle
- Choice: download the original DataCo publication from Mendeley Data (CC BY 4.0, no login), verify SHA-256.
- Alternatives: Kaggle mirror (needs login, was listed as an owner task), synthetic data.
- Reason: same file, open licence, scriptable (`download_dataco()`), so no owner action is needed.

## D10. Point-in-time history features and purged splits
- Choice: an order's label is "known" at its shipping date; history features use only labels known
  strictly before the order date; training rows whose label is still unknown at the split cutoff are dropped.
- Alternatives: plain random split (leaks future), plain date split without purging (small leak at the boundary).
- Reason: mirrors what a live system could actually know at prediction time.

## D11. Forecaster answers "what-if" lane questions from a training snapshot
- Choice: the agent's `delay_risk` tool builds one order row (destination country, shipping mode, typical
  values, end-of-training history rates) and scores it with the LightGBM model.
- Alternatives: no ML in the agent; train a maritime delay model (no free labeled maritime delay data found).
- Reason: keeps the ML model useful in the agent while stating plainly that it is a retail-order baseline.

## D12. Agent tools see only events up to `as_of`
- Choice: every tool filters events to a lookback window ending at `ToolContext.as_of`.
- Alternatives: give tools the whole store and trust the caller.
- Reason: the same tools power the backtest replay; this makes "no peeking at the future" structural.

## D13. Plain-Python agent loop with deterministic guard rails
- Choice: one JSON action per turn, tools as typed functions, guard rails in code, deterministic fallback brief.
- Alternatives: LangGraph / LangChain agents; native tool-calling APIs.
- Reason: 3B local models handle a simple JSON protocol better than nested tool schemas, the loop stays
  inspectable, and guard rails make the output safe even when the model is weak.
## D14. Backtest v2 rules: drop country hits, dedupe stories (no parameters)
- Choice: ignore `CTRY:XX` hits in the flag score; one hit per (date, event type, node) at max weight.
- Alternatives: down-weight country hits by a factor; cluster stories by title similarity; raise τ.
- Reason: parameter-free rules leave nothing to tune on backtest data; a weight or τ change would invite
  fitting the control windows. Pre-registered in `docs/backtest-spec-v2.md`, reported separately (R6b).
## D15. MIT licence
- Choice: MIT. Alternatives: Apache-2.0 (patent grant, longer), no licence (all rights reserved).
- Reason: simplest permissive licence, common for portfolio projects; easy for the owner to change.
## D16. Assisted labeling with disclosed provenance
- Choice: `label_assist` drafts each label with qwen2.5:3b, the owner accepts/edits/retypes it; every row
  records `label_source` (accepted/edited/manual/blank = hand CSV) and `draft_model`.
- Alternatives: blank-template hand labeling only (slow); draft with a model not under evaluation (needs
  a paid or rate-limited API); blind first pass before showing the draft (more prompts per item).
- Reason: owner request to cut labeling time. Drafts anchor the labeler and flatter the drafting model,
  so the scorer must also report the non-`accepted` subset and the write-up must state the process.
