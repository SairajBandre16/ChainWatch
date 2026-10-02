# Progress
Last updated: 2026-10-02

## Current step: 4.2 Agent + brief

## Done
- [x] 0 Bootstrap: repo layout, pyproject (uv, Python 3.12), config, docs stubs, smoke test; pytest + ruff pass

## Plan
### Phase 1: Data and event extraction (branch `phase/1-extraction`)
- [x] 1.1 LLM interface: base client + Ollama/Groq/Gemini/fake, disk cache, JSON retry; 14 tests pass (1f3a802)
- [x] 1.2 News ingestion: 6 RSS feeds + GDELT DOC API + GDELT daily event files, cached in data/raw/http, offline mode; 66-item sample in data/sample/news_sample.jsonl (e96d547)
- [x] 1.3 Event schema + extraction: DisruptionEvent schema, prompt extract_v1, pipeline with retry + failure records, keyword baseline as fake LLM; real run qwen2.5:3b on 66 items: 1 failed, 6 flagged, 226 s (e7c7f6c, 612acbb)
- [x] 1.4 Extraction eval harness: per-field P/R/F1 scorer tested on tiny hand-made set; CSV labeling template (118 items) + import; metrics PENDING LABELS (01021e7)

### Phase 2: Knowledge graph (branch `phase/2-graph`)
- [x] 2.1 Reference data: 58 ports, 35 waypoints (10 chokepoints), 99 sea legs, 23 lanes; validated by tests (828a359)
- [x] 2.2 Graph build + queries: TradeGraph with exposed_lanes, alternate_routes, lane_exposure_score; Red Sea event flags India-Europe lanes, Cape suggested (ab31e2e)
- [x] 2.3 Event-to-graph linking: LOCODE -> alias -> whole-word alias -> difflib fuzzy -> country fallback; link rate on sample events 0.857 (qwen, baseline), see results R2 (9b824f2)

### Phase 3: Risk forecasting (branch `phase/3-forecast`)
- [x] 3.1 Data prep: DataCo downloaded from Mendeley (CC BY 4.0, checksum verified, no Kaggle login); point-in-time history features; purged time split; leakage tests pass (7e509be)
- [x] 3.2 Baselines + model: test ROC AUC majority 0.500, shipping-mode rule 0.724, logreg 0.752, LightGBM 0.776 (results R3) (a299904)
- [x] 3.3 Explainability: exact TreeSHAP via LightGBM pred_contrib; plots in docs/figures/; `explain()` gives top-3 drivers in plain text (results R4) (93c9f11)

### Phase 4: Agent and backtest (branch `phase/4-agent-backtest`)
- [x] 4.1 Tools: 8 typed tools (search/get event, exposed lanes, lane risk, alternate routes, ML delay risk, Open-Meteo weather, list lanes) with as_of time filter; all unit-tested
- [ ] 4.2 Agent + brief
- [ ] 4.3 Backtest engine

### Phase 5: Dashboard and deployment (branch `phase/5-app`)
- [ ] 5.1 API
- [ ] 5.2 Dashboard
- [ ] 5.3 Packaging

### Phase 6: Polish (branch `phase/6-docs`)
- [ ] 6.1 README
- [ ] 6.2 Write-up
- [ ] 6.3 Resume material
- [ ] 6.4 Owner checklist

## Phase summaries

### Phase 1: extraction (merged)
- Built: one LLM interface (Ollama, Groq, Gemini, fake) with disk cache and JSON self-repair retries;
  RSS + GDELT ingestion with offline cache; DisruptionEvent schema, versioned prompt, failure-tolerant
  pipeline; keyword baseline; per-field P/R/F1 scorer and a CSV labeling workflow.
- Measured: failure rate and runtime for qwen2.5:3b (1.5%, 226 s / 66 items) and llama3.2:3b
  (10.6%, 465 s) on the sample (results R1). Accuracy pending owner labels.
- Review: `extraction/prompts/extract_v1.md`, `extraction/schemas.py`, `docs/labeling-guide.md`.

### Phase 2: graph (merged)
- Built: curated ports/chokepoints/legs/lanes, TradeGraph with routes, exposure, alternates and
  explainable exposure score; event-to-node linking.
- Measured: Red Sea event exposes 22 of 23 lanes, Cape detour +4,540 nm / +13.5 days on Nhava Sheva to
  Rotterdam (tests); link rate 0.857-0.875 on sample events (R2). Found and fixed a hallucinated
  port-code link (R2b).
- Review: `data/processed/*.csv` (domain knowledge), `graph/build.py` scoring weights.

### Phase 3: forecast (merged)
- Built: DataCo loader (Mendeley, checksum), point-in-time history features, purged time split with
  leakage tests, 4 models, SHAP plots and `explain()`.
- Measured: test ROC AUC 0.776 (LightGBM) vs 0.752 logreg, 0.724 one-rule, 0.500 majority (R3).
- Review: payment type as a top feature (likely dataset artifact, R4); DataCo is retail, not maritime.

## Blockers (owner action needed)
- **Extraction labels (step 1.4).** Hand-label 100+ news items so extraction P/R/F1 can be reported.
  Open `data/eval/labeling_template.csv`, follow `docs/labeling-guide.md`, then run
  `uv run python -m chainwatch.extraction.eval import-csv`. Until then extraction metrics are "pending labels".

## Decisions pending review
- Graph is a schematic sea network (D7); distances are great-circle approximations.
- GDELT DOC API returns HTTP 429 from this machine; live news uses RSS, history uses GDELT daily event files (D4).
- Local model: `qwen2.5:3b` (already installed) is the default instead of `qwen2.5:7b` from the spec. See `docs/decisions.md`.

## Resume instructions
On branch `phase/4-agent-backtest`, start step 4.2 (tool-calling agent loop + structured brief; fake LLM run end-to-end; real-model example saved to docs/; citation check test). Model file: `uv run python -m chainwatch.forecast.train` recreates models/forecast.joblib (DataCo in data/raw/dataco/, or `download_dataco()`).
