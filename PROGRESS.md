# Progress
Last updated: 2026-10-02

## Current step: build plan complete (owner actions remain; see Owner checklist)

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
- [x] 4.1 Tools: 8 typed tools (search/get event, exposed lanes, lane risk, alternate routes, ML delay risk, Open-Meteo weather, list lanes) with as_of time filter; all unit-tested (288ac13)
- [x] 4.2 Agent + brief: plain-Python JSON tool loop, Brief schema, guard rails (citations, scores, coverage, reroute feasibility), deterministic fallback, rubric; real qwen/llama examples in docs/examples (results R5) (42199a2)
- [x] 4.3 Backtest engine: spec committed first (436013a); replay engine tested on synthetic timelines; frozen keyword run on 164 GDELT days: Suez lead 0 d, Red Sea lead >=30 d (censored), false-alarm rate 0.75 (results R6) (612374a). LLM backtest run (qwen2.5:3b, prefiltered) stopped after 3/164 days (R7); see Owner checklist.

### Phase 5: Dashboard and deployment (branch `phase/5-app`)
- [x] 5.1 API: FastAPI /health /events /lanes /risk /brief /backtest; 6 endpoint tests; served via uvicorn and checked with curl (a217f73)
- [x] 5.2 Dashboard: Streamlit tabs (risk map with bundled Natural Earth land, events, brief, backtest, forecast); headless AppTest passes; started and viewed in Chrome (c0dbacd)
- [x] 5.3 Packaging: Dockerfile builds (image 4 GB) and the container serves Streamlit on 7860; CI workflow (ruff + pytest) validated; tests pass without model/raw data (118 passed, 1 skipped); HF Space files + docs/deployment.md; NOT deployed (6ef494d)

### Phase 6: Polish (branch `phase/6-docs`)
- [x] 6.1 README: architecture, quick start, results table (R1-R6 only), what failed/limitations, screenshot placeholders, data sources (4e26184)
- [x] 6.2 Write-up: docs/writeup.md (problem, approach, before/after table, backtest findings, lessons) (4e26184)
- [x] 6.3 Resume material: docs/resume-bullets.md, 4 bullets, numbers only from results.md, pending items marked (4e26184)
- [x] 6.4 Owner checklist: below

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

### Phase 4: agent and backtest (merged)
- Built: 8 typed agent tools with an `as_of` time filter, a JSON tool-calling loop, code guard rails
  (citations, recomputed scores, coverage, reroute feasibility, dedupe), deterministic fallback brief,
  rubric; backtest spec written first, replay engine, metrics, CLI.
- Measured: agent rubric on the sample scenario (R5, development numbers); backtest with the keyword
  extractor: Suez lead 0 days, Red Sea lead >= 30 days, false-alarm rate 0.75 (R6).
- Review: `docs/backtest-spec.md` (onset dates, threshold), R6 diagnosis, `agent/brief.py` guard rails.

### Phase 5: app (merged)
- Built: FastAPI service, Streamlit dashboard (offline map), Dockerfile, CI, HF Space config.
- Measured: 119 tests pass (R0); docker image builds and serves.
- Review: dashboard visuals (Chrome froze while the LLM backtest used the CPU; first full render looked
  right, please click through all tabs once), `docs/deployment.md`.

### Phase 6: docs (this branch, merged to main)
- Built: README, write-up, resume bullets, owner checklist.
- Review: README "What failed" section and resume bullet wording.

## Owner checklist (everything still needed from you)
1. **Label extraction data (blocker for extraction metrics).** Label 100+ rows of
   `data/eval/labeling_template.csv` following `docs/labeling-guide.md`, run
   `uv run python -m chainwatch.extraction.eval import-csv`, then ask Claude to score qwen2.5:3b,
   llama3.2:3b and the keyword baseline (needs extraction runs over the labeled items) and log R8.
2. **Finish the LLM backtest (optional, ~20 h CPU).** Run
   `uv run python -m chainwatch.backtest.run --extractor ollama:qwen2.5:3b --prefilter --offline`
   overnight; it resumes from `data/raw/backtest/qwen2.5-3b__prefilter/`. Then log R7 results.
3. **Decide on a post-hoc backtest fix.** Story-level dedupe and down-weighting country-level matches
   would address the 0.75 false-alarm rate; it must be reported as a separate labeled run (spec rules).
4. **DataCo data:** nothing to do; it downloads from Mendeley without login (D9).
5. **Check the dashboard by eye:** `uv run streamlit run app/main.py`, click every tab; take the 4
   screenshots listed in the README.
6. **Record a demo** (map -> brief -> backtest tab), 1-2 minutes.
7. **Choose a licence** for the repo (none is set) and add it to the README and HF Space metadata.
8. **Push to GitHub** (not done; no remote configured). Create the repo, `git remote add origin ...`,
   push `main` and the phase branches if you want them. CI runs on push.
9. **Deploy** to Hugging Face Spaces following `docs/deployment.md`, then add the URL to the README.
10. **Fill resume bullets** in `docs/resume-bullets.md` once items 1-3 produce numbers.

## Blockers (owner action needed)
- **Extraction labels (step 1.4).** Hand-label 100+ news items so extraction P/R/F1 can be reported.
  Open `data/eval/labeling_template.csv`, follow `docs/labeling-guide.md`, then run
  `uv run python -m chainwatch.extraction.eval import-csv`. Until then extraction metrics are "pending labels".

## Decisions pending review
- Backtest false-alarm rate 0.75 with the keyword extractor; a post-hoc dedupe/country-weight fix would be a separate labeled run (R6).
- Graph is a schematic sea network (D7); distances are great-circle approximations.
- GDELT DOC API returns HTTP 429 from this machine; live news uses RSS, history uses GDELT daily event files (D4).
- Local model: `qwen2.5:3b` (already installed) is the default instead of `qwen2.5:7b` from the spec. See `docs/decisions.md`.

## Resume instructions
All Build Plan steps are done and merged to `main`. Remaining work is the Owner checklist above.
If labels arrive: run extraction for each model over the labeled news ids, then
`uv run python -m chainwatch.extraction.eval score --pred <file>` per model and log results as R8.
If the LLM backtest finishes: log it as R7 (keep the prefilter deviation note).
Model file: `uv run python -m chainwatch.forecast.train` recreates models/forecast.joblib.
