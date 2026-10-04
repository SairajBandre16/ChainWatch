# Progress
Last updated: 2026-10-04

## Current step: 1.4b done; owner labeling next (Owner checklist item 1)

## Done
- [x] 0 Bootstrap: repo layout, pyproject (uv, Python 3.12), config, docs stubs, smoke test; pytest + ruff pass

## Plan
### Phase 1: Data and event extraction (branch `phase/1-extraction`)
- [x] 1.1 LLM interface: base client + Ollama/Groq/Gemini/fake, disk cache, JSON retry; 14 tests pass (1f3a802)
- [x] 1.2 News ingestion: 6 RSS feeds + GDELT DOC API + GDELT daily event files, cached in data/raw/http, offline mode; 66-item sample in data/sample/news_sample.jsonl (e96d547)
- [x] 1.3 Event schema + extraction: DisruptionEvent schema, prompt extract_v1, pipeline with retry + failure records, keyword baseline as fake LLM; real run qwen2.5:3b on 66 items: 1 failed, 6 flagged, 226 s (e7c7f6c, 612acbb)
- [x] 1.4 Extraction eval harness: per-field P/R/F1 scorer tested on tiny hand-made set; CSV labeling template (118 items) + import; metrics PENDING LABELS (01021e7)
- [x] **1.4b Assisted labeling tool.** Do: a CLI (`uv run python -m chainwatch.extraction.label_assist`)
  that, for each unlabeled row in `data/eval/labeling_template.csv`, shows the source news text, runs
  the existing extraction pipeline (qwen2.5:3b, same schema/prompt as 1.3) to **draft** a candidate
  label, and prompts the owner to `[a]ccept`, `[e]dit`, or `[s]kip`. Only the owner-confirmed value is
  written to the CSV's label columns; nothing is auto-written. Must: (1) never overwrite a row that
  already has a human label, (2) save progress after every row so it's resumable, (3) record in a
  separate column whether the final label was "accepted as drafted" vs "edited" vs "typed from scratch"
  so `docs/labeling-guide.md` and the README can disclose the process honestly, (4) work fully offline
  against the already-cached pipeline (no new network calls beyond the existing LLM client/cache).
  Done when: unit tests cover accept/edit/skip/resume; a dry run over 5 sample rows produces a CSV diff
  the owner can review before committing to the full 100+.
  Result: `extraction/label_assist.py` (accept/edit/manual/skip/quit, atomic save per item, refuses to
  overwrite a label added meanwhile); template gains `label_source` + `draft_model` columns (existing
  content unchanged); `import-csv` skips `draft_unconfirmed` rows and keeps `label_source`; `score`
  also prints the subset without accepted drafts (D16). 13 tests. Dry run qwen2.5:3b on 5 rows: 35 s,
  all 5 drafted "no" (freight-market stories) -> `data/eval/label_assist_preview.csv` (gitignored).
  Interactive run on the real CSV not exercised by Claude; owner does that.

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
- [x] 4.3 Backtest engine: spec committed first (436013a); replay engine tested on synthetic timelines; frozen keyword run on 164 GDELT days: Suez lead 0 d, Red Sea lead >=30 d (censored), false-alarm rate 0.75 (results R6) (612374a). LLM backtest run (qwen2.5:3b, prefiltered) paused after 5/164 days (R7); see Owner checklist.

### Phase 5: Dashboard and deployment (branch `phase/5-app`)
- [x] 5.1 API: FastAPI /health /events /lanes /risk /brief /backtest; 6 endpoint tests; served via uvicorn and checked with curl (a217f73)
- [x] 5.2 Dashboard: Streamlit tabs (risk map with bundled Natural Earth land, events, brief, backtest, forecast); headless AppTest passes; started and viewed in Chrome (c0dbacd)
- [x] 5.3 Packaging: Dockerfile builds (image 4 GB) and the container serves Streamlit on 7860; CI workflow (ruff + pytest) validated; tests pass without model/raw data (118 passed, 1 skipped); HF Space files + docs/deployment.md; NOT deployed (6ef494d)

### Phase 6: Polish (branch `phase/6-docs`)
- [x] 6.1 README: architecture, quick start, results table (R1-R6 only), what failed/limitations, screenshot placeholders, data sources (4e26184)
- [x] 6.2 Write-up: docs/writeup.md (problem, approach, before/after table, backtest findings, lessons) (4e26184)
- [x] 6.3 Resume material: docs/resume-bullets.md, 4 bullets, numbers only from results.md, pending items marked (4e26184)
- [x] 6.4 Owner checklist: below

### Post-plan (branch `phase/7-backtest-v2`)
- [x] 7.1 Backtest v2 (post-hoc): rules pre-registered in docs/backtest-spec-v2.md (1bdb376) before running; `--scoring v2` drops country hits and dedupes stories; false-alarm rate 0.75 -> 0.367, Red Sea lead >=30 d, Suez -1 d (results R6b)
- [x] 7.2 Licence: MIT (LICENSE, pyproject, HF Space metadata, README; D15)

### Post-plan (branch `phase/8-label-assist`)
- [x] 8.1 Assisted labeling tool — see 1.4b above (same task, tracked here for branch history).

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

### Phase 8: assisted labeling (merged)
- Built: LLM-draft, human-confirm labeling CLI with per-item provenance; scorer reports the
  non-accepted subset; labeling guide rewritten with both paths and a bias warning.
- Measured: nothing new (no labels yet). Dry run drafts 5 items in 35 s.
- Review: D16 (anchoring bias: qwen2.5:3b drafts and is also scored; consider drafting with a model
  you do not evaluate via `--provider/--model`), `docs/labeling-guide.md` Option A.

### Phase 6: docs (merged)
- Built: README, write-up, resume bullets, owner checklist.
- Review: README "What failed" section and resume bullet wording.

## Owner checklist (everything still needed from you)
1. **Label extraction data (blocker for extraction metrics).** Once 1.4b (assisted labeling tool) is
   built, run it (`uv run python -m chainwatch.extraction.label_assist`) to review AI-drafted labels
   for 100+ rows of `data/eval/labeling_template.csv` — accept, edit, or retype each one yourself; the
   tool never writes a label without your confirmation. Then run
   `uv run python -m chainwatch.extraction.eval import-csv`, and ask Claude to score qwen2.5:3b,
   llama3.2:3b and the keyword baseline and log R8. (If you'd rather label from scratch with no AI
   draft, `docs/labeling-guide.md` still describes the manual path.)
2. **Finish the LLM backtest (optional, ~20 h CPU).** Run
   `uv run python -m chainwatch.backtest.run --extractor ollama:qwen2.5:3b --prefilter --offline`
   overnight; it resumes from `data/raw/backtest/qwen2.5-3b__prefilter/`. Then log R7 results.
   (Shortcut: narrow the replay window to ~35 days either side of the Red Sea onset date instead of
   all 164 days, if a full run doesn't fit your timeline — ask Claude to add a
   `--days-around-onset N` flag first.)
3. ~~Decide on a post-hoc backtest fix~~ Done 2026-10-03 as R6b (country drop + story dedupe,
   pre-registered): false-alarm rate 0.367. Review `docs/backtest-spec-v2.md` and R6b.
4. **DataCo data:** nothing to do; it downloads from Mendeley without login (D9).
5. **Check the dashboard by eye:** `uv run streamlit run app/main.py`, click every tab; take the 4
   screenshots listed in the README.
6. **Record a demo** (map -> brief -> backtest tab), 1-2 minutes.
7. ~~Choose a licence~~ MIT added 2026-10-03 (D15). Change it if you prefer another.
8. ~~Push to GitHub~~ Done 2026-10-03: https://github.com/SairajBandre16/ChainWatch (public, `main` only; CI green on first push).
9. **Deploy** to Hugging Face Spaces following `docs/deployment.md`, then add the URL to the README.
10. **Fill resume bullets** in `docs/resume-bullets.md` once items 1-3 produce numbers.

## Blockers (owner action needed)
- **Extraction labels (step 1.4, tool being built as 1.4b).** Hand-label (or AI-assisted-label once
  1.4b lands) 100+ news items so extraction P/R/F1 can be reported. Open
  `data/eval/labeling_template.csv`, follow `docs/labeling-guide.md`, then run
  `uv run python -m chainwatch.extraction.eval import-csv`. Until then extraction metrics are
  "pending labels".

## Decisions pending review
- Backtest false-alarm rate 0.75 (R6, frozen spec); post-hoc v2 brings it to 0.367 but Suez detection slips to -1 day (R6b). v2 rules are D14.
- Licence chosen as MIT (D15) and copyright holder written as "Sairaj Bandre"; edit LICENSE if wrong.
- Graph is a schematic sea network (D7); distances are great-circle approximations.
- GDELT DOC API returns HTTP 429 from this machine; live news uses RSS, history uses GDELT daily event files (D4).
- Local model: `qwen2.5:3b` (already installed) is the default instead of `qwen2.5:7b` from the spec. See `docs/decisions.md`.
- **Assisted labeling bias (D16).** Drafts come from qwen2.5:3b, which is also scored. The scorer
  prints the non-accepted subset; the README/write-up must say how labels were made (`stats` command).
  Option: draft with a different model (`--provider groq` needs a free key) to remove the conflict.

## Resume instructions
1.4b is built and merged. Remaining work is owner actions: Owner checklist item 1 (labeling, now via
`uv run python -m chainwatch.extraction.label_assist`) and item 2 (LLM backtest). Build the
`--days-around-onset` shortcut flag only if the owner asks.

If labels arrive: run extraction for each model over the labeled news ids, then
`uv run python -m chainwatch.extraction.eval score --pred <file>` per model and log results as R8.
If the LLM backtest finishes: log it as R7 (keep the prefilter deviation note).
Model file: `uv run python -m chainwatch.forecast.train` recreates models/forecast.joblib.