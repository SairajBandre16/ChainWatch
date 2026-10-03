# ChainWatch

AI early-warning system for supply chain disruptions. It reads global news, extracts disruption events with an LLM, maps them onto a trade-lane knowledge graph, forecasts delay risk with classical ML, and has an agent produce a mitigation brief. The headline result is a **backtest** showing how early the system would have flagged real past disruptions.

---

## HOW TO RUN THIS FILE (read first)

This file is the full build spec. When a session starts in this repo, work autonomously:

1. **Check `PROGRESS.md`.** If it does not exist, this is a fresh start: run the Bootstrap (below), then create `PROGRESS.md` from the Build Plan checklist.
2. **Resume at the first unchecked step** in `PROGRESS.md`. Do not redo completed steps.
3. **Build one step at a time.** For each step: implement, run the tests/acceptance check listed for that step, fix failures, then tick the box in `PROGRESS.md` with a one-line note and make a git commit.
4. **Do not stop to ask for permission between steps.** Keep going through the plan until you finish it, hit a Blocker (see below), or run out of session budget. When stopping, make sure `PROGRESS.md` says exactly where to resume.
5. **Make reasonable decisions yourself.** When a choice is not specified here, pick the simplest free option, and log it in `docs/decisions.md` (2 to 4 lines: choice, alternatives, reason). Do not ask the owner about routine choices.
6. **Never fabricate** data, metrics, or results. Every number in docs or README must come from a real run saved in `docs/results.md`. If something cannot be run (no data, no network, no GPU), say so in `PROGRESS.md` and build the code path plus tests with mocked/sample data instead.

### Blockers (the only reasons to stop and ask the owner)

- Something requires a **paid** service, a credit card, or a secret API key you do not have. (Default to local/free alternatives first; add the key to `.env.example` and make the code work without it where possible.)
- A step needs **manual human work**: hand-labeling the evaluation set, downloading a dataset that needs a login (e.g. Kaggle), or accepting a license.
- Network access is refused for a required data source and no offline fallback exists.
- A destructive action is needed (deleting data, force-pushing, overwriting uncommitted work).

When blocked: write the blocker and the exact thing the owner must do into `PROGRESS.md` under `## Blockers`, then continue with any later step that does not depend on it.

### Autonomy rules

- Allowed without asking: creating/editing files in this repo, installing free packages, running tests/linters, running local scripts, git init/branch/commit.
- Not allowed without asking: `git push`, deploying anywhere, deleting files outside build artifacts, anything involving credentials.
- Work on a branch per phase (`phase/1-extraction`, etc.); merge to `main` locally when a phase's acceptance checks pass.
- Keep commits small with clear messages (`feat(extraction): add event schema and prompt v1`).

---

## Project goals

1. Portfolio project for AI Engineer / Applied AI / ML Engineer roles. Every design choice should produce something measurable and explainable in an interview.
2. Show range: LLM extraction, knowledge graph, tabular ML with explainability, agents, evaluation, deployment.
3. Everything must be **free of cost** (local models, free API tiers, open datasets, free hosting).
4. Domain focus: India to Europe (especially Ireland) trade lanes, drawing on the owner's prior supply chain and automation experience.

## Owner context

- Sairaj, MSc Computer Science student at UCD, Dublin. Strong in Python, RPA, n8n and automation; building up ML and AI engineering depth.
- The owner is learning from this project: write clear, readable code with short comments on *why*, and keep a `docs/learning-notes.md` where each phase adds a short plain-English explanation of the concepts used (what, why, trade-offs).
- Final summary at the end of each phase (in `PROGRESS.md`): what was built, what was measured, what to review.

## Architecture

```
News/RSS/GDELT -> [1 Event Extraction (LLM)] -> structured events
                                  |
                                  v
                      [2 Knowledge Graph] <- ports, routes, suppliers
                                  |
Weather/trade/delay data -> [3 Risk Forecast (LightGBM + SHAP)]
                                  |
                                  v
                      [4 Agent: brief + mitigations]
                                  |
                  [5 Backtest]   [6 Dashboard (Streamlit)]
```

## Tech stack (all free)

- **Language:** Python 3.11+, managed with `uv` (fall back to `venv` + pip if `uv` is unavailable)
- **LLMs:** Ollama (e.g. `qwen2.5:7b`) locally; Gemini or Groq free tier as optional fallback via env keys. All LLM access goes through one interface in `src/chainwatch/llm/` so models are swappable. Include a **mock/fake LLM client** so tests and the pipeline run with no model installed.
- **ML:** scikit-learn, LightGBM, SHAP, pandas
- **Graph:** NetworkX (Neo4j only if clearly needed)
- **Data:** GDELT (via its free HTTP APIs/files), Open-Meteo, DataCo Smart Supply Chain (Kaggle, manual download), UN Comtrade (free tier). Every fetcher must cache to `data/raw/` and work offline from cache.
- **Serving:** FastAPI + Streamlit; Hugging Face Spaces for deployment (deployment itself is owner-approved)
- **Tooling:** pytest, ruff, pre-commit, Docker (phase 5), GitHub Actions CI

## Repo layout (create this in Bootstrap)

```
chainwatch/
  CLAUDE.md
  PROGRESS.md
  README.md
  pyproject.toml
  .env.example
  .gitignore
  data/
    raw/            # gitignored; documented download steps
    processed/
    eval/           # labeled extraction set, backtest event list (committed, small)
    sample/         # tiny committed samples so tests run offline
  notebooks/        # exploration only
  src/chainwatch/
    config.py
    llm/            # base interface, ollama client, gemini/groq clients, fake client
    extraction/     # schemas, prompts/, pipeline, eval
    graph/          # build + query
    forecast/       # features, train, shap, predict
    agent/          # tools, prompts/, brief generation
    backtest/       # replay engine + metrics
    api/            # FastAPI app
  app/              # Streamlit dashboard
  tests/
  docs/
    decisions.md
    results.md
    learning-notes.md
```

## Bootstrap (fresh start only)

1. `git init`, create `.gitignore` (Python, `.env`, `data/raw/`, caches, models), commit.
2. Create the repo layout above with `pyproject.toml` (package `chainwatch` in `src/`), dev deps (pytest, ruff, pre-commit), and a trivial passing test.
3. Create `.env.example` (OLLAMA_HOST, GEMINI_API_KEY, GROQ_API_KEY, all optional) and `config.py` that loads it.
4. Create `README.md` stub, `docs/*.md` stubs, and `PROGRESS.md` from the Build Plan.
5. Run `pytest` and `ruff check .`; both must pass. Commit: `chore: bootstrap project`.

## Build Plan

Each step lists **Do** and **Done when**. A step is not finished until its "Done when" passes.

### Phase 1: Data and event extraction (branch `phase/1-extraction`)

- [ ] **1.1 LLM interface.** Do: base `LLMClient` with `generate()` and `generate_structured(schema)`; Ollama client; fake client returning canned JSON; disk cache keyed by prompt+model. Done when: unit tests pass using the fake client; cache hit verified by test.
- [ ] **1.2 News ingestion.** Do: fetcher for GDELT events/doc API and 3+ supply-chain RSS feeds; normalize to a `NewsItem` model; cache to `data/raw/`. Done when: fetcher works from cache offline; sample of 50+ items saved in `data/sample/`.
- [ ] **1.3 Event schema + extraction.** Do: Pydantic `DisruptionEvent` (type, location, country/port code, severity 1-5, start/end, industries, source, confidence); versioned prompt in `extraction/prompts/`; pipeline with validation and retry on invalid JSON. Done when: runs end-to-end on the sample with the fake client and (if Ollama is available) a real model; invalid output is handled, not crashed on.
- [ ] **1.4 Extraction eval harness.** Do: scorer computing precision/recall/F1 per field against `data/eval/extraction_labels.jsonl`; script to generate a **labeling file template** (items to hand-label) for the owner. Done when: scorer is tested on a tiny hand-made label set. **Owner task:** hand-label 100+ items (see Blockers if labels are missing). Until then, results are marked "pending labels" and never invented.

### Phase 2: Knowledge graph (branch `phase/2-graph`)

- [ ] **2.1 Reference data.** Do: curated CSVs in `data/processed/` for ~50 major ports (UN/LOCODE, country, coordinates) and ~30 key routes/chokepoints (Suez, Red Sea/Bab-el-Mandeb, Cape of Good Hope, Panama, Malacca, Hormuz) with India to Europe lanes (e.g. Nhava Sheva/Mundra/Chennai to Rotterdam/Felixstowe/Dublin/Cork). Done when: loaded and validated by tests.
- [ ] **2.2 Graph build + queries.** Do: NetworkX graph (ports, routes, countries, events); functions: `exposed_lanes(event)`, `alternate_routes(lane, avoid)`, `lane_exposure_score(...)`. Done when: tests prove e.g. a Red Sea event flags India to Europe lanes via Suez and suggests the Cape route as an alternative.
- [ ] **2.3 Event-to-graph linking.** Do: map extracted event locations to graph nodes (alias table + fuzzy matching). Done when: tested on sample events with a reported link rate.

### Phase 3: Risk forecasting (branch `phase/3-forecast`)

- [ ] **3.1 Data prep.** Do: load DataCo (from `data/raw/`; if absent, log a Blocker and continue with a clearly labeled synthetic generator used **only** for tests); build features with strictly time-aware splits. Done when: leakage test passes (no future info in training rows).
- [ ] **3.2 Baselines + model.** Do: naive baseline (historical mean/majority), logistic regression, LightGBM; report metrics (AUC, PR-AUC, MAE as appropriate) in `docs/results.md`. Done when: real-data metrics recorded, or marked pending data.
- [ ] **3.3 Explainability.** Do: SHAP summary + per-prediction explanation function; save plots to `docs/figures/`. Done when: `explain(prediction)` returns top drivers in plain text.

### Phase 4: Agent and backtest (branch `phase/4-agent-backtest`)

- [ ] **4.1 Tools.** Do: agent tools wrapping graph queries, forecaster, event store, and weather (Open-Meteo). Done when: each tool unit-tested.
- [ ] **4.2 Agent + brief.** Do: tool-calling loop (plain Python first; LangGraph only if it clearly helps) producing a structured brief: affected lanes, risk score, drivers, mitigations, cited source events. Done when: runs end-to-end with the fake LLM; real-model run saved as an example in `docs/`; briefs only cite events that exist in the store (tested).
- [ ] **4.3 Backtest engine.** Do: define in `docs/backtest-spec.md` exactly what a "flag" and a "disruption onset" are **before** coding; replay engine over a dated event timeline for Suez blockage (Mar 2021) and Red Sea crisis (from late 2023/2024); report lead time per event and false-alarm rate. Done when: engine tested on a synthetic timeline; real results recorded in `docs/results.md` or marked pending data. Never tune on backtest events.

### Phase 5: Dashboard and deployment (branch `phase/5-app`)

- [ ] **5.1 API.** FastAPI endpoints: `/events`, `/risk`, `/brief`. Done when: endpoint tests pass.
- [ ] **5.2 Dashboard.** Streamlit app: map with risk by lane/port, event feed, brief panel, backtest results page. Works with sample data (no network or model required). Done when: app starts and renders from sample data.
- [ ] **5.3 Packaging.** Dockerfile + GitHub Actions CI (lint, tests). Done when: `docker build` succeeds (if Docker is available) and CI config is valid. Do **not** deploy; prepare the Hugging Face Spaces files and leave a deployment checklist for the owner.

### Phase 6: Polish (branch `phase/6-docs`)

- [ ] **6.1 README.** Architecture diagram, quick start, results table (real numbers only), screenshots placeholder list, limitations and honest "what failed" section.
- [ ] **6.2 Write-up.** `docs/writeup.md`: problem, approach, experiments (before/after), backtest findings, lessons.
- [ ] **6.3 Resume material.** `docs/resume-bullets.md` with 3 to 4 bullet drafts containing **placeholders** for numbers, filled only from `docs/results.md`.
- [ ] **6.4 Owner checklist.** List in `PROGRESS.md` everything the owner still must do: label data, download Kaggle data, record demo, deploy, push to GitHub.

## Conventions

- Type hints everywhere; Pydantic at module boundaries.
- Logic in `src/`, never in notebooks.
- Prompts live in versioned files, not inline.
- Fixed random seeds; reproducible runs.
- Cache every LLM/API call to disk (saves free-tier quota, makes runs reproducible).
- Tests mock the LLM and network; the full test suite must run offline in under a minute.
- Run `ruff check . && ruff format .` and `pytest -q` before every commit.

## Evaluation rules (non-negotiable)

- **Extraction:** report P/R/F1 per field on 100+ hand-labeled items; compare at least two models or prompts.
- **Forecasting:** time-based splits only; always show a baseline next to the model.
- **Agent:** rubric check that briefs identify the right lanes and cite only real events.
- **Backtest:** definitions fixed before running; lead time and false-alarm rate reported; no tuning on backtest events.
- Log every experiment (change, before, after) in `docs/results.md`.

## Commands

```bash
uv sync                                                   # install deps
ollama pull qwen2.5:3b                                    # optional local model (default; see D2)
uv run pytest -q                                          # tests
uv run ruff check . && uv run ruff format .               # lint + format
uv run uvicorn chainwatch.api.main:app --reload           # API
uv run streamlit run app/main.py                          # dashboard
uv run python -m chainwatch.ingest.run --write-sample     # fetch news (RSS + GDELT)
uv run python -m chainwatch.extraction.run --provider ollama --model qwen2.5:3b   # extraction
uv run python -m chainwatch.extraction.eval template|import-csv|score --pred <file>  # eval
uv run python -m chainwatch.graph.linking                 # link-rate report
uv run python -m chainwatch.forecast.train                # forecast models + metrics
uv run python -m chainwatch.forecast.explain              # SHAP plots
uv run python -m chainwatch.agent.run [--provider ollama --model qwen2.5:3b] --save  # brief
uv run python -m chainwatch.backtest.run --event red_sea_2023   # backtest (ids: suez_2021, red_sea_2023)
uv run python -m chainwatch.backtest.run --scoring v2           # post-hoc v2 scoring (R6b)
```

(Keep this section accurate as commands are added.)

## PROGRESS.md format

```
# Progress
Last updated: <date>
## Current step: <id and name>
## Done
- [x] 1.1 LLM interface: <one-line note, commit hash>
## Blockers (owner action needed)
- <what, why, exact steps>
## Decisions pending review
- <anything the owner may want to change>
## Resume instructions
<exactly what to do next>
```

## Definition of done

A reviewer can clone the repo, follow the README, run the dashboard from sample data, and reproduce the extraction metrics, forecast metrics, and backtest lead-time results from `docs/results.md`. All numbers in the README are traceable to a recorded run.