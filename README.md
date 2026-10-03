# ChainWatch

**An early-warning system for supply chain disruptions on India to Europe (and Ireland) shipping lanes.**

ChainWatch reads global news, extracts disruption events with an LLM, maps them onto a trade-lane graph
of ports and chokepoints, adds a classical ML delay-risk model with SHAP explanations, and has an agent
write a grounded mitigation brief. The main result is a backtest that asks: *how early would it have
flagged the 2021 Suez blockage and the 2023 Red Sea crisis, and how often does it cry wolf?*

Everything is free and runs on a laptop: local models through Ollama, open data, no paid APIs.

## Architecture

```
 RSS feeds ─┐                               ┌─ ports, chokepoints, sea legs, lanes (CSV)
 GDELT ─────┼─> [1] Event extraction (LLM) ─┼─> [2] Knowledge graph (NetworkX)
            │      schema + retry + cache   │      routes, exposure, alternates, linking
            │                               │
 DataCo ────┴─> [3] Delay-risk model ───────┤      LightGBM vs baselines, SHAP
                                            v
                                [4] Agent: tool loop + guard rails -> mitigation brief
                                            │
                     [5] Backtest (spec-first replay)    [6] FastAPI + Streamlit dashboard
```

| Module | Path | What it does |
|---|---|---|
| LLM interface | `src/chainwatch/llm/` | One client API for Ollama, Groq, Gemini and a fake; disk cache; JSON self-repair retries |
| Ingestion | `src/chainwatch/ingest/` | 6 RSS feeds, GDELT DOC API, GDELT daily event files; offline HTTP cache |
| Extraction | `src/chainwatch/extraction/` | `DisruptionEvent` schema, versioned prompts, failure-tolerant pipeline, keyword baseline, P/R/F1 scorer |
| Graph | `src/chainwatch/graph/` | 58 ports, 35 waypoints (10 chokepoints), 23 lanes; exposure scoring, alternate routes, event linking |
| Forecast | `src/chainwatch/forecast/` | Point-in-time features, purged time split, 4 models, SHAP `explain()` |
| Agent | `src/chainwatch/agent/` | 8 typed tools, JSON tool-calling loop, guard rails, deterministic fallback, rubric |
| Backtest | `src/chainwatch/backtest/` | Replay engine implementing `docs/backtest-spec.md` |
| API / app | `src/chainwatch/api/`, `app/` | FastAPI endpoints and a Streamlit dashboard that run from sample data |

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Nothing else is needed for the sample data.

```bash
uv sync                                   # install
uv run pytest                             # 119 tests, offline, under a minute
uv run streamlit run app/main.py          # dashboard on sample data
uv run uvicorn chainwatch.api.main:app    # API: /events /lanes /risk /brief /backtest
```

Reproduce the results (each command writes the files that `docs/results.md` cites):

```bash
ollama pull qwen2.5:3b                                            # optional local LLM
uv run python -m chainwatch.ingest.run --write-sample             # fetch news (RSS + GDELT)
uv run python -m chainwatch.extraction.run --provider ollama --model qwen2.5:3b
uv run python -m chainwatch.graph.linking                         # link rate report
uv run python -c "from chainwatch.forecast.data import download_dataco; download_dataco()"
uv run python -m chainwatch.forecast.train                        # forecast metrics (R3)
uv run python -m chainwatch.forecast.explain                      # SHAP plots (R4)
uv run python -m chainwatch.agent.run --provider ollama --model qwen2.5:3b --save   # agent (R5)
uv run python -m chainwatch.backtest.run                          # backtest (R6)
uv run python -m chainwatch.backtest.run --scoring v2             # post-hoc v2 scoring (R6b)
```

Docker: `docker build -t chainwatch . && docker run -p 7860:7860 chainwatch`.

## Results

All numbers come from runs logged in [`docs/results.md`](docs/results.md) (run ids in brackets).

**Late-delivery forecast** (DataCo, 180,519 orders, time-based purged split, test = Jul 2017 to Jan 2018) [R3]

| Model | Test ROC AUC | Test PR-AUC | Test Brier |
|---|---|---|---|
| Majority (train late rate) | 0.500 | 0.551 | 0.247 |
| One rule: shipping-mode history | 0.724 | 0.793 | 0.196 |
| Logistic regression | 0.752 | 0.821 | 0.195 |
| **LightGBM** | **0.776** | **0.839** | **0.182** |

**Backtest** (frozen spec, keyword extractor, 164 GDELT days, 41,937 news items, lane Nhava Sheva to Rotterdam) [R6]

| Disruption | Onset | First warning | Lead time |
|---|---|---|---|
| Suez Canal blocked (Ever Given) | 2021-03-23 | 2021-03-23 | 0 days (same-day detection) |
| Red Sea crisis (carriers suspend transits) | 2023-12-15 | 2023-11-15 | ≥ 30 days (censored at window start) |
| Quiet control periods (60 days) | – | 45 days flagged | **false-alarm rate 0.75** |

Post-hoc v2 scoring [R6b] (rules pre-registered in `docs/backtest-spec-v2.md` before running: drop
country-level matches, count each story once): false-alarm rate **0.75 -> 0.367**, Red Sea lead time
unchanged (≥ 30 days), Suez detected **one day after** onset (-1 day). Optimistic, because the rules
were chosen after inspecting the v1 control windows.

**Graph** [R2, R2c]: a Red Sea event exposes 22 of 23 lanes; the Cape of Good Hope detour adds
4,540 nm (+13.5 days at 14 knots) to Nhava Sheva to Rotterdam. Event-to-graph link rate on sample events: 0.857 to 0.875.

**Agent** [R5, development scenario, not a benchmark]: after fixes found by reading transcripts, both
qwen2.5:3b and llama3.2:3b complete briefs with 100% valid citations; qwen lane recall 0.5 (precision 1.0),
llama lane recall 1.0 (precision 0.22). Guard rails remove invented lanes and infeasible reroutes.

**Extraction** [R1]: on 66 live news items, qwen2.5:3b failed schema validation on 1.5% of items and
llama3.2:3b on 10.6% (after 2 repair retries). **Accuracy (P/R/F1) is pending hand labels**; no extraction
accuracy is claimed until 100+ items are labeled (see `docs/labeling-guide.md`).

## What failed, and limitations

- **The backtest's false-alarm rate (0.75) makes the keyword-based system unusable as is.** Two causes
  found: country-level matches ("India" + any attack keyword) and syndicated copies of one story being
  counted as independent evidence in `1 - prod(1 - w)`. A labeled post-hoc run fixing both (R6b) brings it
  to 0.367, still too high; the remaining flags are real minor incidents (a ship briefly aground in Suez,
  Houthi threats) that keyword rules cannot tell apart from closures.
- **The LLM backtest did not finish.** qwen2.5:3b on a laptop CPU needs roughly 20 hours for the 8,773
  prefiltered items; it resumes from cache (`docs/results.md`, R7).
- **Small LLMs make plain mistakes:** qwen labeled tanker attacks as severe weather; llama invented a port
  code (INNSA for a Saudi port) and a "2 extra days" reroute that does not exist. Guard rails catch the
  structured errors; free-text summaries can still contain invented numbers.
- **DataCo is retail order data, not maritime.** The forecaster gives a baseline delay risk by destination
  and shipping mode. Its second-strongest feature (payment type) is likely a dataset artifact.
- **The graph is schematic:** great-circle legs between hand-placed waypoints, about 5-10% off real sailing
  distances.
- **Two disruptions is a tiny sample;** backtest numbers illustrate behavior, not statistical strength.

## Screenshots (placeholders)

- [ ] Risk map tab with exposed lanes highlighted
- [ ] Brief tab showing a grounded brief with guard-rail notes
- [ ] Backtest tab: daily exposure vs. onset for Suez 2021 and Red Sea 2023
- [ ] Forecast tab: model table and SHAP importance

## Data sources

- News: publisher RSS feeds (titles and short summaries only), [GDELT](https://www.gdeltproject.org/) 1.0 event files and DOC 2.0 API.
- [DataCo Smart Supply Chain](https://data.mendeley.com/datasets/8gx2fvg2k6/5) (Constante et al., 2019, CC BY 4.0).
- Weather: [Open-Meteo](https://open-meteo.com/) forecast API.
- Land outline: [Natural Earth](https://www.naturalearthdata.com/) 1:110m (public domain).
- Ports and lanes: curated by hand (UN/LOCODE codes, approximate coordinates).

## Docs

`docs/results.md` (every number) · `docs/backtest-spec.md` · `docs/decisions.md` ·
`docs/learning-notes.md` · `docs/writeup.md` · `docs/deployment.md` · `PROGRESS.md`
