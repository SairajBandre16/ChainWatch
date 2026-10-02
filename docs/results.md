# Results

Every number quoted in the README or write-up must come from a run logged here.
Format per entry: date, command, data, change, before, after.

## R1. Extraction run on the news sample (no labels yet)
- Date: 2026-10-02
- Data: `data/sample/news_sample.jsonl` (66 RSS items fetched 2026-10-02 from 6 feeds)
- Prompt: `extract_v1`; temperature 0; laptop CPU via Ollama 0.34
- Commands: `uv run python -m chainwatch.extraction.run --provider ollama --model qwen2.5:3b` and
  `uv run python -m chainwatch.extraction.run` (keyword baseline via the fake client)

| Model | Items | Failed after 2 retries | Flagged as disruption | Events | Wall time |
|---|---|---|---|---|---|
| keyword-baseline | 66 | 0 | 7 | 7 | <1 s |
| qwen2.5:3b | 66 | 1 (1.5%) | 6 | 7 | 226 s |
| llama3.2:3b | 66 | 7 (10.6%) | 7 | 8 | 465 s |

Accuracy: **pending labels** (step 1.4 owner task). No precision/recall is claimed from this run.

Qualitative observations from reading the outputs (not metrics):
- qwen2.5:3b labeled two "tanker hit in Strait of Hormuz" items as `severe_weather` instead of
  `conflict_or_attack`; locations were right.
- The one qwen failure was a port code that was not a valid 5-character UN/LOCODE on all 3 attempts.
- llama3.2:3b failures: invalid or missing severity (3), country_code (2), port_code (2), event_type (2),
  confidence (1) (an item can fail on several fields). It gave severity 5 to 6 of its 8 events, gave
  Strait of Hormuz the country code IQ, and gave "Yanbu port" (Saudi Arabia) the code INNSA (Nhava Sheva).
- The keyword baseline mapped "drone strike" to `labor_strike`; fixed by rule order (commit 612acbb)
  before any labels existed, so this is a bug fix, not tuning on eval data.

## R2. Event-to-graph link rate on sample events
- Date: 2026-10-02
- Command: `uv run python -m chainwatch.graph.linking`
- Data: events from R1

| Extraction source | Events | Link rate | Direct link rate (not country fallback) |
|---|---|---|---|
| keyword-baseline | 7 | 0.857 | 0.857 |
| qwen2.5:3b | 7 | 0.857 | 0.714 |
| llama3.2:3b (before fix) | 8 | 0.875 | 0.750 |
| llama3.2:3b (after fix) | 8 | 0.875 | 0.625 |

Change (R2b, commit after 7e509be): the linker trusted any port code that exists in the graph, so
llama's hallucinated INNSA for "Yanbu port" (SA) was linked to Nhava Sheva, a wrong India-Europe exposure.
Now a port code is accepted only if its country matches the event's country. Before: Yanbu -> INNSA
(wrong). After: Yanbu -> CTRY:SA (country fallback). The direct rate dropped because a wrong direct link
became an honest country-level link.

Unlinked: "Iran" (no Iranian port in the graph) and "most global" (not a place).
Sample is small (7 events); treat as a smoke test, not a benchmark.

## R3. Late-delivery forecast on DataCo (time-based split)
- Date: 2026-10-02
- Command: `uv run python -m chainwatch.forecast.train` (metrics also in `docs/metrics/forecast_metrics.json`)
- Data: DataCo Smart Supply Chain (180,519 order lines, 2015-01-01 to 2018-01-31), target `Late_delivery_risk`
  (54.8% positive overall)
- Split by order date, purged (see D10): train 2015-01-01..2016-12-30 (124,604 rows),
  valid 2017-01-01..2017-06-30 (30,402), test 2017-07-01..2018-01-31 (24,369)
- Features: order-time attributes only (shipping mode, scheduled days, market/region/country, segment,
  category, department, payment type, quantity, price, discount, sales, order hour/weekday/month) plus
  point-in-time historical late rates by mode x region, country, and mode
- Choices made on validation only; test evaluated once. LightGBM early stopping picked 92 rounds.

| Model | Valid ROC AUC | Test ROC AUC | Test PR-AUC | Test Brier | Test log loss | Test accuracy |
|---|---|---|---|---|---|---|
| majority (train late rate) | 0.5000 | 0.5000 | 0.5511 | 0.2474 | 0.6879 | 0.5511 |
| mode_history (one rule) | 0.7100 | 0.7243 | 0.7932 | 0.1955 | 0.5692 | 0.6956 |
| logistic regression | 0.7295 | 0.7517 | 0.8207 | 0.1947 | 0.5653 | 0.6954 |
| LightGBM | 0.7556 | 0.7760 | 0.8389 | 0.1823 | 0.5290 | 0.7226 |

Reading: LightGBM beats the one-rule shipping-mode heuristic by +0.052 test AUC and the majority
baseline by +0.276. Most signal comes from shipping mode and scheduled days (see SHAP, R4). DataCo is a
retail order dataset, not maritime; this model gives the base delay risk that the graph exposure adjusts.
