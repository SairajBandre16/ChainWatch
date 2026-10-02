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

## R4. SHAP explanations of the LightGBM forecaster
- Date: 2026-10-02
- Command: `uv run python -m chainwatch.forecast.explain`
- Figures: `docs/figures/shap_importance.png`, `docs/figures/shap_beeswarm.png` (3,000 test rows)

Top features by mean |SHAP| (log-odds, 5,000 random test rows): shipping mode 0.945, payment type 0.302,
recent late rate for shipping mode 0.236, scheduled shipping days 0.236, order hour 0.208,
destination country 0.086. Product, price, quantity and market contribute almost nothing.

Sanity check against raw data (all 180,519 rows): late rate by payment type is TRANSFER 0.485 vs
CASH 0.566, DEBIT 0.572, PAYMENT 0.575, so the payment-type effect is in the data, not a model bug.
It has no plausible logistics cause and is likely an artifact of how DataCo was generated; this is listed
under limitations.

Example output (highest-risk test order): "Late-delivery risk 99% (model baseline 63%). Main drivers:
shipping mode = First Class raises risk (+2.81 log-odds); recent late rate for this shipping mode = 0.95
raises risk (+0.79 log-odds); scheduled shipping days = 1.00 raises risk (+0.57 log-odds)."

## R5. Agent briefs: rubric on the sample scenario (development iterations)
- Date: 2026-10-02
- Command: `uv run python -m chainwatch.agent.run --provider ollama --model <model> --save`
- Scenario: committed sample event store (7 events extracted by qwen2.5:3b from 2026-10-02 RSS), as_of
  2026-10-02, all 23 lanes. Graph ground truth: 2 lanes exposed (AEJEA-NLRTM, AEJEA-INNSA, via Hormuz).
- Prompt `agent_v1`, max 8 tool calls. Rubric scores the LLM's raw draft before guard rails.
- Caveat: this is the only scenario and it was also used to find the bugs below, so these are
  development numbers, not a held-out benchmark.

| Run | Completed | Lane precision | Lane recall | Citation validity | Citation relevance |
|---|---|---|---|---|---|
| qwen2.5:3b, v0 (phrase search) | yes, but wrong | n/a (no lanes) | 0.0 | n/a | n/a |
| qwen2.5:3b, v1 (keyword search) | no (fallback) | - | - | - | - |
| qwen2.5:3b, v2 (tool-name actions accepted) | yes | 1.0 | 0.5 | 1.0 | 0.667 |
| llama3.2:3b, v2 | no (fallback: budget used, no final) | - | - | - | - |
| llama3.2:3b, v3 (3 grace turns, focus "all") | yes | 0.222 | 1.0 | 1.0 | 1.0 |

What changed and why (each a generic fix found by reading the transcript, not a scenario-specific tweak):
- v0 -> v1: qwen searched "sea trade disruption" 5 times, the exact-phrase search matched nothing, and it
  wrote "no disruptions" while both Gulf lanes had exposure 1.0. Search now matches any keyword, and a
  coverage guard rail adds every lane the graph finds exposed.
- v1 -> v2: qwen then wrote `{"action": "lane_risk", ...}` instead of the `call_tool` envelope and repeated
  it 7 times. The loop now accepts a tool name as the action.
- v2 -> v3: llama spent all 8 calls and had no turn left to write the brief; it also asked for
  `focus="all"`, which matched no lanes. Added 3 grace turns and treat "all" as no filter.

Guard rails on the final briefs (both models, v3): 0 invented event ids; llama's 7 unexposed lanes,
2 duplicate lanes and 4 infeasible "alternate route, 2 extra days" claims (no sea route avoids Hormuz)
were removed and replaced by computed "hold or move urgent cargo by air" mitigations. qwen's missing
lane AEJEA-INNSA was added. Remaining known gap: free-text summaries can still contain invented numbers
(qwen wrote "exposure scores ranging from 0.6 to 1.0"); only structured fields are verified.

Example briefs: `docs/examples/brief_{qwen2.5-3b,llama3.2-3b,deterministic}_all.{md,json}`.
