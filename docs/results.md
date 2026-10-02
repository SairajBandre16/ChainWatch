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

Accuracy: **pending labels** (step 1.4 owner task). No precision/recall is claimed from this run.

Qualitative observations from reading the outputs (not metrics):
- qwen2.5:3b labeled two "tanker hit in Strait of Hormuz" items as `severe_weather` instead of
  `conflict_or_attack`; locations were right.
- The one qwen failure was a port code that was not a valid 5-character UN/LOCODE on all 3 attempts.
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

Unlinked: "Iran" (no Iranian port in the graph) and "most global" (not a place).
Sample is small (7 events); treat as a smoke test, not a benchmark.
