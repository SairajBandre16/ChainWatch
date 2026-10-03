# ChainWatch: write-up

## Problem

Shippers moving goods from India to Europe and Ireland depend on a few chokepoints: Bab-el-Mandeb, the Red
Sea and the Suez Canal. When one closes, the Cape of Good Hope detour adds about two weeks. Planners learn
about disruptions from scattered news, often after carriers have already rerouted. The question for this
project: can a free, laptop-scale system turn global news into lane-level warnings early enough to act on,
without drowning planners in false alarms?

## Approach

1. **Extraction.** News from 6 RSS feeds and GDELT is normalized into one `NewsItem` model and cached. An
   LLM fills a small Pydantic schema (`event_type`, location, ISO country, UN/LOCODE, severity 1-5,
   dates, confidence). Invalid JSON or schema violations trigger a repair retry with the validation error
   shown to the model; items that still fail are recorded, not dropped silently.
2. **Graph.** 58 ports, 35 sea regions and chokepoints, ~100 sea legs and 23 named lanes. Events are linked
   to nodes (UN/LOCODE, aliases, whole-word aliases, fuzzy matching, then country). A lane's exposure is
   `1 - prod(1 - severity/5 x confidence x match weight)` over events touching its route, which stays in
   [0, 1] and is explainable by listing its contributions.
3. **Forecast.** A LightGBM late-delivery model on DataCo, with point-in-time history features and a purged
   time split, shown next to a majority baseline, a one-rule baseline and logistic regression. SHAP gives
   plain-text explanations.
4. **Agent.** A plain-Python loop where the model calls 8 typed tools via JSON and then writes a brief.
   Code-level guard rails verify every structured claim against the tools.
5. **Backtest.** Definitions fixed in `docs/backtest-spec.md` before any code: onset dates, a 0.5 flag
   threshold, a 7-day lookback, evaluation windows and quiet control windows.

## Experiments (before -> after)

All numbers are from `docs/results.md`.

| Area | Change | Before | After |
|---|---|---|---|
| Forecast [R3] | Majority -> one rule -> logistic regression -> LightGBM | AUC 0.500 -> 0.724 -> 0.752 | AUC 0.776 |
| Linking [R2] | Port code accepted only if its country matches the event | Saudi "Yanbu port" + hallucinated INNSA linked to Nhava Sheva | linked to `CTRY:SA` (direct link rate 0.75 -> 0.625, but correct) |
| Agent [R5] | Exact-phrase search -> any-keyword search + coverage guard rail | qwen: "no disruptions", lane recall 0.0 | brief completed |
| Agent [R5] | Accept `{"action": "<tool>"}` | qwen stuck for 7 turns, fallback | completed, precision 1.0, recall 0.5 |
| Agent [R5] | 3 grace turns + `focus="all"` | llama ran out of budget, fallback | completed, recall 1.0, precision 0.22 |
| Extraction [R1] | qwen2.5:3b vs llama3.2:3b, same prompt | – | schema failure rate 1.5% vs 10.6%; accuracy pending labels |

## Backtest findings [R6]

With the keyword extractor on 41,937 GDELT items:

- **Suez 2021:** flagged on the onset day (lead time 0). A sudden grounding cannot be predicted from news;
  same-day detection is the realistic best.
- **Red Sea 2023:** the primary lane was already above threshold on the first day of the 30-day window, so
  the lead time is at least 30 days before carriers suspended transits, and at least 4 days before the
  first attack on a merchant ship.
- **But the false-alarm rate in quiet periods is 0.75.** The system never misses because it is almost
  always on. Diagnosis: (a) country-level matches pull in unrelated Indian news; (b) one syndicated story
  (dozens of URLs) is counted as dozens of independent events.

**Post-hoc v2 [R6b].** Two rules, written down and committed before running (`docs/backtest-spec-v2.md`):
drop country-level matches and count each story (date, event type, node) once. Result: false-alarm rate
0.75 -> 0.367, Red Sea lead unchanged, Suez now detected one day *after* onset (-1). That last change is
informative: v1's same-day Suez detection was partly lifted over the threshold by an unrelated Indian
story. The remaining false alarms are real but minor incidents on the route (a ship briefly aground in
Suez in September 2022, Houthi threats in 2019). Because the rules were chosen after looking at the
control windows, 0.367 is optimistic.

The honest headline is therefore: *the plumbing works end to end and catches both disruptions, but with
keyword extraction the signal is too noisy to be useful.* Deduplication and dropping country-level
matches (R6b) halve the noise; the next step is the LLM extractor's severity judgement, so that "briefly
aground" stops counting like "closed".

## Lessons

- **Write evaluation definitions before code.** It removed every temptation to move the threshold.
- **A model that never misses can still be useless.** False-alarm rate on control periods is the number
  that exposed the problem; lead time alone looked great.
- **Small LLMs need code around them.** The most valuable agent work was not prompting; it was guard rails
  (real citations, recomputed scores, coverage, route feasibility) and reading transcripts.
- **Score the raw model, ship the grounded output.** Otherwise guard rails hide how weak the model is.
- **Leakage hides in label timing.** A date split is not enough when labels arrive days after the order.
- **Independence assumptions break on news.** Syndication turns one fact into many "events".

## What is still pending

- Hand labels for extraction P/R/F1 (owner task, `docs/labeling-guide.md`).
- The LLM backtest run (about 20 hours on a laptop CPU; resumes from cache).
- Deployment to Hugging Face Spaces (`docs/deployment.md`).
