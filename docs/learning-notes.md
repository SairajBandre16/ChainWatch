# Learning notes

Plain-English notes on the concepts each phase uses: what it is, why it is used here, and the trade-offs.

## Phase 1: LLM extraction

### One LLM interface (`src/chainwatch/llm/`)
- **What:** every module calls `get_llm_client()` and gets an object with `generate()` (text) and
  `generate_structured(prompt, Schema)` (validated Pydantic object). Ollama, Groq, Gemini and a fake
  client all implement one method, `_complete()`.
- **Why:** swapping models becomes a config change, which makes "compare two models" experiments cheap,
  and the fake client lets tests run offline in milliseconds.
- **Structured output + retry:** LLMs sometimes return broken or schema-violating JSON. We parse, validate
  with Pydantic, and on failure re-ask with the validation error pasted in. This "self-repair" loop fixes
  most small mistakes. Trade-off: extra calls and latency when the model is weak.
- **Disk cache:** the response is stored under a SHA-256 hash of (provider, model, prompts, options).
  Same inputs give the same answer instantly, saving free-tier quota and making experiments reproducible.
  Trade-off: if you change something that affects output but is not in the key, you get stale answers,
  so anything that matters must go in the key.

### News ingestion (`src/chainwatch/ingest/`)
- **Normalize early:** RSS, GDELT DOC and GDELT event files all become one `NewsItem` model. Downstream code
  never cares where an item came from. The id is a hash of the URL, so duplicates across feeds collapse.
- **Cache every HTTP call:** `CachedHttp` stores raw responses in `data/raw/http/`. With `--offline` the
  pipeline replays from disk only, which is how tests and demos run without network.
- **Rate limits and backoff:** GDELT allows one request per 5 seconds. On HTTP 429 or 5xx we wait, double the
  wait, and retry (exponential backoff). Trade-off: a slow source can make a run slow, so failures are logged
  and skipped rather than fatal.
- **Recall-first pre-filter:** the GDELT event filter keeps any URL slug with a logistics word. It lets some
  noise through on purpose; the LLM step decides what is a real disruption. Whole-word matching stops
  "report" from matching "port".

### Event schema and extraction (`src/chainwatch/extraction/`)
- **Two-layer schema:** the LLM fills a small `ExtractedEvent` (type, place, codes, severity, dates,
  confidence). The pipeline then adds provenance (source URL, model, prompt version) to make a
  `DisruptionEvent`. Small models do better when asked for fewer fields, and provenance must never come
  from the model anyway.
- **Enums + aliases:** event types are a fixed list, so results are countable. Near-misses ("strike",
  "weather") are mapped to the enum instead of being rejected, which saves retries.
- **Validators as guard rails:** country codes must be 2 letters, port codes 5 characters, severity 1-5.
  A model reply that breaks a rule triggers the retry loop with the exact error message.
- **Versioned prompts:** prompts live in `extraction/prompts/extract_v1.md`. Every event records the
  prompt version, so a later `extract_v2` can be compared fairly on the same labeled items.
- **Failures are data:** an item that still fails after retries becomes a `status="failed"` record.
  The failure rate is a metric, and one bad article never kills a batch.
- **Keyword baseline:** a rules-only extractor. It doubles as the fake LLM (offline runs) and as the
  floor the real models must beat in the evaluation.

## Phase 2: Knowledge graph

### Why a graph
- A trade lane (Nhava Sheva to Rotterdam) is a path through ports, sea regions and chokepoints. An event
  at any node on that path threatens the lane. "Which lanes does this event touch?" is a graph question.
- **Schematic network:** ~60 ports and ~35 waypoints joined by ~100 legs. Leg length is the great-circle
  (haversine) distance between nodes. It is not a nautical chart, but distances land within about 5-10%
  of real sailing distances, which is enough for "is the Cape detour days or weeks?" reasoning.
- **Node kinds:** port, chokepoint, sea_region, country, event. Routing only walks `sea` edges, so
  linking an event to two ports never creates a fake shortcut between them.

### Queries
- `lane_route`: Dijkstra shortest path by distance (`networkx.shortest_path`).
- `alternate_routes(lane, avoid)`: remove the avoided nodes, then Yen's k-shortest simple paths
  (`networkx.shortest_simple_paths`). Reports extra miles and extra days at 14 knots.
- `exposed_lanes(event)`: a lane is exposed if the event touches its route, an endpoint, or (at half
  weight) an endpoint's country.
- `lane_exposure_score`: `1 - prod(1 - w_i)`, where `w_i = severity/5 x confidence x match weight`.
  This treats events as independent chances of disruption: bounded in [0, 1], grows with each event,
  and every score comes with the list of events that produced it (explainable by construction).

### Linking free text to nodes
- The LLM writes "Hormuz Strait" or "Port of Felixtowe"; the graph has `HORMUZ` and `GBFXT`.
  Linking tries, in order: the UN/LOCODE, an exact alias, an alias inside the text (longest wins),
  fuzzy string similarity (difflib ratio >= 0.85, catches typos), then the country node.
- Each link records which rule fired, so the link rate can be broken down and weak rules spotted.

## Phase 3: Risk forecasting

### Data leakage and time-aware splits
- **Leakage** = training on information that would not exist at prediction time. It gives great test
  scores and a useless model. DataCo has obvious leaks: real shipping days and delivery status *are* the
  answer, and the shipping date reveals it too. These are dropped from features.
- **Subtle leak: label timing.** An order placed on Dec 30 that ships on Jan 3 has a label you only learn on
  Jan 3. A naive date split puts it in training even though, on Jan 1, you could not know it. We record
  `label_known_at` and **purge** such rows from the earlier split.
- **Point-in-time history features:** "late rate for First Class to Western Europe so far" is computed with a
  cumulative sum over labels sorted by when they became known, then `merge_asof` attaches, to each order,
  the last value known strictly before it was placed. Tests recompute this by brute force and also flip
  future labels to prove past features do not move.
- **Smoothing:** a group with 3 orders and 3 late ones should not get rate 1.0. We use
  `(late + 20 x prior) / (n + 20)`, which shrinks small groups toward the overall rate (a Bayesian average).

### Baselines and model choice
- Always show a **baseline** next to a model. "AUC 0.78" means nothing alone; "0.78 vs 0.72 for a
  one-line rule" tells you what the model actually adds.
- **ROC AUC**: probability a random late order is scored above a random on-time one (0.5 = coin flip).
  **PR-AUC**: precision/recall trade-off for the positive class; its floor is the positive rate (0.55).
  **Brier score**: mean squared error of probabilities; rewards calibrated probabilities, lower is better.
- **Early stopping**: LightGBM adds trees until validation loss stops improving for 100 rounds. Validation
  picks the number of trees; the test split is used once, at the end.

### SHAP explanations
- SHAP splits one prediction into per-feature contributions that **add up exactly** to the model output
  (in log-odds for a classifier): `logit(p) = base + sum(contributions)`. A test checks this identity.
- **Global view:** mean |SHAP| per feature ranks what the model relies on overall.
- **Local view:** `explain(order)` lists the top 3 contributions in plain words, e.g. "shipping mode = First
  Class raises risk (+2.81 log-odds)".
- SHAP explains the **model**, not the world. A big SHAP value for payment type means the model uses it,
  which led us to check the raw data and flag a likely dataset artifact.

## Phase 4: Agent

### Tool-calling loop without a framework
- Each turn the model sees: the task, the tool list, and a transcript of earlier calls and results. It
  replies with one JSON action: call a tool, or finish with a brief. Plain Python makes every step visible
  and cacheable; LangGraph would add structure we do not need yet (decision D13).
- **Budgets:** a cap on tool calls stops infinite loops; a few grace turns let the model still finish.
- **Be liberal in what you accept:** small models slip on the protocol (`{"action": "lane_risk"}`). Accepting
  obvious variants is cheaper than re-prompting.

### Guard rails: trust code for facts, the LLM for words
- The LLM is good at reading tool results and writing prose, bad at copying numbers and remembering ids.
  So after it writes a brief, code checks it against the tools: unknown event ids are removed, lane scores
  are recomputed, lanes with zero exposure are dropped, missing exposed lanes are added, impossible reroutes
  are removed, and every lane gets at least one feasible mitigation.
- **Score the raw draft, ship the grounded one.** The rubric measures what the model did by itself
  (lane precision/recall, citation validity/relevance), while users get the corrected brief. Otherwise the
  guard rails would hide how weak the model is.
- **Read transcripts.** Every bug in R5 was found by reading what the agent actually did, not from a metric.

## Phase 4b: Backtesting

- **Write the definitions first.** `docs/backtest-spec.md` fixed onset dates, the flag threshold, windows
  and control periods before any code ran. Otherwise it is too easy to pick a threshold that makes the
  chart look good (that is tuning on the test set).
- **Lead time** = onset date - first on-target warning. Positive means early. A warning only counts if it
  is caused by an event at the right place (on target), so a random port strike does not get credit.
- **False-alarm rate** on quiet control periods is the other half: a system that is always on never
  misses, and is useless. Our keyword run did exactly that (0.75 false-alarm rate).
- **Censoring:** if the lane is already flagged on the first day of the window, the lead time is "at
  least 30 days", not "30 days". Report it that way.
- **Independence assumption:** `1 - prod(1 - w)` assumes each event is independent evidence. News is
  not: one story is syndicated dozens of times. Deduplicating by story (or capping per-node weight) is the
  standard fix, and must be evaluated as a new run.
- **Point-in-time replay:** each day only sees events published up to that day; the engine is tested on
  synthetic timelines where the right answers are known.
- **Post-hoc fixes and pre-registration.** After seeing a bad result it is tempting to tweak until it
  looks good. The honest way: write the fix down and commit it *before* running (`docs/backtest-spec-v2.md`),
  run it once, keep the old run, and say the new number is optimistic because the fix was inspired by the
  same data. Here it halved the false-alarm rate (0.75 -> 0.367) and also revealed that the v1 Suez
  "same-day" detection had been helped by an unrelated story.

## Assisted labeling and anchoring bias (step 1.4b)
- **What:** a model drafts the label; a human accepts, edits or retypes it. Common in industry
  ("model-in-the-loop" annotation) because confirming is faster than typing.
- **Catch:** people tend to accept what they are shown (anchoring). If the drafting model is also a model
  you evaluate, its scores go up for a reason that has nothing to do with quality.
- **Mitigations used:** record per item how the label was made (`label_source`, `draft_model`); report
  metrics on the edited/manual subset too; read the article before the draft. Stronger options: draft
  with a model you do not evaluate, or label a random slice fully blind and compare agreement.
