# Progress
Last updated: 2026-10-02

## Current step: 2.3 Event-to-graph linking

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
- [ ] 2.3 Event-to-graph linking

### Phase 3: Risk forecasting (branch `phase/3-forecast`)
- [ ] 3.1 Data prep
- [ ] 3.2 Baselines + model
- [ ] 3.3 Explainability

### Phase 4: Agent and backtest (branch `phase/4-agent-backtest`)
- [ ] 4.1 Tools
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

## Blockers (owner action needed)
- **Extraction labels (step 1.4).** Hand-label 100+ news items so extraction P/R/F1 can be reported.
  Open `data/eval/labeling_template.csv`, follow `docs/labeling-guide.md`, then run
  `uv run python -m chainwatch.extraction.eval import-csv`. Until then extraction metrics are "pending labels".

## Decisions pending review
- GDELT DOC API returns HTTP 429 from this machine; live news uses RSS, history uses GDELT daily event files (D4).
- Local model: `qwen2.5:3b` (already installed) is the default instead of `qwen2.5:7b` from the spec. See `docs/decisions.md`.

## Resume instructions
On branch `phase/2-graph`, start step 2.3 (event-to-graph linking: alias table + fuzzy matching, report link rate on sample events). Then merge phase 2 to main.
