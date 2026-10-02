# Progress
Last updated: 2026-10-02

## Current step: 1.3 Event schema + extraction

## Done
- [x] 0 Bootstrap: repo layout, pyproject (uv, Python 3.12), config, docs stubs, smoke test; pytest + ruff pass

## Plan
### Phase 1: Data and event extraction (branch `phase/1-extraction`)
- [x] 1.1 LLM interface: base client + Ollama/Groq/Gemini/fake, disk cache, JSON retry; 14 tests pass (1f3a802)
- [x] 1.2 News ingestion: 6 RSS feeds + GDELT DOC API + GDELT daily event files, cached in data/raw/http, offline mode; 66-item sample in data/sample/news_sample.jsonl
- [ ] 1.3 Event schema + extraction
- [ ] 1.4 Extraction eval harness

### Phase 2: Knowledge graph (branch `phase/2-graph`)
- [ ] 2.1 Reference data
- [ ] 2.2 Graph build + queries
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
- None yet.

## Decisions pending review
- GDELT DOC API returns HTTP 429 from this machine; live news uses RSS, history uses GDELT daily event files (D4).
- Local model: `qwen2.5:3b` (already installed) is the default instead of `qwen2.5:7b` from the spec. See `docs/decisions.md`.

## Resume instructions
On branch `phase/1-extraction`, start step 1.3 (Event schema + extraction).
