# ChainWatch

AI early-warning system for supply chain disruptions on India to Europe (especially Ireland) trade lanes.

It reads global news, extracts disruption events with an LLM, maps them onto a trade-lane knowledge graph,
forecasts delay risk with classical ML, and has an agent write a mitigation brief. The headline result is a
backtest showing how early the system would have flagged real past disruptions.

> Status: under construction. See `PROGRESS.md`.

## Quick start

```bash
uv sync
uv run pytest -q
```

All results quoted in this repo come from `docs/results.md`.
