# Decisions log

Each entry: choice, alternatives, reason.

## D1. Python 3.12 via uv
- Choice: pin Python 3.12 with `uv python pin`.
- Alternatives: system Python 3.10 (too old for spec), 3.13/3.14 (some ML wheels lag).
- Reason: satisfies 3.11+ and has wheels for LightGBM, SHAP and friends.

## D2. Default local model qwen2.5:3b
- Choice: default Ollama model `qwen2.5:3b`; `llama3.2:3b` as the second model for comparisons.
- Alternatives: `qwen2.5:7b` (spec suggestion, 4.7 GB download, slower on a laptop CPU).
- Reason: both 3B models are already installed; swap via `CHAINWATCH_LLM_MODEL` at any time.
