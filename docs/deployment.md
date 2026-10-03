# Deployment checklist (owner)

Nothing has been deployed. Deploying is an owner decision. These steps put the dashboard on a free
Hugging Face Space (Docker SDK). The Space shows committed sample data; no secrets are needed.

## 1. Check locally

```bash
docker build -t chainwatch .
docker run -p 7860:7860 chainwatch
# open http://localhost:7860
```

## 2. Create the Space

1. Sign in at https://huggingface.co and create a new Space: SDK **Docker**, hardware **CPU basic (free)**.
2. Clone it: `git clone https://huggingface.co/spaces/<your-user>/chainwatch hf-chainwatch`

## 3. Copy the files

From this repo into the Space clone:

- `Dockerfile`, `.dockerignore`, `pyproject.toml`, `uv.lock`
- `src/`, `app/`, `data/processed/`, `data/sample/`, `data/eval/`, `docs/`
- `deploy/hf-space/README.md` -> the Space's `README.md` (it carries the Space metadata: `sdk: docker`,
  `app_port: 7860`). The licence line (`license: mit`) is already set.

Then `git add . && git commit -m "Deploy ChainWatch" && git push`. The Space builds the image and starts.

## 4. Optional extras

- **Forecaster on the Space:** the trained model (`models/forecast.joblib`) is gitignored. To include the
  what-if panel, train locally (`uv run python -m chainwatch.forecast.train`) and copy `models/` into the
  Space, or add a build step that downloads DataCo and trains (adds about a minute to the build).
- **LLM briefs:** the Space has no Ollama. To enable LLM briefs, set `GROQ_API_KEY` or `GEMINI_API_KEY`
  as a Space secret and use `provider=groq` / `provider=gemini`. Free tiers have rate limits.

## 5. After deploy

- Add the Space URL to the README.
- Record a short demo (map, brief, backtest tab) for the portfolio.
