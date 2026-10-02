# ChainWatch dashboard image. Runs offline from committed sample data.
#   docker build -t chainwatch .
#   docker run -p 7860:7860 chainwatch          # dashboard on http://localhost:7860
FROM python:3.12-slim

# uv for fast, locked installs (same lockfile as local development).
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached layer), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY app ./app
COPY data/processed ./data/processed
COPY data/sample ./data/sample
COPY data/eval ./data/eval
COPY docs ./docs
RUN uv sync --frozen --no-dev

# Hugging Face Spaces (Docker SDK) expects port 7860 and a non-root user.
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD ["uv", "run", "--no-sync", "streamlit", "run", "app/main.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
