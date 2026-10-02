"""CLI: produce a mitigation brief.

python -m chainwatch.agent.run                                  # deterministic, no LLM
python -m chainwatch.agent.run --provider ollama --model qwen2.5:3b --save
python -m chainwatch.agent.run --focus india_ireland --as-of 2024-01-15
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from chainwatch.agent.brief import brief_to_markdown, build_brief_deterministic, ground_brief
from chainwatch.agent.loop import run_agent
from chainwatch.agent.rubric import score_result
from chainwatch.agent.store import EventStore
from chainwatch.agent.tools import ToolContext
from chainwatch.config import DOCS_DIR
from chainwatch.forecast.train import MODEL_DIR, load_models
from chainwatch.llm import get_llm_client

EXAMPLES_DIR = DOCS_DIR / "examples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="omit for the deterministic brief")
    parser.add_argument("--model", default=None)
    parser.add_argument("--focus", default=None)
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: now)")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--save", action="store_true", help="write to docs/examples/")
    args = parser.parse_args()

    as_of = (datetime.strptime(args.as_of, "%Y-%m-%d").replace(tzinfo=UTC, hour=23, minute=59)
             if args.as_of else datetime.now(UTC))  # fmt: skip
    forecaster = load_models() if (MODEL_DIR / "forecast.joblib").exists() else None
    ctx = ToolContext(store=EventStore.default(), forecaster=forecaster, as_of=as_of,
                      lookback_days=args.lookback_days)  # fmt: skip

    steps: list = []
    rubric: dict | None = None
    if args.provider:
        llm = get_llm_client(provider=args.provider, model=args.model)
        result = run_agent(llm, ctx, focus=args.focus, max_steps=args.max_steps)
        brief, steps = result.brief, [s.model_dump() for s in result.steps]
        name = llm.model.replace(":", "-")
        rubric = score_result(result, ctx, args.focus).model_dump()
        print("Rubric (raw LLM draft):", json.dumps(rubric))
    else:
        brief = ground_brief(build_brief_deterministic(ctx, args.focus), ctx)
        name = "deterministic"
    markdown = brief_to_markdown(brief)
    print(markdown)
    if args.save:
        EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        # Build names by hand: Path.with_suffix would treat ".5-3b_all" in "qwen2.5-3b" as a suffix.
        stem = f"brief_{name}_{args.focus or 'all'}"
        (EXAMPLES_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")
        payload = {"brief": brief.model_dump(), "rubric": rubric, "steps": steps}
        (EXAMPLES_DIR / f"{stem}.json").write_text(json.dumps(payload, indent=1, default=str),
                                                   encoding="utf-8")  # fmt: skip
        print(f"Saved -> {EXAMPLES_DIR / stem}.md/.json")


if __name__ == "__main__":
    main()
