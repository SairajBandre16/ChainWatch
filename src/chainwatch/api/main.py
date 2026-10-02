"""FastAPI service over the event store, graph, forecaster and agent.

    uvicorn chainwatch.api.main:app --reload

Works with no network and no LLM: events come from the local store (or the committed sample), briefs are
deterministic unless `provider` is given, and the forecaster is optional.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query

from chainwatch import __version__
from chainwatch.agent.brief import Brief, build_brief_deterministic, ground_brief
from chainwatch.agent.store import EventStore
from chainwatch.agent.tools import ToolContext, call_tool
from chainwatch.config import DOCS_DIR
from chainwatch.forecast.train import MODEL_DIR, TrainedModels, load_models
from chainwatch.graph.build import TradeGraph

app = FastAPI(title="ChainWatch", version=__version__,
              description="Supply chain disruption early warning for India to Europe lanes.")  # fmt: skip


@lru_cache(maxsize=1)
def get_store() -> EventStore:
    return EventStore.default()


@lru_cache(maxsize=1)
def get_forecaster() -> TrainedModels | None:
    path = MODEL_DIR / "forecast.joblib"
    return load_models() if path.exists() else None


def make_context(
    as_of: str | None = Query(None, description="YYYY-MM-DD; default now"),
    lookback_days: int = Query(30, ge=1, le=365),
    store: EventStore = Depends(get_store),
    forecaster: TrainedModels | None = Depends(get_forecaster),
) -> ToolContext:
    when = datetime.now(UTC)
    if as_of:
        try:
            day = datetime.strptime(as_of, "%Y-%m-%d").date()
        except ValueError as exc:
            raise HTTPException(422, "as_of must be YYYY-MM-DD") from exc
        when = datetime.combine(day, time(23, 59, 59), tzinfo=UTC)
    # A fresh graph per request: tool calls add events to it, and requests must not share state.
    return ToolContext(store=store, graph=TradeGraph(), forecaster=forecaster, as_of=when,
                       lookback_days=lookback_days)  # fmt: skip


@app.get("/health")
def health(store: EventStore = Depends(get_store)) -> dict:
    return {"status": "ok", "version": __version__, "events": len(store),
            "forecaster": get_forecaster() is not None}  # fmt: skip


@app.get("/events")
def events(
    text: str | None = None,
    min_severity: int = Query(1, ge=1, le=5),
    limit: int = Query(20, ge=1, le=50),
    ctx: ToolContext = Depends(make_context),
) -> dict:
    return call_tool(ctx, "search_events", {"text": text, "min_severity": min_severity,
                                            "limit": limit})  # fmt: skip


@app.get("/lanes")
def lanes(focus: str | None = None, ctx: ToolContext = Depends(make_context)) -> dict:
    return call_tool(ctx, "list_lanes", {"focus": focus})


@app.get("/risk")
def risk(
    lane_id: str | None = None,
    focus: str | None = None,
    with_delay_risk: bool = False,
    ctx: ToolContext = Depends(make_context),
) -> dict:
    """Exposure score per lane (or one lane), optionally with the ML baseline delay risk."""
    lane_ids = [lane_id] if lane_id else [
        lane["lane_id"] for lane in call_tool(ctx, "list_lanes", {"focus": focus})["lanes"]
    ]  # fmt: skip
    out = []
    for lid in lane_ids:
        result = call_tool(ctx, "lane_risk", {"lane_id": lid})
        if "error" in result:
            raise HTTPException(404, result["error"])
        if with_delay_risk:
            delay = call_tool(ctx, "delay_risk", {"lane_id": lid})
            result["late_probability"] = delay.get("late_probability")
        out.append(result)
    out.sort(key=lambda r: -r["exposure_score"])
    return {"as_of": ctx.as_of.date().isoformat(), "lanes": out}


@app.get("/brief", response_model=Brief)
def brief(
    focus: str | None = None,
    provider: str | None = Query(None, description="LLM provider; omit for deterministic brief"),
    model: str | None = None,
    ctx: ToolContext = Depends(make_context),
) -> Brief:
    if provider:
        from chainwatch.agent.loop import run_agent
        from chainwatch.llm import LLMError, get_llm_client

        try:
            return run_agent(get_llm_client(provider=provider, model=model), ctx, focus).brief
        except (LLMError, ValueError) as exc:
            raise HTTPException(503, f"LLM unavailable: {exc}") from exc
    return ground_brief(build_brief_deterministic(ctx, focus), ctx)


@app.get("/backtest")
def backtest() -> dict:
    """Saved backtest results (produced by `python -m chainwatch.backtest.run`)."""
    runs = {}
    for path in sorted((DOCS_DIR / "metrics").glob("backtest_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs[path.stem.removeprefix("backtest_")] = {
            "disruptions": {k: {kk: vv for kk, vv in v.items() if kk != "primary_signals"}
                            for k, v in data["disruptions"].items()},
            "controls": data["controls"],
            "days": data["days"],
        }  # fmt: skip
    return {"runs": runs}
