"""ChainWatch dashboard.

    streamlit run app/main.py

Runs fully offline from committed sample data: the event store sample, reference CSVs, saved metrics and
figures. The map uses no basemap tiles (lines and dots on a plain background) so it needs no network.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from chainwatch.agent.brief import brief_to_markdown, build_brief_deterministic, ground_brief
from chainwatch.agent.store import EventStore
from chainwatch.agent.tools import ToolContext, call_tool
from chainwatch.config import DOCS_DIR, PROCESSED_DIR
from chainwatch.forecast.train import MODEL_DIR, load_models
from chainwatch.graph.build import TradeGraph

st.set_page_config(page_title="ChainWatch", page_icon="🌊", layout="wide")

# Sequential blue ramp (light -> dark) for exposure magnitude; gray for "no exposure".
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
NO_EXPOSURE = "#b8b7b1"
INK_MUTED = "#52514e"


def hex_to_rgb(hex_color: str) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def exposure_color(score: float) -> list[int]:
    if score <= 0:
        return hex_to_rgb(NO_EXPOSURE) + [140]
    idx = min(int(score * len(RAMP)), len(RAMP) - 1)
    return hex_to_rgb(RAMP[idx]) + [235]


@st.cache_resource
def get_store() -> EventStore:
    return EventStore.default()


@st.cache_resource
def get_forecaster():
    return load_models() if (MODEL_DIR / "forecast.joblib").exists() else None


def make_ctx(as_of: date, lookback: int) -> ToolContext:
    when = datetime.combine(as_of, time(23, 59, 59), tzinfo=UTC)
    return ToolContext(store=get_store(), graph=TradeGraph(), forecaster=get_forecaster(),
                       as_of=when, lookback_days=lookback)  # fmt: skip


# --- sidebar ----------------------------------------------------------------------------------

store = get_store()
latest = max((e.published.date() for e in store.all()), default=date.today())
st.sidebar.title("ChainWatch")
st.sidebar.caption("Early warning for India to Europe trade lanes")
as_of = st.sidebar.date_input("As of", value=latest)
lookback = st.sidebar.slider("Lookback (days)", 1, 60, 30)
focus_options = ["all", "india_europe", "india_ireland", "asia_europe", "gulf_europe", "india_us",
                 "gulf_india", "asia_ireland"]  # fmt: skip
focus_pick = st.sidebar.selectbox("Lane focus", focus_options)
focus = None if focus_pick == "all" else focus_pick
st.sidebar.caption(f"{len(store)} events in store. Forecaster: "
                   f"{'loaded' if get_forecaster() else 'not trained'}.")  # fmt: skip

ctx = make_ctx(as_of, lookback)
tab_map, tab_events, tab_brief, tab_backtest, tab_forecast = st.tabs(
    ["Risk map", "Event feed", "Brief", "Backtest", "Forecast model"]
)

# --- risk map ---------------------------------------------------------------------------------

with tab_map:
    lanes = call_tool(ctx, "list_lanes", {"focus": focus})["lanes"]
    rows = []
    for lane in lanes:
        risk = call_tool(ctx, "lane_risk", {"lane_id": lane["lane_id"]})
        route = ctx.graph.lane_route(lane["lane_id"])
        path = [[ctx.graph.g.nodes[n]["lon"], ctx.graph.g.nodes[n]["lat"]] for n in route.path]
        rows.append({"lane_id": lane["lane_id"], "exposure": risk["exposure_score"], "path": path,
                     "days": route.days, "chokepoints": ", ".join(route.chokepoints),
                     "color": exposure_color(risk["exposure_score"]),
                     "events": len(risk["contributions"])})  # fmt: skip
    lanes_df = pd.DataFrame(rows).sort_values("exposure")  # draw high exposure on top
    nodes = [
        {"id": n, "name": d.get("name", n), "kind": d["kind"], "lon": d["lon"], "lat": d["lat"]}
        for n, d in ctx.graph.g.nodes(data=True)
        if d.get("kind") in {"port", "chokepoint"}
    ]
    nodes_df = pd.DataFrame(nodes)
    nodes_df["color"] = nodes_df["kind"].map(
        {"port": hex_to_rgb(INK_MUTED) + [200], "chokepoint": [11, 11, 11, 230]}
    )
    nodes_df["radius"] = nodes_df["kind"].map({"port": 2.5, "chokepoint": 4})  # pixels

    exposed = lanes_df[lanes_df["exposure"] > 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Lanes exposed", f"{len(exposed)} / {len(lanes_df)}")
    c2.metric("Highest exposure", f"{lanes_df['exposure'].max():.2f}" if len(lanes_df) else "-")
    c3.metric("Events in window", len(ctx.visible_events()))

    land = json.loads((PROCESSED_DIR / "ne_110m_land.geojson").read_text(encoding="utf-8"))
    deck = pdk.Deck(
        map_style=None,  # no tile server: the land outline below is bundled, so this works offline
        initial_view_state=pdk.ViewState(latitude=25, longitude=45, zoom=1.6),
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                land,
                get_fill_color=[214, 212, 204, 255],
                get_line_color=[184, 183, 177, 255],
                line_width_min_pixels=0.5,
            ),
            pdk.Layer(
                "PathLayer",
                lanes_df,
                get_path="path",
                get_color="color",
                width_units="pixels",
                get_width=3,
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                nodes_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                radius_units="pixels",
                stroked=True,
                get_line_color=[252, 252, 251],
                line_width_min_pixels=1,
                pickable=True,
            ),
        ],  # fmt: skip
        tooltip={"html": "<b>{lane_id}{name}</b><br/>exposure {exposure}<br/>{chokepoints}"},
    )
    st.pydeck_chart(deck, height=460)
    st.caption("Lane colour: exposure score (light blue = low, dark blue = high, gray = none). "
               "Black dots: chokepoints. Gray dots: ports. Schematic routes, not sailing tracks. "
               "Land: Natural Earth (public domain).")  # fmt: skip
    st.dataframe(
        lanes_df.sort_values("exposure", ascending=False)[
            ["lane_id", "exposure", "events", "days", "chokepoints"]
        ],
        hide_index=True, width="stretch",
    )  # fmt: skip

# --- events -----------------------------------------------------------------------------------

with tab_events:
    text = st.text_input("Filter (any keyword)", "")
    min_sev = st.slider("Minimum severity", 1, 5, 1)
    found = call_tool(ctx, "search_events", {"text": text or None, "min_severity": min_sev,
                                             "limit": 50})["events"]  # fmt: skip
    if found:
        st.dataframe(pd.DataFrame(found), hide_index=True, width="stretch",
                     column_config={"source_url": st.column_config.LinkColumn("source")})  # fmt: skip
    else:
        st.info("No events in this window. Move 'As of' or widen the lookback.")

# --- brief ------------------------------------------------------------------------------------

with tab_brief:
    st.caption(
        "Deterministic brief from the graph and tools. Every cited event exists in the store."
    )
    brief = ground_brief(build_brief_deterministic(ctx, focus), ctx)
    st.markdown(brief_to_markdown(brief))
    examples = sorted((DOCS_DIR / "examples").glob("brief_*.md"))
    if examples:
        with st.expander("Saved LLM agent briefs (real runs)"):
            pick = st.selectbox("Example", [p.name for p in examples])
            st.markdown((DOCS_DIR / "examples" / pick).read_text(encoding="utf-8"))

# --- backtest ---------------------------------------------------------------------------------

with tab_backtest:
    runs = sorted((DOCS_DIR / "metrics").glob("backtest_*.json"))
    if not runs:
        st.info("No backtest results yet. Run `python -m chainwatch.backtest.run`.")
    else:
        run_name = st.selectbox("Run", [p.stem.removeprefix("backtest_") for p in runs])
        data = json.loads((DOCS_DIR / "metrics" / f"backtest_{run_name}.json").read_text("utf-8"))
        summary = []
        for did, r in data["disruptions"].items():
            p = r["primary"]
            summary.append({"disruption": did, "onset": p["onset"],
                            "first warning": p["first_warning"] or "missed",
                            "lead time (days)": p["lead_time_days"],
                            "lanes warned": r["lanes_warned"],
                            "median lead (days)": r["median_lead_time_days"]})  # fmt: skip
        st.table(pd.DataFrame(summary).set_index("disruption"))
        primary = next(iter(data["controls"]))
        ctrl = data["controls"][primary]
        st.metric(f"False-alarm rate in control windows ({primary})",
                  f"{ctrl['false_alarm_rate']:.0%}",
                  help=f"{ctrl['flagged_days']} of {ctrl['days']} days, {ctrl['episodes']} episodes")  # fmt: skip
        for did, r in data["disruptions"].items():
            sig = pd.DataFrame(r["primary_signals"])
            sig["day"] = pd.to_datetime(sig["day"])
            onset = pd.DataFrame({"day": [pd.to_datetime(r["primary"]["onset"])]})
            base = alt.Chart(sig).encode(x=alt.X("day:T", title=None))
            line = base.mark_line(strokeWidth=2, color=RAMP[2]).encode(
                y=alt.Y("score:Q", title="exposure score", scale=alt.Scale(domain=[0, 1]))
            )
            hover = alt.selection_point(on="pointerover", nearest=True, fields=["day"],
                                        empty=False)  # fmt: skip
            points = (
                base.mark_circle(size=70, color=RAMP[2])
                .encode(
                    y="score:Q",
                    opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                    tooltip=[
                        alt.Tooltip("day:T"),
                        alt.Tooltip("score:Q", format=".2f"),
                        alt.Tooltip("on_target:N", title="on target"),
                    ],  # fmt: skip
                )
                .add_params(hover)
            )
            threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
                strokeDash=[4, 4], color=INK_MUTED).encode(y="y:Q")  # fmt: skip
            onset_rule = (
                alt.Chart(onset).mark_rule(color="#d03b3b", strokeWidth=2).encode(x="day:T")
            )
            st.markdown(f"**{did}** — primary lane exposure per day (dashed: flag threshold 0.5; "
                        "red: disruption onset)")  # fmt: skip
            st.altair_chart((line + points + threshold + onset_rule).properties(height=220),
                            width="stretch")  # fmt: skip
        st.caption(
            "Definitions are fixed in docs/backtest-spec.md. Numbers come from saved runs only."
        )

# --- forecast ---------------------------------------------------------------------------------

with tab_forecast:
    metrics_path = DOCS_DIR / "metrics" / "forecast_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        table = [{"model": name, **{k: v["test"][k] for k in ("roc_auc", "pr_auc", "brier",
                                                                "accuracy")}}
                 for name, v in m["models"].items()]  # fmt: skip
        st.markdown("Late-delivery models on the DataCo **test** split (2017-07 to 2018-01).")
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    for fig in ("shap_importance.png", "shap_beeswarm.png"):
        path = DOCS_DIR / "figures" / fig
        if path.exists():
            st.image(str(path))
    if get_forecaster() is not None:
        st.subheader("What-if: baseline delay risk for a lane")
        lane_pick = st.selectbox("Lane", [lane["lane_id"] for lane in
                                          call_tool(ctx, "list_lanes", {})["lanes"]])  # fmt: skip
        mode = st.selectbox("Shipping mode", ["Standard Class", "Second Class", "First Class",
                                              "Same Day"])  # fmt: skip
        out = call_tool(ctx, "delay_risk", {"lane_id": lane_pick, "shipping_mode": mode})
        st.write(out.get("explanation", out.get("error")))
        st.caption(out.get("note", ""))
