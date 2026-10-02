"""Explain late-delivery predictions with SHAP values.

LightGBM computes exact TreeSHAP values itself (`pred_contrib=True`), the same numbers
`shap.TreeExplainer` gives, without converting categorical columns. Each prediction's log-odds equals
the base value plus the sum of feature contributions, so the explanation adds up exactly.

    python -m chainwatch.forecast.explain     # SHAP plots -> docs/figures/, example explanations
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel

from chainwatch.config import DOCS_DIR
from chainwatch.forecast import data as fd
from chainwatch.forecast.train import TrainedModels, load_models, to_lgb_frame

FIGURE_DIR = DOCS_DIR / "figures"

FRIENDLY = {
    "shipping_mode": "shipping mode",
    "days_scheduled": "scheduled shipping days",
    "market": "market",
    "order_region": "order region",
    "order_country": "destination country",
    "customer_segment": "customer segment",
    "category": "product category",
    "department": "department",
    "payment_type": "payment type",
    "quantity": "quantity",
    "unit_price": "unit price",
    "discount_rate": "discount rate",
    "sales": "order value",
    "order_hour": "order hour",
    "order_dow": "order weekday",
    "order_month": "order month",
    "hist_mode_region_rate": "recent late rate for this mode and region",
    "hist_mode_region_n": "history size for this mode and region",
    "hist_country_rate": "recent late rate for this country",
    "hist_country_n": "history size for this country",
    "hist_mode_rate": "recent late rate for this shipping mode",
    "hist_mode_n": "history size for this shipping mode",
}


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class Driver(BaseModel):
    feature: str
    value: str
    contribution: float  # log-odds; positive raises late risk

    @property
    def direction(self) -> str:
        return "raises" if self.contribution > 0 else "lowers"


class Explanation(BaseModel):
    probability: float
    base_probability: float
    drivers: list[Driver]
    text: str


def shap_matrix(models: TrainedModels, df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """(n_rows x n_features) SHAP values in log-odds, and the base value (same for all rows)."""
    contrib = models.booster.predict(to_lgb_frame(df, models.categories), pred_contrib=True)
    return contrib[:, :-1], float(contrib[0, -1])


def _fmt_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 100 else f"{value:,.0f}"
    return str(value)


def explain(models: TrainedModels, order: pd.Series | pd.DataFrame, top_k: int = 3) -> Explanation:
    """Plain-text explanation of one order's late-delivery risk: the top_k strongest drivers."""
    row = order.to_frame().T if isinstance(order, pd.Series) else order.iloc[[0]]
    row = row.astype({c: float for c in fd.NUMERIC + fd.HISTORY})
    values, base = shap_matrix(models, row)
    contrib = values[0]
    order_idx = np.argsort(-np.abs(contrib))[:top_k]
    drivers = [
        Driver(feature=fd.FEATURES[i], value=_fmt_value(row.iloc[0][fd.FEATURES[i]]),
               contribution=round(float(contrib[i]), 3))
        for i in order_idx
    ]  # fmt: skip
    prob = _sigmoid(base + float(contrib.sum()))
    base_prob = _sigmoid(base)
    parts = [
        f"{FRIENDLY.get(d.feature, d.feature)} = {d.value} {d.direction} risk "
        f"({d.contribution:+.2f} log-odds)"
        for d in drivers
    ]
    text = (
        f"Late-delivery risk {prob:.0%} (model baseline {base_prob:.0%}). "
        f"Main drivers: {'; '.join(parts)}."
    )
    return Explanation(probability=round(prob, 4), base_probability=round(base_prob, 4),
                       drivers=drivers, text=text)  # fmt: skip


def global_importance(models: TrainedModels, df: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, largest first."""
    values, _ = shap_matrix(models, df)
    return pd.Series(np.abs(values).mean(axis=0), index=fd.FEATURES).sort_values(ascending=False)


def save_plots(models: TrainedModels, df: pd.DataFrame, n: int = 3000, seed: int = 42) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    sample = df.sample(min(n, len(df)), random_state=seed)
    values, _ = shap_matrix(models, sample)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    imp = pd.Series(np.abs(values).mean(axis=0), index=fd.FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([FRIENDLY.get(f, f) for f in imp.index], imp.values, color="#3b6ea5")
    ax.set_xlabel("mean |SHAP value| (log-odds of late delivery)")
    ax.set_title("Drivers of late-delivery risk\n(LightGBM, DataCo test split)")
    fig.tight_layout()
    bar_path = FIGURE_DIR / "shap_importance.png"
    fig.savefig(bar_path, dpi=120)
    plt.close(fig)

    # Beeswarm needs numbers for colouring; categorical columns are shown as category codes.
    display = sample[fd.FEATURES].copy()
    for col in fd.CATEGORICAL:
        display[col] = pd.Categorical(display[col]).codes
    display.columns = [FRIENDLY.get(c, c) for c in display.columns]
    shap.summary_plot(values, display, show=False, max_display=12)
    swarm_path = FIGURE_DIR / "shap_beeswarm.png"
    plt.gcf().set_size_inches(8, 6)
    plt.tight_layout()
    plt.savefig(swarm_path, dpi=120)
    plt.close("all")
    return [str(bar_path), str(swarm_path)]


def main() -> None:
    models = load_models()
    split = fd.build_dataset(fd.load_dataco())
    test = split.test
    print("Global importance (mean |SHAP|, test split):")
    print(
        global_importance(models, test.sample(5000, random_state=42)).round(3).head(10).to_string()
    )
    print("Plots:", save_plots(models, test))
    probs = models.predict_lgb(test)
    for label, idx in (
        ("highest risk", int(np.argmax(probs))),
        ("lowest risk", int(np.argmin(probs))),
    ):
        print(f"\n[{label}] {explain(models, test.iloc[idx]).text}")


if __name__ == "__main__":
    main()
