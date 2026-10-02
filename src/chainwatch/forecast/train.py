"""Train and compare late-delivery models on time-based splits.

Models (always reported side by side):
  - majority      predicts the training late rate for everyone (the "do nothing" floor)
  - mode_history  uses the point-in-time late rate of the shipping mode as the score (a one-rule heuristic)
  - logreg        logistic regression on one-hot categoricals + scaled numerics
  - lightgbm      gradient-boosted trees; rounds picked by early stopping on the validation split

The test split is touched once, after all choices were made on validation.

    python -m chainwatch.forecast.train            # real DataCo data
    python -m chainwatch.forecast.train --synthetic  # tiny synthetic run (pipeline check only)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from chainwatch.config import ROOT_DIR
from chainwatch.forecast import data as fd

SEED = 42
MODEL_DIR = ROOT_DIR / "models"
METRICS_PATH = ROOT_DIR / "docs" / "metrics" / "forecast_metrics.json"

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "verbose": -1,
}


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    constant = np.allclose(p, p[0])
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if not constant else 0.5,
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "accuracy": round(float(accuracy_score(y, p >= 0.5)), 4),
        "n": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 4),
    }


def to_lgb_frame(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Categoricals as pandas `category` with categories fixed from training (unseen -> NaN)."""
    X = df[fd.FEATURES].copy()
    for col in fd.CATEGORICAL:
        known = X[col].where(X[col].isin(categories[col]))  # unseen values become NaN explicitly
        X[col] = pd.Categorical(known, categories=categories[col])
    return X


def make_logreg() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
             fd.CATEGORICAL),
            ("num", StandardScaler(), fd.NUMERIC + fd.HISTORY),
        ]
    )  # fmt: skip
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


@dataclass
class TrainedModels:
    logreg: Pipeline
    booster: lgb.Booster
    categories: dict[str, list[str]]
    train_rate: float
    results: dict = field(default_factory=dict)
    # Snapshot of the training data used to score "what-if" orders (see forecast.predict).
    defaults: dict = field(default_factory=dict)
    country_region: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    def predict_lgb(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(to_lgb_frame(df, self.categories))


def train_all(split: fd.Split) -> TrainedModels:
    tr, va, te = split.train, split.valid, split.test
    y_tr, y_va, y_te = (d[fd.TARGET].to_numpy() for d in (tr, va, te))
    results: dict[str, dict] = {}

    train_rate = float(y_tr.mean())
    results["majority"] = {
        "valid": metrics(y_va, np.full(len(va), train_rate)),
        "test": metrics(y_te, np.full(len(te), train_rate)),
    }
    results["mode_history"] = {
        "valid": metrics(y_va, va["hist_mode_rate"].to_numpy()),
        "test": metrics(y_te, te["hist_mode_rate"].to_numpy()),
    }

    t0 = time.perf_counter()
    logreg = make_logreg().fit(tr[fd.FEATURES], y_tr)
    results["logreg"] = {
        "valid": metrics(y_va, logreg.predict_proba(va[fd.FEATURES])[:, 1]),
        "test": metrics(y_te, logreg.predict_proba(te[fd.FEATURES])[:, 1]),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }

    categories = {c: sorted(tr[c].unique().tolist()) for c in fd.CATEGORICAL}
    t0 = time.perf_counter()
    dtrain = lgb.Dataset(to_lgb_frame(tr, categories), y_tr, categorical_feature=fd.CATEGORICAL)
    dvalid = lgb.Dataset(to_lgb_frame(va, categories), y_va, reference=dtrain)
    booster = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    models = TrainedModels(logreg, booster, categories, train_rate)
    results["lightgbm"] = {
        "valid": metrics(y_va, models.predict_lgb(va)),
        "test": metrics(y_te, models.predict_lgb(te)),
        "best_iteration": int(booster.best_iteration),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    models.results = results
    models.defaults, models.country_region, models.history = training_snapshot(tr, train_rate)
    return models


def training_snapshot(tr: pd.DataFrame, prior: float) -> tuple[dict, dict, dict]:
    """Typical feature values, country -> region map, and end-of-training history rates per group.

    Only training rows are used, so a what-if prediction never sees validation or test outcomes.
    """
    defaults = {c: tr[c].mode().iloc[0] for c in fd.CATEGORICAL}
    defaults |= {c: float(tr[c].median()) for c in fd.NUMERIC}
    country_region = tr.groupby("order_country")["order_region"].agg(lambda s: s.mode().iloc[0])
    history: dict = {}
    for name, keys in fd.HISTORY_KEYS.items():
        grouped = tr.groupby(keys)[fd.TARGET].agg(["sum", "count"])
        rate = (grouped["sum"] + fd.PRIOR_STRENGTH * prior) / (grouped["count"] + fd.PRIOR_STRENGTH)
        history[name] = {
            "rate": rate.to_dict(),
            "n": grouped["count"].astype(float).to_dict(),
            "prior": prior,
        }
    return defaults, country_region.to_dict(), history


def save(models: TrainedModels, split: fd.Split, path: Path = METRICS_PATH) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(models, MODEL_DIR / "forecast.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "splits": {
            name: {
                "rows": len(df),
                "from": str(df["order_date"].min().date()),
                "to": str(df["order_date"].max().date()),
            }
            for name, df in (("train", split.train), ("valid", split.valid), ("test", split.test))
        },
        "models": models.results,
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_models() -> TrainedModels:
    return joblib.load(MODEL_DIR / "forecast.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="pipeline check only, not results")
    args = parser.parse_args()
    if args.synthetic:
        split = fd.build_dataset(fd.synthetic_orders(4000), "2016-07-01", "2017-01-01")
        models = train_all(split)
        print(json.dumps({k: v["test"] for k, v in models.results.items()}, indent=1))
        return
    # Import ourselves by package name: under `python -m`, classes defined here live in __main__,
    # and a pickle of a __main__ class cannot be loaded from anywhere else.
    from chainwatch.forecast import train as module

    split = fd.build_dataset(fd.load_dataco())
    models = module.train_all(split)
    module.save(models, split)
    rows = [(name, r["valid"], r["test"]) for name, r in models.results.items()]
    print(f"{'model':14} {'valid AUC':>9} {'test AUC':>9} {'test PR-AUC':>11} {'test Brier':>10}")
    for name, va, te in rows:
        print(
            f"{name:14} {va['roc_auc']:9.4f} {te['roc_auc']:9.4f} {te['pr_auc']:11.4f} {te['brier']:10.4f}"
        )
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
