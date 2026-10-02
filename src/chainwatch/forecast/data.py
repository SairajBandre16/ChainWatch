"""DataCo loading, leak-free feature building, and time-based splits.

Dataset: "DataCo Smart Supply Chain for Big Data Analysis" (Constante et al., 2019, CC BY 4.0),
downloaded from Mendeley Data (no login) by `download_dataco()` into data/raw/dataco/.

Target: `late` = DataCo's Late_delivery_risk (1 if actual shipping days > scheduled days).

Leakage rules, enforced by tests:
  1. Only columns known when the order is placed are features. Outcome columns (real shipping days,
     delivery status, shipping date, order status) are dropped.
  2. An order's label becomes known on its shipping date (= order date + real shipping days).
     History features for an order only use orders whose label was known *before* that order was placed.
  3. Splits are by order date, and training rows whose label was not yet known at the split
     cutoff are purged, so the model never trains on outcomes that were still in the future.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chainwatch.config import RAW_DIR

DATACO_DIR = RAW_DIR / "dataco"
DATACO_CSV = DATACO_DIR / "DataCoSupplyChainDataset.csv"
DATACO_URL = (
    "https://data.mendeley.com/public-files/datasets/8gx2fvg2k6/files/"
    "72784be5-36d3-44fe-b75d-0edbf1999f65/file_downloaded"
)
DATACO_SHA256 = "fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6"

# Raw DataCo column -> our name. Only order-time columns plus what we need to build the label.
COLUMNS = {
    "order date (DateOrders)": "order_date",
    "Days for shipping (real)": "days_real",  # outcome: used for label timing only, never a feature
    "Days for shipment (scheduled)": "days_scheduled",
    "Late_delivery_risk": "late",
    "Shipping Mode": "shipping_mode",
    "Market": "market",
    "Order Region": "order_region",
    "Order Country": "order_country",
    "Customer Segment": "customer_segment",
    "Category Name": "category",
    "Department Name": "department",
    "Type": "payment_type",
    "Order Item Quantity": "quantity",
    "Order Item Product Price": "unit_price",
    "Order Item Discount Rate": "discount_rate",
    "Sales": "sales",
}

# Columns that reveal the outcome. Listed so the leakage test can assert none become features.
LEAKY_RAW_COLUMNS = (
    "Days for shipping (real)",
    "Delivery Status",
    "Late_delivery_risk",
    "shipping date (DateOrders)",
    "Order Status",
)

CATEGORICAL = [
    "shipping_mode", "market", "order_region", "order_country",
    "customer_segment", "category", "department", "payment_type",
]  # fmt: skip
NUMERIC = ["days_scheduled", "quantity", "unit_price", "discount_rate", "sales",
           "order_hour", "order_dow", "order_month"]  # fmt: skip
HISTORY_KEYS = {
    "hist_mode_region": ["shipping_mode", "order_region"],
    "hist_country": ["order_country"],
    "hist_mode": ["shipping_mode"],
}
HISTORY = [f"{name}_{stat}" for name in HISTORY_KEYS for stat in ("rate", "n")]
FEATURES = CATEGORICAL + NUMERIC + HISTORY
TARGET = "late"

PRIOR_STRENGTH = 20.0  # Bayesian smoothing: groups with little history shrink to the global rate


def download_dataco(path: Path = DATACO_CSV) -> Path:
    """Download the public CSV (~96 MB) once and verify its checksum."""
    import httpx

    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", DATACO_URL, follow_redirects=True, timeout=600) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes():
                digest.update(chunk)
                fh.write(chunk)
    if digest.hexdigest() != DATACO_SHA256:
        tmp.unlink()
        raise ValueError("DataCo checksum mismatch; download corrupted or file changed upstream")
    tmp.rename(path)
    return path


def load_dataco(path: Path = DATACO_CSV) -> pd.DataFrame:
    """Read the raw CSV (latin-1 encoded) and keep only the columns we use, renamed."""
    raw = pd.read_csv(path, encoding="latin1", usecols=list(COLUMNS))
    return prepare(raw.rename(columns=COLUMNS))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, add label timing and calendar features. Input uses our column names."""
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    # The label is known once the order has shipped; this drives every point-in-time rule.
    df["label_known_at"] = df["order_date"] + pd.to_timedelta(df["days_real"], unit="D")
    df["order_hour"] = df["order_date"].dt.hour
    df["order_dow"] = df["order_date"].dt.dayofweek
    df["order_month"] = df["order_date"].dt.month
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df.sort_values("order_date", kind="stable").reset_index(drop=True)


def _history_for_group(df: pd.DataFrame, keys: list[str], global_rate: float) -> pd.DataFrame:
    """Point-in-time late rate per group: for each order, use only labels known before it.

    For each group: sort known labels by `label_known_at`, take cumulative sums, then `merge_asof`
    each order onto the last label known strictly before its order date.
    """
    known = df[[*keys, "label_known_at", TARGET]].sort_values("label_known_at")
    known = known.assign(
        cum_late=known.groupby(keys)[TARGET].cumsum(),
        cum_n=known.groupby(keys).cumcount() + 1,
    )
    orders = df[[*keys, "order_date"]].reset_index().sort_values("order_date")
    merged = (
        pd.merge_asof(
            orders,
            known[[*keys, "label_known_at", "cum_late", "cum_n"]],
            left_on="order_date",
            right_on="label_known_at",
            by=keys,
            allow_exact_matches=False,  # strictly before: a label known at the same instant is excluded
        )
        .set_index("index")
        .sort_index()
    )
    n = merged["cum_n"].fillna(0)
    late = merged["cum_late"].fillna(0)
    rate = (late + PRIOR_STRENGTH * global_rate) / (n + PRIOR_STRENGTH)
    return pd.DataFrame({"rate": rate, "n": n}, index=df.index)


def add_history_features(df: pd.DataFrame, prior_until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Add HISTORY columns. The smoothing prior uses only labels known before `prior_until`
    (default: the first order date, i.e. a neutral 0.5) so it cannot leak test outcomes."""
    df = df.copy()
    if prior_until is None:
        global_rate = 0.5
    else:
        seen = df.loc[df["label_known_at"] < prior_until, TARGET]
        global_rate = float(seen.mean()) if len(seen) else 0.5
    for name, keys in HISTORY_KEYS.items():
        hist = _history_for_group(df, keys, global_rate)
        df[f"{name}_rate"] = hist["rate"].astype(float)
        df[f"{name}_n"] = hist["n"].astype(float)
    return df


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    valid_start: pd.Timestamp
    test_start: pd.Timestamp


def time_split(
    df: pd.DataFrame,
    valid_start: str = "2017-01-01",
    test_start: str = "2017-07-01",
) -> Split:
    """Order-date split with purging: a training row is kept only if its label was known before
    the validation period starts (same for valid rows before the test period)."""
    vs, ts = pd.Timestamp(valid_start), pd.Timestamp(test_start)
    train = df[(df["order_date"] < vs) & (df["label_known_at"] < vs)]
    valid = df[(df["order_date"] >= vs) & (df["order_date"] < ts) & (df["label_known_at"] < ts)]
    test = df[df["order_date"] >= ts]
    return Split(train=train, valid=valid, test=test, valid_start=vs, test_start=ts)


def build_dataset(df: pd.DataFrame, valid_start: str = "2017-01-01",
                  test_start: str = "2017-07-01") -> Split:  # fmt: skip
    """prepare() output -> history features -> purged time split."""
    with_hist = add_history_features(df, prior_until=pd.Timestamp(valid_start))
    return time_split(with_hist, valid_start, test_start)


# --- synthetic data: TESTS ONLY ---------------------------------------------------------------


def synthetic_orders(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Small DataCo-shaped frame for unit tests. NOT real data; never used for reported metrics.

    Lateness depends on shipping mode and region so models have signal to find.
    """
    rng = np.random.default_rng(seed)
    modes = np.array(["Standard Class", "Second Class", "First Class", "Same Day"])
    sched = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}
    regions = np.array(["Western Europe", "South Asia", "West of USA", "Northern Europe"])
    mode = rng.choice(modes, n, p=[0.6, 0.2, 0.15, 0.05])
    region = rng.choice(regions, n)
    p_late = (np.select([mode == "First Class", mode == "Second Class"], [0.9, 0.7], 0.35)
              + np.where(region == "South Asia", 0.1, 0.0))  # fmt: skip
    late = rng.random(n) < np.clip(p_late, 0, 1)
    days_scheduled = np.array([sched[m] for m in mode])
    days_real = days_scheduled + np.where(late, rng.integers(1, 3, n), -rng.integers(0, 2, n))
    days_real = np.clip(days_real, 0, None)
    start = pd.Timestamp("2015-01-01")
    order_date = start + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 365, n)), unit="D")
    df = pd.DataFrame(
        {
            "order_date": order_date,
            "days_real": days_real,
            "days_scheduled": days_scheduled,
            "late": late.astype(int),
            "shipping_mode": mode,
            "market": "Europe",
            "order_region": region,
            "order_country": np.where(region == "South Asia", "India", "Ireland"),
            "customer_segment": rng.choice(["Consumer", "Corporate"], n),
            "category": rng.choice(["Cleats", "Electronics", "Toys"], n),
            "department": "Fan Shop",
            "payment_type": rng.choice(["DEBIT", "TRANSFER"], n),
            "quantity": rng.integers(1, 5, n),
            "unit_price": rng.uniform(10, 300, n).round(2),
            "discount_rate": rng.uniform(0, 0.25, n).round(2),
            "sales": rng.uniform(10, 1000, n).round(2),
        }
    )
    return prepare(df)
