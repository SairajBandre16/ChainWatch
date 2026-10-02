import numpy as np
import pandas as pd
import pytest

from chainwatch.forecast import data as fd


@pytest.fixture(scope="module")
def orders() -> pd.DataFrame:
    return fd.synthetic_orders(n=1500, seed=1)


def test_no_leaky_columns_are_features() -> None:
    leaky_renamed = {fd.COLUMNS[c] for c in fd.LEAKY_RAW_COLUMNS if c in fd.COLUMNS}
    assert leaky_renamed == {"days_real", "late"}
    forbidden = leaky_renamed | {
        "label_known_at",
        "delivery_status",
        "order_status",
        "shipping_date",
    }
    assert not forbidden & set(fd.FEATURES)


def test_history_features_only_use_labels_known_before_the_order(orders) -> None:
    df = fd.add_history_features(orders)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(df), 60, replace=False):
        row = df.iloc[i]
        past = df[
            (df["label_known_at"] < row["order_date"])
            & (df["shipping_mode"] == row["shipping_mode"])
            & (df["order_region"] == row["order_region"])
        ]
        expected_rate = (past["late"].sum() + fd.PRIOR_STRENGTH * 0.5) / (
            len(past) + fd.PRIOR_STRENGTH
        )
        assert row["hist_mode_region_n"] == len(past)
        assert row["hist_mode_region_rate"] == pytest.approx(expected_rate)


def test_changing_future_labels_does_not_change_past_features(orders) -> None:
    cutoff = orders["order_date"].quantile(0.6)
    flipped = orders.copy()
    future = flipped["label_known_at"] >= cutoff
    flipped.loc[future, "late"] = 1 - flipped.loc[future, "late"]
    a = fd.add_history_features(orders)
    b = fd.add_history_features(flipped)
    before = a["order_date"] <= cutoff
    pd.testing.assert_frame_equal(a.loc[before, fd.HISTORY], b.loc[before, fd.HISTORY])


def test_time_split_is_ordered_and_purged(orders) -> None:
    split = fd.build_dataset(orders, valid_start="2016-07-01", test_start="2017-01-01")
    assert split.train["order_date"].max() < split.valid_start
    assert split.train["label_known_at"].max() < split.valid_start  # purged
    assert split.valid["order_date"].min() >= split.valid_start
    assert split.valid["label_known_at"].max() < split.test_start
    assert split.test["order_date"].min() >= split.test_start
    assert len(split.train) and len(split.valid) and len(split.test)


def test_prior_uses_only_pre_validation_labels(orders) -> None:
    flipped = orders.copy()
    late_rows = flipped["label_known_at"] >= pd.Timestamp("2016-07-01")
    flipped.loc[late_rows, "late"] = 1
    a = fd.build_dataset(orders, "2016-07-01", "2017-01-01").train
    b = fd.build_dataset(flipped, "2016-07-01", "2017-01-01").train
    pd.testing.assert_frame_equal(a[fd.HISTORY], b[fd.HISTORY])


def test_load_dataco_if_present() -> None:
    if not fd.DATACO_CSV.exists():
        pytest.skip("DataCo CSV not downloaded")
    df = fd.load_dataco()
    assert len(df) > 100_000
    assert set(fd.CATEGORICAL + ["days_scheduled", "late"]) <= set(df.columns)
