import numpy as np
import pytest

from chainwatch.forecast import data as fd
from chainwatch.forecast import train as ft


@pytest.fixture(scope="module")
def trained():
    split = fd.build_dataset(fd.synthetic_orders(3000, seed=3), "2016-07-01", "2017-01-01")
    return split, ft.train_all(split)


def test_all_models_reported_with_baselines(trained) -> None:
    _, models = trained
    assert set(models.results) == {"majority", "mode_history", "logreg", "lightgbm"}
    assert models.results["majority"]["test"]["roc_auc"] == 0.5


def test_models_beat_majority_on_synthetic_signal(trained) -> None:
    _, models = trained
    r = models.results
    assert r["lightgbm"]["test"]["roc_auc"] > 0.6
    assert r["logreg"]["test"]["roc_auc"] > 0.6
    assert r["lightgbm"]["test"]["brier"] < r["majority"]["test"]["brier"]


def test_unseen_categories_are_handled(trained) -> None:
    split, models = trained
    probs = models.predict_lgb(split.test.assign(order_country="Atlantis"))
    assert np.all((probs > 0) & (probs < 1))


def test_metrics_constant_prediction() -> None:
    m = ft.metrics(np.array([0, 1, 1, 0]), np.full(4, 0.5))
    assert m["roc_auc"] == 0.5 and m["accuracy"] == 0.5
