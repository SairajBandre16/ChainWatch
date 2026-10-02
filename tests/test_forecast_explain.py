import numpy as np
import pytest

from chainwatch.forecast import data as fd
from chainwatch.forecast import explain as fx
from chainwatch.forecast import train as ft


@pytest.fixture(scope="module")
def setup():
    split = fd.build_dataset(fd.synthetic_orders(3000, seed=5), "2016-07-01", "2017-01-01")
    return split, ft.train_all(split)


def test_shap_values_add_up_to_prediction(setup) -> None:
    split, models = setup
    rows = split.test.head(50)
    values, base = fx.shap_matrix(models, rows)
    logits = base + values.sum(axis=1)
    np.testing.assert_allclose(1 / (1 + np.exp(-logits)), models.predict_lgb(rows), rtol=1e-6)


def test_explain_returns_plain_text_top_drivers(setup) -> None:
    split, models = setup
    row = split.test.iloc[0]
    exp = fx.explain(models, row, top_k=3)
    assert exp.probability == pytest.approx(models.predict_lgb(split.test.head(1))[0], abs=1e-4)
    assert len(exp.drivers) == 3
    contribs = [abs(d.contribution) for d in exp.drivers]
    assert contribs == sorted(contribs, reverse=True)
    assert exp.text.startswith("Late-delivery risk")
    assert fx.FRIENDLY[exp.drivers[0].feature] in exp.text
    assert ("raises" in exp.text) or ("lowers" in exp.text)


def test_synthetic_signal_is_found(setup) -> None:
    split, models = setup
    imp = fx.global_importance(models, split.test)
    # The synthetic generator makes lateness depend on shipping mode; it must rank near the top.
    assert "shipping_mode" in imp.index[:3] or "hist_mode_rate" in imp.index[:3]
