"""Score "what-if" orders for a trade lane with the trained forecaster.

The agent asks: "what is the baseline late-delivery risk for Standard Class orders to Ireland?"
We build one feature row from the training snapshot (typical values, the destination's DataCo
country name and region, end-of-training history rates) and run the LightGBM model on it.

Note: DataCo is retail order data. This number is a *baseline* delivery risk by destination and
shipping mode; disruption exposure comes from the graph, and the brief reports both side by side.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from chainwatch.forecast import data as fd
from chainwatch.forecast.explain import Explanation, explain
from chainwatch.forecast.train import TrainedModels

# ISO country -> DataCo `Order Country` (DataCo uses Spanish names).
ISO_TO_DATACO = {
    "IE": "Irlanda", "GB": "Reino Unido", "NL": "Países Bajos", "BE": "Bélgica",
    "DE": "Alemania", "FR": "Francia", "ES": "España", "IT": "Italia", "GR": "Grecia",
    "PL": "Polonia", "PT": "Portugal", "IN": "India", "CN": "China", "SG": "Singapur",
    "US": "Estados Unidos", "AE": "Emiratos Árabes Unidos", "LK": "SriLanka", "MT": "Malta",
    "SI": "Eslovenia", "MA": "Marruecos", "EG": "Egipto", "ZA": "Sudáfrica",
}  # fmt: skip

TYPICAL_SCHEDULED_DAYS = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}


class LaneRisk(BaseModel):
    destination_iso: str
    dataco_country: str | None
    shipping_mode: str
    probability: float
    explanation: Explanation
    note: str


def build_order_row(
    models: TrainedModels, destination_iso: str, shipping_mode: str = "Standard Class"
) -> pd.DataFrame:
    row = dict(models.defaults)
    country = ISO_TO_DATACO.get(destination_iso)
    if country is not None and country in models.country_region:
        row["order_country"] = country
        row["order_region"] = models.country_region[country]
    row["shipping_mode"] = shipping_mode
    row["days_scheduled"] = float(TYPICAL_SCHEDULED_DAYS.get(shipping_mode, row["days_scheduled"]))
    for name, keys in fd.HISTORY_KEYS.items():
        hist = models.history[name]
        key = tuple(row[k] for k in keys)
        key = key[0] if len(key) == 1 else key
        row[f"{name}_rate"] = float(hist["rate"].get(key, hist["prior"]))
        row[f"{name}_n"] = float(hist["n"].get(key, 0.0))
    return pd.DataFrame([row])[fd.FEATURES]


def lane_delay_risk(
    models: TrainedModels, destination_iso: str, shipping_mode: str = "Standard Class"
) -> LaneRisk:
    row = build_order_row(models, destination_iso, shipping_mode)
    exp = explain(models, row)
    country = row.iloc[0]["order_country"]
    known = ISO_TO_DATACO.get(destination_iso) == country
    note = (
        f"Baseline late-delivery risk from DataCo order history ({country}, {shipping_mode})."
        if known
        else f"No DataCo history for {destination_iso}; used typical order values."
    )
    return LaneRisk(destination_iso=destination_iso, dataco_country=country if known else None,
                    shipping_mode=shipping_mode, probability=exp.probability, explanation=exp,
                    note=note)  # fmt: skip
