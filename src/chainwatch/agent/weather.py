"""Weather risk at a port or waypoint from Open-Meteo (free, no API key).

Responses are cached per (coordinates, day) through `CachedHttp`, so repeated agent runs on the same day
cost nothing and tests run offline from a stubbed HTTP layer.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel

from chainwatch.ingest.http_cache import CachedHttp

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Beaufort 8 (gale) starts at 62 km/h sustained wind; ports often suspend crane work around there.
GALE_KMH = 62.0
STRONG_KMH = 39.0  # Beaufort 6
HEAVY_RAIN_MM = 50.0


class DayWeather(BaseModel):
    date: str
    wind_max_kmh: float | None
    gust_max_kmh: float | None
    precipitation_mm: float | None


class WeatherRisk(BaseModel):
    lat: float
    lon: float
    days: list[DayWeather]
    level: str  # "low" | "elevated" | "high"
    reason: str


def classify(days: list[DayWeather]) -> tuple[str, str]:
    """Simple, documented thresholds -> (level, reason)."""
    winds = [(d.wind_max_kmh or 0.0, d.date) for d in days]
    rains = [(d.precipitation_mm or 0.0, d.date) for d in days]
    top_wind, wind_day = max(winds, default=(0.0, ""))
    top_rain, rain_day = max(rains, default=(0.0, ""))
    if top_wind >= GALE_KMH:
        return "high", f"gale-force wind {top_wind:.0f} km/h on {wind_day}"
    if top_rain >= HEAVY_RAIN_MM:
        return "high", f"heavy rain {top_rain:.0f} mm on {rain_day}"
    if top_wind >= STRONG_KMH:
        return "elevated", f"strong wind {top_wind:.0f} km/h on {wind_day}"
    return "low", f"max wind {top_wind:.0f} km/h, max rain {top_rain:.0f} mm"


def get_weather_risk(
    lat: float, lon: float, days: int = 7, http: CachedHttp | None = None
) -> WeatherRisk:
    http = http or CachedHttp()
    params = {
        "latitude": f"{lat:.2f}",
        "longitude": f"{lon:.2f}",
        "daily": "wind_speed_10m_max,wind_gusts_10m_max,precipitation_sum",
        "timezone": "UTC",
        "forecast_days": str(days),
    }
    # The forecast changes daily, so today's date is part of the cache key (but not sent upstream).
    today = {"day": datetime.now(UTC).strftime("%Y-%m-%d")}
    payload = json.loads(http.get(FORECAST_URL, params=params, cache_extra=today))
    daily = payload.get("daily", {})
    rows = [
        DayWeather(date=d, wind_max_kmh=w, gust_max_kmh=g, precipitation_mm=p)
        for d, w, g, p in zip(
            daily.get("time", []),
            daily.get("wind_speed_10m_max", []),
            daily.get("wind_gusts_10m_max", []),
            daily.get("precipitation_sum", []),
            strict=False,
        )
    ]
    level, reason = classify(rows)
    return WeatherRisk(lat=lat, lon=lon, days=rows, level=level, reason=reason)
