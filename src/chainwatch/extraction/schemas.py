"""Pydantic models for disruption events.

Two layers:
  - `ExtractedEvent` / `ExtractionOutput`: exactly what the LLM must return (no bookkeeping fields).
  - `DisruptionEvent`: an extracted event plus provenance (source item, model, prompt version).
Keeping the LLM-facing schema small makes it easier for small local models to fill in correctly.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EventType(StrEnum):
    PORT_CLOSURE = "port_closure"
    PORT_CONGESTION = "port_congestion"
    CANAL_OR_STRAIT_BLOCKAGE = "canal_or_strait_blockage"
    LABOR_STRIKE = "labor_strike"
    SEVERE_WEATHER = "severe_weather"
    CONFLICT_OR_ATTACK = "conflict_or_attack"
    ACCIDENT = "accident"
    SANCTIONS_OR_REGULATION = "sanctions_or_regulation"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    OTHER = "other"


# Common LLM phrasings mapped onto the enum, so a near-miss is accepted instead of retried.
_TYPE_ALIASES = {
    "strike": EventType.LABOR_STRIKE,
    "labour_strike": EventType.LABOR_STRIKE,
    "labor_dispute": EventType.LABOR_STRIKE,
    "weather": EventType.SEVERE_WEATHER,
    "storm": EventType.SEVERE_WEATHER,
    "natural_disaster": EventType.SEVERE_WEATHER,
    "congestion": EventType.PORT_CONGESTION,
    "closure": EventType.PORT_CLOSURE,
    "blockage": EventType.CANAL_OR_STRAIT_BLOCKAGE,
    "canal_blockage": EventType.CANAL_OR_STRAIT_BLOCKAGE,
    "conflict": EventType.CONFLICT_OR_ATTACK,
    "attack": EventType.CONFLICT_OR_ATTACK,
    "geopolitical": EventType.CONFLICT_OR_ATTACK,
    "piracy": EventType.CONFLICT_OR_ATTACK,
    "sanctions": EventType.SANCTIONS_OR_REGULATION,
    "regulation": EventType.SANCTIONS_OR_REGULATION,
    "collision": EventType.ACCIDENT,
    "grounding": EventType.ACCIDENT,
    "fire": EventType.ACCIDENT,
    "outage": EventType.INFRASTRUCTURE_FAILURE,
    "cyberattack": EventType.INFRASTRUCTURE_FAILURE,
}


class ExtractedEvent(BaseModel):
    """One disruption as the LLM reports it."""

    event_type: EventType
    location: str = Field(min_length=1, description="Most specific place named, e.g. 'Red Sea'")
    country_code: str | None = Field(default=None, description="ISO 3166-1 alpha-2, e.g. 'YE'")
    port_code: str | None = Field(default=None, description="UN/LOCODE if a port, e.g. 'INNSA'")
    severity: int = Field(ge=1, le=5, description="1 minor/local .. 5 global, weeks-long")
    start_date: date | None = None
    end_date: date | None = None
    industries: list[str] = Field(default_factory=list)
    summary: str = Field(default="", description="One sentence in the model's own words")
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("event_type", mode="before")
    @classmethod
    def _normalize_type(cls, value: object) -> object:
        if isinstance(value, str):
            key = value.strip().lower().replace(" ", "_").replace("-", "_")
            if key in EventType._value2member_map_:
                return key
            return _TYPE_ALIASES.get(key, value)
        return value

    @field_validator("country_code", "port_code", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().upper()
            if value in {"", "NULL", "NONE", "N/A", "UNKNOWN"}:
                return None
        return value

    @field_validator("country_code")
    @classmethod
    def _check_country(cls, value: str | None) -> str | None:
        if value is not None and (len(value) != 2 or not value.isalpha()):
            raise ValueError("country_code must be a 2-letter ISO code or null")
        return value

    @field_validator("port_code")
    @classmethod
    def _check_port(cls, value: str | None) -> str | None:
        value = value.replace(" ", "") if value else value
        if value is not None and (len(value) != 5 or not value.isalnum()):
            raise ValueError("port_code must be a 5-character UN/LOCODE (e.g. INNSA) or null")
        return value

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _blank_date(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "unknown"}:
            return None
        return value

    @field_validator("industries", mode="before")
    @classmethod
    def _lower_industries(cls, value: object) -> object:
        if isinstance(value, list):
            return sorted({str(v).strip().lower() for v in value if str(v).strip()})
        return value


class ExtractionOutput(BaseModel):
    """Top-level LLM reply: an article may describe zero, one, or several disruptions."""

    is_disruption: bool
    events: list[ExtractedEvent] = Field(default_factory=list)


class DisruptionEvent(ExtractedEvent):
    """An extracted event with provenance, as stored and used downstream."""

    event_id: str
    news_id: str
    source_url: str
    source_name: str
    published: datetime
    model: str
    prompt_version: str


class ExtractionRecord(BaseModel):
    """Outcome of running extraction on one news item (kept even when it fails)."""

    news_id: str
    model: str
    prompt_version: str
    status: Literal["ok", "failed"]
    is_disruption: bool = False
    events: list[DisruptionEvent] = Field(default_factory=list)
    error: str | None = None
