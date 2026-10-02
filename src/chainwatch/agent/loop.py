"""Plain-Python tool-calling loop.

Each turn: render the prompt (task + tool list + transcript so far) -> the LLM returns one JSON action
-> either run a tool and append its result, or validate the final brief. The loop is stateless between
turns (the transcript is in the prompt), so every turn is cacheable and easy to inspect.

If the model never produces a valid brief within `max_steps`, we fall back to the deterministic brief and
say so in `generated_by`. Every brief, LLM or not, then passes through `ground_brief` guard rails.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from chainwatch.agent.brief import (
    AffectedLane,
    Brief,
    Mitigation,
    build_brief_deterministic,
    ground_brief,
    risk_level,
)
from chainwatch.agent.prompts import PROMPT_DIR as AGENT_PROMPT_DIR
from chainwatch.agent.tools import TOOLS, ToolContext, call_tool
from chainwatch.extraction.prompts import load_prompt
from chainwatch.llm import LLMClient, LLMError, extract_json

logger = logging.getLogger(__name__)

PROMPT_DIR_NAME = "agent_v1"
OBSERVATION_MAX_CHARS = 1500  # keep the transcript short for small local models
FINAL_GRACE_TURNS = 3


class LLMBriefDraft(BaseModel):
    """What the LLM writes; the rest of `Brief` is filled in by code."""

    summary: str
    affected_lanes: list[AffectedLane] = Field(default_factory=list)
    drivers: list[str] = Field(default_factory=list)
    mitigations: list[Mitigation] = Field(default_factory=list)
    cited_event_ids: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    action: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    error: str | None = None


class AgentResult(BaseModel):
    brief: Brief
    steps: list[AgentStep]
    used_fallback: bool
    raw_brief: Brief | None = None  # the LLM's own draft before guard rails (for the rubric)


def _tool_list() -> str:
    lines = []
    for tool in TOOLS.values():
        props = tool.args_model.model_json_schema().get("properties", {})
        args = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {tool.name}({args}): {tool.description}")
    return "\n".join(lines)


def _transcript(steps: list[AgentStep]) -> str:
    if not steps:
        return "(nothing yet)"
    parts = []
    for i, s in enumerate(steps, 1):
        if s.action == "call_tool":
            obs = json.dumps(s.observation, ensure_ascii=False)[:OBSERVATION_MAX_CHARS]
            parts.append(f"{i}. called {s.tool}({json.dumps(s.args)}) -> {obs}")
        else:
            parts.append(f"{i}. invalid reply: {s.error}")
    return "\n".join(parts)


def run_agent(
    llm: LLMClient,
    ctx: ToolContext,
    focus: str | None = None,
    max_steps: int = 8,
    prompt_name: str = PROMPT_DIR_NAME,
) -> AgentResult:
    template = load_prompt(prompt_name, directory=AGENT_PROMPT_DIR)
    task = (f"Assess disruption risk for the '{focus}' trade lanes and propose mitigations."
            if focus else "Assess disruption risk for all trade lanes and propose mitigations.")  # fmt: skip
    steps: list[AgentStep] = []
    # max_steps tool calls, plus a few turns to write the final brief (or recover from bad JSON).
    for _ in range(max_steps + FINAL_GRACE_TURNS):
        system, user = template.render(
            max_steps=str(max_steps), tools=_tool_list(), task=task,
            as_of=ctx.as_of.date().isoformat(), transcript=_transcript(steps),
        )  # fmt: skip
        try:
            reply = extract_json(llm.generate(user, system=system, json_mode=True))
        except (ValueError, LLMError) as exc:
            steps.append(AgentStep(action="invalid", error=str(exc)[:300]))
            continue
        if not isinstance(reply, dict):
            steps.append(AgentStep(action="invalid", error="reply was not a JSON object"))
            continue
        action = reply.get("action")
        if action in TOOLS:
            # Small models often write {"action": "lane_risk", "args": {...}}; accept that shape too.
            args = reply.get("args")
            if not isinstance(args, dict):
                args = {k: v for k, v in reply.items() if k not in {"action", "tool"}}
            reply = {"action": "call_tool", "tool": action, "args": args}
            action = "call_tool"
        if action == "call_tool" and len([s for s in steps if s.action == "call_tool"]) < max_steps:
            tool, args = reply.get("tool"), reply.get("args") or {}
            obs = call_tool(ctx, str(tool), args if isinstance(args, dict) else {})
            steps.append(AgentStep(action="call_tool", tool=str(tool), args=args, observation=obs))
            continue
        if action == "final":
            try:
                draft = LLMBriefDraft.model_validate(reply.get("brief") or {})
            except ValidationError as exc:
                steps.append(
                    AgentStep(action="invalid", error=f"brief invalid: {exc.errors()[:2]}")
                )
                continue
            brief = Brief(as_of=ctx.as_of.date().isoformat(), focus=focus, risk_score=0.0,
                          risk_level=risk_level(0.0), generated_by=f"llm:{llm.model}",
                          **draft.model_dump())  # fmt: skip
            return AgentResult(brief=ground_brief(brief, ctx), steps=steps, used_fallback=False,
                               raw_brief=brief)  # fmt: skip
        hint = (
            'use {"action": "call_tool", "tool": "<name>", "args": {...}} or '
            '{"action": "final", "brief": {...}}'
        )
        if action == "call_tool":
            hint = 'tool budget used up: reply with {"action": "final", "brief": {...}} now'
        steps.append(AgentStep(action="invalid", error=f"unknown action {action!r}; {hint}"))

    logger.warning("Agent did not finish within %d steps; using deterministic brief", max_steps)
    brief = build_brief_deterministic(ctx, focus)
    brief = brief.model_copy(update={"generated_by": f"fallback after llm:{llm.model}"})
    return AgentResult(brief=ground_brief(brief, ctx), steps=steps, used_fallback=True)
