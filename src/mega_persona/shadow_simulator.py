"""LLM-based behavior simulation for MegaPersona shadow surveys.

The LLMShadowSimulator role-plays each persona and answers survey items as
that person would. This produces truly independent behavior data — the persona
declares traits in one channel, and the LLM-simulated survey responses reveal
behavior in another channel. The gap between declared axes and behavioral axes
is the key experimental signal.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import re
import time
from typing import Any

import numpy as np

from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_survey import (
    ShadowSurvey,
    score_shadow_survey,
)
from src.mega_persona.slots import AXIS_NAMES


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowSurveySimulation:
    persona_id: str
    survey_id: str
    responses: dict[str, int]
    axis_scores: dict[str, float]
    construct_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ShadowBehaviorReport:
    sample_size: int
    survey_count: int
    axis_names: tuple[str, ...]
    behavior_axis_mean: dict[str, float]
    persona_behavior_mae: dict[str, float]
    overall_alignment: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "survey_count": self.survey_count,
            "axis_names": list(self.axis_names),
            "behavior_axis_mean": self.behavior_axis_mean,
            "persona_behavior_mae": self.persona_behavior_mae,
            "overall_alignment": self.overall_alignment,
        }


# ======================================================================
# Aggregation helpers (shared)
# ======================================================================


def aggregate_shadow_behavior(
    personas: list[MegaPersona],
    simulations: list[ShadowSurveySimulation],
    axis_names: tuple[str, ...] = AXIS_NAMES,
) -> ShadowBehaviorReport:
    if not personas:
        return ShadowBehaviorReport(
            sample_size=0,
            survey_count=0,
            axis_names=axis_names,
            behavior_axis_mean={axis: 0.0 for axis in axis_names},
            persona_behavior_mae={axis: 0.0 for axis in axis_names},
            overall_alignment=0.0,
        )

    grouped: dict[str, list[ShadowSurveySimulation]] = {}
    for simulation in simulations:
        grouped.setdefault(simulation.persona_id, []).append(simulation)

    behavior_by_persona: dict[str, dict[str, float]] = {}
    for persona in personas:
        persona_sims = grouped.get(persona.persona_id, [])
        if not persona_sims:
            behavior_by_persona[persona.persona_id] = {axis: 0.5 for axis in axis_names}
            continue
        behavior_by_persona[persona.persona_id] = {
            axis: float(np.mean([sim.axis_scores[axis] for sim in persona_sims]))
            for axis in axis_names
        }

    behavior_axis_mean = {
        axis: float(
            np.mean([behavior_by_persona[persona.persona_id][axis] for persona in personas])
        )
        for axis in axis_names
    }
    persona_behavior_mae = {
        axis: float(
            np.mean(
                [
                    abs(persona.primary_axes()[axis] - behavior_by_persona[persona.persona_id][axis])
                    for persona in personas
                ]
            )
        )
        for axis in axis_names
    }
    overall_mae = float(np.mean(list(persona_behavior_mae.values())))
    return ShadowBehaviorReport(
        sample_size=len(personas),
        survey_count=len({simulation.survey_id for simulation in simulations}),
        axis_names=axis_names,
        behavior_axis_mean=behavior_axis_mean,
        persona_behavior_mae=persona_behavior_mae,
        overall_alignment=float(np.clip(1.0 - overall_mae, 0.0, 1.0)),
    )


def shadow_behavior_axis_matrix(
    personas: list[MegaPersona],
    simulations: list[ShadowSurveySimulation],
    axis_names: tuple[str, ...] = AXIS_NAMES,
) -> np.ndarray:
    grouped: dict[str, list[ShadowSurveySimulation]] = {}
    for simulation in simulations:
        grouped.setdefault(simulation.persona_id, []).append(simulation)

    rows = []
    for persona in personas:
        persona_sims = grouped.get(persona.persona_id, [])
        if not persona_sims:
            rows.append([0.5 for _ in axis_names])
            continue
        rows.append(
            [
                float(np.mean([simulation.axis_scores[axis] for simulation in persona_sims]))
                for axis in axis_names
            ]
        )
    if not rows:
        return np.empty((0, len(axis_names)))
    return np.array(rows, dtype=float)


# ======================================================================
# LLM Shadow Simulator
# ======================================================================

_LLM_SIMULATOR_SYSTEM_PROMPT = """\
You are a behavior simulator for psychological research. You will receive a
detailed persona description and a set of survey items.

Your job is to role-play this person and answer EVERY item as that person
would genuinely answer — not as an idealised version, not as a caricature.

Guidelines:
- Use the persona's thinking style, motivation, values, and self-regulation
  patterns to decide how they would respond to each item.
- Do not infer hidden numeric trait scores. Use only the narrative and
  categorical evidence shown in the prompt.
- The persona may behave differently from what their declared self-image
  suggests — people are not perfectly self-aware.
- Answer every item. If an item does not apply perfectly, give the closest
  honest answer.
- Return ONLY valid JSON: a JSON object mapping item_id (string) to an
  integer 1-5, where 1=strongly disagree, 2=disagree, 3=neutral, 4=agree,
  5=strongly agree.
- Do not include explanations, markdown fences, or extra text."""


def _llm_simulator_user_prompt(persona: MegaPersona, survey: ShadowSurvey) -> str:
    """Build a single prompt containing the persona narrative and all items."""
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    regulation = profile.self_regulation
    thinking = profile.thinking_style
    learning = profile.learning_orientation
    decision = profile.decision_pattern
    challenge = profile.challenge_response
    social = persona.social_creative_profile
    health = persona.mental_health_context
    values = persona.values_identity

    persona_text = f"""\
PERSONA ID: {persona.persona_id}

DEMOGRAPHICS:
- Age: {persona.demographics.age}, Stage: {persona.demographics.grade_or_stage}
- Region: {persona.demographics.region_type}
- Family context: {persona.demographics.family_context}

HOW THIS PERSON THINKS (cognition):
- Dominant thinking mode: {thinking.dominant_mode}
- Evidence preference: {thinking.evidence_preference}
- Typical blind spot: {thinking.typical_blind_spot}

MOTIVATION SYSTEM:
- Primary drive: {motivation.primary_drive}
- Secondary drive: {motivation.secondary_drive}
- Reward preference: {motivation.reward_preference}

LEARNING & ATTENTION:
- Goal orientation: {learning.goal_orientation}
- Learning mode: {learning.preferred_learning_mode}
- Attention pattern: {learning.attention_pattern}
- Curiosity scope: {learning.curiosity_scope}
- Help-seeking style: {learning.help_seeking_style}

SELF-REGULATION:
- Planning style: {regulation.planning_style}

UNDER PRESSURE:
- When facing difficulty: {challenge.under_difficulty}
- When receiving criticism: {challenge.under_criticism}
- After success: {challenge.under_success}

DECISION MAKING:
- Time horizon: {decision.time_horizon}
- Tradeoff style: {decision.tradeoff_style}
- Typical rationale: {decision.typical_rationale}

COGNITIVE-MOTIVATION NARRATIVE:
{profile.narrative}

VALUES & IDENTITY:
- Core values: {', '.join(values.core_values)}
- Identity anchor: {values.identity_anchor}
- Moral tension: {values.moral_tension}
- Aspiration: {values.aspiration}

SOCIAL & CREATIVE PROFILE:
- Collaboration style: {social.collaboration_style}
- Creative mode: {social.creative_mode}
- Social narrative: {social.narrative}

MENTAL HEALTH CONTEXT:
- Coping style: {health.coping_style}
- Protective factors: {', '.join(health.protective_factors)}
- Risk factors: {', '.join(health.risk_factors)}
- Health narrative: {health.narrative}"""

    items_text_parts: list[str] = []
    for item in survey.items:
        items_text_parts.append(f'  "{item.item_id}": "{item.text}"')
    items_text = "\n".join(items_text_parts)

    return f"""\
{persona_text}

---
SURVEY CONTEXT: {survey.context}
---

ANSWER EVERY ITEM BELOW as this person would. Return a JSON object mapping
each item_id to an integer 1-5 (1=strongly disagree → 5=strongly agree).

ITEMS:
{items_text}

Return ONLY the JSON object, no other text."""


class LLMShadowSimulator:
    """LLM-based behavior simulator — the only simulator for real experiments.

    This simulator asks an LLM to genuinely role-play the persona and answer
    survey items independently. The gap between declared persona axes and
    simulated behavior axes is the key signal — it measures how well the
    persona *description* predicts *behavior* in a separate channel.
    """

    def __init__(
        self,
        llm_client,
        temperature: float = 0.3,
        max_tokens: int = 1500,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        max_workers: int = 1,
    ):
        if llm_client is None:
            raise ValueError("llm_client is required for LLMShadowSimulator")
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_workers = max(1, int(max_workers))

    def simulate_population(
        self,
        personas: list[MegaPersona],
        surveys: list[ShadowSurvey],
    ) -> list[ShadowSurveySimulation]:
        total_calls = len(personas) * len(surveys)
        logger.info(
            "Shadow simulation start personas=%s surveys=%s calls=%s max_workers=%s",
            len(personas),
            len(surveys),
            total_calls,
            self.max_workers,
        )
        if self.max_workers <= 1 or total_calls <= 1:
            return self._simulate_population_sequential(personas, surveys, total_calls)

        tasks = []
        call_index = 0
        for persona_index, persona in enumerate(personas, start=1):
            for survey_index, survey in enumerate(surveys, start=1):
                call_index += 1
                tasks.append((call_index, persona_index, survey_index, persona, survey))

        ordered: list[ShadowSurveySimulation | None] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, total_calls)) as executor:
            futures = {}
            for call_index, persona_index, survey_index, persona, survey in tasks:
                logger.info(
                    "Shadow simulation call %s/%s queued persona=%s survey=%s (%s/%s)",
                    call_index,
                    total_calls,
                    persona.persona_id,
                    survey.survey_id,
                    survey_index,
                    len(surveys),
                )
                futures[executor.submit(self.simulate_persona, persona, survey)] = call_index
            for future in as_completed(futures):
                call_index = futures[future]
                ordered[call_index - 1] = future.result()
                logger.info("Shadow simulation call %s/%s done", call_index, total_calls)

        simulations = [simulation for simulation in ordered if simulation is not None]
        logger.info("Shadow simulation done calls=%s", len(simulations))
        return simulations

    def _simulate_population_sequential(
        self,
        personas: list[MegaPersona],
        surveys: list[ShadowSurvey],
        total_calls: int,
    ) -> list[ShadowSurveySimulation]:
        simulations: list[ShadowSurveySimulation] = []
        call_index = 0
        for persona_index, persona in enumerate(personas, start=1):
            logger.info(
                "Shadow simulation persona %s/%s id=%s",
                persona_index,
                len(personas),
                persona.persona_id,
            )
            for survey_index, survey in enumerate(surveys, start=1):
                call_index += 1
                logger.info(
                    "Shadow simulation call %s/%s persona=%s survey=%s (%s/%s)",
                    call_index,
                    total_calls,
                    persona.persona_id,
                    survey.survey_id,
                    survey_index,
                    len(surveys),
                )
                simulations.append(self.simulate_persona(persona, survey))
        logger.info("Shadow simulation done calls=%s", len(simulations))
        return simulations

    def simulate_persona(
        self,
        persona: MegaPersona,
        survey: ShadowSurvey,
    ) -> ShadowSurveySimulation:
        """Generate LLM-simulated responses for one persona on one survey.

        All items in the survey are sent in a single LLM call to save cost.
        """
        user_prompt = _llm_simulator_user_prompt(persona, survey)
        raw = _generate_with_retry(
            llm=self.llm,
            prompt=user_prompt,
            system_prompt=_LLM_SIMULATOR_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            label=f"persona={persona.persona_id} survey={survey.survey_id}",
        )
        responses = _parse_simulator_response(raw, survey)
        scores = score_shadow_survey(survey, responses)
        axis_scores = {
            axis: scores[f"axis.{axis}"]
            for axis in AXIS_NAMES
        }
        construct_scores = {
            key.removeprefix("construct."): value
            for key, value in scores.items()
            if key.startswith("construct.")
        }
        return ShadowSurveySimulation(
            persona_id=persona.persona_id,
            survey_id=survey.survey_id,
            responses=responses,
            axis_scores=axis_scores,
            construct_scores=construct_scores,
        )


def _generate_with_retry(
    *,
    llm,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
    label: str,
) -> str:
    """Retry transient LLM failures so one timeout does not zero a whole seed."""
    transient_markers = (
        "timeout",
        "timed out",
        "connection",
        "rate limit",
        "temporarily",
        "server",
        "overloaded",
    )
    for attempt in range(max_retries + 1):
        try:
            return llm.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            message = str(exc).lower()
            transient = any(marker in message for marker in transient_markers)
            if attempt >= max_retries or not transient:
                raise
            wait_seconds = retry_backoff_seconds * (2 ** attempt)
            logger.warning(
                "Shadow simulator transient failure; retrying attempt=%s/%s wait=%.1fs %s (%s: %s)",
                attempt + 1,
                max_retries,
                wait_seconds,
                label,
                type(exc).__name__,
                exc,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("unreachable retry state")


def _parse_simulator_response(
    raw: str,
    survey: ShadowSurvey,
) -> dict[str, int]:
    """Parse the LLM's JSON response, with fallback for missing/malformed items."""
    import json

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                char = text[idx]
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start:idx + 1])
                        except json.JSONDecodeError:
                            parsed = None
                        break
        if parsed is None:
            logger.warning(
                "Malformed simulator JSON for survey=%s; using regex/neutral fallback (%s: %s)",
                survey.survey_id,
                type(exc).__name__,
                exc,
            )
            return _fallback_simulator_responses(text, survey)

    if not isinstance(parsed, dict):
        logger.warning(
            "Simulator response for survey=%s was %s, expected object; using neutral fallback",
            survey.survey_id,
            type(parsed).__name__,
        )
        return {item.item_id: 3 for item in survey.items}

    responses: dict[str, int] = {}
    for item in survey.items:
        raw_value = parsed.get(item.item_id, 3)
        try:
            value = int(raw_value)
        except (ValueError, TypeError):
            value = 3
        responses[item.item_id] = max(1, min(5, value))
    return responses


def _fallback_simulator_responses(text: str, survey: ShadowSurvey) -> dict[str, int]:
    responses: dict[str, int] = {}
    extracted = 0
    for item in survey.items:
        pattern = rf'["\']{re.escape(item.item_id)}["\']\s*[:=]?\s*["\']?([1-5])'
        match = re.search(pattern, text)
        if match:
            responses[item.item_id] = int(match.group(1))
            extracted += 1
        else:
            responses[item.item_id] = 3
    logger.warning(
        "Simulator fallback recovered %s/%s responses for survey=%s",
        extracted,
        len(survey.items),
        survey.survey_id,
    )
    return responses
