"""LLM-based behavior simulation for MegaPersona shadow surveys.

The LLMShadowSimulator role-plays each persona and answers survey items as
that person would. This produces truly independent behavior data — the persona
declares traits in one channel, and the LLM-simulated survey responses reveal
behavior in another channel. The gap between declared axes and behavioral axes
is the key experimental signal.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import importlib.util
import json
import logging
import re
import time
from typing import Any

import numpy as np

from src.mega_persona.concordia_adapter import (
    build_concordia_agent_bundle,
    survey_action_spec,
    survey_observation,
)
from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_survey import (
    ShadowSurvey,
    score_shadow_survey,
)
from src.mega_persona.slots import AXIS_NAMES, axis_roles_for_target_axes


logger = logging.getLogger(__name__)


SUPPORTED_SHADOW_SIMULATOR_BACKENDS = (
    "llm",
    "concordia",
    "concordia-native",
    "student-realistic",
    "student-realistic-v2",
)


@dataclass(frozen=True)
class ShadowSurveySimulation:
    persona_id: str
    survey_id: str
    responses: dict[str, int]
    axis_scores: dict[str, float]
    construct_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShadowBehaviorReport:
    sample_size: int
    survey_count: int
    axis_names: tuple[str, ...]
    behavior_axis_mean: dict[str, float]
    persona_behavior_mae: dict[str, float]
    overall_alignment: float
    overall_mae: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "survey_count": self.survey_count,
            "axis_names": list(self.axis_names),
            "behavior_axis_mean": self.behavior_axis_mean,
            "persona_behavior_mae": self.persona_behavior_mae,
            "overall_alignment": self.overall_alignment,
            "overall_mae": self.overall_mae,
        }


# ======================================================================
# Aggregation helpers (shared)
# ======================================================================


def aggregate_shadow_behavior(
    personas: list[MegaPersona],
    simulations: list[ShadowSurveySimulation],
    axis_names: tuple[str, ...] = AXIS_NAMES,
    axis_roles: dict[str, str] | None = None,
) -> ShadowBehaviorReport:
    if not personas:
        return ShadowBehaviorReport(
            sample_size=0,
            survey_count=0,
            axis_names=axis_names,
            behavior_axis_mean={axis: 0.0 for axis in axis_names},
            persona_behavior_mae={axis: 0.0 for axis in axis_names},
            overall_alignment=0.0,
            overall_mae=1.0,
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
                    abs(
                        persona.primary_axes(axis_names=axis_names, axis_roles=axis_roles)[axis]
                        - behavior_by_persona[persona.persona_id][axis]
                    )
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
        overall_mae=overall_mae,
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
- For motivation-related items, distinguish intrinsic/autonomous drive from
  externally pressured effort. A person can work hard for approval, grades,
  security, or fear of failure without being highly autonomous.
- Growth, curiosity, and creative-confidence items should be high only when
  the persona shows both internal interest and low dependence on external
  pressure; otherwise use mixed or lower responses.
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
- Intrinsic motivation: {motivation.intrinsic_motivation:.2f}
- External pressure sensitivity: {motivation.external_pressure_sensitivity:.2f}
- Failure sensitivity: {motivation.failure_sensitivity:.2f}
- Reward preference: {motivation.reward_preference}

LEARNING & ATTENTION:
- Goal orientation: {learning.goal_orientation}
- Learning mode: {learning.preferred_learning_mode}
- Attention pattern: {learning.attention_pattern}
- Curiosity scope: {learning.curiosity_scope}
- Help-seeking style: {learning.help_seeking_style}

SELF-REGULATION:
- Planning style: {regulation.planning_style}
- Persistence: {regulation.persistence:.2f}
- Emotional regulation: {regulation.emotional_regulation:.2f}
- Metacognition: {regulation.metacognition:.2f}
- Habit stability: {regulation.habit_stability:.2f}

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

MOTIVATION-AUTONOMY CALIBRATION:
- High motivation-autonomy responses require internal interest, ownership,
  and willingness to continue without praise, grades, or peer approval.
- If the persona is driven by recognition, security, belonging, avoidance, or
  fear of failure, do not automatically score growth/effort items high.
- If external pressure sensitivity or failure sensitivity is high, prefer 2/3
  on autonomy items unless the narrative shows clear self-directed action.
- If intrinsic motivation is high but regulation is weak, answer high for
  wanting to grow but lower for sustained follow-through.

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
        temperature: float = 0.05,
        top_p: float = 0.80,
        max_tokens: int = 1500,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        max_workers: int = 1,
    ):
        if llm_client is None:
            raise ValueError("llm_client is required for LLMShadowSimulator")
        self.llm = llm_client
        self.temperature = temperature
        self.top_p = top_p
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
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            label=f"persona={persona.persona_id} survey={survey.survey_id}",
        )
        responses = _parse_simulator_response(raw, survey)
        scores = score_shadow_survey(survey, responses)
        axis_scores = {
            axis: scores[f"axis.{axis}"]
            for axis in survey.axis_names
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


_CONCORDIA_SIMULATOR_SYSTEM_PROMPT = """\
You are running a compact Concordia-style generative social simulation for
psychological research. Treat the persona as an agent with independent
components: identity, memory, goals, values, social context, coping state,
and moment-by-moment situation appraisal.

For each survey item, silently perform this sequence:
1. Let the agent observe the item in the given survey context.
2. Retrieve relevant memories and stable dispositions from the persona.
3. Let goals, values, self-regulation, social incentives, and mental-health
   context jointly influence the response.
4. Answer as the agent would actually answer, allowing imperfect self-awareness
   and context-dependent behavior.

Return ONLY valid JSON: a JSON object mapping item_id (string) to an integer
1-5, where 1=strongly disagree, 2=disagree, 3=neutral, 4=agree, and 5=strongly
agree. Do not include chain-of-thought, explanations, markdown fences, or extra
text."""


def _concordia_simulator_user_prompt(persona: MegaPersona, survey: ShadowSurvey) -> str:
    base_prompt = _llm_simulator_user_prompt(persona, survey)
    return f"""\
CONCORDIA-STYLE SIMULATION SETUP:
- Agent: {persona.persona_id}
- Game master task: collect honest survey behavior from this agent.
- Active components: observations, identity, memory, goals, values, social
  context, coping state, and decision appraisal.
- Important: use component interaction, not direct copying of declared axis
  scores.

{base_prompt}"""


class ConcordiaShadowSimulator(LLMShadowSimulator):
    """Concordia-style shadow survey simulator.

    This adapter is intentionally kept behind the same interface as
    LLMShadowSimulator. If google-deepmind/concordia is installed, the run is
    tagged as Concordia-available for traceability; the prompt still preserves
    the project contract of one persona-survey call producing Likert JSON.
    """

    def __init__(
        self,
        llm_client,
        temperature: float = 0.05,
        top_p: float = 0.80,
        max_tokens: int = 1500,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        max_workers: int = 1,
    ):
        super().__init__(
            llm_client=llm_client,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_workers=max_workers,
        )
        self.concordia_available = importlib.util.find_spec("concordia") is not None
        if self.concordia_available:
            logger.info("Concordia package detected; using Concordia-style shadow simulator")
        else:
            logger.info(
                "Concordia package not installed; using Concordia-style prompt adapter"
            )

    def simulate_persona(
        self,
        persona: MegaPersona,
        survey: ShadowSurvey,
    ) -> ShadowSurveySimulation:
        user_prompt = _concordia_simulator_user_prompt(persona, survey)
        raw = _generate_with_retry(
            llm=self.llm,
            prompt=user_prompt,
            system_prompt=_CONCORDIA_SIMULATOR_SYSTEM_PROMPT,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            label=f"concordia persona={persona.persona_id} survey={survey.survey_id}",
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


_STUDENT_REALISTIC_SYSTEM_PROMPT = """\
You are a realistic student behavior simulator for psychological research.
You do not simply role-play a persona label. You simulate how a real student
would answer a survey after appraising the situation through motivation,
stress, competence, autonomy, social safety, fatigue, and avoidance.

Principles:
- A student can value growth but still avoid effort when tired, threatened, or
  afraid of failure.
- A student can work hard because of grades, praise, family pressure, or fear
  without having high autonomous motivation.
- A student can show different behavior across items when the item activates
  different psychological mechanisms.
- Use the provided latent student state and context appraisal as mechanisms,
  not as final answers.
- Use the full 1-5 scale. Do not default to 3 unless evidence is genuinely
  mixed or context-dependent.

Return ONLY valid JSON mapping every item_id to an integer 1-5:
1=strongly disagree, 2=disagree, 3=neutral, 4=agree, 5=strongly agree.
Do not include explanations, markdown fences, or extra text."""


_STUDENT_REALISTIC_V2_SYSTEM_PROMPT = """\
You are a realistic student/personality behavior simulator for psychological
research. Your job is to simulate how a real student would answer survey
items after stable traits interact with the immediate context and the
student's current latent state.

Use this mechanism silently:
1. Read stable traits and narrative evidence.
2. Appraise the current situation.
3. Convert traits + context into a latent student state.
4. Interpret what each item is psychologically asking.
5. Apply the response style: acquiescence, social desirability, uncertainty,
   extremity, and fatigue can shape the final Likert choice.

Important:
- Do not answer as an ideal student or as a polished self-description.
- Do not make every positive-sounding item high; distinguish autonomous
  interest from effort driven by fear, reward, grades, or approval.
- Use the full 1-5 scale when evidence is clear.
- Choose 3 only when evidence is genuinely mixed or the item is context-bound.

Return ONLY valid JSON mapping every item_id to an integer 1-5:
1=strongly disagree, 2=disagree, 3=neutral, 4=agree, 5=strongly agree.
Do not include explanations, markdown fences, or extra text."""


def _student_realistic_user_prompt(
    persona: MegaPersona,
    survey: ShadowSurvey,
    student_state: dict[str, float],
    context_appraisal: dict[str, float],
) -> str:
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    regulation = profile.self_regulation
    learning = profile.learning_orientation
    thinking = profile.thinking_style
    challenge = profile.challenge_response
    social = persona.social_creative_profile
    health = persona.mental_health_context
    values = persona.values_identity
    axis_names = survey.axis_names or AXIS_NAMES
    axis_roles = axis_roles_for_target_axes({axis: 0.5 for axis in axis_names})
    axes = persona.primary_axes(axis_names=axis_names, axis_roles=axis_roles)
    items_text = "\n".join(f'  "{item.item_id}": "{item.text}"' for item in survey.items)
    mechanism_text = "\n".join(_item_mechanism_hint(item) for item in survey.items)

    return f"""\
REALISTIC STUDENT SIMULATION

PERSONA ID: {persona.persona_id}
Stage: {persona.demographics.age}, {persona.demographics.grade_or_stage}
Family context: {persona.demographics.family_context}

TRAIT SUMMARY:
- primary axes: {json.dumps(_round_dict(axes), sort_keys=True)}
- thinking mode: {thinking.dominant_mode}
- abstraction level: {thinking.abstraction_level:.2f}
- ambiguity tolerance: {thinking.ambiguity_tolerance:.2f}
- evidence preference: {thinking.evidence_preference}
- typical blind spot: {thinking.typical_blind_spot}
- primary drive: {motivation.primary_drive}
- secondary drive: {motivation.secondary_drive}
- intrinsic motivation: {motivation.intrinsic_motivation:.2f}
- external pressure sensitivity: {motivation.external_pressure_sensitivity:.2f}
- failure sensitivity: {motivation.failure_sensitivity:.2f}
- reward preference: {motivation.reward_preference}
- goal orientation: {learning.goal_orientation}
- curiosity scope: {learning.curiosity_scope}
- help seeking: {learning.help_seeking_style}
- planning style: {regulation.planning_style}
- persistence: {regulation.persistence:.2f}
- emotional regulation: {regulation.emotional_regulation:.2f}
- metacognition: {regulation.metacognition:.2f}
- habit stability: {regulation.habit_stability:.2f}
- peer influence sensitivity: {social.peer_influence_sensitivity:.2f}
- stress load: {health.stress_load:.2f}
- resilience: {health.resilience:.2f}
- coping style: {health.coping_style}
- risk factors: {', '.join(health.risk_factors)}
- protective factors: {', '.join(health.protective_factors)}

NARRATIVE EVIDENCE:
- cognitive-motivation: {profile.narrative}
- identity anchor: {values.identity_anchor}
- moral tension: {values.moral_tension}
- aspiration: {values.aspiration}
- social/creative: {social.narrative}
- mental health: {health.narrative}
- under difficulty: {challenge.under_difficulty}
- under criticism: {challenge.under_criticism}
- after success: {challenge.under_success}

CURRENT SURVEY CONTEXT:
{survey.context}

LATENT STUDENT STATE:
{json.dumps(_round_dict(student_state), ensure_ascii=False, sort_keys=True)}

CONTEXT APPRAISAL:
{json.dumps(_round_dict(context_appraisal), ensure_ascii=False, sort_keys=True)}

ITEM MECHANISM HINTS:
{mechanism_text}

RESPONSE POLICY:
- For autonomy/motivation items, high scores require self-endorsed interest or
  ownership. Effort caused mainly by pressure, reward, or fear should be mixed
  or lower.
- For self-regulation items, combine stable regulation with current stress,
  fatigue, avoidance, and recovery.
- For cognitive/curiosity items, combine abstraction, ambiguity tolerance,
  task interest, and threat appraisal.
- For social items, combine social safety, peer pressure, belonging needs, and
  peer influence sensitivity.
- Answer as behavior in this context, not as a polished self-description.

ITEMS:
{items_text}

Return only a JSON object mapping every item_id to an integer 1-5."""


def _student_realistic_v2_user_prompt(
    persona: MegaPersona,
    survey: ShadowSurvey,
    trait_vector: dict[str, float],
    student_state: dict[str, float],
    context_appraisal: dict[str, float],
    response_style: dict[str, float],
    item_mechanisms: dict[str, str],
) -> str:
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    regulation = profile.self_regulation
    learning = profile.learning_orientation
    thinking = profile.thinking_style
    challenge = profile.challenge_response
    decision = profile.decision_pattern
    social = persona.social_creative_profile
    health = persona.mental_health_context
    values = persona.values_identity
    items_text = "\n".join(f'  "{item.item_id}": "{item.text}"' for item in survey.items)
    mechanisms_text = "\n".join(
        f'- {item_id}: {mechanism}'
        for item_id, mechanism in sorted(item_mechanisms.items())
    )

    return f"""\
REALISTIC STUDENT SIMULATION V2

This is a blind behavioral simulation. Use the student's stable traits,
context appraisal, latent state, response style, and item meaning. Do not infer
or optimize for hidden scoring axes.

PERSONA ID: {persona.persona_id}
Stage: {persona.demographics.age}, {persona.demographics.grade_or_stage}
Region: {persona.demographics.region_type}
Family context: {persona.demographics.family_context}

STABLE TRAIT EVIDENCE:
- thinking mode: {thinking.dominant_mode}
- evidence preference: {thinking.evidence_preference}
- ambiguity tolerance: {thinking.ambiguity_tolerance:.2f}
- abstraction level: {thinking.abstraction_level:.2f}
- blind spot: {thinking.typical_blind_spot}
- primary drive: {motivation.primary_drive}
- secondary drive: {motivation.secondary_drive}
- intrinsic motivation: {motivation.intrinsic_motivation:.2f}
- external pressure sensitivity: {motivation.external_pressure_sensitivity:.2f}
- failure sensitivity: {motivation.failure_sensitivity:.2f}
- reward preference: {motivation.reward_preference}
- learning goal orientation: {learning.goal_orientation}
- learning mode: {learning.preferred_learning_mode}
- attention pattern: {learning.attention_pattern}
- curiosity scope: {learning.curiosity_scope}
- help seeking: {learning.help_seeking_style}
- planning style: {regulation.planning_style}
- persistence: {regulation.persistence:.2f}
- emotional regulation: {regulation.emotional_regulation:.2f}
- metacognition: {regulation.metacognition:.2f}
- habit stability: {regulation.habit_stability:.2f}
- decision time horizon: {decision.time_horizon}
- decision tradeoff style: {decision.tradeoff_style}
- collaboration style: {social.collaboration_style}
- social energy: {social.social_energy:.2f}
- peer influence sensitivity: {social.peer_influence_sensitivity:.2f}
- creative mode: {social.creative_mode}
- stress load: {health.stress_load:.2f}
- resilience: {health.resilience:.2f}
- coping style: {health.coping_style}
- risk factors: {', '.join(health.risk_factors)}
- protective factors: {', '.join(health.protective_factors)}

NARRATIVE EVIDENCE:
- cognitive-motivation narrative: {profile.narrative}
- identity anchor: {values.identity_anchor}
- core values: {', '.join(values.core_values)}
- moral tension: {values.moral_tension}
- aspiration: {values.aspiration}
- social/creative narrative: {social.narrative}
- mental health narrative: {health.narrative}
- under difficulty: {challenge.under_difficulty}
- under criticism: {challenge.under_criticism}
- after success: {challenge.under_success}

DERIVED TRAIT VECTOR:
{json.dumps(_round_dict(trait_vector), ensure_ascii=False, sort_keys=True)}

CURRENT SURVEY CONTEXT:
{survey.context}

CONTEXT APPRAISAL:
{json.dumps(_round_dict(context_appraisal), ensure_ascii=False, sort_keys=True)}

LATENT STUDENT STATE:
{json.dumps(_round_dict(student_state), ensure_ascii=False, sort_keys=True)}

RESPONSE STYLE:
{json.dumps(_round_dict(response_style), ensure_ascii=False, sort_keys=True)}

ITEM MECHANISM INTERPRETATION:
{mechanisms_text}

RESPONSE POLICY:
- For each item, combine stable traits, latent state, and the item mechanism.
- Let response style affect the exact score, but do not let it override strong
  evidence.
- Social desirability may slightly inflate admirable answers; fatigue may
  compress answers toward 3; high extremity may allow 1/5 when evidence is
  clear.
- External pressure can produce hard work without autonomous motivation.
- Stress and avoidance can reduce follow-through even when interest is real.
- Support and competence can unlock more honest growth-oriented responses.

ITEMS:
{items_text}

Return only a JSON object mapping every item_id to an integer 1-5."""


class StudentRealisticShadowSimulator(LLMShadowSimulator):
    """Mechanism-constrained realistic student behavior simulator.

    This backend computes a latent student state and context appraisal before
    asking the LLM for item-level Likert responses. The trace is stored in
    ``ShadowSurveySimulation.metadata`` for auditability.
    """

    def __init__(
        self,
        llm_client,
        temperature: float = 0.05,
        top_p: float = 0.80,
        max_tokens: int = 1800,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        max_workers: int = 1,
    ):
        super().__init__(
            llm_client=llm_client,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_workers=max_workers,
        )
        logger.info("Using realistic student behavior shadow simulator")

    def simulate_persona(
        self,
        persona: MegaPersona,
        survey: ShadowSurvey,
    ) -> ShadowSurveySimulation:
        context_appraisal = _context_appraisal(survey)
        student_state = _student_state(persona, context_appraisal)
        user_prompt = _student_realistic_user_prompt(
            persona,
            survey,
            student_state=student_state,
            context_appraisal=context_appraisal,
        )
        raw = _generate_with_retry(
            llm=self.llm,
            prompt=user_prompt,
            system_prompt=_STUDENT_REALISTIC_SYSTEM_PROMPT,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            label=f"student-realistic persona={persona.persona_id} survey={survey.survey_id}",
        )
        responses = _parse_simulator_response(raw, survey)
        scores = score_shadow_survey(survey, responses)
        axis_scores = {axis: scores[f"axis.{axis}"] for axis in survey.axis_names}
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
            metadata={
                "simulator_backend": "student-realistic",
                "student_state": _round_dict(student_state),
                "context_appraisal": _round_dict(context_appraisal),
                "mechanism": "trait_state_context_item_policy",
            },
        )


class StudentRealisticV2ShadowSimulator(LLMShadowSimulator):
    """Blind mechanism-based realistic student behavior simulator.

    Compared with ``StudentRealisticShadowSimulator``, v2 separates stable
    trait encoding, context appraisal, latent state, response style, and
    item interpretation. The prompt does not expose primary axes or item axis
    weights, making it more suitable as a clean baseline evaluator.
    """

    def __init__(
        self,
        llm_client,
        temperature: float = 0.05,
        top_p: float = 0.80,
        max_tokens: int = 2200,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        max_workers: int = 1,
    ):
        super().__init__(
            llm_client=llm_client,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_workers=max_workers,
        )
        logger.info("Using realistic student behavior shadow simulator v2")

    def simulate_persona(
        self,
        persona: MegaPersona,
        survey: ShadowSurvey,
    ) -> ShadowSurveySimulation:
        trait_vector = _student_trait_vector(persona)
        context_appraisal = _context_appraisal(survey)
        student_state = _student_state_v2(persona, trait_vector, context_appraisal)
        response_style = _student_response_style(persona, trait_vector, student_state)
        item_mechanisms = _blind_item_mechanisms(survey)
        user_prompt = _student_realistic_v2_user_prompt(
            persona,
            survey,
            trait_vector=trait_vector,
            student_state=student_state,
            context_appraisal=context_appraisal,
            response_style=response_style,
            item_mechanisms=item_mechanisms,
        )
        raw = _generate_with_retry(
            llm=self.llm,
            prompt=user_prompt,
            system_prompt=_STUDENT_REALISTIC_V2_SYSTEM_PROMPT,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            label=f"student-realistic-v2 persona={persona.persona_id} survey={survey.survey_id}",
        )
        responses = _parse_simulator_response(raw, survey)
        scores = score_shadow_survey(survey, responses)
        axis_scores = {axis: scores[f"axis.{axis}"] for axis in survey.axis_names}
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
            metadata={
                "simulator_backend": "student-realistic-v2",
                "trait_vector": _round_dict(trait_vector),
                "context_appraisal": _round_dict(context_appraisal),
                "student_state": _round_dict(student_state),
                "response_style": _round_dict(response_style),
                "item_mechanisms": item_mechanisms,
                "mechanism": "trait_context_state_response_style_item_policy_blind",
            },
        )


class ConcordiaNativeShadowSimulator(LLMShadowSimulator):
    """Native Concordia EntityAgent/component shadow survey simulator."""

    def __init__(
        self,
        llm_client,
        temperature: float = 0.05,
        top_p: float = 0.80,
        max_tokens: int = 1500,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        max_workers: int = 1,
    ):
        super().__init__(
            llm_client=llm_client,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_workers=max_workers,
        )
        logger.info("Using Concordia native EntityAgent shadow simulator")

    def simulate_persona(
        self,
        persona: MegaPersona,
        survey: ShadowSurvey,
    ) -> ShadowSurveySimulation:
        raw = _generate_concordia_native_with_retry(
            llm_client=self.llm,
            persona=persona,
            survey=survey,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        responses = _parse_simulator_response(raw, survey)
        scores = score_shadow_survey(survey, responses)
        axis_scores = {axis: scores[f"axis.{axis}"] for axis in survey.axis_names}
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


def build_shadow_simulator(
    *,
    backend: str,
    llm_client,
    max_workers: int = 1,
    temperature: float = 0.05,
    top_p: float = 0.80,
) -> LLMShadowSimulator:
    """Create a shadow-survey simulator backend."""
    if backend == "llm":
        return LLMShadowSimulator(
            llm_client, max_workers=max_workers, temperature=temperature, top_p=top_p
        )
    if backend == "concordia":
        return ConcordiaShadowSimulator(
            llm_client, max_workers=max_workers, temperature=temperature, top_p=top_p
        )
    if backend == "concordia-native":
        return ConcordiaNativeShadowSimulator(
            llm_client, max_workers=max_workers, temperature=temperature, top_p=top_p
        )
    if backend == "student-realistic":
        return StudentRealisticShadowSimulator(
            llm_client, max_workers=max_workers, temperature=temperature, top_p=top_p
        )
    if backend == "student-realistic-v2":
        return StudentRealisticV2ShadowSimulator(
            llm_client, max_workers=max_workers, temperature=temperature, top_p=top_p
        )
    raise ValueError(
        f"Unknown shadow simulator backend: {backend}. "
        f"Expected one of {', '.join(SUPPORTED_SHADOW_SIMULATOR_BACKENDS)}."
    )


def _context_appraisal(survey: ShadowSurvey) -> dict[str, float]:
    text = f"{survey.context} " + " ".join(item.text for item in survey.items)
    lower = text.lower()

    def has_any(words: tuple[str, ...]) -> float:
        return 1.0 if any(word in lower for word in words) else 0.0

    threat = max(
        has_any(("criticism", "deadline", "stress", "pressure", "failure", "conflict")),
        0.6 * has_any(("bullied", "exclusion", "outsider", "unsafe", "disrupt")),
    )
    opportunity = max(
        has_any(("opportunity", "interesting", "exploration", "curious", "creative")),
        0.7 * has_any(("project", "new", "complex", "challenge")),
    )
    external_evaluation = max(
        has_any(("criticism", "approval", "feedback", "checking", "rules")),
        0.7 * has_any(("deadline", "group", "teacher", "adult", "peer")),
    )
    peer_pressure = max(
        has_any(("peer", "group", "approval", "uncool", "embarrassing")),
        0.6 * has_any(("belong", "outsider", "social")),
    )
    ambiguity = max(
        has_any(("unclear", "no single correct", "complex", "unfamiliar")),
        0.7 * has_any(("new", "exploration", "method")),
    )
    support = max(
        has_any(("supportive", "understands", "help", "feedback is firm but fair")),
        0.6 * has_any(("adult", "peer relationships", "belonging")),
    )
    deadline = has_any(("deadline", "competing demands", "schedule", "time"))

    return {
        "threat": _clip01(0.25 + 0.60 * threat + 0.20 * external_evaluation),
        "opportunity": _clip01(0.20 + 0.70 * opportunity + 0.15 * ambiguity),
        "external_evaluation": _clip01(0.20 + 0.70 * external_evaluation),
        "peer_pressure": _clip01(0.15 + 0.75 * peer_pressure),
        "ambiguity": _clip01(0.20 + 0.70 * ambiguity),
        "support": _clip01(0.20 + 0.70 * support),
        "deadline_pressure": _clip01(0.15 + 0.80 * deadline),
    }


def _student_state(persona: MegaPersona, context: dict[str, float]) -> dict[str, float]:
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    regulation = profile.self_regulation
    thinking = profile.thinking_style
    social = persona.social_creative_profile
    health = persona.mental_health_context
    primary_drive_bonus = {
        "autonomy": 0.18,
        "curiosity": 0.15,
        "mastery": 0.12,
        "achievement": 0.04,
        "recognition": -0.06,
        "belonging": -0.05,
        "security": -0.08,
        "avoidance": -0.14,
    }.get(motivation.primary_drive, 0.0)
    intrinsic = motivation.intrinsic_motivation
    external = motivation.external_pressure_sensitivity
    failure = motivation.failure_sensitivity

    stress = _clip01(
        0.38 * health.stress_load
        + 0.24 * context["threat"]
        + 0.18 * failure
        + 0.12 * (1.0 - regulation.emotional_regulation)
        + 0.08 * context["deadline_pressure"]
    )
    fatigue = _clip01(
        0.40 * health.stress_load
        + 0.22 * (1.0 - regulation.habit_stability)
        + 0.22 * context["deadline_pressure"]
        + 0.16 * stress
    )
    perceived_autonomy = _clip01(
        0.42 * intrinsic
        + 0.28 * (1.0 - external)
        + 0.12 * context["opportunity"]
        + primary_drive_bonus
        - 0.15 * context["external_evaluation"]
        - 0.08 * context["peer_pressure"]
    )
    perceived_competence = _clip01(
        0.25 * regulation.persistence
        + 0.25 * regulation.metacognition
        + 0.18 * health.resilience
        + 0.18 * thinking.abstraction_level
        + 0.10 * regulation.habit_stability
        - 0.16 * failure
        - 0.08 * stress
    )
    social_safety = _clip01(
        0.30 * (1.0 - social.peer_influence_sensitivity)
        + 0.20 * context["support"]
        + 0.18 * (1.0 - health.stress_load)
        + 0.16 * social.social_energy
        + 0.16 * health.resilience
        - 0.16 * context["peer_pressure"]
    )
    task_interest = _clip01(
        0.34 * intrinsic
        + 0.22 * thinking.abstraction_level
        + 0.20 * thinking.ambiguity_tolerance
        + 0.18 * context["opportunity"]
        - 0.12 * stress
    )
    avoidance = _clip01(
        0.30 * failure
        + 0.22 * external
        + 0.18 * context["threat"]
        + 0.16 * (1.0 - regulation.persistence)
        + 0.14 * fatigue
        - 0.12 * context["support"]
    )
    recovery_capacity = _clip01(
        0.35 * health.resilience
        + 0.25 * regulation.emotional_regulation
        + 0.20 * regulation.metacognition
        + 0.12 * context["support"]
        + 0.08 * (1.0 - stress)
    )
    return {
        "stress": stress,
        "fatigue": fatigue,
        "perceived_autonomy": perceived_autonomy,
        "perceived_competence": perceived_competence,
        "social_safety": social_safety,
        "task_interest": task_interest,
        "avoidance": avoidance,
        "recovery_capacity": recovery_capacity,
    }


def _student_trait_vector(persona: MegaPersona) -> dict[str, float]:
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    regulation = profile.self_regulation
    thinking = profile.thinking_style
    social = persona.social_creative_profile
    health = persona.mental_health_context

    autonomous_drive_bonus = {
        "autonomy": 0.18,
        "curiosity": 0.15,
        "mastery": 0.12,
        "achievement": 0.04,
        "recognition": -0.05,
        "belonging": -0.04,
        "security": -0.07,
        "avoidance": -0.16,
    }.get(motivation.primary_drive, 0.0)
    defensive_drive_bonus = {
        "avoidance": 0.18,
        "security": 0.12,
        "recognition": 0.08,
        "belonging": 0.06,
        "achievement": 0.04,
    }.get(motivation.primary_drive, 0.0)
    growth_drive_bonus = {
        "curiosity": 0.16,
        "mastery": 0.14,
        "autonomy": 0.10,
        "achievement": 0.06,
    }.get(motivation.primary_drive, 0.0)

    cognitive_flexibility = _clip01(
        0.36 * thinking.abstraction_level
        + 0.34 * thinking.ambiguity_tolerance
        + 0.18 * regulation.metacognition
        + 0.12 * motivation.intrinsic_motivation
    )
    autonomous_orientation = _clip01(
        0.48 * motivation.intrinsic_motivation
        + 0.24 * (1.0 - motivation.external_pressure_sensitivity)
        + 0.12 * regulation.metacognition
        + autonomous_drive_bonus
    )
    defensive_orientation = _clip01(
        0.34 * motivation.failure_sensitivity
        + 0.30 * motivation.external_pressure_sensitivity
        + 0.16 * health.stress_load
        + defensive_drive_bonus
        - 0.12 * health.resilience
    )
    self_regulation_capacity = _clip01(
        0.26 * regulation.persistence
        + 0.24 * regulation.emotional_regulation
        + 0.24 * regulation.metacognition
        + 0.16 * regulation.habit_stability
        + 0.10 * health.resilience
    )
    social_confidence = _clip01(
        0.32 * social.social_energy
        + 0.24 * (1.0 - social.peer_influence_sensitivity)
        + 0.22 * health.resilience
        + 0.12 * regulation.emotional_regulation
        + 0.10 * (1.0 - health.stress_load)
    )
    growth_readiness = _clip01(
        0.28 * motivation.intrinsic_motivation
        + 0.22 * cognitive_flexibility
        + 0.22 * self_regulation_capacity
        + 0.16 * health.resilience
        + growth_drive_bonus
    )
    pressure_reactivity = _clip01(
        0.30 * motivation.external_pressure_sensitivity
        + 0.28 * motivation.failure_sensitivity
        + 0.22 * health.stress_load
        + 0.12 * social.peer_influence_sensitivity
        + 0.08 * (1.0 - regulation.emotional_regulation)
    )
    reflective_accuracy = _clip01(
        0.34 * regulation.metacognition
        + 0.24 * thinking.abstraction_level
        + 0.18 * regulation.emotional_regulation
        + 0.14 * health.resilience
        + 0.10 * (1.0 - defensive_orientation)
    )

    return {
        "cognitive_flexibility": cognitive_flexibility,
        "autonomous_orientation": autonomous_orientation,
        "defensive_orientation": defensive_orientation,
        "self_regulation_capacity": self_regulation_capacity,
        "social_confidence": social_confidence,
        "growth_readiness": growth_readiness,
        "pressure_reactivity": pressure_reactivity,
        "reflective_accuracy": reflective_accuracy,
    }


def _student_state_v2(
    persona: MegaPersona,
    trait_vector: dict[str, float],
    context: dict[str, float],
) -> dict[str, float]:
    base_state = _student_state(persona, context)
    threat = context["threat"]
    support = context["support"]
    opportunity = context["opportunity"]
    evaluation = context["external_evaluation"]
    peer_pressure = context["peer_pressure"]
    ambiguity = context["ambiguity"]

    stress = _clip01(
        0.58 * base_state["stress"]
        + 0.20 * trait_vector["pressure_reactivity"]
        + 0.16 * threat
        - 0.10 * support
    )
    fatigue = _clip01(
        0.62 * base_state["fatigue"]
        + 0.16 * stress
        + 0.12 * trait_vector["defensive_orientation"]
        - 0.08 * trait_vector["self_regulation_capacity"]
    )
    perceived_autonomy = _clip01(
        0.52 * base_state["perceived_autonomy"]
        + 0.30 * trait_vector["autonomous_orientation"]
        + 0.12 * opportunity
        - 0.12 * evaluation
        - 0.06 * peer_pressure
    )
    perceived_competence = _clip01(
        0.48 * base_state["perceived_competence"]
        + 0.28 * trait_vector["self_regulation_capacity"]
        + 0.12 * trait_vector["cognitive_flexibility"]
        + 0.08 * support
        - 0.12 * stress
    )
    social_safety = _clip01(
        0.48 * base_state["social_safety"]
        + 0.28 * trait_vector["social_confidence"]
        + 0.14 * support
        - 0.14 * peer_pressure
    )
    task_interest = _clip01(
        0.46 * base_state["task_interest"]
        + 0.26 * trait_vector["growth_readiness"]
        + 0.14 * opportunity
        + 0.08 * ambiguity
        - 0.12 * stress
    )
    avoidance = _clip01(
        0.48 * base_state["avoidance"]
        + 0.26 * trait_vector["defensive_orientation"]
        + 0.16 * stress
        + 0.08 * fatigue
        - 0.12 * support
    )
    recovery_capacity = _clip01(
        0.50 * base_state["recovery_capacity"]
        + 0.26 * trait_vector["self_regulation_capacity"]
        + 0.14 * support
        - 0.10 * fatigue
    )
    honest_self_access = _clip01(
        0.40 * trait_vector["reflective_accuracy"]
        + 0.24 * perceived_competence
        + 0.16 * social_safety
        + 0.12 * recovery_capacity
        - 0.12 * stress
    )
    state_variability = _clip01(
        0.30 * trait_vector["pressure_reactivity"]
        + 0.24 * ambiguity
        + 0.18 * peer_pressure
        + 0.16 * fatigue
        + 0.12 * (1.0 - recovery_capacity)
    )

    return {
        "stress": stress,
        "fatigue": fatigue,
        "perceived_autonomy": perceived_autonomy,
        "perceived_competence": perceived_competence,
        "social_safety": social_safety,
        "task_interest": task_interest,
        "avoidance": avoidance,
        "recovery_capacity": recovery_capacity,
        "honest_self_access": honest_self_access,
        "state_variability": state_variability,
    }


def _student_response_style(
    persona: MegaPersona,
    trait_vector: dict[str, float],
    state: dict[str, float],
) -> dict[str, float]:
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    social = persona.social_creative_profile
    health = persona.mental_health_context

    acquiescence = _clip01(
        0.30
        + 0.18 * social.peer_influence_sensitivity
        + 0.14 * motivation.external_pressure_sensitivity
        + 0.10 * state["social_safety"]
        - 0.12 * trait_vector["reflective_accuracy"]
    )
    social_desirability = _clip01(
        0.24
        + 0.24 * motivation.external_pressure_sensitivity
        + 0.18 * social.peer_influence_sensitivity
        + 0.14 * state["social_safety"]
        + 0.10 * trait_vector["defensive_orientation"]
    )
    uncertainty_preference = _clip01(
        0.18
        + 0.24 * state["state_variability"]
        + 0.20 * state["fatigue"]
        + 0.16 * (1.0 - trait_vector["reflective_accuracy"])
        + 0.10 * health.stress_load
    )
    extremity_preference = _clip01(
        0.16
        + 0.22 * (1.0 - uncertainty_preference)
        + 0.18 * trait_vector["reflective_accuracy"]
        + 0.14 * state["perceived_competence"]
        + 0.10 * abs(trait_vector["growth_readiness"] - trait_vector["defensive_orientation"])
    )
    fatigue_compression = _clip01(
        0.12
        + 0.36 * state["fatigue"]
        + 0.20 * state["stress"]
        + 0.12 * (1.0 - state["recovery_capacity"])
    )

    return {
        "acquiescence": acquiescence,
        "social_desirability": social_desirability,
        "uncertainty_preference": uncertainty_preference,
        "extremity_preference": extremity_preference,
        "fatigue_compression": fatigue_compression,
    }


def _blind_item_mechanisms(survey: ShadowSurvey) -> dict[str, str]:
    mechanisms: dict[str, str] = {}
    for item in survey.items:
        text = item.text.lower()
        construct = item.construct.replace("_", " ")
        labels: list[str] = []
        if any(word in text for word in ("curious", "understand", "idea", "complex", "different way")):
            labels.append("cognitive curiosity and abstraction")
        if any(word in text for word in ("choose", "own", "because i want", "meaningful", "interest")):
            labels.append("autonomous motivation")
        if any(word in text for word in ("grade", "approval", "praise", "others think", "pressure")):
            labels.append("external pressure or approval")
        if any(word in text for word in ("plan", "keep going", "finish", "habit", "setback", "recover")):
            labels.append("self-regulation and recovery")
        if any(word in text for word in ("avoid", "give up", "failure", "mistake", "criticism")):
            labels.append("avoidance or threat response")
        if any(word in text for word in ("friend", "peer", "group", "belong", "share")):
            labels.append("social safety and peer influence")
        if not labels:
            labels.append("general self-perception in context")
        mechanisms[item.item_id] = f'construct="{construct}"; likely mechanisms: {", ".join(labels)}'
    return mechanisms


def _item_mechanism_hint(item) -> str:
    axes = ", ".join(
        f"{axis}:{weight:.2f}"
        for axis, weight in sorted(item.axis_weights.items())
    ) or "none"
    construct = item.construct.replace("_", " ")
    roles = axis_roles_for_target_axes(item.axis_weights)
    if roles.get("motivation_core") in item.axis_weights:
        focus = "check internal ownership versus external pressure or fear"
    elif roles.get("regulation_core") in item.axis_weights:
        focus = "check stress, fatigue, avoidance, habits, and recovery capacity"
    elif roles.get("cognitive_core") in item.axis_weights:
        focus = "check abstraction, ambiguity tolerance, task interest, and threat"
    else:
        focus = "check context fit and persona-specific evidence"
    direction = "reverse-scored" if item.direction == -1 else "forward-scored"
    return (
        f'- {item.item_id}: construct="{construct}", {direction}, '
        f"axis_weights={axes}; mechanism={focus}."
    )


def _round_dict(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _generate_concordia_native_with_retry(
    *,
    llm_client,
    persona: MegaPersona,
    survey: ShadowSurvey,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    transient_markers = (
        "timeout",
        "timed out",
        "connection",
        "rate limit",
        "temporarily",
        "server",
        "overloaded",
    )
    label = f"concordia-native persona={persona.persona_id} survey={survey.survey_id}"
    for attempt in range(max_retries + 1):
        try:
            axis_roles = axis_roles_for_target_axes({axis: 0.5 for axis in survey.axis_names})
            bundle = build_concordia_agent_bundle(
                persona,
                llm_client,
                axis_names=survey.axis_names,
                axis_roles=axis_roles,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            bundle.agent.observe(survey_observation(survey))
            return bundle.agent.act(survey_action_spec(survey))
        except Exception as exc:
            message = str(exc).lower()
            transient = any(marker in message for marker in transient_markers)
            if attempt >= max_retries or not transient:
                raise
            wait_seconds = retry_backoff_seconds * (2 ** attempt)
            logger.warning(
                "Concordia native simulator transient failure; retrying attempt=%s/%s "
                "wait=%.1fs %s (%s: %s)",
                attempt + 1,
                max_retries,
                wait_seconds,
                label,
                type(exc).__name__,
                exc,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("unreachable Concordia native retry state")


def _generate_with_retry(
    *,
    llm,
    prompt: str,
    system_prompt: str,
    temperature: float,
    top_p: float | None,
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
            sampling_kwargs = {"top_p": top_p} if top_p is not None else {}
            return llm.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **sampling_kwargs,
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
