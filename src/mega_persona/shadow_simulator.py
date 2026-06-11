"""Behavior simulation for MegaPersona shadow surveys."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_survey import (
    ShadowSurvey,
    ShadowSurveyItem,
    score_shadow_survey,
)
from src.mega_persona.slots import AXIS_NAMES


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


class RuleBasedShadowSimulator:
    """Project structured personas into Likert responses.

    This is a deterministic, inspectable baseline. It is useful for offline
    tests and ablations before replacing the simulator with LLM-based
    role-play.
    """

    def __init__(self, noise: float = 0.08, seed: int | None = 17):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def simulate_population(
        self,
        personas: list[MegaPersona],
        surveys: list[ShadowSurvey],
    ) -> list[ShadowSurveySimulation]:
        simulations: list[ShadowSurveySimulation] = []
        for persona in personas:
            for survey in surveys:
                simulations.append(self.simulate_persona(persona, survey))
        return simulations

    def simulate_persona(
        self,
        persona: MegaPersona,
        survey: ShadowSurvey,
    ) -> ShadowSurveySimulation:
        responses = {
            item.item_id: self._response_for_item(persona, item)
            for item in survey.items
        }
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

    def _response_for_item(self, persona: MegaPersona, item: ShadowSurveyItem) -> int:
        target_score = self._target_score(persona, item)
        if self.noise:
            target_score = float(
                np.clip(target_score + self.rng.normal(0.0, self.noise), 0.0, 1.0)
            )
        raw_agreement = target_score if item.direction == 1 else 1.0 - target_score
        return int(np.clip(round(raw_agreement * 4.0) + 1, 1, 5))

    def _target_score(self, persona: MegaPersona, item: ShadowSurveyItem) -> float:
        axes = persona.primary_axes()
        axis_score = 0.5
        if item.axis_weights:
            total_weight = sum(abs(weight) for weight in item.axis_weights.values())
            if total_weight:
                axis_score = sum(
                    axes[axis] * weight
                    for axis, weight in item.axis_weights.items()
                ) / total_weight

        construct_score = _construct_signal(persona, item.construct)
        return float(np.clip(0.7 * axis_score + 0.3 * construct_score, 0.0, 1.0))


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


def _construct_signal(persona: MegaPersona, construct: str) -> float:
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    regulation = profile.self_regulation
    social = persona.social_creative_profile
    health = persona.mental_health_context
    values = persona.values_identity
    axes = persona.primary_axes()

    signals = {
        "cognitive_abstraction": axes["cognitive_abstraction"],
        "ambiguity_tolerance": profile.thinking_style.ambiguity_tolerance,
        "motivation_autonomy": axes["motivation_autonomy"],
        "intrinsic_motivation": motivation.intrinsic_motivation,
        "external_pressure": 1.0 - motivation.external_pressure_sensitivity,
        "self_regulation": axes["self_regulation_resilience"],
        "metacognition": regulation.metacognition,
        "emotional_regulation": regulation.emotional_regulation,
        "stress_load": 1.0 - health.stress_load,
        "resilience": health.resilience,
        "identity_clarity": min(1.0, 0.45 + 0.08 * len(values.core_values)),
        "value_tension": 0.45,
        "belonging": 0.55 if motivation.primary_drive == "belonging" else 0.45,
        "peer_influence": 1.0 - social.peer_influence_sensitivity,
        "creative_orientation": social.expressiveness,
        "risk_appetite": profile.decision_pattern.risk_appetite,
        "help_seeking": 0.75
        if profile.learning_orientation.help_seeking_style in {"proactive", "peer_first", "adult_first"}
        else 0.35,
        "mastery_orientation": 0.75
        if profile.learning_orientation.goal_orientation == "mastery"
        else 0.45,
        "avoidance_orientation": 0.3
        if profile.learning_orientation.goal_orientation == "avoidance"
        else 0.65,
        "future_orientation": 0.8
        if profile.decision_pattern.time_horizon == "long"
        else 0.55
        if profile.decision_pattern.time_horizon == "medium"
        else 0.35,
    }
    return float(np.clip(signals.get(construct, 0.5), 0.0, 1.0))
