"""Evaluation helpers for schema-constrained MegaPersona populations."""

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np

from src.evaluator.metrics import DiversityMetrics
from src.mega_persona.schema import MegaPersona
from src.mega_persona.slots import AXIS_NAMES
from src.mega_persona.validator import ValidationReport, validate_mega_persona


@dataclass
class MegaPersonaEvaluation:
    sample_size: int
    valid_count: int
    validity_rate: float
    near_duplicate_rate: float
    axis_names: tuple[str, ...]
    diversity_metrics: dict[str, float]
    fitness: float
    validation_reports: list[ValidationReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "valid_count": self.valid_count,
            "validity_rate": self.validity_rate,
            "near_duplicate_rate": self.near_duplicate_rate,
            "axis_names": list(self.axis_names),
            "diversity_metrics": self.diversity_metrics,
            "fitness": self.fitness,
        }


def evaluate_mega_personas(
    personas: list[dict[str, Any] | MegaPersona],
    coverage_radius: float = 0.28,
    duplicate_threshold: float = 0.82,
    axis_names: tuple[str, ...] = AXIS_NAMES,
    axis_roles: dict[str, str] | None = None,
) -> MegaPersonaEvaluation:
    """Evaluate a generated MegaPersona population.

    The score deliberately multiplies quality gates with diversity: invalid or
    near-duplicate populations cannot receive a high final fitness.
    """
    reports = [validate_mega_persona(persona) for persona in personas]
    valid_personas = [
        persona if isinstance(persona, MegaPersona) else MegaPersona.model_validate(persona)
        for persona, report in zip(personas, reports)
        if report.is_valid
    ]

    sample_size = len(personas)
    valid_count = len(valid_personas)
    validity_rate = valid_count / sample_size if sample_size else 0.0
    near_duplicate_rate = _near_duplicate_rate(valid_personas, duplicate_threshold)
    axis_matrix = personas_to_axis_matrix(valid_personas, axis_names, axis_roles=axis_roles)

    if len(axis_matrix) == 0:
        diversity_metrics = {
            "coverage": 0.0,
            "convex_hull": 0.0,
            "avg_dist": 0.0,
            "min_dist": 0.0,
            "dispersion": 0.0,
            "kl_divergence": 0.0,
        }
    else:
        diversity_metrics = DiversityMetrics(coverage_radius).fitness(axis_matrix)

    fitness = _combined_fitness(
        diversity_metrics=diversity_metrics,
        validity_rate=validity_rate,
        near_duplicate_rate=near_duplicate_rate,
        dim=len(axis_names),
    )
    return MegaPersonaEvaluation(
        sample_size=sample_size,
        valid_count=valid_count,
        validity_rate=validity_rate,
        near_duplicate_rate=near_duplicate_rate,
        axis_names=axis_names,
        diversity_metrics=diversity_metrics,
        fitness=fitness,
        validation_reports=reports,
    )


def personas_to_axis_matrix(
    personas: list[dict[str, Any] | MegaPersona],
    axis_names: tuple[str, ...] = AXIS_NAMES,
    axis_roles: dict[str, str] | None = None,
) -> np.ndarray:
    rows = []
    for persona_data in personas:
        persona = (
            persona_data
            if isinstance(persona_data, MegaPersona)
            else MegaPersona.model_validate(persona_data)
        )
        axes = persona.primary_axes(axis_names=axis_names, axis_roles=axis_roles)
        rows.append([axes[name] for name in axis_names])
    if not rows:
        return np.empty((0, len(axis_names)))
    return np.array(rows, dtype=float)


def _combined_fitness(
    diversity_metrics: dict[str, float],
    validity_rate: float,
    near_duplicate_rate: float,
    dim: int,
) -> float:
    max_distance = float(np.sqrt(dim))
    avg_dist_norm = _clip01(diversity_metrics.get("avg_dist", 0.0) / max_distance)
    min_dist_norm = _clip01(diversity_metrics.get("min_dist", 0.0) / max_distance)
    coverage = _clip01(diversity_metrics.get("coverage", 0.0))
    diversity_score = 0.65 * coverage + 0.25 * avg_dist_norm + 0.10 * min_dist_norm
    return float(validity_rate * (1.0 - near_duplicate_rate) * diversity_score)


def _near_duplicate_rate(
    personas: list[MegaPersona],
    threshold: float,
) -> float:
    if len(personas) <= 1:
        return 0.0
    duplicate_pairs = 0
    total_pairs = 0
    token_sets = [_persona_tokens(persona) for persona in personas]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            total_pairs += 1
            if _jaccard(token_sets[i], token_sets[j]) >= threshold:
                duplicate_pairs += 1
    return duplicate_pairs / total_pairs if total_pairs else 0.0


def _persona_tokens(persona: MegaPersona) -> set[str]:
    text = " ".join(
        [
            persona.demographics.family_context,
            persona.cognitive_motivation_profile.narrative,
            persona.values_identity.identity_anchor,
            persona.values_identity.moral_tension,
            persona.values_identity.aspiration,
            persona.social_creative_profile.narrative,
            persona.mental_health_context.narrative,
            persona.derived_academic_tendency.reasoning,
        ]
    )
    return set(re.findall(r"[a-zA-Z][a-zA-Z_'-]{2,}", text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
