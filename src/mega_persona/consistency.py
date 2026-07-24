"""Internal consistency scoring for large MegaPersona objects.

The schema validator catches hard invalid cases. This module provides a softer
continuous score for scientific evaluation: personas can be schema-valid while
still containing implausible cross-field tensions or drifting away from the
target Monte Carlo slot.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.mega_persona.schema import MegaPersona
from src.mega_persona.slots import AXIS_NAMES, AXIS_ROLE_MAP, MegaPersonaSlot


@dataclass(frozen=True)
class ConsistencyIssue:
    rule_id: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True)
class PersonaConsistencyReport:
    persona_id: str
    score: float
    axis_alignment: float
    rule_score: float
    issues: list[ConsistencyIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "score": self.score,
            "axis_alignment": self.axis_alignment,
            "rule_score": self.rule_score,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class PopulationConsistencyEvaluation:
    sample_size: int
    mean_score: float
    min_score: float
    axis_alignment_mean: float
    rule_score_mean: float
    axis_target_mae_mean: float
    weighted_issue_rate: float
    consistency_issue_rate: float
    strict_consistency_error: float
    issue_count: int
    rule_counts: dict[str, int]
    reports: list[PersonaConsistencyReport] = field(default_factory=list)

    def to_dict(self, include_reports: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_size": self.sample_size,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "axis_alignment_mean": self.axis_alignment_mean,
            "rule_score_mean": self.rule_score_mean,
            "axis_target_mae_mean": self.axis_target_mae_mean,
            "weighted_issue_rate": self.weighted_issue_rate,
            "consistency_issue_rate": self.consistency_issue_rate,
            "strict_consistency_error": self.strict_consistency_error,
            "issue_count": self.issue_count,
            "rule_counts": self.rule_counts,
        }
        if include_reports:
            payload["reports"] = [report.to_dict() for report in self.reports]
        return payload


def evaluate_population_consistency(
    personas: list[MegaPersona],
    slots: list[MegaPersonaSlot] | None = None,
    axis_names: tuple[str, ...] = AXIS_NAMES,
    axis_roles: dict[str, str] | None = None,
) -> PopulationConsistencyEvaluation:
    """Score target-axis fidelity and cross-field psychological plausibility."""
    slot_by_id = {slot.slot_id: slot for slot in slots or []}
    reports = [
        evaluate_persona_consistency(
            persona,
            slot_by_id.get(persona.persona_id),
            axis_names=axis_names,
            axis_roles=axis_roles,
        )
        for persona in personas
    ]
    if not reports:
        return PopulationConsistencyEvaluation(
            sample_size=0,
            mean_score=0.0,
            min_score=0.0,
            axis_alignment_mean=0.0,
            rule_score_mean=0.0,
            axis_target_mae_mean=1.0,
            weighted_issue_rate=1.0,
            consistency_issue_rate=1.0,
            strict_consistency_error=1.0,
            issue_count=0,
            rule_counts={},
            reports=[],
        )

    rule_counts: dict[str, int] = {}
    weighted_penalty = 0.0
    for report in reports:
        for issue in report.issues:
            rule_counts[issue.rule_id] = rule_counts.get(issue.rule_id, 0) + 1
            weighted_penalty += _issue_penalty(issue)
    axis_target_mae_mean = float(np.mean([1.0 - report.axis_alignment for report in reports]))
    weighted_issue_rate = _clip01(weighted_penalty / len(reports))
    consistency_issue_rate = float(sum(len(report.issues) for report in reports) / len(reports))
    strict_consistency_error = _clip01(0.70 * axis_target_mae_mean + 0.30 * weighted_issue_rate)
    return PopulationConsistencyEvaluation(
        sample_size=len(reports),
        mean_score=float(np.mean([report.score for report in reports])),
        min_score=float(np.min([report.score for report in reports])),
        axis_alignment_mean=float(np.mean([report.axis_alignment for report in reports])),
        rule_score_mean=float(np.mean([report.rule_score for report in reports])),
        axis_target_mae_mean=axis_target_mae_mean,
        weighted_issue_rate=weighted_issue_rate,
        consistency_issue_rate=consistency_issue_rate,
        strict_consistency_error=strict_consistency_error,
        issue_count=sum(len(report.issues) for report in reports),
        rule_counts=dict(sorted(rule_counts.items())),
        reports=reports,
    )


def evaluate_persona_consistency(
    persona: MegaPersona,
    slot: MegaPersonaSlot | None = None,
    axis_names: tuple[str, ...] = AXIS_NAMES,
    axis_roles: dict[str, str] | None = None,
) -> PersonaConsistencyReport:
    issues: list[ConsistencyIssue] = []
    axis_alignment = _axis_alignment(persona, slot, axis_names=axis_names, axis_roles=axis_roles)
    _check_regulation_consistency(persona, issues)
    _check_motivation_consistency(persona, issues)
    _check_challenge_coping_consistency(persona, issues)
    _check_social_consistency(persona, issues)
    _check_performance_consistency(persona, issues, axis_roles=axis_roles)
    _check_stress_recovery_consistency(persona, issues)

    penalty = sum(_issue_penalty(issue) for issue in issues)
    rule_score = _clip01(1.0 - min(0.65, penalty))
    score = _clip01(0.70 * axis_alignment + 0.30 * rule_score)
    return PersonaConsistencyReport(
        persona_id=persona.persona_id,
        score=score,
        axis_alignment=axis_alignment,
        rule_score=rule_score,
        issues=issues,
    )


def _axis_alignment(
    persona: MegaPersona,
    slot: MegaPersonaSlot | None,
    *,
    axis_names: tuple[str, ...],
    axis_roles: dict[str, str] | None,
) -> float:
    if slot is None:
        return 1.0
    compare_axes = tuple(slot.target_axes.keys()) if slot.target_axes else axis_names
    persona_axes = persona.primary_axes(axis_names=compare_axes, axis_roles=axis_roles)
    diffs = [
        abs(float(persona_axes[axis]) - float(slot.target_axes[axis]))
        for axis in compare_axes
        if axis in persona_axes and axis in slot.target_axes
    ]
    if not diffs:
        return 1.0
    return _clip01(1.0 - float(np.mean(diffs)))


def _check_regulation_consistency(
    persona: MegaPersona,
    issues: list[ConsistencyIssue],
) -> None:
    profile = persona.cognitive_motivation_profile
    regulation = profile.self_regulation
    reg_mean = _regulation_mean(persona)
    if regulation.planning_style == "structured" and reg_mean < 0.42:
        _issue(issues, "C1", "error", "Structured planning conflicts with low regulation traits.")
    if regulation.planning_style == "chaotic" and reg_mean > 0.72:
        _issue(issues, "C2", "warning", "Chaotic planning needs explanation when regulation traits are high.")
    if profile.learning_orientation.attention_pattern == "sustained" and reg_mean < 0.45:
        _issue(issues, "C3", "warning", "Sustained attention conflicts with weak persistence and habit stability.")
    if profile.learning_orientation.attention_pattern == "easily_shifted" and reg_mean > 0.72:
        _issue(issues, "C4", "warning", "Easily shifted attention conflicts with high regulation traits.")


def _check_motivation_consistency(
    persona: MegaPersona,
    issues: list[ConsistencyIssue],
) -> None:
    motivation = persona.cognitive_motivation_profile.motivation_system
    drive = motivation.primary_drive
    if drive in {"mastery", "autonomy", "curiosity"} and motivation.intrinsic_motivation < 0.35:
        _issue(issues, "C5", "error", f"{drive} drive conflicts with low intrinsic motivation.")
    if drive in {"recognition", "achievement"} and motivation.external_pressure_sensitivity < 0.20:
        _issue(issues, "C6", "warning", f"{drive} drive usually needs some sensitivity to external evaluation.")
    if motivation.reward_preference == "independence" and motivation.external_pressure_sensitivity > 0.78:
        _issue(issues, "C7", "warning", "Independence reward conflicts with very high external pressure sensitivity.")


def _check_challenge_coping_consistency(
    persona: MegaPersona,
    issues: list[ConsistencyIssue],
) -> None:
    profile = persona.cognitive_motivation_profile
    health = persona.mental_health_context
    reg_mean = _regulation_mean(persona)
    if profile.challenge_response.under_difficulty == "reframes" and reg_mean < 0.35:
        _issue(issues, "C8", "warning", "Reframing difficulty conflicts with low regulation unless explained.")
    if profile.challenge_response.under_difficulty == "freezes" and reg_mean > 0.72:
        _issue(issues, "C9", "warning", "Freezing under difficulty conflicts with high regulation traits.")
    if health.coping_style == "problem_solving" and reg_mean < 0.35:
        _issue(issues, "C10", "warning", "Problem-solving coping conflicts with low regulation.")
    if health.coping_style in {"avoidance", "withdrawal"} and reg_mean > 0.72:
        _issue(issues, "C11", "warning", "Avoidant coping conflicts with high regulation and resilience.")


def _check_social_consistency(
    persona: MegaPersona,
    issues: list[ConsistencyIssue],
) -> None:
    social = persona.social_creative_profile
    if social.collaboration_style == "leader" and social.social_energy < 0.45:
        _issue(issues, "C12", "error", "Leader collaboration style conflicts with low social energy.")
    if social.collaboration_style == "solo" and social.social_energy > 0.78:
        _issue(issues, "C13", "warning", "Solo collaboration style conflicts with very high social energy.")
    if social.expressiveness > 0.82 and social.social_energy < 0.25:
        _issue(issues, "C14", "warning", "Very high expressiveness conflicts with very low social energy.")


def _check_performance_consistency(
    persona: MegaPersona,
    issues: list[ConsistencyIssue],
    axis_roles: dict[str, str] | None = None,
) -> None:
    profile = persona.cognitive_motivation_profile
    health = persona.mental_health_context
    band = persona.derived_academic_tendency.likely_performance_band
    reg_mean = _regulation_mean(persona)
    autonomy = _role_axis_value(persona, "motivation_core", axis_roles=axis_roles)
    if band == "high" and reg_mean < 0.38 and autonomy < 0.45:
        _issue(issues, "C15", "error", "High performance lacks regulation or motivational support.")
    if band == "poor" and reg_mean > 0.76 and autonomy > 0.65 and health.stress_load < 0.45:
        _issue(issues, "C16", "warning", "Poor performance needs clearer contextual friction with strong regulation and motivation.")
    if profile.learning_orientation.goal_orientation == "avoidance" and band == "high" and reg_mean < 0.55:
        _issue(issues, "C17", "warning", "High performance with avoidance orientation needs stronger regulation or support.")


def _check_stress_recovery_consistency(
    persona: MegaPersona,
    issues: list[ConsistencyIssue],
) -> None:
    health = persona.mental_health_context
    reg_mean = _regulation_mean(persona)
    if health.stress_load > 0.78 and health.resilience > 0.78 and reg_mean < 0.42:
        _issue(issues, "C18", "warning", "High stress and high resilience conflict with weak regulation.")
    if health.stress_load < 0.25 and health.risk_factors and health.resilience < 0.25:
        _issue(issues, "C19", "warning", "Low stress and low resilience need more contextual explanation.")


def _regulation_mean(persona: MegaPersona) -> float:
    regulation = persona.cognitive_motivation_profile.self_regulation
    return float(
        np.mean(
            [
                regulation.persistence,
                regulation.emotional_regulation,
                regulation.metacognition,
                regulation.habit_stability,
                persona.mental_health_context.resilience,
            ]
        )
    )


def _issue(
    issues: list[ConsistencyIssue],
    rule_id: str,
    severity: str,
    message: str,
) -> None:
    issues.append(ConsistencyIssue(rule_id=rule_id, severity=severity, message=message))


def _issue_penalty(issue: ConsistencyIssue) -> float:
    return 0.08 if issue.severity == "error" else 0.035


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _role_axis_value(
    persona: MegaPersona,
    role: str,
    axis_roles: dict[str, str] | None = None,
) -> float:
    roles = dict(AXIS_ROLE_MAP)
    if axis_roles:
        roles.update(axis_roles)
    axis_name = roles.get(role)
    if not axis_name:
        return 0.5
    return float(
        persona.primary_axes(axis_names=(axis_name,), axis_roles=roles).get(axis_name, 0.5)
    )
