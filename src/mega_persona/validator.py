"""Symbolic validation rules for MegaPersona objects."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.mega_persona.schema import MegaPersona


@dataclass
class ValidationIssue:
    rule_id: str
    severity: str
    message: str


@dataclass
class ValidationReport:
    is_valid: bool
    schema_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def hard_error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")


def validate_mega_persona(data: dict[str, Any] | MegaPersona) -> ValidationReport:
    """Validate schema plus cross-field psychological consistency rules."""
    issues: list[ValidationIssue] = []
    try:
        persona = data if isinstance(data, MegaPersona) else MegaPersona.model_validate(data)
    except ValidationError as exc:
        return ValidationReport(
            is_valid=False,
            schema_valid=False,
            issues=[
                ValidationIssue(
                    rule_id="SCHEMA",
                    severity="error",
                    message=str(exc),
                )
            ],
        )

    _check_intrinsic_avoidance(persona, issues)
    _check_chaotic_persistence(persona, issues)
    _check_high_stress_resilience(persona, issues)
    _check_external_pressure_consistency(persona, issues)
    _check_trait_extremes(persona, issues)
    _check_high_performance_grounding(persona, issues)
    _check_low_performance_non_deficit(persona, issues)
    _check_primary_axis_alignment(persona, issues)

    return ValidationReport(
        is_valid=not any(issue.severity == "error" for issue in issues),
        schema_valid=True,
        issues=issues,
    )


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _issue(issues: list[ValidationIssue], rule_id: str, severity: str, message: str) -> None:
    issues.append(ValidationIssue(rule_id=rule_id, severity=severity, message=message))


def _check_intrinsic_avoidance(persona: MegaPersona, issues: list[ValidationIssue]) -> None:
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    learning = profile.learning_orientation
    if motivation.intrinsic_motivation >= 0.75 and learning.goal_orientation == "avoidance":
        if not _contains_any(profile.narrative, ["fear", "avoid", "pressure", "shame", "protect"]):
            _issue(
                issues,
                "R1",
                "error",
                "High intrinsic motivation with avoidance orientation needs an explained tension.",
            )


def _check_chaotic_persistence(persona: MegaPersona, issues: list[ValidationIssue]) -> None:
    regulation = persona.cognitive_motivation_profile.self_regulation
    narrative = persona.cognitive_motivation_profile.narrative
    if regulation.persistence >= 0.75 and regulation.planning_style == "chaotic":
        if not _contains_any(narrative, ["deadline", "ritual", "burst", "crisis", "external structure"]):
            _issue(
                issues,
                "R2",
                "warning",
                "High persistence with chaotic planning should explain what sustains effort.",
            )


def _check_high_stress_resilience(persona: MegaPersona, issues: list[ValidationIssue]) -> None:
    health = persona.mental_health_context
    if health.stress_load >= 0.7 and health.resilience >= 0.7 and not health.protective_factors:
        _issue(
            issues,
            "R3",
            "error",
            "High stress plus high resilience requires at least one protective factor.",
        )


def _check_external_pressure_consistency(persona: MegaPersona, issues: list[ValidationIssue]) -> None:
    motivation = persona.cognitive_motivation_profile.motivation_system
    social = persona.social_creative_profile
    if (
        motivation.external_pressure_sensitivity >= 0.75
        and social.peer_influence_sensitivity <= 0.2
        and motivation.reward_preference in {"praise", "grades", "social_status"}
    ):
        _issue(
            issues,
            "R4",
            "warning",
            "High external pressure sensitivity conflicts with very low peer influence.",
        )


def _check_trait_extremes(persona: MegaPersona, issues: list[ValidationIssue]) -> None:
    values = _numeric_trait_values(persona)
    high_count = sum(value >= 0.8 for value in values)
    low_count = sum(value <= 0.2 for value in values)
    if high_count >= len(values) - 1:
        _issue(issues, "R5", "error", "Persona is nearly all-high across numeric traits.")
    if low_count >= len(values) - 1:
        _issue(issues, "R6", "error", "Persona is nearly all-low across numeric traits.")


def _check_high_performance_grounding(persona: MegaPersona, issues: list[ValidationIssue]) -> None:
    derived = persona.derived_academic_tendency
    if derived.likely_performance_band != "high":
        return
    grounding_terms = [
        "habit",
        "strategy",
        "resource",
        "practice",
        "feedback",
        "motivation",
        "planning",
        "support",
    ]
    if not _contains_any(derived.reasoning, grounding_terms):
        _issue(
            issues,
            "R7",
            "error",
            "High likely performance must be grounded in habits, strategy, resources, support, or motivation.",
        )


def _check_low_performance_non_deficit(persona: MegaPersona, issues: list[ValidationIssue]) -> None:
    derived = persona.derived_academic_tendency
    if derived.likely_performance_band not in {"poor", "low"}:
        return
    banned_terms = ["low iq", "stupid", "lazy by nature", "incapable", "unintelligent"]
    if _contains_any(derived.reasoning, banned_terms):
        _issue(
            issues,
            "R8",
            "error",
            "Poor/low performance cannot be explained as fixed ability or moral deficit.",
        )


def _check_primary_axis_alignment(persona: MegaPersona, issues: list[ValidationIssue]) -> None:
    axes = persona.primary_axes()
    for name, value in axes.items():
        if not 0.0 <= value <= 1.0:
            _issue(issues, "R9", "error", f"Primary axis {name} is outside [0, 1].")


def _numeric_trait_values(persona: MegaPersona) -> list[float]:
    profile = persona.cognitive_motivation_profile
    return [
        profile.thinking_style.abstraction_level,
        profile.thinking_style.ambiguity_tolerance,
        profile.motivation_system.intrinsic_motivation,
        profile.motivation_system.external_pressure_sensitivity,
        profile.motivation_system.failure_sensitivity,
        profile.self_regulation.persistence,
        profile.self_regulation.emotional_regulation,
        profile.self_regulation.metacognition,
        profile.self_regulation.habit_stability,
        profile.decision_pattern.risk_appetite,
        persona.social_creative_profile.social_energy,
        persona.social_creative_profile.expressiveness,
        persona.social_creative_profile.peer_influence_sensitivity,
        persona.mental_health_context.stress_load,
        persona.mental_health_context.resilience,
    ]
