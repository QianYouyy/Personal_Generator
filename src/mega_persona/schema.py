"""Structured schema for HACHIMI-style large personas.

The former academic/teaching profile is intentionally modeled as cognitive
and motivational mechanisms: how the person thinks, why they act, and how
they self-regulate under difficulty.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.mega_persona.slots import AXIS_NAMES, AXIS_ROLE_MAP, LEGACY_AXIS_TO_ROLE


Score01 = float


FAMILY_CONTEXT_MAX_LENGTH = 1000
IDENTITY_ANCHOR_MAX_LENGTH = 480
MORAL_TENSION_MAX_LENGTH = 600
ASPIRATION_MAX_LENGTH = 600
SOCIAL_NARRATIVE_MAX_LENGTH = 1500
MENTAL_HEALTH_NARRATIVE_MAX_LENGTH = 1500
DERIVED_REASONING_MAX_LENGTH = 900


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Demographics(StrictModel):
    age: int = Field(ge=10, le=30)
    grade_or_stage: Literal[
        "middle_school",
        "high_school",
        "vocational",
        "undergraduate",
        "graduate",
        "early_career",
    ]
    region_type: Literal["urban", "suburban", "rural", "migrant", "international"]
    family_context: str = Field(min_length=20, max_length=FAMILY_CONTEXT_MAX_LENGTH)


class ThinkingStyle(StrictModel):
    dominant_mode: Literal[
        "analytical",
        "intuitive",
        "associative",
        "practical",
        "reflective",
        "exploratory",
    ]
    abstraction_level: Score01 = Field(ge=0.0, le=1.0)
    ambiguity_tolerance: Score01 = Field(ge=0.0, le=1.0)
    evidence_preference: Literal[
        "data",
        "authority",
        "experience",
        "peer_consensus",
        "personal_values",
    ]
    typical_blind_spot: str = Field(min_length=10, max_length=300)


class MotivationSystem(StrictModel):
    primary_drive: Literal[
        "mastery",
        "achievement",
        "belonging",
        "autonomy",
        "security",
        "recognition",
        "curiosity",
        "avoidance",
    ]
    secondary_drive: str = Field(min_length=3, max_length=120)
    intrinsic_motivation: Score01 = Field(ge=0.0, le=1.0)
    external_pressure_sensitivity: Score01 = Field(ge=0.0, le=1.0)
    failure_sensitivity: Score01 = Field(ge=0.0, le=1.0)
    reward_preference: Literal[
        "praise",
        "progress",
        "grades",
        "independence",
        "social_status",
        "usefulness",
    ]


class LearningOrientation(StrictModel):
    goal_orientation: Literal["mastery", "performance", "avoidance", "mixed"]
    preferred_learning_mode: Literal[
        "reading",
        "discussion",
        "practice",
        "visual",
        "project_based",
        "imitation",
        "trial_and_error",
    ]
    attention_pattern: Literal["sustained", "bursty", "easily_shifted", "hyperfocused"]
    curiosity_scope: Literal["narrow_deep", "broad_shallow", "situational", "low"]
    help_seeking_style: Literal[
        "proactive",
        "reluctant",
        "peer_first",
        "adult_first",
        "avoids_help",
    ]


class SelfRegulation(StrictModel):
    planning_style: Literal[
        "structured",
        "reactive",
        "deadline_driven",
        "ritual_based",
        "chaotic",
    ]
    persistence: Score01 = Field(ge=0.0, le=1.0)
    emotional_regulation: Score01 = Field(ge=0.0, le=1.0)
    metacognition: Score01 = Field(ge=0.0, le=1.0)
    habit_stability: Score01 = Field(ge=0.0, le=1.0)


class ChallengeResponse(StrictModel):
    under_difficulty: Literal[
        "doubles_down",
        "seeks_help",
        "freezes",
        "reframes",
        "distracts",
        "rebels",
    ]
    under_criticism: Literal["defensive", "curious", "ashamed", "motivated", "dismissive"]
    under_success: Literal[
        "confident",
        "complacent",
        "anxious_to_maintain",
        "generous",
        "exploratory",
    ]


class DecisionPattern(StrictModel):
    risk_appetite: Score01 = Field(ge=0.0, le=1.0)
    time_horizon: Literal["short", "medium", "long"]
    tradeoff_style: Literal[
        "safe_choice",
        "optimize_score",
        "protect_identity",
        "seek_growth",
        "maintain_relationships",
    ]
    typical_rationale: str = Field(min_length=20, max_length=500)


class CognitiveMotivationProfile(StrictModel):
    thinking_style: ThinkingStyle
    motivation_system: MotivationSystem
    learning_orientation: LearningOrientation
    self_regulation: SelfRegulation
    challenge_response: ChallengeResponse
    decision_pattern: DecisionPattern
    narrative: str = Field(min_length=250, max_length=1800)


class ValuesIdentity(StrictModel):
    core_values: list[str] = Field(min_length=2, max_length=6)
    identity_anchor: str = Field(min_length=20, max_length=IDENTITY_ANCHOR_MAX_LENGTH)
    moral_tension: str = Field(min_length=20, max_length=MORAL_TENSION_MAX_LENGTH)
    aspiration: str = Field(min_length=20, max_length=ASPIRATION_MAX_LENGTH)


class SocialCreativeProfile(StrictModel):
    social_energy: Score01 = Field(ge=0.0, le=1.0)
    collaboration_style: Literal["leader", "supporter", "observer", "challenger", "mediator", "solo"]
    expressiveness: Score01 = Field(ge=0.0, le=1.0)
    creative_mode: Literal[
        "original",
        "remixing",
        "practical",
        "aesthetic",
        "strategic",
        "low_expression",
    ]
    peer_influence_sensitivity: Score01 = Field(ge=0.0, le=1.0)
    narrative: str = Field(min_length=120, max_length=SOCIAL_NARRATIVE_MAX_LENGTH)


class MentalHealthContext(StrictModel):
    stress_load: Score01 = Field(ge=0.0, le=1.0)
    resilience: Score01 = Field(ge=0.0, le=1.0)
    coping_style: Literal[
        "problem_solving",
        "emotional_support",
        "avoidance",
        "humor",
        "control",
        "withdrawal",
    ]
    protective_factors: list[str] = Field(min_length=1, max_length=5)
    risk_factors: list[str] = Field(min_length=1, max_length=5)
    narrative: str = Field(min_length=120, max_length=MENTAL_HEALTH_NARRATIVE_MAX_LENGTH)


class DerivedAcademicTendency(StrictModel):
    likely_performance_band: Literal["poor", "low", "mid", "high"]
    reasoning: str = Field(min_length=40, max_length=DERIVED_REASONING_MAX_LENGTH)


class MegaPersona(StrictModel):
    persona_id: str = Field(min_length=3, max_length=80)
    demographics: Demographics
    cognitive_motivation_profile: CognitiveMotivationProfile
    values_identity: ValuesIdentity
    social_creative_profile: SocialCreativeProfile
    mental_health_context: MentalHealthContext
    derived_academic_tendency: DerivedAcademicTendency

    def primary_axes(
        self,
        axis_names: tuple[str, ...] = AXIS_NAMES,
        axis_roles: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Return primary continuous axes, optionally projected onto a schema binding."""
        cognitive = self.cognitive_motivation_profile
        motivation = cognitive.motivation_system
        regulation = cognitive.self_regulation
        health = self.mental_health_context
        canonical = {
            "cognitive_core": cognitive.thinking_style.abstraction_level,
            "motivation_core": (
                motivation.intrinsic_motivation
                + (1.0 - motivation.external_pressure_sensitivity)
            )
            / 2.0,
            "regulation_core": (
                regulation.persistence
                + regulation.emotional_regulation
                + regulation.metacognition
                + regulation.habit_stability
                + health.resilience
            )
            / 5.0,
        }
        roles = dict(AXIS_ROLE_MAP)
        if axis_roles:
            roles.update(axis_roles)

        projected: dict[str, float] = {}
        for axis_name in axis_names:
            role = LEGACY_AXIS_TO_ROLE.get(axis_name)
            if role is not None:
                projected[axis_name] = canonical[role]
                continue
            matched_role = next(
                (role_name for role_name, mapped_axis in roles.items() if mapped_axis == axis_name),
                None,
            )
            if matched_role in canonical:
                projected[axis_name] = canonical[matched_role]
            else:
                projected[axis_name] = 0.5
        return projected
