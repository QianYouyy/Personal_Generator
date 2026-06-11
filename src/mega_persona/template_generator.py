"""Rule-based MegaPersona builder used as an offline baseline."""

from src.mega_persona.schema import MegaPersona
from src.mega_persona.slots import MegaPersonaSlot


class RuleBasedMegaPersonaBuilder:
    """Create valid, slot-conditioned MegaPersonas without an LLM."""

    def build_population(self, slots: list[MegaPersonaSlot]) -> list[MegaPersona]:
        return [self.build(slot, index) for index, slot in enumerate(slots, start=1)]

    def build(self, slot: MegaPersonaSlot, index: int = 1) -> MegaPersona:
        axes = slot.target_axes
        constraints = slot.constraints
        abstraction = axes["cognitive_abstraction"]
        autonomy = axes["motivation_autonomy"]
        regulation = axes["self_regulation_resilience"]
        label = slot.quota_label
        theme = THEMES[(index - 1) % len(THEMES)]

        primary_drive = constraints.get("primary_drive", "mastery")
        stage = constraints.get("grade_or_stage", "high_school")
        performance = constraints.get("derived_performance_band", "mid")
        stress_load = _stress_value(constraints.get("stress_band", "mid"), regulation)
        social_energy = _social_energy_value(constraints.get("social_energy_band", "mid"), index)

        data = {
            "persona_id": slot.slot_id,
            "demographics": {
                "age": _age_for_stage(stage, index),
                "grade_or_stage": stage,
                "region_type": _pick(REGION_TYPES, index),
                "family_context": (
                    f"This person is shaped by {theme['setting']}. Their household provides "
                    f"{theme['resource']} but also creates {theme['constraint']}. Daily life has "
                    "enough stability for routines to form, while still leaving real tradeoffs "
                    "around attention, time, privacy, and support."
                ),
            },
            "cognitive_motivation_profile": {
                "thinking_style": {
                    "dominant_mode": _thinking_mode(abstraction, index),
                    "abstraction_level": abstraction,
                    "ambiguity_tolerance": _clip(0.35 + 0.45 * abstraction + 0.15 * autonomy),
                    "evidence_preference": _evidence_preference(abstraction, autonomy, index),
                    "typical_blind_spot": (
                        "They can overuse the strategy that usually works for them and miss "
                        "signals that the current situation needs a different kind of attention."
                    ),
                },
                "motivation_system": {
                    "primary_drive": primary_drive,
                    "secondary_drive": theme["secondary_drive"],
                    "intrinsic_motivation": _clip(0.15 + 0.8 * autonomy),
                    "external_pressure_sensitivity": _clip(1.0 - autonomy),
                    "failure_sensitivity": _clip(0.65 * stress_load + 0.25 * (1.0 - regulation)),
                    "reward_preference": _reward_preference(primary_drive, autonomy),
                },
                "learning_orientation": {
                    "goal_orientation": _goal_orientation(primary_drive, regulation),
                    "preferred_learning_mode": _pick(LEARNING_MODES, index),
                    "attention_pattern": _attention_pattern(regulation),
                    "curiosity_scope": _curiosity_scope(abstraction, autonomy),
                    "help_seeking_style": _help_seeking_style(regulation, social_energy),
                },
                "self_regulation": {
                    "planning_style": _planning_style(regulation),
                    "persistence": _clip(regulation + 0.08),
                    "emotional_regulation": _clip(regulation - 0.03),
                    "metacognition": _clip((regulation + abstraction) / 2.0),
                    "habit_stability": _clip(regulation - 0.08),
                },
                "challenge_response": {
                    "under_difficulty": _difficulty_response(regulation, social_energy),
                    "under_criticism": _criticism_response(autonomy, regulation),
                    "under_success": _success_response(autonomy, abstraction),
                },
                "decision_pattern": {
                    "risk_appetite": _clip(0.25 + 0.45 * autonomy + 0.2 * abstraction),
                    "time_horizon": "long" if regulation >= 0.72 else "medium" if regulation >= 0.38 else "short",
                    "tradeoff_style": _tradeoff_style(primary_drive, autonomy, regulation),
                    "typical_rationale": (
                        "They choose by asking whether the option protects agency, keeps recovery possible, "
                        "and produces evidence they can use for the next decision."
                    ),
                },
                "narrative": _cognition_narrative(
                    theme=theme,
                    label=label,
                    abstraction=abstraction,
                    autonomy=autonomy,
                    regulation=regulation,
                    primary_drive=primary_drive,
                    adaptive_constraints=slot.adaptive_constraints,
                ),
            },
            "values_identity": {
                "core_values": _core_values(primary_drive, theme["value"]),
                "identity_anchor": (
                    f"They see themself as a {theme['identity']}, someone whose worth comes from "
                    "turning pressure into a usable next move rather than performing perfection."
                ),
                "moral_tension": (
                    "They want to stay loyal to people who count on them, but they also resent "
                    "being treated as if reliability means never needing room to experiment."
                ),
                "aspiration": (
                    f"They want to become a person who can use {theme['domain']} as a path toward "
                    "competence, steadier judgment, and a more self-authored future."
                ),
            },
            "social_creative_profile": {
                "social_energy": social_energy,
                "collaboration_style": _collaboration_style(social_energy, autonomy),
                "expressiveness": _clip(0.25 + 0.5 * social_energy + 0.15 * autonomy),
                "creative_mode": _creative_mode(abstraction, autonomy, index),
                "peer_influence_sensitivity": _clip(0.65 - 0.45 * autonomy + 0.25 * social_energy),
                "narrative": (
                    f"Socially, they are most alive around {theme['setting']}. They do not relate "
                    "to peers in a single fixed way: sometimes they observe first, sometimes they "
                    "test a practical contribution, and sometimes they become more expressive when "
                    "the group gives them a visible role. Their creativity is tied to making "
                    f"{theme['domain']} feel workable rather than impressive in the abstract."
                ),
            },
            "mental_health_context": {
                "stress_load": stress_load,
                "resilience": _clip(regulation),
                "coping_style": _coping_style(regulation, autonomy),
                "protective_factors": [theme["protective"], "one trusted relationship"],
                "risk_factors": [theme["risk"], "pressure to appear more consistent than they feel"],
                "narrative": (
                    "Stress shows up through changes in attention, sleep, and willingness to ask for help. "
                    f"Their recovery is strongest when {theme['protective']} is available and weakest when "
                    f"{theme['risk']} becomes the whole frame of the week. This is not a diagnosis; it is "
                    "a behavioral context for understanding when their usual strategies hold or break."
                ),
            },
            "derived_academic_tendency": {
                "likely_performance_band": performance,
                "reasoning": _performance_reasoning(performance, theme, regulation, autonomy),
            },
        }
        return MegaPersona.model_validate(data)


THEMES = (
    {
        "setting": "a small repair workshop and a rotating set of household responsibilities",
        "resource": "practical feedback and visible examples",
        "constraint": "little uninterrupted time",
        "secondary_drive": "being useful when something concrete breaks",
        "value": "usefulness",
        "identity": "practical problem solver",
        "domain": "hands-on repair",
        "protective": "hands-on competence",
        "risk": "fragmented time",
    },
    {
        "setting": "debate meetings, argument notes, and a friend group that cares about fairness",
        "resource": "language for explaining conflict",
        "constraint": "pressure to always have a position",
        "secondary_drive": "being able to defend a view without losing curiosity",
        "value": "fairness",
        "identity": "careful advocate",
        "domain": "dialogue and argument",
        "protective": "a mentor who asks clarifying questions",
        "risk": "public comparison",
    },
    {
        "setting": "family shop routines, customer stories, and evening accounting",
        "resource": "early responsibility and real consequences",
        "constraint": "fatigue from practical obligations",
        "secondary_drive": "making daily systems run more smoothly",
        "value": "responsibility",
        "identity": "quiet organizer",
        "domain": "everyday logistics",
        "protective": "predictable routines",
        "risk": "caregiving spillover",
    },
    {
        "setting": "music files, remix drafts, and late-night attempts to make a sound feel right",
        "resource": "a private creative outlet",
        "constraint": "irregular sleep and uneven feedback",
        "secondary_drive": "finding a personal signature in borrowed material",
        "value": "originality",
        "identity": "iterative maker",
        "domain": "creative production",
        "protective": "private creative practice",
        "risk": "sleep disruption",
    },
)

REGION_TYPES = ("urban", "suburban", "rural", "migrant")
LEARNING_MODES = ("practice", "discussion", "visual", "project_based", "trial_and_error")


def _cognition_narrative(
    theme: dict[str, str],
    label: str,
    abstraction: float,
    autonomy: float,
    regulation: float,
    primary_drive: str,
    adaptive_constraints: list[str],
) -> str:
    constraints_text = " ".join(adaptive_constraints) if adaptive_constraints else (
        "No single trait explains the whole profile."
    )
    return (
        f"This {label} profile thinks through {theme['domain']} as a lived problem rather than "
        "a decorative interest. Their abstraction level shapes whether they begin with principles "
        "or concrete examples, but they usually need evidence that can be tested in action. "
        f"Autonomy is {autonomy:.2f}, so the person is most motivated when the task feels partly "
        f"self-authored; the primary drive is {primary_drive}, which gives effort an emotional "
        "direction. Self-regulation is neither magic nor destiny: when it is strong, they can turn "
        "pressure into sequence; when it is weak, they rely more on context, deadlines, and trusted "
        "people. Their blind spot is repeating a familiar strategy after the situation has changed. "
        f"The target abstraction value is {abstraction:.2f} and regulation is {regulation:.2f}, "
        "but these numbers appear as mechanisms: attention, pacing, help-seeking, and recovery. "
        f"{constraints_text}"
    )


def _performance_reasoning(
    performance: str,
    theme: dict[str, str],
    regulation: float,
    autonomy: float,
) -> str:
    if performance == "high":
        return (
            "High performance is plausible because strategy, practice, feedback, support, and "
            f"motivation around {theme['domain']} create repeatable habits rather than mere talent."
        )
    if performance == "low":
        return (
            "Low performance is plausible because context, uneven routines, limited recovery time, "
            f"and weaker planning around {theme['domain']} interrupt strategy despite real strengths."
        )
    if performance == "poor":
        return (
            "Poor performance is plausible in this setting because pressure, resource limits, and "
            "fragile routines block consistent practice; it is not a fixed ability or moral deficit."
        )
    return (
        f"Mid performance is plausible because regulation ({regulation:.2f}) and autonomy "
        f"({autonomy:.2f}) support some strategy, while stress and practical constraints still "
        "make outcomes uneven across tasks."
    )


def _age_for_stage(stage: str, index: int) -> int:
    ranges = {
        "middle_school": (12, 15),
        "high_school": (15, 18),
        "vocational": (16, 20),
        "undergraduate": (18, 23),
        "graduate": (22, 28),
        "early_career": (22, 30),
    }
    low, high = ranges.get(stage, (15, 22))
    return low + (index % (high - low + 1))


def _stress_value(band: str, regulation: float) -> float:
    base = {
        "low": 0.2,
        "low_mid": 0.38,
        "mid": 0.52,
        "mid_high": 0.68,
        "high": 0.8,
    }.get(band, 0.52)
    return _clip(base + 0.1 * (1.0 - regulation))


def _social_energy_value(band: str, index: int) -> float:
    base = {
        "low": 0.22,
        "low_mid": 0.38,
        "mid": 0.52,
        "mid_high": 0.68,
        "high": 0.82,
    }.get(band, 0.52)
    return _clip(base + ((index % 3) - 1) * 0.04)


def _thinking_mode(abstraction: float, index: int) -> str:
    if abstraction >= 0.72:
        return _pick(("analytical", "reflective", "exploratory"), index)
    if abstraction <= 0.32:
        return _pick(("practical", "intuitive", "associative"), index)
    return _pick(("practical", "reflective", "analytical", "associative"), index)


def _evidence_preference(abstraction: float, autonomy: float, index: int) -> str:
    if abstraction >= 0.7:
        return _pick(("data", "personal_values", "peer_consensus"), index)
    if autonomy >= 0.7:
        return _pick(("experience", "personal_values", "data"), index)
    return _pick(("experience", "authority", "peer_consensus"), index)


def _reward_preference(primary_drive: str, autonomy: float) -> str:
    if primary_drive == "recognition":
        return "praise"
    if primary_drive == "achievement":
        return "grades"
    if primary_drive == "autonomy" or autonomy >= 0.72:
        return "independence"
    if primary_drive == "belonging":
        return "praise"
    return "progress"


def _goal_orientation(primary_drive: str, regulation: float) -> str:
    if primary_drive == "avoidance":
        return "avoidance"
    if primary_drive in {"achievement", "recognition"} and regulation >= 0.5:
        return "performance"
    if primary_drive in {"mastery", "curiosity", "autonomy"}:
        return "mastery"
    return "mixed"


def _attention_pattern(regulation: float) -> str:
    if regulation >= 0.75:
        return "sustained"
    if regulation >= 0.45:
        return "bursty"
    return "easily_shifted"


def _curiosity_scope(abstraction: float, autonomy: float) -> str:
    if abstraction >= 0.7 and autonomy >= 0.55:
        return "narrow_deep"
    if autonomy >= 0.7:
        return "broad_shallow"
    if autonomy <= 0.25:
        return "low"
    return "situational"


def _help_seeking_style(regulation: float, social_energy: float) -> str:
    if regulation >= 0.7:
        return "proactive"
    if social_energy >= 0.65:
        return "peer_first"
    if regulation <= 0.3:
        return "reluctant"
    return "adult_first"


def _planning_style(regulation: float) -> str:
    if regulation >= 0.78:
        return "structured"
    if regulation >= 0.55:
        return "deadline_driven"
    if regulation >= 0.35:
        return "reactive"
    return "chaotic"


def _difficulty_response(regulation: float, social_energy: float) -> str:
    if regulation >= 0.7:
        return "reframes"
    if social_energy >= 0.6:
        return "seeks_help"
    if regulation <= 0.25:
        return "freezes"
    return "doubles_down"


def _criticism_response(autonomy: float, regulation: float) -> str:
    if autonomy >= 0.65 and regulation >= 0.5:
        return "curious"
    if regulation <= 0.3:
        return "ashamed"
    if autonomy <= 0.3:
        return "defensive"
    return "motivated"


def _success_response(autonomy: float, abstraction: float) -> str:
    if abstraction >= 0.7:
        return "exploratory"
    if autonomy >= 0.65:
        return "confident"
    return "anxious_to_maintain"


def _tradeoff_style(primary_drive: str, autonomy: float, regulation: float) -> str:
    if primary_drive == "belonging":
        return "maintain_relationships"
    if autonomy >= 0.7:
        return "seek_growth"
    if regulation <= 0.35:
        return "safe_choice"
    return "optimize_score"


def _core_values(primary_drive: str, theme_value: str) -> list[str]:
    values = [theme_value, "growth"]
    if primary_drive in {"autonomy", "curiosity"}:
        values.append("autonomy")
    elif primary_drive == "belonging":
        values.append("loyalty")
    elif primary_drive in {"security", "avoidance"}:
        values.append("stability")
    else:
        values.append("competence")
    return list(dict.fromkeys(values))


def _collaboration_style(social_energy: float, autonomy: float) -> str:
    if social_energy >= 0.75 and autonomy >= 0.55:
        return "leader"
    if social_energy >= 0.6:
        return "mediator"
    if autonomy >= 0.7:
        return "challenger"
    if social_energy <= 0.3:
        return "observer"
    return "supporter"


def _creative_mode(abstraction: float, autonomy: float, index: int) -> str:
    if abstraction >= 0.7 and autonomy >= 0.6:
        return "original"
    if abstraction >= 0.55:
        return "strategic"
    if autonomy >= 0.55:
        return "remixing"
    return _pick(("practical", "aesthetic", "low_expression"), index)


def _coping_style(regulation: float, autonomy: float) -> str:
    if regulation >= 0.7:
        return "problem_solving"
    if autonomy >= 0.65:
        return "control"
    if regulation <= 0.3:
        return "avoidance"
    return "emotional_support"


def _pick(options: tuple[str, ...], index: int) -> str:
    return options[(index - 1) % len(options)]


def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
