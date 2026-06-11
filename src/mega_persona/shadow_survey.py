"""Non-academic shadow surveys for MegaPersona experiments.

The item text here is original and construct-oriented. It is intended to
capture the role of HACHIMI-style shadow surveys without copying published
questionnaire items verbatim.
"""

from dataclasses import dataclass, field
from typing import Literal

from src.mega_persona.slots import AXIS_NAMES


LikertDirection = Literal[1, -1]


LIKERT_OPTIONS = {
    1: "strongly_disagree",
    2: "disagree",
    3: "neutral",
    4: "agree",
    5: "strongly_agree",
}


@dataclass(frozen=True)
class ShadowSurveyItem:
    item_id: str
    construct: str
    text: str
    direction: LikertDirection = 1
    axis_weights: dict[str, float] = field(default_factory=dict)

    def score(self, response: int) -> float:
        if response not in LIKERT_OPTIONS:
            raise ValueError(f"response must be 1-5, got {response}")
        normalized = (response - 1) / 4.0
        if self.direction == -1:
            return 1.0 - normalized
        return normalized


@dataclass(frozen=True)
class ShadowSurvey:
    survey_id: str
    context: str
    items: tuple[ShadowSurveyItem, ...]

    def item_ids(self) -> list[str]:
        return [item.item_id for item in self.items]


ITEM_BANK = (
    ShadowSurveyItem(
        item_id="cm_abs_01",
        construct="cognitive_abstraction",
        text="I try to find the general principle behind a specific event.",
        axis_weights={"cognitive_abstraction": 1.0},
    ),
    ShadowSurveyItem(
        item_id="cm_abs_02",
        construct="cognitive_abstraction",
        text="I prefer examples that show exactly what to do, even if the bigger idea is unclear.",
        direction=-1,
        axis_weights={"cognitive_abstraction": 0.8},
    ),
    ShadowSurveyItem(
        item_id="cm_amb_01",
        construct="ambiguity_tolerance",
        text="When a situation is unclear, I can keep thinking without needing an immediate answer.",
        axis_weights={"cognitive_abstraction": 0.4},
    ),
    ShadowSurveyItem(
        item_id="cm_amb_02",
        construct="ambiguity_tolerance",
        text="I usually stop engaging when there is no single correct path.",
        direction=-1,
        axis_weights={"cognitive_abstraction": 0.3},
    ),
    ShadowSurveyItem(
        item_id="mot_auto_01",
        construct="motivation_autonomy",
        text="I work harder when I can choose how to approach the task.",
        axis_weights={"motivation_autonomy": 1.0},
    ),
    ShadowSurveyItem(
        item_id="mot_auto_02",
        construct="motivation_autonomy",
        text="Without outside pressure, I rarely keep moving on important goals.",
        direction=-1,
        axis_weights={"motivation_autonomy": 0.9},
    ),
    ShadowSurveyItem(
        item_id="mot_intr_01",
        construct="intrinsic_motivation",
        text="I often continue exploring a topic after the required work is done.",
        axis_weights={"motivation_autonomy": 0.8, "cognitive_abstraction": 0.2},
    ),
    ShadowSurveyItem(
        item_id="mot_press_01",
        construct="external_pressure",
        text="Praise, ranking, or visible approval strongly changes how much effort I give.",
        direction=-1,
        axis_weights={"motivation_autonomy": 0.8},
    ),
    ShadowSurveyItem(
        item_id="reg_plan_01",
        construct="self_regulation",
        text="I break vague goals into small actions before I start.",
        axis_weights={"self_regulation_resilience": 0.9},
    ),
    ShadowSurveyItem(
        item_id="reg_plan_02",
        construct="self_regulation",
        text="I often wait until pressure becomes urgent before I organize my work.",
        direction=-1,
        axis_weights={"self_regulation_resilience": 0.8},
    ),
    ShadowSurveyItem(
        item_id="reg_meta_01",
        construct="metacognition",
        text="After something goes badly, I can usually name what I would change next time.",
        axis_weights={"self_regulation_resilience": 0.7},
    ),
    ShadowSurveyItem(
        item_id="reg_emotion_01",
        construct="emotional_regulation",
        text="Strong feelings may slow me down, but they do not completely decide my next action.",
        axis_weights={"self_regulation_resilience": 0.8},
    ),
    ShadowSurveyItem(
        item_id="well_stress_01",
        construct="stress_load",
        text="Small problems pile up quickly and make normal tasks feel heavy.",
        direction=-1,
        axis_weights={"self_regulation_resilience": 0.5},
    ),
    ShadowSurveyItem(
        item_id="well_recover_01",
        construct="resilience",
        text="After a setback, I can usually recover enough to try a different route.",
        axis_weights={"self_regulation_resilience": 1.0},
    ),
    ShadowSurveyItem(
        item_id="val_identity_01",
        construct="identity_clarity",
        text="I have a stable sense of what kind of person I am trying to become.",
        axis_weights={"motivation_autonomy": 0.4, "self_regulation_resilience": 0.3},
    ),
    ShadowSurveyItem(
        item_id="val_tension_01",
        construct="value_tension",
        text="I often feel pulled between what matters to me and what others expect.",
        direction=-1,
        axis_weights={"motivation_autonomy": 0.4},
    ),
    ShadowSurveyItem(
        item_id="soc_belong_01",
        construct="belonging",
        text="I feel more capable when I know at least one person in the group understands me.",
        axis_weights={"self_regulation_resilience": 0.2},
    ),
    ShadowSurveyItem(
        item_id="soc_peer_01",
        construct="peer_influence",
        text="The mood of people around me can quickly change what I decide to do.",
        direction=-1,
        axis_weights={"motivation_autonomy": 0.3},
    ),
    ShadowSurveyItem(
        item_id="cre_original_01",
        construct="creative_orientation",
        text="I enjoy recombining familiar ideas into something that feels personally useful.",
        axis_weights={"cognitive_abstraction": 0.3, "motivation_autonomy": 0.2},
    ),
    ShadowSurveyItem(
        item_id="risk_01",
        construct="risk_appetite",
        text="I am willing to test an uncertain option when the possible learning is valuable.",
        axis_weights={"motivation_autonomy": 0.3, "cognitive_abstraction": 0.2},
    ),
    ShadowSurveyItem(
        item_id="help_01",
        construct="help_seeking",
        text="When stuck, I can ask for help without feeling that the task no longer belongs to me.",
        axis_weights={"motivation_autonomy": 0.2, "self_regulation_resilience": 0.5},
    ),
    ShadowSurveyItem(
        item_id="goal_mastery_01",
        construct="mastery_orientation",
        text="Improving my method matters to me even when the result is not immediately visible.",
        axis_weights={
            "cognitive_abstraction": 0.3,
            "motivation_autonomy": 0.4,
            "self_regulation_resilience": 0.3,
        },
    ),
    ShadowSurveyItem(
        item_id="goal_avoid_01",
        construct="avoidance_orientation",
        text="I choose easier tasks mainly to avoid exposing what I cannot do yet.",
        direction=-1,
        axis_weights={"motivation_autonomy": 0.4, "self_regulation_resilience": 0.4},
    ),
    ShadowSurveyItem(
        item_id="future_01",
        construct="future_orientation",
        text="My current choices are often guided by the kind of future I want to make possible.",
        axis_weights={"motivation_autonomy": 0.4, "self_regulation_resilience": 0.4},
    ),
)


SURVEY_CONTEXTS = (
    "A new project begins with unclear expectations.",
    "A group task requires coordination with unfamiliar peers.",
    "A person receives criticism after trying seriously.",
    "A deadline is approaching while other demands compete for attention.",
    "An opportunity appears that is interesting but not required.",
    "A person must choose between approval and personal conviction.",
    "A stressful week makes ordinary routines harder to maintain.",
    "A peer group starts treating effort as uncool or embarrassing.",
    "A task has no single correct method and rewards exploration.",
    "A person succeeds once and must decide what to do next.",
    "A conflict emerges between short-term comfort and long-term growth.",
    "A person has to ask for help without losing ownership of the work.",
)


def build_initial_shadow_surveys(
    num_surveys: int = 12,
    items_per_survey: int = 12,
    seed: int = 17,
) -> list[ShadowSurvey]:
    """Build deterministic initial shadow surveys from the local item bank."""
    if num_surveys <= 0:
        return []
    if items_per_survey <= 0:
        raise ValueError("items_per_survey must be positive")

    surveys: list[ShadowSurvey] = []
    bank = list(ITEM_BANK)
    for survey_idx in range(num_surveys):
        start = (survey_idx * 5 + seed) % len(bank)
        chosen = [bank[(start + offset) % len(bank)] for offset in range(items_per_survey)]

        # Ensure every survey touches the three primary axes.
        missing_axes = set(AXIS_NAMES)
        for item in chosen:
            missing_axes -= set(item.axis_weights)
        for replacement_idx, axis in enumerate(sorted(missing_axes), start=1):
            replacement = next(item for item in bank if axis in item.axis_weights)
            chosen[-replacement_idx] = replacement

        surveys.append(
            ShadowSurvey(
                survey_id=f"shadow_{survey_idx + 1:02d}",
                context=SURVEY_CONTEXTS[survey_idx % len(SURVEY_CONTEXTS)],
                items=tuple(chosen),
            )
        )
    return surveys


def score_shadow_survey(
    survey: ShadowSurvey,
    responses: dict[str, int],
) -> dict[str, float]:
    """Score Likert responses into construct scores and projected primary axes."""
    construct_values: dict[str, list[float]] = {}
    axis_weighted_sum = {axis: 0.0 for axis in AXIS_NAMES}
    axis_weight_sum = {axis: 0.0 for axis in AXIS_NAMES}

    for item in survey.items:
        if item.item_id not in responses:
            continue
        item_score = item.score(responses[item.item_id])
        construct_values.setdefault(item.construct, []).append(item_score)
        for axis, weight in item.axis_weights.items():
            axis_weighted_sum[axis] += item_score * weight
            axis_weight_sum[axis] += abs(weight)

    scores = {
        f"construct.{construct}": sum(values) / len(values)
        for construct, values in construct_values.items()
    }
    for axis in AXIS_NAMES:
        if axis_weight_sum[axis] == 0:
            scores[f"axis.{axis}"] = 0.5
        else:
            scores[f"axis.{axis}"] = axis_weighted_sum[axis] / axis_weight_sum[axis]
    return scores
