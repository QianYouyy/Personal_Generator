"""Scientific-scale shadow surveys for MegaPersona experiments.

The HACHIMI paper evaluates personas with CEPS and PISA 2022 shadow surveys.
This module follows that scientific-measurement structure: every local item is
tagged with a CEPS/PISA construct family and scale ID. Item wording is
construct-faithful but original, so the code can run without redistributing
copyrighted questionnaire text.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
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
    instrument: str = "PISA2022"
    scale_id: str = ""
    scale_name: str = ""
    source_note: str = "construct_proxy_not_verbatim"

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
    split: str = "unspecified"
    source_protocol: str = "HACHIMI-style CEPS/PISA shadow survey"

    def item_ids(self) -> list[str]:
        return [item.item_id for item in self.items]


@dataclass(frozen=True)
class ShadowSurveySplit:
    train: tuple[ShadowSurvey, ...]
    validation: tuple[ShadowSurvey, ...]
    test: tuple[ShadowSurvey, ...]

    def to_dict(self) -> dict[str, list[dict]]:
        return {
            "train": [asdict(survey) for survey in self.train],
            "validation": [asdict(survey) for survey in self.validation],
            "test": [asdict(survey) for survey in self.test],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShadowSurveySplit":
        return cls(
            train=tuple(_survey_from_dict(item) for item in data.get("train", [])),
            validation=tuple(_survey_from_dict(item) for item in data.get("validation", [])),
            test=tuple(_survey_from_dict(item) for item in data.get("test", [])),
        )


SCIENTIFIC_SCALE_REGISTRY = {
    "PISA2022:CURIOAGR": "Curiosity agreement",
    "PISA2022:GROSAGR": "Growth mindset agreement",
    "PISA2022:CREATEFF": "Creative self-efficacy",
    "PISA2022:CREATOP": "Creativity and openness to intellect",
    "PISA2022:RELATST": "Student-teacher relationship quality",
    "PISA2022:BELONG": "Sense of belonging at school",
    "PISA2022:BULLIED": "Bullying exposure",
    "PISA2022:PSYCHSYM": "Psychosomatic symptoms / psychological distress",
    "PISA2022:LIFESAT": "Life satisfaction across domains",
    "PISA2022:WORKHOME": "Work-home balance / school-life pressure",
    "CEPS:CESD": "Depressive symptoms scale embedded in CEPS",
    "CEPS:TEACHREL": "Teacher-student relationship and classroom support",
    "CEPS:PEERREL": "Peer relations and school belonging",
    "CEPS:MISBEHAVIOR": "School misbehavior / behavioral self-regulation",
}


ITEM_BANK = (
    ShadowSurveyItem(
        item_id="pisa_curio_01",
        construct="curiosity",
        text="When I encounter something unfamiliar, I want to understand why it works.",
        axis_weights={"cognitive_abstraction": 1.0},
        scale_id="CURIOAGR",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:CURIOAGR"],
    ),
    ShadowSurveyItem(
        item_id="pisa_curio_02",
        construct="curiosity",
        text="I lose interest quickly when a topic does not have an obvious answer.",
        direction=-1,
        axis_weights={"cognitive_abstraction": 0.8},
        scale_id="CURIOAGR",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:CURIOAGR"],
    ),
    ShadowSurveyItem(
        item_id="pisa_creatop_01",
        construct="creativity_openness",
        text="I like trying more than one way to understand a complex situation.",
        axis_weights={"cognitive_abstraction": 0.6, "motivation_autonomy": 0.2},
        scale_id="CREATOP",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:CREATOP"],
    ),
    ShadowSurveyItem(
        item_id="pisa_creatop_02",
        construct="creativity_openness",
        text="I prefer to avoid tasks where the method is not already clearly given.",
        direction=-1,
        axis_weights={"cognitive_abstraction": 0.5},
        scale_id="CREATOP",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:CREATOP"],
    ),
    ShadowSurveyItem(
        item_id="pisa_gros_01",
        construct="growth_mindset",
        text="I can improve at difficult things when I change strategy and keep practicing.",
        axis_weights={"motivation_autonomy": 1.0},
        scale_id="GROSAGR",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:GROSAGR"],
    ),
    ShadowSurveyItem(
        item_id="pisa_gros_02",
        construct="growth_mindset",
        text="When I am not good at something at first, it usually means I should stop investing effort.",
        direction=-1,
        axis_weights={"motivation_autonomy": 0.9},
        scale_id="GROSAGR",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:GROSAGR"],
    ),
    ShadowSurveyItem(
        item_id="pisa_createff_01",
        construct="creative_self_efficacy",
        text="I can usually produce a useful idea when a familiar approach does not fit.",
        axis_weights={"motivation_autonomy": 0.4, "cognitive_abstraction": 0.4},
        scale_id="CREATEFF",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:CREATEFF"],
    ),
    ShadowSurveyItem(
        item_id="pisa_createff_02",
        construct="creative_self_efficacy",
        text="If there is no example to copy, I doubt I can make something worthwhile.",
        direction=-1,
        axis_weights={"motivation_autonomy": 0.4, "cognitive_abstraction": 0.3},
        scale_id="CREATEFF",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:CREATEFF"],
    ),
    ShadowSurveyItem(
        item_id="ceps_misbehavior_01",
        construct="behavioral_self_regulation",
        text="Even when nobody is checking, I can usually keep my behavior within agreed rules.",
        axis_weights={"self_regulation_resilience": 0.9},
        instrument="CEPS",
        scale_id="MISBEHAVIOR",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["CEPS:MISBEHAVIOR"],
    ),
    ShadowSurveyItem(
        item_id="ceps_misbehavior_02",
        construct="behavioral_self_regulation",
        text="When I feel restless, I often disrupt routines or agreements around me.",
        direction=-1,
        axis_weights={"self_regulation_resilience": 0.8},
        instrument="CEPS",
        scale_id="MISBEHAVIOR",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["CEPS:MISBEHAVIOR"],
    ),
    ShadowSurveyItem(
        item_id="pisa_workhome_01",
        construct="work_life_pressure",
        text="Competing demands often make it hard for me to protect time for recovery.",
        direction=-1,
        axis_weights={"self_regulation_resilience": 0.7},
        scale_id="WORKHOME",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:WORKHOME"],
    ),
    ShadowSurveyItem(
        item_id="pisa_workhome_02",
        construct="work_life_pressure",
        text="I can usually adjust my schedule before pressure overwhelms me.",
        axis_weights={"self_regulation_resilience": 0.7},
        scale_id="WORKHOME",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:WORKHOME"],
    ),
    ShadowSurveyItem(
        item_id="pisa_psych_01",
        construct="psychological_distress",
        text="Stress often shows up in my body or mood before I can solve the problem.",
        direction=-1,
        axis_weights={"self_regulation_resilience": 0.8},
        scale_id="PSYCHSYM",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:PSYCHSYM"],
    ),
    ShadowSurveyItem(
        item_id="ceps_cesd_01",
        construct="depressive_symptoms",
        text="After setbacks, low mood can make ordinary responsibilities feel much heavier.",
        direction=-1,
        axis_weights={"self_regulation_resilience": 0.8},
        instrument="CEPS",
        scale_id="CESD",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["CEPS:CESD"],
    ),
    ShadowSurveyItem(
        item_id="pisa_lifesat_01",
        construct="life_satisfaction",
        text="Most days, I can identify at least one part of life that feels meaningful or satisfying.",
        axis_weights={"motivation_autonomy": 0.3, "self_regulation_resilience": 0.3},
        scale_id="LIFESAT",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:LIFESAT"],
    ),
    ShadowSurveyItem(
        item_id="pisa_lifesat_02",
        construct="life_satisfaction",
        text="It is difficult for me to feel satisfied even when things are going acceptably.",
        direction=-1,
        axis_weights={"self_regulation_resilience": 0.4},
        scale_id="LIFESAT",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:LIFESAT"],
    ),
    ShadowSurveyItem(
        item_id="pisa_belong_01",
        construct="school_belonging",
        text="I feel more capable when I know at least one person in the group understands me.",
        axis_weights={"self_regulation_resilience": 0.3},
        scale_id="BELONG",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:BELONG"],
    ),
    ShadowSurveyItem(
        item_id="pisa_belong_02",
        construct="school_belonging",
        text="In group settings, I often feel like an outsider even when I am present.",
        direction=-1,
        axis_weights={"self_regulation_resilience": 0.3},
        scale_id="BELONG",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:BELONG"],
    ),
    ShadowSurveyItem(
        item_id="pisa_relatst_01",
        construct="teacher_relationship",
        text="Supportive adults make it easier for me to keep trying after difficulty.",
        axis_weights={"self_regulation_resilience": 0.3, "motivation_autonomy": 0.2},
        scale_id="RELATST",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:RELATST"],
    ),
    ShadowSurveyItem(
        item_id="ceps_teachrel_01",
        construct="teacher_relationship",
        text="When feedback is firm but fair, I can use it without feeling personally attacked.",
        axis_weights={"self_regulation_resilience": 0.4, "motivation_autonomy": 0.2},
        instrument="CEPS",
        scale_id="TEACHREL",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["CEPS:TEACHREL"],
    ),
    ShadowSurveyItem(
        item_id="pisa_bullied_01",
        construct="social_threat",
        text="Social pressure or exclusion can quickly change how safe I feel participating.",
        direction=-1,
        axis_weights={"motivation_autonomy": 0.2, "self_regulation_resilience": 0.5},
        scale_id="BULLIED",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:BULLIED"],
    ),
    ShadowSurveyItem(
        item_id="ceps_peerrel_01",
        construct="peer_relationship",
        text="Peer relationships usually help me stay connected rather than pull me away from my own goals.",
        axis_weights={
            "motivation_autonomy": 0.3,
            "self_regulation_resilience": 0.3,
        },
        instrument="CEPS",
        scale_id="PEERREL",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["CEPS:PEERREL"],
    ),
    ShadowSurveyItem(
        item_id="pisa_curio_gros_bridge_01",
        construct="curiosity_growth_bridge",
        text="When a challenge exposes what I do not know, I often become more curious rather than less.",
        axis_weights={"cognitive_abstraction": 0.4, "motivation_autonomy": 0.4},
        scale_id="CURIOAGR+GROSAGR",
        scale_name="Curiosity and growth mindset bridge",
    ),
    ShadowSurveyItem(
        item_id="pisa_psych_recover_01",
        construct="distress_recovery",
        text="Even when stress affects me, I can usually recover enough to choose a next step.",
        axis_weights={"self_regulation_resilience": 1.0},
        scale_id="PSYCHSYM",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:PSYCHSYM"],
    ),
    ShadowSurveyItem(
        item_id="pisa_belong_auto_01",
        construct="belonging_autonomy",
        text="Belonging to a group helps me act more like myself, not less.",
        axis_weights={"motivation_autonomy": 0.4, "self_regulation_resilience": 0.3},
        scale_id="BELONG",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["PISA2022:BELONG"],
    ),
    ShadowSurveyItem(
        item_id="ceps_cesd_reverse_01",
        construct="emotional_functioning",
        text="Low energy or sadness rarely interferes with my ability to follow through.",
        axis_weights={"self_regulation_resilience": 0.7},
        instrument="CEPS",
        scale_id="CESD",
        scale_name=SCIENTIFIC_SCALE_REGISTRY["CEPS:CESD"],
    ),
    ShadowSurveyItem(
        item_id="pisa_createff_growth_01",
        construct="creative_growth",
        text="I am willing to revise an idea several times until it becomes clearer.",
        axis_weights={"motivation_autonomy": 0.4, "self_regulation_resilience": 0.4},
        scale_id="CREATEFF+GROSAGR",
        scale_name="Creative self-efficacy and growth mindset bridge",
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
    split: str = "train",
    survey_id_prefix: str | None = None,
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

        prefix = survey_id_prefix or f"shadow_{split}"
        surveys.append(
            ShadowSurvey(
                survey_id=f"{prefix}_{survey_idx + 1:02d}",
                context=SURVEY_CONTEXTS[survey_idx % len(SURVEY_CONTEXTS)],
                items=tuple(chosen),
                split=split,
            )
        )
    return surveys


def build_shadow_survey_splits(
    train_surveys: int = 12,
    validation_surveys: int = 4,
    test_surveys: int = 4,
    items_per_survey: int = 12,
    seed: int = 17,
) -> ShadowSurveySplit:
    """Build fixed train/validation/test shadow survey splits.

    These splits are intentionally independent of candidate genomes. Evolution
    may optimize against train/validation behavior, but test should be reported
    only after candidate selection.
    """
    return ShadowSurveySplit(
        train=tuple(
            build_initial_shadow_surveys(
                num_surveys=train_surveys,
                items_per_survey=items_per_survey,
                seed=seed,
                split="train",
            )
        ),
        validation=tuple(
            build_initial_shadow_surveys(
                num_surveys=validation_surveys,
                items_per_survey=items_per_survey,
                seed=seed + 10000,
                split="validation",
            )
        ),
        test=tuple(
            build_initial_shadow_surveys(
                num_surveys=test_surveys,
                items_per_survey=items_per_survey,
                seed=seed + 20000,
                split="test",
            )
        ),
    )


def write_shadow_survey_splits(splits: ShadowSurveySplit, output_dir: Path) -> dict[str, str]:
    """Persist frozen survey splits and return stable content hashes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    payload = splits.to_dict()
    for split_name, surveys in payload.items():
        path = output_dir / f"{split_name}.json"
        _write_json(path, surveys)
        hashes[split_name] = _sha256_json(surveys)
    _write_json(output_dir / "hashes.json", hashes)
    return hashes


def read_shadow_survey_splits(input_dir: Path) -> ShadowSurveySplit:
    """Load previously frozen survey splits from disk."""
    payload = {}
    for split_name in ("train", "validation", "test"):
        path = input_dir / f"{split_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing frozen shadow survey split: {path}")
        payload[split_name] = _read_json(path)
    return ShadowSurveySplit.from_dict(payload)


def shadow_survey_split_hashes(splits: ShadowSurveySplit) -> dict[str, str]:
    payload = splits.to_dict()
    return {
        split_name: _sha256_json(surveys)
        for split_name, surveys in payload.items()
    }


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


def _survey_from_dict(data: dict[str, Any]) -> ShadowSurvey:
    return ShadowSurvey(
        survey_id=data["survey_id"],
        context=data["context"],
        items=tuple(_item_from_dict(item) for item in data.get("items", [])),
        split=data.get("split", "unspecified"),
        source_protocol=data.get(
            "source_protocol",
            "HACHIMI-style CEPS/PISA shadow survey",
        ),
    )


def _item_from_dict(data: dict[str, Any]) -> ShadowSurveyItem:
    return ShadowSurveyItem(
        item_id=data["item_id"],
        construct=data["construct"],
        text=data["text"],
        direction=data.get("direction", 1),
        axis_weights=dict(data.get("axis_weights", {})),
        instrument=data.get("instrument", "PISA2022"),
        scale_id=data.get("scale_id", ""),
        scale_name=data.get("scale_name", ""),
        source_note=data.get("source_note", "construct_proxy_not_verbatim"),
    )


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
