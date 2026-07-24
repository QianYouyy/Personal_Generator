"""End-to-end experiment runner for MegaPersona-Evolve MVP."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.evaluator.metrics import DiversityMetrics
from src.mega_persona.consistency import (
    PopulationConsistencyEvaluation,
    evaluate_population_consistency,
)
from src.mega_persona.evaluation import MegaPersonaEvaluation, evaluate_mega_personas
from src.mega_persona.generator import MegaPersonaGenerator
from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_simulator import (
    LLMShadowSimulator,
    ShadowBehaviorReport,
    ShadowSurveySimulation,
    aggregate_shadow_behavior,
    shadow_behavior_axis_matrix,
)
from src.mega_persona.shadow_survey import ShadowSurvey, build_initial_shadow_surveys
from src.mega_persona.slots import (
    AXIS_NAMES,
    MegaPersonaSlot,
    SlotSampler,
    axis_names_for_binding,
    axis_roles_for_binding,
    default_schema_binding,
    quota_buckets_for_binding,
)
from src.mega_persona.template_generator import RuleBasedMegaPersonaBuilder


ExperimentMode = Literal["mock", "llm"]


@dataclass(frozen=True)
class MegaPersonaExperimentConfig:
    n: int = 25
    seeds: tuple[int, ...] = (17,)
    mode: ExperimentMode = "mock"
    num_shadow_surveys: int = 12
    items_per_shadow_survey: int = 12
    coverage_radius: float = 0.28
    duplicate_threshold: float = 0.82
    persona_max_workers: int = 1
    shadow_max_workers: int = 1


@dataclass
class MegaPersonaExperimentRun:
    mode: ExperimentMode
    seed: int
    created_at: str
    slots: list[MegaPersonaSlot]
    personas: list[MegaPersona]
    schema_evaluation: MegaPersonaEvaluation
    consistency_evaluation: PopulationConsistencyEvaluation
    shadow_behavior: ShadowBehaviorReport
    slot_diversity_metrics: dict[str, float]
    behavior_diversity_metrics: dict[str, float]
    experiment_score: float
    shadow_simulations: list[ShadowSurveySimulation] = field(default_factory=list)

    def to_dict(self, include_personas: bool = True) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "seed": self.seed,
            "created_at": self.created_at,
            "n_requested": len(self.slots),
            "n_valid_personas": len(self.personas),
            "experiment_score": self.experiment_score,
            "slot_diversity_metrics": self.slot_diversity_metrics,
            "schema_evaluation": self.schema_evaluation.to_dict(),
            "internal_consistency": self.consistency_evaluation.to_dict(),
            "shadow_behavior": self.shadow_behavior.to_dict(),
            "behavior_diversity_metrics": self.behavior_diversity_metrics,
            "slots": [asdict(slot) for slot in self.slots],
            "shadow_simulations": [asdict(simulation) for simulation in self.shadow_simulations],
        }
        if include_personas:
            payload["personas"] = [persona.model_dump() for persona in self.personas]
        return payload


@dataclass
class MegaPersonaExperimentSummary:
    config: MegaPersonaExperimentConfig
    runs: list[MegaPersonaExperimentRun]

    def to_dict(self, include_personas: bool = False) -> dict[str, Any]:
        return {
            "config": {
                **asdict(self.config),
                "seeds": list(self.config.seeds),
            },
            "aggregate": self.aggregate_metrics(),
            "runs": [
                run.to_dict(include_personas=include_personas)
                for run in self.runs
            ],
        }

    def aggregate_metrics(self) -> dict[str, float]:
        if not self.runs:
            return {}
        metric_names = {
            "experiment_score": [run.experiment_score for run in self.runs],
            "schema_fitness": [run.schema_evaluation.fitness for run in self.runs],
            "validity_rate": [run.schema_evaluation.validity_rate for run in self.runs],
            "near_duplicate_rate": [
                run.schema_evaluation.near_duplicate_rate for run in self.runs
            ],
            "internal_consistency": [
                run.consistency_evaluation.mean_score for run in self.runs
            ],
            "internal_consistency_min": [
                run.consistency_evaluation.min_score for run in self.runs
            ],
            "axis_alignment": [
                run.consistency_evaluation.axis_alignment_mean for run in self.runs
            ],
            "shadow_alignment": [
                run.shadow_behavior.overall_alignment for run in self.runs
            ],
            "behavior_coverage": [
                run.behavior_diversity_metrics.get("coverage", 0.0)
                for run in self.runs
            ],
            "slot_coverage": [
                run.slot_diversity_metrics.get("coverage", 0.0)
                for run in self.runs
            ],
        }
        aggregate: dict[str, float] = {}
        for name, values in metric_names.items():
            aggregate[f"{name}.mean"] = float(np.mean(values))
            aggregate[f"{name}.std"] = float(np.std(values))
        return aggregate

    def to_markdown(self) -> str:
        aggregate = self.aggregate_metrics()
        lines = [
            "# MegaPersona Experiment Summary",
            "",
            f"- mode: `{self.config.mode}`",
            f"- n per run: `{self.config.n}`",
            f"- seeds: `{', '.join(str(seed) for seed in self.config.seeds)}`",
            f"- shadow surveys: `{self.config.num_shadow_surveys}`",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Mean | Std |",
            "|---|---:|---:|",
        ]
        for metric in [
            "experiment_score",
            "schema_fitness",
            "validity_rate",
            "near_duplicate_rate",
            "internal_consistency",
            "internal_consistency_min",
            "axis_alignment",
            "shadow_alignment",
            "behavior_coverage",
            "slot_coverage",
        ]:
            lines.append(
                f"| {metric} | {aggregate.get(metric + '.mean', 0.0):.4f} | "
                f"{aggregate.get(metric + '.std', 0.0):.4f} |"
            )

        lines.extend(
            [
                "",
                "## Runs",
                "",
                "| Seed | Score | Valid | Schema Fit | Consistency | Shadow Align | Behavior Cov | Slot Cov |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run in self.runs:
            lines.append(
                f"| {run.seed} | {run.experiment_score:.4f} | "
                f"{run.schema_evaluation.valid_count}/{run.schema_evaluation.sample_size} | "
                f"{run.schema_evaluation.fitness:.4f} | "
                f"{run.consistency_evaluation.mean_score:.4f} | "
                f"{run.shadow_behavior.overall_alignment:.4f} | "
                f"{run.behavior_diversity_metrics.get('coverage', 0.0):.4f} | "
                f"{run.slot_diversity_metrics.get('coverage', 0.0):.4f} |"
            )
        return "\n".join(lines) + "\n"


class MegaPersonaExperimentRunner:
    """Run repeatable MegaPersona experiments across seeds."""

    def __init__(
        self,
        config: MegaPersonaExperimentConfig,
        llm_client=None,
        simulator_llm_client=None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.simulator_llm_client = simulator_llm_client
        if config.mode == "llm" and llm_client is None:
            raise ValueError("llm_client is required when mode='llm'")
        if simulator_llm_client is None:
            raise ValueError("simulator_llm_client is required (LLMShadowSimulator is the only simulator)")

    def run(self) -> MegaPersonaExperimentSummary:
        return MegaPersonaExperimentSummary(
            config=self.config,
            runs=[self.run_seed(seed) for seed in self.config.seeds],
        )

    def run_seed(self, seed: int) -> MegaPersonaExperimentRun:
        schema_binding = default_schema_binding()
        axis_names = axis_names_for_binding(schema_binding)
        axis_roles = axis_roles_for_binding(schema_binding)
        slots = SlotSampler(
            quota_buckets=quota_buckets_for_binding(schema_binding),
            axis_names=axis_names,
        ).sample(n=self.config.n, seed=seed)
        shadow_surveys = build_initial_shadow_surveys(
            num_surveys=self.config.num_shadow_surveys,
            items_per_survey=self.config.items_per_shadow_survey,
            seed=seed,
            schema_binding=schema_binding,
        )
        personas = self._generate_personas(slots)

        schema_evaluation = evaluate_mega_personas(
            personas,
            coverage_radius=self.config.coverage_radius,
            duplicate_threshold=self.config.duplicate_threshold,
            axis_names=axis_names,
            axis_roles=axis_roles,
        )
        consistency_evaluation = evaluate_population_consistency(
            personas,
            slots,
            axis_names=axis_names,
            axis_roles=axis_roles,
        )
        shadow_simulations = LLMShadowSimulator(
            self.simulator_llm_client,
            max_workers=self.config.shadow_max_workers,
        ).simulate_population(personas, shadow_surveys)
        shadow_behavior = aggregate_shadow_behavior(
            personas,
            shadow_simulations,
            axis_names=axis_names,
            axis_roles=axis_roles,
        )
        slot_diversity = _diversity_for_matrix(
            np.array([slot.axis_vector(axis_names) for slot in slots], dtype=float),
            self.config.coverage_radius,
        )
        behavior_diversity = _diversity_for_matrix(
            shadow_behavior_axis_matrix(personas, shadow_simulations, axis_names=axis_names),
            self.config.coverage_radius,
        )
        experiment_score = _experiment_score(
            schema_fitness=schema_evaluation.fitness,
            internal_consistency=consistency_evaluation.mean_score,
            behavior_coverage=behavior_diversity.get("coverage", 0.0),
            shadow_alignment=shadow_behavior.overall_alignment,
            generation_rate=len(personas) / len(slots) if slots else 0.0,
        )

        return MegaPersonaExperimentRun(
            mode=self.config.mode,
            seed=seed,
            created_at=datetime.now().isoformat(),
            slots=slots,
            personas=personas,
            schema_evaluation=schema_evaluation,
            consistency_evaluation=consistency_evaluation,
            shadow_behavior=shadow_behavior,
            slot_diversity_metrics=slot_diversity,
            behavior_diversity_metrics=behavior_diversity,
            experiment_score=experiment_score,
            shadow_simulations=shadow_simulations,
        )

    def _generate_personas(self, slots: list[MegaPersonaSlot]) -> list[MegaPersona]:
        if self.config.mode == "mock":
            return RuleBasedMegaPersonaBuilder().build_population(slots)

        generator = MegaPersonaGenerator(self.llm_client)
        results = generator.generate_from_slots(
            slots,
            max_workers=self.config.persona_max_workers,
        )
        return [result.persona for result in results if result.persona is not None]


def write_experiment_artifacts(
    summary: MegaPersonaExperimentSummary,
    output_dir: Path,
    include_personas: bool = True,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"

    import json

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary.to_dict(include_personas=include_personas), file, ensure_ascii=False, indent=2)
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(summary.to_markdown())
    return json_path, markdown_path


def _diversity_for_matrix(matrix: np.ndarray, coverage_radius: float) -> dict[str, float]:
    if len(matrix) == 0:
        return {
            "coverage": 0.0,
            "convex_hull": 0.0,
            "avg_dist": 0.0,
            "min_dist": 0.0,
            "dispersion": 0.0,
            "kl_divergence": 0.0,
        }
    return DiversityMetrics(coverage_radius=coverage_radius).fitness(matrix)


def compute_experiment_score(
    schema_fitness: float,
    behavior_coverage: float,
    shadow_alignment: float,
    generation_rate: float,
    internal_consistency: float = 1.0,
) -> float:
    """Multiplicative gated score used by both batch experiments and evolution.

    The gated product design ensures that invalid, near-duplicate, or
    behaviorally collapsed populations cannot score highly even if one
    dimension looks strong.  Each gate floors at 0.5 so the landscape
    retains gradient for evolution.
    """
    behavior_gate = 0.5 + 0.5 * float(np.clip(behavior_coverage, 0.0, 1.0))
    alignment_gate = 0.5 + 0.5 * float(np.clip(shadow_alignment, 0.0, 1.0))
    consistency_gate = 0.5 + 0.5 * float(np.clip(internal_consistency, 0.0, 1.0))
    return float(
        schema_fitness
        * consistency_gate
        * behavior_gate
        * alignment_gate
        * generation_rate
    )


# retained for backward-compatibility within this module
_experiment_score = compute_experiment_score
