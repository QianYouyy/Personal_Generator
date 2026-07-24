"""MegaPersona genome utilities, evaluator backend, and artifact store.

The evolvable surface is deliberately narrow and JSON-serializable. The shared
OpenEvolve engine mutates these genomes through src.mega_persona.openevolve_adapter,
while this module keeps the fixed MegaPersona architecture, scientific scoring,
and durable evaluation artifacts.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.evaluator.metrics import DiversityMetrics
from src.mega_persona.consistency import evaluate_population_consistency
from src.mega_persona.evaluation import evaluate_mega_personas
from src.mega_persona.generator import MegaPersonaGenerationResult, MegaPersonaGenerator
from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_simulator import (
    ShadowSurveySimulation,
    aggregate_shadow_behavior,
    build_shadow_simulator,
    shadow_behavior_axis_matrix,
)
from src.mega_persona.shadow_survey import (
    ShadowSurvey,
    ShadowSurveySplit,
    build_shadow_survey_splits,
    read_shadow_survey_splits,
    shadow_survey_split_hashes,
    write_shadow_survey_splits,
)
from src.mega_persona.slots import (
    AXIS_NAMES,
    DEFAULT_QUOTA_BUCKETS,
    MegaPersonaSlot,
    QuotaBucket,
    SlotSampler,
    axis_names_for_binding,
    axis_roles_for_binding,
    build_adaptive_constraints,
    default_schema_binding,
    quota_buckets_for_binding,
    schema_binding_for_genome,
)
from src.mega_persona.template_generator import RuleBasedMegaPersonaBuilder


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MegaEvolutionConfig:
    n: int = 25
    seeds: tuple[int, ...] = (17, 23, 31)
    generator_mode: str = "mock"
    generations: int = 20
    population_size: int = 8
    children_per_generation: int = 6
    elite_count: int = 3
    coverage_radius: float = 0.28
    duplicate_threshold: float = 0.82
    shadow_surveys: int = 12
    validation_shadow_surveys: int = 4
    test_shadow_surveys: int = 4
    items_per_shadow_survey: int = 12
    survey_seed: int = 17
    random_seed: int = 1234
    max_workers: int = 1
    candidate_evaluation_repeats: int = 1
    elite_confirmation_repeats: int = 1
    shadow_max_workers: int = 1
    persona_max_workers: int = 2
    shadow_simulator_backend: str = "llm"
    persona_pipeline: str = "five_agent"
    persona_temperature: float = 0.45
    persona_top_p: float = 0.85
    simulator_temperature: float = 0.05
    simulator_top_p: float = 0.80


# Score aggregation uses the same multiplicative gated formula as the batch
# experiment (see src.mega_persona.experiment.compute_experiment_score).
# The weights below are retained as a reference for ablation studies but are
# NOT used in the default genome_score() computation.


@dataclass
class MegaEvolutionCandidate:
    candidate_id: str
    genome: dict[str, Any]
    generation: int = 0
    parent_id: str | None = None
    fitness: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    evaluated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MegaEvolutionCandidate":
        return cls(**data)


class MegaPersonaEvolver:
    """Evaluation and persistence backend for OpenEvolve MegaPersona genomes."""

    def __init__(
        self,
        config: MegaEvolutionConfig,
        output_dir: Path,
        resume: bool = False,
        llm_client=None,
        simulator_llm_client=None,
        initial_genome: dict[str, Any] | None = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.store = EvolutionStore(output_dir)
        self.rng = np.random.default_rng(config.random_seed)
        self.llm_client = llm_client
        self.simulator_llm_client = simulator_llm_client
        self.initial_genome = normalize_genome(initial_genome) if initial_genome is not None else default_genome()
        self.generation = 0
        self.population: list[MegaEvolutionCandidate] = []
        self.evaluation_count = 0
        self.best_candidate_id: str | None = None
        self.survey_splits: ShadowSurveySplit
        self.survey_hashes: dict[str, str]

        if config.generator_mode not in {"mock", "llm"}:
            raise ValueError("generator_mode must be 'mock' or 'llm'")
        if config.generator_mode == "llm" and llm_client is None:
            raise ValueError("llm_client is required when generator_mode='llm'")
        if simulator_llm_client is None:
            raise ValueError("simulator_llm_client is required for shadow survey simulation")

        if resume:
            self._load_checkpoint()
            self.survey_splits = read_shadow_survey_splits(self.store.surveys_dir)
            self.survey_hashes = shadow_survey_split_hashes(self.survey_splits)
            logger.info(
                "Resumed evolution from %s at generation=%s evaluations=%s",
                self.output_dir,
                self.generation,
                self.evaluation_count,
            )
        else:
            self.store.initialize()
            survey_binding = default_genome()["schema_binding"]
            self.survey_splits = build_shadow_survey_splits(
                train_surveys=self.config.shadow_surveys,
                validation_surveys=self.config.validation_shadow_surveys,
                test_surveys=self.config.test_shadow_surveys,
                items_per_survey=self.config.items_per_shadow_survey,
                seed=self.config.survey_seed,
                schema_binding=survey_binding,
            )
            self.survey_hashes = write_shadow_survey_splits(
                self.survey_splits,
                self.store.surveys_dir,
            )
            self.population = self._initial_population()
            self._save_checkpoint()
            logger.info(
                "Initialized evolution run at %s population=%s generations=%s seeds=%s",
                self.output_dir,
                len(self.population),
                self.config.generations,
                ",".join(str(seed) for seed in self.config.seeds),
            )
            logger.info("Frozen shadow survey hashes: %s", self.survey_hashes)

    def run(self) -> MegaEvolutionCandidate:
        raise RuntimeError(
            "MegaPersonaEvolver.run() has been retired. Use "
            "MegaPersonaOpenEvolveRunner or scripts/run_mega_persona_evolution.py, "
            "which run through src.open_evolve.engine.OpenEvolve."
        )

    def best_candidate(self) -> MegaEvolutionCandidate:
        evaluated = [candidate for candidate in self.population if candidate.evaluated]
        if not evaluated:
            raise RuntimeError("No evaluated candidates yet")
        return max(evaluated, key=lambda candidate: candidate.fitness or float("-inf"))

    def _initial_population(self) -> list[MegaEvolutionCandidate]:
        baseline = MegaEvolutionCandidate(
            candidate_id="candidate_baseline",
            genome=json.loads(json.dumps(self.initial_genome)),
            generation=0,
        )
        candidates = [baseline]
        while len(candidates) < self.config.population_size:
            candidates.append(
                self._mutate_candidate(
                    baseline,
                    generation=0,
                    mutation_scale=0.35,
                )
            )
        return candidates

    def _evaluate_population(self) -> None:
        pending = [candidate for candidate in self.population if not candidate.evaluated]
        if not pending:
            logger.info("Generation %s: no pending candidates", self.generation)
            return
        logger.info(
            "Generation %s: %s pending candidate(s), max_workers=%s",
            self.generation,
            len(pending),
            self.config.max_workers,
        )
        if self.config.max_workers <= 1:
            for candidate in pending:
                result = self._evaluate_candidate_safely(candidate)
                self._record_candidate_result(candidate, result)
            return

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_candidate_safely, candidate): candidate
                for candidate in pending
            }
            for future in as_completed(futures):
                candidate = futures[future]
                result = future.result()
                self._record_candidate_result(candidate, result)

    def _evaluate_candidate_safely(self, candidate: MegaEvolutionCandidate) -> dict[str, Any]:
        try:
            logger.info(
                "Evaluating candidate=%s generation=%s parent=%s",
                candidate.candidate_id,
                candidate.generation,
                candidate.parent_id,
            )
            return self.evaluate_candidate(candidate)
        except Exception as exc:
            logger.error(
                "Candidate evaluation failed: %s error=%s: %s",
                candidate.candidate_id,
                type(exc).__name__,
                exc,
            )
            return failed_evaluation_payload(candidate, exc)

    def _record_candidate_result(
        self,
        candidate: MegaEvolutionCandidate,
        result: dict[str, Any],
    ) -> None:
        candidate.fitness = result["fitness"]
        candidate.metrics = result["metrics"]
        candidate.evaluated = True
        self.evaluation_count += 1
        self.store.write_evaluation(
            evaluation_index=self.evaluation_count,
            candidate=candidate,
            payload=result,
        )
        current_best = self.best_candidate_id
        if current_best is None:
            self.best_candidate_id = candidate.candidate_id
        else:
            old = self._candidate_by_id(current_best)
            if old is None or (candidate.fitness or 0.0) > (old.fitness or 0.0):
                self.best_candidate_id = candidate.candidate_id
        self._save_checkpoint()
        logger.info(
            "Recorded candidate=%s eval=%s fitness=%.4f best=%s checkpoint=%s",
            candidate.candidate_id,
            self.evaluation_count,
            candidate.fitness or 0.0,
            self.best_candidate_id,
            self.store.checkpoint_path,
        )

    def evaluate_candidate(self, candidate: MegaEvolutionCandidate) -> dict[str, Any]:
        per_seed = []
        for seed in self.config.seeds:
            try:
                per_seed.append(self._evaluate_candidate_seed(candidate, seed))
            except Exception as exc:
                logger.error(
                    "Candidate=%s seed=%s failed (%s: %s)",
                    candidate.candidate_id,
                    seed,
                    type(exc).__name__,
                    exc,
                )
                per_seed.append(_failed_seed_result(seed, candidate.candidate_id, exc))

        fitness = float(np.mean([seed_result["score"] for seed_result in per_seed]))
        metrics = _aggregate_seed_metrics(per_seed)
        logger.info("Candidate=%s aggregate fitness=%.4f", candidate.candidate_id, fitness)
        return {
            "candidate": candidate.to_dict(),
            "fitness": fitness,
            "metrics": metrics,
            "per_seed": per_seed,
        }

    def _evaluate_candidate_seed(
        self,
        candidate: MegaEvolutionCandidate,
        seed: int,
    ) -> dict[str, Any]:
        logger.info("Candidate=%s seed=%s: generating slots/personas", candidate.candidate_id, seed)
        schema_binding = schema_binding_for_genome(candidate.genome)
        axis_names = axis_names_for_binding(schema_binding)
        axis_roles = axis_roles_for_binding(schema_binding)
        slots = candidate_slots(candidate.genome, n=self.config.n, seed=seed)
        generation_results = self._generate_persona_results(candidate, slots)
        personas = [result.persona for result in generation_results if result.persona is not None]
        if (
            not personas
            and self.config.generator_mode != "mock"
            and self.config.persona_max_workers > 1
        ):
            logger.warning(
                "Candidate=%s seed=%s: generated 0/%s personas; retrying once with persona_max_workers=1",
                candidate.candidate_id,
                seed,
                len(slots),
            )
            generation_results = self._generate_persona_results(
                candidate,
                slots,
                max_workers_override=1,
            )
            personas = [result.persona for result in generation_results if result.persona is not None]
        logger.info(
            "Candidate=%s seed=%s: generated_personas=%s/%s",
            candidate.candidate_id,
            seed,
            len(personas),
            len(slots),
        )
        if not personas:
            logger.error(
                "Candidate=%s seed=%s: no personas generated after retry; marking seed as failed",
                candidate.candidate_id,
                seed,
            )
            return _no_persona_seed_result(
                seed=seed,
                candidate_id=candidate.candidate_id,
                slots=slots,
                generation_results=generation_results,
                survey_hashes=self.survey_hashes,
            )
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
        simulator = build_shadow_simulator(
            backend=self.config.shadow_simulator_backend,
            llm_client=self.simulator_llm_client,
            max_workers=self.config.shadow_max_workers,
            temperature=self.config.simulator_temperature,
            top_p=self.config.simulator_top_p,
        )
        train_surveys = list(self.survey_splits.train)
        validation_surveys = list(self.survey_splits.validation)
        logger.info(
            "Candidate=%s seed=%s: simulating train+validation shadow surveys",
            candidate.candidate_id,
            seed,
        )
        shadow_simulations = simulator.simulate_population(
            personas,
            train_surveys + validation_surveys,
        )
        train_simulations = _filter_simulations_by_surveys(shadow_simulations, train_surveys)
        validation_simulations = _filter_simulations_by_surveys(
            shadow_simulations,
            validation_surveys,
        )
        train_shadow_behavior = aggregate_shadow_behavior(
            personas,
            train_simulations,
            axis_names=axis_names,
            axis_roles=axis_roles,
        )
        validation_shadow_behavior = aggregate_shadow_behavior(
            personas,
            validation_simulations,
            axis_names=axis_names,
            axis_roles=axis_roles,
        )
        train_behavior_diversity = _diversity_for_matrix(
            shadow_behavior_axis_matrix(personas, train_simulations, axis_names=axis_names),
            self.config.coverage_radius,
        )
        validation_behavior_diversity = _diversity_for_matrix(
            shadow_behavior_axis_matrix(personas, validation_simulations, axis_names=axis_names),
            self.config.coverage_radius,
        )
        slot_diversity = _diversity_for_matrix(
            np.array([slot.axis_vector(axis_names) for slot in slots], dtype=float),
            self.config.coverage_radius,
        )
        seed_score = genome_score(
            genome=candidate.genome,
            schema_fitness=schema_evaluation.fitness,
            internal_consistency=consistency_evaluation.mean_score,
            behavior_coverage=validation_behavior_diversity.get("coverage", 0.0),
            shadow_alignment=validation_shadow_behavior.overall_alignment,
            generation_rate=len(personas) / len(slots) if slots else 0.0,
        )
        logger.info(
            "Candidate=%s seed=%s: score=%.4f schema=%.4f consistency=%.4f val_cov=%.4f val_align=%.4f",
            candidate.candidate_id,
            seed,
            seed_score,
            schema_evaluation.fitness,
            consistency_evaluation.mean_score,
            validation_behavior_diversity.get("coverage", 0.0),
            validation_shadow_behavior.overall_alignment,
        )
        return {
            "seed": seed,
            "status": "ok",
            "score": seed_score,
            "slots": [asdict(slot) for slot in slots],
            "personas": [persona.model_dump() for persona in personas],
            "generation_diagnostics": [
                _generation_result_diagnostic(result) for result in generation_results
            ],
            "schema_evaluation": schema_evaluation.to_dict(),
            "internal_consistency": consistency_evaluation.to_dict(),
            "train_shadow_behavior": train_shadow_behavior.to_dict(),
            "validation_shadow_behavior": validation_shadow_behavior.to_dict(),
            "train_behavior_diversity": train_behavior_diversity,
            "validation_behavior_diversity": validation_behavior_diversity,
            "slot_diversity": slot_diversity,
            "shadow_survey_hashes": self.survey_hashes,
            "train_shadow_simulations": [
                asdict(simulation) for simulation in train_simulations
            ],
            "validation_shadow_simulations": [
                asdict(simulation) for simulation in validation_simulations
            ],
        }

    def evaluate_final_test(self, best: MegaEvolutionCandidate) -> dict[str, Any]:
        """Evaluate the selected candidate on test without using test for selection."""
        best_result = self.store.find_candidate_result(best.candidate_id)
        if best_result is None:
            raise FileNotFoundError(f"missing stored evaluation for best candidate {best.candidate_id}")

        repeat_groups = best_result.get("evaluation_repeat_per_seed")
        if not isinstance(repeat_groups, list) or not repeat_groups:
            repeat_groups = [best_result.get("per_seed", [])]
        successful_seed_results = [
            (repeat_index, seed_result)
            for repeat_index, repeat_group in enumerate(repeat_groups, start=1)
            if isinstance(repeat_group, list)
            for seed_result in repeat_group
            if isinstance(seed_result, dict)
            and seed_result.get("status") == "ok"
            and seed_result.get("personas")
        ]
        if (best.fitness or 0.0) <= 0.0 or not successful_seed_results:
            logger.info(
                "Skipping sealed test for candidate=%s because validation produced no positive successful seed",
                best.candidate_id,
            )
            return {
                "candidate_id": best.candidate_id,
                "candidate_fitness": best.fitness,
                "selection_metric": "validation",
                "test_used_for_selection": False,
                "status": "skipped_no_successful_validation_candidate",
                "survey_hashes": self.survey_hashes,
                "metrics": {},
                "per_seed": [],
                "created_at": datetime.now().isoformat(),
            }

        simulator = build_shadow_simulator(
            backend=self.config.shadow_simulator_backend,
            llm_client=self.simulator_llm_client,
            max_workers=self.config.shadow_max_workers,
            temperature=self.config.simulator_temperature,
            top_p=self.config.simulator_top_p,
        )
        per_seed = []
        for repeat_index, seed_result in successful_seed_results:
            logger.info(
                "Final sealed test candidate=%s repeat=%s/%s seed=%s",
                best.candidate_id,
                repeat_index,
                len(repeat_groups),
                seed_result["seed"],
            )
            personas = [
                MegaPersona.model_validate(persona)
                for persona in seed_result.get("personas", [])
            ]
            slots = [
                MegaPersonaSlot(**slot)
                for slot in seed_result.get("slots", [])
            ]
            if slots:
                axis_names = tuple(slots[0].target_axes.keys()) or AXIS_NAMES
            else:
                axis_names = AXIS_NAMES
            schema_binding = schema_binding_for_genome(best.genome)
            axis_roles = axis_roles_for_binding(schema_binding)
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
            test_simulations = simulator.simulate_population(personas, list(self.survey_splits.test))
            test_shadow_behavior = aggregate_shadow_behavior(
                personas,
                test_simulations,
                axis_names=axis_names,
                axis_roles=axis_roles,
            )
            test_behavior_diversity = _diversity_for_matrix(
                shadow_behavior_axis_matrix(personas, test_simulations, axis_names=axis_names),
                self.config.coverage_radius,
            )
            per_seed.append(
                {
                    "seed": seed_result["seed"],
                    "evaluation_repeat": repeat_index,
                    "candidate_id": best.candidate_id,
                    "schema_evaluation": schema_evaluation.to_dict(),
                    "internal_consistency": consistency_evaluation.to_dict(),
                    "test_shadow_behavior": test_shadow_behavior.to_dict(),
                    "test_behavior_diversity": test_behavior_diversity,
                    "test_shadow_simulations": [
                        asdict(simulation) for simulation in test_simulations
                    ],
                }
            )

        metrics = _aggregate_final_test_metrics(per_seed)
        repeat_metrics = [
            {
                "evaluation_repeat": repeat_index,
                "metrics": _aggregate_final_test_metrics(
                    [item for item in per_seed if item["evaluation_repeat"] == repeat_index]
                ),
            }
            for repeat_index in range(1, len(repeat_groups) + 1)
        ]
        logger.info("Final sealed test metrics: %s", metrics)
        return {
            "candidate_id": best.candidate_id,
            "candidate_fitness": best.fitness,
            "selection_metric": "validation",
            "test_used_for_selection": False,
            "evaluation_repeats": len(repeat_groups),
            "survey_hashes": self.survey_hashes,
            "metrics": metrics,
            "repeat_metrics": repeat_metrics,
            "per_seed": per_seed,
            "created_at": datetime.now().isoformat(),
        }

    def _generate_persona_results(
        self,
        candidate: MegaEvolutionCandidate,
        slots: list[MegaPersonaSlot],
        max_workers_override: int | None = None,
    ) -> list[MegaPersonaGenerationResult]:
        if self.config.generator_mode == "mock":
            personas = RuleBasedMegaPersonaBuilder().build_population(slots)
            return [
                MegaPersonaGenerationResult(
                    slot=slot,
                    persona=persona,
                    validation_report=evaluate_mega_personas([persona]).validation_reports[0],
                    candidate_json=persona.model_dump(),
                )
                for slot, persona in zip(slots, personas)
            ]
        generator = MegaPersonaGenerator(
            self.llm_client,
            temperature=self.config.persona_temperature,
            top_p=self.config.persona_top_p,
            prompt_addendum=prompt_addendum_from_genome(candidate.genome),
            blueprint_builder=lambda slot: blueprint_from_slot(candidate.genome, slot),
            pipeline_mode=self.config.persona_pipeline,
        )
        return generator.generate_from_slots(
            slots,
            max_workers=max_workers_override or self.config.persona_max_workers,
        )

    def _next_generation(self) -> list[MegaEvolutionCandidate]:
        evaluated = sorted(
            [candidate for candidate in self.population if candidate.evaluated],
            key=lambda candidate: candidate.fitness or float("-inf"),
            reverse=True,
        )
        elites = evaluated[: max(1, self.config.elite_count)]
        next_population = [
            MegaEvolutionCandidate.from_dict(candidate.to_dict())
            for candidate in elites
        ]
        children_needed = max(
            self.config.children_per_generation,
            self.config.population_size - len(next_population),
        )
        for child_idx in range(children_needed):
            parent = elites[child_idx % len(elites)]
            base_scale = 0.12 + 0.02 * min(self.generation, 4)
            mutation_mode = _child_mutation_mode(child_idx)
            scale = base_scale * _mutation_mode_scale_multiplier(mutation_mode)
            next_population.append(
                self._mutate_candidate(
                    parent,
                    generation=self.generation + 1,
                    mutation_scale=scale,
                    mutation_mode=mutation_mode,
                )
            )
            if len(next_population) >= self.config.population_size:
                break
        return next_population[: self.config.population_size]

    def _mutate_candidate(
        self,
        parent: MegaEvolutionCandidate,
        generation: int,
        mutation_scale: float,
        mutation_mode: str = "mixed",
    ) -> MegaEvolutionCandidate:
        genome = mutate_genome(parent.genome, self.rng, mutation_scale, mutation_mode=mutation_mode)
        return MegaEvolutionCandidate(
            candidate_id=f"candidate_{generation:04d}_{uuid.uuid4().hex[:8]}",
            genome=genome,
            generation=generation,
            parent_id=parent.candidate_id,
        )

    def _candidate_by_id(self, candidate_id: str) -> MegaEvolutionCandidate | None:
        for candidate in self.population:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def _write_generation_summary(self) -> None:
        self.store.write_generation_summary(
            generation=self.generation,
            population=self.population,
            best=self.best_candidate(),
        )

    def _save_checkpoint(self) -> None:
        state = {
            "config": _config_to_dict(self.config),
            "generation": self.generation,
            "evaluation_count": self.evaluation_count,
            "best_candidate_id": self.best_candidate_id,
            "survey_hashes": self.survey_hashes,
            "rng_state": self.rng.bit_generator.state,
            "population": [candidate.to_dict() for candidate in self.population],
            "updated_at": datetime.now().isoformat(),
        }
        self.store.write_checkpoint(state)

    def _load_checkpoint(self) -> None:
        state = self.store.read_checkpoint()
        self._validate_resume_config(state)
        self.generation = state["generation"]
        self.evaluation_count = state.get("evaluation_count", 0)
        self.best_candidate_id = state.get("best_candidate_id")
        self.population = [
            MegaEvolutionCandidate.from_dict(item)
            for item in state.get("population", [])
        ]
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        saved_hashes = state.get("survey_hashes")
        if saved_hashes:
            loaded_splits = read_shadow_survey_splits(self.store.surveys_dir)
            loaded_hashes = shadow_survey_split_hashes(loaded_splits)
            if loaded_hashes != saved_hashes:
                raise ValueError(
                    "Frozen shadow survey hashes do not match checkpoint. "
                    f"checkpoint={saved_hashes}, current={loaded_hashes}"
                )

    def _validate_resume_config(self, state: dict[str, Any]) -> None:
        saved_config = state.get("config")
        if not saved_config:
            return
        saved_config = _normalize_config_dict(saved_config)
        current_config = _normalize_config_dict(_config_to_dict(self.config))
        ignored = {"generations", "max_workers", "shadow_max_workers", "persona_max_workers"}
        mismatches = {}
        for key, saved_value in saved_config.items():
            if key in ignored:
                continue
            if current_config.get(key) != saved_value:
                mismatches[key] = {
                    "checkpoint": saved_value,
                    "current": current_config.get(key),
                }
        if mismatches:
            raise ValueError(
                "Resume config does not match checkpoint. "
                "Only --generations may change when resuming. "
                f"Mismatches: {mismatches}"
            )


class EvolutionStore:
    """Append-friendly artifact store with atomic JSON writes."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.evaluations_dir = output_dir / "evaluations"
        self.generations_dir = output_dir / "generations"
        self.candidates_dir = output_dir / "candidates"
        self.surveys_dir = output_dir / "shadow_surveys"
        self.checkpoint_path = output_dir / "checkpoint.json"

    def initialize(self) -> None:
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.surveys_dir.mkdir(parents=True, exist_ok=True)

    def write_evaluation(
        self,
        evaluation_index: int,
        candidate: MegaEvolutionCandidate,
        payload: dict[str, Any],
    ) -> None:
        self.initialize()
        eval_dir = self.evaluations_dir / f"eval_{evaluation_index:06d}_{candidate.candidate_id}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(eval_dir / "result.json", payload)
        _atomic_write_json(self.candidates_dir / f"{candidate.candidate_id}.json", candidate.to_dict())

    def write_cached_evaluation_alias(
        self,
        candidate: MegaEvolutionCandidate,
        payload: dict[str, Any],
    ) -> None:
        """Persist a cache-hit candidate without increasing evaluation_count."""
        self.initialize()
        eval_dir = self.evaluations_dir / f"alias_{candidate.candidate_id}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(eval_dir / "result.json", payload)
        _atomic_write_json(self.candidates_dir / f"{candidate.candidate_id}.json", candidate.to_dict())

    def write_generation_summary(
        self,
        generation: int,
        population: list[MegaEvolutionCandidate],
        best: MegaEvolutionCandidate,
    ) -> None:
        self.initialize()
        payload = {
            "generation": generation,
            "best_candidate_id": best.candidate_id,
            "best_fitness": best.fitness,
            "population": [
                {
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "fitness": candidate.fitness,
                    "evaluated": candidate.evaluated,
                    "generation": candidate.generation,
                    "last_mutation": candidate.genome.get("last_mutation"),
                    "last_evolution_operator": candidate.genome.get("last_evolution_operator"),
                }
                for candidate in population
            ],
        }
        _atomic_write_json(self.generations_dir / f"generation_{generation:04d}.json", payload)

    def write_checkpoint(self, state: dict[str, Any]) -> None:
        self.initialize()
        _atomic_write_json(self.checkpoint_path, state)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.initialize()
        _atomic_write_json(self.output_dir / "manifest.json", manifest)

    def read_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        with open(self.checkpoint_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def find_candidate_result(self, candidate_id: str) -> dict[str, Any] | None:
        paths = sorted(self.evaluations_dir.glob("eval_*_*/result.json"))
        paths.extend(sorted(self.evaluations_dir.glob("alias_*/result.json")))
        for path in reversed(paths):
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if payload.get("candidate", {}).get("candidate_id") == candidate_id:
                return payload
        return None

    def write_final_test_report(self, payload: dict[str, Any]) -> None:
        self.initialize()
        _atomic_write_json(self.output_dir / "final_test_report.json", payload)

    def write_final_summary(
        self,
        best: MegaEvolutionCandidate,
        population: list[MegaEvolutionCandidate],
        config: MegaEvolutionConfig,
        final_test_report: dict[str, Any] | None = None,
    ) -> None:
        self.initialize()
        payload = {
            "config": _config_to_dict(config),
            "best": best.to_dict(),
            "population": [candidate.to_dict() for candidate in population],
            "final_test_report": final_test_report,
            "completed_at": datetime.now().isoformat(),
        }
        _atomic_write_json(self.output_dir / "final_summary.json", payload)
        (self.output_dir / "final_summary.md").write_text(
            _final_markdown(payload),
            encoding="utf-8",
        )


def default_genome() -> dict[str, Any]:
    schema_binding = default_schema_binding()
    axis_names = axis_names_for_binding(schema_binding)
    quota_buckets = quota_buckets_for_binding(schema_binding)
    return {
        "genome_version": 3,
        "schema_binding": schema_binding,
        "quota_weights": {
            bucket.label: bucket.weight
            for bucket in quota_buckets
        },
        "axis_bias": {axis: 0.0 for axis in axis_names},
        "axis_stretch": {axis: 1.0 for axis in axis_names},
        "prompt_profile": {
            "mechanism_focus": "balanced",
            "tension_level": "moderate",
            "specificity": "concrete",
            "anti_stereotype": "explicit",
            "axis_binding": "mechanistic",
            "coverage_strategy": "balanced_space",
            "behavioral_signal": "survey_predictive",
        },
        "agent_focus": {
            "demographics": "Keep demographic context ordinary, non-stereotyped, and useful for later constraints.",
            "learning_mindset": "Explain thinking style, motive, and self-regulation through concrete school situations.",
            "values": "Connect values to choices under pressure rather than abstract virtues.",
            "social_creative": "Show social participation, creative confidence, and peer context through observable routines.",
            "health": "Describe stress, recovery, and support patterns with bounded realism.",
        },
        "field_requirements": [
            "Every long narrative field must include at least one observable behavior and one causal reason.",
            "Low-axis signals must remain visible and cannot be rescued by generic competence.",
            "High-axis signals must include a realistic boundary or tradeoff.",
        ],
        "behavior_anchors": {
            "learning": "Trigger: ambiguous assignment. Interpretation: what kind of evidence feels useful. Response: planning, asking, or experimenting behavior.",
            "motivation": "Trigger: external expectation conflicts with interest. Interpretation: approval, autonomy, or obligation. Response: compromise, persistence, or avoidance.",
            "belonging": "Trigger: group task or teacher feedback. Interpretation: safety, comparison, or usefulness of help. Response: help-seeking, withdrawal, or contribution.",
            "stress_recovery": "Trigger: deadline, conflict, or failed attempt. Interpretation: threat versus recoverable friction. Response: latency, coping routine, and later adjustment.",
        },
        "consistency_rules": [
            "Cognition, motivation, self-regulation, social behavior, and health must describe the same student, not five independent profiles.",
            "A limitation in one field must have a plausible echo in at least one other field.",
            "Contrasts are allowed only when a context boundary explains why behavior changes.",
        ],
        "repair_policy": {
            "length": "Prefer concise causal evidence over decorative backstory; stay within schema length limits.",
            "conflict": "If two fields conflict, preserve the target axes and rewrite the weaker field as context-dependent.",
            "coverage": "When diversity collapses, vary mechanism and context instead of adding extreme traits.",
        },
        "blueprint_policy": {
            "core_tension_rule": "Choose one ordinary tension that links the strongest and weakest target axes.",
            "axis_evidence_rule": "For every primary axis, name one observable school-life trigger and one boundary condition.",
            "quota_context_rule": "Use the quota bucket as a context prior, not as a stereotype or final label.",
            "persona_coherence_rule": "All agents must describe the same causal person through repeated behavior anchors.",
        },
        "axis_expression_policy": {
            "low": "Show a visible cost plus a partial workaround; do not rescue the signal into competence.",
            "mid": "Show context-sensitive behavior that changes across task type, audience, or stress level.",
            "high": "Show useful strength plus a boundary where the trait stops helping.",
        },
        "cross_agent_binding_policy": {
            "cognition_to_values": "Values must explain why the cognition pattern feels worth keeping or changing.",
            "cognition_to_social": "Social behavior must echo the person's ambiguity handling and help-seeking style.",
            "regulation_to_health": "Mental health must reuse the same recovery latency and coping routine implied by self-regulation.",
            "values_to_behavior": "Moral tension must predict at least one peer, deadline, or feedback behavior.",
        },
        "behavior_prediction_policy": {
            "ambiguous_task": "Predict first move, evidence preference, and whether the student asks, tests, copies, or delays.",
            "peer_pressure": "Predict whether the student conforms, negotiates, performs, withdraws, or seeks one trusted ally.",
            "failure_feedback": "Predict appraisal, emotional latency, repair action, and what changes next time.",
            "deadline": "Predict planning rhythm, shortcut risk, and recovery when the schedule slips.",
        },
        "critic_policy": {
            "axis_echo": "Each primary axis must appear in at least two fields through behavior, not labels.",
            "behavior_echo": "Ambiguous task, peer pressure, failure feedback, and deadline anchors must be inferable.",
            "contradiction_check": "A cross-field contradiction is allowed only with an explicit context boundary.",
            "length_check": "Trim decorative biography before removing behavior-predictive evidence.",
        },
        "last_evolution_operator": None,
    }


V4_PROBE_SCENARIOS = (
    "ambiguous_task",
    "peer_pressure",
    "failure_feedback",
    "deadline",
)
V4_AXIS_REALIZATION_MODES = (
    "visible_cost",
    "context_switch",
    "tradeoff",
)
V4_INTERACTION_MODES = (
    "strongest_weakest_tension",
    "compensatory",
    "context_switch",
    "independent",
)
V4_ECHO_GRAPHS = (
    "axis_behavior_cross_field",
    "cognition_values_social",
    "regulation_health_feedback",
)
V4_CONTEXT_MODES = (
    "quota_conditioned",
    "support_conditioned",
    "audience_conditioned",
)
V4_REPAIR_PRIORITIES = (
    "preserve_axis_signal",
    "preserve_behavior_trace",
    "preserve_schema_precision",
)
V4_AXIS_ROLES = (
    "cognitive_core",
    "motivation_core",
    "regulation_core",
)


def default_genome_v4() -> dict[str, Any]:
    """Return the low-dimensional structured Genome v4 seed.

    Sampling controls remain present for compatibility with candidate_slots(),
    but v4 operators deliberately leave them fixed. The evolvable surface is a
    small behavior-generation program rendered deterministically into the
    existing generation blueprint.
    """
    v3 = default_genome()
    return {
        "genome_version": 4,
        "schema_binding": v3["schema_binding"],
        "quota_weights": v3["quota_weights"],
        "axis_bias": v3["axis_bias"],
        "axis_stretch": v3["axis_stretch"],
        "probe_assignment": {
            "cognitive_core": "ambiguous_task",
            "motivation_core": "peer_pressure",
            "regulation_core": "failure_feedback",
        },
        "axis_realization": {
            "cognitive_core": {"mode": "tradeoff", "strength": 0.65},
            "motivation_core": {"mode": "context_switch", "strength": 0.60},
            "regulation_core": {"mode": "visible_cost", "strength": 0.70},
        },
        "interaction_mode": "strongest_weakest_tension",
        "echo_graph": "axis_behavior_cross_field",
        "context_modulation": "quota_conditioned",
        "repair_control": {
            "evidence_density": 2,
            "priority": "preserve_axis_signal",
        },
        "last_evolution_operator": None,
    }


def failed_evaluation_payload(
    candidate: MegaEvolutionCandidate,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "candidate": candidate.to_dict(),
        "fitness": 0.0,
        "metrics": {
            "error": type(exc).__name__,
            "message": str(exc),
        },
        "per_seed": [],
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def _failed_seed_result(seed: int, candidate_id: str, exc: Exception) -> dict[str, Any]:
    zero_shadow = _empty_shadow_behavior()
    zero_diversity = _zero_diversity()
    return {
        "seed": seed,
        "candidate_id": candidate_id,
        "status": "failed",
        "score": 0.0,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "slots": [],
        "personas": [],
        "generation_diagnostics": [],
        "schema_evaluation": _empty_schema_evaluation(),
        "internal_consistency": _empty_consistency_evaluation(),
        "train_shadow_behavior": zero_shadow,
        "validation_shadow_behavior": zero_shadow,
        "train_behavior_diversity": zero_diversity,
        "validation_behavior_diversity": zero_diversity,
        "slot_diversity": zero_diversity,
        "shadow_survey_hashes": {},
        "train_shadow_simulations": [],
        "validation_shadow_simulations": [],
    }


def _no_persona_seed_result(
    *,
    seed: int,
    candidate_id: str,
    slots: list[MegaPersonaSlot],
    generation_results: list[MegaPersonaGenerationResult],
    survey_hashes: dict[str, str],
) -> dict[str, Any]:
    zero_shadow = _empty_shadow_behavior()
    zero_diversity = _zero_diversity()
    return {
        "seed": seed,
        "candidate_id": candidate_id,
        "status": "generation_failed_no_personas",
        "score": 0.0,
        "error": {
            "type": "NoPersonasGenerated",
            "message": "No valid personas were generated after the low-concurrency retry.",
        },
        "slots": [asdict(slot) for slot in slots],
        "personas": [],
        "generation_diagnostics": [
            _generation_result_diagnostic(result) for result in generation_results
        ],
        "schema_evaluation": _empty_schema_evaluation(),
        "internal_consistency": _empty_consistency_evaluation(),
        "train_shadow_behavior": zero_shadow,
        "validation_shadow_behavior": zero_shadow,
        "train_behavior_diversity": zero_diversity,
        "validation_behavior_diversity": zero_diversity,
        "slot_diversity": zero_diversity,
        "shadow_survey_hashes": survey_hashes,
        "train_shadow_simulations": [],
        "validation_shadow_simulations": [],
    }


def _generation_result_diagnostic(result: MegaPersonaGenerationResult) -> dict[str, Any]:
    return {
        "slot_id": result.slot.slot_id,
        "quota_label": result.slot.quota_label,
        "valid": result.is_valid,
        "schema_valid": result.validation_report.schema_valid,
        "issue_count": len(result.validation_report.issues),
        "issues": [
            {
                "rule_id": issue.rule_id,
                "severity": issue.severity,
                "message": issue.message,
            }
            for issue in result.validation_report.issues[:8]
        ],
        "candidate_keys": sorted(result.candidate_json.keys()),
    }


def _empty_schema_evaluation() -> dict[str, Any]:
    return {
        "sample_size": 0,
        "valid_count": 0,
        "validity_rate": 0.0,
        "near_duplicate_rate": 0.0,
        "axis_names": list(AXIS_NAMES),
        "diversity_metrics": _zero_diversity(),
        "fitness": 0.0,
    }


def _empty_consistency_evaluation() -> dict[str, Any]:
    return {
        "sample_size": 0,
        "mean_score": 0.0,
        "min_score": 0.0,
        "axis_alignment_mean": 0.0,
        "rule_score_mean": 0.0,
        "axis_target_mae_mean": 1.0,
        "weighted_issue_rate": 1.0,
        "consistency_issue_rate": 1.0,
        "strict_consistency_error": 1.0,
        "issue_count": 0,
        "rule_counts": {},
        "reports": [],
    }


def _empty_shadow_behavior() -> dict[str, Any]:
    return {
        "sample_size": 0,
        "survey_count": 0,
        "axis_names": list(AXIS_NAMES),
        "behavior_axis_mean": {axis: 0.0 for axis in AXIS_NAMES},
        "persona_behavior_mae": {axis: 0.0 for axis in AXIS_NAMES},
        "overall_alignment": 0.0,
        "overall_mae": 1.0,
    }


def _zero_diversity() -> dict[str, float]:
    return {
        "coverage": 0.0,
        "convex_hull": 0.0,
        "avg_dist": 0.0,
        "min_dist": 0.0,
        "dispersion": 0.0,
        "kl_divergence": 0.0,
    }


def mutate_genome(
    genome: dict[str, Any],
    rng: np.random.Generator,
    mutation_scale: float,
    mutation_mode: str = "mixed",
    operator_id: str | None = None,
) -> dict[str, Any]:
    if int(genome.get("genome_version", 3)) == 4:
        return mutate_genome_v4(
            genome,
            rng,
            mutation_scale=mutation_scale,
            operator_id=operator_id,
        )
    mutated = normalize_genome(genome)
    mutated["schema_binding"] = schema_binding_for_genome(mutated)
    axis_names = axis_names_for_binding(mutated["schema_binding"])
    if mutation_mode not in {"mixed", "prompt_only", "operator_only", "numeric_only"}:
        raise ValueError(f"unknown mutation_mode: {mutation_mode}")
    operator = _select_evolution_operator(rng, operator_id=operator_id)
    numeric_mutation = mutation_mode in {"mixed", "numeric_only"}
    prompt_profile_mutation = mutation_mode in {"mixed", "prompt_only"}
    operator_mutation = mutation_mode in {"mixed", "prompt_only", "operator_only"}

    if numeric_mutation:
        for label, weight in mutated["quota_weights"].items():
            mutated["quota_weights"][label] = _clip(weight + rng.normal(0.0, mutation_scale * 0.2), 0.02, 0.6)
        _normalize_weights(mutated["quota_weights"])

        for axis in axis_names:
            mutated["axis_bias"][axis] = _clip(
                mutated["axis_bias"].get(axis, 0.0) + rng.normal(0.0, mutation_scale * 0.15),
                -0.35,
                0.35,
            )
            mutated["axis_stretch"][axis] = _clip(
                mutated["axis_stretch"].get(axis, 1.0) + rng.normal(0.0, mutation_scale * 0.25),
                0.55,
                1.75,
            )

    mutated.setdefault(
        "prompt_profile",
        json.loads(json.dumps(default_genome()["prompt_profile"])),
    )
    if prompt_profile_mutation:
        _mutate_prompt_profile(mutated["prompt_profile"], rng, mutation_scale)
        _mutate_structural_genome(mutated, rng, mutation_scale)
    if operator_mutation:
        _apply_evolution_operator(
            mutated,
            operator,
            rng,
            mutation_scale,
            apply_numeric=mutation_mode != "prompt_only",
        )
        mutated["last_evolution_operator"] = {
            "id": operator["id"],
            "name": operator["name"],
            "instruction": operator["instruction"],
        }
    else:
        mutated["last_evolution_operator"] = None
    mutated["last_mutation"] = {
        "mode": mutation_mode,
        "scale": mutation_scale,
    }
    return mutated


def mutate_genome_v4(
    genome: dict[str, Any],
    rng: np.random.Generator,
    mutation_scale: float,
    operator_id: str | None = None,
) -> dict[str, Any]:
    """Apply one auditable, single-module mutation to Genome v4."""
    mutated = normalize_genome_v4(genome)
    operator = _select_v4_operator(rng, operator_id=operator_id)
    module = str(operator["v4_module"])

    if module == "probe_assignment":
        role = str(rng.choice(V4_AXIS_ROLES))
        current = mutated["probe_assignment"][role]
        choices = [value for value in V4_PROBE_SCENARIOS if value != current]
        mutated["probe_assignment"][role] = str(rng.choice(choices))
    elif module == "axis_realization":
        role = str(rng.choice(V4_AXIS_ROLES))
        realization = mutated["axis_realization"][role]
        if rng.random() < 0.5:
            choices = [value for value in V4_AXIS_REALIZATION_MODES if value != realization["mode"]]
            realization["mode"] = str(rng.choice(choices))
        else:
            step = max(0.05, float(mutation_scale) * 0.5)
            direction = -1.0 if rng.random() < 0.5 else 1.0
            current_strength = float(realization["strength"])
            new_strength = _clip(
                current_strength + direction * step,
                0.35,
                0.90,
            )
            if abs(new_strength - current_strength) <= 1e-12:
                new_strength = _clip(current_strength - direction * step, 0.35, 0.90)
            realization["strength"] = new_strength
    elif module == "interaction_mode":
        choices = [value for value in V4_INTERACTION_MODES if value != mutated["interaction_mode"]]
        mutated["interaction_mode"] = str(rng.choice(choices))
    elif module == "echo_graph":
        choices = [value for value in V4_ECHO_GRAPHS if value != mutated["echo_graph"]]
        mutated["echo_graph"] = str(rng.choice(choices))
    elif module == "context_modulation":
        choices = [value for value in V4_CONTEXT_MODES if value != mutated["context_modulation"]]
        mutated["context_modulation"] = str(rng.choice(choices))
    elif module == "repair_control":
        repair = mutated["repair_control"]
        if rng.random() < 0.5:
            repair["evidence_density"] = 1 + (int(repair["evidence_density"]) % 3)
        else:
            choices = [value for value in V4_REPAIR_PRIORITIES if value != repair["priority"]]
            repair["priority"] = str(rng.choice(choices))
    else:
        raise ValueError(f"unknown Genome v4 mutation module: {module}")

    mutated["last_evolution_operator"] = {
        "id": operator["id"],
        "name": operator["name"],
        "instruction": operator["instruction"],
        "module": module,
    }
    mutated["last_mutation"] = {
        "mode": "structured_v4",
        "module": module,
        "scale": float(mutation_scale),
    }
    return mutated


def prompt_addendum_from_genome(genome: dict[str, Any]) -> str:
    genome = normalize_genome(genome)
    if int(genome.get("genome_version", 3)) == 4:
        return _prompt_addendum_from_genome_v4(genome)
    profile = genome.get("prompt_profile", {})
    mechanism_focus = _profile_choice(profile, "mechanism_focus", "balanced")
    tension_level = _profile_choice(profile, "tension_level", "moderate")
    specificity = _profile_choice(profile, "specificity", "concrete")
    anti_stereotype = _profile_choice(profile, "anti_stereotype", "explicit")
    axis_binding = _profile_choice(profile, "axis_binding", "mechanistic")
    coverage_strategy = _profile_choice(profile, "coverage_strategy", "balanced_space")
    behavioral_signal = _profile_choice(profile, "behavioral_signal", "survey_predictive")
    parts = [
        (
            "Respect all schema length limits. Keep narrative fields concise, concrete, and under their maximum length; "
            "do not add decorative detail that does not improve behavioral prediction."
        ),
        PROMPT_POLICY_BANK["mechanism_focus"][mechanism_focus],
        PROMPT_POLICY_BANK["tension_level"][tension_level],
        PROMPT_POLICY_BANK["specificity"][specificity],
        PROMPT_POLICY_BANK["anti_stereotype"][anti_stereotype],
        PROMPT_POLICY_BANK["axis_binding"][axis_binding],
        PROMPT_POLICY_BANK["coverage_strategy"][coverage_strategy],
        PROMPT_POLICY_BANK["behavioral_signal"][behavioral_signal],
    ]
    # Lineage metadata is mutator-facing only. The selected operator's
    # instruction guides the mutator; injecting it here would leak search-level
    # intent into the persona generation prompt, so it is deliberately omitted.
    parts.append(_structured_prompt_constraints(genome))
    return "\n".join(parts)


def normalize_genome(genome: dict[str, Any]) -> dict[str, Any]:
    """Return a backward-compatible Genome v3 object.

    Older checkpoints may only contain sampling and prompt-profile fields. This
    function fills structural prompt and blueprint modules without changing the
    caller's object in-place.
    """
    if isinstance(genome, dict) and int(genome.get("genome_version", 3)) == 4:
        return normalize_genome_v4(genome)
    base = default_genome()
    normalized = json.loads(json.dumps(base))
    if not isinstance(genome, dict):
        return normalized
    schema_binding = schema_binding_for_genome(genome)
    normalized["schema_binding"] = schema_binding
    axis_names = axis_names_for_binding(schema_binding)
    axis_roles = axis_roles_for_binding(schema_binding)
    quota_buckets = quota_buckets_for_binding(schema_binding)

    source_quota = genome.get("quota_weights", {})
    normalized["quota_weights"] = {
        bucket.label: _clip(
            _safe_float_like(source_quota.get(bucket.label, bucket.weight), bucket.weight),
            0.02,
            0.6,
        )
        for bucket in quota_buckets
    }
    _normalize_weights(normalized["quota_weights"])

    source_bias = genome.get("axis_bias", {})
    source_stretch = genome.get("axis_stretch", {})
    normalized["axis_bias"] = {}
    normalized["axis_stretch"] = {}
    for axis in axis_names:
        legacy_keys = [
            axis,
            "cognitive_abstraction" if axis == axis_roles.get("cognitive_core") else None,
            "motivation_autonomy" if axis == axis_roles.get("motivation_core") else None,
            "self_regulation_resilience" if axis == axis_roles.get("regulation_core") else None,
        ]
        bias_value = _first_present_number(source_bias, legacy_keys, 0.0)
        stretch_value = _first_present_number(source_stretch, legacy_keys, 1.0)
        normalized["axis_bias"][axis] = _clip(bias_value, -0.35, 0.35)
        normalized["axis_stretch"][axis] = _clip(stretch_value, 0.55, 1.75)

    if isinstance(genome.get("prompt_profile"), dict):
        normalized["prompt_profile"].update(genome["prompt_profile"])
    for key in (
        "agent_focus",
        "behavior_anchors",
        "repair_policy",
        "blueprint_policy",
        "axis_expression_policy",
        "cross_agent_binding_policy",
        "behavior_prediction_policy",
        "critic_policy",
    ):
        if isinstance(genome.get(key), dict):
            normalized[key].update(_clean_string_dict(genome[key], normalized[key]))
    for key in ("field_requirements", "consistency_rules"):
        if isinstance(genome.get(key), list):
            normalized[key] = _bounded_string_list(
                genome[key],
                fallback=normalized[key],
                max_items=8,
                max_chars=180,
            )
    if isinstance(genome.get("last_evolution_operator"), dict) or genome.get("last_evolution_operator") is None:
        normalized["last_evolution_operator"] = genome.get("last_evolution_operator")
    if isinstance(genome.get("openevolve_mutation"), dict):
        normalized["openevolve_mutation"] = genome["openevolve_mutation"]
    if isinstance(genome.get("last_mutation"), dict):
        normalized["last_mutation"] = genome["last_mutation"]
    normalized["genome_version"] = 3
    return normalized


def normalize_genome_v4(genome: dict[str, Any]) -> dict[str, Any]:
    """Normalize Genome v4 without accepting new free-form prompt fields."""
    base = default_genome_v4()
    normalized = json.loads(json.dumps(base))
    if not isinstance(genome, dict):
        return normalized

    schema_binding = schema_binding_for_genome(genome)
    normalized["schema_binding"] = schema_binding
    axis_names = axis_names_for_binding(schema_binding)
    quota_buckets = quota_buckets_for_binding(schema_binding)

    source_quota = genome.get("quota_weights", {})
    normalized["quota_weights"] = {
        bucket.label: _clip(
            _safe_float_like(source_quota.get(bucket.label, bucket.weight), bucket.weight),
            0.02,
            0.6,
        )
        for bucket in quota_buckets
    }
    _normalize_weights(normalized["quota_weights"])
    source_bias = genome.get("axis_bias", {})
    source_stretch = genome.get("axis_stretch", {})
    normalized["axis_bias"] = {
        axis: _clip(_safe_float_like(source_bias.get(axis, 0.0), 0.0), -0.35, 0.35)
        for axis in axis_names
    }
    normalized["axis_stretch"] = {
        axis: _clip(_safe_float_like(source_stretch.get(axis, 1.0), 1.0), 0.55, 1.75)
        for axis in axis_names
    }

    raw_probes = genome.get("probe_assignment", {})
    raw_realization = genome.get("axis_realization", {})
    for role in V4_AXIS_ROLES:
        scenario = raw_probes.get(role, base["probe_assignment"][role])
        if scenario in V4_PROBE_SCENARIOS:
            normalized["probe_assignment"][role] = scenario
        realization = raw_realization.get(role, {})
        if not isinstance(realization, dict):
            realization = {}
        mode = realization.get("mode", base["axis_realization"][role]["mode"])
        if mode not in V4_AXIS_REALIZATION_MODES:
            mode = base["axis_realization"][role]["mode"]
        normalized["axis_realization"][role] = {
            "mode": mode,
            "strength": _clip(
                _safe_float_like(
                    realization.get("strength", base["axis_realization"][role]["strength"]),
                    base["axis_realization"][role]["strength"],
                ),
                0.35,
                0.90,
            ),
        }

    for key, choices in (
        ("interaction_mode", V4_INTERACTION_MODES),
        ("echo_graph", V4_ECHO_GRAPHS),
        ("context_modulation", V4_CONTEXT_MODES),
    ):
        value = genome.get(key, base[key])
        normalized[key] = value if value in choices else base[key]

    repair = genome.get("repair_control", {})
    if not isinstance(repair, dict):
        repair = {}
    priority = repair.get("priority", base["repair_control"]["priority"])
    normalized["repair_control"] = {
        "evidence_density": int(
            _clip(
                _safe_float_like(
                    repair.get("evidence_density", base["repair_control"]["evidence_density"]),
                    base["repair_control"]["evidence_density"],
                ),
                1,
                3,
            )
        ),
        "priority": priority if priority in V4_REPAIR_PRIORITIES else base["repair_control"]["priority"],
    }
    if isinstance(genome.get("last_evolution_operator"), dict) or genome.get("last_evolution_operator") is None:
        normalized["last_evolution_operator"] = genome.get("last_evolution_operator")
    if isinstance(genome.get("openevolve_mutation"), dict):
        normalized["openevolve_mutation"] = genome["openevolve_mutation"]
    if isinstance(genome.get("last_mutation"), dict):
        normalized["last_mutation"] = genome["last_mutation"]
    return normalized


def _prompt_addendum_from_genome_v4(genome: dict[str, Any]) -> str:
    repair = genome["repair_control"]
    probe_summary = ", ".join(
        f"{role}={scenario}"
        for role, scenario in genome["probe_assignment"].items()
    )
    return "\n".join(
        (
            "Respect all schema length limits. Treat the Genome v4 generation blueprint as the source of behavioral constraints.",
            f"Structured probe assignment: {probe_summary}.",
            f"Interaction mode: {genome['interaction_mode']}; echo graph: {genome['echo_graph']}.",
            f"Context modulation: {genome['context_modulation']}; evidence density: {repair['evidence_density']}.",
            f"Repair priority: {repair['priority']}. Preserve concrete behavior evidence instead of adding decorative biography.",
        )
    )


def _safe_float_like(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_present_number(
    source: dict[str, Any],
    keys: list[str | None],
    default: float,
) -> float:
    for key in keys:
        if key and key in source:
            return _safe_float_like(source[key], default)
    return float(default)


def _structured_prompt_constraints(genome: dict[str, Any]) -> str:
    lines = ["Evolvable structural controls:"]
    agent_focus = genome.get("agent_focus", {})
    if agent_focus:
        lines.append("Agent focus:")
        for key, value in agent_focus.items():
            lines.append(f"- {key}: {value}")
    field_requirements = genome.get("field_requirements", [])
    if field_requirements:
        lines.append("Field requirements:")
        for item in field_requirements:
            lines.append(f"- {item}")
    behavior_anchors = genome.get("behavior_anchors", {})
    if behavior_anchors:
        lines.append("Behavior anchors:")
        for key, value in behavior_anchors.items():
            lines.append(f"- {key}: {value}")
    consistency_rules = genome.get("consistency_rules", [])
    if consistency_rules:
        lines.append("Internal consistency rules:")
        for item in consistency_rules:
            lines.append(f"- {item}")
    repair_policy = genome.get("repair_policy", {})
    if repair_policy:
        lines.append("Repair policy:")
        for key, value in repair_policy.items():
            lines.append(f"- {key}: {value}")
    blueprint_policy = genome.get("blueprint_policy", {})
    if blueprint_policy:
        lines.append("Genome v3 blueprint policy:")
        for key, value in blueprint_policy.items():
            lines.append(f"- {key}: {value}")
    axis_expression_policy = genome.get("axis_expression_policy", {})
    if axis_expression_policy:
        lines.append("Axis expression policy:")
        for key, value in axis_expression_policy.items():
            lines.append(f"- {key}: {value}")
    cross_agent_binding_policy = genome.get("cross_agent_binding_policy", {})
    if cross_agent_binding_policy:
        lines.append("Cross-agent binding policy:")
        for key, value in cross_agent_binding_policy.items():
            lines.append(f"- {key}: {value}")
    behavior_prediction_policy = genome.get("behavior_prediction_policy", {})
    if behavior_prediction_policy:
        lines.append("Behavior prediction policy:")
        for key, value in behavior_prediction_policy.items():
            lines.append(f"- {key}: {value}")
    critic_policy = genome.get("critic_policy", {})
    if critic_policy:
        lines.append("Blueprint critic policy:")
        for key, value in critic_policy.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def blueprint_from_slot(genome: dict[str, Any], slot: MegaPersonaSlot) -> dict[str, Any]:
    """Build a Genome v3 per-slot generation blueprint.

    This is the main v3 evolvable control surface: the genome no longer only
    appends soft prompt advice. It deterministically converts a slot's target
    axes into stage-specific behavior anchors and cross-agent consistency
    bindings that the generator injects into the multi-agent pipeline.
    """
    genome = normalize_genome(genome)
    if int(genome.get("genome_version", 3)) == 4:
        return _blueprint_from_slot_v4(genome, slot)
    axis_policy = genome.get("axis_expression_policy", {})
    behavior_policy = genome.get("behavior_prediction_policy", {})
    binding_policy = genome.get("cross_agent_binding_policy", {})
    critic_policy = genome.get("critic_policy", {})
    axis_plan = {}
    for axis, value in slot.target_axes.items():
        band = _axis_band(value)
        axis_plan[axis] = {
            "target": round(float(value), 3),
            "band": band,
            "evidence": _axis_evidence_line(axis, value, axis_policy.get(band, "")),
            "boundary": _axis_boundary_line(axis, value),
            "echo_fields": _axis_echo_fields(axis),
        }

    strongest = max(slot.target_axes.items(), key=lambda item: item[1])[0]
    weakest = min(slot.target_axes.items(), key=lambda item: item[1])[0]
    behavior_profile = {
        "ambiguous_task": _behavior_anchor_text(
            behavior_policy.get("ambiguous_task", ""),
            strongest=strongest,
            weakest=weakest,
            quota=slot.quota_label,
        ),
        "peer_pressure": _behavior_anchor_text(
            behavior_policy.get("peer_pressure", ""),
            strongest=strongest,
            weakest=weakest,
            quota=slot.quota_label,
        ),
        "failure_feedback": _behavior_anchor_text(
            behavior_policy.get("failure_feedback", ""),
            strongest=strongest,
            weakest=weakest,
            quota=slot.quota_label,
        ),
        "deadline": _behavior_anchor_text(
            behavior_policy.get("deadline", ""),
            strongest=strongest,
            weakest=weakest,
            quota=slot.quota_label,
        ),
    }
    cross_agent_binding = [
        f"{key}: {value}"
        for key, value in binding_policy.items()
        if isinstance(value, str) and value.strip()
    ]
    critic_checks = [
        f"{key}: {value}"
        for key, value in critic_policy.items()
        if isinstance(value, str) and value.strip()
    ]
    return {
        "blueprint_version": 3,
        "quota_label": slot.quota_label,
        "core_tension": (
            f"Hold {strongest} as the strongest visible axis while preserving "
            f"{weakest} as a real limitation or context-dependent vulnerability."
        ),
        "axis_expression_plan": axis_plan,
        "behavior_prediction_profile": behavior_profile,
        "cross_agent_binding": cross_agent_binding,
        "critic_checks": critic_checks,
        "slot_constraints": {
            key: slot.constraints.get(key)
            for key in (
                "primary_drive",
                "stress_band",
                "social_energy_band",
                "derived_performance_band",
            )
            if key in slot.constraints
        },
    }


def _blueprint_from_slot_v4(
    genome: dict[str, Any],
    slot: MegaPersonaSlot,
) -> dict[str, Any]:
    schema_binding = schema_binding_for_genome(genome)
    role_to_axis = axis_roles_for_binding(schema_binding)
    axis_plan: dict[str, dict[str, Any]] = {}
    scenario_axes: dict[str, list[str]] = {scenario: [] for scenario in V4_PROBE_SCENARIOS}

    for role in V4_AXIS_ROLES:
        axis = role_to_axis.get(role)
        if axis not in slot.target_axes:
            continue
        target = float(slot.target_axes[axis])
        realization = genome["axis_realization"][role]
        scenario = genome["probe_assignment"][role]
        scenario_axes[scenario].append(axis)
        axis_plan[axis] = {
            "target": round(target, 3),
            "band": _axis_band(target),
            "probe": scenario,
            "mode": realization["mode"],
            "strength": round(float(realization["strength"]), 3),
            "evidence": _v4_axis_evidence(axis, scenario, realization),
            "boundary": _v4_axis_boundary(axis, realization["mode"]),
            "echo_fields": _axis_echo_fields(axis),
        }

    strongest = max(slot.target_axes.items(), key=lambda item: item[1])[0]
    weakest = min(slot.target_axes.items(), key=lambda item: item[1])[0]
    behavior_profile = {
        scenario: _v4_behavior_probe(
            scenario,
            axes=scenario_axes[scenario],
            interaction_mode=genome["interaction_mode"],
            context_modulation=genome["context_modulation"],
            quota_label=slot.quota_label,
        )
        for scenario in V4_PROBE_SCENARIOS
    }
    repair = genome["repair_control"]
    return {
        "blueprint_version": 4,
        "quota_label": slot.quota_label,
        "core_tension": _v4_core_tension(
            strongest,
            weakest,
            genome["interaction_mode"],
        ),
        "axis_expression_plan": axis_plan,
        "behavior_prediction_profile": behavior_profile,
        "cross_agent_binding": _v4_echo_bindings(genome["echo_graph"]),
        "critic_checks": _v4_critic_checks(
            priority=repair["priority"],
            evidence_density=repair["evidence_density"],
        ),
        "structured_program": {
            "probe_assignment": genome["probe_assignment"],
            "interaction_mode": genome["interaction_mode"],
            "echo_graph": genome["echo_graph"],
            "context_modulation": genome["context_modulation"],
            "repair_control": repair,
        },
        "slot_constraints": {
            key: slot.constraints.get(key)
            for key in (
                "primary_drive",
                "stress_band",
                "social_energy_band",
                "derived_performance_band",
            )
            if key in slot.constraints
        },
    }


def _v4_axis_evidence(axis: str, scenario: str, realization: dict[str, Any]) -> str:
    mode_text = {
        "visible_cost": "show a visible cost and a partial workaround",
        "context_switch": "show a direction change across two named contexts",
        "tradeoff": "show one benefit and one boundary where it stops helping",
    }[realization["mode"]]
    strength = float(realization["strength"])
    strength_text = "restrained" if strength < 0.50 else "clear" if strength < 0.70 else "strong"
    return (
        f"Use {scenario} as the primary probe for {axis}; {mode_text}; "
        f"require {strength_text} behavioral evidence (signal strength={strength:.2f})."
    )


def _v4_axis_boundary(axis: str, mode: str) -> str:
    return {
        "visible_cost": f"The {axis} signal must retain a measurable cost under pressure.",
        "context_switch": f"The {axis} signal may change only when the context boundary is explicit.",
        "tradeoff": f"The {axis} signal must include both a useful consequence and a limiting consequence.",
    }[mode]


def _v4_behavior_probe(
    scenario: str,
    *,
    axes: list[str],
    interaction_mode: str,
    context_modulation: str,
    quota_label: str,
) -> str:
    axis_text = ", ".join(axes) if axes else "the strongest-weakest axis interaction"
    return (
        f"In {scenario}, make {axis_text} predict trigger, appraisal, first action, and aftereffect. "
        f"Use interaction={interaction_mode}, context={context_modulation}, and quota={quota_label}; "
        "do not quote or paraphrase survey items."
    )


def _v4_core_tension(strongest: str, weakest: str, mode: str) -> str:
    templates = {
        "strongest_weakest_tension": "Let {strongest} help first while {weakest} creates a recurring limit.",
        "compensatory": "Let {strongest} partially compensate for {weakest}, but preserve a residual cost.",
        "context_switch": "Let context decide whether {strongest} or {weakest} dominates observable behavior.",
        "independent": "Keep {strongest} and {weakest} behaviorally separable instead of one general competence factor.",
    }
    return templates[mode].format(strongest=strongest, weakest=weakest)


def _v4_echo_bindings(echo_graph: str) -> list[str]:
    return {
        "axis_behavior_cross_field": [
            "cognition to values: reuse the same appraisal mechanism",
            "cognition to social: reuse the same ambiguity and help-seeking threshold",
            "regulation to health: reuse recovery latency, coping action, and residual cost",
            "values to behavior: reuse the same peer-pressure or deadline decision",
        ],
        "cognition_values_social": [
            "cognition to values: appraisal determines what the student protects",
            "values to social: protected values predict negotiation, conformity, or withdrawal",
            "social to cognition: audience changes evidence seeking without changing the target axis",
        ],
        "regulation_health_feedback": [
            "regulation to health: coping latency and residual strain must match",
            "feedback to cognition: failure appraisal changes the next evidence-seeking action",
            "health to social: support seeking must match stress recovery and peer behavior",
        ],
    }[echo_graph]


def _v4_critic_checks(*, priority: str, evidence_density: int) -> list[str]:
    priority_text = {
        "preserve_axis_signal": "Repair schema issues without smoothing away low or high target-axis signals.",
        "preserve_behavior_trace": "Repair prose while preserving trigger, appraisal, action, and aftereffect.",
        "preserve_schema_precision": "Repair toward concise schema-valid fields without adding unsupported details.",
    }[priority]
    return [
        priority_text,
        f"Require at least {evidence_density} independent behavior evidence units for each primary axis.",
        "Reject contradictions that lack an explicit context boundary.",
    ]


def _axis_band(value: float) -> str:
    if value <= 0.34:
        return "low"
    if value >= 0.66:
        return "high"
    return "mid"


def _axis_evidence_line(axis: str, value: float, policy_text: str) -> str:
    band = _axis_band(value)
    axis_hint = {
        "cognitive": "how the student represents ambiguity and chooses evidence",
        "abstraction": "how the student represents ambiguity and chooses evidence",
        "motivation": "what starts action when nobody is watching",
        "autonomy": "what starts action when nobody is watching",
        "regulation": "how quickly routines recover after friction",
        "resilience": "how quickly routines recover after friction",
    }
    hint = "observable school behavior"
    lowered = axis.lower()
    for token, candidate in axis_hint.items():
        if token in lowered:
            hint = candidate
            break
    policy = policy_text or f"Show {band} expression with a concrete behavior."
    return f"{hint}; {policy}"


def _axis_boundary_line(axis: str, value: float) -> str:
    band = _axis_band(value)
    if band == "low":
        return f"{axis} should create a cost in one situation but retain a partial workaround."
    if band == "high":
        return f"{axis} should help in one situation and stop helping under a clear boundary."
    return f"{axis} should vary by context rather than looking globally high or low."


def _axis_echo_fields(axis: str) -> list[str]:
    lowered = axis.lower()
    if "cogn" in lowered or "abstract" in lowered or "thinking" in lowered:
        return ["cognitive_motivation_profile", "derived_academic_tendency", "values_identity"]
    if "motivation" in lowered or "autonomy" in lowered or "drive" in lowered:
        return ["cognitive_motivation_profile", "values_identity", "social_creative_profile"]
    if "regulation" in lowered or "resilience" in lowered or "recovery" in lowered:
        return ["cognitive_motivation_profile", "mental_health_context", "derived_academic_tendency"]
    return ["cognitive_motivation_profile", "values_identity"]


def _behavior_anchor_text(
    policy_text: str,
    *,
    strongest: str,
    weakest: str,
    quota: str,
) -> str:
    base = policy_text.strip() or "Predict trigger, interpretation, action, and later adjustment."
    return (
        f"{base} Use quota context `{quota}`; make {strongest} visible without erasing "
        f"the weaker signal in {weakest}."
    )


def candidate_slots(
    genome: dict[str, Any],
    n: int,
    seed: int,
) -> list[MegaPersonaSlot]:
    genome = normalize_genome(genome)
    schema_binding = schema_binding_for_genome(genome)
    axis_names = axis_names_for_binding(schema_binding)
    axis_roles = axis_roles_for_binding(schema_binding)
    quota_buckets = tuple(
        QuotaBucket(
            label=bucket.label,
            weight=genome["quota_weights"].get(bucket.label, bucket.weight),
            stage_options=bucket.stage_options,
            motivation_drives=bucket.motivation_drives,
            stress_band=bucket.stress_band,
            social_energy_band=bucket.social_energy_band,
            derived_performance_band=bucket.derived_performance_band,
        )
        for bucket in quota_buckets_for_binding(schema_binding)
    )
    slots = SlotSampler(quota_buckets=quota_buckets, axis_names=axis_names).sample(n=n, seed=seed)
    transformed = []
    for slot in slots:
        axes = {}
        for axis, value in slot.target_axes.items():
            centered = (value - 0.5) * genome["axis_stretch"].get(axis, 1.0)
            axes[axis] = _clip(0.5 + centered + genome["axis_bias"].get(axis, 0.0), 0.0, 1.0)
        constraints = dict(slot.constraints)
        adaptive_constraints = build_adaptive_constraints(axes, constraints, axis_roles=axis_roles)
        transformed.append(
            MegaPersonaSlot(
                slot_id=slot.slot_id,
                quota_label=slot.quota_label,
                target_axes=axes,
                constraints=constraints,
                adaptive_constraints=adaptive_constraints,
            )
        )
    return transformed


def genome_score(
    genome: dict[str, Any],
    schema_fitness: float,
    behavior_coverage: float,
    shadow_alignment: float,
    generation_rate: float,
    internal_consistency: float = 1.0,
) -> float:
    """Multiplicative gated score — same formula as the batch experiment.

    This matches src.mega_persona.experiment.compute_experiment_score.
    The gated product ensures that invalid, near-duplicate, or behaviorally
    collapsed populations receive low scores even when one dimension appears
    strong.
    """
    import numpy as np

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


def _aggregate_seed_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_seed:
        return {}
    numeric_keys = {
        "score": [item["score"] for item in per_seed],
        "schema_fitness": [item["schema_evaluation"]["fitness"] for item in per_seed],
        "validity_rate": [item["schema_evaluation"]["validity_rate"] for item in per_seed],
        "near_duplicate_rate": [item["schema_evaluation"]["near_duplicate_rate"] for item in per_seed],
        "internal_consistency": [
            item.get("internal_consistency", {}).get("mean_score", 0.0) for item in per_seed
        ],
        "internal_consistency_min": [
            item.get("internal_consistency", {}).get("min_score", 0.0) for item in per_seed
        ],
        "axis_alignment": [
            item.get("internal_consistency", {}).get("axis_alignment_mean", 0.0)
            for item in per_seed
        ],
        "axis_target_mae": [
            item.get("internal_consistency", {}).get(
                "axis_target_mae_mean",
                1.0 - item.get("internal_consistency", {}).get("axis_alignment_mean", 0.0),
            )
            for item in per_seed
        ],
        "consistency_issue_rate": [
            item.get("internal_consistency", {}).get("consistency_issue_rate", 0.0)
            for item in per_seed
        ],
        "strict_consistency_error": [
            item.get("internal_consistency", {}).get("strict_consistency_error", 1.0)
            for item in per_seed
        ],
        "train_shadow_alignment": [
            item["train_shadow_behavior"]["overall_alignment"] for item in per_seed
        ],
        "validation_shadow_alignment": [
            item["validation_shadow_behavior"]["overall_alignment"] for item in per_seed
        ],
        "train_shadow_mae": [
            item["train_shadow_behavior"].get(
                "overall_mae",
                1.0 - item["train_shadow_behavior"]["overall_alignment"],
            )
            for item in per_seed
        ],
        "validation_shadow_mae": [
            item["validation_shadow_behavior"].get(
                "overall_mae",
                1.0 - item["validation_shadow_behavior"]["overall_alignment"],
            )
            for item in per_seed
        ],
        "train_behavior_coverage": [
            item["train_behavior_diversity"]["coverage"] for item in per_seed
        ],
        "validation_behavior_coverage": [
            item["validation_behavior_diversity"]["coverage"] for item in per_seed
        ],
        "validation_behavior_avg_dist": [
            item["validation_behavior_diversity"]["avg_dist"] for item in per_seed
        ],
        "validation_behavior_balanced_diversity": [
            _balanced_diversity_score(item["validation_behavior_diversity"])
            for item in per_seed
        ],
        "slot_coverage": [item["slot_diversity"]["coverage"] for item in per_seed],
    }
    return {
        f"{key}.mean": float(np.mean(values))
        for key, values in numeric_keys.items()
    } | {
        f"{key}.std": float(np.std(values))
        for key, values in numeric_keys.items()
    }


def _aggregate_final_test_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_seed:
        return {}
    numeric_keys = {
        "test_schema_fitness": [
            item["schema_evaluation"]["fitness"] for item in per_seed
        ],
        "test_validity_rate": [
            item["schema_evaluation"]["validity_rate"] for item in per_seed
        ],
        "test_near_duplicate_rate": [
            item["schema_evaluation"]["near_duplicate_rate"] for item in per_seed
        ],
        "test_internal_consistency": [
            item.get("internal_consistency", {}).get("mean_score", 0.0) for item in per_seed
        ],
        "test_internal_consistency_min": [
            item.get("internal_consistency", {}).get("min_score", 0.0) for item in per_seed
        ],
        "test_axis_alignment": [
            item.get("internal_consistency", {}).get("axis_alignment_mean", 0.0)
            for item in per_seed
        ],
        "test_axis_target_mae": [
            item.get("internal_consistency", {}).get(
                "axis_target_mae_mean",
                1.0 - item.get("internal_consistency", {}).get("axis_alignment_mean", 0.0),
            )
            for item in per_seed
        ],
        "test_consistency_issue_rate": [
            item.get("internal_consistency", {}).get("consistency_issue_rate", 0.0)
            for item in per_seed
        ],
        "test_strict_consistency_error": [
            item.get("internal_consistency", {}).get("strict_consistency_error", 1.0)
            for item in per_seed
        ],
        "test_shadow_alignment": [
            item["test_shadow_behavior"]["overall_alignment"] for item in per_seed
        ],
        "test_shadow_mae": [
            item["test_shadow_behavior"].get(
                "overall_mae",
                1.0 - item["test_shadow_behavior"]["overall_alignment"],
            )
            for item in per_seed
        ],
        "test_behavior_coverage": [
            item["test_behavior_diversity"]["coverage"] for item in per_seed
        ],
        "test_behavior_avg_dist": [
            item["test_behavior_diversity"]["avg_dist"] for item in per_seed
        ],
        "test_behavior_balanced_diversity": [
            _balanced_diversity_score(item["test_behavior_diversity"])
            for item in per_seed
        ],
    }
    return {
        f"{key}.mean": float(np.mean(values))
        for key, values in numeric_keys.items()
    } | {
        f"{key}.std": float(np.std(values))
        for key, values in numeric_keys.items()
    }


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


def _balanced_diversity_score(metrics: dict[str, float]) -> float:
    """Composite diversity score that avoids rewarding far-apart outliers alone."""
    max_distance = float(np.sqrt(3.0))
    coverage = float(np.clip(metrics.get("coverage", 0.0), 0.0, 1.0))
    avg_dist = float(np.clip(metrics.get("avg_dist", 0.0) / max_distance, 0.0, 1.0))
    min_dist = float(np.clip(metrics.get("min_dist", 0.0) / 0.25, 0.0, 1.0))
    uniformity = float(np.clip(np.exp(metrics.get("kl_divergence", -10.0)), 0.0, 1.0))
    return float(
        0.45 * coverage
        + 0.25 * uniformity
        + 0.20 * avg_dist
        + 0.10 * min_dist
    )


def _filter_simulations_by_surveys(
    simulations: list[ShadowSurveySimulation],
    surveys: list[ShadowSurvey],
) -> list[ShadowSurveySimulation]:
    survey_ids = {survey.survey_id for survey in surveys}
    return [
        simulation
        for simulation in simulations
        if simulation.survey_id in survey_ids
    ]


def _normalize_weights(weights: dict[str, float]) -> None:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        equal = 1.0 / len(weights)
        for key in weights:
            weights[key] = equal
        return
    for key in weights:
        weights[key] = max(weights[key], 0.0) / total


def _child_mutation_mode(child_idx: int) -> str:
    # Interleave conservative and exploratory children so each generation is
    # also a lightweight ablation of what kind of mutation helped.
    modes = ("prompt_only", "operator_only", "mixed", "numeric_only")
    return modes[child_idx % len(modes)]


def _mutation_mode_scale_multiplier(mutation_mode: str) -> float:
    return {
        "prompt_only": 0.55,
        "operator_only": 0.70,
        "mixed": 1.0,
        "numeric_only": 0.80,
    }[mutation_mode]


PROMPT_POLICY_BANK = {
    "mechanism_focus": {
        "balanced": "Explain cognition, motivation, constraints, and social context with balanced emphasis.",
        "cognitive": "Prioritize concrete reasoning mechanisms, evidence preferences, blind spots, and interpretation habits.",
        "motivational": "Prioritize motives, rewards, avoidance patterns, identity protection, and why action starts or stalls.",
        "regulatory": "Prioritize planning, persistence, emotional regulation, recovery, and context-dependent breakdowns.",
        "psychometric": (
            "Prioritize stable response dispositions that can explain repeated survey behavior across related items, "
            "while grounding those dispositions in ordinary choices rather than questionnaire language."
        ),
    },
    "tension_level": {
        "low": "Use subtle tensions; avoid dramatic contradictions unless the slot strongly requires them.",
        "moderate": "Include one clear but plausible internal tension that affects behavior.",
        "high": "Include two interacting tensions, but keep them psychologically coherent and non-caricatured.",
        "bounded": (
            "Show a clear limitation or boundary condition, but also show the student's ordinary adaptation so the "
            "trait remains useful rather than turning into a global flaw."
        ),
    },
    "specificity": {
        "concrete": "Ground traits in specific routines, settings, relationships, and observable choices.",
        "narrative": "Use a strong life-context narrative that makes traits feel causally situated.",
        "behavioral": "Favor behaviorally testable claims that would change survey or simulation responses.",
    },
    "anti_stereotype": {
        "explicit": "Actively avoid stereotypes tied to region, class, age, gender, performance, or stress.",
        "contextual": "When using background context, include both resources and constraints so no trait is deterministic.",
        "counterexample": "Add at least one counter-stereotypical detail that remains plausible for the profile.",
    },
    "axis_binding": {
        "mechanistic": (
            "Bind each primary axis to concrete causal mechanisms: abstraction changes how problems are represented, "
            "autonomy changes what starts action, and self-regulation changes recovery after friction."
        ),
        "orthogonal": (
            "Keep the three primary axes partially independent; avoid making high abstraction automatically imply "
            "high autonomy or high resilience."
        ),
        "contrastive": (
            "For each persona, include at least one axis contrast, such as high curiosity with low planning or "
            "strong persistence with external approval sensitivity."
        ),
    },
    "coverage_strategy": {
        "balanced_space": (
            "Distribute examples across low, mid, and high trait bands; avoid collapsing every trait toward the center."
        ),
        "construct_anchors": (
            "Represent each target construct with two ordinary-life anchors so similar personas can still differ in "
            "survey-relevant behavior without becoming stereotypes."
        ),
        "factorial_probes": (
            "Create small behavior probes that vary the primary axes separately, so coverage comes from combinations "
            "of cognition, motivation, and regulation rather than one overall competence gradient."
        ),
        "edge_cases": (
            "Use plausible edge cases near target-axis boundaries, while preserving non-caricatured constraints and tradeoffs."
        ),
        "within_bucket_variety": (
            "Within the same quota label, vary daily routines, supports, risks, social expression, and failure modes."
        ),
    },
    "behavioral_signal": {
        "survey_predictive": (
            "Write details that would change Likert-style survey answers about curiosity, belonging, stress, recovery, "
            "teacher/peer relations, and creative confidence."
        ),
        "action_predictive": (
            "Ground traits in observable choices under pressure, feedback, deadlines, ambiguity, and group settings."
        ),
        "mixed_evidence": (
            "Include both self-image and behavior evidence, and allow them to diverge in psychologically realistic ways."
        ),
    },
}


EVOLUTION_PROMPT_OPERATORS: tuple[dict[str, Any], ...] = (
    {
        "id": "op01_axis_decoupling",
        "name": "Axis dissociation probes",
        "instruction": (
            "Use axis dissociation probes instead of global competence labels. For every persona, include three short "
            "behavior probes: how they plan an ambiguous task, how they choose under external pressure, and how they "
            "recover after disruption. Each probe must reveal one primary axis while keeping the other two plausible "
            "but not identical, so high abstraction, autonomy, and resilience can vary independently."
        ),
        "profile": {
            "axis_binding": "orthogonal",
            "coverage_strategy": "factorial_probes",
            "behavioral_signal": "action_predictive",
            "specificity": "behavioral",
        },
        "bias": {"cognitive_abstraction": 0.005, "motivation_autonomy": -0.005, "self_regulation_resilience": 0.0},
        "stretch": {"cognitive_abstraction": 1.08, "motivation_autonomy": 1.08, "self_regulation_resilience": 1.08},
        "field_requirements": [
            "Include separate evidence for ambiguous-task planning, external-pressure choice, and disruption recovery.",
            "Do not use one general competence sentence to explain all three primary axes.",
        ],
        "preferred_parent_metric": "coverage_elite",
    },
    {
        "id": "op02_behavioral_evidence",
        "name": "Behavioral evidence checklist",
        "instruction": (
            "For each persona, include concrete evidence for four situations: a deadline, peer pressure, "
            "feedback after failure, and an ambiguous task. These cues should make later simulated behavior inferable."
        ),
        "profile": {"behavioral_signal": "action_predictive", "specificity": "behavioral"},
        "field_requirements": [
            "Each persona must contain evidence for deadline behavior, peer-pressure behavior, feedback behavior, and ambiguous-task behavior.",
        ],
        "preferred_parent_metric": "alignment_elite",
    },
    {
        "id": "op03_shadow_survey_alignment",
        "name": "Cross-survey response anchors",
        "instruction": (
            "Build cross-survey response anchors. For each persona, add one stable anchor for curiosity/learning "
            "approach, one for belonging/help-seeking, and one for stress/recovery. Each anchor must include: "
            "a repeated everyday trigger, the student's usual interpretation, and an observable response. Keep anchors "
            "subtle and natural; never state questionnaire answers, but make related survey responses inferable."
        ),
        "profile": {
            "behavioral_signal": "survey_predictive",
            "specificity": "behavioral",
            "mechanism_focus": "psychometric",
            "axis_binding": "mechanistic",
            "coverage_strategy": "construct_anchors",
        },
        "quota": {"belonging_oriented_collaborator": 0.01, "curious_low_structure": 0.01},
        "bias": {"cognitive_abstraction": 0.005, "motivation_autonomy": 0.0, "self_regulation_resilience": 0.005},
        "behavior_anchors": {
            "learning": "Repeated trigger: ambiguous instruction or unfamiliar task. Usual interpretation: whether uncertainty is interesting, risky, or annoying. Observable response: asks a clarifying question, tests a small example, postpones, or copies a model.",
            "belonging": "Repeated trigger: group work, teacher feedback, or comparison with peers. Usual interpretation: whether help feels safe, exposing, or useful. Observable response: contributes early, waits to be invited, seeks one trusted person, or withdraws.",
            "stress_recovery": "Repeated trigger: missed deadline, poor feedback, or social friction. Usual interpretation: temporary setback versus proof of inadequacy. Observable response: recovery latency, coping routine, help-seeking, or avoidance.",
        },
        "consistency_rules": [
            "Survey-relevant anchors must be echoed across at least two fields, but never phrased as direct questionnaire answers.",
            "If an anchor predicts high curiosity, it must also specify whether confidence and persistence are high, mid, or low.",
        ],
        "preferred_parent_metric": "alignment_elite",
    },
    {
        "id": "op04_within_bucket_contrast",
        "name": "Within-bucket contrast",
        "instruction": (
            "Within the same quota bucket, force contrast on resources, daily routine, support network, risk exposure, "
            "and success definition so personas do not become template variants."
        ),
        "profile": {"coverage_strategy": "within_bucket_variety", "anti_stereotype": "contextual"},
        "quota": {"curious_low_structure": 0.02, "reserved_resilient_observer": 0.02},
        "field_requirements": [
            "Within the same quota label, vary at least one context lever and one behavior lever.",
        ],
        "preferred_parent_metric": "coverage_elite",
    },
    {
        "id": "op05_failure_recovery_cycle",
        "name": "Failure recovery cycle",
        "instruction": (
            "Describe one failure cycle with five parts: trigger, appraisal, coping attempt, short-term outcome, "
            "and later adjustment. The cycle must affect self-regulation or help-seeking."
        ),
        "profile": {"mechanism_focus": "regulatory", "behavioral_signal": "action_predictive"},
        "bias": {"self_regulation_resilience": 0.02},
        "behavior_anchors": {
            "stress_recovery": "Failure cycle anchor: trigger, appraisal, coping attempt, short-term consequence, and later adjustment must be visible without melodrama.",
        },
        "preferred_parent_metric": "consistency_elite",
    },
    {
        "id": "op06_low_axis_fidelity",
        "name": "Low-axis fidelity",
        "instruction": (
            "When a target axis is low, show exactly one realistic cost and one compensating workaround. "
            "The cost must affect a deadline, help-seeking moment, or ambiguous task; the workaround may help "
            "but must not erase the low-axis signal."
        ),
        "profile": {
            "coverage_strategy": "edge_cases",
            "behavioral_signal": "action_predictive",
            "specificity": "behavioral",
        },
        "stretch": {"cognitive_abstraction": 1.06, "motivation_autonomy": 1.06, "self_regulation_resilience": 1.06},
        "field_requirements": [
            "For any low target axis, preserve one visible behavioral cost and one partial workaround.",
        ],
        "preferred_parent_metric": "schema_elite",
    },
    {
        "id": "op07_high_axis_cost",
        "name": "High-axis boundary probes",
        "instruction": (
            "For high-axis personas, use boundary probes rather than generic costs. Show where the high trait works, "
            "where it stops working, and the concrete adjustment that prevents collapse. Examples: abstraction helps "
            "diagnose mistakes but must be simplified under time pressure; autonomy resists approval seeking but still "
            "negotiates constraints; resilience restarts routines but does not erase fatigue or delay."
        ),
        "profile": {
            "tension_level": "bounded",
            "axis_binding": "mechanistic",
            "behavioral_signal": "action_predictive",
            "mechanism_focus": "balanced",
        },
        "bias": {"cognitive_abstraction": 0.01, "motivation_autonomy": 0.01, "self_regulation_resilience": 0.01},
        "stretch": {"cognitive_abstraction": 1.02, "motivation_autonomy": 1.02, "self_regulation_resilience": 1.02},
        "consistency_rules": [
            "A high axis must have a boundary condition, not a generic flaw.",
            "The boundary must not erase the useful side of the high trait.",
        ],
        "preferred_parent_metric": "consistency_elite",
    },
    {
        "id": "op08_validation_conservatism",
        "name": "Validation conservatism",
        "instruction": (
            "Make diversity schema-safe: vary one concrete context lever and one behavior lever while keeping "
            "timeline, demographics, field lengths, and cross-agent facts consistent. Avoid decorative novelty "
            "that does not change behavior."
        ),
        "profile": {
            "anti_stereotype": "contextual",
            "tension_level": "moderate",
            "coverage_strategy": "within_bucket_variety",
            "specificity": "concrete",
        },
        "stretch": {"cognitive_abstraction": 1.02, "motivation_autonomy": 1.02, "self_regulation_resilience": 1.02},
        "repair_policy": {
            "length": "Trim decorative biography first; preserve causal behavior evidence.",
            "conflict": "Resolve contradictions by adding context boundaries rather than deleting the target trait.",
        },
        "preferred_parent_metric": "schema_elite",
    },
    {
        "id": "op09_low_high_axis_tradeoff",
        "name": "Low-high axis tradeoff",
        "instruction": (
            "Pair one low-axis cost with one high-axis cost in the same persona. The contrast must be visible in two "
            "different behavioral settings, such as planning alone versus acting with peers."
        ),
        "profile": {
            "axis_binding": "contrastive",
            "behavioral_signal": "mixed_evidence",
            "tension_level": "high",
        },
        "stretch": {
            "cognitive_abstraction": 1.08,
            "motivation_autonomy": 1.08,
            "self_regulation_resilience": 1.08,
        },
        "consistency_rules": [
            "At least one low-axis and one high-axis signal must coexist through a plausible context boundary.",
        ],
        "preferred_parent_metric": "diversity_elite",
    },
    {
        "id": "op10_contextual_bucket_split",
        "name": "Contextual bucket split",
        "instruction": (
            "Within each quota bucket, split personas by context rather than labels: vary family bandwidth, institutional "
            "support, peer norms, and time scarcity while keeping the same target axes measurable."
        ),
        "profile": {
            "coverage_strategy": "within_bucket_variety",
            "anti_stereotype": "contextual",
            "specificity": "concrete",
        },
        "quota": {
            "belonging_oriented_collaborator": 0.02,
            "externally_driven_performer": -0.01,
            "self_directed_builder": 0.01,
        },
        "field_requirements": [
            "Context variation must change predicted behavior, not only background description.",
        ],
        "preferred_parent_metric": "coverage_elite",
    },
    {
        "id": "op11_decision_trace_evidence",
        "name": "Decision trace evidence",
        "instruction": (
            "For one consequential decision, include a short trace of options considered, rejected alternatives, "
            "emotional friction, and the final action. The trace must reveal abstraction, autonomy, and regulation."
        ),
        "profile": {
            "mechanism_focus": "cognitive",
            "behavioral_signal": "action_predictive",
            "specificity": "behavioral",
        },
        "bias": {"cognitive_abstraction": 0.015},
        "behavior_anchors": {
            "learning": "Decision trace anchor: options considered, rejected alternative, friction, and final action must reveal how evidence is handled.",
        },
        "preferred_parent_metric": "alignment_elite",
    },
    {
        "id": "op12_support_network_asymmetry",
        "name": "Support network asymmetry",
        "instruction": (
            "Make support networks asymmetric: one domain has reliable support while another domain has weak, delayed, "
            "or conditional support. Show how this changes help-seeking and recovery behavior."
        ),
        "profile": {
            "coverage_strategy": "within_bucket_variety",
            "mechanism_focus": "regulatory",
            "behavioral_signal": "survey_predictive",
        },
        "bias": {"self_regulation_resilience": 0.015},
        "behavior_anchors": {
            "belonging": "Support asymmetry anchor: one domain has reliable support and another has delayed or conditional support; help-seeking changes by domain.",
        },
        "preferred_parent_metric": "alignment_elite",
    },
    {
        "id": "op13_autonomy_pressure_test",
        "name": "Autonomy pressure test",
        "instruction": (
            "Expose motivation autonomy under pressure: include one moment where external approval, obligation, or "
            "authority conflicts with personal interest, then show the chosen compromise or refusal."
        ),
        "profile": {
            "mechanism_focus": "motivational",
            "behavioral_signal": "action_predictive",
            "axis_binding": "mechanistic",
        },
        "bias": {"motivation_autonomy": 0.02},
        "stretch": {"motivation_autonomy": 1.10},
        "behavior_anchors": {
            "motivation": "Autonomy pressure anchor: approval, obligation, or authority conflicts with personal interest; final behavior shows compromise, compliance, or refusal.",
        },
        "preferred_parent_metric": "alignment_elite",
    },
    {
        "id": "op14_recovery_latency",
        "name": "Recovery latency",
        "instruction": (
            "Represent resilience through recovery latency rather than optimism: specify how long disruption lasts, "
            "what gets neglected during recovery, and which routine eventually stabilizes behavior."
        ),
        "profile": {
            "mechanism_focus": "regulatory",
            "behavioral_signal": "mixed_evidence",
            "specificity": "behavioral",
        },
        "stretch": {"self_regulation_resilience": 1.14},
        "behavior_anchors": {
            "stress_recovery": "Recovery latency anchor: duration of disruption, neglected routine, stabilizing routine, and residual cost.",
        },
        "preferred_parent_metric": "consistency_elite",
    },
    {
        "id": "op15_survey_discriminating_cues",
        "name": "Construct contrast without leakage",
        "instruction": (
            "Separate nearby latent constructs without leaking survey answers. Use two naturalistic contrasts: one "
            "between curiosity and competence confidence, and one between belonging, approval compliance, or recovery. "
            "Each contrast must be shown through everyday behavior frequency, trigger context, and internal rationale; "
            "avoid phrasing that sounds like a Likert item."
        ),
        "profile": {
            "behavioral_signal": "survey_predictive",
            "specificity": "behavioral",
            "axis_binding": "mechanistic",
            "anti_stereotype": "contextual",
        },
        "stretch": {
            "cognitive_abstraction": 1.02,
            "motivation_autonomy": 1.02,
            "self_regulation_resilience": 1.02,
        },
        "behavior_anchors": {
            "learning": "Discriminating cue: separate curiosity from confidence by showing whether the student explores despite uncertainty or only when already competent.",
            "belonging": "Discriminating cue: separate belonging from approval compliance by showing whether peer/teacher presence changes honest effort or only outward agreement.",
        },
        "field_requirements": [
            "Do not leak survey item wording; use frequency, trigger context, and rationale instead.",
        ],
        "preferred_parent_metric": "alignment_elite",
    },
    {
        "id": "op16_v3_blueprint_binding",
        "name": "Genome v3 blueprint binding",
        "instruction": (
            "Use the Genome v3 blueprint as the primary mutation target. Strengthen the per-slot blueprint so each "
            "persona has: one strongest-weakest axis tension, one low/mid/high expression rule for every primary axis, "
            "four behavior prediction anchors (ambiguous task, peer pressure, failure feedback, deadline), and explicit "
            "cross-agent echoes from cognition to values, social behavior, and mental health. The goal is not subtle "
            "style change; the goal is a diagnosable mechanism change that should be visible in persona text and "
            "shadow-survey behavior. Encourage scientifically plausible exploration: make behavior consequences "
            "specific enough to be measured, but do not force any single metric to dominate the mutation."
        ),
        "profile": {
            "mechanism_focus": "psychometric",
            "behavioral_signal": "survey_predictive",
            "specificity": "behavioral",
            "axis_binding": "mechanistic",
            "coverage_strategy": "factorial_probes",
            "tension_level": "bounded",
        },
        "blueprint_policy": {
            "core_tension_rule": (
                "For every slot, choose the strongest target axis and weakest target axis, then express them as one "
                "ordinary school-life tension that all agents must echo."
            ),
            "axis_evidence_rule": (
                "Each primary axis must have a trigger, interpretation, action, and boundary; use different settings "
                "so axes do not collapse into one general competence gradient."
            ),
            "quota_context_rule": (
                "Use quota bucket only to choose context and resources; never let the bucket become a stereotype or "
                "replace the target-axis mechanism."
            ),
            "persona_coherence_rule": (
                "All five agents must describe one causal person through repeated behavior anchors, not separate profiles."
            ),
        },
        "axis_expression_policy": {
            "low": (
                "Show one visible behavioral cost, one partial workaround, and one situation where the workaround fails."
            ),
            "mid": (
                "Show a stable preference whose execution changes with audience, task clarity, or time pressure."
            ),
            "high": (
                "Show the trait helping first, then a specific boundary where overuse or context reduces its value."
            ),
        },
        "cross_agent_binding_policy": {
            "cognition_to_values": (
                "Values must explain why the cognition pattern is protected, revised, or hidden under pressure."
            ),
            "cognition_to_social": (
                "Social behavior must reuse the same ambiguity handling and help-seeking threshold from cognition."
            ),
            "regulation_to_health": (
                "Mental health must reuse recovery latency, coping routine, and residual cost implied by self-regulation."
            ),
            "values_to_behavior": (
                "Moral tension must predict one peer-pressure, deadline, or feedback behavior."
            ),
        },
        "behavior_prediction_policy": {
            "ambiguous_task": (
                "Predict first move, evidence preference, help-seeking threshold, and whether action starts before certainty."
            ),
            "peer_pressure": (
                "Predict conformity, negotiation, performance, withdrawal, or one-trusted-ally behavior."
            ),
            "failure_feedback": (
                "Predict appraisal, emotional latency, repair action, and what changes on the next attempt."
            ),
            "deadline": (
                "Predict planning rhythm, shortcut risk, and recovery behavior when the schedule slips."
            ),
        },
        "critic_policy": {
            "axis_echo": "Each primary axis must echo across at least two schema sections through behavior, not labels.",
            "behavior_echo": (
                "Ambiguous task, peer pressure, failure feedback, and deadline behavior must be inferable without "
                "using survey item wording."
            ),
            "measurement_check": (
                "Every behavior anchor should be concrete enough that an evaluator could infer a likely survey "
                "response direction, while still allowing novel but plausible combinations of traits."
            ),
            "contradiction_check": (
                "Cross-field contradictions are allowed only when an explicit context boundary explains the shift."
            ),
            "length_check": "Trim decorative biography before removing behavior-predictive evidence.",
        },
        "field_requirements": [
            "Every persona must make the strongest-weakest axis tension visible in at least two fields.",
            "Every persona must include behavior-predictive evidence for ambiguity, peer pressure, feedback, and deadline contexts.",
        ],
        "consistency_rules": [
            "Cognition-to-values, cognition-to-social, and regulation-to-health echoes must all be visible.",
            "If a field contradicts the blueprint, add a context boundary rather than deleting the target-axis signal.",
            "Behavior anchors should be observable and falsifiable, not merely decorative personality prose.",
        ],
        "preferred_parent_metric": "strict_consistency_elite",
    },
    {
        "id": "op17_v3_axis_coverage_grid",
        "name": "Genome v3 axis coverage grid",
        "instruction": (
            "Mutate the Genome v3 blueprint toward coverage. Make low, mid, and high expressions of each primary axis "
            "visibly different across slots, and avoid collapsing cognitive abstraction, motivation autonomy, and "
            "self-regulation resilience into one generic good-student gradient. Exploration should be broad but still "
            "realistic: introduce new axis combinations through concrete school-life contexts rather than extreme labels."
        ),
        "profile": {
            "mechanism_focus": "psychometric",
            "behavioral_signal": "mixed_evidence",
            "specificity": "behavioral",
            "axis_binding": "factorial",
            "coverage_strategy": "axis_grid",
            "tension_level": "bounded",
        },
        "blueprint_policy": {
            "axis_evidence_rule": (
                "For each slot, map every primary axis to a distinct trigger, interpretation, behavior, and boundary; "
                "do not reuse the same behavior as evidence for multiple axes."
            ),
            "core_tension_rule": (
                "Choose axis tensions that cover uncommon but plausible combinations, such as high cognition with low "
                "autonomy, or high autonomy with fragile regulation."
            ),
            "quota_context_rule": (
                "Use quota buckets to vary resources, setting, and support context while preserving the target axis grid."
            ),
        },
        "axis_expression_policy": {
            "low": "Make the trait visibly constrain behavior in one ordinary task and one pressure task.",
            "mid": "Show selective expression: the trait appears in one context but not another.",
            "high": "Show strong expression plus a clear boundary, tradeoff, or blind spot.",
        },
        "field_requirements": [
            "Every persona must show at least one non-obvious axis combination without turning it into pathology.",
            "Axis evidence must be behaviorally separable across cognition, motivation, and regulation.",
        ],
        "consistency_rules": [
            "A wide axis combination is valid only if context makes it plausible.",
            "Do not repair low-axis signals into hidden excellence; keep the target contrast visible.",
        ],
        "preferred_parent_metric": "coverage_elite",
    },
    {
        "id": "op18_v3_behavior_alignment_probes",
        "name": "Genome v3 behavior alignment probes",
        "instruction": (
            "Mutate the Genome v3 blueprint toward stronger behavior prediction. For each persona, make the likely "
            "response to ambiguous tasks, peer pressure, failure feedback, and deadlines inferable from mechanism, not "
            "from survey-like wording. Encourage exploration by varying the causal route from trait to behavior rather "
            "than forcing a single high-score profile."
        ),
        "profile": {
            "mechanism_focus": "behavioral",
            "behavioral_signal": "survey_predictive",
            "specificity": "behavioral",
            "axis_binding": "mechanistic",
            "coverage_strategy": "scenario_probes",
            "tension_level": "moderate",
        },
        "behavior_prediction_policy": {
            "ambiguous_task": (
                "Predict whether the first move is decomposing, asking, copying examples, delaying, or trial testing; "
                "tie it to cognition and autonomy."
            ),
            "peer_pressure": (
                "Predict conformity, negotiation, quiet resistance, performance, or ally-seeking; tie it to values and belonging."
            ),
            "failure_feedback": (
                "Predict appraisal, affect duration, repair strategy, and future avoidance or persistence; tie it to regulation."
            ),
            "deadline": (
                "Predict planning rhythm, shortcut risk, help-seeking, and recovery after slippage; tie it to autonomy and regulation."
            ),
        },
        "critic_policy": {
            "behavior_echo": (
                "Each behavior probe must be inferable from at least two schema sections and should not read like a direct survey answer."
            ),
            "measurement_check": (
                "A reviewer should be able to predict response direction for shadow-survey items while still seeing natural uncertainty."
            ),
        },
        "field_requirements": [
            "Every long narrative field must include one behavior that would change a shadow-survey answer.",
            "Behavior probes must include trigger, interpretation, action, and aftereffect.",
        ],
        "preferred_parent_metric": "shadow_mae_elite",
    },
    {
        "id": "op19_v3_cross_field_coherence",
        "name": "Genome v3 cross-field coherence",
        "instruction": (
            "Mutate the Genome v3 blueprint toward internal coherence. Strengthen causal links across cognition, "
            "motivation, values, social behavior, and mental health so the output reads as one person seen from several "
            "angles. Allow tensions and context-dependent behavior, but require the boundary condition that explains them."
        ),
        "profile": {
            "mechanism_focus": "integrative",
            "behavioral_signal": "mixed_evidence",
            "specificity": "causal",
            "axis_binding": "mechanistic",
            "coverage_strategy": "coherence_first",
            "tension_level": "bounded",
        },
        "blueprint_policy": {
            "persona_coherence_rule": (
                "Before generation, define one causal loop connecting cognition, motivation, regulation, social behavior, "
                "and health; every section must echo part of that loop."
            ),
            "core_tension_rule": (
                "Use one central tension repeatedly instead of adding unrelated contradictions."
            ),
        },
        "cross_agent_binding_policy": {
            "cognition_to_values": "Values must explain why the thinking style is protected, hidden, challenged, or revised.",
            "cognition_to_social": "Social behavior must reuse the same uncertainty tolerance and help-seeking threshold.",
            "regulation_to_health": "Health must reuse recovery latency, coping routine, and residual cost from regulation.",
            "values_to_behavior": "Moral or identity tension must predict one visible action under peer, deadline, or feedback pressure.",
        },
        "critic_policy": {
            "contradiction_check": (
                "Flag any cross-field mismatch unless a concrete boundary condition explains when the behavior changes."
            ),
            "axis_echo": "Each primary axis must appear through repeated behavior, not repeated labels.",
        },
        "consistency_rules": [
            "Contradictions must be resolved by context boundaries, not by averaging the persona into blandness.",
            "At least three sections must reuse the same coping routine, help-seeking rule, or ambiguity-handling pattern.",
        ],
        "preferred_parent_metric": "strict_consistency_elite",
    },
    {
        "id": "op20_v3_realistic_novelty",
        "name": "Genome v3 realistic novelty",
        "instruction": (
            "Mutate the Genome v3 blueprint toward realistic diversity. Seek non-template personas by varying resources, "
            "routines, peer ecology, learning contexts, and coping pathways, while keeping each combination plausible. "
            "The aim is novelty with causal support, not exotic biography or random trait extremes."
        ),
        "profile": {
            "mechanism_focus": "ecological",
            "behavioral_signal": "mixed_evidence",
            "specificity": "situated",
            "axis_binding": "contextual",
            "coverage_strategy": "novel_contexts",
            "tension_level": "moderate",
        },
        "blueprint_policy": {
            "quota_context_rule": (
                "Use quota buckets to choose varied life contexts, resource constraints, social ecology, and institutional expectations."
            ),
            "axis_evidence_rule": (
                "Let the same axis value manifest differently across contexts, so diversity comes from mechanism-by-context interaction."
            ),
        },
        "axis_expression_policy": {
            "low": "Show a concrete workaround that helps sometimes and fails under a named context.",
            "mid": "Show unstable or selective expression across audience, task type, or support availability.",
            "high": "Show a useful strength that creates a different cost in a second context.",
        },
        "behavior_anchors": {
            "learning": (
                "Novelty anchor: vary where learning happens, who observes it, and what evidence feels trustworthy."
            ),
            "stress_recovery": (
                "Novelty anchor: vary recovery pathway through routine, avoidance, social support, reframing, or environmental change."
            ),
        },
        "field_requirements": [
            "Every persona must include one context-specific detail that changes behavior without becoming decorative biography.",
            "Avoid reusing the same successful-student template across low, mid, and high axes.",
        ],
        "preferred_parent_metric": "diversity_elite",
    },
    {
        "id": "op21_v3_schema_precision",
        "name": "Genome v3 schema precision",
        "instruction": (
            "Mutate the Genome v3 blueprint toward schema-safe precision. Improve parseability, bounded length, and field "
            "specificity while preserving behavioral evidence. Exploration should happen through concise causal details, "
            "not through longer prose or extra unsupported attributes."
        ),
        "profile": {
            "mechanism_focus": "structural",
            "behavioral_signal": "action_predictive",
            "specificity": "concise",
            "axis_binding": "mechanistic",
            "coverage_strategy": "schema_safe",
            "tension_level": "bounded",
        },
        "blueprint_policy": {
            "persona_coherence_rule": (
                "Use compact repeated anchors so every schema field supports the same causal person without bloated narrative."
            ),
            "axis_evidence_rule": (
                "Each axis needs only the minimum sufficient trigger, interpretation, behavior, and boundary."
            ),
        },
        "critic_policy": {
            "length_check": (
                "Remove decorative biography, redundant adjectives, and unsupported diagnoses before removing behavior-predictive evidence."
            ),
            "measurement_check": (
                "Every retained sentence should help schema validity, causal coherence, or behavior prediction."
            ),
        },
        "field_requirements": [
            "Prefer short causal sentences over long biography.",
            "Do not add fields or values outside the MegaPersona schema.",
            "Keep all long narrative fields behavior-predictive and length-safe.",
        ],
        "consistency_rules": [
            "Schema safety should preserve the target axis signal rather than smoothing it away.",
            "Conciseness is good only when the behavior mechanism remains inferable.",
        ],
        "preferred_parent_metric": "schema_elite",
    },
    {
        "id": "op22_v4_probe_rewire",
        "name": "Genome v4 probe reassignment",
        "instruction": (
            "Reassign exactly one primary axis to a different observable scenario probe so axis-to-behavior "
            "alignment can be tested without changing any other generation mechanism."
        ),
        "v4_module": "probe_assignment",
        "preferred_parent_metric": "shadow_mae_elite",
    },
    {
        "id": "op23_v4_signal_calibrate",
        "name": "Genome v4 axis signal calibration",
        "instruction": (
            "Change exactly one axis realization mode or signal strength to improve target-axis calibration."
        ),
        "v4_module": "axis_realization",
        "preferred_parent_metric": "axis_target_elite",
    },
    {
        "id": "op24_v4_interaction_rewire",
        "name": "Genome v4 axis interaction rewiring",
        "instruction": (
            "Change only the relationship between strongest and weakest axes to explore separable, compensatory, "
            "or context-dependent behavior combinations."
        ),
        "v4_module": "interaction_mode",
        "preferred_parent_metric": "coverage_elite",
    },
    {
        "id": "op25_v4_echo_graph_rewire",
        "name": "Genome v4 cross-field echo rewiring",
        "instruction": (
            "Change only the deterministic cross-field evidence graph to improve causal consistency across sections."
        ),
        "v4_module": "echo_graph",
        "preferred_parent_metric": "strict_consistency_elite",
    },
    {
        "id": "op26_v4_context_diversify",
        "name": "Genome v4 context modulation",
        "instruction": (
            "Change only how context modulates the same axis mechanism to seek realistic behavioral diversity."
        ),
        "v4_module": "context_modulation",
        "preferred_parent_metric": "diversity_elite",
    },
    {
        "id": "op27_v4_repair_calibrate",
        "name": "Genome v4 repair calibration",
        "instruction": (
            "Change only evidence density or repair priority to improve schema-safe generation while preserving "
            "behavioral evidence."
        ),
        "v4_module": "repair_control",
        "preferred_parent_metric": "schema_elite",
    },
)


def _mutate_prompt_profile(
    profile: dict[str, str],
    rng: np.random.Generator,
    mutation_scale: float,
) -> None:
    changed = False
    mutation_probability = min(0.55, 0.18 + mutation_scale)
    for category, bank in PROMPT_POLICY_BANK.items():
        if rng.random() > mutation_probability:
            continue
        options = list(bank.keys())
        current = profile.get(category)
        choices = [option for option in options if option != current] or options
        profile[category] = str(rng.choice(choices))
        changed = True

    if not changed and rng.random() < 0.35:
        category = str(rng.choice(list(PROMPT_POLICY_BANK.keys())))
        options = list(PROMPT_POLICY_BANK[category].keys())
        current = profile.get(category)
        choices = [option for option in options if option != current] or options
        profile[category] = str(rng.choice(choices))


STRUCTURAL_MUTATION_SNIPPETS = {
    "field_requirements": (
        "Every axis claim must be backed by one routine behavior and one pressure behavior.",
        "Make the strongest trait useful in one context and bounded in another.",
        "Separate self-image from observed behavior when they plausibly diverge.",
        "Use context boundaries to explain contrast rather than treating conflict as inconsistency.",
    ),
    "consistency_rules": (
        "If motivation is externally driven, values and social behavior must show where approval matters.",
        "If regulation is low, health and learning fields must show recovery latency or avoidance cost.",
        "If belonging is high, support-seeking must be specific rather than universally confident.",
        "If curiosity is high, distinguish exploration from performance confidence.",
    ),
    "behavior_anchors": {
        "learning": (
            "Anchor learning in one repeated trigger, one interpretation of uncertainty, and one observable study behavior.",
            "Show whether curiosity persists when competence, time, or peer comparison is uncertain.",
        ),
        "motivation": (
            "Anchor motivation in what starts action: interest, obligation, approval, identity, or avoidance.",
            "Show how the student behaves when personal interest conflicts with external expectations.",
        ),
        "belonging": (
            "Anchor belonging in a help-seeking or group-work moment with a clear safety appraisal.",
            "Show whether social contact changes honest effort, outward agreement, or withdrawal.",
        ),
        "stress_recovery": (
            "Anchor recovery in disruption duration, coping routine, neglected task, and residual cost.",
            "Show whether pressure creates mobilization, freezing, avoidance, or delayed repair.",
        ),
    },
    "blueprint_policy": {
        "core_tension_rule": (
            "Choose the strongest-lowest axis pair as a concrete daily tension, then bind every agent to it.",
            "Use quota context only to choose setting; the causal tension must come from target axes.",
        ),
        "axis_evidence_rule": (
            "Require each primary axis to appear as a trigger, interpretation, action, and boundary.",
            "Use different evidence settings for each axis so the axes do not collapse into one competence score.",
        ),
    },
    "axis_expression_policy": {
        "low": (
            "Show one cost, one workaround, and one situation where the workaround fails.",
            "Keep low expression visible through latency, avoidance, or narrow strategy rather than global deficit.",
        ),
        "mid": (
            "Make mid expression conditional on audience, task clarity, or time pressure.",
            "Show a stable preference with inconsistent execution under one predictable context shift.",
        ),
        "high": (
            "Show strength, overuse risk, and a specific boundary that preserves realism.",
            "Show where the high trait helps first but creates a tradeoff later.",
        ),
    },
    "cross_agent_binding_policy": {
        "cognition_to_values": (
            "Values must explicitly justify why the thinking pattern is protected, hidden, or revised.",
            "Values should turn one cognitive blind spot into a moral or identity tension.",
        ),
        "regulation_to_health": (
            "Health narrative must reuse the same recovery latency implied by self-regulation.",
            "Coping style must show the consequence of planning rhythm under deadline pressure.",
        ),
    },
    "behavior_prediction_policy": {
        "ambiguous_task": (
            "Predict first move, evidence source, help-seeking threshold, and whether action starts before certainty.",
            "Separate curiosity from confidence by showing what the student does when instructions are incomplete.",
        ),
        "failure_feedback": (
            "Predict appraisal, emotional latency, repair action, and whether the next attempt changes strategy.",
            "Show whether feedback is read as information, threat, proof, or social comparison.",
        ),
    },
    "critic_policy": {
        "axis_echo": (
            "Reject profiles where an axis appears in only one field or only as a label.",
            "Require each primary axis to echo across at least two schema sections.",
        ),
        "behavior_echo": (
            "Reject profiles where deadline, peer, ambiguity, or feedback behavior cannot be inferred.",
            "Require behavior anchors to be inferable without mentioning survey items.",
        ),
    },
}


def _mutate_structural_genome(
    genome: dict[str, Any],
    rng: np.random.Generator,
    mutation_scale: float,
) -> None:
    probability = min(0.70, 0.22 + mutation_scale)
    for key in ("field_requirements", "consistency_rules"):
        if rng.random() > probability:
            continue
        snippets = STRUCTURAL_MUTATION_SNIPPETS[key]
        selected = str(rng.choice(snippets))
        current = list(genome.get(key, default_genome()[key]))
        if selected not in current:
            current.append(selected)
        genome[key] = current[-8:]

    if rng.random() <= probability:
        anchor_key = str(rng.choice(list(STRUCTURAL_MUTATION_SNIPPETS["behavior_anchors"].keys())))
        snippets = STRUCTURAL_MUTATION_SNIPPETS["behavior_anchors"][anchor_key]
        genome.setdefault("behavior_anchors", json.loads(json.dumps(default_genome()["behavior_anchors"])))
        genome["behavior_anchors"][anchor_key] = str(rng.choice(snippets))

    v3_keys = (
        "blueprint_policy",
        "axis_expression_policy",
        "cross_agent_binding_policy",
        "behavior_prediction_policy",
        "critic_policy",
    )
    if rng.random() <= probability:
        field = str(rng.choice(v3_keys))
        field_snippets = STRUCTURAL_MUTATION_SNIPPETS[field]
        entry_key = str(rng.choice(list(field_snippets.keys())))
        snippets = field_snippets[entry_key]
        genome.setdefault(field, json.loads(json.dumps(default_genome()[field])))
        genome[field][entry_key] = str(rng.choice(snippets))


def _select_evolution_operator(
    rng: np.random.Generator,
    operator_id: str | None = None,
) -> dict[str, Any]:
    if operator_id is not None:
        for operator in EVOLUTION_PROMPT_OPERATORS:
            if operator["id"] == operator_id:
                return dict(operator)
        raise ValueError(f"unknown evolution operator id: {operator_id}")
    operators = tuple(operator for operator in EVOLUTION_PROMPT_OPERATORS if "_v4_" not in operator["id"])
    return dict(operators[int(rng.integers(0, len(operators)))])


def _select_v4_operator(
    rng: np.random.Generator,
    operator_id: str | None = None,
) -> dict[str, Any]:
    operators = tuple(operator for operator in EVOLUTION_PROMPT_OPERATORS if "_v4_" in operator["id"])
    if operator_id is not None:
        for operator in operators:
            if operator["id"] == operator_id:
                return dict(operator)
        raise ValueError(f"unknown Genome v4 operator id: {operator_id}")
    return dict(operators[int(rng.integers(0, len(operators)))])


def _apply_evolution_operator(
    genome: dict[str, Any],
    operator: dict[str, Any],
    rng: np.random.Generator,
    mutation_scale: float,
    apply_numeric: bool = True,
) -> None:
    genome.update(normalize_genome(genome))
    schema_binding = schema_binding_for_genome(genome)
    axis_names = set(axis_names_for_binding(schema_binding))
    axis_roles = axis_roles_for_binding(schema_binding)
    profile = genome.setdefault(
        "prompt_profile",
        json.loads(json.dumps(default_genome()["prompt_profile"])),
    )
    for category, choice in operator.get("profile", {}).items():
        if category in PROMPT_POLICY_BANK and choice in PROMPT_POLICY_BANK[category]:
            profile[category] = choice
    _merge_structural_operator_fields(genome, operator)
    _merge_structural_operator_fields(genome, _v3_operator_defaults(operator))

    if not apply_numeric:
        return

    for axis, factor in operator.get("stretch", {}).items():
        resolved = _resolve_axis_name(axis, axis_names, axis_roles)
        if resolved in genome.get("axis_stretch", {}):
            jitter = rng.normal(0.0, mutation_scale * 0.04)
            genome["axis_stretch"][resolved] = _clip(genome["axis_stretch"][resolved] * float(factor) + jitter, 0.55, 1.75)

    for axis, delta in operator.get("bias", {}).items():
        resolved = _resolve_axis_name(axis, axis_names, axis_roles)
        if resolved in genome.get("axis_bias", {}):
            jitter = rng.normal(0.0, mutation_scale * 0.02)
            genome["axis_bias"][resolved] = _clip(genome["axis_bias"][resolved] + float(delta) + jitter, -0.35, 0.35)

    for label, delta in operator.get("quota", {}).items():
        if label in genome.get("quota_weights", {}):
            genome["quota_weights"][label] = _clip(genome["quota_weights"][label] + float(delta), 0.02, 0.6)
    if operator.get("quota"):
        _normalize_weights(genome["quota_weights"])


def _merge_structural_operator_fields(genome: dict[str, Any], operator: dict[str, Any]) -> None:
    for key in (
        "agent_focus",
        "behavior_anchors",
        "repair_policy",
        "blueprint_policy",
        "axis_expression_policy",
        "cross_agent_binding_policy",
        "behavior_prediction_policy",
        "critic_policy",
    ):
        if isinstance(operator.get(key), dict):
            base = genome.setdefault(key, json.loads(json.dumps(default_genome()[key])))
            base.update(_clean_string_dict(operator[key], base))
    for key in ("field_requirements", "consistency_rules"):
        if isinstance(operator.get(key), list):
            existing = genome.setdefault(key, json.loads(json.dumps(default_genome()[key])))
            merged = list(existing) + [
                item for item in _bounded_string_list(
                    operator[key],
                    fallback=[],
                    max_items=8,
                    max_chars=180,
                )
                if item not in existing
            ]
            genome[key] = merged[-8:]


def _v3_operator_defaults(operator: dict[str, Any]) -> dict[str, Any]:
    operator_id = str(operator.get("id", ""))
    preferred = str(operator.get("preferred_parent_metric", ""))
    if "alignment" in preferred or "shadow" in operator_id or operator_id in {"op02_behavioral_evidence"}:
        return {
            "behavior_prediction_policy": {
                "ambiguous_task": "Predict first move, evidence preference, and help-seeking threshold in a survey-inferable way.",
                "peer_pressure": "Predict conformity, negotiation, withdrawal, or trusted-ally behavior without naming survey answers.",
                "failure_feedback": "Predict appraisal, emotion latency, repair action, and future strategy change.",
                "deadline": "Predict planning rhythm, shortcut risk, and what happens when the schedule slips.",
            },
            "critic_policy": {
                "behavior_echo": "Behavior anchors must be inferable across learning, belonging, stress, and creative-confidence items.",
            },
        }
    if "coverage" in preferred or "axis" in operator_id or "bucket" in operator_id:
        return {
            "blueprint_policy": {
                "axis_evidence_rule": "Make each primary axis use a different behavior setting so coverage is not one competence gradient.",
            },
            "axis_expression_policy": {
                "low": "Show low expression through one visible cost, one partial workaround, and one failed workaround boundary.",
                "mid": "Show mid expression as context-dependent rather than generic average behavior.",
                "high": "Show high expression through a useful strength and a non-generic boundary condition.",
            },
        }
    if "consistency" in preferred or "recovery" in operator_id or "failure" in operator_id:
        return {
            "cross_agent_binding_policy": {
                "regulation_to_health": "Mental health must reuse recovery latency, coping routine, and residual cost implied by self-regulation.",
                "cognition_to_values": "Values must turn the central cognitive pattern into a plausible identity or moral tension.",
            },
            "critic_policy": {
                "contradiction_check": "Contradictions are acceptable only when a context boundary explains the behavior shift.",
            },
        }
    if "schema" in preferred or "validation" in operator_id or "low_axis" in operator_id:
        return {
            "blueprint_policy": {
                "persona_coherence_rule": "Keep the blueprint short, schema-safe, and focused on causal echoes across fields.",
            },
            "critic_policy": {
                "length_check": "Preserve causal behavior evidence while trimming decorative biography before validation.",
            },
        }
    return {
        "blueprint_policy": {
            "core_tension_rule": "Choose one strongest-weakest axis tension and bind all agent outputs to it.",
        }
    }


def _clean_string_dict(
    value: dict[str, Any],
    fallback: dict[str, str],
    *,
    max_chars: int = 260,
) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, default in fallback.items():
        raw = value.get(key, default)
        if not isinstance(raw, str):
            raw = str(default)
        raw = " ".join(raw.split())
        cleaned[key] = raw[:max_chars].rstrip()
    return cleaned


def _bounded_string_list(
    value: list[Any],
    *,
    fallback: list[str],
    max_items: int,
    max_chars: int,
) -> list[str]:
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())[:max_chars].rstrip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    if cleaned:
        return cleaned
    return list(fallback[:max_items])


def _profile_choice(profile: dict[str, str], category: str, default: str) -> str:
    choice = profile.get(category, default)
    if choice not in PROMPT_POLICY_BANK[category]:
        return default
    return choice


def _resolve_axis_name(
    key: str,
    axis_names: set[str],
    axis_roles: dict[str, str],
) -> str | None:
    if key in axis_names:
        return key
    mapped = axis_roles.get(key)
    if mapped in axis_names:
        return mapped
    legacy_role = {
        "cognitive_abstraction": "cognitive_core",
        "motivation_autonomy": "motivation_core",
        "self_regulation_resilience": "regulation_core",
    }.get(key)
    if legacy_role:
        resolved = axis_roles.get(legacy_role)
        if resolved in axis_names:
            return resolved
    return None


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _config_to_dict(config: MegaEvolutionConfig) -> dict[str, Any]:
    payload = {**asdict(config), "seeds": list(config.seeds)}
    payload["num_islands"] = config.population_size
    payload["population_size_deprecated"] = payload.pop("population_size")
    return payload


def _normalize_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(config)
    # Checkpoints created before repeat-aware selection are single-draw runs.
    payload.setdefault("candidate_evaluation_repeats", 1)
    payload.setdefault("elite_confirmation_repeats", 1)
    payload.setdefault("persona_temperature", 0.45)
    payload.setdefault("persona_top_p", 1.0)
    payload.setdefault("simulator_temperature", 0.25)
    payload.setdefault("simulator_top_p", 1.0)
    if "num_islands" not in payload and "population_size" in payload:
        payload["num_islands"] = payload["population_size"]
    if "population_size_deprecated" not in payload and "population_size" in payload:
        payload["population_size_deprecated"] = payload["population_size"]
    payload.pop("population_size", None)
    return payload


def build_run_manifest(
    config: MegaEvolutionConfig,
    argv: list[str] | None = None,
    resume: bool = False,
    mutator_model_key: str | None = None,
    mutator_model: str | None = None,
    mutator_api_base: str | None = None,
    mutator_api_key_env: str | None = None,
    model_key: str | None = None,
    llm_provider: str | None = None,
    persona_model: str | None = None,
    persona_api_base: str | None = None,
    persona_api_key_env: str | None = None,
    simulator_model_key: str | None = None,
    simulator_model: str | None = None,
    simulator_api_base: str | None = None,
    simulator_api_key_env: str | None = None,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "argv": list(argv or sys.argv),
        "resume": resume,
        "llm_provider": llm_provider,
        "mutator_model_key": mutator_model_key,
        "mutator_model": mutator_model,
        "mutator_api_base": mutator_api_base,
        "mutator_api_key_env": mutator_api_key_env,
        "model_key": model_key,
        "persona_model": persona_model,
        "persona_api_base": persona_api_base,
        "persona_api_key_env": persona_api_key_env,
        "simulator_model_key": simulator_model_key,
        "simulator_model": simulator_model,
        "simulator_api_base": simulator_api_base,
        "simulator_api_key_env": simulator_api_key_env,
        "config": _config_to_dict(config),
        "git": _git_manifest(),
    }


def _git_manifest() -> dict[str, Any]:
    return {
        "commit": _git_command(["git", "rev-parse", "HEAD"]),
        "branch": _git_command(["git", "branch", "--show-current"]),
        "dirty": bool(_git_command(["git", "status", "--short"])),
        "status_short": _git_command(["git", "status", "--short"]),
    }


def _git_command(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _final_markdown(payload: dict[str, Any]) -> str:
    best = payload["best"]
    metrics = best.get("metrics", {})
    lines = [
        "# MegaPersona Open-Evolve Summary",
        "",
        f"- best candidate: `{best['candidate_id']}`",
        f"- best fitness: `{best.get('fitness', 0.0):.4f}`",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for metric in [
        "score",
        "schema_fitness",
        "validity_rate",
        "near_duplicate_rate",
        "internal_consistency",
        "internal_consistency_min",
        "axis_alignment",
        "train_shadow_alignment",
        "validation_shadow_alignment",
        "train_behavior_coverage",
        "validation_behavior_coverage",
        "slot_coverage",
    ]:
        lines.append(
            f"| {metric} | {metrics.get(metric + '.mean', 0.0):.4f} | "
            f"{metrics.get(metric + '.std', 0.0):.4f} |"
        )
    final_test = payload.get("final_test_report") or {}
    test_metrics = final_test.get("metrics", {})
    if test_metrics:
        lines.extend(
            [
                "",
                "## Sealed Test Report",
                "",
                "| Metric | Mean | Std |",
                "|---|---:|---:|",
            ]
        )
        for metric in [
            "test_schema_fitness",
            "test_validity_rate",
            "test_near_duplicate_rate",
            "test_internal_consistency",
            "test_internal_consistency_min",
            "test_axis_alignment",
            "test_shadow_alignment",
            "test_behavior_coverage",
            "test_behavior_avg_dist",
            "test_behavior_balanced_diversity",
        ]:
            lines.append(
                f"| {metric} | {test_metrics.get(metric + '.mean', 0.0):.4f} | "
                f"{test_metrics.get(metric + '.std', 0.0):.4f} |"
            )
        lines.extend(
            [
                "",
                f"- test used for selection: `{final_test.get('test_used_for_selection', False)}`",
            ]
        )
    return "\n".join(lines) + "\n"
