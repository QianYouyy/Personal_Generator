"""Durable Open-Evolve style optimization for MegaPersona experiments.

The evolvable surface is deliberately narrow and JSON-serializable. The fixed
MegaPersona architecture stays stable while evolution tunes sampling strategy,
axis transforms, and shadow-survey selection. The score aggregation weights are
fixed so candidates cannot improve by changing the ruler used to judge them.
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
from src.mega_persona.evaluation import evaluate_mega_personas
from src.mega_persona.generator import MegaPersonaGenerationResult, MegaPersonaGenerator
from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_simulator import (
    LLMShadowSimulator,
    aggregate_shadow_behavior,
    shadow_behavior_axis_matrix,
)
from src.mega_persona.shadow_survey import (
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
    build_adaptive_constraints,
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
    shadow_max_workers: int = 1


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
    """Persistent evolutionary optimizer for MegaPersona candidate genomes."""

    def __init__(
        self,
        config: MegaEvolutionConfig,
        output_dir: Path,
        resume: bool = False,
        llm_client=None,
        simulator_llm_client=None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.store = EvolutionStore(output_dir)
        self.rng = np.random.default_rng(config.random_seed)
        self.llm_client = llm_client
        self.simulator_llm_client = simulator_llm_client
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
            raise ValueError("simulator_llm_client is required (LLMShadowSimulator is the only simulator)")

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
            self.survey_splits = build_shadow_survey_splits(
                train_surveys=self.config.shadow_surveys,
                validation_surveys=self.config.validation_shadow_surveys,
                test_surveys=self.config.test_shadow_surveys,
                items_per_survey=self.config.items_per_shadow_survey,
                seed=self.config.survey_seed,
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
        logger.info("Starting evolution loop target_generations=%s", self.config.generations)
        while self.generation < self.config.generations:
            logger.info("Generation %s: evaluating population", self.generation)
            self._evaluate_population()
            self._write_generation_summary()
            best = self.best_candidate()
            logger.info(
                "Generation %s complete: best=%s fitness=%.4f",
                self.generation,
                best.candidate_id,
                best.fitness or 0.0,
            )
            if self.generation >= self.config.generations:
                break
            self.population = self._next_generation()
            self.generation += 1
            self._save_checkpoint()

        logger.info("Final population evaluation at generation %s", self.generation)
        self._evaluate_population()
        self._write_generation_summary()
        self._save_checkpoint()
        best = self.best_candidate()
        logger.info(
            "Selected best candidate=%s validation_fitness=%.4f; running sealed test",
            best.candidate_id,
            best.fitness or 0.0,
        )
        final_test_report = self.evaluate_final_test(best)
        self.store.write_final_test_report(final_test_report)
        self.store.write_final_summary(best, self.population, self.config, final_test_report)
        logger.info("Evolution finished. Final artifacts written to %s", self.output_dir)
        return best

    def best_candidate(self) -> MegaEvolutionCandidate:
        evaluated = [candidate for candidate in self.population if candidate.evaluated]
        if not evaluated:
            raise RuntimeError("No evaluated candidates yet")
        return max(evaluated, key=lambda candidate: candidate.fitness or float("-inf"))

    def _initial_population(self) -> list[MegaEvolutionCandidate]:
        baseline = MegaEvolutionCandidate(
            candidate_id="candidate_baseline",
            genome=default_genome(),
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
        slots = candidate_slots(candidate.genome, n=self.config.n, seed=seed)
        generation_results = self._generate_persona_results(candidate, slots)
        personas = [result.persona for result in generation_results if result.persona is not None]
        logger.info(
            "Candidate=%s seed=%s: generated_personas=%s/%s",
            candidate.candidate_id,
            seed,
            len(personas),
            len(slots),
        )
        schema_evaluation = evaluate_mega_personas(
            personas,
            coverage_radius=self.config.coverage_radius,
            duplicate_threshold=self.config.duplicate_threshold,
        )
        simulator = LLMShadowSimulator(
            self.simulator_llm_client,
            max_workers=self.config.shadow_max_workers,
        )
        logger.info("Candidate=%s seed=%s: simulating train shadow surveys", candidate.candidate_id, seed)
        train_simulations = simulator.simulate_population(personas, list(self.survey_splits.train))
        logger.info("Candidate=%s seed=%s: simulating validation shadow surveys", candidate.candidate_id, seed)
        validation_simulations = simulator.simulate_population(personas, list(self.survey_splits.validation))
        train_shadow_behavior = aggregate_shadow_behavior(personas, train_simulations)
        validation_shadow_behavior = aggregate_shadow_behavior(personas, validation_simulations)
        train_behavior_diversity = _diversity_for_matrix(
            shadow_behavior_axis_matrix(personas, train_simulations),
            self.config.coverage_radius,
        )
        validation_behavior_diversity = _diversity_for_matrix(
            shadow_behavior_axis_matrix(personas, validation_simulations),
            self.config.coverage_radius,
        )
        slot_diversity = _diversity_for_matrix(
            np.array([slot.axis_vector() for slot in slots], dtype=float),
            self.config.coverage_radius,
        )
        seed_score = genome_score(
            genome=candidate.genome,
            schema_fitness=schema_evaluation.fitness,
            behavior_coverage=validation_behavior_diversity.get("coverage", 0.0),
            shadow_alignment=validation_shadow_behavior.overall_alignment,
            generation_rate=len(personas) / len(slots) if slots else 0.0,
        )
        logger.info(
            "Candidate=%s seed=%s: score=%.4f schema=%.4f val_cov=%.4f val_align=%.4f",
            candidate.candidate_id,
            seed,
            seed_score,
            schema_evaluation.fitness,
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
        """Evaluate the selected best candidate on the sealed test split once."""
        best_result = self.store.find_candidate_result(best.candidate_id)
        if best_result is None:
            raise FileNotFoundError(f"missing stored evaluation for best candidate {best.candidate_id}")

        successful_seed_results = [
            seed_result
            for seed_result in best_result.get("per_seed", [])
            if seed_result.get("status") == "ok" and seed_result.get("personas")
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

        simulator = LLMShadowSimulator(
            self.simulator_llm_client,
            max_workers=self.config.shadow_max_workers,
        )
        per_seed = []
        for seed_result in successful_seed_results:
            logger.info(
                "Final sealed test candidate=%s seed=%s",
                best.candidate_id,
                seed_result["seed"],
            )
            personas = [
                MegaPersona.model_validate(persona)
                for persona in seed_result.get("personas", [])
            ]
            test_simulations = simulator.simulate_population(personas, list(self.survey_splits.test))
            test_shadow_behavior = aggregate_shadow_behavior(personas, test_simulations)
            test_behavior_diversity = _diversity_for_matrix(
                shadow_behavior_axis_matrix(personas, test_simulations),
                self.config.coverage_radius,
            )
            per_seed.append(
                {
                    "seed": seed_result["seed"],
                    "candidate_id": best.candidate_id,
                    "test_shadow_behavior": test_shadow_behavior.to_dict(),
                    "test_behavior_diversity": test_behavior_diversity,
                    "test_shadow_simulations": [
                        asdict(simulation) for simulation in test_simulations
                    ],
                }
            )

        metrics = _aggregate_final_test_metrics(per_seed)
        logger.info("Final sealed test metrics: %s", metrics)
        return {
            "candidate_id": best.candidate_id,
            "candidate_fitness": best.fitness,
            "selection_metric": "validation",
            "test_used_for_selection": False,
            "survey_hashes": self.survey_hashes,
            "metrics": metrics,
            "per_seed": per_seed,
            "created_at": datetime.now().isoformat(),
        }

    def _generate_persona_results(
        self,
        candidate: MegaEvolutionCandidate,
        slots: list[MegaPersonaSlot],
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
            prompt_addendum=prompt_addendum_from_genome(candidate.genome),
        )
        return generator.generate_from_slots(slots)

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
        current_config = _config_to_dict(self.config)
        ignored = {"generations", "max_workers", "shadow_max_workers"}
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
        for path in sorted(self.evaluations_dir.glob("eval_*_*/result.json")):
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
    return {
        "quota_weights": {
            bucket.label: bucket.weight
            for bucket in DEFAULT_QUOTA_BUCKETS
        },
        "axis_bias": {axis: 0.0 for axis in AXIS_NAMES},
        "axis_stretch": {axis: 1.0 for axis in AXIS_NAMES},
        "prompt_profile": {
            "mechanism_focus": "balanced",
            "tension_level": "moderate",
            "specificity": "concrete",
            "anti_stereotype": "explicit",
            "axis_binding": "mechanistic",
            "coverage_strategy": "balanced_space",
            "behavioral_signal": "survey_predictive",
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
        "train_shadow_behavior": zero_shadow,
        "validation_shadow_behavior": zero_shadow,
        "train_behavior_diversity": zero_diversity,
        "validation_behavior_diversity": zero_diversity,
        "slot_diversity": zero_diversity,
        "shadow_survey_hashes": {},
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


def _empty_shadow_behavior() -> dict[str, Any]:
    return {
        "sample_size": 0,
        "survey_count": 0,
        "axis_names": list(AXIS_NAMES),
        "behavior_axis_mean": {axis: 0.0 for axis in AXIS_NAMES},
        "persona_behavior_mae": {axis: 0.0 for axis in AXIS_NAMES},
        "overall_alignment": 0.0,
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
    mutated = json.loads(json.dumps(genome))
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

        for axis in AXIS_NAMES:
            mutated["axis_bias"][axis] = _clip(
                mutated["axis_bias"][axis] + rng.normal(0.0, mutation_scale * 0.15),
                -0.35,
                0.35,
            )
            mutated["axis_stretch"][axis] = _clip(
                mutated["axis_stretch"][axis] + rng.normal(0.0, mutation_scale * 0.25),
                0.55,
                1.75,
            )

    mutated.setdefault(
        "prompt_profile",
        json.loads(json.dumps(default_genome()["prompt_profile"])),
    )
    if prompt_profile_mutation:
        _mutate_prompt_profile(mutated["prompt_profile"], rng, mutation_scale)
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


def prompt_addendum_from_genome(genome: dict[str, Any]) -> str:
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
    operator = genome.get("last_evolution_operator")
    if isinstance(operator, dict) and operator.get("instruction"):
        parts.append(
            "Selected evolution operator "
            f"{operator.get('id', 'unknown')}: {operator['instruction']}"
        )
    return "\n".join(parts)


def candidate_slots(
    genome: dict[str, Any],
    n: int,
    seed: int,
) -> list[MegaPersonaSlot]:
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
        for bucket in DEFAULT_QUOTA_BUCKETS
    )
    slots = SlotSampler(quota_buckets=quota_buckets).sample(n=n, seed=seed)
    transformed = []
    for slot in slots:
        axes = {}
        for axis, value in slot.target_axes.items():
            centered = (value - 0.5) * genome["axis_stretch"].get(axis, 1.0)
            axes[axis] = _clip(0.5 + centered + genome["axis_bias"].get(axis, 0.0), 0.0, 1.0)
        constraints = dict(slot.constraints)
        adaptive_constraints = build_adaptive_constraints(axes, constraints)
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
    return float(schema_fitness * behavior_gate * alignment_gate * generation_rate)


def _aggregate_seed_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_seed:
        return {}
    numeric_keys = {
        "score": [item["score"] for item in per_seed],
        "schema_fitness": [item["schema_evaluation"]["fitness"] for item in per_seed],
        "validity_rate": [item["schema_evaluation"]["validity_rate"] for item in per_seed],
        "near_duplicate_rate": [item["schema_evaluation"]["near_duplicate_rate"] for item in per_seed],
        "train_shadow_alignment": [
            item["train_shadow_behavior"]["overall_alignment"] for item in per_seed
        ],
        "validation_shadow_alignment": [
            item["validation_shadow_behavior"]["overall_alignment"] for item in per_seed
        ],
        "train_behavior_coverage": [
            item["train_behavior_diversity"]["coverage"] for item in per_seed
        ],
        "validation_behavior_coverage": [
            item["validation_behavior_diversity"]["coverage"] for item in per_seed
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
        "test_shadow_alignment": [
            item["test_shadow_behavior"]["overall_alignment"] for item in per_seed
        ],
        "test_behavior_coverage": [
            item["test_behavior_diversity"]["coverage"] for item in per_seed
        ],
        "test_behavior_avg_dist": [
            item["test_behavior_diversity"]["avg_dist"] for item in per_seed
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
    },
    "tension_level": {
        "low": "Use subtle tensions; avoid dramatic contradictions unless the slot strongly requires them.",
        "moderate": "Include one clear but plausible internal tension that affects behavior.",
        "high": "Include two interacting tensions, but keep them psychologically coherent and non-caricatured.",
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
        "name": "Axis decoupling",
        "instruction": (
            "Create explicit counter-axis evidence: at least one high/low contrast among abstraction, autonomy, "
            "and resilience must be visible in behavior. Do not let one high axis imply all axes are high."
        ),
        "profile": {"axis_binding": "orthogonal", "coverage_strategy": "edge_cases"},
        "stretch": {"cognitive_abstraction": 1.08, "motivation_autonomy": 1.04, "self_regulation_resilience": 1.08},
    },
    {
        "id": "op02_behavioral_evidence",
        "name": "Behavioral evidence checklist",
        "instruction": (
            "For each persona, include concrete evidence for four situations: a deadline, peer pressure, "
            "feedback after failure, and an ambiguous task. These cues should make later simulated behavior inferable."
        ),
        "profile": {"behavioral_signal": "action_predictive", "specificity": "behavioral"},
    },
    {
        "id": "op03_shadow_survey_alignment",
        "name": "Shadow survey alignment",
        "instruction": (
            "Encode answer-relevant cues for curiosity, belonging, stress recovery, autonomy, and creative confidence "
            "without stating survey answers directly. Each cue must be grounded in an observable choice."
        ),
        "profile": {"behavioral_signal": "survey_predictive", "specificity": "concrete"},
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
    },
    {
        "id": "op06_low_axis_fidelity",
        "name": "Low-axis fidelity",
        "instruction": (
            "When a target axis is low, show a concrete cost in behavior and do not rescue it with generic competence. "
            "Low values should remain plausible but measurably visible."
        ),
        "profile": {"coverage_strategy": "edge_cases", "behavioral_signal": "mixed_evidence"},
        "stretch": {"cognitive_abstraction": 1.12, "motivation_autonomy": 1.12, "self_regulation_resilience": 1.12},
    },
    {
        "id": "op07_high_axis_cost",
        "name": "High-axis cost",
        "instruction": (
            "When an axis is high, include its behavioral cost: overthinking, stubborn autonomy, over-control, "
            "fatigue, isolation, or conflict with belonging. High values should not become idealized."
        ),
        "profile": {"tension_level": "high", "axis_binding": "contrastive"},
    },
    {
        "id": "op08_validation_conservatism",
        "name": "Validation conservatism",
        "instruction": (
            "Prefer small coherent differences over decorative novelty. Keep field lengths, timeline, demographics, "
            "and cross-agent facts consistent while preserving measurable diversity."
        ),
        "profile": {"anti_stereotype": "contextual", "tension_level": "low"},
        "stretch": {"cognitive_abstraction": 0.98, "motivation_autonomy": 0.98, "self_regulation_resilience": 0.98},
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


def _select_evolution_operator(
    rng: np.random.Generator,
    operator_id: str | None = None,
) -> dict[str, Any]:
    if operator_id is not None:
        for operator in EVOLUTION_PROMPT_OPERATORS:
            if operator["id"] == operator_id:
                return dict(operator)
        raise ValueError(f"unknown evolution operator id: {operator_id}")
    return dict(EVOLUTION_PROMPT_OPERATORS[int(rng.integers(0, len(EVOLUTION_PROMPT_OPERATORS)))])


def _apply_evolution_operator(
    genome: dict[str, Any],
    operator: dict[str, Any],
    rng: np.random.Generator,
    mutation_scale: float,
    apply_numeric: bool = True,
) -> None:
    profile = genome.setdefault(
        "prompt_profile",
        json.loads(json.dumps(default_genome()["prompt_profile"])),
    )
    for category, choice in operator.get("profile", {}).items():
        if category in PROMPT_POLICY_BANK and choice in PROMPT_POLICY_BANK[category]:
            profile[category] = choice

    if not apply_numeric:
        return

    for axis, factor in operator.get("stretch", {}).items():
        if axis in genome.get("axis_stretch", {}):
            jitter = rng.normal(0.0, mutation_scale * 0.04)
            genome["axis_stretch"][axis] = _clip(genome["axis_stretch"][axis] * float(factor) + jitter, 0.55, 1.75)

    for axis, delta in operator.get("bias", {}).items():
        if axis in genome.get("axis_bias", {}):
            jitter = rng.normal(0.0, mutation_scale * 0.02)
            genome["axis_bias"][axis] = _clip(genome["axis_bias"][axis] + float(delta) + jitter, -0.35, 0.35)

    for label, delta in operator.get("quota", {}).items():
        if label in genome.get("quota_weights", {}):
            genome["quota_weights"][label] = _clip(genome["quota_weights"][label] + float(delta), 0.02, 0.6)
    if operator.get("quota"):
        _normalize_weights(genome["quota_weights"])


def _profile_choice(profile: dict[str, str], category: str, default: str) -> str:
    choice = profile.get(category, default)
    if choice not in PROMPT_POLICY_BANK[category]:
        return default
    return choice


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _config_to_dict(config: MegaEvolutionConfig) -> dict[str, Any]:
    return {**asdict(config), "seeds": list(config.seeds)}


def build_run_manifest(
    config: MegaEvolutionConfig,
    argv: list[str] | None = None,
    resume: bool = False,
    model_key: str | None = None,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "argv": list(argv or sys.argv),
        "resume": resume,
        "model_key": model_key,
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
            "test_shadow_alignment",
            "test_behavior_coverage",
            "test_behavior_avg_dist",
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
