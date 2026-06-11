"""Durable Open-Evolve style optimization for MegaPersona experiments.

The evolvable surface is deliberately narrow and JSON-serializable. The fixed
MegaPersona architecture stays stable while evolution tunes sampling strategy,
axis transforms, and shadow-survey selection. The score aggregation weights are
fixed so candidates cannot improve by changing the ruler used to judge them.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.evaluator.metrics import DiversityMetrics
from src.mega_persona.evaluation import evaluate_mega_personas
from src.mega_persona.generator import MegaPersonaGenerator
from src.mega_persona.shadow_simulator import (
    RuleBasedShadowSimulator,
    aggregate_shadow_behavior,
    shadow_behavior_axis_matrix,
)
from src.mega_persona.shadow_survey import build_initial_shadow_surveys
from src.mega_persona.slots import (
    AXIS_NAMES,
    DEFAULT_QUOTA_BUCKETS,
    MegaPersonaSlot,
    QuotaBucket,
    SlotSampler,
    build_adaptive_constraints,
)
from src.mega_persona.template_generator import RuleBasedMegaPersonaBuilder


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
    heldout_shadow_surveys: int = 4
    items_per_shadow_survey: int = 12
    shadow_noise: float = 0.08
    heldout_seed_offset: int = 10000
    random_seed: int = 1234
    max_workers: int = 1


FIXED_SCORE_WEIGHTS = {
    "schema": 0.55,
    "behavior_coverage": 0.25,
    "shadow_alignment": 0.20,
}


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
    ):
        self.config = config
        self.output_dir = output_dir
        self.store = EvolutionStore(output_dir)
        self.rng = np.random.default_rng(config.random_seed)
        self.llm_client = llm_client
        self.generation = 0
        self.population: list[MegaEvolutionCandidate] = []
        self.evaluation_count = 0
        self.best_candidate_id: str | None = None

        if config.generator_mode not in {"mock", "llm"}:
            raise ValueError("generator_mode must be 'mock' or 'llm'")
        if config.generator_mode == "llm" and llm_client is None:
            raise ValueError("llm_client is required when generator_mode='llm'")

        if resume:
            self._load_checkpoint()
        else:
            self.store.initialize()
            self.population = self._initial_population()
            self._save_checkpoint()

    def run(self) -> MegaEvolutionCandidate:
        while self.generation < self.config.generations:
            self._evaluate_population()
            self._write_generation_summary()
            if self.generation >= self.config.generations:
                break
            self.population = self._next_generation()
            self.generation += 1
            self._save_checkpoint()

        self._evaluate_population()
        self._write_generation_summary()
        self._save_checkpoint()
        best = self.best_candidate()
        self.store.write_final_summary(best, self.population, self.config)
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
            return
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
            return self.evaluate_candidate(candidate)
        except Exception as exc:
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

    def evaluate_candidate(self, candidate: MegaEvolutionCandidate) -> dict[str, Any]:
        per_seed = []
        for seed in self.config.seeds:
            slots = candidate_slots(candidate.genome, n=self.config.n, seed=seed)
            personas = self._generate_personas(candidate, slots)
            schema_evaluation = evaluate_mega_personas(
                personas,
                coverage_radius=self.config.coverage_radius,
                duplicate_threshold=self.config.duplicate_threshold,
            )
            train_surveys = build_initial_shadow_surveys(
                num_surveys=self.config.shadow_surveys,
                items_per_survey=self.config.items_per_shadow_survey,
                seed=seed + int(candidate.genome["shadow_survey_seed_offset"]),
            )
            heldout_surveys = build_initial_shadow_surveys(
                num_surveys=self.config.heldout_shadow_surveys,
                items_per_survey=self.config.items_per_shadow_survey,
                seed=(
                    seed
                    + self.config.heldout_seed_offset
                    + int(candidate.genome["shadow_survey_seed_offset"])
                ),
            )
            train_simulations = RuleBasedShadowSimulator(
                noise=self.config.shadow_noise,
                seed=seed,
            ).simulate_population(personas, train_surveys)
            heldout_simulations = RuleBasedShadowSimulator(
                noise=self.config.shadow_noise,
                seed=seed + self.config.heldout_seed_offset,
            ).simulate_population(personas, heldout_surveys)
            train_shadow_behavior = aggregate_shadow_behavior(personas, train_simulations)
            heldout_shadow_behavior = aggregate_shadow_behavior(personas, heldout_simulations)
            train_behavior_diversity = _diversity_for_matrix(
                shadow_behavior_axis_matrix(personas, train_simulations),
                self.config.coverage_radius,
            )
            heldout_behavior_diversity = _diversity_for_matrix(
                shadow_behavior_axis_matrix(personas, heldout_simulations),
                self.config.coverage_radius,
            )
            slot_diversity = _diversity_for_matrix(
                np.array([slot.axis_vector() for slot in slots], dtype=float),
                self.config.coverage_radius,
            )
            seed_score = genome_score(
                genome=candidate.genome,
                schema_fitness=schema_evaluation.fitness,
                behavior_coverage=heldout_behavior_diversity.get("coverage", 0.0),
                shadow_alignment=heldout_shadow_behavior.overall_alignment,
                generation_rate=len(personas) / len(slots) if slots else 0.0,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "score": seed_score,
                    "slots": [asdict(slot) for slot in slots],
                    "personas": [persona.model_dump() for persona in personas],
                    "schema_evaluation": schema_evaluation.to_dict(),
                    "train_shadow_behavior": train_shadow_behavior.to_dict(),
                    "heldout_shadow_behavior": heldout_shadow_behavior.to_dict(),
                    "train_behavior_diversity": train_behavior_diversity,
                    "heldout_behavior_diversity": heldout_behavior_diversity,
                    "slot_diversity": slot_diversity,
                    "train_shadow_simulations": [
                        asdict(simulation) for simulation in train_simulations
                    ],
                    "heldout_shadow_simulations": [
                        asdict(simulation) for simulation in heldout_simulations
                    ],
                }
            )

        fitness = float(np.mean([seed_result["score"] for seed_result in per_seed]))
        metrics = _aggregate_seed_metrics(per_seed)
        return {
            "candidate": candidate.to_dict(),
            "fitness": fitness,
            "metrics": metrics,
            "per_seed": per_seed,
        }

    def _generate_personas(
        self,
        candidate: MegaEvolutionCandidate,
        slots: list[MegaPersonaSlot],
    ):
        if self.config.generator_mode == "mock":
            return RuleBasedMegaPersonaBuilder().build_population(slots)
        generator = MegaPersonaGenerator(
            self.llm_client,
            prompt_addendum=prompt_addendum_from_genome(candidate.genome),
        )
        results = generator.generate_from_slots(slots)
        return [result.persona for result in results if result.persona is not None]

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
            scale = 0.22 + 0.04 * min(self.generation, 6)
            next_population.append(
                self._mutate_candidate(
                    parent,
                    generation=self.generation + 1,
                    mutation_scale=scale,
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
    ) -> MegaEvolutionCandidate:
        genome = mutate_genome(parent.genome, self.rng, mutation_scale)
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

    def _validate_resume_config(self, state: dict[str, Any]) -> None:
        saved_config = state.get("config")
        if not saved_config:
            return
        current_config = _config_to_dict(self.config)
        ignored = {"generations", "max_workers"}
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
        self.checkpoint_path = output_dir / "checkpoint.json"

    def initialize(self) -> None:
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

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

    def write_final_summary(
        self,
        best: MegaEvolutionCandidate,
        population: list[MegaEvolutionCandidate],
        config: MegaEvolutionConfig,
    ) -> None:
        self.initialize()
        payload = {
            "config": _config_to_dict(config),
            "best": best.to_dict(),
            "population": [candidate.to_dict() for candidate in population],
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
        "shadow_survey_seed_offset": 0,
        "prompt_profile": {
            "mechanism_focus": "balanced",
            "tension_level": "moderate",
            "specificity": "concrete",
            "anti_stereotype": "explicit",
        },
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


def mutate_genome(
    genome: dict[str, Any],
    rng: np.random.Generator,
    mutation_scale: float,
) -> dict[str, Any]:
    mutated = json.loads(json.dumps(genome))
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

    mutated["shadow_survey_seed_offset"] = int(
        max(0, mutated["shadow_survey_seed_offset"] + rng.integers(-2, 4))
    )
    mutated.setdefault(
        "prompt_profile",
        json.loads(json.dumps(default_genome()["prompt_profile"])),
    )
    _mutate_prompt_profile(mutated["prompt_profile"], rng)
    return mutated


def prompt_addendum_from_genome(genome: dict[str, Any]) -> str:
    profile = genome.get("prompt_profile", {})
    mechanism_focus = _profile_choice(profile, "mechanism_focus", "balanced")
    tension_level = _profile_choice(profile, "tension_level", "moderate")
    specificity = _profile_choice(profile, "specificity", "concrete")
    anti_stereotype = _profile_choice(profile, "anti_stereotype", "explicit")
    return "\n".join(
        [
            PROMPT_POLICY_BANK["mechanism_focus"][mechanism_focus],
            PROMPT_POLICY_BANK["tension_level"][tension_level],
            PROMPT_POLICY_BANK["specificity"][specificity],
            PROMPT_POLICY_BANK["anti_stereotype"][anti_stereotype],
        ]
    )


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
    weighted = (
        FIXED_SCORE_WEIGHTS["schema"] * schema_fitness
        + FIXED_SCORE_WEIGHTS["behavior_coverage"] * behavior_coverage
        + FIXED_SCORE_WEIGHTS["shadow_alignment"] * shadow_alignment
    )
    return float(max(0.0, weighted) * generation_rate)


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
        "heldout_shadow_alignment": [
            item["heldout_shadow_behavior"]["overall_alignment"] for item in per_seed
        ],
        "train_behavior_coverage": [
            item["train_behavior_diversity"]["coverage"] for item in per_seed
        ],
        "heldout_behavior_coverage": [
            item["heldout_behavior_diversity"]["coverage"] for item in per_seed
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
}


def _mutate_prompt_profile(profile: dict[str, str], rng: np.random.Generator) -> None:
    if rng.random() > 0.45:
        return
    category = rng.choice(list(PROMPT_POLICY_BANK.keys()))
    options = list(PROMPT_POLICY_BANK[category].keys())
    current = profile.get(category)
    choices = [option for option in options if option != current] or options
    profile[category] = str(rng.choice(choices))


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
        "heldout_shadow_alignment",
        "train_behavior_coverage",
        "heldout_behavior_coverage",
        "slot_coverage",
    ]:
        lines.append(
            f"| {metric} | {metrics.get(metric + '.mean', 0.0):.4f} | "
            f"{metrics.get(metric + '.std', 0.0):.4f} |"
        )
    return "\n".join(lines) + "\n"
