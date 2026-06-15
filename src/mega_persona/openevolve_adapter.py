"""Adapter that runs MegaPersona genomes through the OpenEvolve engine."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.mega_persona.evolution import (
    MegaEvolutionCandidate,
    MegaEvolutionConfig,
    MegaPersonaEvolver,
    build_run_manifest,
    default_genome,
    mutate_genome,
)
from src.open_evolve.engine import OpenEvolve


logger = logging.getLogger(__name__)


def genome_to_code(genome: dict[str, Any]) -> str:
    """Serialize a MegaPersona genome as OpenEvolve's evolvable code string."""
    return json.dumps(genome, ensure_ascii=False, sort_keys=True, indent=2)


def genome_from_code(code: str) -> dict[str, Any]:
    """Parse OpenEvolve's code string back into a MegaPersona genome."""
    try:
        genome = json.loads(code)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenEvolve candidate is not a MegaPersona genome JSON: {exc}") from exc
    if not isinstance(genome, dict):
        raise ValueError("OpenEvolve candidate genome must be a JSON object")
    return genome


def genome_hash(genome: dict[str, Any]) -> str:
    encoded = genome_to_code(genome).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MegaGenomeMutator:
    """OpenEvolve mutator for the fixed-architecture MegaPersona genome."""

    def __init__(
        self,
        random_seed: int = 1234,
        base_mutation_scale: float = 0.12,
    ):
        self.rng = np.random.default_rng(random_seed)
        self.base_mutation_scale = base_mutation_scale
        self.mutation_modes = ("prompt_only", "operator_only", "mixed", "numeric_only")

    def mutate(
        self,
        parent_code: str,
        prompt: str | None = None,
        generation: int = 0,
        stagnation: int = 0,
    ) -> str:
        del prompt
        parent = genome_from_code(parent_code)
        mode = str(self.rng.choice(self.mutation_modes))
        stagnation_boost = 1.0 + min(stagnation, 4) * 0.15
        generation_boost = 1.0 + min(generation, 4) * 0.08
        mutation_scale = self.base_mutation_scale * stagnation_boost * generation_boost
        child = mutate_genome(
            parent,
            self.rng,
            mutation_scale=mutation_scale,
            mutation_mode=mode,
        )
        child["openevolve_mutation"] = {
            "mode": mode,
            "scale": mutation_scale,
            "generation": generation,
            "stagnation": stagnation,
        }
        return genome_to_code(child)

    def get_state(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "base_mutation_scale": self.base_mutation_scale,
            "mutation_modes": list(self.mutation_modes),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.base_mutation_scale = state.get("base_mutation_scale", self.base_mutation_scale)
        self.mutation_modes = tuple(state.get("mutation_modes", self.mutation_modes))
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]


class MegaOpenEvolveEvaluator:
    """OpenEvolve evaluator that delegates scientific scoring to MegaPersona."""

    def __init__(self, backend: MegaPersonaEvolver):
        self.backend = backend
        self.num_personas = backend.config.n
        self._code_to_candidate_id = self._load_existing_code_index()

    def evaluate(self, code_str: str) -> dict[str, float]:
        genome = genome_from_code(code_str)
        digest = genome_hash(genome)[:12]
        candidate_id = self._code_to_candidate_id.get(digest)
        if candidate_id is None:
            candidate_id = f"openevolve_{self.backend.evaluation_count + 1:06d}_{digest}"
            self._code_to_candidate_id[digest] = candidate_id

        candidate = MegaEvolutionCandidate(
            candidate_id=candidate_id,
            genome=genome,
            generation=self.backend.generation,
        )
        result = self.backend.evaluate_candidate(candidate)
        candidate.fitness = result["fitness"]
        candidate.metrics = result["metrics"]
        candidate.evaluated = True
        candidate.metrics["openevolve_genome_hash"] = digest

        self.backend.evaluation_count += 1
        self.backend.store.write_evaluation(
            evaluation_index=self.backend.evaluation_count,
            candidate=candidate,
            payload=result,
        )
        self.backend.best_candidate_id = self._best_candidate_id_after(candidate)
        self.backend._save_checkpoint()
        return open_evolve_fitness_from_payload(result)

    def candidate_id_for_code(self, code_str: str) -> str | None:
        return self._code_to_candidate_id.get(genome_hash(genome_from_code(code_str))[:12])

    def _best_candidate_id_after(self, candidate: MegaEvolutionCandidate) -> str:
        current = self.backend.best_candidate_id
        if current is None:
            return candidate.candidate_id
        old_payload = self.backend.store.find_candidate_result(current)
        old_fitness = (
            old_payload.get("fitness", float("-inf"))
            if old_payload is not None
            else float("-inf")
        )
        if (candidate.fitness or 0.0) > old_fitness:
            return candidate.candidate_id
        return current

    def _load_existing_code_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        candidates_dir = self.backend.store.candidates_dir
        if not candidates_dir.exists():
            return index
        for path in candidates_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                genome = payload.get("genome")
                candidate_id = payload.get("candidate_id")
                if isinstance(genome, dict) and candidate_id:
                    index[genome_hash(genome)[:12]] = candidate_id
            except Exception:
                continue
        return index


class MegaPersonaOpenEvolveRunner:
    """Convenience wrapper that wires MegaPersona evaluation into OpenEvolve."""

    def __init__(
        self,
        config: MegaEvolutionConfig,
        output_dir: Path,
        resume: bool = False,
        llm_client=None,
        simulator_llm_client=None,
        children_per_island: int = 1,
        base_mutation_scale: float = 0.12,
    ):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir = output_dir / "mega_eval"
        self.open_evolve_dir = output_dir / "open_evolve"
        self.open_evolve_dir.mkdir(parents=True, exist_ok=True)
        self.children_per_island = children_per_island
        self.resume = resume

        if resume and not (self.open_evolve_dir / "checkpoint.json").exists():
            raise FileNotFoundError(
                f"OpenEvolve checkpoint not found: {self.open_evolve_dir / 'checkpoint.json'}"
            )

        backend_resume = resume and (self.eval_dir / "checkpoint.json").exists()
        self.backend = MegaPersonaEvolver(
            config=config,
            output_dir=self.eval_dir,
            resume=backend_resume,
            llm_client=llm_client,
            simulator_llm_client=simulator_llm_client,
        )
        self.mutator = MegaGenomeMutator(
            random_seed=config.random_seed,
            base_mutation_scale=base_mutation_scale,
        )
        self.evaluator = MegaOpenEvolveEvaluator(self.backend)
        self.engine = self._load_or_create_engine(resume=resume)

    def run(self, argv: list[str] | None = None, model_key: str | None = None) -> MegaEvolutionCandidate:
        manifest = build_run_manifest(
            config=self.config,
            argv=argv,
            resume=self.resume,
            model_key=model_key,
        )
        manifest["engine"] = "src.open_evolve.engine.OpenEvolve"
        manifest["open_evolve_checkpoint_dir"] = str(self.open_evolve_dir)
        manifest["mega_eval_dir"] = str(self.eval_dir)
        manifest["shadow_survey_hashes"] = self.backend.survey_hashes
        self._write_json(self.output_dir / "manifest.json", manifest)

        best = self.engine.run(
            max_generations=self.config.generations,
            children_per_island=self.children_per_island,
        )
        if best is None:
            raise RuntimeError("OpenEvolve finished without a best candidate")

        best_candidate = self._best_mega_candidate(best.code, best.fitness)
        final_test_report = self.backend.evaluate_final_test(best_candidate)
        self.backend.store.write_final_test_report(final_test_report)
        self.backend.store.write_final_summary(
            best_candidate,
            [best_candidate],
            self.config,
            final_test_report,
        )
        self._write_root_summary(best_candidate, best.fitness, final_test_report)
        return best_candidate

    def _load_or_create_engine(self, resume: bool) -> OpenEvolve:
        checkpoint = self.open_evolve_dir / "checkpoint.json"
        if resume and checkpoint.exists():
            engine = OpenEvolve.from_checkpoint(
                str(checkpoint),
                mutator=self.mutator,
                evaluator=self.evaluator,
                questionnaires=list(self.backend.survey_splits.validation),
            )
            engine.checkpoint_path = self.open_evolve_dir
            return engine

        seed_codes = {"mega_default": genome_to_code(default_genome())}
        return OpenEvolve(
            mutator=self.mutator,
            evaluator=self.evaluator,
            questionnaires=list(self.backend.survey_splits.validation),
            seed_codes=seed_codes,
            initial_seed_distribution={"mega_default": self.config.population_size},
            num_islands=self.config.population_size,
            checkpoint_path=self.open_evolve_dir,
        )

    def _best_mega_candidate(
        self,
        best_code: str,
        open_evolve_fitness: dict[str, float],
    ) -> MegaEvolutionCandidate:
        genome = genome_from_code(best_code)
        candidate_id = self.evaluator.candidate_id_for_code(best_code)
        if candidate_id is None:
            raise FileNotFoundError("Best OpenEvolve genome has no stored MegaPersona evaluation")
        payload = self.backend.store.find_candidate_result(candidate_id)
        if payload is None:
            raise FileNotFoundError(f"missing stored evaluation for best candidate {candidate_id}")
        candidate = MegaEvolutionCandidate(
            candidate_id=candidate_id,
            genome=genome,
            fitness=payload["fitness"],
            metrics=payload["metrics"] | {"open_evolve_fitness": open_evolve_fitness},
            evaluated=True,
        )
        return candidate

    def _write_root_summary(
        self,
        best: MegaEvolutionCandidate,
        open_evolve_fitness: dict[str, float],
        final_test_report: dict[str, Any],
    ) -> None:
        payload = {
            "engine": "src.open_evolve.engine.OpenEvolve",
            "config": {**asdict(self.config), "seeds": list(self.config.seeds)},
            "best": best.to_dict(),
            "open_evolve_fitness": open_evolve_fitness,
            "final_test_report": final_test_report,
            "mega_eval_dir": str(self.eval_dir),
            "open_evolve_checkpoint_dir": str(self.open_evolve_dir),
            "completed_at": datetime.now().isoformat(),
        }
        self._write_json(self.output_dir / "final_summary.json", payload)
        (self.output_dir / "final_summary.md").write_text(
            _root_summary_markdown(payload),
            encoding="utf-8",
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def open_evolve_fitness_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    """Map MegaPersona validation metrics into OpenEvolve's elite slots."""
    metrics = payload.get("metrics", {})
    fitness = float(payload.get("fitness", 0.0))
    return {
        "coverage": fitness,
        "convex_hull": float(metrics.get("validation_behavior_coverage.mean", 0.0)),
        "avg_dist": float(metrics.get("validation_shadow_alignment.mean", 0.0)),
        "min_dist": float(metrics.get("schema_fitness.mean", 0.0)),
        "dispersion": float(metrics.get("slot_coverage.mean", 0.0)),
        "kl_divergence": 1.0 - float(metrics.get("near_duplicate_rate.mean", 1.0)),
        "mega_fitness": fitness,
    }


def _root_summary_markdown(payload: dict[str, Any]) -> str:
    best = payload["best"]
    test_metrics = payload.get("final_test_report", {}).get("metrics", {})
    lines = [
        "# MegaPersona OpenEvolve Summary",
        "",
        f"- engine: `{payload['engine']}`",
        f"- best candidate: `{best['candidate_id']}`",
        f"- validation fitness: `{best.get('fitness', 0.0):.4f}`",
        f"- mega eval dir: `{payload['mega_eval_dir']}`",
        f"- OpenEvolve checkpoint dir: `{payload['open_evolve_checkpoint_dir']}`",
        "",
        "## OpenEvolve Fitness",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in payload.get("open_evolve_fitness", {}).items():
        lines.append(f"| {key} | {value:.4f} |")
    if test_metrics:
        lines.extend(["", "## Sealed Test", "", "| Metric | Value |", "|---|---:|"])
        for key, value in test_metrics.items():
            lines.append(f"| {key} | {value:.4f} |")
    return "\n".join(lines) + "\n"
