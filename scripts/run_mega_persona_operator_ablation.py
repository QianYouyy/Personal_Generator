"""Run a fixed-parent operator ablation for MegaPersona evolution.

This script does not change the main evolution loop. It takes one already
evaluated parent genome, creates controlled variants, evaluates them with the
same durable evaluator, and writes a compact comparison report.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona import (  # noqa: E402
    EVOLUTION_PROMPT_OPERATORS,
    MegaEvolutionCandidate,
    MegaEvolutionConfig,
    MegaPersonaEvolver,
    build_run_manifest,
    mutate_genome,
)
from src.utils.llm_client import LLMClient  # noqa: E402


DEFAULT_OPERATORS = (
    "op06_low_axis_fidelity",
    "op04_within_bucket_contrast",
    "op07_high_axis_cost",
    "op02_behavioral_evidence",
    "op09_low_high_axis_tradeoff",
    "op14_recovery_latency",
)
DEFAULT_MUTATION_MODES = (
    "parent_replay",
    "prompt_only",
    "operator_only",
    "mixed",
    "numeric_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed-parent ablation of MegaPersona evolution operators."
    )
    parser.add_argument("--source-run", required=True, help="Previous run directory containing final_summary.json.")
    parser.add_argument(
        "--parent-candidate-id",
        default=None,
        help="Candidate id to use as parent. Defaults to final_summary best.",
    )
    parser.add_argument("--operators", default=",".join(DEFAULT_OPERATORS))
    parser.add_argument("--mutation-modes", default=",".join(DEFAULT_MUTATION_MODES))
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--mutation-scale", type=float, default=0.08)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seeds", default="17,23")
    parser.add_argument("--generator-mode", choices=["mock", "llm"], default="mock")
    parser.add_argument("--simulator-model-key", default="llm.simulator_model")
    parser.add_argument("--coverage-radius", type=float, default=0.28)
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--shadow-surveys", type=int, default=3)
    parser.add_argument("--validation-shadow-surveys", type=int, default=2)
    parser.add_argument("--test-shadow-surveys", type=int, default=2)
    parser.add_argument("--items-per-shadow-survey", type=int, default=8)
    parser.add_argument("--survey-seed", type=int, default=17)
    parser.add_argument("--random-seed", type=int, default=20260614)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--shadow-max-workers",
        type=int,
        default=1,
        help="Parallel LLM calls inside each shadow-survey simulation batch.",
    )
    parser.add_argument("--model-key", default="llm.persona_model")
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="After validation ablation, evaluate the selected best on the sealed test split.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/results/mega_persona_operator_ablation_<timestamp>",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/checkpoint.json instead of creating a fresh ablation population.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    checkpoint_path = output_dir / "checkpoint.json"
    if args.resume:
        if not checkpoint_path.exists():
            print(
                "Cannot resume: checkpoint not found at "
                f"{checkpoint_path}\n"
                "Start a fresh ablation without --resume, or pass the output directory "
                "that contains the previous checkpoint.json.",
                file=sys.stderr,
            )
            raise SystemExit(2)
    else:
        _ensure_fresh_output_dir(output_dir)
    _setup_logging(output_dir)

    source_run = Path(args.source_run)
    source_candidate = load_source_candidate(source_run, args.parent_candidate_id)
    operators = _parse_csv(args.operators)
    mutation_modes = _parse_csv(args.mutation_modes)
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())

    logging.info(
        "Operator ablation output_dir=%s source_run=%s parent=%s",
        output_dir,
        source_run,
        source_candidate["candidate_id"],
    )
    config = MegaEvolutionConfig(
        n=args.n,
        seeds=seeds,
        generator_mode=args.generator_mode,
        generations=0,
        population_size=1,
        children_per_generation=1,
        elite_count=1,
        coverage_radius=args.coverage_radius,
        duplicate_threshold=args.duplicate_threshold,
        shadow_surveys=args.shadow_surveys,
        validation_shadow_surveys=args.validation_shadow_surveys,
        test_shadow_surveys=args.test_shadow_surveys,
        items_per_shadow_survey=args.items_per_shadow_survey,
        survey_seed=args.survey_seed,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        shadow_max_workers=args.shadow_max_workers,
    )

    logging.info("Loading LLM clients generator_mode=%s", args.generator_mode)
    gen_llm = LLMClient.from_config(args.model_key) if args.generator_mode == "llm" else None
    sim_llm = LLMClient.from_config(args.simulator_model_key)

    evolver = MegaPersonaEvolver(
        config=config,
        output_dir=output_dir,
        resume=args.resume,
        llm_client=gen_llm,
        simulator_llm_client=sim_llm,
    )
    manifest = build_run_manifest(
        config=config,
        argv=sys.argv,
        resume=False,
        model_key=args.model_key if args.generator_mode == "llm" else None,
    )
    manifest["ablation"] = {
        "source_run": str(source_run),
        "parent_candidate_id": source_candidate["candidate_id"],
        "operators": operators,
        "mutation_modes": mutation_modes,
        "replicates": args.replicates,
        "mutation_scale": args.mutation_scale,
        "run_final_test": args.run_final_test,
    }
    manifest["shadow_survey_hashes"] = evolver.survey_hashes
    manifest["resume"] = args.resume
    evolver.store.write_manifest(manifest)

    if not args.resume:
        candidates = build_ablation_candidates(
            parent_candidate=source_candidate,
            operators=operators,
            mutation_modes=mutation_modes,
            replicates=args.replicates,
            mutation_scale=args.mutation_scale,
            random_seed=args.random_seed,
        )
        evolver.population = candidates
        evolver.evaluation_count = 0
        evolver.best_candidate_id = None
        evolver._save_checkpoint()
    else:
        candidates = evolver.population

    logging.info("Evaluating %s ablation candidate(s)", len(candidates))
    evolver._evaluate_population()
    best = evolver.best_candidate()
    final_test_report = evolver.evaluate_final_test(best) if args.run_final_test else None
    if final_test_report is not None:
        evolver.store.write_final_test_report(final_test_report)
    evolver.store.write_final_summary(best, evolver.population, config, final_test_report)

    summary = summarize_ablation_results(evolver.population, parent_candidate_id=source_candidate["candidate_id"])
    summary["config"] = manifest["ablation"]
    summary["best_candidate_id"] = best.candidate_id
    summary["best_fitness"] = best.fitness
    summary["final_test_report"] = final_test_report
    _atomic_write_json(output_dir / "ablation_summary.json", summary)
    (output_dir / "ablation_summary.md").write_text(
        _ablation_markdown(summary),
        encoding="utf-8",
    )
    logging.info("Ablation finished. Best=%s fitness=%.4f", best.candidate_id, best.fitness or 0.0)
    logging.info("Summary: %s", output_dir / "ablation_summary.md")


def load_source_candidate(source_run: Path, candidate_id: str | None = None) -> dict[str, Any]:
    final_summary_path = source_run / "final_summary.json"
    if not final_summary_path.exists():
        raise FileNotFoundError(f"final_summary.json not found: {final_summary_path}")
    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    if candidate_id is None:
        candidate_id = final_summary.get("best", {}).get("candidate_id")
    if not candidate_id:
        raise ValueError("parent candidate id is required when final_summary has no best candidate")

    for candidate in final_summary.get("population", []):
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    best = final_summary.get("best")
    if isinstance(best, dict) and best.get("candidate_id") == candidate_id:
        return best

    candidate_path = source_run / "candidates" / f"{candidate_id}.json"
    if candidate_path.exists():
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"candidate not found in source run: {candidate_id}")


def build_ablation_candidates(
    parent_candidate: dict[str, Any],
    operators: list[str],
    mutation_modes: list[str],
    replicates: int,
    mutation_scale: float,
    random_seed: int,
) -> list[MegaEvolutionCandidate]:
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    _validate_operators(operators)
    _validate_mutation_modes(mutation_modes)

    parent_id = parent_candidate["candidate_id"]
    parent_genome = parent_candidate["genome"]
    candidates: list[MegaEvolutionCandidate] = []
    sequence = 0

    if "parent_replay" in mutation_modes:
        for replicate in range(1, replicates + 1):
            genome = json.loads(json.dumps(parent_genome))
            genome["last_mutation"] = {"mode": "parent_replay", "scale": 0.0}
            candidates.append(
                MegaEvolutionCandidate(
                    candidate_id=f"ablation_{sequence:04d}_parent_replay_r{replicate:02d}",
                    genome=genome,
                    generation=0,
                    parent_id=parent_id,
                )
            )
            sequence += 1

    rng = np.random.default_rng(random_seed)
    for replicate in range(1, replicates + 1):
        if "numeric_only" in mutation_modes:
            genome = mutate_genome(
                parent_genome,
                rng,
                mutation_scale,
                mutation_mode="numeric_only",
            )
            candidates.append(
                MegaEvolutionCandidate(
                    candidate_id=f"ablation_{sequence:04d}_numeric_only_r{replicate:02d}",
                    genome=genome,
                    generation=1,
                    parent_id=parent_id,
                )
            )
            sequence += 1

        for operator_id in operators:
            for mode in mutation_modes:
                if mode in {"parent_replay", "numeric_only"}:
                    continue
                genome = mutate_genome(
                    parent_genome,
                    rng,
                    mutation_scale,
                    mutation_mode=mode,
                    operator_id=operator_id,
                )
                candidates.append(
                    MegaEvolutionCandidate(
                        candidate_id=f"ablation_{sequence:04d}_{operator_id}_{mode}_r{replicate:02d}",
                        genome=genome,
                        generation=1,
                        parent_id=parent_id,
                    )
                )
                sequence += 1
    return candidates


def summarize_ablation_results(
    candidates: list[MegaEvolutionCandidate],
    parent_candidate_id: str,
) -> dict[str, Any]:
    rows = [_candidate_row(candidate) for candidate in candidates]
    parent_values = [
        row["fitness"]
        for row in rows
        if row["mutation_mode"] == "parent_replay" and row["fitness"] is not None
    ]
    parent_fitness = float(np.mean(parent_values)) if parent_values else None
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["mutation_mode"], row["operator_id"] or "none")].append(row)

    grouped = []
    for (mode, operator_id), items in sorted(groups.items()):
        fitness_values = [item["fitness"] for item in items if item["fitness"] is not None]
        mean = float(np.mean(fitness_values)) if fitness_values else None
        std = float(np.std(fitness_values)) if len(fitness_values) > 1 else 0.0
        best = max(items, key=lambda item: item["fitness"] if item["fitness"] is not None else float("-inf"))
        grouped.append(
            {
                "mutation_mode": mode,
                "operator_id": None if operator_id == "none" else operator_id,
                "n": len(items),
                "fitness_mean": mean,
                "fitness_std": std,
                "best_candidate_id": best["candidate_id"],
                "best_fitness": best["fitness"],
                "delta_vs_parent": (
                    None
                    if mean is None or parent_fitness is None
                    else mean - parent_fitness
                ),
                "beats_parent": (
                    False
                    if best["fitness"] is None or parent_fitness is None
                    else best["fitness"] > parent_fitness
                ),
            }
        )

    return {
        "parent_candidate_id": parent_candidate_id,
        "parent_replay_fitness": parent_fitness,
        "parent_replay_n": len(parent_values),
        "parent_replay_std": float(np.std(parent_values)) if len(parent_values) > 1 else 0.0,
        "rows": sorted(rows, key=lambda row: row["fitness"] if row["fitness"] is not None else float("-inf"), reverse=True),
        "groups": sorted(
            grouped,
            key=lambda item: item["fitness_mean"] if item["fitness_mean"] is not None else float("-inf"),
            reverse=True,
        ),
    }


def _candidate_row(candidate: MegaEvolutionCandidate) -> dict[str, Any]:
    operator = candidate.genome.get("last_evolution_operator")
    mutation = candidate.genome.get("last_mutation") or {}
    mutation_mode = mutation.get("mode")
    if mutation_mode == "parent_replay":
        operator_id = "source_parent"
        operator_name = "Source parent"
    else:
        operator_id = operator.get("id") if isinstance(operator, dict) else None
        operator_name = operator.get("name") if isinstance(operator, dict) else None
    return {
        "candidate_id": candidate.candidate_id,
        "parent_id": candidate.parent_id,
        "fitness": candidate.fitness,
        "mutation_mode": mutation_mode,
        "mutation_scale": mutation.get("scale"),
        "operator_id": operator_id,
        "operator_name": operator_name,
        "schema_fitness": candidate.metrics.get("schema_fitness.mean"),
        "validation_behavior_coverage": candidate.metrics.get("validation_behavior_coverage.mean"),
        "validation_shadow_alignment": candidate.metrics.get("validation_shadow_alignment.mean"),
        "score_std": candidate.metrics.get("score.std"),
    }


def _ablation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MegaPersona Operator Ablation",
        "",
        f"- Source parent: `{summary['parent_candidate_id']}`",
        f"- Parent replay fitness: `{_fmt(summary.get('parent_replay_fitness'))}`",
        f"- Parent replay n/std: `{summary.get('parent_replay_n', 0)}` / `{_fmt(summary.get('parent_replay_std'))}`",
        f"- Best candidate: `{summary.get('best_candidate_id')}`",
        f"- Best fitness: `{_fmt(summary.get('best_fitness'))}`",
        "",
        "## Grouped Results",
        "",
        "| Rank | Mode | Operator | N | Mean Fitness | Std | Delta vs Parent | Beats Parent |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, group in enumerate(summary["groups"], start=1):
        lines.append(
            "| {rank} | `{mode}` | `{operator}` | {n} | {mean} | {std} | {delta} | {beats} |".format(
                rank=rank,
                mode=group["mutation_mode"],
                operator=group["operator_id"] or "none",
                n=group["n"],
                mean=_fmt(group["fitness_mean"]),
                std=_fmt(group["fitness_std"]),
                delta=_fmt(group["delta_vs_parent"]),
                beats="yes" if group["beats_parent"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Ranking",
            "",
            "| Rank | Candidate | Mode | Operator | Fitness | Schema | Val Coverage | Val Alignment | Score Std |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary["rows"], start=1):
        lines.append(
            "| {rank} | `{candidate}` | `{mode}` | `{operator}` | {fitness} | {schema} | {cov} | {align} | {std} |".format(
                rank=rank,
                candidate=row["candidate_id"],
                mode=row["mutation_mode"],
                operator=row["operator_id"] or "none",
                fitness=_fmt(row["fitness"]),
                schema=_fmt(row["schema_fitness"]),
                cov=_fmt(row["validation_behavior_coverage"]),
                align=_fmt(row["validation_shadow_alignment"]),
                std=_fmt(row["score_std"]),
            )
        )
    return "\n".join(lines) + "\n"


def _validate_operators(operators: list[str]) -> None:
    known = {operator["id"] for operator in EVOLUTION_PROMPT_OPERATORS}
    unknown = [operator for operator in operators if operator not in known]
    if unknown:
        raise ValueError(f"unknown operator id(s): {unknown}")


def _validate_mutation_modes(mutation_modes: list[str]) -> None:
    allowed = {"parent_replay", "prompt_only", "operator_only", "mixed", "numeric_only"}
    unknown = [mode for mode in mutation_modes if mode not in allowed]
    if unknown:
        raise ValueError(f"unknown mutation mode(s): {unknown}")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _ensure_fresh_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output dir is not empty: {output_dir}. Use a new output directory for each ablation."
        )


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/results") / f"mega_persona_operator_ablation_{timestamp}"


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    for noisy_logger in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
