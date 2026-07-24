"""Measure the MegaPersona evaluation noise floor.

Evaluates the same genome(s) K times under a finished run's exact evaluation
configuration (same slots, same shadow-survey splits, same models), so the
observed per-metric variance reflects only LLM/simulator stochasticity. The
resulting standard deviation is the noise floor: candidate fitness differences
smaller than ~2 sigma are not resolvable and should not be interpreted as real
improvement or plateau escape.

Usage:

    python scripts/measure_evaluation_noise_floor.py \
      --source-run data/results/<run> \
      --repeats 5 \
      --output-dir data/results/<run>/noise_floor
"""

import argparse
from dataclasses import fields
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona import MegaEvolutionConfig
from src.mega_persona.evolution import (
    MegaEvolutionCandidate,
    MegaPersonaEvolver,
    default_genome,
    default_genome_v4,
)
from src.utils.llm_client import LLMClient

logger = logging.getLogger("noise_floor")

# Metrics highlighted at the top of the report; every other numeric metric is
# still aggregated and listed below them.
_KEY_METRICS = (
    "fitness",
    "schema_fitness.mean",
    "generation_rate.mean",
    "validation_behavior_coverage.mean",
    "validation_behavior_balanced_diversity.mean",
    "validation_shadow_alignment.mean",
    "validation_shadow_mae.mean",
    "axis_target_mae.mean",
    "internal_consistency.mean",
    "internal_consistency_min.mean",
    "strict_consistency_error.mean",
    "consistency_issue_rate.mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure MegaPersona evaluation noise floor by repeatedly "
        "evaluating the same genome under a finished run's configuration."
    )
    parser.add_argument(
        "--source-run",
        required=True,
        help="Finished run directory; manifest.json supplies the full evaluation config.",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        help="Candidate id from the source run to measure (repeatable). "
        "Default: the run's best candidate plus the default seed genome.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <source-run>/noise_floor",
    )
    parser.add_argument("--shadow-max-workers", type=int, default=None)
    return parser.parse_args()


def _config_from_manifest(manifest: dict, shadow_max_workers: int | None) -> MegaEvolutionConfig:
    raw = dict(manifest.get("config") or {})
    valid = {field.name for field in fields(MegaEvolutionConfig)}
    filtered = {key: value for key, value in raw.items() if key in valid}
    if "seeds" in filtered:
        filtered["seeds"] = tuple(int(seed) for seed in filtered["seeds"])
    if shadow_max_workers is not None:
        filtered["shadow_max_workers"] = shadow_max_workers
    return MegaEvolutionConfig(**filtered)


def _llm_from_manifest(manifest: dict, role: str) -> LLMClient:
    provider = manifest.get("llm_provider")
    if role == "persona":
        model = manifest.get("persona_model")
        api_base = manifest.get("persona_api_base")
        api_key_env = manifest.get("persona_api_key_env")
        model_key = manifest.get("model_key")
    else:
        model = manifest.get("simulator_model")
        api_base = manifest.get("simulator_api_base")
        api_key_env = manifest.get("simulator_api_key_env")
        model_key = manifest.get("simulator_model_key")
    if provider:
        return LLMClient.from_provider(
            provider,
            role=role,
            model=model,
            api_key_env=api_key_env,
            base_url=api_base,
        )
    if model:
        return LLMClient(
            model=model,
            base_url=api_base,
            api_key_env=api_key_env or "OPENAI_API_KEY",
        )
    return LLMClient.from_config(model_key, base_url=api_base)


def _load_genomes(source_run: Path, candidate_ids: list[str] | None) -> dict[str, dict]:
    genomes: dict[str, dict] = {}
    if candidate_ids:
        for candidate_id in candidate_ids:
            path = source_run / "mega_eval" / "candidates" / f"{candidate_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            genomes[candidate_id] = payload["genome"]
        return genomes

    summary_path = source_run / "mega_eval" / "final_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary.get("best") or {}
        if isinstance(best.get("genome"), dict):
            genomes[f"best:{best.get('candidate_id', 'unknown')}"] = best["genome"]
    genome_version = max(
        (int(genome.get("genome_version", 3)) for genome in genomes.values()),
        default=3,
    )
    genomes["seed_default"] = default_genome_v4() if genome_version == 4 else default_genome()
    return genomes


def _aggregate(repeats: list[dict]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for payload in repeats:
        metrics = payload.get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, []).append(float(value))
        values.setdefault("fitness", []).append(float(payload.get("fitness", 0.0)))
    # Keep only metrics present in every repeat so stats are comparable.
    complete = {
        key: series
        for key, series in values.items()
        if len(series) == len(repeats)
    }
    stats: dict[str, dict[str, float]] = {}
    for key, series in complete.items():
        arr = np.asarray(series, dtype=float)
        stats[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
        }
    return stats


def _markdown_report(all_stats: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = [
        "# Evaluation Noise Floor",
        "",
        "Same genome, same slots, same survey splits, repeated evaluations.",
        "Differences below 2*std are within noise and not resolvable.",
        "",
    ]
    for label, stats in all_stats.items():
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Metric | Mean | Std | 2*std | Min | Max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        ordered = [key for key in _KEY_METRICS if key in stats]
        ordered += sorted(key for key in stats if key not in _KEY_METRICS)
        for key in ordered:
            row = stats[key]
            lines.append(
                f"| {key} | {row['mean']:.4f} | {row['std']:.4f} | "
                f"{2 * row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    source_run = Path(args.source_run)
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    config = _config_from_manifest(manifest, args.shadow_max_workers)
    output_dir = Path(args.output_dir) if args.output_dir else source_run / "noise_floor"
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_llm = _llm_from_manifest(manifest, "persona") if config.generator_mode == "llm" else None
    sim_llm = _llm_from_manifest(manifest, "simulator")
    backend = MegaPersonaEvolver(
        config=config,
        output_dir=output_dir / "mega_eval",
        resume=False,
        llm_client=gen_llm,
        simulator_llm_client=sim_llm,
        initial_genome=(
            default_genome_v4()
            if int(manifest.get("genome_version", 3)) == 4
            else default_genome()
        ),
    )

    manifest_hashes = manifest.get("shadow_survey_hashes") or {}
    if manifest_hashes and backend.survey_hashes != manifest_hashes:
        raise SystemExit(
            "Shadow survey split mismatch with the source run; refusing to "
            "measure noise on different surveys.\n"
            f"  source: {manifest_hashes}\n  rebuilt: {backend.survey_hashes}"
        )
    logger.info("Shadow survey splits verified identical to source run")

    genomes = _load_genomes(source_run, args.candidate_id)
    logger.info(
        "Measuring %d genomes x %d repeats (n=%d seeds=%s simulator=%s)",
        len(genomes),
        args.repeats,
        config.n,
        config.seeds,
        config.shadow_simulator_backend,
    )

    all_stats: dict[str, dict[str, dict[str, float]]] = {}
    raw: dict[str, list[dict]] = {}
    for label, genome in genomes.items():
        repeats: list[dict] = []
        for rep in range(args.repeats):
            candidate = MegaEvolutionCandidate(
                candidate_id=f"noisefloor_{label.replace(':', '_')}_rep{rep:02d}",
                genome=genome,
                generation=0,
            )
            payload = backend.evaluate_candidate(candidate)
            repeats.append(payload)
            logger.info("%s rep%d fitness=%.4f", label, rep, payload.get("fitness", 0.0))
        all_stats[label] = _aggregate(repeats)
        raw[label] = repeats

    result = {
        "source_run": str(source_run),
        "repeats": args.repeats,
        "config": {field.name: getattr(config, field.name) for field in fields(MegaEvolutionConfig)},
        "survey_hashes": backend.survey_hashes,
        "stats": all_stats,
    }
    (output_dir / "noise_floor.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report = _markdown_report(all_stats)
    (output_dir / "noise_floor.md").write_text(report, encoding="utf-8")

    print("\n" + report)
    for label, stats in all_stats.items():
        fitness = stats.get("fitness", {})
        print(
            f"[{label}] fitness noise floor: mean={fitness.get('mean', 0):.4f} "
            f"std={fitness.get('std', 0):.4f} -> differences < "
            f"{2 * fitness.get('std', 0):.4f} are not resolvable"
        )


if __name__ == "__main__":
    main()
