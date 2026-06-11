"""Run repeatable MegaPersona experiment batches and export reports."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona import (
    MegaPersonaExperimentConfig,
    MegaPersonaExperimentRunner,
    write_experiment_artifacts,
)
from src.utils.llm_client import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MegaPersona experiment batches.")
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument(
        "--seeds",
        default="17,23,31",
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--shadow-surveys", type=int, default=12)
    parser.add_argument("--items-per-shadow-survey", type=int, default=12)
    parser.add_argument("--coverage-radius", type=float, default=0.28)
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--shadow-noise", type=float, default=0.08)
    parser.add_argument("--model-key", default="llm.persona_model")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/results/mega_persona_experiment_<timestamp>",
    )
    parser.add_argument(
        "--no-personas",
        action="store_true",
        help="Do not include full persona JSON in summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    config = MegaPersonaExperimentConfig(
        n=args.n,
        seeds=seeds,
        mode=args.mode,
        num_shadow_surveys=args.shadow_surveys,
        items_per_shadow_survey=args.items_per_shadow_survey,
        coverage_radius=args.coverage_radius,
        duplicate_threshold=args.duplicate_threshold,
        shadow_noise=args.shadow_noise,
    )
    llm = LLMClient.from_config(args.model_key) if args.mode == "llm" else None
    summary = MegaPersonaExperimentRunner(config=config, llm_client=llm).run()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    json_path, markdown_path = write_experiment_artifacts(
        summary,
        output_dir=output_dir,
        include_personas=not args.no_personas,
    )
    aggregate = summary.aggregate_metrics()
    print(f"Saved MegaPersona experiment JSON to {json_path}")
    print(f"Saved MegaPersona experiment report to {markdown_path}")
    print(
        "Aggregate: "
        f"score={aggregate.get('experiment_score.mean', 0.0):.4f}, "
        f"schema={aggregate.get('schema_fitness.mean', 0.0):.4f}, "
        f"shadow_alignment={aggregate.get('shadow_alignment.mean', 0.0):.4f}, "
        f"behavior_coverage={aggregate.get('behavior_coverage.mean', 0.0):.4f}"
    )


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/results") / f"mega_persona_experiment_{timestamp}"


if __name__ == "__main__":
    main()
