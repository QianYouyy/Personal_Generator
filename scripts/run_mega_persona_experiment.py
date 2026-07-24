"""Run repeatable MegaPersona experiment batches and export reports."""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time

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
    parser.add_argument("--seeds", default="17,23,31", help="Comma-separated random seeds.")
    parser.add_argument("--shadow-surveys", type=int, default=12)
    parser.add_argument("--items-per-shadow-survey", type=int, default=12)
    parser.add_argument("--coverage-radius", type=float, default=0.28)
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--persona-max-workers", type=int, default=1,
                        help="Concurrent persona generation slots. Each slot may also run internal agents in parallel.")
    parser.add_argument("--shadow-max-workers", type=int, default=1,
                        help="Concurrent shadow survey simulation calls.")
    parser.add_argument("--model-key", default="llm.persona_model")
    parser.add_argument("--simulator-model-key", default="llm.simulator_model",
                        help="Config key for the LLM simulator model.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to data/results/mega_persona_experiment_<timestamp>")
    parser.add_argument("--no-personas", action="store_true",
                        help="Do not include full persona JSON in summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    _setup_logging(output_dir)
    logging.info(
        "MegaPersona experiment output_dir=%s mode=%s n=%s seeds=%s persona_workers=%s shadow_workers=%s",
        output_dir,
        args.mode,
        args.n,
        args.seeds,
        args.persona_max_workers,
        args.shadow_max_workers,
    )
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    config = MegaPersonaExperimentConfig(
        n=args.n,
        seeds=seeds,
        mode=args.mode,
        num_shadow_surveys=args.shadow_surveys,
        items_per_shadow_survey=args.items_per_shadow_survey,
        coverage_radius=args.coverage_radius,
        duplicate_threshold=args.duplicate_threshold,
        persona_max_workers=args.persona_max_workers,
        shadow_max_workers=args.shadow_max_workers,
    )
    logging.info("Loading LLM clients mode=%s", args.mode)
    gen_llm = LLMClient.from_config(args.model_key) if args.mode == "llm" else None
    sim_llm = LLMClient.from_config(args.simulator_model_key)
    logging.info("Starting experiment run")
    summary = MegaPersonaExperimentRunner(
        config=config,
        llm_client=gen_llm,
        simulator_llm_client=sim_llm,
    ).run()
    json_path, markdown_path = write_experiment_artifacts(
        summary, output_dir=output_dir, include_personas=not args.no_personas,
    )
    aggregate = summary.aggregate_metrics()
    elapsed = time.time() - started_at
    logging.info("Saved MegaPersona experiment JSON to %s", json_path)
    logging.info("Saved MegaPersona experiment report to %s", markdown_path)
    logging.info(
        "Aggregate score=%.4f schema=%.4f shadow_alignment=%.4f behavior_coverage=%.4f elapsed=%.1fs",
        aggregate.get("experiment_score.mean", 0.0),
        aggregate.get("schema_fitness.mean", 0.0),
        aggregate.get("shadow_alignment.mean", 0.0),
        aggregate.get("behavior_coverage.mean", 0.0),
        elapsed,
    )
    logging.info("Run log: %s", output_dir / "run.log")
    print(f"Saved MegaPersona experiment JSON to {json_path}")
    print(f"Saved MegaPersona experiment report to {markdown_path}")
    print(
        "Aggregate: "
        f"score={aggregate.get('experiment_score.mean', 0.0):.4f}, "
        f"schema={aggregate.get('schema_fitness.mean', 0.0):.4f}, "
        f"shadow_alignment={aggregate.get('shadow_alignment.mean', 0.0):.4f}, "
        f"behavior_coverage={aggregate.get('behavior_coverage.mean', 0.0):.4f}"
    )
    print(f"Elapsed: {elapsed:.1f}s")


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/results") / f"mega_persona_experiment_{timestamp}"


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
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
    for noisy_logger in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
