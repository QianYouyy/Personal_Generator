"""Run durable Open-Evolve style optimization for MegaPersona experiments."""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona import MegaEvolutionConfig, MegaPersonaEvolver, build_run_manifest
from src.utils.llm_client import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run durable MegaPersona evolution.")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seeds", default="17,23,31")
    parser.add_argument("--generator-mode", choices=["mock", "llm"], default="mock")
    parser.add_argument("--simulator-model-key", default="llm.simulator_model",
                        help="Config key for the LLM simulator model.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--children-per-generation", type=int, default=6)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--coverage-radius", type=float, default=0.28)
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--shadow-surveys", type=int, default=12)
    parser.add_argument("--validation-shadow-surveys", type=int, default=4)
    parser.add_argument("--test-shadow-surveys", type=int, default=4)
    parser.add_argument(
        "--heldout-shadow-surveys",
        type=int,
        default=None,
        help="Deprecated alias for --validation-shadow-surveys.",
    )
    parser.add_argument("--items-per-shadow-survey", type=int, default=12)
    parser.add_argument("--survey-seed", type=int, default=17)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--shadow-max-workers",
        type=int,
        default=1,
        help="Parallel LLM calls inside each shadow-survey simulation batch.",
    )
    parser.add_argument("--model-key", default="llm.persona_model")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/results/mega_persona_evolution_<timestamp>",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/checkpoint.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    checkpoint_path = output_dir / "checkpoint.json"
    if args.resume and not checkpoint_path.exists():
        print(
            "Cannot resume: checkpoint not found at "
            f"{checkpoint_path}\n"
            "Start a fresh run without --resume, or pass the output directory "
            "that contains the previous checkpoint.json.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _setup_logging(output_dir)
    logging.info("MegaPersona evolution output_dir=%s resume=%s", output_dir, args.resume)
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    validation_shadow_surveys = (
        args.heldout_shadow_surveys
        if args.heldout_shadow_surveys is not None
        else args.validation_shadow_surveys
    )
    config = MegaEvolutionConfig(
        n=args.n,
        seeds=seeds,
        generator_mode=args.generator_mode,
        generations=args.generations,
        population_size=args.population_size,
        children_per_generation=args.children_per_generation,
        elite_count=args.elite_count,
        coverage_radius=args.coverage_radius,
        duplicate_threshold=args.duplicate_threshold,
        shadow_surveys=args.shadow_surveys,
        validation_shadow_surveys=validation_shadow_surveys,
        test_shadow_surveys=args.test_shadow_surveys,
        items_per_shadow_survey=args.items_per_shadow_survey,
        survey_seed=args.survey_seed,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        shadow_max_workers=args.shadow_max_workers,
    )
    # Generator LLM client (for LLM persona generation)
    logging.info("Loading LLM clients generator_mode=%s", args.generator_mode)
    gen_llm = LLMClient.from_config(args.model_key) if args.generator_mode == "llm" else None
    # Simulator LLM client (LLMShadowSimulator is always used now)
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
        resume=args.resume,
        model_key=args.model_key if args.generator_mode == "llm" else None,
    )
    manifest["shadow_survey_hashes"] = evolver.survey_hashes
    manifest["shadow_survey_dir"] = str(evolver.store.surveys_dir)
    evolver.store.write_manifest(
        manifest
    )
    logging.info("Manifest written to %s", output_dir / "manifest.json")
    best = evolver.run()
    logging.info("Saved durable MegaPersona evolution run to %s", output_dir)
    logging.info("Best candidate: %s", best.candidate_id)
    logging.info("Best fitness: %.4f", best.fitness or 0.0)
    logging.info("Checkpoint: %s", output_dir / "checkpoint.json")
    logging.info("Final summary: %s", output_dir / "final_summary.md")
    logging.info("Run log: %s", output_dir / "run.log")


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/results") / f"mega_persona_evolution_{timestamp}"


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
