"""Run durable Open-Evolve style optimization for MegaPersona experiments."""

import argparse
from datetime import datetime
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
    )
    # Generator LLM client (for LLM persona generation)
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
    best = evolver.run()
    print(f"Saved durable MegaPersona evolution run to {output_dir}")
    print(f"Best candidate: {best.candidate_id}")
    print(f"Best fitness: {best.fitness:.4f}")
    print(f"Checkpoint: {output_dir / 'checkpoint.json'}")
    print(f"Final summary: {output_dir / 'final_summary.md'}")


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/results") / f"mega_persona_evolution_{timestamp}"


if __name__ == "__main__":
    main()
