"""Run MegaPersona evolution through src.open_evolve.engine.OpenEvolve."""

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona import MegaEvolutionConfig
from src.mega_persona.evolution import EVOLUTION_PROMPT_OPERATORS
from src.mega_persona.openevolve_adapter import MegaPersonaOpenEvolveRunner
from src.utils.llm_client import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MegaPersona with the shared OpenEvolve island engine."
    )
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seeds", default="17,23,31")
    parser.add_argument("--generator-mode", choices=["mock", "llm"], default="mock")
    parser.add_argument(
        "--persona-pipeline",
        choices=["five_agent", "single_call", "compact"],
        default="five_agent",
        help=(
            "LLM persona generation pipeline. five_agent is the decomposed "
            "5-call architecture; single_call is an integrated 1-call "
            "architecture. compact is accepted as a legacy alias for single_call."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "deepseek"],
        default=None,
        help="Optional OpenAI-compatible provider preset for both persona and simulator models.",
    )
    parser.add_argument("--mutator-model-key", default="llm.mutator_model")
    parser.add_argument(
        "--mutator-model",
        default=None,
        help="Direct mutator model override for the OpenEvolve mutation stage.",
    )
    parser.add_argument(
        "--mutator-api-base",
        default=None,
        help="Optional OpenAI-compatible API base URL for the mutator model.",
    )
    parser.add_argument(
        "--mutator-api-key-env",
        default=None,
        help="Optional environment variable name containing the mutator API key.",
    )
    parser.add_argument("--model-key", default="llm.persona_model")
    parser.add_argument(
        "--persona-model",
        default=None,
        help="Direct persona generation model override for --llm-provider runs.",
    )
    parser.add_argument(
        "--persona-api-base",
        default=None,
        help="Optional OpenAI-compatible API base URL for the persona model.",
    )
    parser.add_argument(
        "--persona-api-key-env",
        default=None,
        help="Optional environment variable name containing the persona API key.",
    )
    parser.add_argument("--simulator-model-key", default="llm.simulator_model")
    parser.add_argument(
        "--simulator-backend",
        choices=[
            "llm",
            "concordia",
            "concordia-native",
            "student-realistic",
            "student-realistic-v2",
        ],
        default="llm",
        help="Shadow-survey simulator backend.",
    )
    parser.add_argument(
        "--simulator-model",
        default=None,
        help="Direct simulator model override. If set, --simulator-model-key is ignored.",
    )
    parser.add_argument(
        "--simulator-api-base",
        default=None,
        help="Optional OpenAI-compatible API base URL for the simulator model.",
    )
    parser.add_argument(
        "--simulator-api-key-env",
        default=None,
        help="Optional environment variable name containing the simulator API key.",
    )
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument(
        "--num-islands",
        dest="num_islands",
        type=int,
        default=8,
        help="Number of OpenEvolve islands.",
    )
    parser.add_argument(
        "--population-size",
        dest="num_islands",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--children-per-island",
        type=int,
        default=1,
        help="OpenEvolve children generated per island per generation.",
    )
    parser.add_argument("--elite-count", type=int, default=3,
                        help="Recorded in manifest; OpenEvolve uses metric elites per island.")
    parser.add_argument("--coverage-radius", type=float, default=0.28)
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--shadow-surveys", type=int, default=12)
    parser.add_argument("--validation-shadow-surveys", type=int, default=4)
    parser.add_argument("--test-shadow-surveys", type=int, default=4)
    parser.add_argument("--items-per-shadow-survey", type=int, default=12)
    parser.add_argument("--survey-seed", type=int, default=17)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument(
        "--candidate-max-workers",
        "--max-workers",
        dest="candidate_max_workers",
        type=int,
        default=1,
        help="Parallel OpenEvolve candidate evaluations per generation.",
    )
    parser.add_argument(
        "--candidate-evaluation-repeats",
        type=int,
        default=1,
        help=(
            "Independent full evaluations per candidate used for selection. "
            "OpenEvolve elites and MCTS receive the repeat mean; repeat uncertainty "
            "is stored in each evaluation payload. Use 3 or more for confirmatory runs."
        ),
    )
    parser.add_argument(
        "--elite-confirmation-repeats",
        type=int,
        default=1,
        help=(
            "Optional full-evaluation confirmation budget for elite challengers. "
            "The default 1 disables additional evaluations."
        ),
    )
    parser.add_argument(
        "--persona-max-workers",
        type=int,
        default=2,
        help="Parallel persona generations inside each candidate/seed evaluation.",
    )
    parser.add_argument("--shadow-max-workers", type=int, default=1)
    parser.add_argument(
        "--persona-temperature",
        type=float,
        default=0.45,
        help="Persona generator sampling temperature.",
    )
    parser.add_argument(
        "--persona-top-p",
        type=float,
        default=0.85,
        help="Persona generator nucleus-sampling probability.",
    )
    parser.add_argument(
        "--simulator-temperature",
        type=float,
        default=0.05,
        help="Shadow evaluator sampling temperature; kept low to reduce selection noise.",
    )
    parser.add_argument(
        "--simulator-top-p",
        type=float,
        default=0.80,
        help="Shadow evaluator nucleus-sampling probability.",
    )
    parser.add_argument("--base-mutation-scale", type=float, default=0.12)
    parser.add_argument(
        "--genome-version",
        type=int,
        choices=(3, 4),
        default=3,
        help=(
            "Genome representation. v3 keeps the existing mixed prompt/blueprint surface; "
            "v4 uses a low-dimensional structured generation program with deterministic mutation."
        ),
    )
    parser.add_argument(
        "--extinction-interval",
        type=int,
        default=None,
        help="Override OpenEvolve fixed extinction interval in generations.",
    )
    parser.add_argument(
        "--fixed-operator",
        default=None,
        help=(
            "Use exactly one evolution operator for every mutation, e.g. "
            "op06_low_axis_fidelity. This is for controlled single-operator "
            "validation experiments."
        ),
    )
    parser.add_argument(
        "--operator-family",
        choices=("all", "v3", "v4", "legacy"),
        default=None,
        help=(
            "Random operator pool when --fixed-operator is not set. Defaults to v4 for "
            "--genome-version 4, otherwise all (legacy + v3)."
        ),
    )
    parser.add_argument(
        "--search-strategy",
        choices=("openevolve", "hybrid_mcts"),
        default="openevolve",
        help=(
            "Operator-selection strategy. openevolve preserves the existing random "
            "operator choice; hybrid_mcts uses an online MCTS-style tree over operator sequences."
        ),
    )
    parser.add_argument(
        "--mcts-depth",
        type=int,
        default=3,
        help="Maximum operator-sequence depth tracked by --search-strategy hybrid_mcts.",
    )
    parser.add_argument(
        "--mcts-exploration-c",
        type=float,
        default=1.4,
        help="UCT exploration constant for --search-strategy hybrid_mcts.",
    )
    parser.add_argument(
        "--mcts-progressive-widening",
        action="store_true",
        help="Progressively widen available operator actions for hybrid MCTS.",
    )
    parser.add_argument(
        "--mcts-reward-profile",
        choices=("legacy", "structured"),
        default="legacy",
        help=(
            "Reward profile for --search-strategy hybrid_mcts. legacy keeps the "
            "original absolute-delta reward; structured uses the layered plateau-aware "
            "reward (relative deltas, bounded coverage/diversity guard, historical-best "
            "progress bonus, reward standardization)."
        ),
    )
    parser.add_argument(
        "--mcts-plateau-stagnation",
        type=int,
        default=4,
        help=(
            "Generations without global_best improvement before the structured reward "
            "profile enters plateau mode (progress bonus doubled, UCT exploration boosted)."
        ),
    )
    parser.add_argument(
        "--mcts-reward-weight-mode",
        choices=("fixed", "deficit"),
        default="fixed",
        help=(
            "Optimization-metric weights for --mcts-reward-profile structured. fixed uses "
            "the static weight table; deficit scales each metric's weight by how far the "
            "parent lags that metric's historical best (normalized into shares), so the "
            "reward rotates toward lagging metrics during plateaus."
        ),
    )
    parser.add_argument(
        "--parent-selection",
        choices=("operator_preferred", "objective_rotation"),
        default="operator_preferred",
        help=(
            "Parent-selection strategy. operator_preferred keeps the historical behavior "
            "(operator preferred_parent_metric, uniform elite fallback). objective_rotation "
            "round-robins each child's parent across global/coverage/diversity/strict/"
            "shadow-MAE elites so all objective bests participate in mutation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/results/mega_persona_evolution_<timestamp>",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/open_evolve/checkpoint.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_evaluation_repeats < 1:
        raise SystemExit("--candidate-evaluation-repeats must be >= 1")
    if args.elite_confirmation_repeats < args.candidate_evaluation_repeats:
        raise SystemExit(
            "--elite-confirmation-repeats must be >= --candidate-evaluation-repeats"
        )
    if not 0.0 <= args.persona_temperature <= 2.0:
        raise SystemExit("--persona-temperature must be between 0 and 2")
    if not 0.0 <= args.simulator_temperature <= 2.0:
        raise SystemExit("--simulator-temperature must be between 0 and 2")
    if not 0.0 < args.persona_top_p <= 1.0:
        raise SystemExit("--persona-top-p must be in (0, 1]")
    if not 0.0 < args.simulator_top_p <= 1.0:
        raise SystemExit("--simulator-top-p must be in (0, 1]")
    operator_family = args.operator_family or ("v4" if args.genome_version == 4 else "all")
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    open_evolve_checkpoint = output_dir / "open_evolve" / "checkpoint.json"
    if args.resume and not open_evolve_checkpoint.exists():
        print(
            "Cannot resume: OpenEvolve checkpoint not found at "
            f"{open_evolve_checkpoint}\n"
            "Start a fresh run without --resume, or pass the output directory "
            "that contains open_evolve/checkpoint.json.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    _setup_logging(output_dir)
    logging.info("MegaPersona OpenEvolve output_dir=%s resume=%s", output_dir, args.resume)
    if args.fixed_operator:
        _validate_fixed_operator(args.fixed_operator)
        logging.info("Fixed evolution operator enabled: %s", args.fixed_operator)
    logging.info("Genome version: v%s", args.genome_version)
    logging.info("Evolution operator family: %s", operator_family)
    logging.info(
        "Candidate selection evaluations: base_repeats=%s elite_confirmation_repeats=%s "
        "aggregation=mean",
        args.candidate_evaluation_repeats,
        args.elite_confirmation_repeats,
    )
    logging.info(
        "Sampling: persona temperature=%.2f top_p=%.2f; simulator temperature=%.2f top_p=%.2f",
        args.persona_temperature,
        args.persona_top_p,
        args.simulator_temperature,
        args.simulator_top_p,
    )
    logging.info(
        "Search strategy: %s mcts_depth=%s mcts_c=%.3f progressive_widening=%s "
        "reward_profile=%s plateau_stagnation=%s reward_weight_mode=%s parent_selection=%s",
        args.search_strategy,
        args.mcts_depth,
        args.mcts_exploration_c,
        args.mcts_progressive_widening,
        args.mcts_reward_profile,
        args.mcts_plateau_stagnation,
        args.mcts_reward_weight_mode,
        args.parent_selection,
    )
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    config = MegaEvolutionConfig(
        n=args.n,
        seeds=seeds,
        generator_mode=args.generator_mode,
        generations=args.generations,
        population_size=args.num_islands,
        children_per_generation=args.children_per_island,
        elite_count=args.elite_count,
        coverage_radius=args.coverage_radius,
        duplicate_threshold=args.duplicate_threshold,
        shadow_surveys=args.shadow_surveys,
        validation_shadow_surveys=args.validation_shadow_surveys,
        test_shadow_surveys=args.test_shadow_surveys,
        items_per_shadow_survey=args.items_per_shadow_survey,
        survey_seed=args.survey_seed,
        random_seed=args.random_seed,
        max_workers=args.candidate_max_workers,
        candidate_evaluation_repeats=args.candidate_evaluation_repeats,
        elite_confirmation_repeats=args.elite_confirmation_repeats,
        shadow_max_workers=args.shadow_max_workers,
        persona_max_workers=args.persona_max_workers,
        shadow_simulator_backend=args.simulator_backend,
        persona_pipeline=args.persona_pipeline,
        persona_temperature=args.persona_temperature,
        persona_top_p=args.persona_top_p,
        simulator_temperature=args.simulator_temperature,
        simulator_top_p=args.simulator_top_p,
    )

    logging.info(
        "Loading LLM clients generator_mode=%s provider=%s",
        args.generator_mode,
        args.llm_provider or "config",
    )
    # Genome v4 mutations are deterministic structured patches; the mutator
    # LLM is intentionally not loaded or called for this experimental surface.
    mutator_llm = None if args.genome_version == 4 else _load_mutator_llm(args)
    gen_llm = _load_generation_llm(args) if args.generator_mode == "llm" else None
    sim_llm = _load_simulator_llm(args)
    logging.info(
        "Models mutator=%s persona=%s simulator=%s persona_pipeline=%s",
        getattr(mutator_llm, "model", None),
        getattr(gen_llm, "model", None) if gen_llm else None,
        getattr(sim_llm, "model", None),
        args.persona_pipeline,
    )

    runner = MegaPersonaOpenEvolveRunner(
        config=config,
        output_dir=output_dir,
        resume=args.resume,
        mutator_llm_client=mutator_llm,
        llm_client=gen_llm,
        simulator_llm_client=sim_llm,
        children_per_island=args.children_per_island,
        base_mutation_scale=args.base_mutation_scale,
        fixed_operator_id=args.fixed_operator,
        operator_family=operator_family,
        genome_version=args.genome_version,
        search_strategy=args.search_strategy,
        mcts_depth=args.mcts_depth,
        mcts_exploration_c=args.mcts_exploration_c,
        mcts_progressive_widening=args.mcts_progressive_widening,
        mcts_reward_profile=args.mcts_reward_profile,
        mcts_plateau_stagnation=args.mcts_plateau_stagnation,
        mcts_reward_weight_mode=args.mcts_reward_weight_mode,
        parent_selection=args.parent_selection,
        extinction_interval=args.extinction_interval,
    )
    best = runner.run(
        argv=sys.argv,
        mutator_model_key=(
            args.mutator_model_key
            if not args.llm_provider and not args.mutator_model
            else None
        ),
        mutator_model=getattr(mutator_llm, "model", args.mutator_model),
        mutator_api_base=args.mutator_api_base or (
            _provider_defaults(args.llm_provider).get("api_base") if args.llm_provider else None
        ),
        mutator_api_key_env=args.mutator_api_key_env or (
            _provider_defaults(args.llm_provider).get("api_key_env") if args.llm_provider else None
        ),
        model_key=(
            args.model_key
            if args.generator_mode == "llm" and not args.llm_provider and not args.persona_model
            else None
        ),
        llm_provider=args.llm_provider,
        persona_model=getattr(gen_llm, "model", args.persona_model) if gen_llm else None,
        persona_api_base=args.persona_api_base or (
            _provider_defaults(args.llm_provider).get("api_base") if args.llm_provider else None
        ),
        persona_api_key_env=args.persona_api_key_env or (
            _provider_defaults(args.llm_provider).get("api_key_env") if args.llm_provider else None
        ),
        simulator_model_key=(
            args.simulator_model_key
            if args.simulator_model is None and not args.llm_provider
            else None
        ),
        simulator_model=getattr(sim_llm, "model", args.simulator_model),
        simulator_api_base=args.simulator_api_base or (
            _provider_defaults(args.llm_provider).get("api_base") if args.llm_provider else None
        ),
        simulator_api_key_env=args.simulator_api_key_env or (
            _provider_defaults(args.llm_provider).get("api_key_env") if args.llm_provider else None
        ),
    )
    logging.info("Saved MegaPersona OpenEvolve run to %s", output_dir)
    logging.info("Best candidate: %s", best.candidate_id)
    logging.info("Best fitness: %.4f", best.fitness or 0.0)
    logging.info("OpenEvolve checkpoint: %s", open_evolve_checkpoint)
    logging.info("MegaPersona evaluation dir: %s", output_dir / "mega_eval")
    logging.info("Final summary: %s", output_dir / "final_summary.md")
    logging.info("Run log: %s", output_dir / "run.log")


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/results") / f"mega_persona_evolution_{timestamp}"


def _validate_fixed_operator(operator_id: str) -> None:
    known = {operator["id"] for operator in EVOLUTION_PROMPT_OPERATORS}
    if operator_id not in known:
        print(
            f"Unknown --fixed-operator: {operator_id}\n"
            f"Known operators: {', '.join(sorted(known))}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _load_simulator_llm(args: argparse.Namespace) -> LLMClient:
    if args.llm_provider:
        provider_defaults = _provider_defaults(args.llm_provider)
        api_key_env = args.simulator_api_key_env or provider_defaults.get("api_key_env")
        api_base = args.simulator_api_base
        if api_base is None:
            api_base = provider_defaults.get("api_base")
        return LLMClient.from_provider(
            args.llm_provider,
            role="simulator",
            model=args.simulator_model,
            api_key_env=api_key_env,
            base_url=api_base,
        )

    api_key = os.environ.get(args.simulator_api_key_env) if args.simulator_api_key_env else None
    if args.simulator_model:
        return LLMClient(
            model=args.simulator_model,
            api_key=api_key,
            base_url=args.simulator_api_base,
            api_key_env=args.simulator_api_key_env or "OPENAI_API_KEY",
        )
    return LLMClient.from_config(
        args.simulator_model_key,
        api_key=api_key,
        base_url=args.simulator_api_base,
    )


def _load_mutator_llm(args: argparse.Namespace) -> LLMClient:
    if args.llm_provider:
        provider_defaults = _provider_defaults(args.llm_provider)
        api_key_env = args.mutator_api_key_env or provider_defaults.get("api_key_env")
        api_base = args.mutator_api_base
        if api_base is None:
            api_base = provider_defaults.get("api_base")
        return LLMClient.from_provider(
            args.llm_provider,
            role="mutator",
            model=args.mutator_model,
            api_key_env=api_key_env,
            base_url=api_base,
        )

    api_key = os.environ.get(args.mutator_api_key_env) if args.mutator_api_key_env else None
    if args.mutator_model:
        return LLMClient(
            model=args.mutator_model,
            api_key=api_key,
            base_url=args.mutator_api_base,
            api_key_env=args.mutator_api_key_env or "OPENAI_API_KEY",
        )
    return LLMClient.from_config(
        args.mutator_model_key,
        api_key=api_key,
        base_url=args.mutator_api_base,
    )


def _load_generation_llm(args: argparse.Namespace) -> LLMClient:
    if args.llm_provider:
        provider_defaults = _provider_defaults(args.llm_provider)
        api_key_env = args.persona_api_key_env or provider_defaults.get("api_key_env")
        api_base = args.persona_api_base
        if api_base is None:
            api_base = provider_defaults.get("api_base")
        return LLMClient.from_provider(
            args.llm_provider,
            role="persona",
            model=args.persona_model,
            api_key_env=api_key_env,
            base_url=api_base,
        )

    api_key = os.environ.get(args.persona_api_key_env) if args.persona_api_key_env else None
    if args.persona_model:
        return LLMClient(
            model=args.persona_model,
            api_key=api_key,
            base_url=args.persona_api_base,
            api_key_env=args.persona_api_key_env or "OPENAI_API_KEY",
        )
    return LLMClient.from_config(
        args.model_key,
        api_key=api_key,
        base_url=args.persona_api_base,
    )


def _provider_defaults(provider: str) -> dict[str, str | None]:
    if provider == "deepseek":
        return {
            "api_base": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
        }
    return {
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY",
    }


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
