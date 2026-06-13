"""Generate MegaPersona populations for the new experiment path."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona import (
    LLMShadowSimulator,
    MegaPersonaGenerator,
    RuleBasedMegaPersonaBuilder,
    SlotSampler,
    aggregate_shadow_behavior,
    build_initial_shadow_surveys,
    evaluate_mega_personas,
)
from src.utils.llm_client import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate schema-constrained MegaPersonas.")
    parser.add_argument("--n", type=int, default=25, help="Number of personas to generate.")
    parser.add_argument("--seed", type=int, default=17, help="Slot sampling seed.")
    parser.add_argument(
        "--model-key",
        default="llm.persona_model",
        help="Config key for the persona-generation LLM client.",
    )
    parser.add_argument(
        "--simulator-model-key",
        default="llm.simulator_model",
        help="Config key for the shadow-simulator LLM client.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to data/generated_personas/mega_personas_<timestamp>.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate slots and initial shadow surveys; do not call any LLM.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate rule-based baseline personas without calling an LLM for generation (simulator still uses LLM).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slots = SlotSampler().sample(n=args.n, seed=args.seed)
    shadow_surveys = build_initial_shadow_surveys()

    output_path = _output_path(args.output, args.dry_run)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        payload = {
            "mode": "dry_run",
            "created_at": datetime.now().isoformat(),
            "slots": [asdict(slot) for slot in slots],
            "shadow_surveys": [asdict(survey) for survey in shadow_surveys],
        }
        _write_json(output_path, payload)
        print(f"Saved dry-run MegaPersona experiment spec to {output_path}")
        return

    if args.mock:
        valid_personas = RuleBasedMegaPersonaBuilder().build_population(slots)
        generation_payload = {
            "personas": [persona.model_dump() for persona in valid_personas],
            "validation_reports": [],
            "model_key": None,
        }
        mode = "mock"
    else:
        gen_llm = LLMClient.from_config(args.model_key)
        generator = MegaPersonaGenerator(gen_llm)
        results = generator.generate_from_slots(slots)
        valid_personas = [result.persona for result in results if result.persona is not None]
        generation_payload = {
            "personas": [
                result.persona.model_dump() if result.persona else None
                for result in results
            ],
            "validation_reports": [
                {
                    "is_valid": result.validation_report.is_valid,
                    "schema_valid": result.validation_report.schema_valid,
                    "issues": [asdict(issue) for issue in result.validation_report.issues],
                }
                for result in results
            ],
            "model_key": args.model_key,
        }
        mode = "generated"

    evaluation = evaluate_mega_personas(valid_personas)

    # All shadow surveys are simulated via LLM — the only simulator now.
    sim_llm = LLMClient.from_config(args.simulator_model_key)
    shadow_simulations = LLMShadowSimulator(sim_llm).simulate_population(
        valid_personas, shadow_surveys,
    )
    shadow_behavior = aggregate_shadow_behavior(valid_personas, shadow_simulations)

    payload = {
        "mode": mode,
        "created_at": datetime.now().isoformat(),
        "model_key": generation_payload["model_key"],
        "slots": [asdict(slot) for slot in slots],
        "personas": generation_payload["personas"],
        "validation_reports": generation_payload["validation_reports"],
        "evaluation": evaluation.to_dict(),
        "shadow_behavior": shadow_behavior.to_dict(),
        "shadow_simulations": [asdict(simulation) for simulation in shadow_simulations],
        "shadow_surveys": [asdict(survey) for survey in shadow_surveys],
    }
    _write_json(output_path, payload)
    print(
        "Saved MegaPersona generation result to "
        f"{output_path} (valid={evaluation.valid_count}/{evaluation.sample_size}, "
        f"fitness={evaluation.fitness:.4f}, "
        f"shadow_alignment={shadow_behavior.overall_alignment:.4f})"
    )


def _output_path(path: str | None, dry_run: bool) -> Path:
    if path:
        return Path(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "mega_persona_spec" if dry_run else "mega_personas"
    return Path("data/generated_personas") / f"{stem}_{timestamp}.json"


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
