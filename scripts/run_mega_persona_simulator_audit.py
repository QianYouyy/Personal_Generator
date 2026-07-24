"""Offline A/B/C audit for MegaPersona shadow-survey simulators.

Experiment 1 freezes one persona set and one shadow-survey set, then asks
multiple simulator backends to answer the exact same persona-survey matrix.
This isolates simulator quality from OpenEvolve selection dynamics.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator.metrics import DiversityMetrics
from src.mega_persona.evaluation import evaluate_mega_personas, personas_to_axis_matrix
from src.mega_persona.generator import MegaPersonaGenerator
from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_simulator import (
    SUPPORTED_SHADOW_SIMULATOR_BACKENDS,
    ShadowSurveySimulation,
    aggregate_shadow_behavior,
    build_shadow_simulator,
    shadow_behavior_axis_matrix,
)
from src.mega_persona.shadow_survey import (
    ShadowSurvey,
    build_initial_shadow_surveys,
)
from src.mega_persona.slots import AXIS_NAMES, SlotSampler
from src.mega_persona.template_generator import RuleBasedMegaPersonaBuilder
from src.utils.llm_client import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen offline audit across MegaPersona simulator backends."
    )
    parser.add_argument("--persona-mode", choices=["mock", "llm"], default="mock")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--persona-seed", type=int, default=17)
    parser.add_argument("--survey-seed", type=int, default=17017)
    parser.add_argument("--shadow-surveys", type=int, default=4)
    parser.add_argument("--items-per-shadow-survey", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--backends",
        default="llm,concordia,concordia-native",
        help="Comma-separated simulator backends.",
    )
    parser.add_argument("--model-key", default="llm.persona_model")
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "deepseek"],
        default=None,
        help="Optional OpenAI-compatible provider preset for persona and simulator models.",
    )
    parser.add_argument("--persona-model", default=None)
    parser.add_argument("--persona-api-base", default=None)
    parser.add_argument("--persona-api-key-env", default=None)
    parser.add_argument("--simulator-model-key", default="llm.simulator_model")
    parser.add_argument("--simulator-model", default=None)
    parser.add_argument("--simulator-api-base", default=None)
    parser.add_argument("--simulator-api-key-env", default=None)
    parser.add_argument("--persona-max-workers", type=int, default=1)
    parser.add_argument("--shadow-max-workers", type=int, default=1)
    parser.add_argument("--coverage-radius", type=float, default=0.28)
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument(
        "--persona-json",
        default=None,
        help="Optional frozen persona JSON. Supports a list or {'personas': [...]} payload.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/results/mega_persona_simulator_audit_<timestamp>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    _setup_logging(output_dir)

    started_at = time.time()
    backends = _parse_backends(args.backends)
    logging.info(
        "MegaPersona simulator audit output_dir=%s persona_mode=%s n=%s backends=%s repeats=%s",
        output_dir,
        args.persona_mode,
        args.n,
        ",".join(backends),
        args.repeats,
    )

    personas = _load_or_generate_personas(args, output_dir)
    surveys = build_initial_shadow_surveys(
        num_surveys=args.shadow_surveys,
        items_per_survey=args.items_per_shadow_survey,
        seed=args.survey_seed,
        split="audit",
        survey_id_prefix="shadow_audit",
    )
    _write_frozen_inputs(output_dir, personas, surveys, args)

    simulator_llm = _load_simulator_llm(args)
    backend_results: dict[str, Any] = {}
    for backend in backends:
        backend_results[backend] = _run_backend(
            backend=backend,
            personas=personas,
            surveys=surveys,
            repeats=max(1, args.repeats),
            llm_client=simulator_llm,
            max_workers=args.shadow_max_workers,
            coverage_radius=args.coverage_radius,
            output_dir=output_dir,
        )

    summary = {
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.time() - started_at,
        "config": _config_payload(args, backends),
        "frozen_inputs": {
            "persona_count": len(personas),
            "survey_count": len(surveys),
            "item_count": sum(len(survey.items) for survey in surveys),
            "persona_sha256": _sha256_json([persona.model_dump() for persona in personas]),
            "survey_sha256": _sha256_json([asdict(survey) for survey in surveys]),
        },
        "persona_schema_evaluation": evaluate_mega_personas(
            personas,
            coverage_radius=args.coverage_radius,
            duplicate_threshold=args.duplicate_threshold,
        ).to_dict(),
        "backend_results": backend_results,
        "paired_backend_deltas": _paired_backend_deltas(backend_results),
    }
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "summary.md"
    _write_json(summary_path, summary)
    report_path.write_text(_to_markdown(summary), encoding="utf-8")

    logging.info("Saved simulator audit summary JSON to %s", summary_path)
    logging.info("Saved simulator audit report to %s", report_path)
    logging.info("Run log: %s", output_dir / "run.log")
    print(f"Saved simulator audit summary JSON to {summary_path}")
    print(f"Saved simulator audit report to {report_path}")


def _load_or_generate_personas(args: argparse.Namespace, output_dir: Path) -> list[MegaPersona]:
    if args.persona_json:
        path = Path(args.persona_json)
        logging.info("Loading frozen personas from %s", path)
        data = json.loads(path.read_text(encoding="utf-8"))
        payload = data.get("personas", data) if isinstance(data, dict) else data
        personas = [MegaPersona.model_validate(item) for item in payload]
        if args.n and len(personas) > args.n:
            personas = personas[: args.n]
        logging.info("Loaded %s frozen personas", len(personas))
        return personas

    slots = SlotSampler().sample(n=args.n, seed=args.persona_seed)
    _write_json(output_dir / "frozen_slots.json", [asdict(slot) for slot in slots])
    if args.persona_mode == "mock":
        personas = RuleBasedMegaPersonaBuilder().build_population(slots)
    else:
        logging.info("Loading persona-generation LLM client")
        llm = _load_persona_llm(args)
        results = MegaPersonaGenerator(llm).generate_from_slots(
            slots,
            max_workers=args.persona_max_workers,
        )
        personas = [result.persona for result in results if result.persona is not None]
    logging.info("Generated frozen persona set valid=%s requested=%s", len(personas), len(slots))
    return personas


def _run_backend(
    *,
    backend: str,
    personas: list[MegaPersona],
    surveys: list[ShadowSurvey],
    repeats: int,
    llm_client,
    max_workers: int,
    coverage_radius: float,
    output_dir: Path,
) -> dict[str, Any]:
    logging.info("Starting backend=%s repeats=%s", backend, repeats)
    backend_dir = output_dir / "simulations" / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    repeat_payloads: list[dict[str, Any]] = []
    all_repeat_simulations: list[list[ShadowSurveySimulation]] = []

    for repeat_index in range(1, repeats + 1):
        repeat_started = time.time()
        logging.info("Backend=%s repeat=%s/%s start", backend, repeat_index, repeats)
        simulator = build_shadow_simulator(
            backend=backend,
            llm_client=llm_client,
            max_workers=max_workers,
        )
        simulations = simulator.simulate_population(personas, surveys)
        elapsed = time.time() - repeat_started
        all_repeat_simulations.append(simulations)
        payload = {
            "backend": backend,
            "repeat": repeat_index,
            "elapsed_seconds": elapsed,
            "simulations": [asdict(simulation) for simulation in simulations],
        }
        _write_json(backend_dir / f"repeat_{repeat_index:02d}.json", payload)
        repeat_metrics = _metrics_for_simulations(
            personas=personas,
            surveys=surveys,
            simulations=simulations,
            coverage_radius=coverage_radius,
            elapsed_seconds=elapsed,
        )
        repeat_payloads.append({"repeat": repeat_index, **repeat_metrics})
        logging.info(
            "Backend=%s repeat=%s done elapsed=%.1fs coverage=%.4f alignment=%.4f valid=%.4f",
            backend,
            repeat_index,
            elapsed,
            repeat_metrics["behavior_diversity"].get("coverage", 0.0),
            repeat_metrics["behavior_alignment"]["overall_alignment"],
            repeat_metrics["response_quality"]["complete_response_rate"],
        )

    aggregate = _aggregate_repeat_metrics(repeat_payloads)
    stability = _stability_metrics(all_repeat_simulations, surveys) if repeats > 1 else {}
    return {
        "repeats": repeat_payloads,
        "aggregate": aggregate,
        "stability": stability,
    }


def _metrics_for_simulations(
    *,
    personas: list[MegaPersona],
    surveys: list[ShadowSurvey],
    simulations: list[ShadowSurveySimulation],
    coverage_radius: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    behavior_report = aggregate_shadow_behavior(personas, simulations)
    behavior_matrix = shadow_behavior_axis_matrix(personas, simulations)
    behavior_diversity = DiversityMetrics(coverage_radius=coverage_radius).fitness(behavior_matrix)
    persona_matrix = personas_to_axis_matrix(personas)
    correlations = _axis_correlations(persona_matrix, behavior_matrix)
    response_quality = _response_quality(surveys, simulations)
    discrimination = _discrimination_metrics(behavior_matrix)
    calls = len(personas) * len(surveys)
    item_total = calls * (len(surveys[0].items) if surveys else 0)
    return {
        "elapsed_seconds": elapsed_seconds,
        "calls": calls,
        "items": item_total,
        "seconds_per_call": elapsed_seconds / calls if calls else 0.0,
        "seconds_per_item": elapsed_seconds / item_total if item_total else 0.0,
        "behavior_alignment": behavior_report.to_dict(),
        "behavior_diversity": behavior_diversity,
        "axis_correlations": correlations,
        "response_quality": response_quality,
        "behavior_discrimination": discrimination,
    }


def _response_quality(
    surveys: list[ShadowSurvey],
    simulations: list[ShadowSurveySimulation],
) -> dict[str, Any]:
    survey_by_id = {survey.survey_id: survey for survey in surveys}
    total_expected = 0
    total_present = 0
    valid_values = 0
    response_counts = {str(score): 0 for score in range(1, 6)}
    for simulation in simulations:
        survey = survey_by_id[simulation.survey_id]
        expected = survey.item_ids()
        total_expected += len(expected)
        for item_id in expected:
            if item_id not in simulation.responses:
                continue
            total_present += 1
            value = simulation.responses[item_id]
            if value in range(1, 6):
                valid_values += 1
                response_counts[str(value)] += 1

    entropy = _normalized_entropy(list(response_counts.values()))
    neutral = response_counts["3"] / valid_values if valid_values else 0.0
    extreme = (response_counts["1"] + response_counts["5"]) / valid_values if valid_values else 0.0
    return {
        "expected_responses": total_expected,
        "present_responses": total_present,
        "complete_response_rate": total_present / total_expected if total_expected else 0.0,
        "valid_value_rate": valid_values / total_present if total_present else 0.0,
        "response_counts": response_counts,
        "response_entropy": entropy,
        "neutral_rate": neutral,
        "extreme_rate": extreme,
    }


def _discrimination_metrics(matrix: np.ndarray) -> dict[str, float]:
    if len(matrix) == 0:
        return {"axis_std_mean": 0.0, "mean_pairwise_distance": 0.0, "min_pairwise_distance": 0.0}
    distances = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            distances.append(float(np.linalg.norm(matrix[i] - matrix[j])))
    return {
        "axis_std_mean": float(np.mean(np.std(matrix, axis=0))),
        "mean_pairwise_distance": float(np.mean(distances)) if distances else 0.0,
        "min_pairwise_distance": float(np.min(distances)) if distances else 0.0,
    }


def _axis_correlations(persona_matrix: np.ndarray, behavior_matrix: np.ndarray) -> dict[str, float]:
    correlations: dict[str, float] = {}
    if len(persona_matrix) != len(behavior_matrix) or len(persona_matrix) < 2:
        return {axis: 0.0 for axis in AXIS_NAMES}
    for idx, axis in enumerate(AXIS_NAMES):
        left = persona_matrix[:, idx]
        right = behavior_matrix[:, idx]
        if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
            correlations[axis] = 0.0
        else:
            correlations[axis] = float(np.corrcoef(left, right)[0, 1])
    return correlations


def _stability_metrics(
    repeats: list[list[ShadowSurveySimulation]],
    surveys: list[ShadowSurvey],
) -> dict[str, float]:
    if len(repeats) <= 1:
        return {}
    survey_items = {survey.survey_id: survey.item_ids() for survey in surveys}
    repeat_maps = [_simulation_map(simulations) for simulations in repeats]
    response_consistency_scores: list[float] = []
    axis_consistency_scores: list[float] = []
    for i in range(len(repeat_maps)):
        for j in range(i + 1, len(repeat_maps)):
            shared_keys = sorted(set(repeat_maps[i]) & set(repeat_maps[j]))
            for key in shared_keys:
                left = repeat_maps[i][key]
                right = repeat_maps[j][key]
                item_ids = survey_items[key[1]]
                for item_id in item_ids:
                    if item_id in left.responses and item_id in right.responses:
                        response_consistency_scores.append(
                            1.0 - abs(left.responses[item_id] - right.responses[item_id]) / 4.0
                        )
                axis_delta = np.mean(
                    [
                        abs(left.axis_scores[axis] - right.axis_scores[axis])
                        for axis in AXIS_NAMES
                    ]
                )
                axis_consistency_scores.append(float(1.0 - np.clip(axis_delta, 0.0, 1.0)))
    return {
        "response_consistency": float(np.mean(response_consistency_scores))
        if response_consistency_scores
        else 0.0,
        "axis_score_consistency": float(np.mean(axis_consistency_scores))
        if axis_consistency_scores
        else 0.0,
    }


def _simulation_map(
    simulations: list[ShadowSurveySimulation],
) -> dict[tuple[str, str], ShadowSurveySimulation]:
    return {
        (simulation.persona_id, simulation.survey_id): simulation
        for simulation in simulations
    }


def _aggregate_repeat_metrics(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "behavior_coverage": [
            item["behavior_diversity"].get("coverage", 0.0)
            for item in repeats
        ],
        "overall_alignment": [
            item["behavior_alignment"]["overall_alignment"]
            for item in repeats
        ],
        "complete_response_rate": [
            item["response_quality"]["complete_response_rate"]
            for item in repeats
        ],
        "valid_value_rate": [
            item["response_quality"]["valid_value_rate"]
            for item in repeats
        ],
        "response_entropy": [
            item["response_quality"]["response_entropy"]
            for item in repeats
        ],
        "neutral_rate": [
            item["response_quality"]["neutral_rate"]
            for item in repeats
        ],
        "axis_std_mean": [
            item["behavior_discrimination"]["axis_std_mean"]
            for item in repeats
        ],
        "mean_pairwise_distance": [
            item["behavior_discrimination"]["mean_pairwise_distance"]
            for item in repeats
        ],
        "seconds_per_call": [
            item["seconds_per_call"]
            for item in repeats
        ],
    }
    aggregate = {}
    for key, values in keys.items():
        aggregate[key] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "std": float(np.std(values)) if values else 0.0,
            "ci95": _bootstrap_ci(values),
        }
    aggregate["axis_correlation_mean"] = _mean_axis_correlations(repeats)
    return aggregate


def _mean_axis_correlations(repeats: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for axis in AXIS_NAMES:
        values = [item["axis_correlations"].get(axis, 0.0) for item in repeats]
        result[axis] = float(np.mean(values)) if values else 0.0
    return result


def _paired_backend_deltas(backend_results: dict[str, Any]) -> dict[str, Any]:
    """Report simple metric deltas against the first backend as a reference."""
    if not backend_results:
        return {}
    reference = next(iter(backend_results))
    ref_agg = backend_results[reference]["aggregate"]
    deltas = {"reference": reference, "deltas": {}}
    for backend, result in backend_results.items():
        if backend == reference:
            continue
        backend_delta = {}
        for metric in ("behavior_coverage", "overall_alignment", "response_entropy", "axis_std_mean"):
            backend_delta[metric] = (
                result["aggregate"][metric]["mean"] - ref_agg[metric]["mean"]
            )
        deltas["deltas"][backend] = backend_delta
    return deltas


def _bootstrap_ci(values: list[float], trials: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(12345)
    samples = []
    arr = np.array(values, dtype=float)
    for _ in range(trials):
        samples.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = np.array([count / total for count in counts if count > 0], dtype=float)
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / float(np.log(len(counts))) if len(counts) > 1 else 0.0


def _to_markdown(summary: dict[str, Any]) -> str:
    config = summary["config"]
    frozen = summary["frozen_inputs"]
    lines = [
        "# MegaPersona Simulator Audit",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- persona_mode: `{config['persona_mode']}`",
        f"- personas: `{frozen['persona_count']}`",
        f"- surveys: `{frozen['survey_count']}`",
        f"- items per survey: `{config['items_per_shadow_survey']}`",
        f"- repeats: `{config['repeats']}`",
        f"- backends: `{', '.join(config['backends'])}`",
        f"- persona_sha256: `{frozen['persona_sha256']}`",
        f"- survey_sha256: `{frozen['survey_sha256']}`",
        "",
        "## Backend Comparison",
        "",
        "| Backend | Coverage | Alignment | Complete | Entropy | Neutral | Axis Std | Sec/Call | Stability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, result in summary["backend_results"].items():
        agg = result["aggregate"]
        stability = result.get("stability", {})
        lines.append(
            f"| {backend} | "
            f"{agg['behavior_coverage']['mean']:.4f} | "
            f"{agg['overall_alignment']['mean']:.4f} | "
            f"{agg['complete_response_rate']['mean']:.4f} | "
            f"{agg['response_entropy']['mean']:.4f} | "
            f"{agg['neutral_rate']['mean']:.4f} | "
            f"{agg['axis_std_mean']['mean']:.4f} | "
            f"{agg['seconds_per_call']['mean']:.2f} | "
            f"{stability.get('axis_score_consistency', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Axis Correlations",
            "",
            "| Backend | " + " | ".join(AXIS_NAMES) + " |",
            "|---" + "|---:" * len(AXIS_NAMES) + "|",
        ]
    )
    for backend, result in summary["backend_results"].items():
        corr = result["aggregate"]["axis_correlation_mean"]
        lines.append(
            f"| {backend} | "
            + " | ".join(f"{corr.get(axis, 0.0):.4f}" for axis in AXIS_NAMES)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- Complete/valid response rate should be near 1.0 before any quality conclusion.",
            "- Coverage and axis std indicate behavioral spread; very low values suggest collapsed responses.",
            "- Alignment and axis correlations indicate persona-behavior consistency.",
            "- Stability is only meaningful when repeats > 1.",
            "- Final conclusions should use the same frozen persona and survey hashes above.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_frozen_inputs(
    output_dir: Path,
    personas: list[MegaPersona],
    surveys: list[ShadowSurvey],
    args: argparse.Namespace,
) -> None:
    _write_json(output_dir / "frozen_personas.json", [persona.model_dump() for persona in personas])
    _write_json(output_dir / "frozen_surveys.json", [asdict(survey) for survey in surveys])
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "config": _config_payload(args, _parse_backends(args.backends)),
            "persona_sha256": _sha256_json([persona.model_dump() for persona in personas]),
            "survey_sha256": _sha256_json([asdict(survey) for survey in surveys]),
        },
    )


def _config_payload(args: argparse.Namespace, backends: list[str]) -> dict[str, Any]:
    return {
        "persona_mode": args.persona_mode,
        "n": args.n,
        "persona_seed": args.persona_seed,
        "survey_seed": args.survey_seed,
        "shadow_surveys": args.shadow_surveys,
        "items_per_shadow_survey": args.items_per_shadow_survey,
        "repeats": args.repeats,
        "backends": backends,
        "model_key": args.model_key if args.persona_mode == "llm" else None,
        "simulator_model_key": args.simulator_model_key if args.simulator_model is None else None,
        "simulator_model": args.simulator_model,
        "persona_max_workers": args.persona_max_workers,
        "shadow_max_workers": args.shadow_max_workers,
        "coverage_radius": args.coverage_radius,
        "duplicate_threshold": args.duplicate_threshold,
        "persona_json": args.persona_json,
    }


def _parse_backends(raw: str) -> list[str]:
    backends = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [backend for backend in backends if backend not in SUPPORTED_SHADOW_SIMULATOR_BACKENDS]
    if unknown:
        raise ValueError(
            f"Unknown backend(s): {', '.join(unknown)}. "
            f"Expected one of {', '.join(SUPPORTED_SHADOW_SIMULATOR_BACKENDS)}."
        )
    if not backends:
        raise ValueError("at least one backend is required")
    return backends


def _load_simulator_llm(args: argparse.Namespace) -> LLMClient:
    import os

    if args.llm_provider:
        provider_defaults = _provider_defaults(args.llm_provider)
        return LLMClient.from_provider(
            args.llm_provider,
            role="simulator",
            model=args.simulator_model,
            api_key_env=args.simulator_api_key_env or provider_defaults.get("api_key_env"),
            base_url=args.simulator_api_base or provider_defaults.get("api_base"),
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


def _load_persona_llm(args: argparse.Namespace) -> LLMClient:
    import os

    if args.llm_provider:
        provider_defaults = _provider_defaults(args.llm_provider)
        return LLMClient.from_provider(
            args.llm_provider,
            role="persona",
            model=args.persona_model,
            api_key_env=args.persona_api_key_env or provider_defaults.get("api_key_env"),
            base_url=args.persona_api_base or provider_defaults.get("api_base"),
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/results") / f"mega_persona_simulator_audit_{timestamp}"


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
