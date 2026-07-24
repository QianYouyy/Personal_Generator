"""Run sealed-test evaluation for the multi-objective best candidates of a finished run.

This is a post-hoc selection-rule diagnostic. It reuses the run's persisted
shadow-survey splits and evaluates each multi-objective best candidate
(global/coverage/diversity/strict/shadow-MAE/axis-target/schema/research) on the
sealed test split, so we can see whether global-fitness selection leaves better
generalizing candidates unselected. The sealed test is still never used for
selection: every report keeps ``test_used_for_selection: False``.

Example:

    python scripts/run_multi_objective_sealed_test.py \
        data/results/mega_persona_v3_structured_mcts_fixed_single_call_deepseek_n8_g15_seed23_20260719
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona.evolution import (
    MegaEvolutionConfig,
    MegaEvolutionCandidate,
    MegaPersonaEvolver,
)
from src.mega_persona.openevolve_adapter import multi_objective_best_candidates
from src.utils.llm_client import LLMClient
from src.utils.logger import logger

TEST_METRIC_COLUMNS = (
    "test_behavior_coverage.mean",
    "test_behavior_balanced_diversity.mean",
    "test_shadow_mae.mean",
    "test_strict_consistency_error.mean",
    "test_axis_target_mae.mean",
    "test_schema_fitness.mean",
    "test_internal_consistency.mean",
)

_CONFIG_FIELDS = {field.name for field in fields(MegaEvolutionConfig)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the multi-objective best candidates of a finished MegaPersona "
            "run on the sealed test split and write a side-by-side comparison."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Run dir (or its mega_eval dir).")
    parser.add_argument("--llm-provider", default=None, help="Override manifest llm_provider.")
    parser.add_argument("--simulator-model", default=None, help="Override manifest simulator model.")
    parser.add_argument("--persona-model", default=None, help="Override manifest persona model.")
    parser.add_argument(
        "--shadow-max-workers",
        type=int,
        default=None,
        help="Parallel shadow-survey simulation calls (defaults to the run config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the comparison reports. Defaults to the run directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mega_eval_dir = _find_mega_eval_dir(args.run_dir)
    run_dir = mega_eval_dir.parent if mega_eval_dir.name == "mega_eval" else mega_eval_dir
    manifest = _load_manifest(run_dir)
    config = _config_from_manifest(manifest, args)

    report = multi_objective_best_candidates(mega_eval_dir / "candidates")
    roles = report.get("roles", {})
    if not roles:
        raise SystemExit(f"No multi-objective candidates found under {mega_eval_dir / 'candidates'}")

    persona_client, simulator_client = _load_llm_clients(manifest, args)
    evolver = MegaPersonaEvolver(
        config=config,
        output_dir=mega_eval_dir,
        resume=True,
        llm_client=persona_client,
        simulator_llm_client=simulator_client,
    )
    manifest_hashes = manifest.get("shadow_survey_hashes")
    if isinstance(manifest_hashes, dict) and manifest_hashes != evolver.survey_hashes:
        logger.warning(
            "Survey split hashes differ from manifest; continuing with the persisted splits."
        )

    comparison = build_sealed_test_comparison(evolver, roles, mega_eval_dir / "candidates")
    comparison.update(
        {
            "run_dir": str(run_dir),
            "candidates_scanned": report.get("candidates_scanned", 0),
            "survey_hashes": evolver.survey_hashes,
            "test_used_for_selection": False,
        }
    )
    markdown = sealed_test_comparison_markdown(comparison)

    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "multi_objective_final_test.json"
    md_path = output_dir / "multi_objective_final_test.md"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def build_sealed_test_comparison(
    evolver: MegaPersonaEvolver,
    roles: dict,
    candidates_dir: Path,
) -> dict:
    """Evaluate every distinct role candidate on the sealed test split."""
    payloads = _load_candidate_payloads(candidates_dir)
    candidate_roles: dict[str, list[str]] = {}
    role_rows: dict[str, dict] = {}
    for role, item in roles.items():
        candidate_id = str(item.get("candidate_id", ""))
        if not candidate_id or candidate_id not in payloads:
            continue
        candidate_roles.setdefault(candidate_id, []).append(role)
        role_rows[candidate_id] = item

    entries = []
    for candidate_id, candidate_role_names in candidate_roles.items():
        payload = payloads[candidate_id]
        candidate = MegaEvolutionCandidate(
            candidate_id=candidate_id,
            genome=payload.get("genome", {}),
            generation=int(payload.get("generation", 0) or 0),
            parent_id=payload.get("parent_id"),
            fitness=payload.get("fitness"),
            metrics=payload.get("metrics", {}),
            evaluated=True,
        )
        logger.info(
            f"Sealed test for {candidate_id} ({', '.join(candidate_role_names)})"
        )
        test_report = evolver.evaluate_final_test(candidate)
        metrics = test_report.get("metrics", {})
        entries.append(
            {
                "candidate_id": candidate_id,
                "roles": candidate_role_names,
                "generation": candidate.generation,
                "operator_id": role_rows[candidate_id].get("operator_id"),
                "validation_fitness": payload.get("fitness"),
                "status": test_report.get("status", "ok"),
                "test_metrics": {
                    key: metrics.get(key)
                    for key in TEST_METRIC_COLUMNS
                    if metrics.get(key) is not None
                },
            }
        )
    return {"entries": entries}


def sealed_test_comparison_markdown(comparison: dict) -> str:
    """Render the side-by-side sealed-test comparison table."""
    lines = [
        "# Multi-Objective Sealed-Test Comparison",
        "",
        f"- run: `{comparison.get('run_dir', '')}`",
        f"- candidates scanned: `{int(comparison.get('candidates_scanned', 0))}`",
        "- test used for selection: `False`",
        "",
        "| Roles | Candidate | Gen | Operator | Val fitness | Test coverage | Test diversity | Test shadow MAE | Test strict err | Test axis MAE | Status |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in comparison.get("entries", []):
        metrics = entry.get("test_metrics", {})
        lines.append(
            "| "
            f"{', '.join(entry.get('roles', []))} | "
            f"`{entry.get('candidate_id', '')}` | "
            f"{int(entry.get('generation', 0))} | "
            f"`{entry.get('operator_id') or ''}` | "
            f"{_fmt(entry.get('validation_fitness'))} | "
            f"{_fmt(metrics.get('test_behavior_coverage.mean'))} | "
            f"{_fmt(metrics.get('test_behavior_balanced_diversity.mean'))} | "
            f"{_fmt(metrics.get('test_shadow_mae.mean'))} | "
            f"{_fmt(metrics.get('test_strict_consistency_error.mean'))} | "
            f"{_fmt(metrics.get('test_axis_target_mae.mean'))} | "
            f"{entry.get('status', '')} |"
        )
    return "\n".join(lines) + "\n"


def _load_candidate_payloads(candidates_dir: Path) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    if not candidates_dir.exists():
        return payloads
    for path in sorted(candidates_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate_id = payload.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            payloads[candidate_id] = payload
    return payloads


def _find_mega_eval_dir(path: Path) -> Path:
    if path.name == "mega_eval" and (path / "checkpoint.json").exists():
        return path
    if (path / "mega_eval" / "checkpoint.json").exists():
        return path / "mega_eval"
    if (path / "checkpoint.json").exists():
        return path
    raise FileNotFoundError(
        f"Could not find mega_eval checkpoint under {path}. Expected a finished run directory."
    )


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _config_from_manifest(manifest: dict, args: argparse.Namespace) -> MegaEvolutionConfig:
    raw = manifest.get("config", {})
    config_kwargs = {key: value for key, value in raw.items() if key in _CONFIG_FIELDS}
    if "seeds" in config_kwargs:
        config_kwargs["seeds"] = tuple(int(seed) for seed in config_kwargs["seeds"])
    config = MegaEvolutionConfig(**config_kwargs)
    if args.shadow_max_workers:
        config = replace(config, shadow_max_workers=int(args.shadow_max_workers))
    return config


def _load_llm_clients(manifest: dict, args: argparse.Namespace) -> tuple[LLMClient, LLMClient]:
    provider = args.llm_provider or manifest.get("llm_provider")
    if provider:
        simulator_client = LLMClient.from_provider(
            provider,
            role="simulator",
            model=args.simulator_model or manifest.get("simulator_model"),
            api_key_env=manifest.get("simulator_api_key_env"),
            base_url=manifest.get("simulator_api_base"),
        )
        persona_client = LLMClient.from_provider(
            provider,
            role="persona",
            model=args.persona_model or manifest.get("persona_model"),
            api_key_env=manifest.get("persona_api_key_env") or manifest.get("simulator_api_key_env"),
            base_url=manifest.get("persona_api_base") or manifest.get("simulator_api_base"),
        )
        return persona_client, simulator_client
    logger.info("manifest has no llm_provider; falling back to config-file model keys")
    return (
        LLMClient.from_config("llm.persona_model"),
        LLMClient.from_config("llm.simulator_model"),
    )


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
