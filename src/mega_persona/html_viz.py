"""Generate a self-contained interactive HTML report for MegaPersona results.

The report uses Plotly.js (loaded from CDN) for interactive charts and embeds
all data as JSON so it can be opened directly in a browser with no server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.mega_persona.schema import MegaPersona
from src.mega_persona.slots import AXIS_NAMES, axis_names_for_binding, schema_binding_for_genome

_TEMPLATE_PATH = Path(__file__).parent / "html_template.html"

# Embedded fallback template (minimal, used when the template file is missing)
_EMBEDDED_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{{TITLE}}</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script></head>
<body><pre id="data">{{JSON_DATA}}</pre></body></html>"""

MAX_SCATTER_POINTS = 500


def generate_html_report(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained HTML visualization report.

    Args:
        input_path: Evolution run directory, batch summary.json, or single JSON file.
        output_path: Where to write the HTML file. Defaults to ``<input>/report.html``.

    Returns:
        The path to the generated HTML file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = (
            input_path / "report.html"
            if input_path.is_dir()
            else input_path.parent / "report.html"
        )
    else:
        output_path = Path(output_path)

    data = _extract_viz_data(input_path)
    template = _load_template()
    html = template.replace("{{TITLE}}", _title_from_path(input_path))
    # Escape </script> sequences that could appear in persona narratives
    json_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    json_str = json_str.replace("</", "<\\/")
    html = html.replace("{{JSON_DATA}}", json_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _title_from_path(path: Path) -> str:
    if path.is_dir():
        return path.name
    return path.stem


def _load_template() -> str:
    if _TEMPLATE_PATH.exists():
        return _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _EMBEDDED_TEMPLATE


# ---------------------------------------------------------------------------
# Data extraction dispatcher
# ---------------------------------------------------------------------------


def _extract_viz_data(input_path: Path) -> dict[str, Any]:
    if input_path.is_dir() and (input_path / "final_summary.json").exists():
        return _extract_evolution_data(input_path)
    if input_path.is_dir() and (input_path / "summary.json").exists():
        return _extract_experiment_data(input_path / "summary.json")
    if input_path.is_file() and input_path.suffix == ".json":
        return _extract_generation_data(input_path)
    raise FileNotFoundError(f"Unsupported MegaPersona result path: {input_path}")


# ---------------------------------------------------------------------------
# Evolution run
# ---------------------------------------------------------------------------


def _extract_evolution_data(evo_dir: Path) -> dict[str, Any]:
    final = _read_json(evo_dir / "final_summary.json")
    best_id = final["best"]["candidate_id"]
    best_result = _find_evaluation_result(evo_dir, best_id)

    # Fitness history from generation files
    fitness_history = _build_fitness_history(evo_dir)

    # Scatter data from best candidate's first seed result
    scatter_data = _build_scatter_data(best_result)

    # Genome, metrics, per-seed from best result
    genome = final["best"].get("genome", {})
    metrics = final["best"].get("metrics", {})
    per_seed = _build_per_seed(best_result)

    return {
        "title": evo_dir.name,
        "type": "evolution",
        "fitness_history": fitness_history,
        "scatter_data": scatter_data,
        "genome": genome,
        "metrics": metrics,
        "per_seed": per_seed,
        "config": final.get("config", {}),
        "run_info": {
            "best_fitness": final["best"].get("fitness"),
            "best_candidate_id": best_id,
            "completed_at": final.get("completed_at", ""),
            "generations": final.get("config", {}).get("generations", 0),
        },
    }


def _build_fitness_history(evo_dir: Path) -> list[dict[str, Any]]:
    gen_dir = evo_dir / "generations"
    history: list[dict[str, Any]] = []
    if not gen_dir.exists():
        return history
    for path in sorted(gen_dir.glob("generation_*.json")):
        gen_data = _read_json(path)
        entry: dict[str, Any] = {
            "generation": gen_data.get("generation", 0),
            "best_fitness": gen_data.get("best_fitness"),
        }
        # If there's sub-metric data on the best candidate for this generation,
        # include it so the fitness curve can show per-metric traces.
        best_id = gen_data.get("best_candidate_id")
        if best_id:
            for pop in gen_data.get("population", []):
                if pop.get("candidate_id") == best_id:
                    entry["best_candidate"] = pop
                    break
        history.append(entry)
    return history


def _build_scatter_data(best_result: dict[str, Any] | None) -> dict[str, Any]:
    axis_names = _infer_axis_names(
        genome=(best_result or {}).get("candidate", {}).get("genome"),
        slots=((best_result or {}).get("per_seed", [{}])[0].get("slots", []) if best_result and best_result.get("per_seed") else []),
    )
    empty = {
        "slot_axes": [],
        "persona_axes": [],
        "behavior_axes": [],
        "axis_names": list(axis_names),
    }
    if not best_result:
        return empty
    per_seed = best_result.get("per_seed", [])
    if not per_seed:
        return empty
    first = per_seed[0]

    slot_axes = _extract_slot_axes(first.get("slots", []), axis_names=axis_names)
    persona_axes = _extract_persona_axes(first.get("personas", []), axis_names=axis_names)
    behavior_axes = _extract_behavior_axes(first, axis_names=axis_names)

    # Truncate if too large
    if len(persona_axes) > MAX_SCATTER_POINTS:
        persona_axes = persona_axes[:MAX_SCATTER_POINTS]
    if len(behavior_axes) > MAX_SCATTER_POINTS:
        behavior_axes = behavior_axes[:MAX_SCATTER_POINTS]

    return {
        "slot_axes": slot_axes,
        "persona_axes": persona_axes,
        "behavior_axes": behavior_axes,
        "axis_names": list(axis_names),
    }


def _build_per_seed(best_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not best_result:
        return []
    per_seed = best_result.get("per_seed", [])
    result: list[dict[str, Any]] = []
    for seed_entry in per_seed:
        schema_ev = seed_entry.get("schema_evaluation", {})
        heldout = seed_entry.get("validation_shadow_behavior") or seed_entry.get("heldout_shadow_behavior", {})
        hd_div = seed_entry.get("validation_behavior_diversity") or seed_entry.get("heldout_behavior_diversity", {})
        result.append({
            "seed": seed_entry.get("seed"),
            "score": seed_entry.get("score"),
            "schema_fitness": schema_ev.get("fitness"),
            "validity_rate": schema_ev.get("validity_rate"),
            "near_duplicate_rate": schema_ev.get("near_duplicate_rate"),
            "shadow_alignment": heldout.get("overall_alignment"),
            "behavior_coverage": hd_div.get("coverage"),
            "slot_coverage": seed_entry.get("slot_diversity", {}).get("coverage"),
            "scatter": {
                "slot_axes": _extract_slot_axes(seed_entry.get("slots", [])),
                "persona_axes": _extract_persona_axes(seed_entry.get("personas", [])),
                "behavior_axes": _extract_behavior_axes(seed_entry),
            },
        })
    return result


# ---------------------------------------------------------------------------
# Batch experiment (summary.json)
# ---------------------------------------------------------------------------


def _extract_experiment_data(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    runs = summary.get("runs", [])
    first = runs[0] if runs else {}
    axis_names = _infer_axis_names(genome=first.get("genome"), slots=first.get("slots", []))

    slot_axes = _extract_slot_axes(first.get("slots", []), axis_names=axis_names)
    persona_data = first.get("personas", [])
    persona_axes = _extract_persona_axes(persona_data, axis_names=axis_names)
    behavior_axes = _extract_behavior_axes_from_sims(
        persona_data,
        first.get("shadow_simulations", []),
        axis_names=axis_names,
    )

    per_seed: list[dict[str, Any]] = []
    for run in runs:
        schema_ev = run.get("schema_evaluation", {})
        shadow = run.get("shadow_behavior", {})
        beh_div = run.get("behavior_diversity_metrics", {})
        per_seed.append({
            "seed": run.get("seed"),
            "score": run.get("experiment_score"),
            "schema_fitness": schema_ev.get("fitness"),
            "validity_rate": schema_ev.get("validity_rate"),
            "near_duplicate_rate": schema_ev.get("near_duplicate_rate"),
            "shadow_alignment": shadow.get("overall_alignment"),
            "behavior_coverage": beh_div.get("coverage"),
            "slot_coverage": run.get("slot_diversity_metrics", {}).get("coverage"),
        })

    return {
        "title": summary_path.parent.name if summary_path.parent.name else "experiment",
        "type": "experiment",
        "fitness_history": [],
        "scatter_data": {
            "slot_axes": slot_axes,
            "persona_axes": persona_axes,
            "behavior_axes": behavior_axes,
            "axis_names": list(axis_names),
        },
        "genome": {},
        "metrics": summary.get("aggregate", {}),
        "per_seed": per_seed,
        "config": summary.get("config", {}),
        "run_info": {
            "best_fitness": per_seed[0]["score"] if per_seed else 0,
            "best_candidate_id": "batch_experiment",
            "completed_at": first.get("created_at", ""),
            "n_runs": len(runs),
        },
    }


# ---------------------------------------------------------------------------
# Single JSON / checkpoint
# ---------------------------------------------------------------------------


def _extract_generation_data(json_path: Path) -> dict[str, Any]:
    payload = _read_json(json_path)

    # Check if it's a checkpoint-style payload
    if "history" in payload:
        fitness_history = [
            {"generation": i, "best_fitness": fit}
            for i, fit in enumerate(payload.get("history", []))
        ]
        best = payload.get("best", {})
    else:
        fitness_history = []
        best = payload

    axis_names = _infer_axis_names(genome=best.get("genome"), slots=payload.get("slots", []))
    slots = payload.get("slots", [])
    personas = payload.get("personas", [])
    sims = payload.get("shadow_simulations", [])
    slot_axes = _extract_slot_axes(slots, axis_names=axis_names)
    persona_axes = _extract_persona_axes(personas, axis_names=axis_names)
    behavior_axes = _extract_behavior_axes_from_sims(personas, sims, axis_names=axis_names)

    return {
        "title": json_path.stem,
        "type": "generation",
        "fitness_history": fitness_history,
        "scatter_data": {
            "slot_axes": slot_axes,
            "persona_axes": persona_axes,
            "behavior_axes": behavior_axes,
            "axis_names": list(axis_names),
        },
        "genome": best.get("genome", {}),
        "metrics": payload.get("evaluation", payload.get("metrics", {})),
        "per_seed": [],
        "config": payload.get("config", {}),
        "run_info": {
            "best_fitness": best.get("fitness", payload.get("experiment_score", 0)),
            "best_candidate_id": best.get("candidate_id", json_path.stem),
        },
    }


# ---------------------------------------------------------------------------
# Axis extraction helpers (mirrors visualization.py logic)
# ---------------------------------------------------------------------------


def _extract_slot_axes(
    slots: list[dict[str, Any]],
    axis_names: tuple[str, ...] = AXIS_NAMES,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for slot in slots:
        target = slot.get("target_axes", {})
        rows.append([float(target.get(axis, 0.5)) for axis in axis_names])
    if not rows:
        return []
    # Truncate
    if len(rows) > MAX_SCATTER_POINTS:
        rows = rows[:MAX_SCATTER_POINTS]
    return rows


def _extract_persona_axes(
    personas: list[dict[str, Any]],
    axis_names: tuple[str, ...] = AXIS_NAMES,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for data in personas:
        if not data:
            continue
        try:
            persona = MegaPersona.model_validate(data)
            axes = persona.primary_axes(axis_names=axis_names)
            rows.append([float(axes[axis]) for axis in axis_names])
        except Exception:
            continue
    if len(rows) > MAX_SCATTER_POINTS:
        rows = rows[:MAX_SCATTER_POINTS]
    return rows


def _extract_behavior_axes(
    seed_entry: dict[str, Any],
    axis_names: tuple[str, ...] = AXIS_NAMES,
) -> list[list[float]]:
    """Extract behavior axes from a per-seed evolution result."""
    persona_ids = {
        p.get("persona_id", "")
        for p in seed_entry.get("personas", [])
        if p and p.get("persona_id")
    }
    # Prefer validation simulations, fall back to legacy held-out, then train.
    sims = seed_entry.get("validation_shadow_simulations") or seed_entry.get(
        "heldout_shadow_simulations"
    ) or seed_entry.get(
        "train_shadow_simulations", []
    )
    return _extract_behavior_axes_from_sims(
        seed_entry.get("personas", []), sims, axis_names=axis_names
    )


def _extract_behavior_axes_from_sims(
    personas: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
    axis_names: tuple[str, ...] = AXIS_NAMES,
) -> list[list[float]]:
    """Average per-persona axis_scores from simulation results."""
    grouped: dict[str, list[dict[str, float]]] = {}
    for p in personas:
        pid = p.get("persona_id", "") if p else ""
        if pid:
            grouped.setdefault(pid, [])
    for sim in simulations:
        pid = sim.get("persona_id", "")
        if pid in grouped:
            grouped[pid].append(sim.get("axis_scores", {}))
    rows: list[list[float]] = []
    for persona in personas:
        pid = persona.get("persona_id", "") if persona else ""
        axis_scores_list = grouped.get(pid, [])
        if not axis_scores_list:
            rows.append([0.5 for _ in axis_names])
            continue
        rows.append([
            float(np.mean([scores.get(axis, 0.5) for scores in axis_scores_list]))
            for axis in axis_names
        ])
    if len(rows) > MAX_SCATTER_POINTS:
        rows = rows[:MAX_SCATTER_POINTS]
    return rows


def _infer_axis_names(
    *,
    genome: dict[str, Any] | None = None,
    slots: list[dict[str, Any]] | None = None,
) -> tuple[str, ...]:
    if isinstance(genome, dict):
        return axis_names_for_binding(schema_binding_for_genome(genome))
    if slots:
        target_axes = slots[0].get("target_axes", {})
        if isinstance(target_axes, dict) and target_axes:
            return tuple(target_axes.keys())
    return AXIS_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_evaluation_result(
    evo_dir: Path, candidate_id: str
) -> dict[str, Any] | None:
    evals_dir = evo_dir / "evaluations"
    if not evals_dir.exists():
        return None
    for path in sorted(evals_dir.glob("eval_*_*/result.json")):
        payload = _read_json(path)
        if payload.get("candidate", {}).get("candidate_id") == candidate_id:
            return payload
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
