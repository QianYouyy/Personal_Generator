#!/usr/bin/env python3
"""Post-hoc behavior-grid (MAP-Elites style) occupancy analysis for a finished run.

Maps every persisted persona of every evaluated candidate into a behavior
descriptor grid and reports:

- target occupancy: which grid cells the slot sampler asked for;
- realized occupancy: which cells the generator actually produced
  (per-persona mean shadow-survey axis scores);
- per-generation cumulative realized occupancy (does evolution expand the
  behavior footprint or stall?);
- per-cell best quality (1 - |realized - target| MAE), i.e. what a
  MAP-Elites archive would have kept;
- pool union vs. the single global-best candidate's footprint.

Zero API cost: only reads mega_eval/evaluations/*/result.json.

Usage:
    python scripts/report_behavior_grid_occupancy.py --source-run data/results/<run>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _bucket(value: float, buckets: int) -> int:
    idx = int(value * buckets)
    return min(max(idx, 0), buckets - 1)


def _cell(vector: dict[str, float], axes: list[str], buckets: int) -> tuple[int, ...]:
    return tuple(_bucket(float(vector.get(a, 0.5)), buckets) for a in axes)


def _load_results(mega_eval_dir: Path) -> list[dict]:
    results = []
    eval_dir = mega_eval_dir / "evaluations"
    for child in sorted(eval_dir.iterdir()):
        if child.name.startswith("alias_") or not child.is_dir():
            continue
        result_path = child / "result.json"
        if not result_path.exists():
            continue
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
    return results


def _persona_records(result: dict, split: str) -> list[dict]:
    """Flatten one evaluation into per-persona realized/target records."""
    records = []
    sim_key = {
        "validation": "validation_shadow_simulations",
        "train": "train_shadow_simulations",
    }.get(split)
    for per_seed in result.get("per_seed", []):
        if per_seed.get("status") != "ok":
            continue
        slots = {s["slot_id"]: s for s in per_seed.get("slots", []) if "slot_id" in s}
        sim_lists: list[list[dict]] = []
        if sim_key:
            sim_lists = [per_seed.get(sim_key, [])]
        else:  # all
            sim_lists = [
                per_seed.get("train_shadow_simulations", []),
                per_seed.get("validation_shadow_simulations", []),
            ]
        by_persona: dict[str, list[dict]] = defaultdict(list)
        for sims in sim_lists:
            for sim in sims or []:
                pid = sim.get("persona_id")
                if pid and sim.get("axis_scores"):
                    by_persona[pid].append(sim["axis_scores"])
        for pid, score_dicts in by_persona.items():
            axes = sorted(score_dicts[0].keys())
            realized = {
                a: sum(float(s.get(a, 0.5)) for s in score_dicts) / len(score_dicts)
                for a in axes
            }
            slot = slots.get(pid)
            target = slot.get("target_axes") if slot else None
            quality = None
            if target:
                mae = sum(abs(realized[a] - float(target.get(a, 0.5))) for a in axes) / len(axes)
                quality = 1.0 - mae
            records.append(
                {
                    "persona_id": pid,
                    "realized": realized,
                    "target": target,
                    "quality": quality,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="Finished run directory.")
    parser.add_argument("--buckets", type=int, default=3, help="Grid buckets per axis.")
    parser.add_argument(
        "--split",
        choices=["validation", "train", "all"],
        default="validation",
        help="Which shadow simulations define realized behavior.",
    )
    args = parser.parse_args()

    run_dir = Path(args.source_run)
    mega_eval_dir = run_dir / "mega_eval"
    if not mega_eval_dir.exists():
        raise SystemExit(f"mega_eval not found under {run_dir}")

    results = _load_results(mega_eval_dir)
    if not results:
        raise SystemExit(f"no evaluation results under {mega_eval_dir}/evaluations")

    buckets = args.buckets
    axes: list[str] = []
    pool: list[dict] = []
    for result in results:
        candidate = result.get("candidate", {})
        generation = candidate.get("generation", 0)
        candidate_id = candidate.get("candidate_id", "?")
        fitness = result.get("fitness")
        for rec in _persona_records(result, args.split):
            if not axes:
                axes = sorted(rec["realized"].keys())
            rec.update(
                {
                    "candidate_id": candidate_id,
                    "generation": generation,
                    "candidate_fitness": fitness,
                    "realized_cell": _cell(rec["realized"], axes, buckets),
                    "target_cell": _cell(rec["target"], axes, buckets) if rec["target"] else None,
                }
            )
            pool.append(rec)

    if not pool:
        raise SystemExit("no persona shadow simulations found")

    n_cells = buckets ** len(axes)
    target_cells = {r["target_cell"] for r in pool if r["target_cell"]}
    realized_cells = {r["realized_cell"] for r in pool}

    # Per-generation cumulative realized occupancy.
    gens = sorted({r["generation"] for r in pool})
    cumulative: list[tuple[int, int]] = []
    seen: set[tuple[int, ...]] = set()
    for g in gens:
        seen |= {r["realized_cell"] for r in pool if r["generation"] == g}
        cumulative.append((g, len(seen)))

    # Per-candidate footprint.
    by_candidate: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for r in pool:
        by_candidate[r["candidate_id"]].add(r["realized_cell"])
    footprints = sorted(len(c) for c in by_candidate.values())

    # Cell best quality (MAP-Elites archive view).
    cell_best: dict[tuple[int, ...], dict] = {}
    for r in pool:
        if r["quality"] is None:
            continue
        cur = cell_best.get(r["realized_cell"])
        if cur is None or r["quality"] > cur["quality"]:
            cell_best[r["realized_cell"]] = {
                "quality": r["quality"],
                "candidate_id": r["candidate_id"],
                "generation": r["generation"],
            }

    # Global-best candidate footprint vs pool.
    best_result = max(results, key=lambda r: r.get("fitness") or 0.0)
    best_id = best_result.get("candidate", {}).get("candidate_id", "?")
    best_cells = by_candidate.get(best_id, set())

    # Target vs realized spread per axis (central-tendency check).
    spread = {}
    for i, a in enumerate(axes):
        t_vals = [float(r["target"][a]) for r in pool if r["target"]]
        r_vals = [r["realized"][a] for r in pool]
        t_var = _variance(t_vals)
        r_var = _variance(r_vals)
        spread[a] = {
            "target_std": math.sqrt(t_var),
            "realized_std": math.sqrt(r_var),
            "ratio": (math.sqrt(r_var) / math.sqrt(t_var)) if t_var > 0 else None,
        }

    # Contributors: how many distinct candidates supply the archive.
    archive_candidates = {v["candidate_id"] for v in cell_best.values()}
    archive_gens = sorted({v["generation"] for v in cell_best.values()})

    summary = {
        "run_dir": str(run_dir),
        "split": args.split,
        "axes": axes,
        "buckets_per_axis": buckets,
        "grid_cells_total": n_cells,
        "evaluations": len(results),
        "personas_scored": len(pool),
        "target_cells_used": len(target_cells),
        "realized_cells_used": len(realized_cells),
        "realized_occupancy_pct": round(100.0 * len(realized_cells) / n_cells, 1),
        "per_generation_cumulative_realized": cumulative,
        "candidate_footprint": {
            "min": footprints[0],
            "median": footprints[len(footprints) // 2],
            "max": footprints[-1],
        },
        "global_best_candidate": best_id,
        "global_best_footprint_cells": len(best_cells),
        "global_best_footprint_pct_of_pool": round(
            100.0 * len(best_cells) / max(1, len(realized_cells)), 1
        ),
        "archive_cells_with_quality": len(cell_best),
        "archive_distinct_contributor_candidates": len(archive_candidates),
        "archive_contributor_generations": archive_gens,
        "archive_mean_best_quality": round(
            sum(v["quality"] for v in cell_best.values()) / max(1, len(cell_best)), 4
        ),
        "axis_spread": spread,
    }

    out_dir = run_dir / "behavior_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "behavior_grid_occupancy.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "behavior_grid_occupancy.md").write_text(
        _render_markdown(summary, cell_best, axes, buckets, realized_cells, target_cells),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwritten: {out_dir}/behavior_grid_occupancy.json/.md")


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _render_markdown(
    summary: dict,
    cell_best: dict,
    axes: list[str],
    buckets: int,
    realized_cells: set,
    target_cells: set,
) -> str:
    lines = [
        "# Behavior Grid Occupancy (MAP-Elites post-hoc)",
        "",
        f"- run: `{summary['run_dir']}`",
        f"- split: `{summary['split']}`",
        f"- grid: {len(axes)} axes x {buckets} buckets = {summary['grid_cells_total']} cells",
        f"- evaluations: {summary['evaluations']}, personas scored: {summary['personas_scored']}",
        "",
        "## Headline",
        "",
        f"- target cells used by slot sampler: **{summary['target_cells_used']}**",
        f"- realized cells produced by generator: **{summary['realized_cells_used']}** "
        f"({summary['realized_occupancy_pct']}% of grid)",
        f"- global best `{summary['global_best_candidate']}` footprint: "
        f"**{summary['global_best_footprint_cells']}** cells "
        f"({summary['global_best_footprint_pct_of_pool']}% of pool union)",
        f"- archive cells with quality: {summary['archive_cells_with_quality']}, "
        f"contributed by {summary['archive_distinct_contributor_candidates']} distinct candidates "
        f"across generations {summary['archive_contributor_generations']}",
        f"- archive mean best quality (1 - MAE): {summary['archive_mean_best_quality']}",
        "",
        "## Per-generation cumulative realized occupancy",
        "",
        "| Generation | Cumulative cells |",
        "|---:|---:|",
    ]
    for gen, count in summary["per_generation_cumulative_realized"]:
        lines.append(f"| {gen} | {count} |")
    lines += [
        "",
        "## Axis spread (target vs realized std)",
        "",
        "| Axis | target std | realized std | ratio |",
        "|---|---:|---:|---:|",
    ]
    for a, s in summary["axis_spread"].items():
        ratio = f"{s['ratio']:.2f}" if s["ratio"] is not None else "n/a"
        lines.append(f"| {a} | {s['target_std']:.3f} | {s['realized_std']:.3f} | {ratio} |")

    # 2D projections: fix each axis in turn, show the other two.
    labels = ["L", "M", "H"] if buckets == 3 else [str(i) for i in range(buckets)]
    for fixed in range(len(axes)):
        other = [i for i in range(len(axes)) if i != fixed]
        lines += [
            "",
            f"## Projection: {axes[other[0]]} x {axes[other[1]]} (any {axes[fixed]})",
            "",
            "| bucket | " + " | ".join(labels) + " |",
            "|---|" + "---:|" * buckets,
        ]
        for b0 in reversed(range(buckets)):
            row = []
            for b1 in range(buckets):
                count = 0
                for cell in realized_cells:
                    if cell[other[0]] == b0 and cell[other[1]] == b1:
                        count += 1
                row.append(str(count) if count else ".")
            lines.append(f"| {labels[b0]} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Archive (best quality per realized cell)",
        "",
        "| Cell (" + ", ".join(axes) + ") | quality | candidate | gen |",
        "|---|---:|---|---:|",
    ]
    for cell in sorted(cell_best, key=lambda c: -cell_best[c]["quality"]):
        v = cell_best[cell]
        cell_str = ",".join(labels[b] for b in cell)
        lines.append(
            f"| {cell_str} | {v['quality']:.3f} | `{v['candidate_id']}` | {v['generation']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
