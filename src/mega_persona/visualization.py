"""Visualization helpers for MegaPersona experiment and evolution outputs."""

from pathlib import Path
from typing import Any
import json
import os
import tempfile

import numpy as np

from src.mega_persona.schema import MegaPersona
from src.mega_persona.slots import AXIS_NAMES


def visualize_result_path(input_path: Path, output_dir: Path | None = None) -> list[Path]:
    """Create figures for a MegaPersona result directory or JSON file."""
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path / "figures" if input_path.is_dir() else input_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir() and (input_path / "final_summary.json").exists():
        return visualize_evolution_dir(input_path, output_dir)
    if input_path.is_dir() and (input_path / "summary.json").exists():
        return visualize_experiment_summary(input_path / "summary.json", output_dir)
    if input_path.is_file() and input_path.name == "final_summary.json":
        return visualize_evolution_dir(input_path.parent, output_dir)
    if input_path.is_file() and input_path.name == "summary.json":
        return visualize_experiment_summary(input_path, output_dir)
    if input_path.is_file():
        return visualize_generation_file(input_path, output_dir)
    raise FileNotFoundError(f"Unsupported MegaPersona result path: {input_path}")


def visualize_evolution_dir(evolution_dir: Path, output_dir: Path) -> list[Path]:
    written: list[Path] = []
    final_summary = _read_json(evolution_dir / "final_summary.json")
    best_id = final_summary["best"]["candidate_id"]
    best_result = _find_evaluation_result(evolution_dir, best_id)
    if best_result is None:
        raise FileNotFoundError(f"Could not find evaluation result for {best_id}")

    generation_files = sorted((evolution_dir / "generations").glob("generation_*.json"))
    generations = [_read_json(path) for path in generation_files]
    if generations:
        written.append(plot_fitness_curve(generations, output_dir / "fitness_over_generations.png"))

    written.extend(_plot_seed_result(best_result["per_seed"][0], output_dir, prefix="best"))
    written.append(plot_best_genome(final_summary["best"]["genome"], output_dir / "best_genome.png"))
    written.append(plot_metric_bars(final_summary["best"].get("metrics", {}), output_dir / "best_metrics.png"))
    return written


def visualize_experiment_summary(summary_path: Path, output_dir: Path) -> list[Path]:
    summary = _read_json(summary_path)
    runs = summary.get("runs", [])
    if not runs:
        return []
    written: list[Path] = []
    first = runs[0]
    written.extend(_plot_run_payload(first, output_dir, prefix=f"seed_{first.get('seed', 'run')}"))
    aggregate = summary.get("aggregate", {})
    if aggregate:
        written.append(plot_metric_bars(aggregate, output_dir / "aggregate_metrics.png"))
    return written


def visualize_generation_file(json_path: Path, output_dir: Path) -> list[Path]:
    payload = _read_json(json_path)
    written: list[Path] = []
    written.extend(_plot_run_payload(payload, output_dir, prefix="generation"))
    if payload.get("evaluation"):
        written.append(plot_metric_bars(payload["evaluation"], output_dir / "generation_metrics.png"))
    return written


def plot_fitness_curve(generations: list[dict[str, Any]], save_path: Path) -> Path:
    plt = _plt()
    xs = [item["generation"] for item in generations]
    ys = [item.get("best_fitness") or 0.0 for item in generations]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", color="#2563eb", linewidth=2)
    ax.set_title("Best Fitness over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.grid(True, alpha=0.25)
    _save(fig, save_path)
    return save_path


def plot_axis_scatter(
    points: np.ndarray,
    title: str,
    save_path: Path,
    labels: tuple[str, ...] = AXIS_NAMES,
) -> Path:
    plt = _plt()
    fig = plt.figure(figsize=(8, 6))
    if points.size == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No points", ha="center", va="center")
        ax.set_axis_off()
    elif points.shape[1] >= 3:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=np.arange(len(points)),
            cmap="viridis",
            s=70,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        fig.colorbar(scatter, ax=ax, shrink=0.65, label="Index")
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=np.arange(len(points)),
            cmap="viridis",
            s=70,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="Index")
    fig.suptitle(title)
    _save(fig, save_path)
    return save_path


def plot_best_genome(genome: dict[str, Any], save_path: Path) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    quota = genome.get("quota_weights", {})
    axes[0].bar(range(len(quota)), list(quota.values()), color="#0f766e")
    axes[0].set_title("Quota Weights")
    axes[0].set_xticks(range(len(quota)))
    axes[0].set_xticklabels(list(quota.keys()), rotation=45, ha="right", fontsize=8)

    bias = genome.get("axis_bias", {})
    axes[1].bar(range(len(bias)), list(bias.values()), color="#b45309")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Axis Bias")
    axes[1].set_xticks(range(len(bias)))
    axes[1].set_xticklabels(list(bias.keys()), rotation=35, ha="right", fontsize=8)

    stretch = genome.get("axis_stretch", {})
    axes[2].bar(range(len(stretch)), list(stretch.values()), color="#6d28d9")
    axes[2].axhline(1, color="black", linewidth=0.8)
    axes[2].set_title("Axis Stretch")
    axes[2].set_xticks(range(len(stretch)))
    axes[2].set_xticklabels(list(stretch.keys()), rotation=35, ha="right", fontsize=8)

    fig.suptitle("Best Candidate Genome")
    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def plot_metric_bars(metrics: dict[str, Any], save_path: Path) -> Path:
    plt = _plt()
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not key.endswith(".std")
    }
    if not numeric:
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

    fig, ax = plt.subplots(figsize=(max(8, len(numeric) * 0.75), 4.8))
    ax.bar(range(len(numeric)), list(numeric.values()), color="#334155")
    ax.set_xticks(range(len(numeric)))
    ax.set_xticklabels(list(numeric.keys()), rotation=45, ha="right", fontsize=8)
    ax.set_title("Metrics")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def _plot_seed_result(seed_result: dict[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    slots = seed_result.get("slots", [])
    personas = seed_result.get("personas", [])
    heldout = seed_result.get("heldout_shadow_simulations") or seed_result.get("shadow_simulations") or []

    slot_axes = np.array(
        [[slot["target_axes"][axis] for axis in AXIS_NAMES] for slot in slots],
        dtype=float,
    ) if slots else np.empty((0, len(AXIS_NAMES)))
    persona_axes = _persona_axis_matrix(personas)
    behavior_axes = _behavior_axis_matrix(personas, heldout)

    return [
        plot_axis_scatter(slot_axes, "Slot Target Axes", output_dir / f"{prefix}_slot_axes.png"),
        plot_axis_scatter(persona_axes, "Persona Primary Axes", output_dir / f"{prefix}_persona_axes.png"),
        plot_axis_scatter(behavior_axes, "Held-out Behavior Axes", output_dir / f"{prefix}_behavior_axes.png"),
    ]


def _plot_run_payload(payload: dict[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    slots = payload.get("slots", [])
    personas = [persona for persona in payload.get("personas", []) if persona]
    simulations = payload.get("shadow_simulations", [])
    slot_axes = np.array(
        [[slot["target_axes"][axis] for axis in AXIS_NAMES] for slot in slots],
        dtype=float,
    ) if slots else np.empty((0, len(AXIS_NAMES)))
    persona_axes = _persona_axis_matrix(personas)
    behavior_axes = _behavior_axis_matrix(personas, simulations)
    return [
        plot_axis_scatter(slot_axes, "Slot Target Axes", output_dir / f"{prefix}_slot_axes.png"),
        plot_axis_scatter(persona_axes, "Persona Primary Axes", output_dir / f"{prefix}_persona_axes.png"),
        plot_axis_scatter(behavior_axes, "Behavior Axes", output_dir / f"{prefix}_behavior_axes.png"),
    ]


def _persona_axis_matrix(personas: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for data in personas:
        axes = MegaPersona.model_validate(data).primary_axes()
        rows.append([axes[axis] for axis in AXIS_NAMES])
    if not rows:
        return np.empty((0, len(AXIS_NAMES)))
    return np.array(rows, dtype=float)


def _behavior_axis_matrix(
    personas: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
) -> np.ndarray:
    persona_ids = [persona["persona_id"] for persona in personas]
    grouped: dict[str, list[dict[str, float]]] = {persona_id: [] for persona_id in persona_ids}
    for simulation in simulations:
        persona_id = simulation.get("persona_id")
        if persona_id in grouped:
            grouped[persona_id].append(simulation.get("axis_scores", {}))

    rows = []
    for persona_id in persona_ids:
        axis_scores = grouped.get(persona_id) or []
        if not axis_scores:
            rows.append([0.5 for _ in AXIS_NAMES])
        else:
            rows.append(
                [
                    float(np.mean([scores.get(axis, 0.5) for scores in axis_scores]))
                    for axis in AXIS_NAMES
                ]
            )
    if not rows:
        return np.empty((0, len(AXIS_NAMES)))
    return np.array(rows, dtype=float)


def _find_evaluation_result(evolution_dir: Path, candidate_id: str) -> dict[str, Any] | None:
    for result_path in sorted((evolution_dir / "evaluations").glob("eval_*_*/result.json")):
        payload = _read_json(result_path)
        if payload.get("candidate", {}).get("candidate_id") == candidate_id:
            return payload
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _plt():
    try:
        if not os.environ.get("MPLCONFIGDIR"):
            cache_dir = Path(tempfile.gettempdir()) / "mega_persona_matplotlib"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("Please install matplotlib to use MegaPersona visualization.") from exc


def _save(fig, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    _plt().close(fig)
