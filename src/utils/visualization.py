"""可视化模块 — 人格分布与进化曲线."""

import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def _ensure_matplotlib():
    """确保 matplotlib 可用."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无 GUI 后端
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        return plt, Axes3D
    except ImportError:
        raise ImportError("请先安装 matplotlib: pip install matplotlib")


def plot_persona_distribution(
    Z: np.ndarray,
    dimensions: List[str],
    title: str = "人格分布",
    save_path: str = None,
):
    """绘制人格在多样性轴上的分布散点图.

    Args:
        Z: 人格分布矩阵 (N, K)，K 为维度数
        dimensions: 维度名称列表
        title: 图表标题
        save_path: 保存路径（None 则显示）
    """
    plt, _ = _ensure_matplotlib()
    k = Z.shape[1]

    if k == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(len(Z)),
                             cmap="tab10", s=100, alpha=0.7, edgecolors="black")
        ax.set_xlabel(dimensions[0], fontsize=12)
        ax.set_ylabel(dimensions[1], fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        # 添加标签
        for i, (x, y) in enumerate(Z):
            ax.annotate(str(i+1), (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        plt.colorbar(scatter, label="Persona Index")

    elif k == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2],
                             c=np.arange(len(Z)), cmap="tab10",
                             s=100, alpha=0.7, edgecolors="black")
        ax.set_xlabel(dimensions[0], fontsize=11)
        ax.set_ylabel(dimensions[1], fontsize=11)
        ax.set_zlabel(dimensions[2], fontsize=11)
        ax.set_title(title, fontsize=14)
        plt.colorbar(scatter, label="Persona Index")

    else:
        # K > 3: 画成对散点图矩阵
        fig, axes = plt.subplots(k, k, figsize=(k*2.5, k*2.5))
        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                if i == j:
                    ax.hist(Z[:, i], bins=10, alpha=0.7, color="steelblue")
                    ax.set_title(dimensions[i], fontsize=10)
                else:
                    ax.scatter(Z[:, j], Z[:, i], c=np.arange(len(Z)),
                               cmap="tab10", s=30, alpha=0.6)
                    ax.set_xlabel(dimensions[j], fontsize=8)
                    ax.set_ylabel(dimensions[i], fontsize=8)
                ax.grid(True, alpha=0.3)
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_evolution_curve(
    history: List[Dict],
    metric_names: Optional[List[str]] = None,
    save_path: str = None,
):
    """绘制进化曲线（各指标随轮数变化）.

    Args:
        history: OpenEvolve 的 history 列表，每项含 generation 和 best_fitness
        metric_names: 要绘制的指标列表（None 则全部）
        save_path: 保存路径
    """
    plt, _ = _ensure_matplotlib()

    if not history:
        print("[Viz] 历史数据为空")
        return

    generations = [h["generation"] for h in history]

    # 确定要绘制的指标
    if metric_names is None:
        sample = history[0].get("best_fitness", {})
        metric_names = list(sample.keys())

    n_metrics = len(metric_names)
    cols = 3
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = []
        for h in history:
            bf = h.get("best_fitness", {})
            values.append(bf.get(metric, 0))

        ax.plot(generations, values, color=colors[idx], linewidth=2,
                marker="o", markersize=3)
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"{metric}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Evolution Curves", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_island_heatmap(
    islands_data: Dict[int, Dict[str, float]],
    save_path: str = None,
):
    """绘制岛屿热力图（各岛屿在各指标上的最佳适应度）.

    Args:
        islands_data: {island_id: {metric: value}}
        save_path: 保存路径
    """
    plt, _ = _ensure_matplotlib()

    island_ids = sorted(islands_data.keys())
    if not island_ids:
        print("[Viz] 岛屿数据为空")
        return

    metrics = list(next(iter(islands_data.values())).keys())
    data = np.array([[islands_data[i].get(m, 0) for m in metrics]
                     for i in island_ids])

    fig, ax = plt.subplots(figsize=(max(8, len(metrics)*1.5), max(6, len(island_ids)*0.8)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=data.min(), vmax=data.max())

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(island_ids)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"Island {i}" for i in island_ids], fontsize=9)

    # 在每个格子中标注数值
    for i in range(len(island_ids)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{data[i, j]:.3f}",
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Island Fitness Heatmap", fontsize=14)
    plt.colorbar(im, ax=ax, label="Fitness")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_coverage_comparison(
    Z_list: List[np.ndarray],
    labels: List[str],
    dimensions: List[str],
    save_path: str = None,
):
    """对比多组人格分布（如不同 seed 或不同代的结果）.

    Args:
        Z_list: 多组 Z 矩阵
        labels: 每组标签
        dimensions: 维度名称
        save_path: 保存路径
    """
    plt, _ = _ensure_matplotlib()
    k = Z_list[0].shape[1]
    n_groups = len(Z_list)

    if k == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
        for i, (Z, label) in enumerate(zip(Z_list, labels)):
            ax.scatter(Z[:, 0], Z[:, 1], c=[colors[i]], label=label,
                       s=80, alpha=0.6, edgecolors="black")
        ax.set_xlabel(dimensions[0], fontsize=12)
        ax.set_ylabel(dimensions[1], fontsize=12)
        ax.set_title("Persona Distribution Comparison", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    else:
        # 画成对对比
        cols = min(n_groups, 3)
        rows = (n_groups + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if rows == 1:
            axes = [axes] if cols == 1 else axes.flatten().tolist()
        else:
            axes = axes.flatten().tolist()

        for i, (Z, label) in enumerate(zip(Z_list, labels)):
            ax = axes[i]
            if k >= 2:
                ax.scatter(Z[:, 0], Z[:, 1], s=80, alpha=0.6,
                           edgecolors="black")
                ax.set_xlabel(dimensions[0], fontsize=10)
                ax.set_ylabel(dimensions[1], fontsize=10)
            ax.set_title(label, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        for idx in range(n_groups, len(axes)):
            axes[idx].axis("off")

        fig.suptitle("Persona Distribution Comparison", fontsize=14)
        plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def generate_all_visualizations(
    Z: np.ndarray,
    dimensions: List[str],
    history: List[Dict],
    islands_data: Dict[int, Dict[str, float]],
    output_dir: str = "data/results/visualizations",
):
    """一键生成所有可视化图表.

    Args:
        Z: 人格分布矩阵
        dimensions: 维度名称
        history: 进化历史
        islands_data: 岛屿数据
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Viz] 生成可视化图表...")

    plot_persona_distribution(
        Z, dimensions,
        title="Persona Distribution (Best Generation)",
        save_path=output_dir / "persona_distribution.png",
    )

    plot_evolution_curve(
        history,
        save_path=output_dir / "evolution_curves.png",
    )

    plot_island_heatmap(
        islands_data,
        save_path=output_dir / "island_heatmap.png",
    )

    print(f"[Viz] 所有图表已保存到: {output_dir}")
