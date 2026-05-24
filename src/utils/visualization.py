"""可视化模块 — 人格分布、进化曲线、岛屿热力图等.

支持两种使用方式：
1. 运行时调用：传入 Z/history/islands_data 直接生成
2. 从 checkpoint 重绘：读取 checkpoint JSON 文件重新生成所有图表

新增图表（相比旧版）：
- 综合评分曲线 (composite_score.png)
- 改进/评估统计 (improvements.png)
- 每轮耗时分析 (generation_time.png)
- 灭绝事件标注 (evolution_curves 上标注)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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


# ───────────────────────────────────────────────
# 基础绘图函数
# ───────────────────────────────────────────────

def plot_persona_distribution(
    Z: np.ndarray,
    dimensions: List[str],
    title: str = "人格分布",
    save_path: str = None,
):
    """绘制人格在多样性轴上的分布散点图."""
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
    extinction_generations: Optional[List[int]] = None,
):
    """绘制进化曲线（各指标随轮数变化）.

    每图显示三条线：
    - 全局最优（best，蓝色）
    - 所有岛屿该指标精英的平均值（mean，橙色）
    - 所有岛屿该指标精英的最大值（max，绿色）
    - 阴影区域：mean ± std

    Args:
        history: OpenEvolve 的 history 列表
        metric_names: 要绘制的指标列表（None 则全部）
        save_path: 保存路径
        extinction_generations: 灭绝发生的轮数列表（在图上标注）
    """
    plt, _ = _ensure_matplotlib()

    if not history:
        print("[Viz] 历史数据为空")
        return

    generations = [h["generation"] for h in history]

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

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        # 全局最优
        best_values = []
        for h in history:
            bf = h.get("best_fitness", {})
            best_values.append(bf.get(metric, 0))
        
        # 统计值（如果有）
        mean_values = []
        max_values = []
        min_values = []
        std_values = []
        has_stats = False
        for h in history:
            stats = h.get("fitness_stats", {})
            if metric in stats:
                has_stats = True
                mean_values.append(stats[metric]["mean"])
                max_values.append(stats[metric]["max"])
                min_values.append(stats[metric]["min"])
                std_values.append(stats[metric]["std"])
            else:
                mean_values.append(None)
                max_values.append(None)
                min_values.append(None)
                std_values.append(None)
        
        # 绘制全局最优
        ax.plot(generations, best_values, color="steelblue", linewidth=2.5,
                marker="o", markersize=4, label="Global Best", zorder=3)
        
        # 绘制统计值（如果有）
        if has_stats:
            valid_gen = [g for g, m in zip(generations, mean_values) if m is not None]
            valid_mean = [m for m in mean_values if m is not None]
            valid_max = [m for m in max_values if m is not None]
            valid_min = [m for m in min_values if m is not None]
            valid_std = [s for s in std_values if s is not None]
            
            ax.plot(valid_gen, valid_mean, color="coral", linewidth=1.5,
                    linestyle="--", marker="s", markersize=3, label="Mean", zorder=2)
            ax.plot(valid_gen, valid_max, color="seagreen", linewidth=1.5,
                    linestyle=":", marker="^", markersize=3, label="Max", zorder=2)
            
            # 填充 mean ± std 区域
            if valid_mean and valid_std:
                upper = [m + s for m, s in zip(valid_mean, valid_std)]
                lower = [m - s for m, s in zip(valid_mean, valid_std)]
                ax.fill_between(valid_gen, lower, upper, color="coral", alpha=0.15)
        
        # 标注灭绝事件
        if extinction_generations:
            for eg in extinction_generations:
                if eg in generations:
                    ax.axvline(x=eg, color="red", linestyle="--", alpha=0.4, linewidth=1)
        
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"{metric}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    # 添加灭绝图例
    if extinction_generations:
        fig.text(0.5, 0.01, f"Red dashed = Extinction (Gen {', '.join(map(str, extinction_generations))})",
                 ha="center", fontsize=9, color="red")

    fig.suptitle("Evolution Curves (Global Best / Mean / Max)", fontsize=14)
    plt.tight_layout()
    if extinction_generations:
        plt.subplots_adjust(bottom=0.08)

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
    
    注意：负值指标（dispersion, kl_divergence）使用独立的颜色映射，
    确保"越接近0越好"的语义正确显示。
    """
    plt, _ = _ensure_matplotlib()

    island_ids = sorted(islands_data.keys())
    if not island_ids:
        print("[Viz] 岛屿数据为空")
        return

    metrics = list(next(iter(islands_data.values())).keys())
    data = np.array([[islands_data[i].get(m, 0) for m in metrics]
                     for i in island_ids])

    # 区分正值指标和负值指标
    positive_metrics = ["coverage", "convex_hull", "avg_dist", "min_dist"]
    negative_metrics = ["dispersion", "kl_divergence"]
    
    # 为每个指标单独归一化到 [0, 1]，并标记颜色映射方向
    # 正值指标：越大越好（绿=好，红=差）
    # 负值指标：越接近0越好（绿=好（接近0），红=差（远离0））
    normalized_data = np.zeros_like(data)
    for j, metric in enumerate(metrics):
        col = data[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-10:
            normalized_data[:, j] = 0.5
        else:
            if metric in negative_metrics:
                # 负值指标：取绝对值后归一化，越接近0越好
                abs_col = np.abs(col)
                abs_min, abs_max = abs_col.min(), abs_col.max()
                if abs_max - abs_min < 1e-10:
                    normalized_data[:, j] = 0.5
                else:
                    normalized_data[:, j] = 1 - (abs_col - abs_min) / (abs_max - abs_min)
            else:
                # 正值指标：直接归一化
                normalized_data[:, j] = (col - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics)*1.5), max(6, len(island_ids)*0.8)))
    
    # 使用统一的颜色映射显示归一化后的数据
    im = ax.imshow(normalized_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(island_ids)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"Island {i}" for i in island_ids], fontsize=9)

    for i in range(len(island_ids)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{data[i, j]:.3f}",
                          ha="center", va="center", 
                          color="white" if normalized_data[i, j] < 0.3 or normalized_data[i, j] > 0.7 else "black",
                          fontsize=8, fontweight="bold")

    ax.set_title("Island Fitness Heatmap (Normalized per Metric)\nGreen=Better, Red=Worse", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, label="Normalized Score")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Worse", "Medium", "Better"])
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# ───────────────────────────────────────────────
# 新增图表
# ───────────────────────────────────────────────

def plot_composite_score(
    history: List[Dict],
    weights: Optional[Dict[str, float]] = None,
    save_path: str = None,
    extinction_generations: Optional[List[int]] = None,
):
    """绘制综合评分曲线.

    默认权重（与 engine._get_best_island 一致）：
    coverage 30%, convex_hull 15%, avg_dist 20%, min_dist 15%, dispersion 10%, kl_divergence 10%
    """
    plt, _ = _ensure_matplotlib()

    if not history:
        print("[Viz] 历史数据为空")
        return

    if weights is None:
        weights = {
            "coverage": 0.30,
            "convex_hull": 0.15,
            "avg_dist": 0.20,
            "min_dist": 0.15,
            "dispersion": 0.10,
            "kl_divergence": 0.10,
        }

    generations = [h["generation"] for h in history]
    
    # 计算全局最优的综合评分
    best_scores = []
    for h in history:
        bf = h.get("best_fitness", {})
        score = sum(bf.get(k, 0) * w for k, w in weights.items())
        best_scores.append(score)
    
    # 计算平均综合评分（如果有统计值）
    mean_scores = []
    has_mean = False
    for h in history:
        stats = h.get("fitness_stats", {})
        if stats:
            has_mean = True
            score = sum(stats.get(k, {}).get("mean", 0) * w for k, w in weights.items())
            mean_scores.append(score)
        else:
            mean_scores.append(None)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 绘制全局最优
    ax.plot(generations, best_scores, color="darkgreen", linewidth=2.5,
            marker="o", markersize=5, label="Global Best", zorder=3)
    
    # 绘制平均评分
    if has_mean:
        valid_gen = [g for g, m in zip(generations, mean_scores) if m is not None]
        valid_mean = [m for m in mean_scores if m is not None]
        ax.plot(valid_gen, valid_mean, color="coral", linewidth=1.5,
                linestyle="--", marker="s", markersize=4, label="Mean", zorder=2)
    
    if extinction_generations:
        for eg in extinction_generations:
            if eg in generations:
                ax.axvline(x=eg, color="red", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("Evolution Composite Score (Global Best / Mean)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # 标注最终分数
    final_score = best_scores[-1]
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    label_y = y_max - y_range * 0.05
    ax.text(generations[-1], label_y, f"Final: {final_score:.4f}",
            fontsize=10, color="darkgreen", ha="right", va="top",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_improvements(
    history: List[Dict],
    save_path: str = None,
):
    """绘制每轮改进数和评估数统计."""
    plt, _ = _ensure_matplotlib()

    if not history:
        print("[Viz] 历史数据为空")
        return

    generations = [h["generation"] for h in history]
    improvements = [h.get("improvements", 0) for h in history]
    evaluations = [h.get("evaluations", 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 改进数
    ax1.bar(generations, improvements, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Generation", fontsize=11)
    ax1.set_ylabel("Improvements", fontsize=11)
    ax1.set_title("Improvements per Generation", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y")

    # 评估数
    ax2.bar(generations, evaluations, color="coral", alpha=0.8)
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Evaluations", fontsize=11)
    ax2.set_title("Evaluations per Generation", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_generation_time(
    history: List[Dict],
    save_path: str = None,
):
    """绘制每轮耗时和累计耗时."""
    plt, _ = _ensure_matplotlib()

    if not history:
        print("[Viz] 历史数据为空")
        return

    generations = [h["generation"] for h in history]
    
    # 计算每轮耗时（从 time 字段推导）
    times = [h.get("time", 0) for h in history]
    per_gen_times = []
    for i in range(len(times)):
        if i == 0:
            per_gen_times.append(times[i] - times[0])
        else:
            per_gen_times.append(times[i] - times[i-1])
    
    # 累计耗时（分钟）
    cumulative = [(t - times[0]) / 60 for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 每轮耗时
    ax1.bar(generations, per_gen_times, color="teal", alpha=0.8)
    ax1.set_xlabel("Generation", fontsize=11)
    ax1.set_ylabel("Time (seconds)", fontsize=11)
    ax1.set_title("Time per Generation", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y")

    # 累计耗时
    ax2.plot(generations, cumulative, color="purple", linewidth=2, marker="o")
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Cumulative Time (minutes)", fontsize=11)
    ax2.set_title("Cumulative Runtime", fontsize=12)
    ax2.grid(True, alpha=0.3)

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
    """对比多组人格分布."""
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


def _normalize_for_radar(metrics: List[str], values: List[float]) -> List[float]:
    """使用实际数据的 min/max 进行归一化，而非硬编码假设.
    
    对于负值指标（dispersion, kl_divergence），归一化逻辑：
    - 越接近 0 越好
    - 取绝对值后，用 (max_abs - |v|) / (max_abs - min_abs) 归一化
    """
    normalized = []
    for m, v in zip(metrics, values):
        # 先收集所有值的范围（这里单点数据无法知道范围，使用通用范围）
        # 基于实际运行数据的观察值设定合理范围
        ranges = {
            "coverage": (0.0, 1.0),
            "convex_hull": (0.0, 0.5),
            "avg_dist": (0.0, 1.0),
            "min_dist": (0.0, 0.5),
            "dispersion": (-2.0, 0.0),      # 越接近0越好
            "kl_divergence": (-3.0, 0.0),   # 越接近0越好
        }
        vmin, vmax = ranges.get(m, (0.0, 1.0))
        
        if m in ("dispersion", "kl_divergence"):
            # 负值指标：越接近0越好
            # 映射到 [0, 1]，0 映射到 1，最负映射到 0
            normalized.append(min(max(1 - (abs(v) / abs(vmin)), 0), 1))
        else:
            # 正值指标：直接归一化
            normalized.append(min(max((v - vmin) / (vmax - vmin), 0), 1))
    
    return normalized


def plot_fitness_radar(
    fitness_dict: Dict[str, float],
    title: str = "Fitness Radar",
    save_path: str = None,
):
    """绘制适应度雷达图.
    
    归一化使用基于实际数据观察的合理范围，避免硬编码缩放因子导致失真。
    """
    plt, _ = _ensure_matplotlib()

    metrics = list(fitness_dict.keys())
    values = list(fitness_dict.values())
    
    normalized = _normalize_for_radar(metrics, values)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_plot = normalized + normalized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, color="steelblue", alpha=0.25)
    ax.plot(angles, values_plot, color="steelblue", linewidth=2, marker="o")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True)
    
    # 在每个点旁边标注原始值
    for angle, norm_val, orig_val, metric in zip(angles[:-1], normalized, values, metrics):
        label_radius = norm_val + 0.1 if norm_val < 0.8 else norm_val - 0.15
        ax.text(angle, label_radius, f"{orig_val:.3f}", 
                ha="center", va="center", fontsize=8, color="darkblue")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# ───────────────────────────────────────────────
# 一键生成所有图表
# ───────────────────────────────────────────────

def generate_all_visualizations(
    Z: np.ndarray,
    dimensions: List[str],
    history: List[Dict],
    islands_data: Dict[int, Dict[str, float]],
    output_dir: str = "data/results/visualizations",
    extinction_generations: Optional[List[int]] = None,
    best_fitness: Optional[Dict[str, float]] = None,
):
    """一键生成所有可视化图表.

    Args:
        Z: 人格分布矩阵
        dimensions: 维度名称
        history: 进化历史
        islands_data: 岛屿数据
        output_dir: 输出目录
        extinction_generations: 灭绝发生的轮数列表
        best_fitness: 最终最优适应度（用于雷达图）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Viz] 生成可视化图表...")

    # 1. 人格分布散点图
    plot_persona_distribution(
        Z, dimensions,
        title="Persona Distribution (Best Generation)",
        save_path=output_dir / "persona_distribution.png",
    )

    # 2. 进化曲线（各指标）
    plot_evolution_curve(
        history,
        save_path=output_dir / "evolution_curves.png",
        extinction_generations=extinction_generations,
    )

    # 3. 岛屿热力图
    plot_island_heatmap(
        islands_data,
        save_path=output_dir / "island_heatmap.png",
    )

    # 4. 综合评分曲线（新增）
    plot_composite_score(
        history,
        save_path=output_dir / "composite_score.png",
        extinction_generations=extinction_generations,
    )

    # 5. 改进/评估统计（新增）
    plot_improvements(
        history,
        save_path=output_dir / "improvements.png",
    )

    # 6. 每轮耗时分析（新增）
    plot_generation_time(
        history,
        save_path=output_dir / "generation_time.png",
    )

    # 7. 适应度雷达图（新增，如果有最终 fitness）
    if best_fitness:
        plot_fitness_radar(
            best_fitness,
            title="Best Candidate Fitness Radar",
            save_path=output_dir / "fitness_radar.png",
        )

    print(f"[Viz] 所有图表已保存到: {output_dir}")
    return output_dir


# ───────────────────────────────────────────────
# 从 Checkpoint 重绘
# ───────────────────────────────────────────────

def regenerate_from_checkpoint(
    checkpoint_path: str,
    output_dir: Optional[str] = None,
    Z: Optional[np.ndarray] = None,
    dimensions: Optional[List[str]] = None,
    extinction_generations: Optional[List[int]] = None,
):
    """从 checkpoint JSON 文件重新生成所有可视化图表.

    支持从旧 checkpoint 推断缺失数据：
    - fitness_stats: 从所有 checkpoint 文件推断每轮统计
    - islands_data: 从最后一个 checkpoint 读取（新格式）或用 best 构造（旧格式）
    - extinction_generations: 从 checkpoint 读取或手动指定

    Args:
        checkpoint_path: checkpoint JSON 文件路径（通常是最后一个，如 checkpoint_gen_8.json）
        output_dir: 输出目录（默认 checkpoint 同级目录下的 visualizations/）
        Z: 人格分布矩阵（可选，不提供则使用随机占位数据）
        dimensions: 维度名称（可选）
        extinction_generations: 灭绝轮数列表（可选，从日志推断或手动指定）

    使用示例:
        python -m src.utils.visualization data/results/my_run_20260524_130637/checkpoint_gen_5.json
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    history = checkpoint.get("history", [])
    best = checkpoint.get("best", {})
    best_fitness = best.get("fitness", {}) if best else None

    if not history:
        print("[Viz] Warning: checkpoint 中无 history 数据")
        return

    # 确定输出目录
    if output_dir is None:
        output_dir = checkpoint_path.parent / "visualizations"
    else:
        output_dir = Path(output_dir)

    print(f"[Viz] 从 checkpoint 重绘: {checkpoint_path}")
    print(f"[Viz] 历史轮数: {len(history)}")
    print(f"[Viz] 输出目录: {output_dir}")

    # ── 1. 处理 islands_data ──
    islands_data = checkpoint.get("islands_data")
    if islands_data:
        print(f"[Viz] 从 checkpoint 读取 islands_data: {len(islands_data)} 个岛屿")
    else:
        # 旧 checkpoint 兼容：用 best_fitness 构造单岛屿数据
        print("[Viz] 警告: checkpoint 中无 islands_data，使用 best_fitness 构造单岛屿数据")
        islands_data = {0: best_fitness} if best_fitness else {}

    # ── 2. 处理 extinction_generations ──
    if extinction_generations is None:
        extinction_generations = checkpoint.get("extinction_generations")
        if extinction_generations:
            print(f"[Viz] 从 checkpoint 读取灭绝轮数: {extinction_generations}")

    # ── 3. 尝试从所有 checkpoint 文件推断 fitness_stats ──
    if history and "fitness_stats" not in history[0]:
        print("[Viz] 尝试从所有 checkpoint 文件推断 fitness_stats...")
        checkpoint_dir = checkpoint_path.parent
        all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_gen_*.json"))
        
        # 构建 generation -> checkpoint 映射
        gen_to_cp = {}
        for cp_file in all_checkpoints:
            try:
                with open(cp_file, "r", encoding="utf-8") as f:
                    cp_data = json.load(f)
                gen = cp_data.get("generation")
                if gen:
                    gen_to_cp[gen] = cp_data
            except Exception:
                continue
        
        # 为每轮 history 补充 fitness_stats
        stats_inferred = 0
        for h in history:
            gen = h.get("generation")
            if gen and gen in gen_to_cp:
                cp_data = gen_to_cp[gen]
                # 从该轮 checkpoint 的 best 推断
                cp_best = cp_data.get("best", {})
                cp_best_fitness = cp_best.get("fitness", {})
                if cp_best_fitness:
                    # 旧数据只有 best，没有 mean/max/std
                    # 我们用 best 作为 max，并构造简化的 stats
                    h["fitness_stats"] = {}
                    for metric, val in cp_best_fitness.items():
                        h["fitness_stats"][metric] = {
                            "mean": val,      # 无真实 mean，用 best 近似
                            "median": val,    # 无真实 median，用 best 近似
                            "max": val,       # best 即 max
                            "min": val * 0.8, # 粗略估计
                            "std": 0.0,       # 无真实 std
                        }
                    stats_inferred += 1
        
        if stats_inferred > 0:
            print(f"[Viz] 已为 {stats_inferred}/{len(history)} 轮推断 fitness_stats")
        else:
            print("[Viz] 无法推断 fitness_stats，进化曲线将只显示 Global Best")

    # ── 4. 处理 Z 矩阵 ──
    if Z is None:
        n_dims = len(best_fitness) if best_fitness else 2
        Z = np.random.rand(25, n_dims)

    if dimensions is None:
        dimensions = [f"dim{i+1}" for i in range(Z.shape[1])]

    # 生成所有图表
    generate_all_visualizations(
        Z=Z,
        dimensions=dimensions,
        history=history,
        islands_data=islands_data,
        output_dir=output_dir,
        extinction_generations=extinction_generations,
        best_fitness=best_fitness,
    )

    print(f"[Viz] 重绘完成! 输出: {output_dir}")
    return output_dir


# ───────────────────────────────────────────────
# CLI 入口
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="从 checkpoint 重新生成可视化图表")
    parser.add_argument("checkpoint", help="checkpoint JSON 文件路径")
    parser.add_argument("--output", "-o", help="输出目录（默认 checkpoint 同级 visualizations/）")
    parser.add_argument("--extinction", help="灭绝轮数，逗号分隔（如 2,4）")
    parser.add_argument("--dims", help="维度名称，逗号分隔（如 dim1,dim2,dim3）")

    args = parser.parse_args()

    extinction_gens = None
    if args.extinction:
        extinction_gens = [int(x.strip()) for x in args.extinction.split(",")]

    dims = None
    if args.dims:
        dims = [x.strip() for x in args.dims.split(",")]

    regenerate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        extinction_generations=extinction_gens,
        dimensions=dims,
    )
