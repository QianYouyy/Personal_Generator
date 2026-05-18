"""测试可视化模块."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.utils.visualization import (
    plot_persona_distribution,
    plot_evolution_curve,
    plot_island_heatmap,
    plot_coverage_comparison,
    generate_all_visualizations,
)


def test_2d_distribution():
    print("=" * 60)
    print("测试 2D 人格分布散点图")
    print("=" * 60)

    np.random.seed(42)
    Z = np.random.rand(25, 2)
    dimensions = ["adaptability", "risk_tolerance"]

    plot_persona_distribution(
        Z, dimensions,
        title="Test 2D Distribution",
        save_path="data/results/visualizations/test_2d.png",
    )
    print("✅ 2D 散点图已保存")


def test_3d_distribution():
    print("\n" + "=" * 60)
    print("测试 3D 人格分布散点图")
    print("=" * 60)

    np.random.seed(42)
    Z = np.random.rand(25, 3)
    dimensions = ["adaptability", "risk_tolerance", "social_compliance"]

    plot_persona_distribution(
        Z, dimensions,
        title="Test 3D Distribution",
        save_path="data/results/visualizations/test_3d.png",
    )
    print("✅ 3D 散点图已保存")


def test_evolution_curve():
    print("\n" + "=" * 60)
    print("测试进化曲线")
    print("=" * 60)

    history = []
    for gen in range(1, 21):
        history.append({
            "generation": gen,
            "best_fitness": {
                "coverage": 0.5 + 0.02 * gen + np.random.normal(0, 0.02),
                "convex_hull": 0.3 + 0.01 * gen + np.random.normal(0, 0.01),
                "avg_dist": 0.4 + 0.015 * gen + np.random.normal(0, 0.015),
                "min_dist": 0.1 + 0.005 * gen + np.random.normal(0, 0.005),
                "dispersion": -(0.2 - 0.005 * gen),
                "kl_divergence": -(1.0 - 0.02 * gen),
            },
        })

    plot_evolution_curve(
        history,
        save_path="data/results/visualizations/test_evolution.png",
    )
    print("✅ 进化曲线已保存")


def test_island_heatmap():
    print("\n" + "=" * 60)
    print("测试岛屿热力图")
    print("=" * 60)

    islands_data = {}
    for i in range(10):
        islands_data[i] = {
            "coverage": np.random.uniform(0.5, 1.0),
            "convex_hull": np.random.uniform(0.1, 0.8),
            "avg_dist": np.random.uniform(0.3, 0.7),
            "min_dist": np.random.uniform(0.05, 0.2),
            "dispersion": np.random.uniform(-0.5, -0.1),
            "kl_divergence": np.random.uniform(-2.0, -0.5),
        }

    plot_island_heatmap(
        islands_data,
        save_path="data/results/visualizations/test_heatmap.png",
    )
    print("✅ 岛屿热力图已保存")


def test_comparison():
    print("\n" + "=" * 60)
    print("测试分布对比")
    print("=" * 60)

    np.random.seed(42)
    Z1 = np.random.rand(25, 2) * 0.3 + 0.35  # 聚簇
    Z2 = np.random.rand(25, 2)  # 分散

    plot_coverage_comparison(
        [Z1, Z2],
        labels=["Clustered (Bad)", "Spread (Good)"],
        dimensions=["dim1", "dim2"],
        save_path="data/results/visualizations/test_comparison.png",
    )
    print("✅ 分布对比图已保存")


def test_all_in_one():
    print("\n" + "=" * 60)
    print("测试一键生成所有图表")
    print("=" * 60)

    np.random.seed(42)
    Z = np.random.rand(25, 2)
    dimensions = ["adaptability", "risk_tolerance"]

    history = []
    for gen in range(1, 11):
        history.append({
            "generation": gen,
            "best_fitness": {
                "coverage": 0.6 + 0.03 * gen,
                "avg_dist": 0.4 + 0.02 * gen,
            },
        })

    islands_data = {i: {"coverage": np.random.uniform(0.5, 1.0),
                        "avg_dist": np.random.uniform(0.3, 0.7)}
                    for i in range(5)}

    generate_all_visualizations(
        Z, dimensions, history, islands_data,
        output_dir="data/results/visualizations/all_in_one",
    )
    print("✅ 所有图表已保存")


def main():
    test_2d_distribution()
    test_3d_distribution()
    test_evolution_curve()
    test_island_heatmap()
    test_comparison()
    test_all_in_one()

    print("\n" + "=" * 60)
    print("所有可视化测试通过!")
    print("图表保存在: data/results/visualizations/")
    print("=" * 60)


if __name__ == "__main__":
    main()
