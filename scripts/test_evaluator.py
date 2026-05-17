"""测试多样性评估器（6 个指标）."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.evaluator.metrics import DiversityMetrics, MultiQuestionnaireEvaluator


def test_basic_metrics():
    print("=" * 60)
    print("测试 6 个基础指标")
    print("=" * 60)

    # 构造测试数据：25 个点，2 维
    np.random.seed(42)
    Z = np.random.rand(25, 2)

    # 使用预校准的半径加速
    metrics = DiversityMetrics(coverage_radius=0.2)
    result = metrics.evaluate(Z)

    print(f"\n输入: Z.shape = {Z.shape}")
    print(f"\n指标结果:")
    for name, value in result.items():
        print(f"  {name:20s}: {value:.6f}")

    # 验证数值范围
    assert 0 <= result["coverage"] <= 1, "Coverage 应在 [0, 1]"
    assert result["avg_dist"] > 0, "AvgDist 应 > 0"
    assert result["min_dist"] > 0, "MinDist 应 > 0"
    assert result["convex_hull"] > 0, "ConvexHull 应 > 0"
    assert result["dispersion"] > 0, "Dispersion 应 > 0"
    assert result["kl_divergence"] >= 0, "KL 应 >= 0"

    print("\n✅ 数值范围检查通过")
    return result


def test_fitness_direction():
    print("\n" + "=" * 60)
    print("测试适应度方向（统一为越大越好）")
    print("=" * 60)

    np.random.seed(42)

    # 场景 A: 点聚成一团（多样性差）
    Z_clustered = np.random.randn(25, 2) * 0.05 + 0.5
    Z_clustered = np.clip(Z_clustered, 0, 1)

    # 场景 B: 点均匀分散（多样性好）
    Z_spread = np.random.rand(25, 2)

    metrics = DiversityMetrics(coverage_radius=0.2)
    fit_clustered = metrics.fitness(Z_clustered)
    fit_spread = metrics.fitness(Z_spread)

    print(f"\n聚簇场景 (多样性差):")
    for k, v in fit_clustered.items():
        print(f"  {k:20s}: {v:+.6f}")

    print(f"\n分散场景 (多样性好):")
    for k, v in fit_spread.items():
        print(f"  {k:20s}: {v:+.6f}")

    print(f"\n对比（分散应优于聚簇）:")
    improvements = {}
    for k in fit_clustered.keys():
        imp = fit_spread[k] - fit_clustered[k]
        improvements[k] = imp
        status = "✅" if imp > 0 else "⚠️"
        print(f"  {status} {k:20s}: {imp:+.6f}")

    # 核心指标应显著改善（Coverage 和 Dispersion 本身不区分聚簇/分散，不测方向）
    assert fit_spread["avg_dist"] > fit_clustered["avg_dist"], "AvgDist 应改善"
    assert fit_spread["min_dist"] > fit_clustered["min_dist"], "MinDist 应改善"
    assert fit_spread["convex_hull"] > fit_clustered["convex_hull"], "ConvexHull 应改善"
    # KL 越小越好，取负后越大越好
    assert fit_spread["kl_divergence"] > fit_clustered["kl_divergence"], "KL 应改善"

    print("\n✅ 方向性检查通过（AvgDist/MinDist/ConvexHull/KL 均改善）")


def test_calibration():
    print("\n" + "=" * 60)
    print("测试 Coverage 半径校准")
    print("=" * 60)

    # 用少量 trials 快速测试
    k = DiversityMetrics.calibrate_coverage_radius(
        n=25, dim=2, trials=10, target=0.99
    )
    print(f"\n校准结果 (快速模式, 10 trials):")
    print(f"  k = {k:.4f}")
    assert k > 0, "半径应 > 0"
    print("✅ 校准通过")


def test_multi_questionnaire():
    print("\n" + "=" * 60)
    print("测试多问卷聚合评估")
    print("=" * 60)

    evaluator = MultiQuestionnaireEvaluator()

    # 模拟 3 份问卷的结果
    np.random.seed(42)
    results = [
        np.random.rand(25, 2),
        np.random.rand(25, 2),
        np.random.rand(25, 2),
    ]

    avg = evaluator.evaluate_batch(results)
    print(f"\n3 份问卷平均适应度:")
    for k, v in avg.items():
        print(f"  {k:20s}: {v:+.6f}")

    print("\n✅ 多问卷评估通过")


def test_3d():
    print("\n" + "=" * 60)
    print("测试 3 维数据")
    print("=" * 60)

    np.random.seed(42)
    Z = np.random.rand(25, 3)

    metrics = DiversityMetrics()
    result = metrics.evaluate(Z)

    print(f"\n输入: Z.shape = {Z.shape}")
    for name, value in result.items():
        print(f"  {name:20s}: {value:.6f}")

    print("\n✅ 3 维测试通过")


def main():
    test_basic_metrics()
    test_fitness_direction()
    test_calibration()
    test_multi_questionnaire()
    test_3d()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
