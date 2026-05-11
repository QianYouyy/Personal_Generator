"""多样性评估器 - 6 个指标."""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from scipy.spatial import cKDTree


class DiversityEvaluator:
    """计算人格分布的 6 个多样性指标."""

    NUM_RANDOM_POINTS = 10000

    def __init__(self, coverage_radius: float):
        self.k = coverage_radius  # 校准后的 Coverage 半径

    def coverage(self, Z: np.ndarray) -> float:
        """覆盖率: 随机点被球覆盖的比例."""
        # TODO: 在 K 维空间撒点，计算覆盖比例
        raise NotImplementedError("Phase 2 实现中")

    def convex_hull_volume(self, Z: np.ndarray) -> float:
        """凸包体积."""
        if len(Z) <= Z.shape[1]:
            return 0.0
        try:
            hull = ConvexHull(Z)
            return hull.volume
        except Exception:
            return 0.0

    def avg_pairwise_dist(self, Z: np.ndarray) -> float:
        """平均两两距离."""
        dists = cdist(Z, Z)
        # 取上三角（排除对角线）
        return dists[np.triu_indices_from(dists, k=1)].mean()

    def min_pairwise_dist(self, Z: np.ndarray) -> float:
        """最小两两距离."""
        dists = cdist(Z, Z)
        # 排除对角线（距离为0）
        np.fill_diagonal(dists, np.inf)
        return dists.min()

    def dispersion(self, Z: np.ndarray) -> float:
        """最大空白区半径（随机参考点到最近 z_i 的最大距离）."""
        # TODO: 撒随机参考点，找最大距离
        raise NotImplementedError("Phase 2 实现中")

    def kl_divergence(self, Z: np.ndarray) -> float:
        """KL 散度: Z 的经验分布 vs Sobol 准随机参考分布."""
        # TODO: 用核密度估计 + Sobol 采样计算 KL
        raise NotImplementedError("Phase 2 实现中")

    def evaluate(self, Z: np.ndarray) -> dict:
        """计算全部 6 个指标."""
        return {
            "coverage": self.coverage(Z),
            "convex_hull": self.convex_hull_volume(Z),
            "avg_dist": self.avg_pairwise_dist(Z),
            "min_dist": self.min_pairwise_dist(Z),
            "dispersion": self.dispersion(Z),
            "kl_divergence": self.kl_divergence(Z),
        }

    @staticmethod
    def calibrate_coverage_radius(n: int = 25, dim: int = 2, trials: int = 1000) -> float:
        """校准 Coverage 半径 k.

        从 Sobol 分布采样 N 个点，找覆盖 99% 空间的最小半径，
        重复 trials 次取平均.
        """
        # TODO: Phase 2 实现
        raise NotImplementedError("Phase 2 实现中")
