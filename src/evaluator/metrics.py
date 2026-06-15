"""多样性评估器 — 6 个多样性指标."""

import warnings
from typing import Dict, List

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from scipy.stats import qmc


class DiversityMetrics:
    """计算人格分布的 6 个多样性指标.

    指标列表:
      1. Coverage       — 球覆盖比例（越大越好）
      2. ConvexHull     — 凸包体积（越大越好）
      3. AvgDist        — 平均两两距离（越大越好）
      4. MinDist        — 最小两两距离（越大越好）
      5. Dispersion     — 最大空白区半径（越小越好 → 符号反转）
      6. KL Divergence  — 与 Sobol 分布的 KL 散度（越小越好 → 符号反转）
    """

    NUM_RANDOM_POINTS = 1000   # 从10000减少到1000
    COVERAGE_TARGET = 0.99
    COVERAGE_CALIBRATION_TRIALS = 100  # 从1000减少到100
    DEFAULT_RANDOM_SEED = 20260613

    def __init__(self, coverage_radius: float = None, random_seed: int | None = None):
        """
        Args:
            coverage_radius: Coverage 球的半径 k。
                若为 None，则自动根据 N=25, dim=2/3 校准。
            random_seed: 参考点采样种子。默认固定，保证同一输入重复评估一致。
        """
        self.k = coverage_radius
        self.random_seed = self.DEFAULT_RANDOM_SEED if random_seed is None else random_seed

    # ------------------------------------------------------------------
    # 6 个核心指标
    # ------------------------------------------------------------------

    def coverage(self, Z: np.ndarray) -> float:
        """覆盖率: 随机参考点被球覆盖的比例.

        算法:
          1. 在 Z 的包围盒内撒 NUM_RANDOM_POINTS 个随机点
          2. 对每个随机点，检查是否落在任一 z_i 为中心、半径 k 的球内
          3. 返回被覆盖的比例
        """
        if len(Z) == 0:
            return 0.0

        n, dim = Z.shape
        k = self._get_coverage_radius(n, dim)

        # 在单位空间 [0,1]^d 内撒点（与校准时的采样空间一致）
        random_points = self._reference_points(dim, salt=1)

        # 用 cKDTree 加速近邻查询
        tree = cKDTree(Z)
        distances, _ = tree.query(random_points, k=1)
        covered = (distances <= k).sum()

        return covered / self.NUM_RANDOM_POINTS

    def convex_hull_volume(self, Z: np.ndarray) -> float:
        """凸包体积.

        注意: 低维数据（点数 ≤ 维数）时返回 0。
        """
        if len(Z) <= Z.shape[1]:
            return 0.0
        try:
            hull = ConvexHull(Z)
            return hull.volume
        except Exception:
            return 0.0

    def avg_pairwise_dist(self, Z: np.ndarray) -> float:
        """平均两两欧氏距离."""
        if len(Z) <= 1:
            return 0.0
        dists = cdist(Z, Z)
        # 取上三角（排除对角线）
        triu_idx = np.triu_indices_from(dists, k=1)
        return float(dists[triu_idx].mean())

    def min_pairwise_dist(self, Z: np.ndarray) -> float:
        """最小两两欧氏距离."""
        if len(Z) <= 1:
            return 0.0
        dists = cdist(Z, Z)
        np.fill_diagonal(dists, np.inf)
        return float(dists.min())

    def dispersion(self, Z: np.ndarray) -> float:
        """最大空白区半径: 随机参考点到最近 z_i 的最大距离.

        越小表示覆盖越均匀。在适应度计算中会符号反转。
        
        在单位空间 [0,1]^d 内撒点（与 coverage 一致）。
        """
        if len(Z) == 0:
            return 0.0

        n, dim = Z.shape

        # 在单位空间 [0,1]^d 内撒点
        random_points = self._reference_points(dim, salt=2)

        tree = cKDTree(Z)
        distances, _ = tree.query(random_points, k=1)
        return float(distances.max())

    def kl_divergence(self, Z: np.ndarray, num_ref_points: int = 10000) -> float:
        """KL 散度: Z 的经验分布 vs Sobol 准随机参考分布.

        在固定的单位空间 [0, 1]^d 内比较分布均匀性。
        越小表示 Z 的分布越接近均匀。在适应度计算中会符号反转。
        """
        if len(Z) == 0:
            return 0.0

        n, dim = Z.shape
        Z = np.clip(Z, 0.0, 1.0)

        # 使用固定单位空间直方图估计分布。
        # 注意：如果按 Z 自己的包围盒归一化，聚成一小团的点也会被误判为均匀。
        # bin 数根据样本量自适应：每维约 N^(1/d) 个 bin，确保平均每 bin 有若干点
        bins_per_dim = max(3, int(n ** (1.0 / dim)))
        bins = [bins_per_dim] * dim
        ranges = [(0.0, 1.0) for _ in range(dim)]

        # Z 的经验分布
        hist_z, _ = np.histogramdd(Z, bins=bins, range=ranges)
        hist_z = hist_z.flatten() + 1e-10
        hist_z = hist_z / hist_z.sum()

        # 单位空间上的均匀参考分布
        hist_ref = np.ones_like(hist_z, dtype=float)
        hist_ref = hist_ref / hist_ref.sum()

        # KL(Z || Ref)
        kl = entropy(hist_z, hist_ref)
        return float(kl)

    # ------------------------------------------------------------------
    # 聚合评估
    # ------------------------------------------------------------------

    def evaluate(self, Z: np.ndarray) -> Dict[str, float]:
        """计算全部 6 个指标.

        Returns:
            dict: {metric_name: value}
        """
        return {
            "coverage": self.coverage(Z),
            "convex_hull": self.convex_hull_volume(Z),
            "avg_dist": self.avg_pairwise_dist(Z),
            "min_dist": self.min_pairwise_dist(Z),
            "dispersion": self.dispersion(Z),
            "kl_divergence": self.kl_divergence(Z),
        }

    def fitness(self, Z: np.ndarray) -> Dict[str, float]:
        """计算适应度分数（统一为"越大越好"）.

        Dispersion 和 KL Divergence 取负值（或倒数），使其方向一致。
        """
        metrics = self.evaluate(Z)
        fitness = {
            "coverage": metrics["coverage"],
            "convex_hull": metrics["convex_hull"],
            "avg_dist": metrics["avg_dist"],
            "min_dist": metrics["min_dist"],
            # 符号反转: 越小越好 → 越大越好
            "dispersion": -metrics["dispersion"],
            "kl_divergence": -metrics["kl_divergence"],
        }
        return fitness

    # ------------------------------------------------------------------
    # Coverage 半径校准
    # ------------------------------------------------------------------

    @classmethod
    def calibrate_coverage_radius(
        cls,
        n: int = 25,
        dim: int = 2,
        trials: int = None,
        target: float = None,
    ) -> float:
        """校准 Coverage 半径 k.

        从 Sobol 分布采样 N 个点，找覆盖 target 比例空间的最小半径，
        重复 trials 次取平均。

        Args:
            n: 每次采样的点数（默认 25，匹配评估时的 personas_per_evaluation）
            dim: 维度数
            trials: 重复次数（默认 1000）
            target: 目标覆盖比例（默认 0.99）

        Returns:
            float: 校准后的半径 k
        """
        trials = trials or cls.COVERAGE_CALIBRATION_TRIALS
        target = target or cls.COVERAGE_TARGET

        radii = []
        for _ in range(trials):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    points = cls._sobol_points(dim, n)
            except Exception:
                points = np.random.rand(n, dim)

            # 二分查找最小半径
            r_low, r_high = 0.0, 2.0  # 在归一化空间，最大距离约 sqrt(dim)
            for _ in range(30):  # 二分 30 次足够精确
                r_mid = (r_low + r_high) / 2.0

                # 检查覆盖比例
                rng = np.random.default_rng(20260613 + dim * 1009 + n * 9176 + len(radii))
                random_pts = rng.random((cls.NUM_RANDOM_POINTS, dim))
                tree = cKDTree(points)
                dists, _ = tree.query(random_pts, k=1)
                coverage = (dists <= r_mid).mean()

                if coverage >= target:
                    r_high = r_mid
                else:
                    r_low = r_mid

            radii.append((r_low + r_high) / 2.0)

        return float(np.mean(radii))

    @staticmethod
    def _sobol_points(dim: int, n: int) -> np.ndarray:
        """生成 Sobol 点，并避免 n 非 2 次幂时的均匀性警告."""
        if n <= 0:
            return np.empty((0, dim))
        m = int(np.ceil(np.log2(n)))
        sampler = qmc.Sobol(
            d=dim,
            scramble=True,
            seed=DiversityMetrics.DEFAULT_RANDOM_SEED + dim * 1009 + n * 9176,
        )
        return sampler.random_base2(m=m)[:n]

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_coverage_radius(self, n: int, dim: int) -> float:
        """获取 Coverage 半径（优先使用预校准值，否则动态计算）."""
        if self.k is not None:
            return self.k
        # 动态校准（按 (n, dim) 缓存）
        cache_key = (n, dim)
        if not hasattr(self, "_cached_radius"):
            self._cached_radius = {}
        if cache_key not in self._cached_radius:
            print(f"  [Evaluator] 校准 Coverage 半径 (N={n}, dim={dim})...")
            self._cached_radius[cache_key] = self.calibrate_coverage_radius(n=n, dim=dim)
            print(f"  [Evaluator] 校准完成: k = {self._cached_radius[cache_key]:.4f}")
        return self._cached_radius[cache_key]

    def _reference_points(self, dim: int, salt: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed + dim * 1009 + salt * 9176)
        return rng.random((self.NUM_RANDOM_POINTS, dim))


class MultiQuestionnaireEvaluator:
    """跨多份问卷的聚合评估器.

    对 40 份问卷分别计算 M(Z)，取平均作为最终适应度。
    """

    def __init__(self, coverage_radius: float = None):
        self.metrics = DiversityMetrics(coverage_radius)

    def evaluate_batch(
        self,
        questionnaire_results: List[np.ndarray],
    ) -> Dict[str, float]:
        """评估多份问卷的结果并取平均.

        Args:
            questionnaire_results: 每份问卷的 Z 矩阵列表
                每个 Z 形状为 (N, K)

        Returns:
            dict: 6 个指标的平均值
        """
        all_metrics = []
        for i, Z in enumerate(questionnaire_results):
            if Z is None or len(Z) == 0:
                continue
            m = self.metrics.fitness(Z)
            all_metrics.append(m)

        if not all_metrics:
            return {name: 0.0 for name in DiversityMetrics().evaluate(np.zeros((1, 2))).keys()}

        # 取平均
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = float(np.mean(values))

        return avg_metrics
