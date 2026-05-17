"""Open-Evolve 进化引擎 — 替代 AlphaEvolve."""

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from src.open_evolve.mutator import Mutator
from src.open_evolve.evaluator import PersonaCodeEvaluator
from src.open_evolve.code_templates import SEED_CODES
from src.utils.config import get_config


@dataclass
class Candidate:
    """进化候选（人格生成器代码）."""

    code: str
    fitness: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    island_id: int = 0
    seed_name: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "fitness": self.fitness,
            "generation": self.generation,
            "island_id": self.island_id,
            "seed_name": self.seed_name,
        }


class Island:
    """单个进化岛屿.

    维护 6 个精英位（对应 6 个指标）。
    若候选在任一指标上打破历史纪录，即成为该指标的新精英。
    """

    METRIC_NAMES = [
        "coverage", "convex_hull", "avg_dist",
        "min_dist", "dispersion", "kl_divergence",
    ]

    def __init__(self, island_id: int):
        self.id = island_id
        self.elites: Dict[str, Candidate] = {}  # metric_name → Candidate
        self.generation = 0
        self.last_improvement_gen = 0

    def update_elite(self, candidate: Candidate) -> Tuple[bool, List[str]]:
        """若候选在任一指标上打破纪录，更新精英.

        Returns:
            (是否成功上位, 打破纪录的指标列表)
        """
        improved_metrics = []
        for metric in self.METRIC_NAMES:
            current_best = self.elites.get(metric)
            if current_best is None:
                self.elites[metric] = copy.deepcopy(candidate)
                improved_metrics.append(metric)
            elif candidate.fitness.get(metric, float('-inf')) > current_best.fitness.get(metric, float('-inf')):
                self.elites[metric] = copy.deepcopy(candidate)
                improved_metrics.append(metric)

        if improved_metrics:
            self.last_improvement_gen = self.generation

        return len(improved_metrics) > 0, improved_metrics

    def get_best_candidate(self) -> Optional[Candidate]:
        """获取该岛屿当前最优候选（按覆盖率的精英）."""
        return self.elites.get("coverage") or next(iter(self.elites.values()), None)

    def get_all_elites(self) -> List[Candidate]:
        """获取所有精英（去重）."""
        seen_codes = set()
        elites = []
        for c in self.elites.values():
            if c.code not in seen_codes:
                seen_codes.add(c.code)
                elites.append(c)
        return elites

    def reset(self, new_seed: Candidate):
        """重置岛屿（从其他岛屿复制精英）."""
        self.elites = {}
        self.generation = 0
        self.last_improvement_gen = 0
        # 将新种子作为所有指标的初始精英
        for metric in self.METRIC_NAMES:
            self.elites[metric] = copy.deepcopy(new_seed)


class OpenEvolve:
    """Open-Evolve 进化引擎.

    核心机制:
      - 10 个并行岛屿，每岛 6 个精英位
      - LLM 驱动的代码变异算子
      - 若候选在任一指标打破纪录即上位
      - 每 100 轮/8 小时触发周期性灭绝
      - 反馈机制：观察 empirical data 指导改进
    """

    def __init__(
        self,
        mutator: Mutator,
        evaluator: PersonaCodeEvaluator,
        questionnaires: List,
    ):
        self.mutator = mutator
        self.evaluator = evaluator
        self.questionnaires = questionnaires

        cfg = get_config()
        self.num_islands = cfg.get("open_evolve.num_islands", 10)
        self.extinction_interval = cfg.get("open_evolve.extinction_interval", 100)
        self.extinction_hours = cfg.get("open_evolve.extinction_interval_hours", 8)

        # 初始种子分配
        distribution = cfg.get("open_evolve.initial_seed_distribution", {
            "seed1": 4, "seed2": 3, "seed3": 3,
        })

        self.islands = [Island(i) for i in range(self.num_islands)]
        self._initialize_islands(distribution)

        self.generation = 0
        self.start_time = time.time()
        self.history: List[dict] = []
        self.checkpoint_path = Path("data/results")
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def _initialize_islands(self, distribution: Dict[str, int]):
        """初始种子轮询分配."""
        seed_list = []
        for seed_name, count in distribution.items():
            code = SEED_CODES.get(seed_name, SEED_CODES["seed1"])
            for _ in range(count):
                seed_list.append((seed_name, code))

        # 轮询分配到各岛屿
        for i, island in enumerate(self.islands):
            seed_name, code = seed_list[i % len(seed_list)]
            # 评估初始种子
            print(f"  [Init] 评估岛屿 {i} 的初始种子 {seed_name}...")
            fitness = self.evaluator.evaluate(code)
            candidate = Candidate(
                code=code,
                fitness=fitness,
                generation=0,
                island_id=i,
                seed_name=seed_name,
            )
            island.update_elite(candidate)
            print(f"  [Init] 岛屿 {i} 初始化完成: { {k: f'{v:.3f}' for k, v in fitness.items()} }")

    def evolve_once(self) -> dict:
        """单轮进化.

        对每个岛屿:
          1. 选择父代（随机选一个精英）
          2. 变异生成子代
          3. 评估子代
          4. 尝试更新精英
        """
        self.generation += 1
        print(f"\n{'='*60}")
        print(f"Generation {self.generation}")
        print(f"{'='*60}")

        round_stats = {
            "generation": self.generation,
            "improvements": 0,
            "evaluations": 0,
            "time": time.time(),
        }

        for island in self.islands:
            island.generation = self.generation

            # 1. 选择父代
            elites = island.get_all_elites()
            if not elites:
                continue
            parent = random.choice(elites)

            # 2. 变异
            try:
                child_code = self.mutator.mutate(parent.code)
            except Exception as e:
                print(f"  [Island {island.id}] 变异失败: {e}")
                continue

            # 3. 评估
            child_fitness = self.evaluator.evaluate(child_code)
            round_stats["evaluations"] += 1

            child = Candidate(
                code=child_code,
                fitness=child_fitness,
                generation=self.generation,
                island_id=island.id,
                seed_name=parent.seed_name,
            )

            # 4. 更新精英
            improved, metrics = island.update_elite(child)
            if improved:
                round_stats["improvements"] += 1
                print(f"  [Island {island.id}] 🏆 打破纪录: {metrics}")

        # 检查灭绝
        self._check_extinction()

        # 记录历史
        best = self.get_global_best()
        if best:
            round_stats["best_fitness"] = best.fitness
            print(f"\n  本轮最优: { {k: f'{v:.3f}' for k, v in best.fitness.items()} }")

        self.history.append(round_stats)
        return round_stats

    def _check_extinction(self):
        """检查是否触发周期性灭绝."""
        # 按轮数触发
        if self.generation > 0 and self.generation % self.extinction_interval == 0:
            self._trigger_extinction()
            return

        # 按时间触发
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours >= self.extinction_hours:
            self._trigger_extinction()

    def _trigger_extinction(self):
        """触发灭绝：重置表现差的岛屿，从最优岛复制精英."""
        print(f"\n{'='*60}")
        print("🔥 周期性灭绝触发!")
        print(f"{'='*60}")

        # 找出最优岛屿
        best_island = max(self.islands, key=lambda i: (
            i.elites.get("coverage", Candidate("")).fitness.get("coverage", 0)
        ))
        best_seed = best_island.get_best_candidate()

        if best_seed is None:
            return

        # 重置所有岛屿（除了最优岛）
        for island in self.islands:
            if island.id == best_island.id:
                continue

            # 检查该岛屿是否需要重置（长时间无改进）
            gens_since_improvement = self.generation - island.last_improvement_gen
            if gens_since_improvement > self.extinction_interval // 2:
                print(f"  [Island {island.id}] 已 {gens_since_improvement} 轮无改进，重置...")
                island.reset(best_seed)

    def get_global_best(self) -> Optional[Candidate]:
        """获取全局最优候选."""
        all_elites = []
        for island in self.islands:
            all_elites.extend(island.get_all_elites())

        if not all_elites:
            return None

        # 按 coverage 排序
        return max(all_elites, key=lambda c: c.fitness.get("coverage", 0))

    def run(self, max_generations: int = None, max_hours: float = None):
        """主进化循环.

        Args:
            max_generations: 最大进化轮数
            max_hours: 最大运行时间（小时）
        """
        print("=" * 60)
        print("Open-Evolve 进化引擎启动")
        print("=" * 60)

        while True:
            # 检查终止条件
            if max_generations and self.generation >= max_generations:
                print(f"\n达到最大轮数 {max_generations}，停止进化。")
                break

            elapsed_hours = (time.time() - self.start_time) / 3600
            if max_hours and elapsed_hours >= max_hours:
                print(f"\n达到最大时间 {max_hours}h，停止进化。")
                break

            # 单轮进化
            self.evolve_once()

            # 保存 checkpoint
            self._save_checkpoint()

        # 输出最终结果
        best = self.get_global_best()
        if best:
            print(f"\n{'='*60}")
            print("进化完成!")
            print(f"最优代码来自: 岛屿 {best.island_id}, 种子 {best.seed_name}, 代数 {best.generation}")
            print(f"适应度: { {k: f'{v:.4f}' for k, v in best.fitness.items()} }")
            print(f"{'='*60}")

        return best

    def _save_checkpoint(self):
        """保存进化状态."""
        checkpoint = {
            "generation": self.generation,
            "history": self.history,
            "best": self.get_global_best().to_dict() if self.get_global_best() else None,
        }
        path = self.checkpoint_path / f"checkpoint_gen_{self.generation}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
