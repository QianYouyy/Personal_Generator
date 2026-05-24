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
from src.utils.logger import logger


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
        self.total_evaluations = 0
        self.total_improvements = 0

    def update_elite(self, candidate: Candidate) -> Tuple[bool, List[str]]:
        """若候选在任一指标上打破纪录，更新精英."""
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
            self.total_improvements += len(improved_metrics)

        self.total_evaluations += 1
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

    def get_stats(self) -> dict:
        """获取岛屿统计信息."""
        best = self.get_best_candidate()
        return {
            "id": self.id,
            "elites": len(self.elites),
            "evaluations": self.total_evaluations,
            "improvements": self.total_improvements,
            "last_improvement": self.last_improvement_gen,
            "gens_since_improvement": self.generation - self.last_improvement_gen,
            "best_coverage": best.fitness.get("coverage", 0) if best else 0,
        }

    def reset(self, new_seed: Candidate, current_gen: int = 0):
        """重置岛屿（从其他岛屿复制精英）."""
        self.elites = {}
        # 保留当前 generation 计数，不重置为0
        self.last_improvement_gen = current_gen
        for metric in self.METRIC_NAMES:
            self.elites[metric] = copy.deepcopy(new_seed)


class OpenEvolve:
    """Open-Evolve 进化引擎."""

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
        # 动态灭绝阈值：根据总轮数自适应（至少2轮）
        self.extinction_stagnation_threshold = max(2, self.extinction_interval // 10)
        # 灭绝模式: "interval" | "stagnation" | "adaptive"
        self.extinction_mode = cfg.get("open_evolve.extinction_mode", "adaptive")

        distribution = cfg.get("open_evolve.initial_seed_distribution", {
            "seed1": 4, "seed2": 3, "seed3": 3,
        })

        self.islands = [Island(i) for i in range(self.num_islands)]
        self._initialize_islands(distribution)

        self.generation = 0
        self.start_time = time.time()
        self.history: List[dict] = []
        self._extinction_log: List[int] = []  # 记录灭绝发生的轮数
        
        # checkpoint 保存到统一输出目录
        from src.utils.output_manager import output_manager
        if output_manager.base_dir is None:
            output_manager.setup("default")
        self.checkpoint_path = output_manager.outputs_dir

    def _initialize_islands(self, distribution: Dict[str, int]):
        """初始种子轮询分配."""
        seed_list = []
        for seed_name, count in distribution.items():
            code = SEED_CODES.get(seed_name, SEED_CODES["seed1"])
            for _ in range(count):
                seed_list.append((seed_name, code))

        logger.info(f"初始化 {self.num_islands} 个岛屿...")
        for i, island in enumerate(self.islands):
            seed_name, code = seed_list[i % len(seed_list)]
            logger.info(f"  [Island {i}] 评估初始种子 {seed_name}...")
            fitness = self.evaluator.evaluate(code)
            candidate = Candidate(
                code=code,
                fitness=fitness,
                generation=0,
                island_id=i,
                seed_name=seed_name,
            )
            island.update_elite(candidate)
            logger.success(f"  [Island {i}] 初始化完成 | Coverage: {fitness.get('coverage', 0):.3f} | AvgDist: {fitness.get('avg_dist', 0):.3f}")

    def evolve_once(self) -> dict:
        """单轮进化."""
        self.generation += 1
        gen_start = time.time()

        logger.section(f"Generation {self.generation}/{self.max_generations if hasattr(self, 'max_generations') else '?'}")

        round_stats = {
            "generation": self.generation,
            "improvements": 0,
            "evaluations": 0,
            "time": time.time(),
        }

        # 每轮处理所有岛屿
        total_islands = len(self.islands)
        for idx, island in enumerate(self.islands):
            island.generation = self.generation

            # 进度条
            logger.progress(idx + 1, total_islands, f"Island {island.id}")

            # 1. 选择父代
            elites = island.get_all_elites()
            if not elites:
                logger.warn(f"[Island {island.id}] 无精英，跳过")
                continue
            parent = random.choice(elites)
            logger.debug(f"[Island {island.id}] 选择父代 (gen={parent.generation}, seed={parent.seed_name})")

            # 2. 变异
            try:
                logger.debug(f"[Island {island.id}] 执行变异...")
                child_code = self.mutator.mutate(parent.code)
                logger.debug(f"[Island {island.id}] 变异完成，代码长度: {len(child_code)} chars")
            except Exception as e:
                logger.error(f"[Island {island.id}] 变异失败: {e}")
                continue

            # 3. 评估
            logger.debug(f"[Island {island.id}] 评估子代...")
            eval_start = time.time()
            child_fitness = self.evaluator.evaluate(child_code)
            eval_time = time.time() - eval_start
            round_stats["evaluations"] += 1
            logger.debug(f"[Island {island.id}] 评估完成 ({eval_time:.1f}s)")

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
                logger.success(f"[Island {island.id}] 🏆 打破 {len(metrics)} 项纪录: {metrics}")
                for m in metrics:
                    logger.metric(m, child_fitness.get(m, 0))
            else:
                logger.debug(f"[Island {island.id}] 未打破纪录")

        # 检查灭绝
        self._check_extinction()

        # 记录历史
        best = self.get_global_best()
        if best:
            round_stats["best_fitness"] = best.fitness
            logger.success(f"本轮最优 (Island {best.island_id}):")
            for k, v in best.fitness.items():
                logger.metric(k, v)

        # 岛屿统计
        logger.info("岛屿状态:")
        for island in self.islands:
            stats = island.get_stats()
            logger.info(f"  Island {stats['id']}: evals={stats['evaluations']}, "
                       f"improvements={stats['improvements']}, "
                       f"last_improve={stats['last_improvement']}, "
                       f"coverage={stats['best_coverage']:.3f}")

        gen_time = time.time() - gen_start
        logger.info(f"本轮耗时: {gen_time:.1f}s | 评估: {round_stats['evaluations']} | 改进: {round_stats['improvements']}")

        self.history.append(round_stats)
        return round_stats

    def _check_extinction(self):
        """检查是否触发周期性灭绝.
        
        三种触发方式：
        1. 固定间隔触发（每 _effective_interval 轮）
        2. 停滞触发：岛屿超过 _effective_stagnation 轮无改进
        3. 时间触发（长时间运行）
        
        优先级：固定间隔 > 时间 > 停滞检测
        避免连续两轮触发（保护期1轮）
        """
        # 使用实际生效的参数
        effective_interval = getattr(self, '_effective_interval', self.extinction_interval)
        effective_stagnation = getattr(self, '_effective_stagnation', self.extinction_stagnation_threshold)
        
        # 检查保护期（避免连续触发）
        last_extinction = getattr(self, '_last_extinction_gen', 0)
        if self.generation - last_extinction <= 1:
            return
        
        # 方式1：固定间隔触发
        if self.generation > 0 and self.generation % effective_interval == 0:
            self._last_extinction_gen = self.generation
            self._trigger_extinction(reason=f"达到 {effective_interval} 轮间隔")
            return

        # 方式2：时间触发
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours >= self.extinction_hours:
            self._last_extinction_gen = self.generation
            self._trigger_extinction(reason=f"达到 {self.extinction_hours} 小时")
            return

        # 方式3：动态停滞触发（根据总轮数自适应）
        stagnated = []
        for island in self.islands:
            gens_since = self.generation - island.last_improvement_gen
            if gens_since >= effective_stagnation:
                stagnated.append((island.id, gens_since))
        
        if stagnated:
            self._last_extinction_gen = self.generation
            logger.info(f"检测到 {len(stagnated)} 个岛屿停滞（阈值: {effective_stagnation} 轮）: {stagnated}")
            self._trigger_extinction(reason=f"停滞检测", only_stagnated=stagnated)

    def _get_best_island(self) -> Island:
        """多指标综合选最优岛屿（用于灭绝时选择最优种子）.
        
        评分权重：
        - coverage: 30%（覆盖度，核心指标）
        - convex_hull: 15%（凸包体积，空间覆盖）
        - avg_dist: 20%（平均距离，分散度）
        - min_dist: 15%（最小距离，避免聚集）
        - dispersion: 10%（离散度）
        - kl_divergence: 10%（KL散度，已取负值，越大越好）
        """
        def score(island):
            best = island.get_best_candidate()
            if not best:
                return 0
            f = best.fitness
            return (
                f.get("coverage", 0) * 0.4 +
                f.get("convex_hull", 0) * 0.15 +
                f.get("avg_dist", 0) * 0.15 +
                f.get("min_dist", 0) * 0.1 +
                f.get("dispersion", 0) * 0.1 +
                (1 - f.get("kl_divergence", 1)) * 0.1
            )
        return max(self.islands, key=score)

    def _trigger_extinction(self, reason: str = "", only_stagnated: list = None):
        """触发灭绝.
        
        策略：
        - 固定间隔：重置后50%的岛屿（保留一半多样性）
        - 停滞检测：只重置真正停滞的岛屿
        
        Args:
            reason: 触发原因
            only_stagnated: 仅重置指定的停滞岛屿 [(id, gens_since), ...]
        """
        logger.section(f"🔥 周期性灭绝触发! ({reason})")
        self._extinction_log.append(self.generation)

        best_island = self._get_best_island()
        best_seed = best_island.get_best_candidate()

        if best_seed is None:
            logger.warn("无最优种子可复制，跳过灭绝")
            return

        logger.info(f"最优岛屿: Island {best_island.id} (综合评分)")
        logger.info("最优适应度:")
        for k, v in best_seed.fitness.items():
            logger.metric(k, v)

        reset_count = 0
        
        if only_stagnated:
            # 停滞检测：只重置停滞的岛屿
            for island in self.islands:
                if island.id == best_island.id:
                    continue
                stagnated_ids = [sid for sid, _ in only_stagnated]
                if island.id not in stagnated_ids:
                    continue
                gens_since = next(gs for sid, gs in only_stagnated if sid == island.id)
                logger.warn(f"[Island {island.id}] 已 {gens_since} 轮无改进，重置...")
                island.reset(best_seed, current_gen=self.generation)
                reset_count += 1
        else:
            # 固定间隔：重置后50%的岛屿（保留最优+随机保留一半）
            non_best_islands = [i for i in self.islands if i.id != best_island.id]
            
            # 按适应度排序，重置较差的50%
            non_best_islands.sort(key=lambda i: (
                i.elites.get("coverage", Candidate("")).fitness.get("coverage", 0)
            ))
            
            # 重置后50%
            num_to_reset = max(1, len(non_best_islands) // 2)
            islands_to_reset = non_best_islands[:num_to_reset]
            islands_to_keep = non_best_islands[num_to_reset:]
            
            logger.info(f"岛屿状态:")
            logger.info(f"  最优保留: Island {best_island.id}")
            logger.info(f"  保留: {[i.id for i in islands_to_keep]}")
            logger.info(f"  重置: {[i.id for i in islands_to_reset]}")
            
            for island in islands_to_reset:
                logger.warn(f"[Island {island.id}] 固定间隔重置...")
                island.reset(best_seed, current_gen=self.generation)
                reset_count += 1

        logger.success(f"重置了 {reset_count} 个岛屿，保留了 {self.num_islands - reset_count} 个岛屿")

    def get_global_best(self) -> Optional[Candidate]:
        """获取全局最优候选（多指标综合评分）.
        
        用于：
        1. 记录历史最优（供可视化使用）
        2. 最终评估时选择最优代码
        
        评分权重（与 _get_best_island 一致）：
        - coverage: 40%（覆盖度，核心指标）
        - convex_hull: 15%（凸包体积，空间覆盖）
        - avg_dist: 15%（平均距离，分散度）
        - min_dist: 10%（最小距离，避免聚集）
        - dispersion: 10%（离散度）
        - kl_divergence: 10%（KL散度，越接近均匀分布越好）
        """
        all_elites = []
        for island in self.islands:
            all_elites.extend(island.get_all_elites())

        if not all_elites:
            return None

        def score(candidate):
            f = candidate.fitness
            return (
                f.get("coverage", 0) * 0.30 +
                f.get("convex_hull", 0) * 0.15 +
                f.get("avg_dist", 0) * 0.20 +
                f.get("min_dist", 0) * 0.15 +
                f.get("dispersion", 0) * 0.10 +
                f.get("kl_divergence", 0) * 0.10
            )

        return max(all_elites, key=score)

    def _print_extinction_logic(self, max_generations: int = None):
        """打印灭绝逻辑配置."""
        logger.section("🧬 灭绝逻辑配置")
        
        # 根据总轮数动态调整
        if max_generations:
            # 自适应阈值
            adaptive_interval = max(2, max_generations // 2)
            adaptive_stagnation = max(2, max_generations // 3)
            
            logger.info(f"实验总轮数: {max_generations}")
            logger.info(f"灭绝模式: {self.extinction_mode}")
            logger.info("")
            logger.info("【自适应参数】")
            logger.info(f"  固定间隔灭绝: 每 {adaptive_interval} 轮")
            logger.info(f"  停滞检测阈值: {adaptive_stagnation} 轮无改进")
            logger.info("")
            logger.info("【原始配置】")
            logger.info(f"  extinction_interval: {self.extinction_interval}")
            logger.info(f"  stagnation_threshold: {self.extinction_stagnation_threshold}")
            logger.info("")
            
            # 实际生效的参数
            self._effective_interval = min(self.extinction_interval, adaptive_interval)
            self._effective_stagnation = min(self.extinction_stagnation_threshold, adaptive_stagnation)
            
            logger.info("【实际生效参数】")
            logger.info(f"  有效间隔: {self._effective_interval} 轮")
            logger.info(f"  有效停滞阈值: {self._effective_stagnation} 轮")
            logger.info("")
            
            # 预测触发轮数
            triggers = []
            for g in range(1, max_generations + 1):
                if g % self._effective_interval == 0:
                    triggers.append(f"Gen {g}(间隔)")
            logger.info(f"预计触发轮数: {', '.join(triggers) if triggers else '无'}")
            
        else:
            logger.info("未设置最大轮数，使用原始配置:")
            logger.info(f"  固定间隔: {self.extinction_interval} 轮")
            logger.info(f"  停滞阈值: {self.extinction_stagnation_threshold} 轮")
            self._effective_interval = self.extinction_interval
            self._effective_stagnation = self.extinction_stagnation_threshold
        
        logger.info("")
        logger.info("【灭绝策略】")
        logger.info("  1. 固定间隔: 每N轮重置所有非最优岛屿")
        logger.info("  2. 停滞检测: 超过阈值无改进的岛屿被重置")
        logger.info("  3. 最优保留: 最优岛屿不会被重置")
        logger.info("  4. 多指标选优: coverage(40%) + convex_hull(15%) + avg_dist(15%) + min_dist(10%) + dispersion(10%) + (1-KL)(10%)")

    def run(self, max_generations: int = None, max_hours: float = None):
        """主进化循环."""
        self.max_generations = max_generations
        logger.section("Open-Evolve 进化引擎启动")
        logger.info(f"配置: 岛屿={self.num_islands}, 最大轮数={max_generations}")
        logger.info(f"人格数: {self.evaluator.num_personas} 人/问卷 | 问卷数: {len(self.questionnaires)} 份")
        logger.info(f"每轮评估: {self.num_islands} 岛屿 × {len(self.questionnaires)} 问卷 × {self.evaluator.num_personas} 人格 = {self.num_islands * len(self.questionnaires) * self.evaluator.num_personas} 次 API 调用")
        
        # 打印灭绝逻辑
        self._print_extinction_logic(max_generations)

        while True:
            if max_generations and self.generation >= max_generations:
                logger.success(f"达到最大轮数 {max_generations}，停止进化")
                break

            elapsed_hours = (time.time() - self.start_time) / 3600
            if max_hours and elapsed_hours >= max_hours:
                logger.success(f"达到最大时间 {max_hours}h，停止进化")
                break

            self.evolve_once()
            self._save_checkpoint()

        best = self.get_global_best()
        if best:
            logger.section("进化完成!")
            logger.success(f"最优代码: 岛屿 {best.island_id}, 种子 {best.seed_name}, 代数 {best.generation}")
            logger.info("最终适应度:")
            for k, v in best.fitness.items():
                logger.metric(k, v)

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
        logger.debug(f"Checkpoint 已保存: {path}")
        
        # 更新后台状态文件
        self._update_background_status()

    def _update_background_status(self):
        """更新后台状态文件（供 run_background.py 轮询使用）."""
        try:
            from src.utils.output_manager import output_manager
            if output_manager.base_dir is None:
                return
            
            status_file = output_manager.base_dir / ".background_status"
            
            best = self.get_global_best()
            status = {
                "current_generation": self.generation,
                "total_generations": getattr(self, 'max_generations', None),
                "current_eval": sum(island.total_evaluations for island in self.islands),
                "best_fitness": best.fitness if best else {},
                "last_update": datetime.now().isoformat(),
            }
            
            # 读取现有状态保留 start_time
            if status_file.exists():
                old = json.loads(status_file.read_text())
                status["start_time"] = old.get("start_time", datetime.now().isoformat())
                status["pid"] = old.get("pid")
                status["status"] = "running" if self.generation < status.get("total_generations", float('inf')) else "completed"
            else:
                status["start_time"] = datetime.now().isoformat()
                status["status"] = "running"
            
            status_file.write_text(json.dumps(status, indent=2))
        except Exception:
            pass  # 静默失败，不影响主流程
