"""OpenEvolve island engine used by MegaPersona."""

import copy
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple
from pathlib import Path
import json

from src.utils.config import get_config
from src.utils.logger import logger


def _candidate_global_score(candidate: Optional["Candidate"]) -> float:
    if candidate is None:
        return 0.0
    fitness = candidate.fitness or {}
    return float(
        fitness.get(
            "global_best",
            fitness.get("mega_fitness", fitness.get("coverage", 0.0)),
        )
    )


DEFAULT_PARENT_OBJECTIVE_ROLES = [
    "global_best",
    "coverage_elite",
    "diversity_elite",
    "strict_consistency_elite",
    "shadow_mae_elite",
]


class MutatorProtocol(Protocol):
    def mutate(
        self,
        parent_code: str,
        prompt: str | None = None,
        generation: int = 0,
        stagnation: int = 0,
        operator_id: str | None = None,
    ) -> str:
        ...

    def get_state(self) -> dict[str, Any]:
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        ...


class EvaluatorProtocol(Protocol):
    def evaluate(self, code_str: str) -> dict[str, float]:
        ...

    def evaluate_with_context(
        self,
        code_str: str,
        *,
        parent_id: str | None = None,
    ) -> dict[str, float]:
        ...

    def candidate_id_for_code(self, code_str: str) -> str | None:
        ...

    def confirm_evaluation(
        self,
        code_str: str,
        *,
        parent_id: str | None = None,
        required_repeats: int | None = None,
    ) -> dict[str, float]:
        ...


@dataclass
class Candidate:
    """进化候选（人格生成器代码）."""

    code: str
    fitness: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    island_id: int = 0
    seed_name: str = ""
    candidate_id: str | None = None
    parent_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "fitness": self.fitness,
            "generation": self.generation,
            "island_id": self.island_id,
            "seed_name": self.seed_name,
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
        }


class Island:
    """单个进化岛屿.

    维护多个精英位（对应不同优化目标）。
    若候选在任一指标上打破历史纪录，即成为该指标的新精英。
    """

    METRIC_NAMES = [
        "global_best",
        "research_score_v2",
        "coverage_elite",
        "alignment_elite",
        "shadow_mae_elite",
        "consistency_elite",
        "axis_target_elite",
        "issue_rate_elite",
        "strict_consistency_elite",
        "diversity_elite",
        "schema_elite",
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
        improved_metrics = self.improvement_metrics(candidate)
        for metric in improved_metrics:
            self.elites[metric] = copy.deepcopy(candidate)

        if improved_metrics:
            self.last_improvement_gen = self.generation
            self.total_improvements += len(improved_metrics)

        self.total_evaluations += 1
        return len(improved_metrics) > 0, improved_metrics

    def improvement_metrics(self, candidate: Candidate) -> List[str]:
        """Return elite slots this candidate would improve without mutating state."""
        improved_metrics = []
        for metric in self.METRIC_NAMES:
            current_best = self.elites.get(metric)
            if current_best is None or candidate.fitness.get(
                metric, float("-inf")
            ) > current_best.fitness.get(metric, float("-inf")):
                improved_metrics.append(metric)
        return improved_metrics

    def get_best_candidate(self) -> Optional[Candidate]:
        """获取该岛屿当前综合最优候选."""
        return self.elites.get("global_best") or next(iter(self.elites.values()), None)

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
            "best_global": best.fitness.get("global_best", 0) if best else 0,
        }

    def reset(self, best_elites: Dict[str, Candidate], current_gen: int = 0):
        """重置岛屿：保留部分原有精英，部分替换为最优岛屿精英.

        策略（对比择优）：
        对于每个指标，保留被重置岛屿和最优岛屿中该指标更优的精英。
        这样保证每个指标都不退化，同时保留被重置岛屿的局部优势。

        Args:
            best_elites: 最优岛屿的 elites 字典 {metric: Candidate}
            current_gen: 当前轮数
        """
        # 按指标逐个对比，保留更优者
        for metric in self.METRIC_NAMES:
            old_elite = self.elites.get(metric)
            new_elite = best_elites.get(metric)

            if old_elite is None:
                # 原岛屿没有该指标精英，直接复制
                if new_elite is not None:
                    self.elites[metric] = copy.deepcopy(new_elite)
            elif new_elite is None:
                # 最优岛屿没有该指标精英，保留原有
                pass
            else:
                # 对比该指标上的适应度，保留更优者
                old_val = old_elite.fitness.get(metric, float('-inf'))
                new_val = new_elite.fitness.get(metric, float('-inf'))
                if new_val > old_val:
                    self.elites[metric] = copy.deepcopy(new_elite)
                # 否则保留原有（不做任何操作）

        # 保留当前 generation 计数，不重置为0
        self.last_improvement_gen = current_gen


class OpenEvolve:
    """Open-Evolve 进化引擎."""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        mutator: MutatorProtocol,
        evaluator: EvaluatorProtocol,
        questionnaires: List,
    ) -> "OpenEvolve":
        """从 checkpoint 恢复进化状态.
        
        Args:
            checkpoint_path: checkpoint JSON 文件路径
            mutator: 变异算子（需重新传入）
            evaluator: 评估器（需重新传入）
            questionnaires: 问卷列表（需重新传入）
            
        Returns:
            OpenEvolve: 恢复状态的进化引擎实例
        """
        import json
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
        
        logger.section(f"从 Checkpoint 恢复: {checkpoint_path}")
        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        
        # 创建新实例（不调用 __init__ 的初始化逻辑）
        instance = cls.__new__(cls)
        instance.mutator = mutator
        instance.evaluator = evaluator
        instance.questionnaires = questionnaires
        instance.parent_selection = "operator_preferred"
        instance.parent_objective_roles = list(DEFAULT_PARENT_OBJECTIVE_ROLES)
        instance._parent_rotation_cursor = 0
        
        # 恢复变异算子状态（如果 checkpoint 中有）
        if "mutator_state" in checkpoint and hasattr(instance.mutator, 'set_state'):
            instance.mutator.set_state(checkpoint["mutator_state"])
            logger.info("变异算子状态已从 checkpoint 恢复")
        elif "mutator_state" in checkpoint:
            logger.warn("Mutator 不支持 set_state()，跳过变异状态恢复")
        
        # 恢复灭绝机制状态（优先从 checkpoint 恢复，确保行为一致）
        if "extinction_state" in checkpoint:
            extinction_state = checkpoint["extinction_state"]
            instance.extinction_interval = extinction_state["extinction_interval"]
            instance.extinction_hours = extinction_state["extinction_hours"]
            instance.extinction_stagnation_threshold = extinction_state["extinction_stagnation_threshold"]
            instance.extinction_mode = extinction_state["extinction_mode"]
            instance._effective_interval = extinction_state["_effective_interval"]
            instance._effective_stagnation = extinction_state["_effective_stagnation"]
            instance._last_extinction_gen = extinction_state["_last_extinction_gen"]
            logger.info("灭绝机制状态已从 checkpoint 恢复")
        else:
            # 旧版 checkpoint：从配置文件读取（行为可能不一致）
            cfg = get_config()
            instance.extinction_interval = cfg.get("open_evolve.extinction_interval", 100)
            instance.extinction_hours = cfg.get("open_evolve.extinction_interval_hours", 8)
            instance.extinction_stagnation_threshold = max(2, instance.extinction_interval // 10)
            instance.extinction_mode = cfg.get("open_evolve.extinction_mode", "adaptive")
            logger.warn("Checkpoint 中无灭绝机制状态，使用配置文件默认值")
        
        # 恢复运行配置（优先从 checkpoint 恢复）
        if "run_config" in checkpoint:
            run_config = checkpoint["run_config"]
            instance.num_islands = run_config["num_islands"]
            instance.max_generations = run_config.get("max_generations")
            instance.children_per_island = run_config.get("children_per_island", 3)
            instance.max_workers = max(1, int(run_config.get("max_workers", 1)))
            instance.mutation_max_workers = max(1, int(run_config.get("mutation_max_workers", 6)))
            instance.parent_selection = run_config.get("parent_selection", "operator_preferred")
            instance.parent_objective_roles = list(
                run_config.get("parent_objective_roles", DEFAULT_PARENT_OBJECTIVE_ROLES)
            )
            instance._parent_rotation_cursor = int(run_config.get("parent_rotation_cursor", 0))
            logger.info(
                f"运行配置已恢复: {instance.num_islands} 岛屿, "
                f"children={instance.children_per_island}, "
                f"mutation_workers={instance.mutation_max_workers}, "
                f"max_workers={instance.max_workers}"
            )
        else:
            cfg = get_config()
            instance.num_islands = cfg.get("open_evolve.num_islands", 10)
            instance.max_generations = None
            instance.children_per_island = 3
            instance.max_workers = max(1, int(cfg.get("open_evolve.max_workers", 1)))
            instance.mutation_max_workers = 6
        
        # 恢复基本状态
        instance.generation = checkpoint["generation"]
        instance.history = checkpoint.get("history", [])
        instance._extinction_log = checkpoint.get("extinction_generations", [])
        instance.start_time = time.time()  # 重置计时
        
        # checkpoint 保存路径
        from src.utils.output_manager import output_manager
        if output_manager.base_dir is None:
            output_manager.setup("default")
        instance.checkpoint_path = output_manager.outputs_dir
        
        # 恢复岛屿状态
        instance.islands = []
        if "islands" in checkpoint:
            islands_state = checkpoint["islands"]
            elite_codes_dir = path.parent / islands_state.get("_elite_codes_dir", "")
            
            for island_id in range(instance.num_islands):
                island = Island(island_id)
                island_data = islands_state.get(str(island_id), {})
                
                # 恢复岛屿统计
                island.generation = island_data.get("generation", instance.generation)
                island.last_improvement_gen = island_data.get("last_improvement_gen", 0)
                island.total_evaluations = island_data.get("total_evaluations", 0)
                island.total_improvements = island_data.get("total_improvements", 0)
                
                # 恢复 elites
                elites_meta = island_data.get("elites", {})
                for metric, meta in elites_meta.items():
                    code_file = elite_codes_dir / meta["code_file"] if "code_file" in meta else None
                    if code_file and code_file.exists():
                        code = code_file.read_text(encoding="utf-8")
                    else:
                        # 回退：尝试从 code_file 路径直接读取
                        alt_path = path.parent / meta.get("code_file", "")
                        if alt_path.exists():
                            code = alt_path.read_text(encoding="utf-8")
                        else:
                            logger.warn(f"[恢复] Island {island_id} {metric} 的代码文件不存在，跳过")
                            continue
                    
                    candidate = Candidate(
                        code=code,
                        fitness=meta["fitness"],
                        generation=meta["generation"],
                        island_id=meta["island_id"],
                        seed_name=meta["seed_name"],
                        candidate_id=meta.get("candidate_id"),
                        parent_id=meta.get("parent_id"),
                    )
                    island.elites[metric] = candidate
                
                instance.islands.append(island)
            
            logger.info(f"已恢复 {len(instance.islands)} 个岛屿，共 {sum(len(i.elites) for i in instance.islands)} 个 elite")
        else:
            # 旧版 checkpoint 没有 islands 字段，无法恢复详细状态
            logger.warn("Checkpoint 中没有岛屿详细状态，无法断点续跑")
            # 创建空岛屿
            instance.islands = [Island(i) for i in range(instance.num_islands)]
        
        # 恢复 seed_baselines
        if "seed_baselines" in checkpoint:
            instance.seed_baselines = {}
            for seed_name, fitness in checkpoint["seed_baselines"].items():
                instance.seed_baselines[seed_name] = [fitness]  # 包装成列表保持兼容
        
        logger.success(f"Checkpoint 恢复完成: Gen {instance.generation}")
        return instance

    def __init__(
        self,
        mutator: MutatorProtocol,
        evaluator: EvaluatorProtocol,
        questionnaires: List,
        seed_codes: Optional[Dict[str, str]] = None,
        initial_seed_distribution: Optional[Dict[str, int]] = None,
        num_islands: Optional[int] = None,
        max_workers: Optional[int] = None,
        checkpoint_path: Optional[str | Path] = None,
        initialize: bool = True,
    ):
        self.mutator = mutator
        self.evaluator = evaluator
        self.questionnaires = questionnaires
        self.seed_codes = seed_codes or {}

        cfg = get_config()
        self.num_islands = num_islands or cfg.get("open_evolve.num_islands", 10)
        self.max_workers = max(1, int(max_workers or cfg.get("open_evolve.max_workers", 1)))
        self.mutation_max_workers = 6
        self.extinction_interval = cfg.get("open_evolve.extinction_interval", 100)
        self.extinction_hours = cfg.get("open_evolve.extinction_interval_hours", 8)
        # Kept for checkpoint compatibility; stagnation-triggered extinction is disabled.
        self.extinction_stagnation_threshold = max(2, self.extinction_interval // 10)
        # 灭绝模式保留配置字段；当前只启用固定间隔和时间触发。
        self.extinction_mode = cfg.get("open_evolve.extinction_mode", "adaptive")

        distribution = initial_seed_distribution or cfg.get("open_evolve.initial_seed_distribution", {
            "seed": self.num_islands,
        })

        self.islands = [Island(i) for i in range(self.num_islands)]
        self.seed_baselines: Dict[str, List[Dict]] = {}

        # Parent-selection strategy. "operator_preferred" keeps the historical
        # behavior (operator preferred_parent_metric -> uniform elite fallback).
        # "objective_rotation" round-robins each child's parent across the
        # objective elite roles below so global/coverage/diversity/strict/
        # shadow-MAE elites all participate in mutation.
        self.parent_selection = "operator_preferred"
        self.parent_objective_roles = list(DEFAULT_PARENT_OBJECTIVE_ROLES)
        self._parent_rotation_cursor = 0

        self.generation = 0
        self.start_time = time.time()
        self.history: List[dict] = []
        self._extinction_log: List[int] = []  # 记录灭绝发生的轮数
        
        # checkpoint 保存到统一输出目录
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        else:
            from src.utils.output_manager import output_manager
            if output_manager.base_dir is None:
                output_manager.setup("default")
            self.checkpoint_path = output_manager.outputs_dir

        if initialize:
            self._initialize_islands(distribution)

    def _initialize_islands(self, distribution: Dict[str, int]):
        """初始种子轮询分配."""
        if not self.seed_codes:
            raise ValueError("OpenEvolve requires at least one seed code")

        seed_list = []
        for seed_name, count in distribution.items():
            fallback_code = next(iter(self.seed_codes.values()))
            code = self.seed_codes.get(seed_name, fallback_code)
            for _ in range(count):
                seed_list.append((seed_name, code))

        logger.info(f"初始化 {self.num_islands} 个岛屿...")
        
        for i, island in enumerate(self.islands):
            seed_name, code = seed_list[i % len(seed_list)]
            logger.info(f"  [Island {i}] 评估初始种子 {seed_name}...")
            fitness = self._evaluate_code(code)
            candidate = Candidate(
                code=code,
                fitness=fitness,
                generation=0,
                island_id=i,
                seed_name=seed_name,
                candidate_id=self._candidate_id_for_code(code),
            )
            candidate = self._confirm_potential_elite(candidate, force=True)
            island.update_elite(candidate)
            # Baselines must use the confirmed aggregate that entered the
            # island elite, not the initial single-draw estimate.
            fitness = candidate.fitness
            
            # 记录baseline
            if seed_name not in self.seed_baselines:
                self.seed_baselines[seed_name] = []
            self.seed_baselines[seed_name].append(fitness)
            
            logger.success(
                f"  [Island {i}] 初始化完成 | "
                f"Global: {fitness.get('global_best', 0):.3f} | "
                f"Coverage: {fitness.get('coverage_elite', 0):.3f} | "
                f"Alignment: {fitness.get('alignment_elite', 0):.3f}"
            )
        
        # 打印seed baseline汇总
        logger.section("Seed Baseline 汇总")
        for seed_name, fitness_list in self.seed_baselines.items():
            avg_fitness = {}
            for key in fitness_list[0].keys():
                avg_fitness[key] = sum(f[key] for f in fitness_list) / len(fitness_list)
            logger.success(f"{seed_name} (平均 {len(fitness_list)} 个岛屿):")
            for k, v in avg_fitness.items():
                logger.metric(k, v)

    def _mutate_single(
        self,
        parent_code: str,
        island_id: int,
        child_idx: int,
        children_per_island: int,
        operator_id: Optional[str] = None,
    ) -> Optional[str]:
        """变异单个候选解（用于线程池并行）."""
        max_retries = 3
        
        # 计算动态温度参数
        island = self.islands[island_id] if island_id < len(self.islands) else None
        stagnation = 0
        if island:
            stagnation = self.generation - island.last_improvement_gen
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"[Island {island_id}] 候选 {child_idx+1}/{children_per_island} 变异... (attempt {attempt + 1}/{max_retries})")
                try:
                    child_code = self.mutator.mutate(
                        parent_code,
                        generation=self.generation,
                        stagnation=stagnation,
                        operator_id=operator_id,
                    )
                except TypeError:
                    child_code = self.mutator.mutate(
                        parent_code,
                        generation=self.generation,
                        stagnation=stagnation,
                    )
                return child_code
            except Exception as e:
                logger.warn(f"[Island {island_id}] 候选 {child_idx+1} 变异失败 (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error(f"[Island {island_id}] 候选 {child_idx+1} 变异失败 {max_retries} 次，跳过")
        return None

    def _evaluate_code(
        self,
        code_str: str,
        *,
        parent_id: str | None = None,
    ) -> dict[str, float]:
        contextual_evaluate = getattr(self.evaluator, "evaluate_with_context", None)
        if callable(contextual_evaluate):
            return contextual_evaluate(code_str, parent_id=parent_id)
        return self.evaluator.evaluate(code_str)

    def _candidate_id_for_code(self, code_str: str) -> str | None:
        resolver = getattr(self.evaluator, "candidate_id_for_code", None)
        if not callable(resolver):
            return None
        return resolver(code_str)

    def _evaluate_single(
        self,
        child_code: str,
        island_id: int,
        child_idx: int,
        total_children: int,
        seed_name: str,
        parent_id: str | None = None,
    ) -> Optional[Candidate]:
        """评估单个候选解（用于线程池并行）."""
        try:
            logger.debug(f"[Island {island_id}] 候选 {child_idx+1}/{total_children} 评估中...")
            child_fitness = self._evaluate_code(child_code, parent_id=parent_id)
            return Candidate(
                code=child_code,
                fitness=child_fitness,
                generation=self.generation,
                island_id=island_id,
                seed_name=seed_name,
                candidate_id=self._candidate_id_for_code(child_code),
                parent_id=parent_id,
            )
        except Exception as e:
            logger.error(f"[Island {island_id}] 候选 {child_idx+1} 评估失败: {e}")
            return None

    def evolve_once(self) -> dict:
        """Run one evolution generation with global candidate evaluation parallelism."""
        self.generation += 1
        gen_start = time.time()
        
        # 每岛候选数（大群体进化参数）
        children_per_island = getattr(self, 'children_per_island', 3)
        self.children_per_island = children_per_island
        max_workers = max(1, int(getattr(self, "max_workers", 1)))
        mutation_max_workers = max(1, int(getattr(self, "mutation_max_workers", 6)))

        logger.section(f"Generation {self.generation}/{self.max_generations if hasattr(self, 'max_generations') else '?'}")
        logger.info(f"大群体进化: {len(self.islands)} 岛屿 × {children_per_island} 候选 = {len(self.islands) * children_per_island} 评估/轮")
        logger.info(
            f"并行策略: mutation最多 {mutation_max_workers} 个线程，"
            f"全局最多 {max_workers} 个候选同时评估"
        )

        round_stats = {
            "generation": self.generation,
            "improvements": 0,
            "evaluations": 0,
            "time": time.time(),
        }

        mutation_jobs = []

        # Parent/operator selection remains serial so stateful policies such as
        # MCTS keep deterministic ordering. The expensive LLM mutator calls are
        # then dispatched concurrently.
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
            
            # 2. 准备变异任务
            for child_idx in range(children_per_island):
                operator_id = self._choose_operator_id(island.id, child_idx)
                parent = self._select_parent_for_operator(island, elites, operator_id)
                mutation_jobs.append(
                    {
                        "island": island,
                        "parent_code": parent.code,
                        "child_idx": child_idx,
                        "total_children": children_per_island,
                        "seed_name": parent.seed_name,
                        "operator_id": operator_id,
                        "parent_fitness": dict(parent.fitness),
                        "parent_id": parent.candidate_id or self._candidate_id_for_code(parent.code),
                    }
                )

        evaluation_jobs = []
        if not mutation_jobs:
            logger.warn("本轮无可变异候选，跳过评估")
        else:
            logger.info(
                f"并行变异 {len(mutation_jobs)} 个候选解，"
                f"max_workers={min(mutation_max_workers, len(mutation_jobs))}"
            )
            mutation_start = time.time()

            if mutation_max_workers <= 1:
                for job_index, job in enumerate(mutation_jobs):
                    child_code = self._mutate_single(
                        job["parent_code"],
                        job["island"].id,
                        job["child_idx"],
                        job["total_children"],
                        operator_id=job.get("operator_id"),
                    )
                    if child_code is not None:
                        job = dict(job)
                        job["child_code"] = child_code
                        evaluation_jobs.append(job)
                    logger.progress(job_index + 1, len(mutation_jobs), "candidate mutation")
            else:
                mutated_jobs: list[Optional[dict]] = [None] * len(mutation_jobs)
                with ThreadPoolExecutor(max_workers=min(mutation_max_workers, len(mutation_jobs))) as executor:
                    futures = {
                        executor.submit(
                            self._mutate_single,
                            job["parent_code"],
                            job["island"].id,
                            job["child_idx"],
                            job["total_children"],
                            operator_id=job.get("operator_id"),
                        ): (job_index, job)
                        for job_index, job in enumerate(mutation_jobs)
                    }

                    completed = 0
                    for future in as_completed(futures):
                        job_index, job = futures[future]
                        try:
                            child_code = future.result()
                            if child_code is not None:
                                job = dict(job)
                                job["child_code"] = child_code
                                mutated_jobs[job_index] = job
                        except Exception as e:
                            logger.error(
                                f"[Island {job['island'].id}] 候选 {job['child_idx']+1} "
                                f"变异线程异常: {e}"
                            )
                        completed += 1
                        logger.progress(completed, len(mutation_jobs), "candidate mutation")

                evaluation_jobs = [job for job in mutated_jobs if job is not None]

            logger.info(f"并行变异完成: {time.time() - mutation_start:.1f}s")

        if not evaluation_jobs:
            logger.warn("本轮无成功变异候选，跳过评估")
        else:
            logger.info(
                f"全局并行评估 {len(evaluation_jobs)} 个候选解，"
                f"max_workers={min(max_workers, len(evaluation_jobs))}"
            )
            eval_start = time.time()

            if max_workers <= 1:
                for job_index, job in enumerate(evaluation_jobs):
                    self._record_evaluated_job(job, job_index, len(evaluation_jobs), round_stats)
            else:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(evaluation_jobs))) as executor:
                    futures = {
                        executor.submit(
                            self._evaluate_single,
                            job["child_code"],
                            job["island"].id,
                            job["child_idx"],
                            children_per_island,
                            job["seed_name"],
                            job.get("parent_id"),
                        ): (job_index, job)
                        for job_index, job in enumerate(evaluation_jobs)
                    }

                    completed = 0
                    for future in as_completed(futures):
                        job_index, job = futures[future]
                        island = job["island"]
                        child_idx = job["child_idx"]
                        try:
                            child = future.result()
                            self._update_island_with_child(
                                island,
                                child_idx,
                                child,
                                round_stats,
                                operator_id=job.get("operator_id"),
                                parent_fitness=job.get("parent_fitness"),
                            )
                        except Exception as e:
                            logger.error(f"[Island {island.id}] 候选 {child_idx+1} 评估线程异常: {e}")
                        completed += 1
                        logger.progress(completed, len(evaluation_jobs), "candidate eval")

            logger.info(f"全局并行评估完成: {time.time() - eval_start:.1f}s")

        # 检查灭绝
        self._check_extinction()

        # 记录历史
        best = self.get_global_best()
        if best:
            round_stats["best_fitness"] = best.fitness
            logger.success(f"本轮最优 (Island {best.island_id}):")
            self._log_fitness_with_baseline(best)
        
        # 记录每轮统计：平均、中位数、最优、最差（用于可视化）
        all_fitness_values = {}
        for metric in Island.METRIC_NAMES:
            values = []
            for island in self.islands:
                elite = island.elites.get(metric)
                if elite:
                    values.append(elite.fitness.get(metric, 0))
            if values:
                import numpy as np
                all_fitness_values[metric] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "max": float(np.max(values)),
                    "min": float(np.min(values)),
                    "std": float(np.std(values)),
                }
        round_stats["fitness_stats"] = all_fitness_values

        # 岛屿统计
        logger.info("岛屿状态:")
        for island in self.islands:
            stats = island.get_stats()
            logger.info(f"  Island {stats['id']}: evals={stats['evaluations']}, "
                       f"improvements={stats['improvements']}, "
                       f"last_improve={stats['last_improvement']}, "
                       f"global={stats['best_global']:.3f}")

        gen_time = time.time() - gen_start
        logger.info(f"本轮耗时: {gen_time:.1f}s | 评估: {round_stats['evaluations']} | 改进: {round_stats['improvements']}")

        self.history.append(round_stats)
        return round_stats

    def _record_evaluated_job(self, job: dict, job_index: int, total_jobs: int, round_stats: dict) -> None:
        island = job["island"]
        child_idx = job["child_idx"]
        child = self._evaluate_single(
            job["child_code"],
            island.id,
            child_idx,
            job.get("total_children", getattr(self, "children_per_island", 1)),
            job["seed_name"],
            job.get("parent_id"),
        )
        self._update_island_with_child(
            island,
            child_idx,
            child,
            round_stats,
            operator_id=job.get("operator_id"),
            parent_fitness=job.get("parent_fitness"),
        )
        logger.progress(job_index + 1, total_jobs, "candidate eval")

    def _choose_operator_id(self, island_id: int, child_idx: int) -> Optional[str]:
        chooser = getattr(self.mutator, "choose_operator", None)
        if chooser is None:
            return None
        island = self.islands[island_id] if island_id < len(self.islands) else None
        stagnation = self.generation - island.last_improvement_gen if island else 0
        try:
            operator = chooser(
                generation=self.generation,
                stagnation=stagnation,
                island_id=island_id,
                child_idx=child_idx,
            )
        except TypeError:
            operator = chooser(self.generation, stagnation)
        if isinstance(operator, dict) and isinstance(operator.get("id"), str):
            return operator["id"]
        if isinstance(operator, str):
            return operator
        return None

    def _select_parent_for_operator(
        self,
        island: Island,
        elites: List[Candidate],
        operator_id: Optional[str],
    ) -> Candidate:
        if self._global_stagnation() >= 4:
            plateau_parent = self._select_plateau_parent(island)
            if plateau_parent is not None:
                return plateau_parent

        if self.parent_selection == "objective_rotation":
            roles = self.parent_objective_roles or ["global_best"]
            role = roles[self._parent_rotation_cursor % len(roles)]
            self._parent_rotation_cursor += 1
            parent = island.elites.get(role) or island.elites.get("global_best")
            if parent is not None:
                logger.debug(
                    f"[Island {island.id}] objective_rotation parent role={role}"
                )
                return parent

        metric_getter = getattr(self.mutator, "preferred_parent_metric", None)
        metric = None
        if metric_getter is not None:
            try:
                metric = metric_getter(operator_id)
            except Exception:
                metric = None
        if isinstance(metric, str):
            candidate = island.elites.get(metric)
            if candidate is not None:
                return candidate
        return random.choice(elites)

    def _global_stagnation(self) -> int:
        best = self.get_global_best()
        if best is None:
            return 0
        return max(0, int(self.generation) - int(best.generation))

    def _select_plateau_parent(self, island: Island) -> Optional[Candidate]:
        """During global plateaus, force search away from only global_best.

        Coverage/diversity elites are tried first so strict-consistency search
        does not keep narrowing the behavior space after the global score stalls.
        """
        priority_metrics = [
            "coverage_elite",
            "diversity_elite",
            "strict_consistency_elite",
            "shadow_mae_elite",
            "axis_target_elite",
            "issue_rate_elite",
            "research_score_v2",
        ]
        candidates: list[Candidate] = []
        seen_codes: set[str] = set()
        for metric in priority_metrics:
            candidate = island.elites.get(metric)
            if candidate is None or candidate.code in seen_codes:
                continue
            seen_codes.add(candidate.code)
            candidates.append(candidate)
        if not candidates:
            return None
        return random.choice(candidates)

    def _update_island_with_child(
        self,
        island: Island,
        child_idx: int,
        child: Optional[Candidate],
        round_stats: dict,
        operator_id: Optional[str] = None,
        parent_fitness: Optional[dict[str, float]] = None,
    ) -> None:
        if child is None:
            return

        round_stats["evaluations"] += 1
        potential_metrics = island.improvement_metrics(child)
        if potential_metrics:
            child = self._confirm_potential_elite(child)
        improved, metrics = island.update_elite(child)
        self._record_mutation_result(
            operator_id=operator_id,
            parent_fitness=parent_fitness,
            child=child,
            improved=improved,
            improved_metrics=metrics,
            child_idx=child_idx,
        )
        if improved:
            round_stats["improvements"] += 1
            logger.success(f"[Island {island.id}] 候选 {child_idx+1} 🏆 打破 {len(metrics)} 项纪录: {metrics}")
            for metric in metrics:
                self._log_metric_with_baseline(metric, child.fitness.get(metric, 0), child)
        else:
            logger.debug(f"[Island {island.id}] 候选 {child_idx+1} 未打破纪录")

    def _confirm_potential_elite(
        self,
        candidate: Candidate,
        *,
        force: bool = False,
    ) -> Candidate:
        confirmer = getattr(self.evaluator, "confirm_evaluation", None)
        required = max(
            1,
            int(getattr(self.evaluator, "elite_confirmation_repeats", 1)),
        )
        base = max(
            1,
            int(getattr(self.evaluator, "candidate_evaluation_repeats", 1)),
        )
        if not callable(confirmer) or required <= base:
            return candidate
        logger.info(
            f"[Island {candidate.island_id}] candidate={candidate.candidate_id or '?'} "
            f"触发精英确认: {base} -> {required} 次"
        )
        try:
            confirmed_fitness = confirmer(
                candidate.code,
                parent_id=candidate.parent_id,
                required_repeats=required,
            )
        except Exception as exc:
            if force:
                raise
            logger.warn(
                f"[Island {candidate.island_id}] 精英确认失败，保留单次结果: {exc}"
            )
            return candidate
        return Candidate(
            code=candidate.code,
            fitness=confirmed_fitness,
            generation=candidate.generation,
            island_id=candidate.island_id,
            seed_name=candidate.seed_name,
            candidate_id=candidate.candidate_id,
            parent_id=candidate.parent_id,
        )

    def _record_mutation_result(
        self,
        *,
        operator_id: Optional[str],
        parent_fitness: Optional[dict[str, float]],
        child: Candidate,
        improved: bool,
        improved_metrics: list[str],
        child_idx: int,
    ) -> None:
        recorder = getattr(self.mutator, "record_result", None)
        if recorder is None:
            return
        if bool((child.fitness or {}).get("phenotype_cache_hit")):
            logger.debug(
                f"[Island {child.island_id}] 候选 {child_idx+1} 为 phenotype cache hit，"
                "跳过 MCTS 结果回传"
            )
            return
        try:
            recorder(
                operator_id=operator_id,
                parent_fitness=parent_fitness,
                child_fitness=child.fitness,
                generation=self.generation,
                island_id=child.island_id,
                child_idx=child_idx,
                improved=improved,
                improved_metrics=improved_metrics,
            )
        except Exception as exc:
            logger.warn(f"变异结果回传失败，跳过策略更新: {exc}")

    def _baseline_for_candidate(self, candidate: Optional[Candidate]) -> dict[str, float]:
        """Return the averaged seed baseline for a candidate."""
        if candidate is None:
            return {}
        if candidate.seed_name in self.seed_baselines:
            return self._average_fitness_list(self.seed_baselines[candidate.seed_name])
        all_baselines = [
            fitness
            for fitness_list in self.seed_baselines.values()
            for fitness in fitness_list
            if isinstance(fitness, dict)
        ]
        return self._average_fitness_list(all_baselines)

    @staticmethod
    def _average_fitness_list(fitness_list: list[dict]) -> dict[str, float]:
        if not fitness_list:
            return {}
        keys = set()
        for fitness in fitness_list:
            keys.update(fitness.keys())
        averaged = {}
        for key in keys:
            values = [
                float(fitness.get(key, 0.0))
                for fitness in fitness_list
                if isinstance(fitness, dict)
            ]
            averaged[key] = sum(values) / len(values) if values else 0.0
        return averaged

    def _log_metric_with_baseline(
        self,
        metric: str,
        current: float,
        candidate: Optional[Candidate],
    ) -> None:
        baseline = self._baseline_for_candidate(candidate).get(metric)
        if baseline is None:
            logger.metric(metric, current)
            return
        logger.metric_delta(metric, float(baseline), float(current))

    def _log_fitness_with_baseline(self, candidate: Candidate) -> None:
        for metric, current in candidate.fitness.items():
            self._log_metric_with_baseline(metric, current, candidate)

    def _check_extinction(self):
        """检查是否触发周期性灭绝.
        
        两种触发方式：
        1. 固定间隔触发（每 _effective_interval 轮）
        2. 时间触发（长时间运行）
        
        优先级：固定间隔 > 时间
        避免连续两轮触发（保护期1轮）
        """
        # 使用实际生效的参数
        effective_interval = getattr(self, '_effective_interval', self.extinction_interval)
        
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

    def _get_best_island(self) -> Island:
        """选择综合表现最强的岛屿，用于灭绝时复制 elite archive."""
        def score(island):
            return _candidate_global_score(island.get_best_candidate())
        return max(self.islands, key=score)

    def _trigger_extinction(self, reason: str = "", only_stagnated: list = None):
        """触发灭绝.
        
        策略：
        - 固定间隔：重置后50%的岛屿（保留一半多样性）
        
        Args:
            reason: 触发原因
            only_stagnated: legacy argument kept for checkpoint/code compatibility.
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
        self._log_fitness_with_baseline(best_seed)
        
        # 获取最优岛屿的完整 elites（用于混合替换）
        best_elites = best_island.elites

        reset_count = 0
        
        # 固定间隔/时间触发：重置后50%的岛屿（保留最优+随机保留一半）
        non_best_islands = [i for i in self.islands if i.id != best_island.id]
        
        # 按综合适应度排序，重置较差的50%
        non_best_islands.sort(key=lambda i: _candidate_global_score(i.get_best_candidate()))
        
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
            island.reset(best_elites, current_gen=self.generation)
            reset_count += 1

        logger.success(f"重置了 {reset_count} 个岛屿，保留了 {self.num_islands - reset_count} 个岛屿")

    def get_global_best(self) -> Optional[Candidate]:
        """获取全局综合最优候选.
        
        用于：
        1. 记录历史最优（供可视化使用）
        2. 最终评估时选择最优代码
        """
        all_elites = []
        for island in self.islands:
            all_elites.extend(island.get_all_elites())

        if not all_elites:
            return None

        return max(all_elites, key=_candidate_global_score)

    def _print_extinction_logic(self, max_generations: int = None):
        """打印灭绝逻辑配置."""
        logger.section("🧬 灭绝逻辑配置")
        
        # 根据总轮数动态调整
        if max_generations is not None:
            # 自适应阈值
            adaptive_interval = max(2, max_generations // 2)
            logger.info(f"实验总轮数: {max_generations}")
            logger.info(f"灭绝模式: {self.extinction_mode}")
            logger.info("")
            logger.info("【自适应参数】")
            logger.info(f"  固定间隔灭绝: 每 {adaptive_interval} 轮")
            logger.info("")
            logger.info("【原始配置】")
            logger.info(f"  extinction_interval: {self.extinction_interval}")
            logger.info("")
            
            # 实际生效的参数
            self._effective_interval = min(self.extinction_interval, adaptive_interval)
            
            logger.info("【实际生效参数】")
            logger.info(f"  有效间隔: {self._effective_interval} 轮")
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
            self._effective_interval = self.extinction_interval
        
        logger.info("")
        logger.info("【灭绝策略】")
        logger.info("  1. 固定间隔: 每N轮重置所有非最优岛屿")
        logger.info("  2. 最优保留: 最优岛屿不会被重置")
        logger.info(
            "  3. Elite archive: global_best / research_score_v2 / coverage_elite / "
            "shadow_mae_elite / strict_consistency_elite / diversity_elite / schema_elite"
        )

    def run(
        self,
        max_generations: int = None,
        max_hours: float = None,
        children_per_island: int = 3,
        max_workers: int | None = None,
    ):
        """主进化循环.
        
        Args:
            max_generations: 最大进化轮数
            max_hours: 最大运行时间（小时）
            children_per_island: 每轮每岛产生的候选解数量（大群体进化参数，默认3）
        """
        self.max_generations = max_generations
        self.children_per_island = children_per_island
        if max_workers is not None:
            self.max_workers = max(1, int(max_workers))
        
        logger.section("Open-Evolve 进化引擎启动")
        logger.info(
            f"配置: 岛屿={self.num_islands}, 最大轮数={max_generations}, "
            f"mutation并发={getattr(self, 'mutation_max_workers', 6)}, "
            f"candidate并发={self.max_workers}"
        )
        logger.info(f"大群体进化: 每岛 {children_per_island} 候选 × {self.num_islands} 岛屿 = {children_per_island * self.num_islands} 评估/轮")
        num_personas = getattr(self.evaluator, "num_personas", "?")
        evaluation_repeats = max(
            1,
            int(getattr(self.evaluator, "candidate_evaluation_repeats", 1)),
        )
        elite_confirmation_repeats = max(
            evaluation_repeats,
            int(getattr(self.evaluator, "elite_confirmation_repeats", evaluation_repeats)),
        )
        logger.info(f"人格数: {num_personas} 人/问卷 | 问卷数: {len(self.questionnaires)} 份")
        logger.info(
            f"候选评估: 基础={evaluation_repeats} 次，"
            f"潜在精英确认={elite_confirmation_repeats} 次"
        )
        
        # 计算 API 调用量
        if isinstance(num_personas, int):
            evals_per_round = (
                self.num_islands
                * children_per_island
                * len(self.questionnaires)
                * num_personas
                * evaluation_repeats
            )
            logger.info(
                f"每轮 API 调用: {self.num_islands} 岛 × {children_per_island} 候选 "
                f"× {len(self.questionnaires)} 问卷 × {num_personas} 人格 "
                f"× {evaluation_repeats} 次评估 = {evals_per_round} 次"
            )
            if elite_confirmation_repeats > evaluation_repeats:
                logger.info(
                    "潜在精英会按需追加评估；实际 API 调用量取决于触发确认的候选数"
                )
        
        # 打印灭绝逻辑
        self._print_extinction_logic(max_generations)

        while True:
            if max_generations is not None and self.generation >= max_generations:
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
            self._log_fitness_with_baseline(best)

        return best

    def _save_checkpoint(self):
        """保存进化状态.
        
        Checkpoint 结构:
          - generation: 当前轮数
          - history: 历史统计
          - best: 全局最优（代码内联）
          - seed_baselines: 初始种子基线
          - islands: 各岛屿完整状态（elites 元数据，代码引用）
          - islands_data: 各岛屿当前适应度（简化版，用于可视化）
          - extinction_generations: 灭绝日志
          - elite_codes_dir: elites 代码文件存放目录
        """
        checkpoint = {
            "generation": self.generation,
            "history": self.history,
            "best": self.get_global_best().to_dict() if self.get_global_best() else None,
        }
        
        # 保存灭绝机制运行时状态（确保恢复后行为一致）
        checkpoint["extinction_state"] = {
            "extinction_interval": self.extinction_interval,
            "extinction_hours": self.extinction_hours,
            "extinction_stagnation_threshold": self.extinction_stagnation_threshold,
            "extinction_mode": self.extinction_mode,
            "_effective_interval": getattr(self, '_effective_interval', self.extinction_interval),
            "_effective_stagnation": getattr(self, '_effective_stagnation', self.extinction_stagnation_threshold),
            "_last_extinction_gen": getattr(self, '_last_extinction_gen', 0),
        }
        
        # 保存seed baseline（初始化完成后才有）
        if hasattr(self, "seed_baselines"):
            checkpoint["seed_baselines"] = {}
            for seed_name, fitness_list in self.seed_baselines.items():
                avg_fitness = {}
                for key in fitness_list[0].keys():
                    avg_fitness[key] = sum(f[key] for f in fitness_list) / len(fitness_list)
                checkpoint["seed_baselines"][seed_name] = avg_fitness
        
        # 保存所有岛屿的当前 elites 数据（用于热力图重绘）
        islands_data = {}
        for island in self.islands:
            data = {}
            for metric, candidate in island.elites.items():
                data[metric] = candidate.fitness.get(metric, 0)
            islands_data[island.id] = data
        checkpoint["islands_data"] = islands_data
        
        # 保存灭绝日志
        if hasattr(self, '_extinction_log') and self._extinction_log:
            checkpoint["extinction_generations"] = self._extinction_log
        
        # 保存运行参数（确保恢复后配置一致）
        checkpoint["run_config"] = {
            "num_islands": self.num_islands,
            "max_generations": getattr(self, 'max_generations', None),
            "children_per_island": getattr(self, 'children_per_island', 3),
            "max_workers": getattr(self, "max_workers", 1),
            "mutation_max_workers": getattr(self, "mutation_max_workers", 6),
            "parent_selection": getattr(self, "parent_selection", "operator_preferred"),
            "parent_objective_roles": list(
                getattr(self, "parent_objective_roles", DEFAULT_PARENT_OBJECTIVE_ROLES)
            ),
            "parent_rotation_cursor": int(getattr(self, "_parent_rotation_cursor", 0)),
        }
        
        # 保存变异算子状态
        if hasattr(self, 'mutator') and self.mutator is not None:
            checkpoint["mutator_state"] = self.mutator.get_state()
        
        # 保存岛屿详细状态（elites 元数据，代码存到单独文件）
        checkpoint["islands"] = self._serialize_islands()
        
        path = self.checkpoint_path / f"checkpoint_gen_{self.generation}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        latest_path = self.checkpoint_path / "checkpoint.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.debug(f"Checkpoint 已保存: {path}")
        
        # 更新后台状态文件
        self._update_background_status()
    
    def _serialize_islands(self) -> dict:
        """序列化所有岛屿状态.
        
        elites 的代码保存到单独文件，checkpoint 中只存引用路径。
        这样 checkpoint 文件不会过大，同时支持断点续跑。
        """
        # 创建 elites 代码目录
        elite_codes_dir = self.checkpoint_path / f"elite_codes_gen_{self.generation}"
        elite_codes_dir.mkdir(exist_ok=True)
        
        islands_state = {}
        for island in self.islands:
            elites_meta = {}
            for metric, candidate in island.elites.items():
                # 保存代码到单独文件
                code_filename = f"island_{island.id}_{metric}_gen{candidate.generation}.py"
                code_path = elite_codes_dir / code_filename
                code_path.write_text(candidate.code, encoding="utf-8")
                
                # checkpoint 中只存元数据
                elites_meta[metric] = {
                    "fitness": candidate.fitness,
                    "generation": candidate.generation,
                    "island_id": candidate.island_id,
                    "seed_name": candidate.seed_name,
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "code_file": str(code_path.relative_to(self.checkpoint_path)),
                }
            
            islands_state[str(island.id)] = {
                "elites": elites_meta,
                "generation": island.generation,
                "last_improvement_gen": island.last_improvement_gen,
                "total_evaluations": island.total_evaluations,
                "total_improvements": island.total_improvements,
            }
        
        islands_state["_elite_codes_dir"] = str(elite_codes_dir.relative_to(self.checkpoint_path))
        return islands_state

    def _update_background_status(self):
        """Best-effort status file update for legacy output-manager runs."""
        try:
            from datetime import datetime
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
