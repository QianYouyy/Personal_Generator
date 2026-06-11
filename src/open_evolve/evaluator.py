"""Open-Evolve 评估器 — 执行人格生成器代码并评估其适应度."""

import builtins
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import numpy as np
from scipy.stats import qmc

from src.qgenerator.generator import Questionnaire
from src.simulator.concordia_sim import ConcordiaSimulator
from src.evaluator.metrics import DiversityMetrics
from src.utils.llm_client import LLMClient
from src.utils.config import get_config
from src.utils.logger import logger


class PersonaCodeEvaluator:
    """评估人格生成器代码的适应度.

    流程:
      1. exec() 执行代码字符串，获取人格生成器类
      2. 对每份问卷生成 N 个人格 (使用 persona_model)
      3. 模拟回答 → Z 矩阵 (使用 simulator_model)
      4. 计算 6 个指标
      5. 取平均
    """

    def __init__(
        self,
        questionnaires: List[Questionnaire],
        llm_client,
        num_personas: int = 25,
    ):
        self.questionnaires = questionnaires
        self.llm = llm_client
        self.num_personas = num_personas
        # 人格生成使用 persona_model (gpt-5.4) 以保证质量
        self.persona_llm = LLMClient.from_config("llm.persona_model")
        # 模拟器使用 simulator_model，线程数从配置读取
        cfg = get_config()
        max_workers = cfg.get("simulator.max_workers", 5)
        self.simulator = ConcordiaSimulator(llm_client, max_workers=max_workers)
        self.metrics = DiversityMetrics()
        self.total_evals = 0
        self.total_time = 0.0
        self._last_Z = None  # 保存最后一次评估的 Z 矩阵用于可视化

    def evaluate(self, code_str: str) -> Dict[str, float]:
        """评估代码字符串."""
        start_time = time.time()
        self.total_evals += 1
        eval_id = self.total_evals

        logger.debug(f"[Eval #{eval_id}] 开始评估...")

        try:
            namespace = self._execute_candidate_code(code_str)

            if "SeedPersonaGenerator" not in namespace:
                raise ValueError("代码中未定义 SeedPersonaGenerator 类")

            GeneratorClass = namespace["SeedPersonaGenerator"]
            # 使用 persona_model (gpt-5.4) 进行人格生成，质量更好
            generator = GeneratorClass(self.persona_llm)

            if not hasattr(generator, "stage1") or not hasattr(generator, "stage2"):
                raise ValueError("人格生成器缺少 stage1 或 stage2 方法")

            # 对每份问卷评估
            all_fitness = []
            logger.debug(f"[Eval #{eval_id}] 评估 {len(self.questionnaires)} 份问卷...")

            for q_idx, q in enumerate(self.questionnaires):
                q_start = time.time()
                try:
                    # 生成人格
                    logger.debug(f"[Eval #{eval_id}] 问卷 {q_idx+1}/{len(self.questionnaires)}: {q.brief[:40]}...")

                    tags = generator.stage1(q.context, q.dimensions, self.num_personas)
                    if tags is None:
                        logger.warn(f"[Eval #{eval_id}]   Stage1 返回 None，跳过")
                        continue
                    if not isinstance(tags, list):
                        logger.warn(f"[Eval #{eval_id}]   Stage1 返回类型错误: {type(tags).__name__}，跳过")
                        continue
                    tags = self._normalize_text_list(
                        tags,
                        self.num_personas,
                        lambda i: f"fallback persona niche {i + 1}: balanced but distinct on {', '.join(q.dimensions)}",
                    )
                    if len(tags) == 0:
                        logger.warn(f"[Eval #{eval_id}]   Stage1 返回空列表，跳过")
                        continue
                    logger.debug(f"[Eval #{eval_id}]   Stage1 完成: {len(tags)} 个标签")

                    descriptions = generator.stage2(q.context, q.dimensions, tags)
                    if descriptions is None:
                        logger.warn(f"[Eval #{eval_id}]   Stage2 返回 None，跳过")
                        continue
                    if not isinstance(descriptions, list):
                        logger.warn(f"[Eval #{eval_id}]   Stage2 返回类型错误: {type(descriptions).__name__}，跳过")
                        continue
                    descriptions = self._normalize_text_list(
                        descriptions,
                        self.num_personas,
                        lambda i: (
                            f"Persona {i + 1} is shaped by the requested context and differs from peers "
                            f"across {', '.join(q.dimensions)}. Their choices are internally consistent "
                            f"with the tag: {tags[i] if i < len(tags) else 'balanced'}."
                        ),
                    )
                    if len(descriptions) == 0:
                        logger.warn(f"[Eval #{eval_id}]   Stage2 返回空列表，跳过")
                        continue
                    logger.debug(f"[Eval #{eval_id}]   Stage2 完成: {len(descriptions)} 个人格")

                    personas = [{"description": desc} for desc in descriptions]

                    # 模拟回答
                    q_dict = q.to_dict()
                    Z = self.simulator.simulate(personas, q_dict)
                    logger.debug(f"[Eval #{eval_id}]   模拟完成: Z.shape={Z.shape}")

                    # 评估
                    fitness = self.metrics.fitness(Z)
                    all_fitness.append(fitness)
                    q_time = time.time() - q_start
                    logger.debug(f"[Eval #{eval_id}]   问卷评估完成 ({q_time:.1f}s): "
                                f"Coverage={fitness['coverage']:.3f}, "
                                f"ConvexHull={fitness['convex_hull']:.3f}, "
                                f"AvgDist={fitness['avg_dist']:.3f}, "
                                f"MinDist={fitness['min_dist']:.3f}, "
                                f"Dispersion={fitness['dispersion']:.3f}, "
                                f"KL={fitness['kl_divergence']:.3f}")

                except Exception as e:
                    logger.warn(f"[Eval #{eval_id}] 问卷 {q_idx+1} 评估失败: {e}")
                    continue

            if not all_fitness:
                logger.error(f"[Eval #{eval_id}] 所有问卷评估失败")
                return self._default_fitness()

            # 取平均
            avg_fitness = {}
            for key in all_fitness[0].keys():
                values = [f[key] for f in all_fitness]
                avg_fitness[key] = float(np.mean(values))

            # 保存最后一个 Z 矩阵用于可视化（取第一份问卷的 Z）
            self._last_Z = Z

            elapsed = time.time() - start_time
            self.total_time += elapsed
            logger.debug(f"[Eval #{eval_id}] 评估完成 ({elapsed:.1f}s): "
                        f"Coverage={avg_fitness['coverage']:.3f}, "
                        f"ConvexHull={avg_fitness['convex_hull']:.3f}, "
                        f"AvgDist={avg_fitness['avg_dist']:.3f}, "
                        f"MinDist={avg_fitness['min_dist']:.3f}, "
                        f"Dispersion={avg_fitness['dispersion']:.3f}, "
                        f"KL={avg_fitness['kl_divergence']:.3f}")
            return avg_fitness

        except Exception as e:
            logger.error(f"[Eval #{eval_id}] 评估失败: {e}")
            return self._default_fitness()

    @classmethod
    def _execute_candidate_code(cls, code_str: str) -> dict:
        """用受限命名空间执行候选代码，阻止文件/系统级能力被直接调用."""
        namespace = {
            "__builtins__": cls._safe_builtins(),
            "__name__": "__candidate__",
            "np": np,
            "numpy": np,
            "qmc": qmc,
            "math": math,
            "random": random,
            "re": re,
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "as_completed": as_completed,
        }
        exec(code_str, namespace)
        return namespace

    @classmethod
    def _safe_builtins(cls) -> dict:
        names = [
            "__build_class__",
            "__import__",
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "Exception",
            "float",
            "getattr",
            "hasattr",
            "int",
            "isinstance",
            "len",
            "list",
            "max",
            "min",
            "object",
            "print",
            "range",
            "round",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "type",
            "ValueError",
            "zip",
        ]
        safe = {name: getattr(builtins, name) for name in names}
        safe["__import__"] = cls._safe_import
        return safe

    @staticmethod
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed = {
            "concurrent",
            "concurrent.futures",
            "math",
            "random",
            "re",
            "time",
            "typing",
            "numpy",
            "scipy",
            "scipy.stats",
        }
        if level != 0:
            raise ImportError("relative imports are not allowed in candidate code")
        if name not in allowed:
            raise ImportError(f"import '{name}' is not allowed in candidate code")
        return __import__(name, globals, locals, fromlist, level)

    @staticmethod
    def _normalize_text_list(values, target_len: int, fallback_factory) -> List[str]:
        """过滤 None/Exception/Error 字符串，并补齐到目标长度."""
        cleaned = []
        if isinstance(values, list):
            for value in values:
                if value is None or isinstance(value, BaseException):
                    continue
                text = str(value).strip()
                if not text or text.lower().startswith("error"):
                    continue
                cleaned.append(text)

        for i in range(len(cleaned), target_len):
            cleaned.append(fallback_factory(i))
        return cleaned[:target_len]

    @staticmethod
    def _default_fitness() -> Dict[str, float]:
        """默认适应度."""
        return {
            "coverage": 0.0,
            "convex_hull": 0.0,
            "avg_dist": 0.0,
            "min_dist": 0.0,
            "dispersion": -1e9,
            "kl_divergence": -1e9,
        }

    def get_stats(self) -> dict:
        """获取评估器统计."""
        return {
            "total_evaluations": self.total_evals,
            "total_time": self.total_time,
            "avg_time": self.total_time / self.total_evals if self.total_evals > 0 else 0,
        }
