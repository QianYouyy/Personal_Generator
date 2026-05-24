"""Open-Evolve 评估器 — 执行人格生成器代码并评估其适应度."""

import time
from typing import Dict, List
import numpy as np

from src.qgenerator.generator import Questionnaire
from src.simulator.concordia_sim import ConcordiaSimulator
from src.evaluator.metrics import DiversityMetrics, MultiQuestionnaireEvaluator
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

    def evaluate(self, code_str: str) -> Dict[str, float]:
        """评估代码字符串."""
        start_time = time.time()
        self.total_evals += 1
        eval_id = self.total_evals

        logger.debug(f"[Eval #{eval_id}] 开始评估...")

        try:
            # 在安全命名空间中执行代码
            namespace = {}
            exec(code_str, namespace)

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
                    if len(tags) < self.num_personas:
                        tags = tags + [tags[-1]] * (self.num_personas - len(tags))
                    tags = tags[:self.num_personas]
                    logger.debug(f"[Eval #{eval_id}]   Stage1 完成: {len(tags)} 个标签")

                    descriptions = generator.stage2(q.context, q.dimensions, tags)
                    if len(descriptions) < self.num_personas:
                        descriptions = descriptions + [descriptions[-1]] * (self.num_personas - len(descriptions))
                    descriptions = descriptions[:self.num_personas]
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

    @staticmethod
    def _default_fitness() -> Dict[str, float]:
        """默认适应度."""
        return {
            "coverage": 0.0,
            "convex_hull": 0.0,
            "avg_dist": 0.0,
            "min_dist": 0.0,
            "dispersion": 0.0,
            "kl_divergence": 0.0,
        }

    def get_stats(self) -> dict:
        """获取评估器统计."""
        return {
            "total_evaluations": self.total_evals,
            "total_time": self.total_time,
            "avg_time": self.total_time / self.total_evals if self.total_evals > 0 else 0,
        }
