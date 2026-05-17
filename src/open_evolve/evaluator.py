"""Open-Evolve 评估器 — 执行人格生成器代码并评估其适应度."""

import time
from typing import Dict, List
import numpy as np

from src.qgenerator.generator import Questionnaire
from src.simulator.concordia_sim import ConcordiaSimulator
from src.evaluator.metrics import DiversityMetrics, MultiQuestionnaireEvaluator


class PersonaCodeEvaluator:
    """评估人格生成器代码的适应度.

    流程:
      1. exec() 执行代码字符串，获取人格生成器类
      2. 对每份问卷生成 N 个人格
      3. 模拟回答 → Z 矩阵
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
        self.simulator = ConcordiaSimulator(llm_client)
        self.metrics = DiversityMetrics()

    def evaluate(self, code_str: str) -> Dict[str, float]:
        """评估代码字符串.

        Args:
            code_str: 人格生成器的 Python 代码字符串

        Returns:
            dict: 6 个指标的平均适应度
        """
        start_time = time.time()

        try:
            # 在安全命名空间中执行代码
            namespace = {}
            exec(code_str, namespace)

            # 获取人格生成器类
            if "SeedPersonaGenerator" not in namespace:
                raise ValueError("代码中未定义 SeedPersonaGenerator 类")

            GeneratorClass = namespace["SeedPersonaGenerator"]
            generator = GeneratorClass(self.llm)

            # 检查必要的方法
            if not hasattr(generator, "stage1") or not hasattr(generator, "stage2"):
                raise ValueError("人格生成器缺少 stage1 或 stage2 方法")

            # 对每份问卷评估
            all_fitness = []
            for q in self.questionnaires:
                try:
                    # 生成人格
                    tags = generator.stage1(q.context, q.dimensions, self.num_personas)
                    if len(tags) < self.num_personas:
                        tags = tags + [tags[-1]] * (self.num_personas - len(tags))
                    tags = tags[:self.num_personas]

                    descriptions = generator.stage2(q.context, q.dimensions, tags)
                    if len(descriptions) < self.num_personas:
                        descriptions = descriptions + [descriptions[-1]] * (self.num_personas - len(descriptions))
                    descriptions = descriptions[:self.num_personas]

                    personas = [{"description": desc} for desc in descriptions]

                    # 模拟回答
                    q_dict = q.to_dict()
                    Z = self.simulator.simulate(personas, q_dict)

                    # 评估
                    fitness = self.metrics.fitness(Z)
                    all_fitness.append(fitness)

                except Exception as e:
                    print(f"    [Evaluator] 问卷评估失败: {e}")
                    continue

            if not all_fitness:
                return self._default_fitness()

            # 取平均
            avg_fitness = {}
            for key in all_fitness[0].keys():
                values = [f[key] for f in all_fitness]
                avg_fitness[key] = float(np.mean(values))

            elapsed = time.time() - start_time
            print(f"    [Evaluator] 评估完成 ({elapsed:.1f}s): { {k: f'{v:.3f}' for k, v in avg_fitness.items()} }")
            return avg_fitness

        except Exception as e:
            print(f"    [Evaluator] 评估失败: {e}")
            return self._default_fitness()

    @staticmethod
    def _default_fitness() -> Dict[str, float]:
        """默认适应度（评估失败时返回）."""
        return {
            "coverage": 0.0,
            "convex_hull": 0.0,
            "avg_dist": 0.0,
            "min_dist": 0.0,
            "dispersion": 0.0,
            "kl_divergence": 0.0,
        }
