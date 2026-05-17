"""端到端测试：人格生成 → 模拟回答 → 多样性评估."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.utils.llm_client import LLMClient
from src.persona_generator.generator import PersonaGenerator
from src.simulator.concordia_sim import ConcordiaSimulator
from src.evaluator.metrics import DiversityMetrics
from src.qgenerator.generator import QGenerator, Questionnaire


def load_questionnaire(path: str) -> Questionnaire:
    """加载单份问卷."""
    questionnaires = QGenerator.load(path)
    return questionnaires[0]


def main():
    print("=" * 70)
    print("端到端测试：人格生成 → 模拟回答 → 多样性评估")
    print("=" * 70)

    # 加载 LLM
    print("\n[1/4] 初始化 LLM...")
    llm = LLMClient.from_config("llm.qgenerator_model")

    # 加载问卷
    print("\n[2/4] 加载问卷...")
    q = load_questionnaire("data/questionnaires/test_5.json")
    print(f"  问卷: {q.brief}")
    print(f"  维度: {q.dimensions}")
    print(f"  题项: {len(q.items)}")

    # 生成人格（用 seed1，n=5 快速测试）
    print("\n[3/4] 生成人格（seed1, n=5）...")
    persona_gen = PersonaGenerator(llm, seed="seed1")
    personas = persona_gen.generate(
        context=q.context,
        dimensions=q.dimensions,
        n=5,
    )
    print(f"  生成 {len(personas)} 个人格")
    for i, p in enumerate(personas):
        print(f"    {i+1}. {p.high_level_tag[:50]}...")

    # 模拟回答
    print("\n[4/4] 模拟回答 & 评估...")
    simulator = ConcordiaSimulator(llm)

    # 把 Persona 对象转为 dict
    persona_dicts = [{"description": p.description} for p in personas]
    q_dict = q.to_dict()

    Z = simulator.simulate(persona_dicts, q_dict)

    # 评估
    print("\n  多样性指标:")
    metrics = DiversityMetrics(coverage_radius=0.2)
    fitness = metrics.fitness(Z)
    for name, value in fitness.items():
        print(f"    {name:20s}: {value:+.6f}")

    # 输出 Z 矩阵
    print(f"\n  Z 矩阵 (人格 × 维度):")
    print(f"    {Z}")

    print("\n" + "=" * 70)
    print("端到端测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
