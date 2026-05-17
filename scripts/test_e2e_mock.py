"""端到端测试（Mock 数据）— 验证 pipeline 逻辑，不调用 API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.simulator.concordia_sim import ConcordiaSimulator
from src.evaluator.metrics import DiversityMetrics


class MockLLM:
    """Mock LLM，用于模拟器测试."""
    def generate(self, prompt, **kwargs):
        # 从 prompt 中解析选项数量，返回随机选择
        import re
        match = re.search(r'0-(\d+)', prompt)
        if match:
            max_choice = int(match.group(1))
            return str(np.random.randint(0, max_choice + 1))
        return "2"  # 默认中间值


def main():
    print("=" * 70)
    print("端到端测试（Mock 数据）：模拟回答 → 多样性评估")
    print("=" * 70)

    # 构造模拟数据
    np.random.seed(42)

    # 5 个人格（文本描述）
    personas = [
        {"description": f"Persona {i+1}: 高适应性、中等风险承受的人格..."}
        for i in range(5)
    ]

    # 1 份问卷（2 维，每维 3 题）
    questionnaire = {
        "context": "AGI 导致失业的社会情境",
        "dimensions": ["adaptability", "risk_tolerance"],
        "items": [
            {"statement": "我能快速适应新环境", "choices": ["SD", "D", "N", "A", "SA"], "dimension": "adaptability"},
            {"statement": "我喜欢尝试新事物", "choices": ["SD", "D", "N", "A", "SA"], "dimension": "adaptability"},
            {"statement": "变化让我感到兴奋", "choices": ["SD", "D", "N", "A", "SA"], "dimension": "adaptability"},
            {"statement": "我愿意冒险追求高回报", "choices": ["SD", "D", "N", "A", "SA"], "dimension": "risk_tolerance"},
            {"statement": "我喜欢刺激的体验", "choices": ["SD", "D", "N", "A", "SA"], "dimension": "risk_tolerance"},
            {"statement": "我倾向于保守决策", "choices": ["SD", "D", "N", "A", "SA"], "dimension": "risk_tolerance"},
        ],
    }

    # 模拟回答
    print("\n[1/2] 模拟回答...")
    llm = MockLLM()
    simulator = ConcordiaSimulator(llm)
    Z = simulator.simulate(personas, questionnaire)

    print(f"\n  Z 矩阵 (人格 × 维度):\n{Z}")

    # 评估
    print("\n[2/2] 多样性评估...")
    metrics = DiversityMetrics(coverage_radius=0.3)
    fitness = metrics.fitness(Z)

    print(f"\n  适应度分数（越大越好）:")
    for name, value in fitness.items():
        print(f"    {name:20s}: {value:+.6f}")

    # 验证数据结构
    assert Z.shape == (5, 2), f"Z 形状应为 (5, 2)，实际 {Z.shape}"
    assert all(0 <= z <= 1 for z in Z.flatten()), "Z 值应在 [0, 1]"

    print("\n" + "=" * 70)
    print("端到端 Mock 测试通过!")
    print("=" * 70)


if __name__ == "__main__":
    main()
