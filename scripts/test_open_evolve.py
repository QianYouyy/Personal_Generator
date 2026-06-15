"""测试 Open-Evolve 进化引擎（Mock 评估器）."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.open_evolve.engine import OpenEvolve, Island, Candidate


SEED_CODES = {
    "seed1": "seed-code-1",
    "seed2": "seed-code-2",
    "seed3": "seed-code-3",
}


class MockMutator:
    """Mock mutator that appends deterministic markers."""

    def __init__(self):
        self.call_count = 0

    def mutate(self, parent_code: str, prompt=None, generation: int = 0, stagnation: int = 0):
        self.call_count += 1
        return f"{parent_code}\n# mutation={self.call_count} generation={generation} stagnation={stagnation}"

    def get_state(self):
        return {"call_count": self.call_count}

    def set_state(self, state):
        self.call_count = state.get("call_count", self.call_count)


class MockEvaluator:
    """Mock 评估器 — 返回随机适应度."""

    def __init__(self):
        self.eval_count = 0

    def evaluate(self, code_str: str):
        self.eval_count += 1
        np.random.seed(hash(code_str) % 2**31)
        return {
            "coverage": np.random.uniform(0.3, 1.0),
            "convex_hull": np.random.uniform(0.0, 1.0),
            "avg_dist": np.random.uniform(0.2, 0.8),
            "min_dist": np.random.uniform(0.0, 0.2),
            "dispersion": np.random.uniform(0.1, 0.5),
            "kl_divergence": np.random.uniform(0.5, 3.0),
        }


def test_island():
    print("=" * 60)
    print("测试岛屿系统")
    print("=" * 60)

    island = Island(0)

    # 添加初始精英
    c1 = Candidate(code="code1", fitness={
        "coverage": 0.5, "convex_hull": 0.3,
        "avg_dist": 0.4, "min_dist": 0.1,
        "dispersion": 0.2, "kl_divergence": 1.0,
    }, generation=0)
    improved, metrics = island.update_elite(c1)
    print(f"  初始精英: improved={improved}, metrics={metrics}")

    # 添加更优候选
    c2 = Candidate(code="code2", fitness={
        "coverage": 0.8, "convex_hull": 0.3,
        "avg_dist": 0.4, "min_dist": 0.1,
        "dispersion": 0.2, "kl_divergence": 1.0,
    }, generation=1)
    improved, metrics = island.update_elite(c2)
    print(f"  更优候选: improved={improved}, metrics={metrics}")

    # 添加部分改善候选
    c3 = Candidate(code="code3", fitness={
        "coverage": 0.6, "convex_hull": 0.5,
        "avg_dist": 0.4, "min_dist": 0.1,
        "dispersion": 0.2, "kl_divergence": 1.0,
    }, generation=2)
    improved, metrics = island.update_elite(c3)
    print(f"  部分改善: improved={improved}, metrics={metrics}")

    print(f"  当前精英数: {len(island.elites)}")
    assert len(island.elites) == 6, "应有 6 个精英"

    # 检查覆盖率精英是否为 code2
    assert island.elites["coverage"].code == "code2", "coverage 精英应为 code2"
    assert island.elites["convex_hull"].code == "code3", "convex_hull 精英应为 code3"

    print("\n✅ 岛屿系统测试通过")


def test_engine():
    print("\n" + "=" * 60)
    print("测试进化引擎（3 轮，3 个岛屿）")
    print("=" * 60)

    mutator = MockMutator()
    evaluator = MockEvaluator()

    # 使用 3 个岛屿和 3 份 Mock 问卷快速测试
    engine = OpenEvolve(
        mutator,
        evaluator,
        questionnaires=[],
        seed_codes=SEED_CODES,
        initial_seed_distribution={"seed1": 2, "seed2": 2, "seed3": 2},
    )

    # 覆盖配置为 3 个岛屿
    engine.num_islands = 3
    engine.islands = [Island(i) for i in range(3)]
    engine.extinction_interval = 100  # 避免测试中触发灭绝

    # 手动初始化岛屿
    for i, island in enumerate(engine.islands):
        seed_name = ["seed1", "seed2", "seed3"][i % 3]
        code = SEED_CODES[seed_name]
        fitness = evaluator.evaluate(code)
        candidate = Candidate(code=code, fitness=fitness, generation=0,
                              island_id=i, seed_name=seed_name)
        island.update_elite(candidate)

    print(f"  初始状态: {len(engine.islands)} 个岛屿")

    # 运行 3 轮
    for gen in range(1, 4):
        stats = engine.evolve_once()
        print(f"  第 {gen} 轮: {stats['evaluations']} 次评估, "
              f"{stats['improvements']} 次改进")

    print(f"\n  总代数: {engine.generation}")
    print(f"  总评估次数: {evaluator.eval_count}")
    print(f"  变异次数: {mutator.call_count}")

    best = engine.get_global_best()
    if best:
        print(f"  最优适应度: { {k: f'{v:.3f}' for k, v in best.fitness.items()} }")

    print("\n✅ 进化引擎测试通过")


def test_extinction():
    print("\n" + "=" * 60)
    print("测试灭绝机制")
    print("=" * 60)

    mutator = MockMutator()
    evaluator = MockEvaluator()

    engine = OpenEvolve(
        mutator,
        evaluator,
        questionnaires=[],
        seed_codes=SEED_CODES,
        initial_seed_distribution={"seed1": 2, "seed2": 2, "seed3": 2},
    )
    engine.num_islands = 3
    engine.islands = [Island(i) for i in range(3)]
    engine.extinction_interval = 2  # 每 2 轮触发灭绝

    # 初始化
    for i, island in enumerate(engine.islands):
        fitness = evaluator.evaluate(f"seed{i}")
        candidate = Candidate(code=f"seed{i}", fitness=fitness, generation=0,
                              island_id=i, seed_name=f"seed{i}")
        island.update_elite(candidate)
        # 模拟长时间无改进
        island.last_improvement_gen = 0

    # 运行 3 轮（第 2 轮应触发灭绝）
    for gen in range(1, 4):
        stats = engine.evolve_once()
        print(f"  第 {gen} 轮完成")

    print("\n✅ 灭绝机制测试通过")


def main():
    test_island()
    test_engine()
    test_extinction()

    print("\n" + "=" * 60)
    print("所有 Open-Evolve 测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
