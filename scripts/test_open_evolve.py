"""测试 Open-Evolve 进化引擎（Mock 评估器）."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

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
            "global_best": np.random.uniform(0.3, 1.0),
            "research_score_v2": np.random.uniform(0.3, 1.0),
            "coverage_elite": np.random.uniform(0.0, 1.0),
            "alignment_elite": np.random.uniform(0.2, 0.8),
            "shadow_mae_elite": np.random.uniform(0.2, 0.8),
            "consistency_elite": np.random.uniform(0.0, 1.0),
            "axis_target_elite": np.random.uniform(0.0, 1.0),
            "issue_rate_elite": np.random.uniform(0.0, 1.0),
            "strict_consistency_elite": np.random.uniform(0.0, 1.0),
            "diversity_elite": np.random.uniform(0.1, 0.5),
            "schema_elite": np.random.uniform(0.5, 1.0),
        }


class RecordingMutator(MockMutator):
    """Mock mutator that records MCTS-style result callbacks."""

    def __init__(self):
        super().__init__()
        self.records = []

    def record_result(self, **kwargs):
        self.records.append(kwargs)


class LineageEvaluator:
    """Evaluator that exposes candidate IDs and records parent context."""

    def __init__(self):
        self.ids = {}
        self.records = []

    def evaluate(self, code_str: str):
        return self.evaluate_with_context(code_str)

    def evaluate_with_context(self, code_str: str, *, parent_id=None):
        candidate_id = self.ids.setdefault(code_str, f"lineage_{len(self.ids) + 1:03d}")
        self.records.append((candidate_id, parent_id))
        score = 0.5 + 0.01 * len(self.records)
        return {
            "global_best": score,
            "research_score_v2": score,
            "coverage_elite": score,
            "alignment_elite": score,
            "shadow_mae_elite": score,
            "consistency_elite": score,
            "axis_target_elite": score,
            "issue_rate_elite": score,
            "strict_consistency_elite": score,
            "diversity_elite": score,
            "schema_elite": score,
        }

    def candidate_id_for_code(self, code_str: str):
        return self.ids.get(code_str)


class EliteConfirmingEvaluator:
    candidate_evaluation_repeats = 1
    elite_confirmation_repeats = 3

    def __init__(self, confirmed_score=0.4):
        self.confirmed_score = confirmed_score
        self.confirm_calls = 0

    def confirm_evaluation(self, code_str, *, parent_id=None, required_repeats=None):
        self.confirm_calls += 1
        return {metric: self.confirmed_score for metric in Island.METRIC_NAMES}


def test_island():
    print("=" * 60)
    print("测试岛屿系统")
    print("=" * 60)

    island = Island(0)

    # 添加初始精英
    c1 = Candidate(code="code1", fitness={
        "global_best": 0.5, "research_score_v2": 0.45, "coverage_elite": 0.3,
        "alignment_elite": 0.4, "shadow_mae_elite": 0.4, "consistency_elite": 0.1,
        "axis_target_elite": 0.3, "issue_rate_elite": 0.8, "strict_consistency_elite": 0.5,
        "diversity_elite": 0.2, "schema_elite": 0.6,
    }, generation=0)
    improved, metrics = island.update_elite(c1)
    print(f"  初始精英: improved={improved}, metrics={metrics}")

    # 添加更优候选
    c2 = Candidate(code="code2", fitness={
        "global_best": 0.8, "research_score_v2": 0.45, "coverage_elite": 0.3,
        "alignment_elite": 0.4, "shadow_mae_elite": 0.4, "consistency_elite": 0.1,
        "axis_target_elite": 0.3, "issue_rate_elite": 0.8, "strict_consistency_elite": 0.5,
        "diversity_elite": 0.2, "schema_elite": 0.6,
    }, generation=1)
    improved, metrics = island.update_elite(c2)
    print(f"  更优候选: improved={improved}, metrics={metrics}")

    # 添加部分改善候选
    c3 = Candidate(code="code3", fitness={
        "global_best": 0.6, "research_score_v2": 0.46, "coverage_elite": 0.5,
        "alignment_elite": 0.4, "shadow_mae_elite": 0.45, "consistency_elite": 0.1,
        "axis_target_elite": 0.35, "issue_rate_elite": 0.8, "strict_consistency_elite": 0.55,
        "diversity_elite": 0.2, "schema_elite": 0.6,
    }, generation=2)
    improved, metrics = island.update_elite(c3)
    print(f"  部分改善: improved={improved}, metrics={metrics}")

    print(f"  当前精英数: {len(island.elites)}")
    assert len(island.elites) == len(Island.METRIC_NAMES), "应有完整精英槽位"

    # 检查综合与覆盖精英可由不同 candidate 占据
    assert island.elites["global_best"].code == "code2", "global_best 精英应为 code2"
    assert island.elites["coverage_elite"].code == "code3", "coverage_elite 精英应为 code3"
    assert island.elites["strict_consistency_elite"].code == "code3", "strict_consistency_elite 精英应为 code3"

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


def test_objective_rotation_parent_selection():
    print("=" * 60)
    print("测试 objective_rotation 父代选择")
    print("=" * 60)

    engine = OpenEvolve(
        mutator=MockMutator(),
        evaluator=MockEvaluator(),
        questionnaires=[],
        seed_codes=SEED_CODES,
        num_islands=1,
        initialize=False,
    )
    island = engine.islands[0]
    base = {
        "global_best": 0.5, "research_score_v2": 0.5, "coverage_elite": 0.3,
        "alignment_elite": 0.4, "shadow_mae_elite": 0.4, "consistency_elite": 0.5,
        "axis_target_elite": 0.4, "issue_rate_elite": 0.5,
        "strict_consistency_elite": 0.4, "diversity_elite": 0.3, "schema_elite": 0.6,
    }
    c_global = Candidate(code="c_global", fitness={**base, "global_best": 0.9}, generation=0)
    c_cov = Candidate(code="c_cov", fitness={**base, "coverage_elite": 0.9}, generation=0)
    island.update_elite(c_global)
    island.update_elite(c_cov)
    elites = island.get_all_elites()

    assert engine.parent_selection == "operator_preferred"  # default unchanged
    default_parent = engine._select_parent_for_operator(island, elites, None)
    assert default_parent.code in {"c_global", "c_cov"}
    assert engine._parent_rotation_cursor == 0  # rotation never engages by default

    engine.parent_selection = "objective_rotation"
    picked = [engine._select_parent_for_operator(island, elites, None) for _ in range(5)]
    assert picked[0].code == "c_global"  # role: global_best
    assert picked[1].code == "c_cov"     # role: coverage_elite
    # remaining roles fall back to the elites holding those slots (c_global)
    assert all(parent.code == "c_global" for parent in picked[2:])
    # cursor wraps around to global_best
    assert engine._select_parent_for_operator(island, elites, None).code == "c_global"

    with TemporaryDirectory() as tmp:
        engine.checkpoint_path = Path(tmp)
        engine._save_checkpoint()
        restored = OpenEvolve.from_checkpoint(
            str(Path(tmp) / "checkpoint.json"),
            mutator=MockMutator(),
            evaluator=MockEvaluator(),
            questionnaires=[],
        )
        assert restored.parent_selection == "objective_rotation"
        assert restored.parent_objective_roles == engine.parent_objective_roles
        assert restored._parent_rotation_cursor == engine._parent_rotation_cursor
        restored_island = restored.islands[0]
        restored_elites = restored_island.get_all_elites()
        # Cursor was advanced 6 times above, so the next role is coverage_elite.
        assert restored._select_parent_for_operator(restored_island, restored_elites, None).code == "c_cov"

    # plateau branch still takes precedence and does not consume the rotation cursor
    engine.generation = 10  # best generation is 0 -> stagnation 10 >= 4
    cursor_before = engine._parent_rotation_cursor
    plateau_parent = engine._select_parent_for_operator(island, elites, None)
    assert plateau_parent.code in {"c_global", "c_cov"}
    assert engine._parent_rotation_cursor == cursor_before

    print("✅ objective_rotation 父代选择测试通过")


def test_phenotype_cache_hit_skips_mcts_record():
    print("=" * 60)
    print("测试 phenotype cache hit 不回传 MCTS")
    print("=" * 60)

    mutator = RecordingMutator()
    engine = OpenEvolve(
        mutator=mutator,
        evaluator=MockEvaluator(),
        questionnaires=[],
        seed_codes=SEED_CODES,
        num_islands=1,
        initialize=False,
    )

    cached_child = Candidate(
        code="cached",
        fitness={
            "global_best": 0.7,
            "coverage_elite": 0.4,
            "phenotype_cache_hit": True,
        },
        generation=1,
        island_id=0,
    )
    engine._record_mutation_result(
        operator_id="op21_v3_schema_precision",
        parent_fitness={"global_best": 0.6},
        child=cached_child,
        improved=True,
        improved_metrics=["global_best"],
        child_idx=0,
    )
    assert mutator.records == []

    fresh_child = Candidate(
        code="fresh",
        fitness={"global_best": 0.8, "coverage_elite": 0.4},
        generation=1,
        island_id=0,
    )
    engine._record_mutation_result(
        operator_id="op21_v3_schema_precision",
        parent_fitness={"global_best": 0.6},
        child=fresh_child,
        improved=True,
        improved_metrics=["global_best"],
        child_idx=1,
    )
    assert len(mutator.records) == 1
    assert mutator.records[0]["operator_id"] == "op21_v3_schema_precision"

    print("✅ phenotype cache hit 跳过 MCTS 回传测试通过")


def test_potential_elite_is_confirmed_before_selection_and_mcts():
    evaluator = EliteConfirmingEvaluator(confirmed_score=0.4)
    mutator = RecordingMutator()
    engine = OpenEvolve(
        mutator=mutator,
        evaluator=evaluator,
        questionnaires=[],
        seed_codes=SEED_CODES,
        num_islands=1,
        initialize=False,
    )
    island = engine.islands[0]
    incumbent = Candidate(
        code="incumbent",
        fitness={metric: 0.5 for metric in Island.METRIC_NAMES},
        island_id=0,
    )
    island.update_elite(incumbent)
    lucky_single_draw = Candidate(
        code="challenger",
        fitness={metric: 0.9 for metric in Island.METRIC_NAMES},
        generation=1,
        island_id=0,
        candidate_id="challenger-id",
    )
    round_stats = {"evaluations": 0, "improvements": 0}
    engine._update_island_with_child(
        island,
        0,
        lucky_single_draw,
        round_stats,
        operator_id="op-test",
        parent_fitness=incumbent.fitness,
    )
    assert evaluator.confirm_calls == 1
    assert island.elites["global_best"].code == "incumbent"
    assert round_stats == {"evaluations": 1, "improvements": 0}
    assert mutator.records[0]["child_fitness"]["global_best"] == 0.4
    assert mutator.records[0]["improved"] is False
    print("✅ potential elite 确认后再选择并回传 MCTS")


def test_candidate_lineage_round_trip():
    print("=" * 60)
    print("测试 candidate / parent 谱系透传与 checkpoint 恢复")
    print("=" * 60)

    with TemporaryDirectory() as tmp:
        evaluator = LineageEvaluator()
        engine = OpenEvolve(
            mutator=MockMutator(),
            evaluator=evaluator,
            questionnaires=[],
            seed_codes={"seed1": "seed-code-1"},
            initial_seed_distribution={"seed1": 1},
            num_islands=1,
            max_workers=1,
            checkpoint_path=tmp,
        )
        engine.mutation_max_workers = 1
        engine.extinction_interval = 100
        parent = engine.islands[0].get_best_candidate()
        assert parent is not None
        assert parent.candidate_id == "lineage_001"
        assert parent.parent_id is None

        engine.children_per_island = 1
        engine.evolve_once()
        child = engine.get_global_best()
        assert child is not None
        assert child.candidate_id == "lineage_002"
        assert child.parent_id == parent.candidate_id
        assert evaluator.records[-1] == (child.candidate_id, parent.candidate_id)

        engine._save_checkpoint()
        restored = OpenEvolve.from_checkpoint(
            str(Path(tmp) / "checkpoint.json"),
            mutator=MockMutator(),
            evaluator=evaluator,
            questionnaires=[],
        )
        restored_child = restored.get_global_best()
        assert restored_child is not None
        assert restored_child.candidate_id == child.candidate_id
        assert restored_child.parent_id == parent.candidate_id

    print("✅ candidate / parent 谱系测试通过")


def main():
    test_island()
    test_engine()
    test_extinction()
    test_objective_rotation_parent_selection()
    test_phenotype_cache_hit_skips_mcts_record()
    test_potential_elite_is_confirmed_before_selection_and_mcts()
    test_candidate_lineage_round_trip()

    print("\n" + "=" * 60)
    print("所有 Open-Evolve 测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
