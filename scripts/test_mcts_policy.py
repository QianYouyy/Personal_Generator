"""Unit tests for the MegaPersona operator MCTS policy reward profiles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.mega_persona.mcts_policy import (
    _STRUCTURED_OPT_WEIGHTS as _BASE_WEIGHTS,
    OperatorMCTSConfig,
    OperatorMCTSPolicy,
)


OPERATORS = [{"id": "op_a"}, {"id": "op_b"}, {"id": "op_c"}]

PARENT = {
    "global_best": 0.25,
    "coverage_elite": 0.40,
    "diversity_elite": 0.30,
    "shadow_mae_elite": 0.50,
    "axis_target_elite": 0.50,
    "strict_consistency_elite": 0.50,
    "research_score_v2": 0.20,
    "alignment_elite": 0.60,
    "consistency_elite": 0.60,
    "schema_elite": 0.40,
    "issue_rate_elite": 0.50,
}


def _structured_policy(**config_overrides) -> OperatorMCTSPolicy:
    config_kwargs = {"reward_profile": "structured", "plateau_stagnation": 4}
    config_kwargs.update(config_overrides)
    return OperatorMCTSPolicy(
        operators=list(OPERATORS),
        rng=np.random.default_rng(7),
        config=OperatorMCTSConfig(**config_kwargs),
    )


def test_structured_reward_failed_child():
    policy = _structured_policy()
    child = dict(PARENT)
    child["global_best"] = 0.0
    reward = policy._reward_structured(
        parent_fitness=PARENT,
        child_fitness=child,
        improved=False,
        improved_metrics=[],
        new_bests={},
        stagnation=0,
    )
    assert reward == -1.0, reward
    print("✅ structured reward hard gate")


def test_structured_reward_guard_tolerance():
    policy = _structured_policy()
    same = policy._reward_structured(
        parent_fitness=PARENT,
        child_fitness=dict(PARENT),
        improved=False,
        improved_metrics=[],
        new_bests={},
        stagnation=0,
    )
    assert same == 0.0, same

    tolerant_child = dict(PARENT)
    tolerant_child["coverage_elite"] = 0.395  # 1.25% drop, below the 2% tolerance
    tolerant = policy._reward_structured(
        parent_fitness=PARENT,
        child_fitness=tolerant_child,
        improved=False,
        improved_metrics=[],
        new_bests={},
        stagnation=0,
    )
    expected_tolerant = 0.05 * (0.395 - 0.40) / 0.40  # optimization term only, no guard
    assert abs(tolerant - expected_tolerant) < 1e-9, (tolerant, expected_tolerant)

    collapsed_child = dict(PARENT)
    collapsed_child["coverage_elite"] = 0.20  # 50% drop
    collapsed = policy._reward_structured(
        parent_fitness=PARENT,
        child_fitness=collapsed_child,
        improved=False,
        improved_metrics=[],
        new_bests={},
        stagnation=0,
    )
    expected_collapsed = 0.05 * (0.20 - 0.40) / 0.40 - 0.5 * (0.5 - 0.02)
    assert abs(collapsed - expected_collapsed) < 1e-9, (collapsed, expected_collapsed)
    assert collapsed < tolerant < same
    print("✅ structured reward bounded coverage guard")


def test_structured_reward_progress_and_plateau():
    policy = _structured_policy()
    policy.metric_best = {"coverage_elite": 0.40}
    policy.metric_stale = {"coverage_elite": 25}  # stale beyond threshold (20)

    child = dict(PARENT)  # identical to parent: optimization term is zero
    new_bests = {"coverage_elite": 0.44}
    calm = policy._reward_structured(
        parent_fitness=PARENT,
        child_fitness=child,
        improved=True,
        improved_metrics=["coverage_elite"],
        new_bests=new_bests,
        stagnation=0,
    )
    progress = 0.5 * (0.44 - 0.40) / 0.40
    expected_calm = progress + 0.02 * 1
    assert abs(calm - expected_calm) < 1e-9, (calm, expected_calm)

    plateau = policy._reward_structured(
        parent_fitness=PARENT,
        child_fitness=child,
        improved=True,
        improved_metrics=["coverage_elite"],
        new_bests=new_bests,
        stagnation=4,
    )
    expected_plateau = expected_calm + progress + 0.05 * 1  # doubled progress + stale bonus
    assert abs(plateau - expected_plateau) < 1e-9, (plateau, expected_plateau)
    assert plateau > calm
    print("✅ structured reward progress bonus and plateau mode")


def test_structured_tracking_and_stagnation():
    policy = _structured_policy()
    base_child = dict(PARENT)

    policy.record_result(
        operator_id="op_a",
        parent_fitness=PARENT,
        child_fitness=base_child,
        generation=0,
        island_id=0,
        child_idx=0,
        improved=True,
        improved_metrics=list(PARENT),
    )
    assert policy.best_global_gen == 0
    assert policy.metric_best["global_best"] == PARENT["global_best"]

    stale_child = dict(PARENT)
    stale_child["global_best"] = 0.20  # below historical best: no refresh
    policy.record_result(
        operator_id="op_a",
        parent_fitness=PARENT,
        child_fitness=stale_child,
        generation=6,
        island_id=0,
        child_idx=0,
        improved=False,
        improved_metrics=[],
    )
    assert policy.best_global_gen == 0
    assert policy._last_stagnation == 6  # 6 - 0, cross-island plateau signal
    assert policy.metric_stale["global_best"] == 1
    print("✅ structured metric tracking and plateau detection")


def test_structured_reward_standardization():
    policy = _structured_policy()
    rng = np.random.default_rng(11)
    for index in range(30):
        child = {
            key: float(np.clip(value + rng.normal(0.0, 0.02), 0.01, 1.0))
            for key, value in PARENT.items()
        }
        policy.record_result(
            operator_id=str(OPERATORS[index % len(OPERATORS)]["id"]),
            parent_fitness=PARENT,
            child_fitness=child,
            generation=index // 3,
            island_id=index % 2,
            child_idx=0,
            improved=False,
            improved_metrics=[],
        )
    assert policy._reward_n == 30
    root = policy.nodes[()]
    for stats in root["children"].values():
        mean = stats["value_sum"] / max(1, stats["visits"])
        assert abs(mean) <= 3.0, mean  # standardized z-values are clipped to [-3, 3]
    print("✅ structured reward standardization bounds")


def test_structured_state_roundtrip():
    policy = _structured_policy(plateau_stagnation=5)
    child = dict(PARENT)
    child["coverage_elite"] = 0.45
    policy.record_result(
        operator_id="op_b",
        parent_fitness=PARENT,
        child_fitness=child,
        generation=2,
        island_id=1,
        child_idx=0,
        improved=True,
        improved_metrics=["coverage_elite"],
    )
    state = policy.get_state()

    restored = _structured_policy()
    restored.set_state(state)
    assert restored.config.reward_profile == "structured"
    assert restored.config.plateau_stagnation == 5
    assert restored.metric_best == policy.metric_best
    assert restored.metric_stale == policy.metric_stale
    assert restored.best_global_gen == policy.best_global_gen
    assert restored._reward_n == policy._reward_n
    assert restored._reward_mean == policy._reward_mean
    assert restored._reward_m2 == policy._reward_m2
    assert restored._last_stagnation == policy._last_stagnation
    print("✅ structured state roundtrip")


def test_legacy_profile_uses_legacy_reward():
    policy = OperatorMCTSPolicy(
        operators=list(OPERATORS),
        rng=np.random.default_rng(3),
    )
    policy.record_result(
        operator_id="op_a",
        parent_fitness=PARENT,
        child_fitness=dict(PARENT),
        generation=0,
        island_id=0,
        child_idx=0,
        improved=True,
        improved_metrics=["global_best"],
    )
    expected = OperatorMCTSPolicy._reward(
        parent_fitness=PARENT,
        child_fitness=dict(PARENT),
        improved=True,
        improved_metrics=["global_best"],
    )
    root = policy.nodes[()]
    stats = root["children"]["op_a"]
    assert abs(stats["value_sum"] - expected) < 1e-12, (stats["value_sum"], expected)
    assert policy._reward_n == 0  # legacy path never touches standardization state
    print("✅ legacy profile unchanged")


def test_plateau_boosts_uct_exploration():
    def rigged_policy(profile: str) -> OperatorMCTSPolicy:
        policy = OperatorMCTSPolicy(
            operators=[{"id": "op_a"}, {"id": "op_b"}],
            rng=np.random.default_rng(5),
            config=OperatorMCTSConfig(reward_profile=profile, plateau_stagnation=4),
        )
        policy.nodes[()] = {
            "visits": 11,
            "value_sum": 20.0,
            "children": {
                "op_a": {"visits": 1, "value_sum": 0.0},  # mean 0.0, rarely tried
                "op_b": {"visits": 10, "value_sum": 20.0},  # mean 2.0, well exploited
            },
        }
        return policy

    calm = rigged_policy("structured")
    calm._last_stagnation = 0
    assert calm._select_operator_id(()) == "op_b"  # exploitation wins at base c

    plateaued = rigged_policy("structured")
    plateaued._last_stagnation = 10
    assert plateaued._select_operator_id(()) == "op_a"  # boosted exploration flips

    legacy = rigged_policy("legacy")
    legacy._last_stagnation = 10
    assert legacy._select_operator_id(()) == "op_b"  # legacy never boosts
    print("✅ plateau-aware UCT exploration boost")


def test_deficit_weights_rotate_to_lagging_metric():
    policy = _structured_policy(reward_weight_mode="deficit")
    policy.metric_best = dict(PARENT)
    policy.metric_best["shadow_mae_elite"] = 0.80  # parent lags only on shadow MAE

    weights = policy._effective_opt_weights(PARENT)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    top_metric = max(weights, key=weights.get)
    assert top_metric == "shadow_mae_elite", weights
    assert weights["shadow_mae_elite"] > 0.7, weights["shadow_mae_elite"]

    saturated_parent = dict(policy.metric_best)
    saturated_weights = policy._effective_opt_weights(saturated_parent)
    total_base = sum(_BASE_WEIGHTS.values())
    for key, weight in saturated_weights.items():
        expected = _BASE_WEIGHTS[key] / total_base  # no deficit: shares follow base weights
        assert abs(weight - expected) < 1e-9, (key, weight, expected)

    fixed_policy = _structured_policy(reward_weight_mode="fixed")
    fixed_policy.metric_best = dict(policy.metric_best)
    assert fixed_policy._effective_opt_weights(PARENT) == _BASE_WEIGHTS
    print("✅ deficit weights rotate toward lagging metrics")


def test_deficit_reward_amplifies_lagging_metric_improvement():
    child = dict(PARENT)
    child["shadow_mae_elite"] = 0.60  # +20% relative improvement on the lagging metric

    deficit_policy = _structured_policy(reward_weight_mode="deficit")
    deficit_policy.metric_best = dict(PARENT)
    deficit_policy.metric_best["shadow_mae_elite"] = 0.80
    deficit_reward = deficit_policy._reward_structured(
        parent_fitness=PARENT,
        child_fitness=child,
        improved=False,
        improved_metrics=[],
        new_bests={},
        stagnation=0,
    )

    fixed_policy = _structured_policy(reward_weight_mode="fixed")
    fixed_reward = fixed_policy._reward_structured(
        parent_fitness=PARENT,
        child_fitness=child,
        improved=False,
        improved_metrics=[],
        new_bests={},
        stagnation=0,
    )
    assert abs(fixed_reward - 0.20 * 0.2) < 1e-9, fixed_reward
    assert deficit_reward > fixed_reward, (deficit_reward, fixed_reward)
    print("✅ deficit mode amplifies lagging-metric improvement")


def test_deficit_summary_and_weight_state_roundtrip():
    policy = _structured_policy(reward_weight_mode="deficit")
    policy.metric_best = dict(PARENT)
    for index in range(3):
        policy.record_result(
            operator_id="op_a",
            parent_fitness=PARENT,
            child_fitness=dict(PARENT),
            generation=index,
            island_id=0,
            child_idx=0,
            improved=False,
            improved_metrics=[],
        )
    summary = policy.summary()
    assert summary["reward_weight_mode"] == "deficit"
    assert abs(sum(summary["mean_opt_weights"].values()) - 1.0) < 1e-9
    assert summary["last_opt_weights"]

    restored = _structured_policy(reward_weight_mode="deficit")
    restored.set_state(policy.get_state())
    assert restored.config.reward_weight_mode == "deficit"
    assert restored._opt_weight_sum == policy._opt_weight_sum
    assert restored._opt_weight_count == policy._opt_weight_count
    print("✅ deficit summary and weight state roundtrip")


def main():
    test_structured_reward_failed_child()
    test_structured_reward_guard_tolerance()
    test_structured_reward_progress_and_plateau()
    test_structured_tracking_and_stagnation()
    test_structured_reward_standardization()
    test_structured_state_roundtrip()
    test_legacy_profile_uses_legacy_reward()
    test_plateau_boosts_uct_exploration()
    test_deficit_weights_rotate_to_lagging_metric()
    test_deficit_reward_amplifies_lagging_metric_improvement()
    test_deficit_summary_and_weight_state_roundtrip()
    print("✅ MCTS policy reward profile tests passed")


if __name__ == "__main__":
    main()
