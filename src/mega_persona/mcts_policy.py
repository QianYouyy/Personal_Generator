"""Online MCTS-style operator policy for MegaPersona genome evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OperatorMCTSConfig:
    max_depth: int = 3
    exploration_c: float = 1.4
    progressive_widening: bool = False
    reward_profile: str = "legacy"
    plateau_stagnation: int = 4
    reward_weight_mode: str = "fixed"


# Structured reward profile constants. The structured profile is plateau-aware:
# it uses relative deltas, a bounded coverage/diversity guard, progress bonuses
# against policy-tracked historical bests, and reward standardization so the
# exploitation term stays on a scale comparable with the UCT exploration term.
_STRUCTURED_OPT_WEIGHTS = {
    "global_best": 0.30,
    "shadow_mae_elite": 0.20,
    "axis_target_elite": 0.15,
    "strict_consistency_elite": 0.15,
    "research_score_v2": 0.10,
    "alignment_elite": 0.05,
    "consistency_elite": 0.05,
    "schema_elite": 0.05,
    "issue_rate_elite": 0.05,
    "coverage_elite": 0.05,
    "diversity_elite": 0.05,
}
_STRUCTURED_GUARD_METRICS = ("coverage_elite", "diversity_elite")
_STRUCTURED_GUARD_TOLERANCE = 0.02
_STRUCTURED_GUARD_WEIGHT = 0.5
_STRUCTURED_PROGRESS_WEIGHT = 0.5
_STRUCTURED_IMPROVED_BONUS = 0.02
_STRUCTURED_STALE_BONUS = 0.05
_STRUCTURED_STALE_THRESHOLD = 20
_REWARD_STD_FLOOR = 0.02
_REWARD_Z_CLIP = 3.0
# Floor that keeps every metric direction alive in deficit-adaptive weights.
_REWARD_WEIGHT_FLOOR = 0.02
_EPS = 1e-6


@dataclass
class PendingAction:
    path: tuple[str, ...]
    operator_id: str
    generation: int
    island_id: int
    child_idx: int


class OperatorMCTSPolicy:
    """A shallow online tree policy over evolution-operator sequences.

    The expensive rollout is the existing MegaPersona validation evaluation. The
    policy therefore only chooses the next operator, then backpropagates the
    observed validation reward through the chosen operator path.
    """

    def __init__(
        self,
        operators: list[dict[str, Any]],
        rng: np.random.Generator,
        config: OperatorMCTSConfig | None = None,
    ):
        self.operators = [dict(operator) for operator in operators]
        self.operator_by_id = {str(operator["id"]): dict(operator) for operator in self.operators}
        self.rng = rng
        self.config = config or OperatorMCTSConfig()
        self.nodes: dict[tuple[str, ...], dict[str, Any]] = {}
        self.island_paths: dict[int, tuple[str, ...]] = {}
        self.pending: dict[tuple[int, int, int], PendingAction] = {}
        self.total_results = 0
        self.metric_best: dict[str, float] = {}
        self.metric_stale: dict[str, int] = {}
        self.best_global_gen: int | None = None
        self._last_stagnation = 0
        self._reward_n = 0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._last_opt_weights: dict[str, float] = {}
        self._opt_weight_sum: dict[str, float] = {}
        self._opt_weight_count = 0
        self._lock = threading.Lock()
        self._ensure_node(())

    def choose_operator(
        self,
        *,
        generation: int = 0,
        island_id: int | None = None,
        child_idx: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            path = self._path_for_island(island_id)
            operator_id = self._select_operator_id(path)
            if island_id is not None and child_idx is not None:
                self.pending[(int(generation), int(island_id), int(child_idx))] = PendingAction(
                    path=path,
                    operator_id=operator_id,
                    generation=int(generation),
                    island_id=int(island_id),
                    child_idx=int(child_idx),
                )
            return dict(self.operator_by_id[operator_id])

    def record_result(
        self,
        *,
        operator_id: str,
        parent_fitness: dict[str, float] | None,
        child_fitness: dict[str, float] | None,
        generation: int,
        island_id: int | None,
        child_idx: int | None,
        improved: bool,
        improved_metrics: list[str] | None = None,
    ) -> None:
        if not operator_id or operator_id not in self.operator_by_id:
            return
        with self._lock:
            pending_key = None
            action = None
            if island_id is not None and child_idx is not None:
                pending_key = (int(generation), int(island_id), int(child_idx))
                action = self.pending.pop(pending_key, None)
            path = action.path if action is not None else self._path_for_island(island_id)
            action_path = self._trim_path(path + (operator_id,))
            parent = parent_fitness or {}
            child = child_fitness or {}
            if str(self.config.reward_profile) == "structured":
                numeric_child = {
                    str(key): float(value)
                    for key, value in child.items()
                    if isinstance(value, (int, float))
                }
                new_bests = {
                    key: value
                    for key, value in numeric_child.items()
                    if key not in self.metric_best or value > self.metric_best[key]
                }
                stagnation = self._plateau_stagnation(generation)
                raw_reward = self._reward_structured(
                    parent_fitness=parent,
                    child_fitness=numeric_child,
                    improved=improved,
                    improved_metrics=improved_metrics or [],
                    new_bests=new_bests,
                    stagnation=stagnation,
                )
                for key, value in numeric_child.items():
                    if key in new_bests:
                        self.metric_best[key] = value
                        self.metric_stale[key] = 0
                    else:
                        self.metric_stale[key] = self.metric_stale.get(key, 0) + 1
                if new_bests.get("global_best", 0.0) > 0.0:
                    self.best_global_gen = int(generation)
                self._last_stagnation = stagnation
                reward = self._standardize_reward(raw_reward)
            else:
                reward = self._reward(
                    parent_fitness=parent,
                    child_fitness=child,
                    improved=improved,
                    improved_metrics=improved_metrics or [],
                )
            self._backpropagate(action_path, reward)
            self.total_results += 1
            if island_id is not None and (improved or reward > 0.0):
                self.island_paths[int(island_id)] = action_path

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config": asdict(self.config),
                "nodes": {
                    "|".join(path): {
                        "visits": node["visits"],
                        "value_sum": node["value_sum"],
                        "children": node["children"],
                    }
                    for path, node in self.nodes.items()
                },
                "island_paths": {
                    str(island_id): list(path)
                    for island_id, path in self.island_paths.items()
                },
                "total_results": self.total_results,
                "reward_tracking": {
                    "metric_best": dict(self.metric_best),
                    "metric_stale": dict(self.metric_stale),
                    "best_global_gen": self.best_global_gen,
                    "last_stagnation": self._last_stagnation,
                    "reward_n": self._reward_n,
                    "reward_mean": self._reward_mean,
                    "reward_m2": self._reward_m2,
                    "last_opt_weights": dict(self._last_opt_weights),
                    "opt_weight_sum": dict(self._opt_weight_sum),
                    "opt_weight_count": self._opt_weight_count,
                },
            }

    def set_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            config = state.get("config")
            if isinstance(config, dict):
                self.config = OperatorMCTSConfig(
                    max_depth=int(config.get("max_depth", self.config.max_depth)),
                    exploration_c=float(config.get("exploration_c", self.config.exploration_c)),
                    progressive_widening=bool(
                        config.get("progressive_widening", self.config.progressive_widening)
                    ),
                    reward_profile=str(config.get("reward_profile", self.config.reward_profile)),
                    plateau_stagnation=int(
                        config.get("plateau_stagnation", self.config.plateau_stagnation)
                    ),
                    reward_weight_mode=str(
                        config.get("reward_weight_mode", self.config.reward_weight_mode)
                    ),
                )
            self.nodes = {}
            for raw_path, node in state.get("nodes", {}).items():
                path = tuple(part for part in raw_path.split("|") if part)
                self.nodes[path] = {
                    "visits": int(node.get("visits", 0)),
                    "value_sum": float(node.get("value_sum", 0.0)),
                    "children": {
                        str(operator_id): {
                            "visits": int(stats.get("visits", 0)),
                            "value_sum": float(stats.get("value_sum", 0.0)),
                        }
                        for operator_id, stats in node.get("children", {}).items()
                        if operator_id in self.operator_by_id
                    },
                }
            self.island_paths = {
                int(island_id): self._trim_path(tuple(path))
                for island_id, path in state.get("island_paths", {}).items()
                if isinstance(path, list)
            }
            self.total_results = int(state.get("total_results", 0))
            tracking = state.get("reward_tracking")
            if isinstance(tracking, dict):
                self.metric_best = {
                    str(key): float(value)
                    for key, value in tracking.get("metric_best", {}).items()
                }
                self.metric_stale = {
                    str(key): int(value)
                    for key, value in tracking.get("metric_stale", {}).items()
                }
                best_gen = tracking.get("best_global_gen")
                self.best_global_gen = int(best_gen) if best_gen is not None else None
                self._last_stagnation = int(tracking.get("last_stagnation", 0))
                self._reward_n = int(tracking.get("reward_n", 0))
                self._reward_mean = float(tracking.get("reward_mean", 0.0))
                self._reward_m2 = float(tracking.get("reward_m2", 0.0))
                self._last_opt_weights = {
                    str(key): float(value)
                    for key, value in tracking.get("last_opt_weights", {}).items()
                }
                self._opt_weight_sum = {
                    str(key): float(value)
                    for key, value in tracking.get("opt_weight_sum", {}).items()
                }
                self._opt_weight_count = int(tracking.get("opt_weight_count", 0))
            self.pending = {}
            self._ensure_node(())

    def summary(self, top_k: int = 12) -> dict[str, Any]:
        with self._lock:
            root = self._ensure_node(())
            operator_stats = []
            for operator in self.operators:
                operator_id = str(operator["id"])
                stats = root["children"].get(operator_id, {"visits": 0, "value_sum": 0.0})
                visits = int(stats["visits"])
                operator_stats.append(
                    {
                        "operator_id": operator_id,
                        "visits": visits,
                        "mean_reward": float(stats["value_sum"] / visits) if visits else 0.0,
                    }
                )
            operator_stats.sort(key=lambda item: (item["mean_reward"], item["visits"]), reverse=True)
            paths = []
            for path, node in self.nodes.items():
                if not path:
                    continue
                visits = int(node["visits"])
                paths.append(
                    {
                        "path": list(path),
                        "visits": visits,
                        "mean_reward": float(node["value_sum"] / visits) if visits else 0.0,
                    }
                )
            paths.sort(key=lambda item: (item["mean_reward"], item["visits"]), reverse=True)
            mean_opt_weights = {}
            if self._opt_weight_count > 0:
                mean_opt_weights = {
                    key: value / self._opt_weight_count
                    for key, value in sorted(self._opt_weight_sum.items())
                }
            return {
                "config": asdict(self.config),
                "total_results": self.total_results,
                "reward_profile": str(self.config.reward_profile),
                "reward_weight_mode": str(self.config.reward_weight_mode),
                "last_stagnation": self._last_stagnation,
                "last_opt_weights": dict(self._last_opt_weights),
                "mean_opt_weights": mean_opt_weights,
                "root_operator_stats": operator_stats,
                "top_paths": paths[:top_k],
            }

    def _path_for_island(self, island_id: int | None) -> tuple[str, ...]:
        if island_id is None:
            return ()
        return self._trim_path(self.island_paths.get(int(island_id), ()))

    def _trim_path(self, path: tuple[str, ...]) -> tuple[str, ...]:
        max_depth = max(1, int(self.config.max_depth))
        clean = tuple(operator_id for operator_id in path if operator_id in self.operator_by_id)
        return clean[-max_depth:]

    def _ensure_node(self, path: tuple[str, ...]) -> dict[str, Any]:
        path = self._trim_path(path)
        if path not in self.nodes:
            self.nodes[path] = {"visits": 0, "value_sum": 0.0, "children": {}}
        return self.nodes[path]

    def _select_operator_id(self, path: tuple[str, ...]) -> str:
        node = self._ensure_node(path)
        available_ids = [str(operator["id"]) for operator in self.operators]
        children = node["children"]
        allowed_ids = available_ids
        if self.config.progressive_widening:
            width = min(len(available_ids), max(1, int(math.sqrt(max(1, node["visits"]))) + 1))
            known = sorted(
                children,
                key=lambda operator_id: (
                    children[operator_id]["value_sum"] / max(1, children[operator_id]["visits"]),
                    children[operator_id]["visits"],
                ),
                reverse=True,
            )
            untried = [operator_id for operator_id in available_ids if operator_id not in children]
            allowed_ids = (known + untried)[:width]
        untried_allowed = [operator_id for operator_id in allowed_ids if operator_id not in children]
        if untried_allowed:
            return str(self.rng.choice(untried_allowed))
        parent_visits = max(1, int(node["visits"]))
        exploration_c = float(self.config.exploration_c)
        if (
            str(self.config.reward_profile) == "structured"
            and self._last_stagnation >= int(self.config.plateau_stagnation)
        ):
            exploration_c *= 1.0 + 0.2 * min(self._last_stagnation, 8)

        def uct(operator_id: str) -> float:
            stats = children[operator_id]
            visits = max(1, int(stats["visits"]))
            mean = float(stats["value_sum"]) / visits
            exploration = exploration_c * math.sqrt(math.log(parent_visits + 1.0) / visits)
            return mean + exploration

        return max(allowed_ids, key=uct)

    def _backpropagate(self, action_path: tuple[str, ...], reward: float) -> None:
        for depth in range(len(action_path)):
            prefix = action_path[:depth]
            operator_id = action_path[depth]
            node = self._ensure_node(prefix)
            node["visits"] += 1
            node["value_sum"] += reward
            child_stats = node["children"].setdefault(operator_id, {"visits": 0, "value_sum": 0.0})
            child_stats["visits"] += 1
            child_stats["value_sum"] += reward
        leaf = self._ensure_node(action_path)
        leaf["visits"] += 1
        leaf["value_sum"] += reward

    @staticmethod
    def _reward(
        *,
        parent_fitness: dict[str, float],
        child_fitness: dict[str, float],
        improved: bool,
        improved_metrics: list[str],
    ) -> float:
        parent_global = float(parent_fitness.get("global_best", 0.0))
        child_global = float(child_fitness.get("global_best", 0.0))
        coverage_delta = _delta(parent_fitness, child_fitness, "coverage_elite")
        diversity_delta = _delta(parent_fitness, child_fitness, "diversity_elite")
        strict_delta = _delta(parent_fitness, child_fitness, "strict_consistency_elite")
        shadow_delta = _delta(parent_fitness, child_fitness, "shadow_mae_elite")
        reward = child_global - parent_global
        reward += 0.16 * coverage_delta
        reward += 0.10 * _delta(parent_fitness, child_fitness, "alignment_elite")
        reward += 0.14 * shadow_delta
        reward += 0.08 * _delta(parent_fitness, child_fitness, "consistency_elite")
        reward += 0.12 * strict_delta
        reward += 0.08 * _delta(parent_fitness, child_fitness, "axis_target_elite")
        reward += 0.05 * _delta(parent_fitness, child_fitness, "issue_rate_elite")
        reward += 0.10 * _delta(parent_fitness, child_fitness, "research_score_v2")
        reward += 0.08 * _delta(parent_fitness, child_fitness, "schema_elite")
        reward += 0.14 * diversity_delta
        reward -= 0.35 * max(0.0, -coverage_delta)
        reward -= 0.35 * max(0.0, -diversity_delta)
        reward -= 0.20 * max(0.0, -_delta(parent_fitness, child_fitness, "alignment_elite"))
        reward -= 0.25 * max(0.0, -shadow_delta)
        reward -= 0.20 * max(0.0, -_delta(parent_fitness, child_fitness, "consistency_elite"))
        reward -= 0.25 * max(0.0, -strict_delta)
        if strict_delta > 0.0:
            reward -= 0.20 * max(0.0, -coverage_delta)
            reward -= 0.20 * max(0.0, -diversity_delta)
        if shadow_delta > 0.0:
            reward -= 0.10 * max(0.0, -coverage_delta)
            reward -= 0.10 * max(0.0, -diversity_delta)
        if improved:
            reward += 0.01 * max(1, len(improved_metrics))
        return float(reward)

    def _plateau_stagnation(self, generation: int) -> int:
        """Generations since the policy last saw global_best improve.

        Mirrors OpenEvolve._global_stagnation but from the policy's own
        cross-island result stream, so no engine signature changes are needed.
        """
        if self.best_global_gen is None:
            return 0
        return max(0, int(generation) - int(self.best_global_gen))

    def _effective_opt_weights(self, parent_fitness: dict[str, float]) -> dict[str, float]:
        """Per-metric optimization weights for the structured reward.

        `fixed` returns the static base weights. `deficit` scales each base
        weight by how far the parent lags the policy-tracked historical best
        on that metric (plus a floor so no direction dies), then normalizes
        the weights into shares: saturated directions lose weight and lagging
        directions gain it, so the reward rotates across metric combinations
        instead of always chasing the fixed global_best share.
        """
        if str(self.config.reward_weight_mode) != "deficit":
            return dict(_STRUCTURED_OPT_WEIGHTS)
        weights: dict[str, float] = {}
        for key, base in _STRUCTURED_OPT_WEIGHTS.items():
            best = self.metric_best.get(key)
            parent_value = float(parent_fitness.get(key, 0.0))
            deficit = max(0.0, (best if best is not None else parent_value) - parent_value)
            weights[key] = base * (deficit + _REWARD_WEIGHT_FLOOR)
        total = sum(weights.values()) or 1.0
        return {key: value / total for key, value in weights.items()}

    def _reward_structured(
        self,
        *,
        parent_fitness: dict[str, float],
        child_fitness: dict[str, float],
        improved: bool,
        improved_metrics: list[str],
        new_bests: dict[str, float],
        stagnation: int,
    ) -> float:
        """Layered plateau-aware reward: hard gate, soft guard, optimization,
        historical-best progress, and a plateau escape bonus.

        Uses relative deltas so a fixed coverage/diversity tolerance is
        meaningful, and only penalizes guard-metric drops beyond the tolerance
        instead of the legacy asymmetric penalty block that pushed mean rewards
        negative during plateaus.
        """
        child_global = float(child_fitness.get("global_best", 0.0))
        if child_global <= 0.0:
            return -1.0
        opt_weights = self._effective_opt_weights(parent_fitness)
        reward = 0.0
        for key, weight in opt_weights.items():
            parent_value = float(parent_fitness.get(key, 0.0))
            child_value = float(child_fitness.get(key, 0.0))
            reward += weight * (child_value - parent_value) / max(abs(parent_value), _EPS)
        if str(self.config.reward_weight_mode) == "deficit":
            self._last_opt_weights = dict(opt_weights)
            for key, value in opt_weights.items():
                self._opt_weight_sum[key] = self._opt_weight_sum.get(key, 0.0) + value
            self._opt_weight_count += 1
        progress = 0.0
        stale_improved = 0
        for key, value in new_bests.items():
            previous = self.metric_best.get(key)
            if previous is None:
                continue
            progress += max(0.0, (value - previous) / max(abs(previous), _EPS))
            if self.metric_stale.get(key, 0) >= _STRUCTURED_STALE_THRESHOLD:
                stale_improved += 1
        reward += _STRUCTURED_PROGRESS_WEIGHT * progress
        guard_drop = 0.0
        for key in _STRUCTURED_GUARD_METRICS:
            parent_value = float(parent_fitness.get(key, 0.0))
            child_value = float(child_fitness.get(key, 0.0))
            relative_drop = (parent_value - child_value) / max(abs(parent_value), _EPS)
            guard_drop += max(0.0, relative_drop - _STRUCTURED_GUARD_TOLERANCE)
        reward -= _STRUCTURED_GUARD_WEIGHT * guard_drop
        if improved:
            reward += _STRUCTURED_IMPROVED_BONUS * max(1, len(improved_metrics))
        if stagnation >= int(self.config.plateau_stagnation):
            reward += _STRUCTURED_PROGRESS_WEIGHT * progress
            reward += _STRUCTURED_STALE_BONUS * stale_improved
        return float(reward)

    def _standardize_reward(self, reward: float) -> float:
        """Z-score rewards with Welford running stats before backpropagation.

        Keeps the UCT exploitation term on a unit scale so exploration_c stays
        meaningful; the std floor avoids amplifying near-zero plateau noise.
        """
        self._reward_n += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_n
        self._reward_m2 += delta * (reward - self._reward_mean)
        variance = self._reward_m2 / max(1, self._reward_n - 1)
        std = max(math.sqrt(max(variance, 0.0)), _REWARD_STD_FLOOR)
        z_value = (reward - self._reward_mean) / std
        return float(np.clip(z_value, -_REWARD_Z_CLIP, _REWARD_Z_CLIP))


def _delta(parent: dict[str, float], child: dict[str, float], key: str) -> float:
    return float(child.get(key, 0.0)) - float(parent.get(key, 0.0))
