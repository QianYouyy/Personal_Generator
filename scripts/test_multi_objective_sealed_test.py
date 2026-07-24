"""Unit tests for the multi-objective sealed-test comparison script."""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_multi_objective_sealed_test import (
    _config_from_manifest,
    _load_candidate_payloads,
    build_sealed_test_comparison,
    sealed_test_comparison_markdown,
)


def _write_candidate(candidates_dir: Path, candidate_id: str, fitness: float) -> None:
    payload = {
        "candidate_id": candidate_id,
        "generation": 3,
        "parent_id": None,
        "fitness": fitness,
        "genome": {"schema_binding": {"axis_names": ["a"]}},
        "metrics": {"validation_behavior_coverage.mean": 0.4},
    }
    (candidates_dir / f"{candidate_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class MockEvolver:
    def __init__(self):
        self.evaluated_ids = []

    def evaluate_final_test(self, candidate):
        self.evaluated_ids.append(candidate.candidate_id)
        if candidate.candidate_id == "cand_skip":
            return {"status": "skipped_no_successful_validation_candidate", "metrics": {}}
        return {
            "status": "ok",
            "metrics": {
                "test_behavior_coverage.mean": 0.41,
                "test_behavior_balanced_diversity.mean": 0.33,
                "test_shadow_mae.mean": 0.21,
                "test_strict_consistency_error.mean": 0.05,
                "test_axis_target_mae.mean": 0.06,
                "test_schema_fitness.mean": 0.45,
                "test_internal_consistency.mean": 0.9,
                "unrelated_metric.mean": 1.0,
            },
        }


def test_build_comparison_dedup_and_metrics():
    with TemporaryDirectory() as tmp:
        candidates_dir = Path(tmp) / "candidates"
        candidates_dir.mkdir()
        _write_candidate(candidates_dir, "cand_a", 0.28)
        _write_candidate(candidates_dir, "cand_b", 0.26)
        _write_candidate(candidates_dir, "cand_skip", 0.20)

        roles = {
            "global_best": {"candidate_id": "cand_a", "operator_id": "op17"},
            "coverage_best": {"candidate_id": "cand_b", "operator_id": "op04"},
            "diversity_best": {"candidate_id": "cand_b", "operator_id": "op04"},
            "schema_best": {"candidate_id": "cand_skip", "operator_id": "op09"},
            "missing_best": {"candidate_id": "cand_missing", "operator_id": "op00"},
        }
        evolver = MockEvolver()
        comparison = build_sealed_test_comparison(evolver, roles, candidates_dir)

        # cand_b holds two roles but is evaluated once; missing candidate ignored.
        assert evolver.evaluated_ids == ["cand_a", "cand_b", "cand_skip"]
        entries = {entry["candidate_id"]: entry for entry in comparison["entries"]}
        assert entries["cand_b"]["roles"] == ["coverage_best", "diversity_best"]
        assert entries["cand_a"]["status"] == "ok"
        assert entries["cand_a"]["test_metrics"]["test_shadow_mae.mean"] == 0.21
        assert "unrelated_metric.mean" not in entries["cand_a"]["test_metrics"]
        assert entries["cand_skip"]["status"] == "skipped_no_successful_validation_candidate"
        assert entries["cand_skip"]["test_metrics"] == {}
    print("✅ sealed-test comparison dedup and metric extraction")


def test_markdown_renders_table():
    comparison = {
        "run_dir": "data/results/demo",
        "candidates_scanned": 12,
        "entries": [
            {
                "candidate_id": "cand_a",
                "roles": ["global_best"],
                "generation": 7,
                "operator_id": "op17",
                "validation_fitness": 0.28,
                "status": "ok",
                "test_metrics": {
                    "test_behavior_coverage.mean": 0.41,
                    "test_behavior_balanced_diversity.mean": 0.33,
                    "test_shadow_mae.mean": 0.21,
                    "test_strict_consistency_error.mean": 0.05,
                    "test_axis_target_mae.mean": 0.06,
                },
            }
        ],
    }
    markdown = sealed_test_comparison_markdown(comparison)
    assert "cand_a" in markdown
    assert "0.4100" in markdown
    assert "test used for selection: `False`" in markdown
    print("✅ sealed-test comparison markdown")


def test_config_from_manifest_filters_and_overrides():
    manifest = {
        "config": {
            "n": 8,
            "seeds": [23],
            "shadow_simulator_backend": "student-realistic-v2",
            "num_islands": 8,  # not a MegaEvolutionConfig field: must be dropped
            "population_size_deprecated": 8,
        }
    }
    args = argparse.Namespace(shadow_max_workers=24)
    config = _config_from_manifest(manifest, args)
    assert config.n == 8
    assert config.seeds == (23,)
    assert config.shadow_simulator_backend == "student-realistic-v2"
    assert config.shadow_max_workers == 24

    args_default = argparse.Namespace(shadow_max_workers=None)
    config_default = _config_from_manifest(manifest, args_default)
    assert config_default.shadow_max_workers == 1  # dataclass default preserved
    print("✅ manifest config reconstruction")


def test_load_candidate_payloads_skips_unreadable():
    with TemporaryDirectory() as tmp:
        candidates_dir = Path(tmp)
        _write_candidate(candidates_dir, "cand_ok", 0.2)
        (candidates_dir / "broken.json").write_text("{not json", encoding="utf-8")
        payloads = _load_candidate_payloads(candidates_dir)
        assert list(payloads) == ["cand_ok"]
    print("✅ candidate payload loading")


def main():
    test_build_comparison_dedup_and_metrics()
    test_markdown_renders_table()
    test_config_from_manifest_filters_and_overrides()
    test_load_candidate_payloads_skips_unreadable()
    print("✅ multi-objective sealed-test script tests passed")


if __name__ == "__main__":
    main()
