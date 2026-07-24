"""Smoke tests for the MegaPersona OpenEvolve adapter."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona.evolution import default_genome
from src.mega_persona.openevolve_adapter import (
    MegaGenomeMutator,
    MegaOpenEvolveEvaluator,
    genome_from_code,
    genome_hash,
    genome_phenotype_hash,
    genome_to_code,
    multi_objective_best_candidates,
    open_evolve_fitness_from_payload,
)
from src.mega_persona.mcts_policy import OperatorMCTSPolicy
from src.open_evolve.engine import OpenEvolve


class MockMegaEvaluator:
    num_personas = 2

    def __init__(self):
        self.count = 0

    def evaluate(self, code_str: str):
        self.count += 1
        genome = genome_from_code(code_str)
        specificity = genome.get("prompt_profile", {}).get("specificity", "")
        bonus = 0.01 if specificity == "behavioral" else 0.0
        return {
            "global_best": 0.20 + bonus + self.count * 0.001,
            "coverage_elite": 0.10,
            "alignment_elite": 0.70,
            "consistency_elite": 0.80,
            "diversity_elite": 0.60,
            "schema_elite": 0.50,
        }


class MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "mock-mutator"
        self.calls = []

    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2048, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            raise RuntimeError("No mock responses left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_genome_round_trip():
    genome = default_genome()
    code = genome_to_code(genome)
    assert genome_from_code(code) == genome
    print("✅ genome JSON round-trip")


def test_fitness_mapping():
    payload = {
        "fitness": 0.25,
        "metrics": {
            "validation_behavior_coverage.mean": 0.30,
            "validation_behavior_avg_dist.mean": 0.42,
            "validation_behavior_balanced_diversity.mean": 0.55,
            "validation_shadow_alignment.mean": 0.80,
            "internal_consistency.mean": 0.90,
            "schema_fitness.mean": 0.50,
            "slot_coverage.mean": 0.60,
            "near_duplicate_rate.mean": 0.10,
        },
    }
    mapped = open_evolve_fitness_from_payload(payload)
    assert mapped["global_best"] == 0.25
    assert mapped["coverage_elite"] == 0.30
    assert mapped["alignment_elite"] == 0.80
    assert mapped["consistency_elite"] == 0.90
    assert mapped["diversity_elite"] == 0.55
    assert mapped["schema_elite"] == 0.50
    cache_mapped = open_evolve_fitness_from_payload(
        {**payload, "phenotype_cache_hit": True}
    )
    assert cache_mapped["phenotype_cache_hit"] is True
    print("✅ fitness mapping")


def test_open_evolve_accepts_mega_genome_seed():
    with TemporaryDirectory() as tmp:
        seed_codes = {"mega_default": genome_to_code(default_genome())}
        engine = OpenEvolve(
            mutator=MegaGenomeMutator(random_seed=7),
            evaluator=MockMegaEvaluator(),
            questionnaires=[],
            seed_codes=seed_codes,
            initial_seed_distribution={"mega_default": 2},
            num_islands=2,
            checkpoint_path=Path(tmp),
        )
        best = engine.run(max_generations=1, children_per_island=1)
        assert best is not None
        assert (Path(tmp) / "checkpoint.json").exists()
        assert genome_from_code(best.code)
    print("✅ OpenEvolve engine accepts MegaPersona genome seeds")


def test_llm_mutator_path():
    parent = default_genome()
    child = json.loads(json.dumps(parent))
    child["prompt_profile"]["specificity"] = "behavioral"
    child["axis_bias"]["motivation_autonomy"] = 0.12
    mutator = MegaGenomeMutator(
        random_seed=7,
        llm_client=MockLLM([json.dumps(child, ensure_ascii=False)]),
    )
    mutated = genome_from_code(mutator.mutate(genome_to_code(parent), generation=2, stagnation=1))
    assert mutated["prompt_profile"]["specificity"] == "behavioral"
    # Numeric jitter is applied on top of the LLM patch: close to 0.12, not exact.
    assert abs(mutated["axis_bias"]["motivation_autonomy"] - 0.12) < 0.08
    assert mutated["axis_bias"]["motivation_autonomy"] != 0.0
    assert mutated["openevolve_mutation"]["backend"] == "llm"
    assert mutated["openevolve_mutation"]["generation"] == 2
    print("✅ LLM mutator path")


def test_llm_mutator_patch_path():
    parent = default_genome()
    patch = {
        "patch": {
            "prompt_profile": {"specificity": "behavioral"},
            "axis_bias": {"motivation_autonomy": 0.11},
            "behavior_anchors": {
                "stress_recovery": "Trigger: failed attempt. Response: delayed repair with one concrete help-seeking action."
            },
            "blueprint_policy": {
                "core_tension_rule": "Bind the strongest and weakest axes through one repeated classroom tension."
            },
            "behavior_prediction_policy": {
                "deadline": "Predict planning rhythm, shortcut risk, and repair behavior after delay."
            },
        }
    }
    mutator = MegaGenomeMutator(
        random_seed=7,
        llm_client=MockLLM([json.dumps(patch, ensure_ascii=False)]),
    )
    mutated = genome_from_code(mutator.mutate(genome_to_code(parent), generation=2, stagnation=1))
    assert mutated["prompt_profile"]["specificity"] == "behavioral"
    # Numeric jitter is applied on top of the LLM patch: close to 0.11, not exact.
    assert abs(mutated["axis_bias"]["motivation_autonomy"] - 0.11) < 0.08
    assert "delayed repair" in mutated["behavior_anchors"]["stress_recovery"]
    assert "classroom tension" in mutated["blueprint_policy"]["core_tension_rule"]
    assert "shortcut risk" in mutated["behavior_prediction_policy"]["deadline"]
    assert mutated["openevolve_mutation"]["backend"] == "llm"
    print("✅ LLM mutator patch path")


def test_llm_mutator_fallback_to_rule():
    parent = default_genome()
    mutator = MegaGenomeMutator(
        random_seed=7,
        llm_client=MockLLM(["not-json", "still-not-json"]),
    )
    mutated = genome_from_code(mutator.mutate(genome_to_code(parent), generation=3, stagnation=2))
    assert mutated["openevolve_mutation"]["backend"] == "rule"
    assert "fallback_reason" in mutated["openevolve_mutation"]
    assert set(mutated["quota_weights"]) == set(parent["quota_weights"])
    print("✅ LLM mutator fallback")


def test_fixed_operator_mutator():
    parent = default_genome()
    mutator = MegaGenomeMutator(random_seed=7, fixed_operator_id="op06_low_axis_fidelity")
    for generation in range(3):
        chosen = mutator.choose_operator(generation=generation, island_id=generation)
        assert chosen["id"] == "op06_low_axis_fidelity"
        mutated = genome_from_code(mutator.mutate(genome_to_code(parent), generation=generation))
        assert mutated["openevolve_mutation"]["operator_id"] == "op06_low_axis_fidelity"
    print("✅ fixed operator mutator")


def test_hybrid_mcts_policy_state():
    mutator = MegaGenomeMutator(
        random_seed=7,
        operator_family="v3",
        search_strategy="hybrid_mcts",
        mcts_depth=2,
    )
    chosen = mutator.choose_operator(generation=1, island_id=0, child_idx=0)
    assert chosen["id"].startswith("op")
    mutator.record_result(
        operator_id=chosen["id"],
        parent_fitness={
            "global_best": 0.20,
            "coverage_elite": 0.30,
            "alignment_elite": 0.80,
            "consistency_elite": 0.90,
            "schema_elite": 0.40,
            "diversity_elite": 0.20,
        },
        child_fitness={
            "global_best": 0.25,
            "coverage_elite": 0.35,
            "alignment_elite": 0.81,
            "consistency_elite": 0.91,
            "schema_elite": 0.42,
            "diversity_elite": 0.22,
        },
        generation=1,
        island_id=0,
        child_idx=0,
        improved=True,
        improved_metrics=["global_best", "coverage_elite"],
    )
    summary = mutator.mcts_summary()
    assert summary is not None
    assert summary["total_results"] == 1
    assert summary["root_operator_stats"][0]["visits"] >= 1

    restored = MegaGenomeMutator(
        random_seed=11,
        operator_family="v3",
        search_strategy="hybrid_mcts",
    )
    restored.set_state(mutator.get_state())
    restored_summary = restored.mcts_summary()
    assert restored_summary is not None
    assert restored_summary["total_results"] == 1
    print("✅ hybrid MCTS policy state")


def test_hybrid_mcts_reward_protects_diversity():
    parent = {
        "global_best": 0.25,
        "coverage_elite": 0.40,
        "alignment_elite": 0.80,
        "shadow_mae_elite": 0.80,
        "consistency_elite": 0.90,
        "strict_consistency_elite": 0.90,
        "axis_target_elite": 0.90,
        "issue_rate_elite": 0.90,
        "research_score_v2": 0.24,
        "schema_elite": 0.45,
        "diversity_elite": 0.35,
    }
    protected_child = dict(parent)
    protected_child.update(
        {
            "global_best": 0.26,
            "shadow_mae_elite": 0.82,
            "strict_consistency_elite": 0.94,
            "research_score_v2": 0.25,
        }
    )
    collapsed_child = dict(protected_child)
    collapsed_child.update(
        {
            "coverage_elite": 0.32,
            "diversity_elite": 0.27,
        }
    )

    protected_reward = OperatorMCTSPolicy._reward(
        parent_fitness=parent,
        child_fitness=protected_child,
        improved=True,
        improved_metrics=["strict_consistency_elite", "shadow_mae_elite"],
    )
    collapsed_reward = OperatorMCTSPolicy._reward(
        parent_fitness=parent,
        child_fitness=collapsed_child,
        improved=True,
        improved_metrics=["strict_consistency_elite", "shadow_mae_elite"],
    )
    assert collapsed_reward < protected_reward
    assert collapsed_reward < 0.0
    print("✅ hybrid MCTS reward protects coverage/diversity")


def test_multi_objective_candidate_report():
    with TemporaryDirectory() as tmp:
        candidates_dir = Path(tmp) / "candidates"
        candidates_dir.mkdir()
        base_genome = default_genome()

        def write_candidate(candidate_id, fitness, generation, operator_id, metrics):
            genome = json.loads(json.dumps(base_genome))
            genome["openevolve_mutation"] = {"operator_id": operator_id}
            payload = {
                "candidate_id": candidate_id,
                "genome": genome,
                "generation": generation,
                "parent_id": None,
                "fitness": fitness,
                "metrics": metrics,
                "evaluated": True,
            }
            (candidates_dir / f"{candidate_id}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

        write_candidate(
            "fitness_best",
            0.30,
            3,
            "op17",
            {
                "schema_fitness.mean": 0.45,
                "validation_behavior_coverage.mean": 0.35,
                "validation_behavior_balanced_diversity.mean": 0.30,
                "validation_shadow_alignment.mean": 0.82,
                "validation_shadow_mae.mean": 0.18,
                "strict_consistency_error.mean": 0.06,
                "axis_target_mae.mean": 0.08,
            },
        )
        write_candidate(
            "coverage_best",
            0.27,
            4,
            "op18",
            {
                "schema_fitness.mean": 0.44,
                "validation_behavior_coverage.mean": 0.48,
                "validation_behavior_balanced_diversity.mean": 0.39,
                "validation_shadow_alignment.mean": 0.80,
                "validation_shadow_mae.mean": 0.20,
                "strict_consistency_error.mean": 0.07,
                "axis_target_mae.mean": 0.09,
            },
        )
        write_candidate(
            "strict_best",
            0.28,
            5,
            "op20",
            {
                "schema_fitness.mean": 0.46,
                "validation_behavior_coverage.mean": 0.34,
                "validation_behavior_balanced_diversity.mean": 0.31,
                "validation_shadow_alignment.mean": 0.84,
                "validation_shadow_mae.mean": 0.16,
                "strict_consistency_error.mean": 0.03,
                "axis_target_mae.mean": 0.05,
            },
        )

        report = multi_objective_best_candidates(candidates_dir)
        roles = report["roles"]
        assert report["candidates_scanned"] == 3
        assert roles["global_best"]["candidate_id"] == "fitness_best"
        assert roles["coverage_best"]["candidate_id"] == "coverage_best"
        assert roles["diversity_best"]["candidate_id"] == "coverage_best"
        assert roles["strict_consistency_best"]["candidate_id"] == "strict_best"
        assert roles["shadow_mae_best"]["candidate_id"] == "strict_best"
    print("✅ multi-objective candidate report")


def test_llm_mutator_edit_audit():
    parent = default_genome()
    patch = {
        "patch": {
            "blueprint_policy": {
                "core_tension_rule": "Bind the strongest and weakest axes through one repeated classroom tension."
            },
            "axis_bias": {"motivation_autonomy": 0.11},
        },
        "declared_edits": ["blueprint_policy.core_tension_rule", "critic_policy"],
    }
    mutator = MegaGenomeMutator(
        random_seed=7,
        llm_client=MockLLM([json.dumps(patch, ensure_ascii=False)]),
    )
    mutated = genome_from_code(mutator.mutate(genome_to_code(parent), generation=2, stagnation=0))
    audit = mutated["openevolve_mutation"]
    assert audit["backend"] == "llm"
    assert audit["declared_edits"] == ["blueprint_policy.core_tension_rule", "critic_policy"]
    assert set(audit["actual_edits"]) == {"blueprint_policy", "axis_bias"}
    assert audit["undeclared_edits"] == ["axis_bias"]
    assert audit["phantom_edits"] == ["critic_policy"]
    assert "_mutation_edit_audit" not in mutated
    print("✅ LLM mutator edit audit")


def test_llm_mutator_noop_retry():
    parent = default_genome()
    noop_patch = {"patch": {}, "declared_edits": ["blueprint_policy"]}
    real_patch = {
        "patch": {
            "blueprint_policy": {
                "core_tension_rule": "Tie the weakest axis to one repeated peer-feedback scene."
            }
        },
        "declared_edits": ["blueprint_policy.core_tension_rule"],
    }
    client = MockLLM([
        json.dumps(noop_patch, ensure_ascii=False),
        json.dumps(real_patch, ensure_ascii=False),
    ])
    mutator = MegaGenomeMutator(random_seed=7, llm_client=client)
    mutated = genome_from_code(mutator.mutate(genome_to_code(parent), generation=1, stagnation=0))
    audit = mutated["openevolve_mutation"]
    assert audit["noop_retries"] == 1
    assert audit["actual_edits"] == ["blueprint_policy"]
    assert "peer-feedback scene" in mutated["blueprint_policy"]["core_tension_rule"]
    assert len(client.calls) == 2
    assert "NO effective change" in client.calls[1]["prompt"]
    print("✅ LLM mutator no-op retry")


def test_llm_mutator_noop_retry_exhausted():
    parent = default_genome()
    noop_patch = {"patch": {}, "declared_edits": []}
    client = MockLLM([
        json.dumps(noop_patch, ensure_ascii=False),
        json.dumps(noop_patch, ensure_ascii=False),
    ])
    mutator = MegaGenomeMutator(random_seed=7, llm_client=client)
    mutated = genome_from_code(mutator.mutate(genome_to_code(parent), generation=1, stagnation=0))
    audit = mutated["openevolve_mutation"]
    assert audit["noop_retries"] == 2
    assert audit["actual_edits"] == []
    assert len(client.calls) == 2
    print("✅ LLM mutator no-op retry exhausted")


def test_llm_mutator_numeric_jitter():
    parent = default_genome()
    patch = {
        "patch": {
            "blueprint_policy": {
                "core_tension_rule": "Bind the strongest and weakest axes through one repeated classroom tension."
            }
        },
        "declared_edits": ["blueprint_policy.core_tension_rule"],
    }
    mutator = MegaGenomeMutator(
        random_seed=7,
        llm_client=MockLLM([json.dumps(patch, ensure_ascii=False)]),
    )
    mutated = genome_from_code(mutator.mutate(genome_to_code(parent), generation=1, stagnation=0))
    audit = mutated["openevolve_mutation"]
    # Mutator-facing audit stays clean: only the text edit is listed.
    assert audit["actual_edits"] == ["blueprint_policy"]
    # But the numeric surface moved via jitter and is reported separately.
    jitter = audit.get("numeric_jitter") or {}
    assert jitter.get("axis_bias")
    assert jitter.get("axis_stretch")
    assert any(
        mutated["axis_bias"][axis] != parent["axis_bias"][axis]
        for axis in jitter["axis_bias"]
    )
    assert all(-0.35 <= value <= 0.35 for value in mutated["axis_bias"].values())
    assert all(0.55 <= value <= 1.75 for value in mutated["axis_stretch"].values())
    print("✅ LLM mutator numeric jitter")


def test_genome_phenotype_hash_excludes_lineage_metadata():
    base = default_genome()
    variant = json.loads(json.dumps(base))
    variant["last_evolution_operator"] = {"id": "op16", "instruction": "x"}
    variant["last_mutation"] = {"mode": "mixed", "scale": 0.2}
    variant["openevolve_mutation"] = {"backend": "llm", "timestamp": "2026-07-20"}
    assert genome_phenotype_hash(base) == genome_phenotype_hash(variant)
    assert genome_hash(base) != genome_hash(variant)
    changed = json.loads(json.dumps(base))
    changed["blueprint_policy"]["core_tension_rule"] = "A different rule entirely."
    assert genome_phenotype_hash(base) != genome_phenotype_hash(changed)
    print("✅ genome phenotype hash")


def test_evaluator_phenotype_cache():
    from types import SimpleNamespace

    class StubStore:
        def __init__(self, root):
            self.candidates_dir = Path(root) / "candidates"
            self.candidates_dir.mkdir(parents=True)
            self.payloads = {}

        def find_candidate_result(self, candidate_id):
            return self.payloads.get(candidate_id)

        def write_evaluation(self, *, evaluation_index, candidate, payload):
            payload = {**payload, "candidate": candidate.to_dict()}
            self.payloads[candidate.candidate_id] = payload
            record = {"candidate_id": candidate.candidate_id, "genome": candidate.genome}
            (self.candidates_dir / f"{candidate.candidate_id}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )

        def write_cached_evaluation_alias(self, *, candidate, payload):
            self.payloads[candidate.candidate_id] = payload
            (self.candidates_dir / f"{candidate.candidate_id}.json").write_text(
                json.dumps(candidate.to_dict(), ensure_ascii=False), encoding="utf-8"
            )

    class StubBackend:
        def __init__(self, root):
            self.config = SimpleNamespace(n=2)
            self.store = StubStore(root)
            self.evaluation_count = 0
            self.best_candidate_id = None
            self.calls = 0

        def evaluate_candidate(self, candidate):
            self.calls += 1
            return {
                "fitness": 0.42,
                "metrics": {"validation_behavior_coverage.mean": 0.3},
                "per_seed": [],
            }

        def _save_checkpoint(self):
            pass

    with TemporaryDirectory() as tmp:
        backend = StubBackend(tmp)
        evaluator = MegaOpenEvolveEvaluator(backend)
        genome_a = default_genome()
        genome_b = json.loads(json.dumps(genome_a))
        genome_b["last_evolution_operator"] = {
            "id": "op17_v3_axis_coverage_grid",
            "name": "x",
            "instruction": "y",
        }
        genome_b["openevolve_mutation"] = {"backend": "llm", "timestamp": "later"}
        first = evaluator.evaluate(genome_to_code(genome_a))
        assert backend.calls == 1
        assert backend.evaluation_count == 1
        source_candidate_id = evaluator.candidate_id_for_code(genome_to_code(genome_a))
        second = evaluator.evaluate(genome_to_code(genome_b))
        assert backend.calls == 1  # phenotype cache hit, no re-evaluation
        assert backend.evaluation_count == 1  # alias does not count as a real evaluation
        assert {k: v for k, v in first.items() if k != "phenotype_cache_hit"} == {
            k: v for k, v in second.items() if k != "phenotype_cache_hit"
        }
        assert second["phenotype_cache_hit"] is True
        alias_candidate_id = evaluator.candidate_id_for_code(genome_to_code(genome_b))
        assert alias_candidate_id != source_candidate_id
        alias_payload = backend.store.find_candidate_result(alias_candidate_id)
        assert alias_payload["phenotype_cache_hit"] is True
        assert alias_payload["phenotype_cache_source_candidate_id"] == source_candidate_id
        assert alias_payload["candidate"]["candidate_id"] == alias_candidate_id
        assert alias_payload["candidate"]["genome"] == genome_b
        assert alias_payload["metrics"]["openevolve_phenotype_hash"]

        # A fresh evaluator over the same store also dedups (resume path).
        resumed = MegaOpenEvolveEvaluator(backend)
        resumed.evaluate(genome_to_code(genome_b))
        assert backend.calls == 1
    print("✅ evaluator phenotype cache")


def main():
    test_genome_round_trip()
    test_fitness_mapping()
    test_open_evolve_accepts_mega_genome_seed()
    test_llm_mutator_path()
    test_llm_mutator_patch_path()
    test_llm_mutator_edit_audit()
    test_llm_mutator_noop_retry()
    test_llm_mutator_noop_retry_exhausted()
    test_llm_mutator_numeric_jitter()
    test_llm_mutator_fallback_to_rule()
    test_fixed_operator_mutator()
    test_hybrid_mcts_policy_state()
    test_hybrid_mcts_reward_protects_diversity()
    test_multi_objective_candidate_report()
    test_genome_phenotype_hash_excludes_lineage_metadata()
    test_evaluator_phenotype_cache()
    print("✅ MegaPersona OpenEvolve adapter tests passed")


if __name__ == "__main__":
    main()
