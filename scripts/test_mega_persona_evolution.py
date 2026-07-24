"""Smoke tests for MegaPersona's OpenEvolve integration."""

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_mega_persona_generator import MockMegaPersonaLLM
from src.mega_persona import (
    EVOLUTION_PROMPT_OPERATORS,
    MegaEvolutionCandidate,
    MegaEvolutionConfig,
    MegaPersonaOpenEvolveRunner,
    RuleBasedMegaPersonaBuilder,
    blueprint_from_slot,
    candidate_slots,
    default_genome,
    default_genome_v4,
    mutate_genome,
    prompt_addendum_from_genome,
    validate_mega_persona,
)
from src.mega_persona.slots import schema_binding_for_genome
from src.mega_persona.slots import build_adaptive_constraints
from src.mega_persona.generator import _blueprint_hard_constraints
from src.mega_persona.shadow_survey import build_initial_shadow_surveys, score_shadow_survey
from scripts.run_mega_persona_operator_ablation import (
    build_ablation_candidates,
    summarize_ablation_results,
)


class _MockSimulatorLLM:
    """Returns neutral Likert responses for every simulate_persona call."""

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.0,
                 max_tokens: int = 1500, **kwargs) -> str:
        import json
        # Extract item_ids from the prompt
        lines = prompt.split("\n")
        ids = []
        for line in lines:
            if line.strip().startswith('"') and '":' in line:
                item_id = line.strip().split('"')[1] if line.strip().startswith('"') else None
                if item_id:
                    ids.append(item_id)
        if not ids:
            return '{"unknown": 3}'
        return json.dumps({iid: 3 for iid in ids})


_MOCK_SIM = _MockSimulatorLLM()


def test_openevolve_persistence_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "evolution"
        config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=1,
            population_size=2,
            children_per_generation=1,
            elite_count=1,
            shadow_surveys=2,
            items_per_shadow_survey=6,
            random_seed=9,
        )
        first = MegaPersonaOpenEvolveRunner(
            config=config,
            output_dir=output_dir,
            simulator_llm_client=_MOCK_SIM,
            children_per_island=1,
        ).run()
        assert first.fitness is not None
        mega_eval_dir = output_dir / "mega_eval"
        assert (output_dir / "open_evolve" / "checkpoint.json").exists()
        assert (output_dir / "final_summary.json").exists()
        assert (mega_eval_dir / "final_summary.json").exists()
        assert (mega_eval_dir / "final_test_report.json").exists()
        assert (mega_eval_dir / "shadow_surveys" / "train.json").exists()
        assert (mega_eval_dir / "shadow_surveys" / "validation.json").exists()
        assert (mega_eval_dir / "shadow_surveys" / "test.json").exists()
        assert (mega_eval_dir / "shadow_surveys" / "hashes.json").exists()
        evals_after_first = sorted((mega_eval_dir / "evaluations").glob("eval_*"))
        assert len(evals_after_first) >= 2

        checkpoint = json.loads((mega_eval_dir / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["evaluation_count"] >= 2
        assert set(checkpoint["survey_hashes"]) == {"train", "validation", "test"}

        resume_config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=2,
            population_size=2,
            children_per_generation=1,
            elite_count=1,
            shadow_surveys=2,
            items_per_shadow_survey=6,
            random_seed=9,
        )
        second = MegaPersonaOpenEvolveRunner(
            config=resume_config,
            output_dir=output_dir,
            resume=True,
            simulator_llm_client=_MOCK_SIM,
            children_per_island=1,
        ).run()
        assert second.fitness is not None
        evals_after_resume = sorted((mega_eval_dir / "evaluations").glob("eval_*"))
        assert len(evals_after_resume) > len(evals_after_first)

        final_summary = json.loads((output_dir / "final_summary.json").read_text(encoding="utf-8"))
        assert final_summary["best"]["fitness"] is not None
        candidate_payloads = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (mega_eval_dir / "candidates").glob("*.json")
        ]
        assert any(payload["generation"] > 0 for payload in candidate_payloads)
        assert final_summary["best"]["generation"] >= 0
        result_path = next((mega_eval_dir / "evaluations").glob("eval_*/*"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        seed_result = result["per_seed"][0]
        assert "train_shadow_behavior" in seed_result
        assert "validation_shadow_behavior" in seed_result
        assert "validation_behavior_diversity" in seed_result
        assert "internal_consistency" in seed_result
        assert 0.0 <= seed_result["internal_consistency"]["mean_score"] <= 1.0
        assert "test_shadow_behavior" not in seed_result
        assert "test_behavior_diversity" not in seed_result
        assert "shadow_survey_hashes" in seed_result
        assert "internal_consistency.mean" in result["metrics"]
        assert "axis_alignment.mean" in result["metrics"]
        assert "validation_shadow_alignment.mean" in result["metrics"]
        assert "test_shadow_alignment.mean" not in result["metrics"]
        test_report = json.loads((mega_eval_dir / "final_test_report.json").read_text(encoding="utf-8"))
        assert test_report["test_used_for_selection"] is False
        assert "test_schema_fitness.mean" in test_report["metrics"]
        assert "test_internal_consistency.mean" in test_report["metrics"]
        assert "test_axis_alignment.mean" in test_report["metrics"]
        assert "test_shadow_alignment.mean" in test_report["metrics"]
        assert "test_behavior_coverage.mean" in test_report["metrics"]
        assert "test_behavior_balanced_diversity.mean" in test_report["metrics"]


def test_resume_rejects_config_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "evolution"
        config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=1,
            population_size=2,
            children_per_generation=1,
            elite_count=1,
            shadow_surveys=2,
            items_per_shadow_survey=6,
        )
        MegaPersonaOpenEvolveRunner(
            config=config,
            output_dir=output_dir,
            simulator_llm_client=_MOCK_SIM,
            children_per_island=1,
        ).run()

        mismatched = MegaEvolutionConfig(
            n=5,
            seeds=(17,),
            generations=2,
            population_size=2,
            children_per_generation=1,
            elite_count=1,
            shadow_surveys=2,
            items_per_shadow_survey=6,
        )
        try:
            MegaPersonaOpenEvolveRunner(
                config=mismatched,
                output_dir=output_dir,
                resume=True,
                simulator_llm_client=_MOCK_SIM,
                children_per_island=1,
            )
        except ValueError as exc:
            assert "Resume config does not match checkpoint" in str(exc)
        else:
            raise AssertionError("Expected resume config mismatch to fail")


def test_openevolve_manifest_and_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "parallel_evolution"
        config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=1,
            population_size=2,
            children_per_generation=1,
            elite_count=1,
            shadow_surveys=2,
            validation_shadow_surveys=2,
            test_shadow_surveys=1,
            items_per_shadow_survey=6,
            max_workers=2,
            shadow_max_workers=2,
        )
        best = MegaPersonaOpenEvolveRunner(
            config=config,
            output_dir=output_dir,
            simulator_llm_client=_MOCK_SIM,
            children_per_island=1,
        ).run(argv=["test"], model_key=None)
        assert best.fitness is not None
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "open_evolve" / "checkpoint.json").exists()
        assert len(list((output_dir / "mega_eval" / "evaluations").glob("eval_*"))) >= 2

        resume_config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=2,
            population_size=2,
            children_per_generation=1,
            elite_count=1,
            shadow_surveys=2,
            validation_shadow_surveys=2,
            test_shadow_surveys=1,
            items_per_shadow_survey=6,
            max_workers=1,
            shadow_max_workers=1,
        )
        resumed = MegaPersonaOpenEvolveRunner(
            config=resume_config,
            output_dir=output_dir,
            resume=True,
            simulator_llm_client=_MOCK_SIM,
            children_per_island=1,
        ).run()
        assert resumed.fitness is not None


def test_prompt_addendum_from_genome():
    assert len(EVOLUTION_PROMPT_OPERATORS) >= 16
    assert len({operator["id"] for operator in EVOLUTION_PROMPT_OPERATORS}) == len(EVOLUTION_PROMPT_OPERATORS)
    operator_instructions = " ".join(operator["instruction"] for operator in EVOLUTION_PROMPT_OPERATORS)
    assert "deadline" in operator_instructions
    assert "peer pressure" in operator_instructions
    assert "failure cycle" in operator_instructions
    assert "field lengths" in operator_instructions
    assert "recovery latency" in operator_instructions
    assert "support networks" in operator_instructions
    assert "external approval" in operator_instructions
    assert "Genome v3 blueprint" in operator_instructions
    op16 = next(operator for operator in EVOLUTION_PROMPT_OPERATORS if operator["id"] == "op16_v3_blueprint_binding")
    assert "blueprint_policy" in op16
    assert "axis_expression_policy" in op16
    assert "cross_agent_binding_policy" in op16
    assert "behavior_prediction_policy" in op16
    assert "critic_policy" in op16

    genome = default_genome()
    genome["prompt_profile"] = {
        "mechanism_focus": "motivational",
        "tension_level": "high",
        "specificity": "behavioral",
        "anti_stereotype": "counterexample",
        "axis_binding": "orthogonal",
        "coverage_strategy": "edge_cases",
        "behavioral_signal": "mixed_evidence",
    }
    addendum = prompt_addendum_from_genome(genome)
    assert "Respect all schema length limits" in addendum
    assert "motives" in addendum
    assert "two interacting tensions" in addendum
    assert "behaviorally testable" in addendum
    assert "counter-stereotypical" in addendum
    assert "partially independent" in addendum
    assert "edge cases" in addendum
    assert "self-image and behavior evidence" in addendum


def test_genome_v3_blueprint_from_slot():
    genome = default_genome()
    assert genome["genome_version"] == 3
    slot = candidate_slots(genome, n=1, seed=17)[0]
    blueprint = blueprint_from_slot(genome, slot)
    assert blueprint["blueprint_version"] == 3
    assert blueprint["axis_expression_plan"]
    assert set(blueprint["axis_expression_plan"]) == set(slot.target_axes)
    assert "ambiguous_task" in blueprint["behavior_prediction_profile"]
    assert blueprint["cross_agent_binding"]
    assert blueprint["critic_checks"]


def test_genome_v4_structured_blueprint():
    genome = default_genome_v4()
    assert genome["genome_version"] == 4
    assert "prompt_profile" not in genome
    assert "blueprint_policy" not in genome
    slot = candidate_slots(genome, n=1, seed=17)[0]
    blueprint = blueprint_from_slot(genome, slot)
    assert blueprint["blueprint_version"] == 4
    assert set(blueprint["axis_expression_plan"]) == set(slot.target_axes)
    assert set(blueprint["behavior_prediction_profile"]) == {
        "ambiguous_task",
        "peer_pressure",
        "failure_feedback",
        "deadline",
    }
    assert blueprint["structured_program"]["probe_assignment"] == genome["probe_assignment"]
    assert "Genome v4 generation blueprint" in prompt_addendum_from_genome(genome)
    hard_constraints = _blueprint_hard_constraints(blueprint, "cognition")
    assert "signal strength=" in hard_constraints
    assert "Blueprint critic requirement" in hard_constraints


def test_genome_v4_operators_change_one_module():
    import numpy as np

    base = default_genome_v4()
    expected_modules = {
        "op22_v4_probe_rewire": "probe_assignment",
        "op23_v4_signal_calibrate": "axis_realization",
        "op24_v4_interaction_rewire": "interaction_mode",
        "op25_v4_echo_graph_rewire": "echo_graph",
        "op26_v4_context_diversify": "context_modulation",
        "op27_v4_repair_calibrate": "repair_control",
    }
    ignored = {"last_evolution_operator", "last_mutation", "openevolve_mutation"}
    for index, (operator_id, module) in enumerate(expected_modules.items()):
        child = mutate_genome(
            base,
            np.random.default_rng(100 + index),
            0.12,
            operator_id=operator_id,
        )
        changed = {
            key
            for key in set(base) | set(child)
            if key not in ignored and base.get(key) != child.get(key)
        }
        assert changed == {module}, (operator_id, changed)
        assert child["last_mutation"]["module"] == module
        assert child["quota_weights"] == base["quota_weights"]
        assert child["axis_bias"] == base["axis_bias"]
        assert child["axis_stretch"] == base["axis_stretch"]


def test_genome_v4_runner_manifest_and_seed():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "v4_evolution"
        config = MegaEvolutionConfig(
            n=2,
            seeds=(17,),
            generations=1,
            population_size=2,
            children_per_generation=1,
            elite_count=1,
            shadow_surveys=1,
            validation_shadow_surveys=1,
            test_shadow_surveys=1,
            items_per_shadow_survey=6,
            candidate_evaluation_repeats=2,
            elite_confirmation_repeats=3,
        )
        best = MegaPersonaOpenEvolveRunner(
            config=config,
            output_dir=output_dir,
            simulator_llm_client=_MOCK_SIM,
            children_per_island=1,
            genome_version=4,
            operator_family="v4",
            fixed_operator_id="op22_v4_probe_rewire",
        ).run()
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert best.genome["genome_version"] == 4
        assert manifest["genome_version"] == 4
        assert manifest["operator_family"] == "v4"
        assert manifest["config"]["candidate_evaluation_repeats"] == 2
        assert manifest["config"]["elite_confirmation_repeats"] == 3
        assert set(manifest["operator_pool"]) == {
            "op22_v4_probe_rewire",
            "op23_v4_signal_calibrate",
            "op24_v4_interaction_rewire",
            "op25_v4_echo_graph_rewire",
            "op26_v4_context_diversify",
            "op27_v4_repair_calibrate",
        }
        checkpoint = json.loads(
            (output_dir / "mega_eval" / "checkpoint.json").read_text(encoding="utf-8")
        )
        assert all(
            candidate["genome"]["genome_version"] == 4
            for candidate in checkpoint["population"]
        )
        test_report = json.loads(
            (output_dir / "mega_eval" / "final_test_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert test_report["evaluation_repeats"] == 3
        assert {item["evaluation_repeat"] for item in test_report["per_seed"]} == {1, 2, 3}


def test_mutation_records_evolution_operator():
    import numpy as np

    mutated = mutate_genome(default_genome(), np.random.default_rng(3), 0.2)
    operator = mutated["last_evolution_operator"]
    assert operator["id"] in {item["id"] for item in EVOLUTION_PROMPT_OPERATORS}
    addendum = prompt_addendum_from_genome(mutated)
    # Operator instructions guide the mutator only; they must not leak into
    # the persona generation prompt.
    assert "Selected evolution operator" not in addendum
    assert operator["instruction"] not in addendum


def test_mutation_modes_are_diagnostic():
    import numpy as np

    base = default_genome()
    prompt_only = mutate_genome(
        base,
        np.random.default_rng(4),
        0.2,
        mutation_mode="prompt_only",
        operator_id="op05_failure_recovery_cycle",
    )
    assert prompt_only["quota_weights"] == base["quota_weights"]
    assert prompt_only["axis_bias"] == base["axis_bias"]
    assert prompt_only["axis_stretch"] == base["axis_stretch"]
    assert prompt_only["last_mutation"]["mode"] == "prompt_only"
    assert prompt_only["last_evolution_operator"]["id"] == "op05_failure_recovery_cycle"

    numeric_only = mutate_genome(
        base,
        np.random.default_rng(5),
        0.2,
        mutation_mode="numeric_only",
    )
    assert numeric_only["last_mutation"]["mode"] == "numeric_only"
    assert numeric_only["last_evolution_operator"] is None
    assert numeric_only["prompt_profile"] == base["prompt_profile"]
    assert numeric_only["axis_bias"] != base["axis_bias"]


def test_schema_bound_genome_can_rename_axes():
    import numpy as np

    genome = default_genome()
    genome["schema_binding"]["axis_names"] = [
        "thinking_depth",
        "motivation_drive",
        "recovery_control",
    ]
    genome["schema_binding"]["axis_roles"] = {
        "cognitive_core": "thinking_depth",
        "motivation_core": "motivation_drive",
        "regulation_core": "recovery_control",
    }
    genome["axis_bias"] = {
        "thinking_depth": 0.0,
        "motivation_drive": 0.0,
        "recovery_control": 0.0,
    }
    genome["axis_stretch"] = {
        "thinking_depth": 1.0,
        "motivation_drive": 1.0,
        "recovery_control": 1.0,
    }
    mutated = mutate_genome(
        genome,
        np.random.default_rng(12),
        0.15,
        mutation_mode="mixed",
        operator_id="op13_autonomy_pressure_test",
    )
    binding = schema_binding_for_genome(mutated)
    assert tuple(binding["axis_names"]) == (
        "thinking_depth",
        "motivation_drive",
        "recovery_control",
    )
    assert set(mutated["axis_bias"]) == {"thinking_depth", "motivation_drive", "recovery_control"}
    assert set(mutated["axis_stretch"]) == {"thinking_depth", "motivation_drive", "recovery_control"}
    slots = candidate_slots(mutated, n=2, seed=17)
    assert set(slots[0].target_axes) == {"thinking_depth", "motivation_drive", "recovery_control"}
    hints = build_adaptive_constraints(
        {
            "thinking_depth": 0.4,
            "motivation_drive": 0.85,
            "recovery_control": 0.55,
        },
        {"primary_drive": "security"},
        axis_roles=binding["axis_roles"],
    )
    assert any("security or recognition needs" in hint for hint in hints)


def test_schema_bound_shadow_surveys_remap_axes():
    binding = {
        "axis_names": ["thinking_depth", "motivation_drive", "recovery_control"],
        "axis_roles": {
            "cognitive_core": "thinking_depth",
            "motivation_core": "motivation_drive",
            "regulation_core": "recovery_control",
        },
        "quota_buckets": default_genome()["schema_binding"]["quota_buckets"],
    }
    surveys = build_initial_shadow_surveys(
        num_surveys=1,
        items_per_survey=8,
        seed=17,
        schema_binding=binding,
    )
    survey = surveys[0]
    assert survey.axis_names == ("thinking_depth", "motivation_drive", "recovery_control")
    assert all(
        set(item.axis_weights).issubset(set(survey.axis_names))
        for item in survey.items
    )
    responses = {item.item_id: 3 for item in survey.items}
    scores = score_shadow_survey(survey, responses)
    assert "axis.thinking_depth" in scores
    assert "axis.motivation_drive" in scores
    assert "axis.recovery_control" in scores


def test_schema_bound_builder_and_validator_follow_renamed_axes():
    genome = default_genome()
    genome["schema_binding"]["axis_names"] = [
        "thinking_depth",
        "motivation_drive",
        "recovery_control",
    ]
    genome["schema_binding"]["axis_roles"] = {
        "cognitive_core": "thinking_depth",
        "motivation_core": "motivation_drive",
        "regulation_core": "recovery_control",
    }
    genome["axis_bias"] = {
        "thinking_depth": 0.0,
        "motivation_drive": 0.0,
        "recovery_control": 0.0,
    }
    genome["axis_stretch"] = {
        "thinking_depth": 1.0,
        "motivation_drive": 1.0,
        "recovery_control": 1.0,
    }
    slot = candidate_slots(genome, n=1, seed=23)[0]
    persona = RuleBasedMegaPersonaBuilder().build(slot)
    report = validate_mega_persona(
        persona,
        axis_names=tuple(genome["schema_binding"]["axis_names"]),
        axis_roles=genome["schema_binding"]["axis_roles"],
    )
    assert report.schema_valid is True
    derived_axes = persona.primary_axes(
        axis_names=tuple(genome["schema_binding"]["axis_names"]),
        axis_roles=genome["schema_binding"]["axis_roles"],
    )
    assert set(derived_axes) == {"thinking_depth", "motivation_drive", "recovery_control"}


def test_operator_ablation_candidate_design():
    parent = {
        "candidate_id": "candidate_parent",
        "genome": default_genome(),
    }
    candidates = build_ablation_candidates(
        parent_candidate=parent,
        operators=["op01_axis_decoupling", "op02_behavioral_evidence"],
        mutation_modes=["parent_replay", "prompt_only", "operator_only", "mixed", "numeric_only"],
        replicates=2,
        mutation_scale=0.08,
        random_seed=11,
    )
    # parent replay and numeric_only are one per replicate; the
    # three operator-bound modes are crossed with operators and replicates.
    assert len(candidates) == 2 + 2 + (2 * 3 * 2)
    assert candidates[0].genome["last_mutation"]["mode"] == "parent_replay"
    assert candidates[0].genome["last_evolution_operator"] is None
    assert {candidate.parent_id for candidate in candidates} == {"candidate_parent"}

    modes = [candidate.genome["last_mutation"]["mode"] for candidate in candidates]
    assert modes.count("parent_replay") == 2
    assert modes.count("numeric_only") == 2
    assert modes.count("prompt_only") == 4
    assert modes.count("operator_only") == 4
    assert modes.count("mixed") == 4

    operator_ids = {
        candidate.genome["last_evolution_operator"]["id"]
        for candidate in candidates
        if isinstance(candidate.genome.get("last_evolution_operator"), dict)
    }
    assert operator_ids == {"op01_axis_decoupling", "op02_behavioral_evidence"}


def test_operator_ablation_summary_groups_against_parent():
    parent = MegaEvolutionCandidate(
        candidate_id="ablation_0000_parent_replay",
        genome={**default_genome(), "last_mutation": {"mode": "parent_replay", "scale": 0.0}},
        parent_id="candidate_parent",
        fitness=0.2,
        evaluated=True,
    )
    child_genome = mutate_genome(
        default_genome(),
        __import__("numpy").random.default_rng(2),
        0.08,
        mutation_mode="prompt_only",
        operator_id="op02_behavioral_evidence",
    )
    child = MegaEvolutionCandidate(
        candidate_id="ablation_0001_op02_behavioral_evidence_prompt_only_r01",
        genome=child_genome,
        parent_id="candidate_parent",
        fitness=0.21,
        evaluated=True,
    )
    summary = summarize_ablation_results([parent, child], parent_candidate_id="candidate_parent")
    assert summary["parent_replay_fitness"] == 0.2
    assert summary["parent_replay_n"] == 1
    assert summary["parent_replay_std"] == 0.0
    assert summary["rows"][0]["candidate_id"] == child.candidate_id
    op_group = next(group for group in summary["groups"] if group["operator_id"] == "op02_behavioral_evidence")
    assert op_group["beats_parent"] is True
    assert round(op_group["delta_vs_parent"], 6) == 0.01


def test_llm_mode_uses_prompt_genome():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "evolution_llm"
        llm = MockMegaPersonaLLM()
        config = MegaEvolutionConfig(
            n=1,
            seeds=(17,),
            generator_mode="llm",
            generations=0,
            population_size=1,
            children_per_generation=1,
            elite_count=1,
            shadow_surveys=1,
            validation_shadow_surveys=1,
            test_shadow_surveys=1,
            items_per_shadow_survey=6,
        )
        best = MegaPersonaOpenEvolveRunner(
            config=config,
            output_dir=output_dir,
            llm_client=llm,
            simulator_llm_client=_MOCK_SIM,
            children_per_island=1,
        ).run()
        assert best.fitness is not None
        assert llm.calls
        assert "Evolved generation policy addendum" in llm.calls[0]["system_prompt"]


def main():
    test_openevolve_persistence_and_resume()
    test_resume_rejects_config_mismatch()
    test_openevolve_manifest_and_artifacts()
    test_prompt_addendum_from_genome()
    test_genome_v3_blueprint_from_slot()
    test_genome_v4_structured_blueprint()
    test_genome_v4_operators_change_one_module()
    test_genome_v4_runner_manifest_and_seed()
    test_mutation_records_evolution_operator()
    test_mutation_modes_are_diagnostic()
    test_schema_bound_genome_can_rename_axes()
    test_schema_bound_shadow_surveys_remap_axes()
    test_schema_bound_builder_and_validator_follow_renamed_axes()
    test_operator_ablation_candidate_design()
    test_operator_ablation_summary_groups_against_parent()
    test_llm_mode_uses_prompt_genome()
    print("MegaPersona evolution tests passed.")


if __name__ == "__main__":
    main()
