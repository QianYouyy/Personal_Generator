"""Smoke tests for durable MegaPersona evolution."""

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_mega_persona_generator import MockMegaPersonaLLM
from src.mega_persona import (
    MegaEvolutionConfig,
    MegaPersonaEvolver,
    build_run_manifest,
    default_genome,
    prompt_addendum_from_genome,
)


def test_evolution_persistence_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "evolution"
        config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=0,
            population_size=3,
            children_per_generation=2,
            elite_count=1,
            shadow_surveys=2,
            items_per_shadow_survey=6,
            random_seed=9,
        )
        first = MegaPersonaEvolver(config=config, output_dir=output_dir).run()
        assert first.fitness is not None
        assert (output_dir / "checkpoint.json").exists()
        assert (output_dir / "final_summary.json").exists()
        evals_after_first = sorted((output_dir / "evaluations").glob("eval_*"))
        assert len(evals_after_first) == 3

        checkpoint = json.loads((output_dir / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["evaluation_count"] == 3

        resume_config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=1,
            population_size=3,
            children_per_generation=2,
            elite_count=1,
            shadow_surveys=2,
            items_per_shadow_survey=6,
            random_seed=9,
        )
        second = MegaPersonaEvolver(
            config=resume_config,
            output_dir=output_dir,
            resume=True,
        ).run()
        assert second.fitness is not None
        evals_after_resume = sorted((output_dir / "evaluations").glob("eval_*"))
        assert len(evals_after_resume) > len(evals_after_first)

        final_summary = json.loads((output_dir / "final_summary.json").read_text(encoding="utf-8"))
        assert final_summary["best"]["fitness"] is not None
        result_path = next((output_dir / "evaluations").glob("eval_*/*"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        seed_result = result["per_seed"][0]
        assert "train_shadow_behavior" in seed_result
        assert "heldout_shadow_behavior" in seed_result
        assert "heldout_behavior_diversity" in seed_result
        assert "heldout_shadow_alignment.mean" in result["metrics"]


def test_resume_rejects_config_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "evolution"
        config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=0,
            population_size=3,
            children_per_generation=2,
            elite_count=1,
            shadow_surveys=2,
            items_per_shadow_survey=6,
        )
        MegaPersonaEvolver(config=config, output_dir=output_dir).run()

        mismatched = MegaEvolutionConfig(
            n=5,
            seeds=(17,),
            generations=1,
            population_size=3,
            children_per_generation=2,
            elite_count=1,
            shadow_surveys=2,
            items_per_shadow_survey=6,
        )
        try:
            MegaPersonaEvolver(config=mismatched, output_dir=output_dir, resume=True)
        except ValueError as exc:
            assert "Resume config does not match checkpoint" in str(exc)
        else:
            raise AssertionError("Expected resume config mismatch to fail")


def test_parallel_evolution_and_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "parallel_evolution"
        config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=1,
            population_size=4,
            children_per_generation=2,
            elite_count=1,
            shadow_surveys=2,
            heldout_shadow_surveys=2,
            items_per_shadow_survey=6,
            max_workers=2,
        )
        evolver = MegaPersonaEvolver(config=config, output_dir=output_dir)
        evolver.store.write_manifest(
            build_run_manifest(config, argv=["test"], resume=False, model_key=None)
        )
        best = evolver.run()
        assert best.fitness is not None
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "checkpoint.json").exists()
        assert len(list((output_dir / "evaluations").glob("eval_*"))) >= 4

        resume_config = MegaEvolutionConfig(
            n=4,
            seeds=(17,),
            generations=2,
            population_size=4,
            children_per_generation=2,
            elite_count=1,
            shadow_surveys=2,
            heldout_shadow_surveys=2,
            items_per_shadow_survey=6,
            max_workers=1,
        )
        resumed = MegaPersonaEvolver(
            config=resume_config,
            output_dir=output_dir,
            resume=True,
        ).run()
        assert resumed.fitness is not None


def test_prompt_addendum_from_genome():
    genome = default_genome()
    genome["prompt_profile"] = {
        "mechanism_focus": "motivational",
        "tension_level": "high",
        "specificity": "behavioral",
        "anti_stereotype": "counterexample",
    }
    addendum = prompt_addendum_from_genome(genome)
    assert "motives" in addendum
    assert "two interacting tensions" in addendum
    assert "behaviorally testable" in addendum
    assert "counter-stereotypical" in addendum


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
            heldout_shadow_surveys=1,
            items_per_shadow_survey=6,
        )
        best = MegaPersonaEvolver(
            config=config,
            output_dir=output_dir,
            llm_client=llm,
        ).run()
        assert best.fitness is not None
        assert llm.calls
        assert "Evolved generation policy addendum" in llm.calls[0]["system_prompt"]


def main():
    test_evolution_persistence_and_resume()
    test_resume_rejects_config_mismatch()
    test_parallel_evolution_and_manifest()
    test_prompt_addendum_from_genome()
    test_llm_mode_uses_prompt_genome()
    print("MegaPersona evolution tests passed.")


if __name__ == "__main__":
    main()
