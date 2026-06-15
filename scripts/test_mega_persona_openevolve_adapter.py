"""Smoke tests for the MegaPersona OpenEvolve adapter."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona.evolution import default_genome
from src.mega_persona.openevolve_adapter import (
    MegaGenomeMutator,
    genome_from_code,
    genome_to_code,
    open_evolve_fitness_from_payload,
)
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
            "coverage": 0.20 + bonus + self.count * 0.001,
            "convex_hull": 0.10,
            "avg_dist": 0.70,
            "min_dist": 0.50,
            "dispersion": 0.60,
            "kl_divergence": 0.90,
            "mega_fitness": 0.20 + bonus,
        }


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
            "validation_shadow_alignment.mean": 0.80,
            "schema_fitness.mean": 0.50,
            "slot_coverage.mean": 0.60,
            "near_duplicate_rate.mean": 0.10,
        },
    }
    mapped = open_evolve_fitness_from_payload(payload)
    assert mapped["coverage"] == 0.25
    assert mapped["convex_hull"] == 0.30
    assert mapped["kl_divergence"] == 0.90
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


def main():
    test_genome_round_trip()
    test_fitness_mapping()
    test_open_evolve_accepts_mega_genome_seed()
    print("✅ MegaPersona OpenEvolve adapter tests passed")


if __name__ == "__main__":
    main()
