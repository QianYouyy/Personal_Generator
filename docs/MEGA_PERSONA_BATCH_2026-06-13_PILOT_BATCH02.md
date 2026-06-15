# MegaPersona-Evolve Pilot Batch 02 Report

- Date: 2026-06-13
- Batch: `mega_persona_pilot_batch02_20260613`
- Output dir: `data/results/mega_persona_pilot_batch02_20260613`
- Status: pipeline passed; suitable as a pilot validation run, not yet a final scientific run.

## Run Setup

- generator mode: `llm`
- personas per seed: `n=8`
- seeds: `17`
- generations: `1`
- population size: `3`
- children per generation: `2`
- elite count: `1`
- train shadow surveys: `4`
- validation shadow surveys: `2`
- sealed test shadow surveys: `2`
- items per shadow survey: `8`
- max workers: `1`

This produced 5 evaluated candidates:

- 3 initial candidates in generation 0
- 2 children in generation 1

## Main Result

The previous batch failed at the generation/JSON layer. This pilot batch passed that layer.

- all 5 candidates generated `8/8` valid personas
- no candidate-level JSON failures
- no invalid final personas
- final sealed test ran successfully
- best candidate: `candidate_0000_8db98a05`
- best validation fitness: `0.2119`

Candidate summary:

| Candidate | Fitness | Personas | Validity | Schema Fitness | Validation Behavior Coverage | Validation Alignment | Slot Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `candidate_baseline` | 0.2118 | 8 | 1.0000 | 0.4010 | 0.2120 | 0.7438 | 0.4800 |
| `candidate_0000_11a0c72f` | 0.1800 | 8 | 1.0000 | 0.3367 | 0.2270 | 0.7430 | 0.4810 |
| `candidate_0000_8db98a05` | 0.2119 | 8 | 1.0000 | 0.3837 | 0.2380 | 0.7843 | 0.4710 |
| `candidate_0001_f3c87c1b` | 0.1851 | 8 | 1.0000 | 0.3455 | 0.2110 | 0.7693 | 0.5020 |
| `candidate_0001_7a461d92` | 0.2082 | 8 | 1.0000 | 0.3846 | 0.2070 | 0.7945 | 0.5020 |

Sealed test for selected best candidate:

| Metric | Value |
|---|---:|
| test shadow alignment | 0.7336 |
| test behavior coverage | 0.1710 |
| test behavior avg distance | 0.2296 |

## Interpretation

This batch validates the repaired pipeline:

1. The explicit schema contracts in the prompts solved the total generation failure seen in batch 01.
2. The Open-Evolve loop, checkpointing, frozen survey splits, validation selection, and sealed test path all worked.
3. Validation alignment is reasonably high for a pilot (`0.7843` best), and sealed test alignment remains acceptable (`0.7336`).
4. Behavior coverage is still low. Best validation behavior coverage is `0.2380`, and sealed test coverage drops to `0.1710`.

The main scientific bottleneck is now coverage/generalization, not schema validity.

## Remaining Concerns

- This pilot uses only one seed, so all reported standard deviations are `0.0`; it does not measure seed stability.
- The test set has only 2 shadow surveys with 8 items each, so sealed test estimates are noisy.
- Validation-to-test behavior coverage drops from `0.2380` to `0.1710`, suggesting possible small-sample instability.
- Fitness differences are tiny: baseline `0.2118` vs best `0.2119`. This is not enough evidence that evolution improved the generator.
- The current best is still from generation 0, not a later evolved child.

## Code Fix Applied After Analysis

One reproducibility issue was found after this pilot:

- `DiversityMetrics.coverage()` and `DiversityMetrics.dispersion()` used unseeded random reference points.
- This has been fixed by using deterministic reference points with a fixed default seed.
- Sobol-based automatic coverage-radius calibration now also uses a fixed seed.

Verification after the fix:

- `PYTHONDONTWRITEBYTECODE=1 python scripts/test_evaluator.py`
- `PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_experiment.py`
- `PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_evolution.py`
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile src/evaluator/metrics.py src/mega_persona/evolution.py src/mega_persona/generator.py src/mega_persona/prompts.py`

All passed.

## Recommendation

Run one medium pilot before the full formal experiment:

- `n=16`
- `seeds=17,23`
- `generations=3`
- `population-size=5`
- `children-per-generation=4`
- `shadow-surveys=8`
- `validation-shadow-surveys=3`
- `test-shadow-surveys=3`
- `items-per-shadow-survey=10`

Proceed to a formal run only if:

- validity remains near `1.0`
- best validation fitness clearly exceeds baseline by at least `0.03`
- behavior coverage improves above `0.30`
- validation/test alignment gap remains below `0.10`
- an evolved child beats the generation-0 candidates

