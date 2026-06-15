# MegaPersona Mutation Diagnostic - 2026-06-14

## Purpose

This run tests the diagnostic mutation modes added after the operator smoke
test. The goal is to see whether generation-1 children can beat generation-0
candidates, and which mutation mode appears most promising.

## Run

- Output directory: `data/results/mega_persona_mutation_diagnostic_20260614`
- Mode: LLM persona generator + LLM shadow simulator
- `n`: 6 personas per seed
- Seeds: `17`
- Generations: 1
- Population size: 6
- Children per generation: 4
- Elite count: 2
- Train/validation/test shadow surveys: 3 / 2 / 2
- Items per shadow survey: 8

## Candidate Results

| Rank | Candidate | Gen | Parent | Mutation | Operator | Fitness | Schema | Val Cov | Val Align |
|---:|---|---:|---|---|---|---:|---:|---:|---:|
| 1 | `candidate_0001_d61f7cb3` | 1 | `candidate_0000_ffe3d57c` | `mixed` | `op03_behavioral_prediction` | 0.2091 | 0.3719 | 0.2590 | 0.7864 |
| 2 | `candidate_0000_ffe3d57c` | 0 | `candidate_baseline` | `mixed` | `op16_failure_modes` | 0.2061 | 0.3724 | 0.2650 | 0.7498 |
| 3 | `candidate_0001_215c7d06` | 1 | `candidate_0000_ffe3d57c` | `prompt_only` | `op20_validation_guardrail` | 0.2044 | 0.3697 | 0.2560 | 0.7609 |
| 4 | `candidate_0001_4f571dbb` | 1 | `candidate_baseline` | `numeric_only` | - | 0.1877 | 0.3418 | 0.2380 | 0.7742 |
| 5 | `candidate_0001_4a1cbb29` | 1 | `candidate_baseline` | `operator_only` | `op01_axis_orthogonality` | 0.1843 | 0.3350 | 0.2290 | 0.7909 |
| 6 | `candidate_baseline` | 0 | - | - | - | 0.1773 | 0.3217 | 0.2510 | 0.7618 |
| 7 | `candidate_0000_37f18239` | 0 | `candidate_baseline` | `mixed` | `op05_anti_stereotype` | 0.1762 | 0.3227 | 0.2360 | 0.7667 |
| 8 | `candidate_0000_e99ffbc1` | 0 | `candidate_baseline` | `mixed` | `op09_low_axis_fidelity` | 0.1743 | 0.3204 | 0.2320 | 0.7666 |
| 9 | `candidate_0000_60aea021` | 0 | `candidate_baseline` | `mixed` | `op10_mid_axis_texture` | 0.1653 | 0.3013 | 0.2130 | 0.8092 |
| 10 | `candidate_0000_a8e46466` | 0 | `candidate_baseline` | `mixed` | `op20_validation_guardrail` | 0.1458 | 0.2781 | 0.1940 | 0.7560 |

## Sealed Test

The selected best candidate was `candidate_0001_d61f7cb3`.

- Test alignment: 0.8237
- Test behavior coverage: 0.2340
- Test behavior average distance: 0.4164

## Interpretation

This is the first small run where a generation-1 child beats the best
generation-0 candidate.

- Best generation-0 candidate: `candidate_0000_ffe3d57c`, fitness 0.2061
- Best generation-1 child: `candidate_0001_d61f7cb3`, fitness 0.2091
- Absolute gain: +0.0030

The gain is small, but it is directionally important because it comes from a
child of the best generation-0 candidate. The winning child used:

- Mutation mode: `mixed`
- Operator: `op03_behavioral_prediction`

The near-best child used:

- Mutation mode: `prompt_only`
- Operator: `op20_validation_guardrail`
- Fitness: 0.2044

This suggests that prompt/operator changes are not purely harmful. However,
`numeric_only` and `operator_only` children from baseline did not catch the best
generation-0 candidate in this run.

## Scientific Caution

This is still not enough to claim robust Open-Evolve improvement:

1. Only one seed was used.
2. The improvement is small (+0.0030).
3. The best generation-0 candidate was already a strong initialization sample.
4. Operator effects are confounded with parent quality; the winning `mixed`
   child inherited from the best generation-0 candidate.

The result is best interpreted as a positive pilot signal.

## Next Step

Run a two-seed confirmation with the same diagnostic mutation modes:

- `n=6`
- `seeds=17,23`
- `generations=1`
- `population-size=6`
- `children-per-generation=4`

Success criterion for moving toward a larger formal run:

- at least one generation-1 child beats the best generation-0 candidate on mean
  validation fitness
- the winning child does not lose sealed-test coverage by more than 0.03
  compared with the current best pilot runs
