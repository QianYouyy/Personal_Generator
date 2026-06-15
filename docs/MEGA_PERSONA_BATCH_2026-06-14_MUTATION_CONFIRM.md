# MegaPersona Mutation Confirmation - 2026-06-14

## Purpose

This run repeats the mutation diagnostic setup with two seeds (`17,23`) to check
whether the previous one-seed positive signal is stable.

## Run

- Output directory: `data/results/mega_persona_mutation_confirm_20260614`
- Mode: LLM persona generator + LLM shadow simulator
- `n`: 6 personas per seed
- Seeds: `17,23`
- Generations: 1
- Population size: 6
- Children per generation: 4
- Elite count: 2
- Train/validation/test shadow surveys: 3 / 2 / 2
- Items per shadow survey: 8

## Candidate Results

| Rank | Candidate | Gen | Parent | Mutation | Operator | Fitness | Schema | Val Cov | Val Align |
|---:|---|---:|---|---|---|---:|---:|---:|---:|
| 1 | `candidate_0000_5f68df59` | 0 | `candidate_baseline` | `mixed` | `op16_failure_modes` | 0.2086 | 0.3729 | 0.2760 | 0.7537 |
| 2 | `candidate_0001_2e7bf7fe` | 1 | `candidate_0000_5f68df59` | `prompt_only` | `op20_validation_guardrail` | 0.2023 | 0.3691 | 0.2695 | 0.7275 |
| 3 | `candidate_0001_c2ab8c7b` | 1 | `candidate_0000_5f68df59` | `mixed` | `op03_behavioral_prediction` | 0.2021 | 0.3636 | 0.2715 | 0.7480 |
| 4 | `candidate_baseline` | 0 | - | - | - | 0.1971 | 0.3550 | 0.2470 | 0.7797 |
| 5 | `candidate_0001_07d30f68` | 1 | `candidate_baseline` | `numeric_only` | - | 0.1961 | 0.3492 | 0.2530 | 0.7936 |
| 6 | `candidate_0001_0fbf04c7` | 1 | `candidate_baseline` | `operator_only` | `op01_axis_orthogonality` | 0.1930 | 0.3406 | 0.2545 | 0.7982 |
| 7 | `candidate_0000_b6f6d92b` | 0 | `candidate_baseline` | `mixed` | `op20_validation_guardrail` | 0.1921 | 0.3348 | 0.2825 | 0.7890 |
| 8 | `candidate_0000_5cef094f` | 0 | `candidate_baseline` | `mixed` | `op10_mid_axis_texture` | 0.1853 | 0.3423 | 0.2215 | 0.7740 |
| 9 | `candidate_0000_f8dead0a` | 0 | `candidate_baseline` | `mixed` | `op09_low_axis_fidelity` | 0.1814 | 0.3257 | 0.2485 | 0.7855 |
| 10 | `candidate_0000_aba7dbcf` | 0 | `candidate_baseline` | `mixed` | `op05_anti_stereotype` | 0.1800 | 0.3297 | 0.2270 | 0.7803 |

## Sealed Test

The selected best candidate was `candidate_0000_5f68df59`.

- Test alignment: 0.7730
- Test behavior coverage: 0.2385
- Test behavior average distance: 0.4034

## Interpretation

This two-seed confirmation did **not** replicate the previous one-seed result
where a generation-1 child became the best candidate. The best candidate is
again a generation-0 initial mutation:

- Best generation-0: `candidate_0000_5f68df59`, fitness 0.2086
- Best generation-1: `candidate_0001_2e7bf7fe`, fitness 0.2023
- Gap: -0.0063

However, the generation-1 children of the best generation-0 candidate remained
close:

- `prompt_only + op20_validation_guardrail`: 0.2023
- `mixed + op03_behavioral_prediction`: 0.2021

This suggests the operator system is not destructive, but it is not yet
reliably improving over strong initialization.

## Reliability

- All evaluated seeds completed successfully.
- All candidates had validity rate 1.0.
- No timeout/API failures appeared in the final log.
- Only a few validation revisions were needed, all recovered.

## Scientific Conclusion

Current evidence supports:

1. The durable evolution pipeline is working.
2. The operator/mutation diagnostics are now measurable.
3. `op16_failure_modes` remains a strong initialization operator.
4. Generation-1 improvement is possible but not stable across seeds.

Current evidence does **not** yet support:

1. A robust claim that Open-Evolve reliably improves over initialization.
2. A claim that `op03_behavioral_prediction` is consistently superior.

## Next Step

Before increasing run size, add a targeted ablation that repeatedly applies the
most promising operators (`op16`, `op20`, `op03`) from the same strong parent
across multiple seeds. This will separate operator quality from parent quality.
