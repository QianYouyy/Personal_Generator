# MegaPersona-Evolve Medium Batch 03 Report

- Date: 2026-06-13
- Batch: `mega_persona_medium_batch03_20260613`
- Output dir: `data/results/mega_persona_medium_batch03_20260613`
- Status: medium-scale pilot completed; strong pipeline evidence, partial evolution evidence.

## Executive Summary

This batch completed successfully and is the strongest run so far.

Main conclusion:

- The MegaPersona generation pipeline is now stable at medium scale.
- Schema validity and semantic non-duplication are good.
- Behavior coverage crossed the target pilot threshold of `0.30`.
- Sealed test performance is close to validation performance, suggesting limited validation/test drift.
- However, the best candidate is still an initial generation-0 mutant, not a later evolved child. This means the run supports the pipeline and search space design, but does not yet prove that multi-generation evolution is reliably improving the generator.

## Run Configuration

Recovered from `manifest.json`:

- generator mode: `llm`
- personas per seed: `n=16`
- seeds: `17,23`
- generations: `3`
- population size: `5`
- children per generation: `4`
- elite count: `2`
- train shadow surveys: `8`
- validation shadow surveys: `3`
- sealed test shadow surveys: `3`
- items per shadow survey: `10`
- max workers: `1`

Expected evaluation size:

- per candidate: `16 personas/seed * 2 seeds = 32 personas`
- evaluated candidates: `14`
- total seed evaluations: `28`

## Best Candidate

Selected best candidate:

- candidate: `candidate_0000_8a2d6494`
- generation: `0`
- parent: `candidate_baseline`
- validation fitness: `0.2985`
- fitness std across seeds: `0.0081`

Validation metrics:

| Metric | Mean | Std |
|---|---:|---:|
| score | 0.2985 | 0.0081 |
| schema fitness | 0.5011 | 0.0064 |
| validity rate | 1.0000 | 0.0000 |
| near duplicate rate | 0.0000 | 0.0000 |
| train shadow alignment | 0.7834 | 0.0056 |
| validation shadow alignment | 0.7880 | 0.0041 |
| train behavior coverage | 0.3320 | 0.0110 |
| validation behavior coverage | 0.3320 | 0.0160 |
| slot coverage | 0.7795 | 0.0255 |

Sealed test metrics:

| Metric | Mean | Std |
|---|---:|---:|
| test shadow alignment | 0.7717 | 0.0047 |
| test behavior coverage | 0.3060 | 0.0070 |
| test behavior avg distance | 0.3019 | 0.0010 |

Validation-to-test gap:

- alignment gap: `0.7880 - 0.7717 = 0.0163`
- behavior coverage gap: `0.3320 - 0.3060 = 0.0260`

This gap is acceptable for a medium pilot.

## Candidate Ranking

| Rank | Candidate | Gen | Parent | Fitness | Personas Saved | Failed Seeds | Validity | Schema | Val Coverage | Val Alignment | Slot Coverage |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `candidate_0000_8a2d6494` | 0 | `candidate_baseline` | 0.2985 | 32 | 0 | 1.0000 | 0.5011 | 0.3320 | 0.7880 | 0.7795 |
| 2 | `candidate_0002_06a3d59f` | 2 | `candidate_0000_8a2d6494` | 0.2754 | 31 | 0 | 1.0000 | 0.4676 | 0.3395 | 0.8125 | 0.7625 |
| 3 | `candidate_0001_aced520a` | 1 | `candidate_0000_8a2d6494` | 0.2735 | 30 | 0 | 1.0000 | 0.4947 | 0.3160 | 0.7926 | 0.7800 |
| 4 | `candidate_0001_2cbae1b2` | 1 | `candidate_0000_8a2d6494` | 0.2669 | 31 | 0 | 1.0000 | 0.4784 | 0.2910 | 0.7815 | 0.7905 |
| 5 | `candidate_0003_16ed35c9` | 3 | `candidate_0002_06a3d59f` | 0.2394 | 29 | 0 | 1.0000 | 0.4489 | 0.3025 | 0.8021 | 0.7735 |
| 6 | `candidate_baseline` | 0 | None | 0.1676 | 16 | 1 | 0.5000 | 0.2826 | 0.1715 | 0.3833 | 0.3830 |
| 7 | `candidate_0000_26ddf87b` | 0 | `candidate_baseline` | 0.1512 | 16 | 1 | 0.5000 | 0.2649 | 0.1455 | 0.3841 | 0.3750 |
| 8 | `candidate_0000_3325815e` | 0 | `candidate_baseline` | 0.1511 | 16 | 1 | 0.5000 | 0.2537 | 0.1800 | 0.3759 | 0.3760 |
| 9 | `candidate_0000_c1465b8b` | 0 | `candidate_baseline` | 0.1428 | 16 | 1 | 0.5000 | 0.2588 | 0.1155 | 0.3967 | 0.3660 |
| 10 | `candidate_0002_8823fdb6` | 2 | `candidate_0001_aced520a` | 0.1416 | 16 | 1 | 0.5000 | 0.2418 | 0.1485 | 0.4028 | 0.3745 |
| 11 | `candidate_0003_72211d5c` | 3 | `candidate_0000_8a2d6494` | 0.1238 | 16 | 1 | 0.5000 | 0.2201 | 0.1285 | 0.3949 | 0.4010 |
| 12 | `candidate_0001_6f67ca94` | 1 | `candidate_baseline` | 0.1216 | 14 | 1 | 0.5000 | 0.2402 | 0.1525 | 0.3865 | 0.3830 |
| 13 | `candidate_0003_891c421b` | 3 | `candidate_0000_8a2d6494` | 0.1169 | 14 | 1 | 0.5000 | 0.2323 | 0.1520 | 0.3823 | 0.3630 |
| 14 | `candidate_0002_267eeeda` | 2 | `candidate_0000_8a2d6494` | 0.1107 | 14 | 1 | 0.5000 | 0.2330 | 0.1135 | 0.3852 | 0.3820 |

## Generation-Level Trend

Best candidate per generation:

| Generation | Best Candidate | Fitness | Candidates |
|---:|---|---:|---:|
| 0 | `candidate_0000_8a2d6494` | 0.2985 | 5 |
| 1 | `candidate_0001_aced520a` | 0.2735 | 3 |
| 2 | `candidate_0002_06a3d59f` | 0.2754 | 3 |
| 3 | `candidate_0003_16ed35c9` | 0.2394 | 3 |

Interpretation:

- Evolution preserved several good descendants, but did not surpass the best initial mutant.
- Later candidates often improved alignment but lost schema/diversity fitness, so overall score declined.
- The current mutation/search settings may be too noisy or too small to consistently improve over a strong initial candidate.

## Comparison With Earlier Batches

| Batch | Status | Best Fitness | Validity | Val Behavior Coverage | Test Behavior Coverage | Test Alignment |
|---|---|---:|---:|---:|---:|---:|
| batch01 `mega_persona_formal_run` | failed pipeline | 0.0000 | 0.0000 | 0.0000 | n/a | n/a |
| batch02 `pilot_batch02` | pipeline passed | 0.2119 | 1.0000 | 0.2380 | 0.1710 | 0.7336 |
| batch03 `medium_batch03` | medium pilot passed | 0.2985 | 1.0000 | 0.3320 | 0.3060 | 0.7717 |

Medium batch 03 is a clear improvement over pilot batch 02.

## Reliability And Failure Modes

The run completed, but transient API failures affected candidate ranking:

- seed evaluations: `28`
- successful seed evaluations: `19`
- failed seed evaluations: `9`
- failed seed error type: `APITimeoutError` only

Observations:

- The best candidate had no failed seeds.
- Several low-ranked candidates lost one full seed because shadow simulation timed out.
- Per-slot generation resilience worked: slot-level failures did not crash candidate evaluation.
- Per-seed timeout during shadow simulation still zeroes the seed. This is scientifically conservative, but it can make candidate ranking depend on external API stability.

This is the main engineering issue to fix before a larger formal experiment.

## Scientific Interpretation

Supported by this run:

1. The repaired MegaPersona generator can produce schema-valid, non-duplicate personas at medium scale.
2. The selected persona generator has meaningful validation/test generalization.
3. Behavior-space coverage is no longer collapsed.
4. Frozen validation/test splits and sealed test evaluation are functioning.

Not yet supported:

1. Strong evidence that Open-Evolve improves the generator over generations.
2. Stable final conclusions about the best genome, because only one medium batch has been run.
3. Final publication-level measurement validity, because the shadow surveys are still local construct proxies rather than official CEPS/PISA item texts.

## Recommended Next Actions

Before the next large run, the first and third items below have now been implemented in code:

1. Add retry/backoff around LLM shadow simulation calls, especially `APITimeoutError`. Done after this analysis.
2. Optionally re-evaluate the top 3 candidates from this batch with the same frozen survey splits after retry support, to reduce API-noise bias.
3. Reduce mutation scale or add a local exploitation phase around `candidate_0000_8a2d6494`, since descendants did not beat it. Done after this analysis.
4. Increase generations only after transient failures are controlled.

Suggested next experiment:

- reuse the same scale or slightly larger
- `n=16`
- `seeds=17,23,31`
- `generations=4`
- `population-size=6`
- `children-per-generation=4`
- use the retry/prompt-evolution fixes added after this batch

Success criteria for moving to formal:

- best evolved child beats best generation-0 candidate by at least `0.03`
- validation behavior coverage remains above `0.30`
- sealed test behavior coverage remains above `0.28`
- validation/test alignment gap remains below `0.05`
- no more than 5% seed evaluations fail due to API/transient errors

## Follow-Up Code Changes

Implemented after this analysis:

1. Added retry/backoff around shadow-survey LLM simulation calls so transient `APITimeoutError` no longer immediately zeroes a seed.
2. Added retry/backoff around persona-generation LLM calls so a transient agent timeout does not immediately invalidate a slot.
3. Expanded the evolvable `prompt_profile` from 4 coarse switches to 7 policy dimensions:
   - `mechanism_focus`
   - `tension_level`
   - `specificity`
   - `anti_stereotype`
   - `axis_binding`
   - `coverage_strategy`
   - `behavioral_signal`
4. Added prompt policies that explicitly target axis orthogonality, within-bucket variety, edge cases, and survey-predictive behavioral evidence.
5. Reduced child mutation scale after the initial population so later generations perform more local exploitation around good candidates instead of destructively jumping away from them.
6. Added a 20-item evolution operator bank. Each mutated candidate now records
   `last_evolution_operator` in its genome, and the selected operator instruction
   is injected into the LLM generation prompt.
7. Added tests for transient retry behavior, the expanded prompt addendum, and
   evolution-operator persistence.
