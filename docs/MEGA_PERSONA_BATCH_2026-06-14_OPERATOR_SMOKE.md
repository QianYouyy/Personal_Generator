# MegaPersona Operator Smoke Test - 2026-06-14

## Purpose

This short run tests whether the new 20-item evolution operator bank can produce
a generation-1 child that beats the generation-0 candidates. This is a smoke
test, not a formal scientific run.

## Run

- Output directory: `data/results/mega_persona_operator_smoke2_20260614`
- Mode: LLM persona generator + LLM shadow simulator
- `n`: 6 personas per seed
- Seeds: `17`
- Generations: 1
- Population size: 4
- Children per generation: 3
- Elite count: 2
- Train/validation/test shadow surveys: 3 / 2 / 2
- Items per shadow survey: 8

The first attempted run at `data/results/mega_persona_operator_smoke_20260614`
was interrupted because the sandboxed process could not reach the LLM API and
all calls failed with `APIConnectionError`.

## Candidate Results

| Rank | Candidate | Gen | Parent | Operator | Fitness | Schema | Val Cov | Val Align |
|---:|---|---:|---|---|---:|---:|---:|---:|
| 1 | `candidate_baseline` | 0 | - | - | 0.1867 | 0.3366 | 0.2500 | 0.7754 |
| 2 | `candidate_0001_63fb985b` | 1 | `candidate_baseline` | `op16_failure_modes` | 0.1849 | 0.3354 | 0.2410 | 0.7769 |
| 3 | `candidate_0000_539e568b` | 0 | `candidate_baseline` | `op09_low_axis_fidelity` | 0.1793 | 0.3296 | 0.2320 | 0.7659 |
| 4 | `candidate_0000_090dcd1d` | 0 | `candidate_baseline` | `op10_mid_axis_texture` | 0.1734 | 0.3225 | 0.2120 | 0.7742 |
| 5 | `candidate_0001_d3daba75` | 1 | `candidate_0000_539e568b` | `op01_axis_orthogonality` | 0.1693 | 0.3201 | 0.1910 | 0.7765 |
| 6 | `candidate_0000_760aa52d` | 0 | `candidate_baseline` | `op20_validation_guardrail` | 0.1685 | 0.3025 | 0.2400 | 0.7967 |

## Sealed Test

The selected best candidate was still `candidate_baseline`.

- Test alignment: 0.7411
- Test behavior coverage: 0.1900
- Test behavior average distance: 0.2219

## Interpretation

This smoke test did not show a generation-1 improvement over baseline. However,
the closest child, `op16_failure_modes`, was very close to baseline:

- Baseline fitness: 0.1867
- `op16_failure_modes` child fitness: 0.1849
- Difference: -0.0018

This is better than the earlier pattern where evolved children were clearly
worse, but it is still not evidence that the operator bank improves the
generator. The sample is too small and uses only one seed.

## Notes

- All completed candidates generated 6/6 valid personas.
- No timeout failures occurred in the successful run.
- Two candidates required one revision for overlong `social_creative_profile.narrative`.
- The current bottleneck is still evaluation cost: even a small LLM run spends
  most time in multi-agent persona generation and shadow-survey simulation.

## Next Step

Run a slightly larger but still controlled operator test with two seeds and a
smaller population, or add a direct operator ablation script that evaluates each
operator on the same parent with `n=4` to estimate operator direction before
running full evolution.

## Follow-Up Code Optimization

Implemented after this smoke test:

1. Split genome mutation into diagnostic modes:
   - `prompt_only`: changes prompt/operator instructions without changing
     quota weights or axis transforms.
   - `operator_only`: applies one evolution operator without the extra random
     prompt-profile mutation.
   - `mixed`: keeps the previous combined search behavior.
   - `numeric_only`: changes quota/axis sampling without injecting an operator.
2. Generation summaries now persist `last_mutation` and
   `last_evolution_operator` for every candidate.
3. The global prompt addendum now includes an explicit schema-length guardrail
   to reduce overlong narrative revisions.

Rationale: the previous operator test could not tell whether a child underperformed
because of the operator text itself or because quota/axis noise moved the sampling
space. The next short run will be easier to interpret.
