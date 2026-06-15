# MegaPersona Operator Ablation V2 Partial Analysis - 2026-06-15

## Status

This is a **partial analysis** of the ongoing run:

- Output directory: `data/results/mega_persona_operator_ablation_v2_20260614`
- Analysis time: 2026-06-15
- Current checkpoint: 27 / 52 candidates evaluated
- Pending candidates: 25
- Current best candidate: `ablation_0014_op04_within_bucket_contrast_mixed_r01`
- Current best fitness: `0.2189`

The run was not completed on 2026-06-14 and is still continuing. Therefore, the
results below should be treated as a checkpoint-level trend, not as a final
operator conclusion.

## Experiment Design

This run evaluates the new 8-operator bank introduced after the previous
20-operator experiments showed unstable improvement.

Compared with the old operator bank, the new operators are more behavior-facing:

| Operator | Intended intervention |
|---|---|
| `op01_axis_decoupling` | Force visible high/low contrasts across abstraction, autonomy, and resilience |
| `op02_behavioral_evidence` | Add deadline, peer pressure, failure feedback, and ambiguous-task evidence |
| `op03_shadow_survey_alignment` | Add survey-inferable cues without writing survey answers directly |
| `op04_within_bucket_contrast` | Increase contrast within the same quota bucket |
| `op05_failure_recovery_cycle` | Add trigger-appraisal-coping-outcome-adjustment failure cycles |
| `op06_low_axis_fidelity` | Preserve low-axis behavioral costs |
| `op07_high_axis_cost` | Add realistic costs for high-axis traits |
| `op08_validation_conservatism` | Preserve consistency while keeping measurable diversity |

The ablation compares each operator under:

- `prompt_only`
- `operator_only`
- `mixed`
- plus `parent_replay` and `numeric_only` controls

The planned run contains 52 candidates:

- 2 parent replay candidates
- 2 numeric-only candidates
- 8 operators x 3 mutation modes x 2 replicates = 48 candidates

At this checkpoint, the first replicate is essentially complete, and the second
replicate is pending.

## Completed Candidate Summary

Current parent replay baseline:

| Parent replay | Fitness |
|---|---:|
| `ablation_0000_parent_replay_r01` | `0.2038` |
| `ablation_0001_parent_replay_r02` | `0.2072` |
| Mean | `0.2055` |
| Std | `0.0017` |

Top completed candidates:

| Rank | Candidate | Mode | Operator | Fitness | Schema | Val Coverage | Val Alignment |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `ablation_0014_op04_within_bucket_contrast_mixed_r01` | `mixed` | `op04_within_bucket_contrast` | `0.2189` | `0.3755` | `0.3285` | `0.7546` |
| 2 | `ablation_0008_op02_behavioral_evidence_mixed_r01` | `mixed` | `op02_behavioral_evidence` | `0.2189` | `0.3782` | `0.3385` | `0.7293` |
| 3 | `ablation_0013_op04_within_bucket_contrast_operator_only_r01` | `operator_only` | `op04_within_bucket_contrast` | `0.2126` | `0.3722` | `0.2960` | `0.7638` |
| 4 | `ablation_0011_op03_shadow_survey_alignment_mixed_r01` | `mixed` | `op03_shadow_survey_alignment` | `0.2120` | `0.3797` | `0.2795` | `0.7445` |
| 5 | `ablation_0006_op02_behavioral_evidence_prompt_only_r01` | `prompt_only` | `op02_behavioral_evidence` | `0.2114` | `0.3707` | `0.2925` | `0.7647` |
| 6 | `ablation_0019_op06_low_axis_fidelity_operator_only_r01` | `operator_only` | `op06_low_axis_fidelity` | `0.2109` | `0.3664` | `0.3180` | `0.7490` |
| 7 | `ablation_0023_op07_high_axis_cost_mixed_r01` | `mixed` | `op07_high_axis_cost` | `0.2109` | `0.3750` | `0.2795` | `0.7581` |

## Preliminary Interpretation

The early signal is better than the previous 20-operator bank.

Most importantly:

- `op04_within_bucket_contrast + mixed` is currently about `+0.0134` above the
  parent replay mean.
- `op02_behavioral_evidence + mixed` is also about `+0.0134` above the parent
  replay mean.
- `op04_within_bucket_contrast` appears in both the current best mixed candidate
  and a strong operator-only candidate.
- `op02_behavioral_evidence` appears strong in both mixed and prompt-only modes.

This differs from the previous operator ablation, where the best child was only
slightly above parent replay and the group-level signal was weak.

The most promising operators so far are:

1. `op04_within_bucket_contrast`
2. `op02_behavioral_evidence`
3. `op03_shadow_survey_alignment`
4. `op06_low_axis_fidelity`
5. `op07_high_axis_cost`

However, most groups currently have only one replicate. This means the result
is still directional rather than confirmatory.

## Reliability Notes

All 27 completed candidates produced seed-level status `ok` in the saved
evaluation files:

- completed candidate evaluations: 27
- completed seed evaluations: 54
- saved seed statuses: all `ok`

But the log shows that API/network instability increased after the run crossed
midnight:

- several `APIConnectionError` retries
- several `APITimeoutError` retries
- at least two slot-level generation failures in an incomplete/late candidate
- the process nevertheless continued because slot-level and seed-level isolation
are working

One important example:

- `ablation_0025_op08_validation_conservatism_operator_only_r01` has fitness
  `0.1617`
- seed 17 generated 6 / 6 personas and scored `0.2135`
- seed 23 generated only 4 / 6 personas and scored `0.1099`
- therefore this candidate's low aggregate score is likely confounded by
  network/API failure rather than pure operator quality

This means the final v2 conclusion should separate:

1. operator quality
2. LLM/network reliability
3. generation-rate penalties caused by external failures

## Current Scientific Reading

Supported at this partial checkpoint:

1. The new 8-operator bank is more promising than the old 20-operator bank.
2. Behavior-evidence style operators appear to move metrics more strongly.
3. `mixed` mode is again producing the strongest individual candidates.
4. Parent replay repetition is useful because baseline itself varies from
   `0.2038` to `0.2072`.

Not yet supported:

1. A final claim that any operator is stable.
2. A claim that `op04` or `op02` is definitively best.
3. A decision to scale to a larger formal run.

## Next Actions

1. Let the current run finish if possible.
2. When complete, analyze group means and win rates across both replicates.
3. Treat any candidates affected by visible slot-level API failures with
   caution, especially if generation rate dropped below 6 / 6.
4. If the run does not finish because of network instability, resume from the
   same output directory rather than starting over.
5. After completion, keep operators that beat repeated parent replay in group
   mean and show at least repeated wins across replicates.

## Current Bottom Line

At the 27 / 52 checkpoint, the v2 operator design is showing a materially better
early signal than the previous operator bank. The strongest current evidence
points to `op04_within_bucket_contrast` and `op02_behavioral_evidence`,
especially in `mixed` mode. The run is not complete, and later API instability
means final interpretation should wait until all candidates are evaluated or
until the run is cleanly resumed.
