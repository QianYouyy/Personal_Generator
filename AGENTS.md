# AGENTS.md — MegaPersona-Evolve Project Context

Last updated: 2026-07-19

This file is the fast onboarding note for Codex or other coding agents. Read it before making changes, then inspect the specific files mentioned by the user.

## Project Goal

This repository implements **MegaPersona-Evolve**: a schema-constrained, coverage-guided synthetic persona generation experiment.

The project combines:

- DeepMind-style spatial coverage and evolutionary optimization.
- HACHIMI-style structured large persona schema, symbolic validation, and shadow survey evaluation.
- A real `OpenEvolve` island engine that optimizes a schema-aware MegaPersona genome through LLM mutation.

The scientific goal is to generate persona populations that are:

- schema-valid;
- semantically non-duplicative;
- diverse across cognitive, motivational, self-regulation, social, and mental-health axes;
- behaviorally consistent under held-out shadow surveys.

The current system now treats evolution as a three-stage model pipeline:

- `mutator_model`: mutates parent genome -> child genome
- `persona_model`: executes the generator and produces personas
- `simulator_model`: simulates held-out shadow-survey behavior

Persona generation has two explicit pipeline modes:

- `--persona-pipeline five_agent`: decomposed 5-call architecture with staged sections.
- `--persona-pipeline single_call`: integrated 1-call architecture that writes the
  complete persona at once.
- `--persona-pipeline compact`: legacy alias for `single_call`.

Treat pipeline mode as an experimental condition, not a hierarchy where
`five_agent` is inherently more formal. Report results separately by pipeline
mode because API cost, coherence profile, and failure modes differ.

## Concurrency Policy

For this project, default to **maximum practical concurrency**, especially for
smoke experiments. The user prefers fast completion over conservative
throughput.

Interpret the worker flags this way:

- `--candidate-max-workers`: parallel candidate evaluations
- `--persona-max-workers`: parallel persona generations inside one candidate
- `--shadow-max-workers`: parallel shadow-survey simulation calls

Operational rule for future runs:

- Smoke runs should use aggressive worker counts by default, not the minimum.
- If the user asks for a quick smoke or provider sanity check, bias toward
  finishing fast rather than minimizing concurrency.
- Only dial worker counts down when debugging determinism, provider instability,
  or a specific race condition.

Practical guidance:

- `shadow-max-workers` is usually the main bottleneck for runtime.
- If the run log shows `Shadow simulation start ... max_workers=1` or `2`, that
  is usually too conservative for this user's preferred workflow.
- Keep the laptop awake during long runs; perceived slowness can also come from
  sleep interruptions.

## Current Main Flow

The current official evolution entrypoint is:

```bash
python scripts/run_mega_persona_evolution.py
```

This script runs through:

```text
scripts/run_mega_persona_evolution.py
  -> src.mega_persona.openevolve_adapter.MegaPersonaOpenEvolveRunner
  -> src.open_evolve.engine.OpenEvolve
  -> src.mega_persona.evolution.MegaPersonaEvolver as evaluator/artifact backend only
```

Important: `MegaPersonaEvolver.run()` is retired and intentionally raises. Do not re-enable the old custom evolution loop. `MegaPersonaEvolver` remains useful for candidate evaluation, shadow survey scoring, sealed test evaluation, and durable artifacts.

## What Actually Evolves

The project does not evolve arbitrary Python source code. It evolves a
schema-aware JSON genome under a fixed generation/evaluation architecture.

Current genome surface:

- `schema_binding`
  - `axis_names`
  - `axis_roles`
  - `quota_buckets`
- `quota_weights`
- `axis_bias`
- `axis_stretch`
- `prompt_profile`
- `agent_focus`
- `field_requirements`
- `behavior_anchors`
- `consistency_rules`
- `repair_policy`
- `blueprint_policy`
- `axis_expression_policy`
- `cross_agent_binding_policy`
- `behavior_prediction_policy`
- `critic_policy`
- `last_evolution_operator`

This is important for new chats: if the user wants to change persona
attributes or rename axes, do not assume the whole generator is invalid. The
current implementation is already `schema_binding`-aware across slot sampling,
hard constraints, validator axis checks, shadow surveys, simulators, and
visualization.

Important 2026-06-27 update: `Genome v3` is now the default. The evolved
surface is no longer only prompt advice. Each candidate genome builds a
per-slot `generation_blueprint` through `blueprint_from_slot()`, and the
generator injects that blueprint into the selected generation pipeline, hard
constraints, and lightweight blueprint critic.

Genome v3 evolves:

- how target axes become observable behavior anchors;
- how strongest/weakest axis tensions are selected;
- how cognition, values, social behavior, and mental-health fields echo one
  another;
- how ambiguity, peer pressure, feedback, and deadlines should predict shadow
  survey behavior;
- what the revision stage should treat as missing blueprint evidence.

Old v2 checkpoints still normalize forward, but new experiments should be
reported as Genome v3 unless they intentionally load an old run.

## Key Files

| Path | Purpose |
|---|---|
| `scripts/run_mega_persona_evolution.py` | Canonical OpenEvolve runner for MegaPersona |
| `src/mega_persona/openevolve_adapter.py` | Bridges MegaPersona JSON genomes to `OpenEvolve`, including the LLM mutator path |
| `src/open_evolve/engine.py` | Shared island-based OpenEvolve engine |
| `src/mega_persona/evolution.py` | Genome utilities, mutation operators, evaluator backend, persistent store |
| `src/mega_persona/generator.py` | MegaPersona LLM generation pipeline; `five_agent` and `single_call` architectures |
| `src/mega_persona/consistency.py` | Internal consistency and axis-alignment scoring |
| `src/mega_persona/shadow_survey.py` | Scientific shadow survey registry and train/validation/test split |
| `src/mega_persona/shadow_simulator.py` | LLM simulator for persona-to-Likert behavior |
| `src/mega_persona/experiment.py` | Shared score formula and batch experiment utilities |
| `scripts/run_mega_persona_operator_ablation.py` | Fixed-parent controlled operator ablation |
| `scripts/visualize_mega_persona_evolution_dashboard.py` | Diagnostic HTML dashboard for per-generation and per-operator metrics |
| `docs/TECHNICAL_ROUTE_REPORT.md` | Current technical route summary |
| `docs/MEGA_PERSONA_EXPERIMENT.md` | Experiment protocol and CLI guidance |

## Current Output Layout

For the canonical OpenEvolve runner:

```text
data/results/<run_name>/
├── manifest.json
├── run.log
├── final_summary.json
├── final_summary.md
├── open_evolve/
│   ├── checkpoint.json
│   ├── checkpoint_gen_*.json
│   └── elite_codes_gen_*/
└── mega_eval/
    ├── checkpoint.json
    ├── final_summary.json
    ├── final_test_report.json
    ├── shadow_surveys/
    ├── candidates/
    └── evaluations/
```

Resume uses:

```text
data/results/<run_name>/open_evolve/checkpoint.json
```

Operator ablation runs still use a flatter output layout because they are controlled evaluation studies, not the main OpenEvolve loop.

## Scientific Evaluation Rules

Evolution selection must use only train/validation shadow behavior.

The sealed test split is only for the final selected best candidate. Do not use test metrics for candidate selection, prompt/operator tuning, or checkpoint ranking.

The shared score is:

```text
score =
  schema_fitness
  × (0.5 + 0.5 × behavior_coverage)
  × (0.5 + 0.5 × shadow_alignment)
  × generation_rate
```

The intent is to penalize collapsed behavior or poor alignment even when schema validity looks good.

Also watch these consistency-related metrics in newer runs:

- `internal_consistency.mean`
- `internal_consistency_min.mean`
- `axis_alignment.mean`

## 2026-06-26 Negative Result And Genome v3 Validation Plan

The latest `Genome v2` run should be preserved as a negative/diagnostic result:

```text
data/results/mega_persona_openai_genome_v2_n16_g20_20260626
```

Observed:

- baseline fitness: `0.335616`
- best fitness: `0.343820`
- relative gain: `+2.44%`
- best appeared at `gen5`, then plateaued through `gen20`
- best improved schema / behavior coverage / diversity slightly, but reduced
  validation alignment, axis alignment, internal consistency, and slot coverage.

Interpretation:

- Do not present this as successful evolution.
- The individual metric changes are small enough that they look close to random
  fluctuation.
- Changing fitness weights alone is not a convincing fix, because the single
  metrics themselves do not move meaningfully.
- The core issue is likely that `Genome v2` evolves prompt advice, not a strong
  generation mechanism.

Avoid repeating this ineffective path:

- Do not keep scaling all 15 mixed operators merely to chase tiny aggregate gains.
- Do not claim success from aggregate fitness if per-metric deltas are tiny or
  achieved by trading off alignment/consistency.
- Do not tune fitness before checking that the evolved object has causal leverage.

Genome v3 validation plan:

- Use `--fixed-operator <operator_id>` to isolate one operator at a time.
- Keep `n`, seed, surveys, simulator, model, and worker settings fixed.
- Compare per-metric deltas against baseline, not only aggregate fitness.
- First check whether v3 blueprint fields actually change generated personas
  and shadow behavior beyond noise.
- Treat an operator as promising only if it moves its target metric beyond
  noise while not degrading alignment, consistency, or slot coverage.

Example fixed-operator run:

```bash
python scripts/run_mega_persona_evolution.py \
  --llm-provider openai \
  --generator-mode llm \
  --n 16 \
  --seeds 17 \
  --generations 10 \
  --num-islands 8 \
  --children-per-island 1 \
  --fixed-operator op14_recovery_latency \
  --simulator-backend student-realistic-v2 \
  --shadow-surveys 8 \
  --validation-shadow-surveys 3 \
  --test-shadow-surveys 3 \
  --items-per-shadow-survey 10 \
  --candidate-max-workers 4 \
  --persona-max-workers 8 \
  --shadow-max-workers 32 \
  --output-dir data/results/mega_persona_fixed_op14_n16_g10
```

Current direction:

- Use `Genome v3` blueprint-first generation as the default experimental
  surface.
- Treat `persona_pipeline` as a design variable. Compare `five_agent` and
  `single_call` directly when the research question is generation architecture.
- Evolution should affect generator mechanisms through blueprint policy,
  axis-expression policy, cross-agent binding, behavior prediction anchors,
  critic policy, and targeted repair policy.
- The goal is to test mechanism-level evolution rather than only prompt addenda.

## Current Operator Evidence

Historical operator evidence before the 2026-06-26 negative result:

```text
docs/MEGA_PERSONA_BATCH_2026-06-16_OPENEVOLVE_MEDIUM.md
```

Main findings:

- Operator bank currently has 15 operators: `op01` to `op15`.
- `op06_low_axis_fidelity` is the strongest signal in the medium OpenEvolve run.
- Best candidate: `openevolve_000022_84f5f9724380`.
- Best fitness: `0.373588`.
- Baseline fitness: `0.342958`.
- Improvement: about `+8.9%`.
- `op04_within_bucket_contrast` remains a stable positive operator.
- `op07_high_axis_cost` and `op02_behavioral_evidence` remain worth testing.
- `operator_only` is currently the strongest mutation mode.
- `mixed` has higher variance and should be used sparingly.
- `numeric_only` is weak as a standalone mode.
- `op01_axis_decoupling` is high risk and caused schema/alignment collapses, including one 0-fitness candidate.

Recommended next search policy:

- Increase sampling weight for `op06_low_axis_fidelity` and `op04_within_bucket_contrast`.
- Keep `op07_high_axis_cost` and `op02_behavioral_evidence` as medium-priority operators.
- Downweight or temporarily remove `op01_axis_decoupling` until it is rewritten.
- Prefer `operator_only` and restrained `prompt_only`; lower `mixed` frequency.

## 2026-07 MCTS, Metrics, And Plateau Findings

Recent Genome v3 work added `hybrid_mcts` as an operator search strategy.
Treat MCTS as a history-aware operator selection policy, not as a replacement
for the evaluator or fitness function.

Current evidence:

- MCTS can change the search path compared with random v3 operator sampling.
- In one completed seed17 comparison, MCTS improved validation fitness over
  the random v3 pool checkpoint by about `+2.9%`.
- The improvement is not yet a stable final result: sealed-test generalization
  and cross-seed operator preference remain uncertain.
- Different reward designs produce different operator preferences. A
  strict-aware reward can improve consistency/axis targeting while narrowing
  behavior coverage or diversity.

Use refined metrics as diagnostics, not as a claim that every metric must
improve together:

- `axis_target_mae`: whether generated personas match target slot axes;
  lower is better.
- `shadow_mae`: behavior prediction error on shadow surveys; lower is better.
- `strict_consistency_error`: stricter cross-field/self-consistency error;
  lower is better.
- `validation_behavior_coverage` and
  `validation_behavior_balanced_diversity`: whether behavior response space is
  broad and non-collapsed; higher is generally better, but these should be
  treated as soft search signals rather than hard constraints.

Important interpretation:

- Do not claim that evolution should improve every metric simultaneously.
  These metrics expose which direction the search is moving and whether there
  is a trade-off or search bias.
- Several recent runs show early or mid-run plateau behavior: best candidates
  often appear around gen5 or gen9, then global best stops refreshing.
- `multi-objective elite archive` already exists. The next design question is
  how to use existing strict/coverage/diversity elites during plateau escape,
  not how to add elite storage again.
- Coverage/diversity should not be hard constraints. Prefer soft protection:
  reduce future sampling priority for paths with clear convergence, but keep
  low-diversity/high-consistency candidates for diagnosis.
- Reward design is structured rather than simple linear weighting: basic
  validity/generation reliability first, coverage/diversity soft protection
  second, then optimize fitness, shadow MAE, axis target MAE, and strict
  consistency. See the structured reward profile below.

### Structured MCTS Reward Profile (2026-07-19)

`--mcts-reward-profile {legacy,structured}` selects the MCTS reward design.
`legacy` is the default and keeps historical runs comparable. `structured` is
the plateau-aware design:

- Layered reward: hard gate (failed evaluation -> reward `-1`), bounded
  coverage/diversity soft guard (only relative drops beyond 2% are penalized),
  then weighted relative improvement over parent across the 11 elite metrics.
- `--mcts-reward-weight-mode {fixed,deficit}` controls the optimization
  weights. `fixed` uses the static weight table. `deficit` scales each
  metric's weight by how far the parent lags that metric's policy-tracked
  historical best (normalized into shares, floor 0.02), so saturated
  directions lose weight and lagging directions gain it; the reward rotates
  across metric combinations instead of always chasing the fixed
  `global_best` share. `mcts_summary` reports `mean_opt_weights` and
  `last_opt_weights` so the rotation is observable in run artifacts.
- Progress bonus: the policy tracks the historical best of every elite metric
  across islands; beating it adds a positive term, so plateau runs still get
  positive feedback when non-global metrics improve.
- Plateau detection is internal to `OperatorMCTSPolicy`: stagnation =
  generations since `global_best` last improved. After
  `--mcts-plateau-stagnation` generations (default 4), the progress bonus
  doubles, improvements on metrics stale for 20+ results get an extra bonus,
  and the UCT exploration constant is boosted by `1 + 0.2 x min(stagnation, 8)`.
- Rewards are z-score standardized (Welford running stats, std floor 0.02,
  clip +/-3) before backpropagation so the UCT exploration term stays on a
  comparable scale; the legacy profile's absolute-delta rewards were ~1e-3 and
  were drowned out by `exploration_c=1.4`.
- The shared fitness formula, the engine, and the engine's plateau parent
  selection are unchanged; this only reshapes the operator-selection reward.

### Multi-Objective Parent Selection (2026-07-19)

`--parent-selection {operator_preferred,objective_rotation}` controls which
elite becomes the parent of each child. `operator_preferred` is the default and
the historical behavior. `objective_rotation` round-robins each child's parent
across five objective elite roles — `global_best`, `coverage_elite`,
`diversity_elite`, `strict_consistency_elite`, `shadow_mae_elite` — so every
objective's best candidate participates in mutation instead of letting one
high-fitness template dominate reproduction.

- Implemented in `src/open_evolve/engine.py` (`parent_selection`,
  `parent_objective_roles`, `_parent_rotation_cursor`); the plateau parent
  branch (stagnation >= 4) still takes precedence and does not consume the
  rotation cursor.
- This changes the supply side of search (who reproduces), which is upstream
  of and orthogonal to MCTS reward shaping (how produced children are
  credited). It composes with `--mcts-reward-weight-mode deficit`: rotated
  parents give the deficit computation different reference points.
- Motivation: seed17/seed23 comparisons showed `deficit` changes search
  preference but is not uniformly better than `fixed`; rather than tuning
  reward weights further, parent-level multi-objective participation is the
  more stable lever.
- Seed23 rotation result: final validation fitness slightly lower than
  `deficit`, but the candidate pool widened (higher coverage-best,
  diversity-best, strict-best, shadow-MAE-best) and sealed-test diversity /
  shadow MAE were the best of the three compared runs. The bottleneck then
  moved to final selection, motivating the sealed-test comparison below.

### Multi-Objective Sealed-Test Comparison (2026-07-19)

`scripts/run_multi_objective_sealed_test.py <run_dir>` evaluates the
multi-objective best candidates of any finished run (from
`multi_objective_best_candidates`) on the sealed test split and writes
`multi_objective_final_test.json/.md` next to the run's final summary.

- Reuses the run's persisted survey splits (`resume=True` on the backend) and
  the manifest's LLM provider settings; `--llm-provider`, model, and
  `--shadow-max-workers` overrides exist for other cases.
- Purpose: decide whether global-fitness final selection systematically
  ignores better-generalizing coverage/diversity/strict candidates. It audits
  the selection rule; `test_used_for_selection: False` still holds and no
  winner is re-picked per run.
- If sealed tests show role candidates (e.g. coverage_best) generalize better
  than global_best, the next design question is a final-selection/soft
  reranking rule, not more MCTS reward tuning.

Recent report:

```text
docs/mega_persona_experiment_records/WEEKLY_REPORT_2026-07-14.md
```

## 2026-07-24 Next Direction: Goal-Conditioned Generator + Persona Archive

Confirmed next-phase design (not yet implemented): keep the OpenEvolve
island + MCTS loop unchanged, but move evaluation from scalar-fitness slots to
curriculum-sampled target cells with a `PersonaArchive` bookkeeping layer
(MAP-Elites style) inside the evaluator. The deliverable is a goal-conditioned
generator `Gθ`, validated by clearing the archive and regenerating the whole
persona space. Full design, motivation evidence (behavior-grid occupancy,
noise floor), and the six locked decisions are in:

```text
docs/MEGA_PERSONA_GOAL_CONDITIONED_GENERATOR_DESIGN.md
```

Analysis tool added for this: `scripts/report_behavior_grid_occupancy.py`
(post-hoc behavior-grid occupancy of any finished run, zero API cost).

## Common Commands

Small LLM smoke run:

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-model-key llm.simulator_model \
  --simulator-backend llm \
  --n 10 \
  --seeds 17,23 \
  --generations 3 \
  --num-islands 4 \
  --children-per-island 1 \
  --shadow-surveys 3 \
  --validation-shadow-surveys 2 \
  --test-shadow-surveys 2 \
  --items-per-shadow-survey 8 \
  --candidate-max-workers 4 \
  --persona-max-workers 6 \
  --shadow-max-workers 8 \
  --output-dir data/results/mega_persona_openevolve_smoke
```

Schema-aware Stage-2 run with true LLM mutation:

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --mutator-model-key llm.mutator_model \
  --model-key llm.persona_model \
  --simulator-model-key llm.simulator_model \
  --simulator-backend student-realistic \
  --n 32 \
  --seeds 17,23 \
  --generations 10 \
  --num-islands 8 \
  --children-per-island 1 \
  --elite-count 3 \
  --shadow-surveys 8 \
  --validation-shadow-surveys 4 \
  --test-shadow-surveys 4 \
  --items-per-shadow-survey 10 \
  --candidate-max-workers 6 \
  --persona-max-workers 6 \
  --shadow-max-workers 12 \
  --output-dir data/results/mega_persona_stage2_schema_aware
```

DeepSeek provider run:

```bash
export DEEPSEEK_API_KEY=...
python scripts/run_mega_persona_evolution.py \
  --llm-provider deepseek \
  --generator-mode llm \
  --simulator-backend student-realistic \
  --n 32 \
  --seeds 17,23 \
  --generations 10 \
  --num-islands 8 \
  --children-per-island 1 \
  --elite-count 3 \
  --shadow-surveys 8 \
  --validation-shadow-surveys 4 \
  --test-shadow-surveys 4 \
  --items-per-shadow-survey 10 \
  --candidate-max-workers 6 \
  --persona-max-workers 6 \
  --shadow-max-workers 12 \
  --output-dir data/results/mega_persona_deepseek_stage2
```

Concordia-style simulator smoke run:

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-backend concordia \
  --simulator-model-key llm.simulator_model \
  --n 10 \
  --seeds 17,23 \
  --generations 3 \
  --num-islands 4 \
  --children-per-island 1 \
  --shadow-surveys 3 \
  --validation-shadow-surveys 2 \
  --test-shadow-surveys 2 \
  --items-per-shadow-survey 8 \
  --candidate-max-workers 4 \
  --persona-max-workers 4 \
  --shadow-max-workers 4 \
  --output-dir data/results/mega_persona_concordia_smoke
```

Concordia native simulator smoke run:

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-backend concordia-native \
  --simulator-model-key llm.simulator_model \
  --n 6 \
  --seeds 17 \
  --generations 1 \
  --num-islands 2 \
  --children-per-island 1 \
  --shadow-surveys 2 \
  --validation-shadow-surveys 1 \
  --test-shadow-surveys 1 \
  --items-per-shadow-survey 6 \
  --candidate-max-workers 1 \
  --persona-max-workers 1 \
  --shadow-max-workers 2 \
  --output-dir data/results/mega_persona_concordia_native_smoke
```

Resume:

```bash
python scripts/run_mega_persona_evolution.py \
  ...same args... \
  --generations 5 \
  --output-dir data/results/mega_persona_openevolve_smoke \
  --resume
```

Operator ablation:

```bash
python scripts/run_mega_persona_operator_ablation.py \
  --source-run data/results/<source_run> \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-model-key llm.simulator_model \
  --n 6 \
  --seeds 17,23 \
  --replicates 2 \
  --mutation-scale 0.08 \
  --output-dir data/results/mega_persona_operator_ablation_next
```

Core tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_evolution.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_openevolve_adapter.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_open_evolve.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mcts_policy.py
```

## Development Notes

- Prefer `rg` for search.
- Use `apply_patch` for manual file edits.
- Do not delete or overwrite user experiment outputs unless explicitly asked.
- Do not kill long-running MegaPersona experiments unless the user explicitly asks.
- Keep train/validation/test split integrity intact.
- If adding or changing operators, update tests and the latest technical notes.
- If changing output layout or CLI flags, update `README.md`, `docs/MEGA_PERSONA_EXPERIMENT.md`, and this file.
- Avoid introducing a second evolution mechanism. The main evolution mechanism is `src.open_evolve.engine.OpenEvolve`.
- If changing schema/axes, update the schema binding path first, not scattered legacy axis strings.
- When proposing smoke commands, prefer high worker counts by default so the
  smoke finishes quickly.

## 2026-07-20 Mutation Audit Changes

Behavior changes made to make mutations measurable and cheaper:

- The selected operator instruction is no longer injected into the generation
  prompt addendum (`prompt_addendum_from_genome`). Operator instructions are
  mutator-facing only; `last_evolution_operator` remains in the genome as
  lineage metadata.
- The LLM mutator is now asked to return `declared_edits` alongside `patch`.
  Each child's `openevolve_mutation` metadata records `declared_edits`,
  `actual_edits`, `undeclared_edits`, and `phantom_edits` (top-level evolvable
  fields), so per-field mutation effects can be audited post hoc.
- `MegaOpenEvolveEvaluator` dedups by `genome_phenotype_hash` (the genome minus
  lineage metadata) in addition to the exact genome hash: children that differ
  only in mutation metadata reuse the first non-zero-persona evaluation instead
  of paying full cost. Cache hits still write a lightweight alias candidate and
  `evaluations/alias_<candidate_id>/result.json` with
  `phenotype_cache_source_candidate_id`, so the current child keeps its own
  lineage/operator metadata without increasing the true evaluation count. The
  phenotype index is rebuilt from stored candidates on resume.
- Concurrent candidates now register an in-flight phenotype owner before the
  expensive evaluation starts. Other threads wait for that owner and then write
  their own alias records, preventing parallel cache misses from evaluating the
  same phenotype multiple times.
- OpenEvolve candidates and elite checkpoints persist both `candidate_id` and
  `parent_id`. The engine passes parent context into the MegaPersona evaluator,
  including phenotype-cache aliases, so operator chains can be reconstructed
  after a run or resume instead of leaving every stored parent as `null`.
- Phenotype cache-hit children are not backpropagated into the MCTS operator
  policy. They may still enter the island archive if their reused fitness is an
  elite, but they do not count as fresh operator evidence because no new
  phenotype was evaluated.
- No-op retry: if the edit audit finds `actual_edits` empty, the LLM mutator
  retries once with explicit "no effective change" feedback. `noop_retries` in
  the mutation metadata counts discarded no-op drafts (0 = clean first try,
  2 = retry still produced nothing).
- Numeric surface: after the edit audit, every LLM child gets a small Gaussian
  jitter on `axis_bias`/`axis_stretch` at the rule-mutation magnitudes, because
  the LLM rarely edits numeric fields on its own. Jittered axes are reported
  under `numeric_jitter`, separate from the mutator-facing `actual_edits`, so
  no-op detection and per-field attribution stay clean.
- Mutator network resilience: the mutator's LLM call now goes through
  `_generate_with_retry` (same transient-error classification and backoff as
  the generator/simulator), so a transient connection error no longer silently
  demotes the child to rule mutation.
- `scripts/measure_evaluation_noise_floor.py --source-run <run>` re-evaluates
  the run's best genome and the default seed genome K times under the run's
  exact config (same slots/surveys/models, verified via survey hashes) and
  writes `noise_floor.json/.md`. Fitness differences below 2*std are within
  noise; use this to size large runs and to judge plateau claims.

## 2026-07-24 Experimental Genome v4

Genome v3 remains the default for backward compatibility. Genome v4 is an
explicit experimental surface enabled by `--genome-version 4`; it automatically
uses `--operator-family v4` unless overridden.

Genome v4 replaces the high-dimensional free-form prompt surface with a small
structured behavior-generation program:

- `probe_assignment`: maps the three schema-bound axis roles to observable scenarios;
- `axis_realization`: bounded realization mode and signal strength per axis role;
- `interaction_mode`: strongest/weakest-axis relationship;
- `echo_graph`: deterministic cross-field evidence graph;
- `context_modulation`: how context changes expression of the same mechanism;
- `repair_control`: evidence density and repair priority.

`quota_weights`, `axis_bias`, and `axis_stretch` remain in the JSON for sampler
compatibility but are frozen by v4 operators. `blueprint_from_slot()` renders
the v4 program deterministically into the existing generation blueprint.

The v4 operator pool is `op22` through `op27`. Every operator mutates exactly
one module, records that module under `actual_edits`, and uses the
`structured_v4` backend. The LLM mutator and v3 numeric jitter are intentionally
skipped so operator rewards remain attributable. Keep initial v4 validation on
`search_strategy=openevolve`; test MCTS only after fixed-operator effects exceed
the measured evaluation noise floor.

## 2026-07-24 Noise-Aware Candidate Selection

The v4 noise-floor audit showed that the apparent smoke improvement was not
resolvable: repeated best-genome fitness was `0.23497 +/- 0.00922` (std) versus
`0.23416 +/- 0.01202` for the default v4 seed. Coverage/diversity were noisier
still. Treat all historical single-evaluation search gains as exploratory.

`--candidate-evaluation-repeats` controls the number of independent
full-pipeline evaluations per phenotype. The default experiment protocol is
`--candidate-evaluation-repeats 1 --elite-confirmation-repeats 1`; no candidate
receives automatic additional evaluations. Use larger repeat counts only for an explicit
noise-floor or confirmatory ablation.
`MegaOpenEvolveEvaluator` averages fitness and every numeric metric before
OpenEvolve elite updates or MCTS backpropagation, and stores repeat values,
std, and SEM. Phenotype caches are reused only when they contain at least the
requested repeat count, including resume. All stored repeat persona populations
are evaluated on the sealed test after selection, so validation and test use
the same statistical unit while test remains excluded from selection. New-run
sampling defaults are persona `temperature=0.45, top_p=0.85` and simulator
`temperature=0.05, top_p=0.80`.

## Quick Orientation For New Chats

If a new conversation starts, first read:

1. `AGENTS.md`
2. `README.md`
3. `docs/TECHNICAL_ROUTE_REPORT.md`
4. `docs/MEGA_PERSONA_EXPERIMENT.md`
5. `src/mega_persona/openevolve_adapter.py`
5. `scripts/run_mega_persona_evolution.py`

Then inspect the files relevant to the user's current request.
