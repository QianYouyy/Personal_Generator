# CLAUDE.md - Personal Generator / MegaPersona-Evolve

Last updated: 2026-06-27

This file gives Claude a fast project overview. For coding-agent onboarding, also read `AGENTS.md`; it is the most compact current context file.

## Project Overview

This project implements **MegaPersona-Evolve**: a schema-constrained and coverage-guided synthetic persona generation experiment.

It combines:

- DeepMind-style spatial coverage and evolutionary optimization.
- HACHIMI-style structured large persona schemas, symbolic validation, and shadow survey evaluation.
- A real `OpenEvolve` island engine for optimizing a schema-aware MegaPersona genome through LLM mutation.

Core goal: generate diverse, valid, controllable MegaPersona populations that cover personality space while maintaining behavioral consistency on held-out shadow surveys.

The current system uses three distinct model stages:

- `mutator_model`: evolves parent genome into child genome
- `persona_model`: generates structured MegaPersona outputs
- `simulator_model`: simulates shadow-survey behavior

Persona generation pipeline modes:

- `--persona-pipeline five_agent`: decomposed 5-call architecture with staged sections.
- `--persona-pipeline single_call`: integrated 1-call architecture that writes the
  complete persona at once.
- `--persona-pipeline compact`: legacy alias for `single_call`.

Treat pipeline mode as an experimental condition. Do not assume `five_agent`
is inherently more scientific; compare architectures under the same seeds,
survey splits, simulator, and model when the research question is generation
architecture.

## Concurrency Policy

For this repository, prefer **maximum practical concurrency** by default,
especially for smoke tests and provider sanity checks. The user explicitly
prefers fast completion over conservative worker settings.

Worker flags:

- `--candidate-max-workers`: parallel candidate evaluations
- `--persona-max-workers`: parallel persona generations inside one candidate
- `--shadow-max-workers`: parallel shadow-survey calls

Guideline:

- Smoke runs should bias toward finishing quickly.
- Do not default to `1` or `2` workers unless debugging, reproducing a race,
  or handling provider instability.
- `shadow-max-workers` is often the dominant runtime lever.
- Also remember that machine sleep can dominate perceived slowdown.

## Current Architecture

```text
SlotSampler (axis spread + quota buckets)
    -> MegaPersona Generator (five_agent or single_call architecture)
    -> Symbolic Validator (schema legality, anti-duplicate checks)
    -> Shadow Survey Simulator (persona -> Likert responses -> behavior axes)
    -> Evaluation: schema_fitness x behavior_coverage x shadow_alignment x generation_rate
       + internal consistency / axis alignment diagnostics
    -> OpenEvolve Engine (island-based evolution over schema-aware MegaPersona JSON genomes)
    -> Sealed test evaluation for final best candidate only
```

Important: the project no longer uses a separate custom "Open-Evolve style" main loop for MegaPersona. The official evolution flow is now:

```text
scripts/run_mega_persona_evolution.py
  -> src.mega_persona.openevolve_adapter.MegaPersonaOpenEvolveRunner
  -> src.open_evolve.engine.OpenEvolve
  -> src.mega_persona.evolution.MegaPersonaEvolver as evaluator/artifact backend only
```

`MegaPersonaEvolver.run()` is intentionally retired and raises an error. Do not re-enable it.

## What Actually Evolves

The project does not evolve arbitrary Python source files. It evolves a
schema-aware JSON genome under a fixed generation/evaluation pipeline.

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

This means the project can now tolerate primary-axis renaming or schema
reorganization much better than earlier versions. New conversations should not
assume legacy axis names are hard requirements everywhere in the stack.

Important status update from 2026-06-27: `Genome v3` is now the default.
It no longer only evolves prompt addenda. Each genome builds a per-slot
`generation_blueprint` with `blueprint_from_slot()`, and the generator injects
that blueprint into the selected generation pipeline, hard constraints, and
lightweight blueprint critic.

Genome v3 evolves:

- blueprint policy;
- axis-expression policy;
- cross-agent binding rules;
- behavior prediction anchors for ambiguity, peer pressure, failure feedback,
  and deadlines;
- critic policy for missing blueprint echoes and cross-field contradictions.

Old Genome v2 checkpoints still normalize forward, but new experiments should
be described as Genome v3 unless they intentionally load old artifacts.

## Negative Result To Preserve

Latest diagnostic run:

```text
data/results/mega_persona_openai_genome_v2_n16_g20_20260626
```

Observed:

- baseline fitness: `0.335616`
- best fitness: `0.343820`
- relative gain: `+2.44%`
- best appeared at `gen5`; no later generation improved it through `gen20`
- schema / coverage / diversity moved slightly upward, while validation
  alignment, axis alignment, consistency, and slot coverage moved slightly
  downward.

Interpretation:

- Do not describe this as successful evolution.
- The per-metric deltas are very small and likely noise-scale.
- Changing the aggregate fitness formula alone is not enough if individual
  metrics do not move meaningfully.
- The likely bottleneck is the evolved object: current operators mostly provide
  prompt advice rather than changing the generator's causal mechanism.

Genome v3 validation design:

- First test one operator at a time using `--fixed-operator`.
- Keep all other conditions fixed: `n`, seed, survey split, simulator, model,
  and worker settings.
- Judge success by per-metric deltas and non-degradation constraints, not by
  aggregate fitness alone.
- Promising operators should produce target-metric movement beyond noise while
  preserving alignment, consistency, and slot coverage.
- Specifically check whether v3 blueprint fields create visible changes in
  generated personas and shadow behavior, not merely tiny random metric drift.

## Key Modules

### `src/mega_persona/`

| File | Purpose |
|---|---|
| `schema.py` | Canonical `MegaPersona` dataclass |
| `slots.py` | `SlotSampler`, quota buckets, axis-level targets, schema binding helpers |
| `prompts.py` | Multi-agent prompts for cognition, values, social, mental health, demographics |
| `generator.py` | `MegaPersonaGenerator`, `five_agent` and `single_call` LLM generation architectures |
| `validator.py` | Symbolic schema validation and anti-duplicate checks |
| `consistency.py` | Internal consistency and axis-alignment scoring |
| `template_generator.py` | Non-LLM baseline generator |
| `shadow_survey.py` | Scientific scale registry and train/validation/test shadow survey splits |
| `shadow_simulator.py` | LLM role-play simulator for Likert survey responses |
| `evaluation.py` | Schema-level diversity and coverage metrics |
| `experiment.py` | Batch experiment runner and shared score formula |
| `evolution.py` | Genome utilities, operator definitions, evaluator backend, artifact store |
| `openevolve_adapter.py` | Adapter from MegaPersona genomes to `OpenEvolve` code strings |
| `visualization.py` | Static result visualization |
| `html_viz.py` | Interactive HTML visualization |

### `src/open_evolve/`

Shared OpenEvolve island engine. The MegaPersona main evolution runner uses `src.open_evolve.engine.OpenEvolve` directly.

### `src/evaluator/`

Diversity metrics such as coverage, convex hull, distances, and KL-style distribution measures.

### `src/utils/`

Config loading, LLM client, logging, async queue, and output helpers.

## Main Scripts

```bash
# Generate MegaPersonas without evolution
python scripts/generate_mega_personas.py --n 10 --seed 17 --mock

# Run end-to-end batch experiment
python scripts/run_mega_persona_experiment.py --n 25 --seeds 17,23 --mode mock

# Canonical MegaPersona evolution via OpenEvolve
python scripts/run_mega_persona_evolution.py --n 25 --generator-mode mock --generations 5

# Alias for the canonical OpenEvolve runner
python scripts/run_mega_persona_openevolve.py --n 25 --generator-mode mock --generations 5

# Controlled single-operator evolution experiment
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

# Fixed-parent operator ablation
python scripts/run_mega_persona_operator_ablation.py \
  --source-run data/results/<source_run> \
  --generator-mode mock \
  --n 6 \
  --replicates 2 \
  --mutation-scale 0.08

# Visualize results
python scripts/visualize_mega_persona_results.py <run_dir>
```

Small LLM OpenEvolve smoke run:

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

DeepSeek run:

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

Resume by increasing `--generations` and adding `--resume` with the same output directory.

## Tests

Core tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_evolution.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_openevolve_adapter.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_open_evolve.py
```

Other useful tests:

```bash
python scripts/test_mega_persona_schema.py
python scripts/test_mega_persona_generator.py
python scripts/test_mega_persona_experiment.py
python scripts/test_mega_persona_runner.py
python scripts/test_evaluator.py
python scripts/test_visualization.py
```

## Configuration

Main config:

```text
configs/default.yaml
```

Important model keys:

- `llm.mutator_model`: mutation/operator model
- `llm.persona_model`: MegaPersona generation model
- `llm.simulator_model`: shadow survey simulator model

For provider-specific runs, see `llm.providers.<provider>` in
`configs/default.yaml`. DeepSeek now fails fast if API key or base URL is
missing.

For future coding-agent decisions, if the task is “run a quick smoke” or
“verify the provider/integration path”, choose the high-concurrency variant by
default unless the user asks for a conservative run.

## Scoring Formula

```text
score =
  schema_fitness
  x (0.5 + 0.5 x behavior_coverage)
  x (0.5 + 0.5 x shadow_alignment)
  x generation_rate
```

This multiplicative gated formula prevents candidates with collapsed behavior or poor shadow alignment from scoring highly only because their schema is valid.

Newer runs also expose:

- `internal_consistency.mean`
- `internal_consistency_min.mean`
- `axis_alignment.mean`

## Train / Validation / Test Rule

Evolution selection uses train and validation shadow behavior only.

The test split is sealed. It should only be used for the final selected best candidate. Do not tune prompts, operators, or selection logic using test metrics.

Shadow survey splits are frozen and hashed at run start so every candidate is evaluated against identical train/validation/test sets.

## Evolution Operators

Current operator IDs from `EVOLUTION_PROMPT_OPERATORS`:

- `op01_axis_decoupling`
- `op02_behavioral_evidence`
- `op03_shadow_survey_alignment`
- `op04_within_bucket_contrast`
- `op05_failure_recovery_cycle`
- `op06_low_axis_fidelity`
- `op07_high_axis_cost`
- `op08_validation_conservatism`
- `op09_low_high_axis_tradeoff`
- `op10_contextual_bucket_split`
- `op11_decision_trace_evidence`
- `op12_support_network_asymmetry`
- `op13_autonomy_pressure_test`
- `op14_recovery_latency`
- `op15_survey_discriminating_cues`

Mutation modes:

- `parent_replay`
- `prompt_only`
- `operator_only`
- `mixed`
- `numeric_only`

## Latest Operator Evidence

Latest completed evolution report:

```text
docs/MEGA_PERSONA_BATCH_2026-06-16_OPENEVOLVE_MEDIUM.md
```

Main findings:

- Best candidate: `openevolve_000022_84f5f9724380`
- Best fitness: `0.373588`
- Baseline fitness: `0.342958`
- Improvement: about `+8.9%`
- `op06_low_axis_fidelity` is the strongest signal in the medium OpenEvolve run
- `op04_within_bucket_contrast` remains a stable positive operator
- `op07_high_axis_cost` and `op02_behavioral_evidence` remain worth testing
- `op01_axis_decoupling` is high risk and produced schema/alignment collapses
- `operator_only` is currently the strongest mutation mode
- `mixed` mode is high variance and should be used sparingly
- `numeric_only` is weak as a standalone mode

Recommended next search policy:

- Increase sampling weight for `op06_low_axis_fidelity` and `op04_within_bucket_contrast`
- Keep `op07_high_axis_cost` and `op02_behavioral_evidence` as medium-priority operators
- Downweight or temporarily remove `op01_axis_decoupling`
- Prefer `operator_only` and restrained `prompt_only`; lower `mixed` frequency

## 2026-07 MCTS And Metrics Update

Recent Genome v3 experiments added `hybrid_mcts` as a history-aware operator
selection policy. MCTS should be treated as a search strategy over mutation
operators, not as a replacement for the evaluator or the main fitness formula.

Current interpretation:

- MCTS can change the operator search path compared with random v3 sampling.
- One completed seed17 comparison showed about `+2.9%` validation fitness over
  the random v3 pool checkpoint, but sealed-test generalization and cross-seed
  stability are not yet proven.
- Strict-aware MCTS improved axis targeting and strict consistency in one
  n8/g15 run, but behavior coverage/diversity decreased.
- A later diversity-guard run did not recover test diversity, suggesting that
  simply adding penalties to reward is not enough.

Use refined metrics to diagnose search direction:

- `axis_target_mae`: generated persona vs target slot-axis error; lower is
  better.
- `shadow_mae`: shadow-survey behavior prediction error; lower is better.
- `strict_consistency_error`: strict cross-field consistency error; lower is
  better.
- `validation_behavior_coverage` and
  `validation_behavior_balanced_diversity`: behavior-space coverage and
  diversity; higher is generally better, but use them as soft search signals,
  not hard constraints.

Important:

- Do not imply that evolution must improve all metrics simultaneously. The
  refined metrics expose trade-offs and search bias.
- Recent runs show plateau behavior: best candidates often appear around gen5
  or gen9, then global best stops refreshing.
- `multi-objective elite archive` already exists. The next question is how to
  use existing strict/coverage/diversity elites during plateau escape.
- Prefer structured reward design over simple linear weighting: validity and
  generation reliability first, coverage/diversity soft protection second, then
  optimize fitness, shadow MAE, axis target MAE, and strict consistency. This
  is implemented as `--mcts-reward-profile structured` (plateau-aware layered
  reward with progress bonuses and reward standardization); `legacy` remains
  the default for historical comparability. `--mcts-reward-weight-mode deficit`
  additionally rotates per-metric reward weights toward lagging metrics.
  `--parent-selection objective_rotation` round-robins child parents across
  global/coverage/diversity/strict/shadow-MAE elites so all objective bests
  participate in mutation (supply-side multi-objective search).
  `scripts/run_multi_objective_sealed_test.py <run_dir>` audits finished runs
  by evaluating every multi-objective best candidate on the sealed test split
  (selection-rule diagnostic; test is still never used for selection).
  See `AGENTS.md` for details.

Latest concise report:

```text
docs/mega_persona_experiment_records/WEEKLY_REPORT_2026-07-14.md
```

## Documentation Priority

When updating the project, keep these files aligned:

1. `README.md`
2. `docs/TECHNICAL_ROUTE_REPORT.md`
3. `docs/MEGA_PERSONA_EXPERIMENT.md`
4. `AGENTS.md`
5. `CLAUDE.md`

## Output Layout

Canonical OpenEvolve runs:

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

Resume path:

```text
data/results/<run_name>/open_evolve/checkpoint.json
```

Operator ablation runs use a flatter layout with:

```text
ablation_summary.json
ablation_summary.md
checkpoint.json
candidates/
evaluations/
shadow_surveys/
```

## Development Conventions

1. Run scripts from the repository root.
2. Keep `train/validation/test` separation intact.
3. Do not use sealed test results for selection or prompt/operator tuning.
4. Do not reintroduce a second MegaPersona evolution mechanism.
5. Do not overwrite or delete user experiment outputs unless explicitly requested.
6. Update `README.md`, `AGENTS.md`, and relevant docs when changing CLI flags, output layout, or experimental protocol.
7. Prefer `rg` for code search.
8. Use `apply_patch` for manual code edits.

## Fast Orientation

For a new work session, read these first:

1. `AGENTS.md`
2. `README.md`
3. `docs/MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md`
4. `scripts/run_mega_persona_evolution.py`
5. `src/mega_persona/openevolve_adapter.py`
6. `src/open_evolve/engine.py`
