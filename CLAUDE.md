# CLAUDE.md - Personal Generator / MegaPersona-Evolve

Last updated: 2026-06-15

This file gives Claude a fast project overview. For coding-agent onboarding, also read `AGENTS.md`; it is the most compact current context file.

## Project Overview

This project implements **MegaPersona-Evolve**: a schema-constrained and coverage-guided synthetic persona generation experiment.

It combines:

- DeepMind-style spatial coverage and evolutionary optimization.
- HACHIMI-style structured large persona schemas, symbolic validation, and shadow survey evaluation.
- A real `OpenEvolve` island engine for optimizing MegaPersona prompt/genome variants.

Core goal: generate diverse, valid, controllable MegaPersona populations that cover personality space while maintaining behavioral consistency on held-out shadow surveys.

## Current Architecture

```text
SlotSampler (axis spread + quota buckets)
    -> MegaPersona Generator (5-agent parallel LLM pipeline)
    -> Symbolic Validator (schema legality, anti-duplicate checks)
    -> Shadow Survey Simulator (persona -> Likert responses -> behavior axes)
    -> Evaluation: schema_fitness x behavior_coverage x shadow_alignment x generation_rate
    -> OpenEvolve Engine (island-based evolution over MegaPersona JSON genomes)
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

## Key Modules

### `src/mega_persona/`

| File | Purpose |
|---|---|
| `schema.py` | Canonical `MegaPersona` dataclass |
| `slots.py` | `SlotSampler`, quota buckets, axis-level targets |
| `prompts.py` | Multi-agent prompts for cognition, values, social, mental health, demographics |
| `generator.py` | `MegaPersonaGenerator`, 5-agent LLM pipeline with shared whiteboard |
| `validator.py` | Symbolic schema validation and anti-duplicate checks |
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
  --n 10 \
  --seeds 17,23 \
  --generations 3 \
  --population-size 4 \
  --children-per-island 1 \
  --shadow-surveys 3 \
  --validation-shadow-surveys 2 \
  --test-shadow-surveys 2 \
  --items-per-shadow-survey 8 \
  --shadow-max-workers 4 \
  --output-dir data/results/mega_persona_openevolve_smoke
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

## Scoring Formula

```text
score =
  schema_fitness
  x (0.5 + 0.5 x behavior_coverage)
  x (0.5 + 0.5 x shadow_alignment)
  x generation_rate
```

This multiplicative gated formula prevents candidates with collapsed behavior or poor shadow alignment from scoring highly only because their schema is valid.

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

Mutation modes:

- `parent_replay`
- `prompt_only`
- `operator_only`
- `mixed`
- `numeric_only`

## Latest Operator Evidence

Latest completed ablation:

```text
docs/MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md
```

Main findings:

- Best candidate: `ablation_0039_op04_within_bucket_contrast_mixed_r02`
- Best fitness: `0.225122`
- Parent replay mean fitness: `0.205497`
- Improvement: `+0.019626`, about 11.6x parent replay std
- `op04_within_bucket_contrast` is the most stable positive operator
- `op02_behavioral_evidence` has high upside but higher failure risk
- `op01_axis_decoupling` is high risk and produced schema/alignment collapses
- `mixed` mode has the highest upside but high variance
- `operator_only` is more conservative and useful as a stable branch
- `numeric_only` is weak as a standalone mode

Recommended next search policy:

- Increase sampling weight for `op04_within_bucket_contrast`
- Keep `op02_behavioral_evidence`, but add stricter guardrails or mode restrictions
- Downweight or temporarily remove `op01_axis_decoupling`
- Prefer `mixed` for exploration and `operator_only` for stable branches

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
