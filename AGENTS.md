# AGENTS.md — MegaPersona-Evolve Project Context

Last updated: 2026-06-15

This file is the fast onboarding note for Codex or other coding agents. Read it before making changes, then inspect the specific files mentioned by the user.

## Project Goal

This repository implements **MegaPersona-Evolve**: a schema-constrained, coverage-guided synthetic persona generation experiment.

The project combines:

- DeepMind-style spatial coverage and evolutionary optimization.
- HACHIMI-style structured large persona schema, symbolic validation, and shadow survey evaluation.
- A real `OpenEvolve` island engine that optimizes MegaPersona prompt/genome variants.

The scientific goal is to generate persona populations that are:

- schema-valid;
- semantically non-duplicative;
- diverse across cognitive, motivational, self-regulation, social, and mental-health axes;
- behaviorally consistent under held-out shadow surveys.

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

## Key Files

| Path | Purpose |
|---|---|
| `scripts/run_mega_persona_evolution.py` | Canonical OpenEvolve runner for MegaPersona |
| `src/mega_persona/openevolve_adapter.py` | Bridges MegaPersona JSON genomes to `OpenEvolve` code strings |
| `src/open_evolve/engine.py` | Shared island-based OpenEvolve engine |
| `src/mega_persona/evolution.py` | Genome utilities, mutation operators, evaluator backend, persistent store |
| `src/mega_persona/generator.py` | 5-agent MegaPersona LLM generation pipeline |
| `src/mega_persona/shadow_survey.py` | Scientific shadow survey registry and train/validation/test split |
| `src/mega_persona/shadow_simulator.py` | LLM simulator for persona-to-Likert behavior |
| `src/mega_persona/experiment.py` | Shared score formula and batch experiment utilities |
| `scripts/run_mega_persona_operator_ablation.py` | Fixed-parent controlled operator ablation |
| `docs/MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md` | Latest completed operator ablation result |
| `docs/MEGA_PERSONA_TECHNICAL_REPORT_CN_2026-06-15.md` | Chinese technical report |

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

## Current Operator Evidence

Latest completed ablation:

```text
docs/MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md
```

Main findings:

- `op04_within_bucket_contrast` is the strongest and most stable positive operator.
- Best candidate: `ablation_0039_op04_within_bucket_contrast_mixed_r02`.
- Best fitness: `0.225122`.
- Parent replay mean: `0.205497`.
- Improvement: `+0.019626`, about 11.6 times the parent replay std.
- `mixed` mode gives the highest upside but has higher variance.
- `operator_only` is more conservative and can be useful as a stable branch.
- `numeric_only` is weak as a standalone mode.
- `op01_axis_decoupling` is high risk and caused schema/alignment collapses, including one 0-fitness candidate.

Recommended next search policy:

- Increase sampling weight for `op04_within_bucket_contrast`.
- Keep `op02_behavioral_evidence`, but add stricter guards or restrict its modes.
- Downweight or temporarily remove `op01_axis_decoupling` until it is rewritten.
- Prefer `mixed` for exploration and `operator_only` for stable branches.

## Common Commands

Small LLM smoke run:

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

## Quick Orientation For New Chats

If a new conversation starts, first read:

1. `AGENTS.md`
2. `README.md`
3. `docs/MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md`
4. `src/mega_persona/openevolve_adapter.py`
5. `scripts/run_mega_persona_evolution.py`

Then inspect the files relevant to the user's current request.
