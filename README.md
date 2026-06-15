# MegaPersona-Evolve

Schema-constrained, coverage-guided MegaPersona generation with OpenEvolve optimization.

This project is no longer a general DeepMind persona-generator reproduction. It is now a clean MegaPersona experiment focused on generating diverse, valid, behaviorally consistent large personas.

## Goal

Generate MegaPersona populations that are:

- valid under a structured schema;
- diverse across cognitive, motivational, self-regulation, social, and mental-health axes;
- non-duplicative;
- behaviorally aligned under held-out shadow surveys;
- optimizable through the shared `OpenEvolve` island engine.

## Architecture

```text
SlotSampler
  -> MegaPersona 5-agent LLM generator
  -> Symbolic validator
  -> Shadow survey simulator
  -> Scientific fitness
  -> OpenEvolve island engine
  -> Sealed final test for the selected best candidate
```

The official evolution path is:

```text
scripts/run_mega_persona_evolution.py
  -> src.mega_persona.openevolve_adapter.MegaPersonaOpenEvolveRunner
  -> src.open_evolve.engine.OpenEvolve
  -> src.mega_persona.evolution.MegaPersonaEvolver as evaluator/artifact backend
```

`MegaPersonaEvolver.run()` is intentionally retired. Do not use or restore the old custom evolution loop.

## Repository Layout

```text
configs/
  default.yaml
data/
  generated_personas/
  questionnaires/
  results/
docs/
  MEGA_PERSONA_EXPERIMENT.md
  MEGA_PERSONA_TECHNICAL_REPORT_CN_2026-06-15.md
  MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md
  TECHNICAL_ROUTE_REPORT.md
scripts/
  generate_mega_personas.py
  run_mega_persona_experiment.py
  run_mega_persona_evolution.py
  run_mega_persona_openevolve.py
  run_mega_persona_operator_ablation.py
  visualize_mega_persona_results.py
src/
  evaluator/
  mega_persona/
  open_evolve/
  utils/
```

## Main Commands

Generate MegaPersonas without evolution:

```bash
python scripts/generate_mega_personas.py --n 10 --seed 17 --mock
```

Run a batch experiment:

```bash
python scripts/run_mega_persona_experiment.py --n 25 --seeds 17,23 --mode mock
```

Run canonical OpenEvolve optimization:

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

Resume an existing OpenEvolve run:

```bash
python scripts/run_mega_persona_evolution.py \
  ...same args... \
  --generations 5 \
  --output-dir data/results/mega_persona_openevolve_smoke \
  --resume
```

Run operator ablation:

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

Visualize a result directory:

```bash
python scripts/visualize_mega_persona_results.py data/results/<run_name>
```

## Output Layout

Canonical OpenEvolve runs write:

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

## Scoring

The shared scientific score is:

```text
score =
  schema_fitness
  x (0.5 + 0.5 x behavior_coverage)
  x (0.5 + 0.5 x shadow_alignment)
  x generation_rate
```

This gates high schema validity with behavioral coverage and persona-behavior alignment.

## Experimental Rules

- Train and validation shadow surveys can be used for candidate evaluation.
- The test split is sealed and should only be used for the final selected best candidate.
- Shadow survey splits are frozen and hashed at run start.
- Do not use test metrics for operator tuning, prompt tuning, or model selection.

## Current Operator Evidence

Latest retained report:

```text
docs/MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md
```

Main result:

- `op04_within_bucket_contrast` is the strongest and most stable positive operator.
- Best candidate fitness was `0.225122` versus parent replay mean `0.205497`.
- `op02_behavioral_evidence` has high upside but needs guardrails.
- `op01_axis_decoupling` is high risk and should be downweighted or rewritten.
- `mixed` mode has high upside but high variance.
- `operator_only` is useful as a stable branch.

## Tests

Core tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_evolution.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_openevolve_adapter.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_open_evolve.py
```

Additional tests:

```bash
python scripts/test_mega_persona_schema.py
python scripts/test_mega_persona_generator.py
python scripts/test_mega_persona_experiment.py
python scripts/test_mega_persona_runner.py
python scripts/test_evaluator.py
python scripts/test_visualization.py
```

## Agent Context

For new coding-agent conversations, read:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `docs/MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md`
4. `scripts/run_mega_persona_evolution.py`
5. `src/mega_persona/openevolve_adapter.py`
