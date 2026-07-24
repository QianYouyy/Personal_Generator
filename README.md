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
  -> MegaPersona LLM generator (five_agent or single_call architecture)
  -> Symbolic validator
  -> Shadow survey simulator
  -> Scientific fitness
  -> OpenEvolve island engine
  -> Sealed final test for the selected best candidate
```

The current pipeline uses three separate LLM roles:

```text
mutator_model
  -> OpenEvolve mutation stage
  -> parent genome -> child genome

persona_model
  -> MegaPersona generation stage
  -> slot-conditioned large persona JSON

simulator_model
  -> shadow-survey simulation stage
  -> behavior responses / behavior axes
```

OpenEvolve is therefore not just selecting among local rule mutations anymore.
The main path now uses an LLM mutator to evolve the generator genome, with
rule-based mutation kept only as a safe fallback when the mutator returns
invalid JSON.

Persona generation architecture is an experimental variable:

- `--persona-pipeline five_agent`: decomposed 5-call generator with staged
  demographics, cognition/motivation, values, social/creative, and mental-health sections.
- `--persona-pipeline single_call`: integrated 1-call generator that writes the
  complete persona in one pass.
- `--persona-pipeline compact`: legacy alias for `single_call`.

Do not assume the 5-agent architecture is inherently better just because it was
inspired by HACHIMI-style decomposition. Compare pipeline modes under matched
seeds, survey splits, simulator backend, and model settings.

The official evolution path is:

```text
scripts/run_mega_persona_evolution.py
  -> src.mega_persona.openevolve_adapter.MegaPersonaOpenEvolveRunner
  -> src.open_evolve.engine.OpenEvolve
  -> src.mega_persona.evolution.MegaPersonaEvolver as evaluator/artifact backend
```

`MegaPersonaEvolver.run()` is intentionally retired. Do not use or restore the old custom evolution loop.

## Schema-Aware Genome

The evolvable object is a schema-aware genome rather than arbitrary Python
source code. The fixed generation architecture remains stable while the genome
changes how the system samples, constrains, and prompts personas.

Current genome surface:

- `schema_binding`
  - `axis_names`
  - `axis_roles`
  - `quota_buckets`
- `quota_weights`
- `axis_bias`
- `axis_stretch`
- `prompt_profile`
- `last_evolution_operator`

`schema_binding` is the key recent upgrade. If the project renames or redefines
the primary persona axes, the genome, slot sampler, validator, shadow surveys,
simulators, and visualization layer can now follow the new axis binding
instead of silently assuming the legacy axis names.

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
  --persona-max-workers 4 \
  --shadow-max-workers 4 \
  --output-dir data/results/mega_persona_openevolve_smoke
```

Run a Stage-2 style schema-aware experiment with true LLM mutation:

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
  --candidate-max-workers 5 \
  --persona-max-workers 5 \
  --shadow-max-workers 10 \
  --output-dir data/results/mega_persona_stage2_schema_aware
```

Run with DeepSeek through the OpenAI-compatible API:

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
  --candidate-max-workers 5 \
  --persona-max-workers 5 \
  --shadow-max-workers 10 \
  --output-dir data/results/mega_persona_deepseek_stage2
```

When `--llm-provider deepseek` is enabled:

- `mutator_model` comes from `llm.providers.deepseek.mutator_model`
- `persona_model` comes from `llm.providers.deepseek.persona_model`
- `simulator_model` comes from `llm.providers.deepseek.simulator_model`

If DeepSeek API configuration is incomplete, the runner fails fast. Missing
`DEEPSEEK_API_KEY` or missing `api_base` is treated as an error rather than
falling back to OpenAI defaults.

Override DeepSeek model names when needed:

```bash
python scripts/run_mega_persona_evolution.py \
  --llm-provider deepseek \
  --mutator-model deepseek-v4-flash \
  --persona-model deepseek-v4-pro \
  --simulator-model deepseek-v4-flash \
  --generator-mode llm \
  --simulator-backend student-realistic \
  --n 8 \
  --seeds 17 \
  --generations 1 \
  --num-islands 2 \
  --output-dir data/results/mega_persona_deepseek_smoke
```

## Model Configuration

Stage-specific model defaults live in [configs/default.yaml](/Users/qianjun/Documents/programs/Personal_Generator/configs/default.yaml).

Current structure:

```yaml
llm:
  providers:
    openai:
      mutator_model: "gpt-5.4-mini"
      persona_model: "gpt-4o-mini"
      simulator_model: "gpt-4o-mini"
    deepseek:
      api_base: "https://api.deepseek.com"
      api_key_env: "DEEPSEEK_API_KEY"
      mutator_model: "deepseek-v4-flash"
      persona_model: "deepseek-v4-flash"
      simulator_model: "deepseek-v4-flash"
```

Recommended mental model:

- `mutator_model`: evolves the generator genome
- `persona_model`: executes the generator and writes personas
- `simulator_model`: evaluates personas through shadow-survey behavior

Use the Concordia-style shadow simulator backend:

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

Use the native Concordia agent/component backend:

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

Run the frozen three-simulator offline audit:

```bash
python scripts/run_mega_persona_simulator_audit.py \
  --persona-mode mock \
  --n 12 \
  --persona-seed 17 \
  --survey-seed 17017 \
  --shadow-surveys 3 \
  --items-per-shadow-survey 6 \
  --repeats 1 \
  --backends llm,concordia,concordia-native \
  --simulator-model-key llm.simulator_model \
  --shadow-max-workers 3 \
  --output-dir data/results/mega_persona_simulator_audit_smoke
```

See `docs/MEGA_PERSONA_SIMULATOR_AUDIT_PROTOCOL.md` for the strict protocol and
medium/formal settings.

Run the realistic student simulator audit, including the blind v2 baseline:

```bash
python scripts/run_mega_persona_simulator_audit.py \
  --persona-mode mock \
  --n 60 \
  --persona-seed 17 \
  --survey-seed 17017 \
  --shadow-surveys 8 \
  --items-per-shadow-survey 10 \
  --repeats 2 \
  --backends llm,concordia-native,student-realistic,student-realistic-v2 \
  --simulator-model-key llm.simulator_model \
  --shadow-max-workers 8 \
  --output-dir data/results/mega_persona_simulator_audit_student_realistic_v2_medium
```

`student-realistic-v2` keeps a mechanism trace (`trait_vector`,
`context_appraisal`, `student_state`, `response_style`, `item_mechanisms`) but
does not expose primary axes or item axis weights in the simulator prompt.

Use a stronger or third-party OpenAI-compatible simulator model:

```bash
export STRONG_SIMULATOR_API_KEY=...
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-backend concordia \
  --simulator-model <strong-simulator-model> \
  --simulator-api-base <https://openai-compatible-base/v1> \
  --simulator-api-key-env STRONG_SIMULATOR_API_KEY \
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
  --output-dir data/results/mega_persona_concordia_strong_sim
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

Internal consistency is also tracked in the current evaluator path:

- `internal_consistency.mean`
- `internal_consistency_min.mean`
- `axis_alignment.mean`

These metrics make it easier to distinguish “coverage through realistic
diversity” from “coverage through contradictory personas”.

## Experimental Rules

- Train and validation shadow surveys can be used for candidate evaluation.
- The test split is sealed and should only be used for the final selected best candidate.
- Shadow survey splits are frozen and hashed at run start.
- Do not use test metrics for operator tuning, prompt tuning, or model selection.

## Changing the Persona Schema

You can now change the primary persona axis names without rewriting the whole
experiment stack, as long as the new schema is expressed through
`schema_binding`.

The current implementation already propagates schema-aware axis bindings into:

- slot sampling
- adaptive constraints
- rule-based persona baseline
- LLM generation hard constraints
- validator axis checks
- shadow surveys
- student-realistic simulators
- Concordia native behavior calibration
- visualization / HTML report extraction

## Experimental Genome v4

Genome v3 remains the default for historical reproducibility. The experimental
Genome v4 path is enabled explicitly:

```bash
python scripts/run_mega_persona_evolution.py \
  --genome-version 4 \
  --operator-family v4 \
  --search-strategy openevolve \
  ...
```

V4 evolves a low-dimensional structured behavior program rather than free-form
prompt policy. Operators `op22` through `op27` each change exactly one module;
sampling weights/bias/stretch are fixed, and mutations do not call the mutator
LLM or apply v3 numeric jitter. First measure the evaluation noise floor and
run fixed-operator screens. Use MCTS only after these operators show repeatable
target-metric effects.

Candidate selection is stochastic in LLM mode. The default evolution protocol
uses one evaluation per candidate:
`--candidate-evaluation-repeats 1 --elite-confirmation-repeats 1`. This keeps
the search budget and selection rule simple. Use
`--candidate-evaluation-repeats 3` only for an explicitly repeated ablation;
repeat fitness values and per-metric mean/std/SEM are persisted, while test
results remain excluded from selection.

New-run sampling defaults are `persona temperature=0.45, top_p=0.85` and
`simulator temperature=0.05, top_p=0.80`. This reduces evaluator noise while
preserving generation diversity; both are exposed as CLI overrides.

## Historical Operator Evidence

Latest retained report:

```text
docs/MEGA_PERSONA_BATCH_2026-06-16_OPENEVOLVE_MEDIUM.md
```

Main result:

- The operator bank now has 15 operators, `op01` through `op15`.
- `op06_low_axis_fidelity` is the strongest signal in the medium OpenEvolve run.
- Best candidate fitness was `0.373588` versus baseline `0.342958`.
- `op04_within_bucket_contrast` remains a stable positive operator.
- `op07_high_axis_cost` and `op02_behavioral_evidence` remain worth testing.
- `op01_axis_decoupling` is high risk and should be downweighted or rewritten.
- `operator_only` is currently the strongest mutation mode.
- `mixed` mode has high variance and should be used sparingly.

## Tests

Core tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_evolution.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_openevolve_adapter.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_open_evolve.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_mcts_policy.py
```

Additional tests:

```bash
python scripts/test_mega_persona_schema.py
python scripts/test_mega_persona_generator.py
python scripts/test_mega_persona_experiment.py
python scripts/test_mega_persona_runner.py
python scripts/test_evaluator.py
python scripts/test_visualization.py
python scripts/test_multi_objective_sealed_test.py
```

Noise-floor measurement:

```bash
python scripts/measure_evaluation_noise_floor.py \
  --source-run data/results/<run> \
  --repeats 5
```

## Agent Context

For new coding-agent conversations, read:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `docs/MEGA_PERSONA_BATCH_2026-06-15_OPERATOR_ABLATION_V2_FINAL.md`
4. `scripts/run_mega_persona_evolution.py`
5. `src/mega_persona/openevolve_adapter.py`
