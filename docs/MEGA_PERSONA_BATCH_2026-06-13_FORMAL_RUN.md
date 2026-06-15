# MegaPersona-Evolve Batch Report

- Date: 2026-06-13
- Batch: `mega_persona_formal_run`
- Output dir: `data/results/mega_persona_formal_run`
- Status: failed pilot batch; useful for pipeline debugging, not usable as scientific result.

## Run Setup

Command configuration recovered from `manifest.json`:

- generator mode: `llm`
- personas per seed: `n=25`
- seeds: `17,23,31`
- target generations: `5`
- population size: `8`
- children per generation: `6`
- elite count: `3`
- train shadow surveys: `12`
- validation shadow surveys: `4`
- sealed test shadow surveys: `4`
- items per shadow survey: `12`
- max workers: `1`

Expected full evaluation size per candidate:

- `25 personas/seed * 3 seeds = 75 personas/candidate`

## Observed Results

The run completed structurally, but no candidate produced a valid successful evaluation.

- evaluation records: `33`
- best candidate: `candidate_baseline`
- best fitness: `0.0000`
- valid generated personas observed in logs: `0`
- slot generations observed in logs: `607`
- slot validation failures after revision: `287`
- evaluations with saved `per_seed` persona payloads: `0`
- final sealed test: produced empty metrics because the selected best candidate also failed validation.

Error distribution across `evaluations/*/result.json`:

| Error type | Count |
|---|---:|
| `JSONDecodeError` | 28 |
| `AgentOutputError` | 2 |
| `APITimeoutError` | 1 |
| `APIConnectionError` | 1 |
| `TypeError` | 1 |

## Diagnosis

This batch should be treated as a pipeline failure, not as evidence about the evolution algorithm.

Main failure modes:

1. The LLM generation agents were asked to satisfy the MegaPersona schema, but the prompts did not explicitly include the exact enum values and length constraints for each section. This likely caused repeated schema violations even after revision.
2. A single malformed JSON response could abort the whole candidate evaluation. This erased partial seed-level diagnostics and made it hard to identify where the generation failed.
3. Per-slot validation issues were logged only as `issues=1`, without rule IDs or messages, so the run was not sufficiently auditable.
4. The final sealed test was attempted even when the selected candidate had zero validation fitness and no saved personas. That makes the artifact look more complete than it actually is.
5. The simulator already had a better fallback path than the persona generator did; generation needed the same kind of robust fallback.

## Fixes Applied After This Batch

Code changes now in the workspace:

- Added explicit per-agent schema contracts to `src/mega_persona/prompts.py`.
- Added JSON repair for malformed agent and revision outputs in `src/mega_persona/generator.py`.
- Added per-slot exception capture so one bad slot no longer aborts the whole candidate.
- Added detailed validation issue logging with rule ID, severity, and message.
- Added seed-level isolation in `src/mega_persona/evolution.py`, so one failed seed no longer discards diagnostics from other seeds.
- Added `generation_diagnostics` to per-seed result payloads.
- Added a sealed-test guard: if validation produced no positive successful seed, final test is skipped with status `skipped_no_successful_validation_candidate`.
- Kept simulator malformed JSON fallback in `src/mega_persona/shadow_simulator.py`.

Verification after fixes:

- `PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_generator.py`
- `PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_experiment.py`
- `PYTHONDONTWRITEBYTECODE=1 python scripts/test_mega_persona_evolution.py`
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile src/mega_persona/generator.py src/mega_persona/evolution.py src/mega_persona/prompts.py src/mega_persona/shadow_simulator.py`

All passed.

## Recommendation For Next Batch

Do not use this failed batch as the main scientific result. Keep it as `batch01_failed_pipeline_debug`.

Start a clean second batch after the fixes, with a smaller pilot first:

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --simulator-model-key llm.simulator_model \
  --n 8 \
  --seeds 17 \
  --generations 1 \
  --population-size 3 \
  --children-per-generation 2 \
  --elite-count 1 \
  --shadow-surveys 4 \
  --validation-shadow-surveys 2 \
  --test-shadow-surveys 2 \
  --items-per-shadow-survey 8 \
  --max-workers 1 \
  --output-dir data/results/mega_persona_pilot_batch02_20260613
```

Proceed to the larger formal run only if this pilot produces:

- validity rate above `0.6`
- at least one candidate with non-zero fitness
- non-empty `generation_diagnostics`
- non-skipped sealed test only after a successful validation-selected candidate

