# MegaPersona-Evolve Experiment Design

## Research Goal

This experiment studies whether schema-constrained large personas can be
generated with both high semantic quality and broad coverage over a target
behavioral space.

The project treats HACHIMI as a reference for structured, validated large
personas and DeepMind's persona generator as a reference for coverage-guided
evaluation. It does not attempt to replicate either project directly.

## Core Claim

A useful large-persona generator should satisfy four properties at the same
time:

1. Schema validity: generated personas obey a structured psychological schema.
2. Coverage: generated personas spread across chosen primary axes.
3. Non-duplication: personas are not superficial rewrites of one another.
4. Behavioral measurability: personas produce distinct and aligned responses on
   shadow surveys.

## Experimental Pipeline

```text
SlotSampler
  -> quota buckets + Sobol primary-axis coordinates
MegaPersonaGenerator
  -> demographics
  -> cognition and motivation
  -> values and identity
  -> social and creativity
  -> mental health
  -> symbolic validation and optional revision
ShadowSurvey
  -> non-academic construct items
ShadowSimulator
  -> Likert responses and behavior-axis scores
Evaluation
  -> validity, coverage, distance, duplicate penalty, behavior alignment
```

## Primary Axes

The MVP uses three axes:

| Axis | Meaning | Source |
|---|---|---|
| `cognitive_abstraction` | Concrete-to-abstract reasoning tendency | Thinking style |
| `motivation_autonomy` | Self-authored motivation vs external pressure | Motivation system |
| `self_regulation_resilience` | Planning, persistence, recovery, coping | Self-regulation + mental health |

These axes are intentionally low-dimensional. The full schema stays rich, but
coverage is computed over a compact target space to avoid dimensional collapse.

## Shadow Surveys

The initial shadow surveys are original, non-academic Likert items. They are
construct-oriented rather than copied from published instruments.

Construct families:

- cognitive abstraction
- ambiguity tolerance
- autonomous and intrinsic motivation
- external pressure sensitivity
- self-regulation and metacognition
- emotional regulation
- stress and recovery
- identity clarity and value tension
- belonging, peer influence, creativity, risk, help-seeking

## Baselines

| Mode | Command | Purpose |
|---|---|---|
| `dry-run` | `scripts/generate_mega_personas.py --dry-run` | Inspect slots and surveys only |
| `mock` | `scripts/generate_mega_personas.py --mock` | Offline rule-based baseline |
| `llm` | `scripts/generate_mega_personas.py --model-key llm.persona_model` | Fixed multi-agent LLM generator |

The `mock` mode is not the final method. It is a reproducible lower-bound
baseline for checking that metrics and artifacts work before spending LLM calls.

## Metrics

### Schema Metrics

- `validity_rate`: proportion of generated personas passing schema and hard rules
- `near_duplicate_rate`: pairwise lexical near-duplicate rate
- `coverage`: random-ball coverage in primary-axis space
- `avg_dist`: average pairwise distance
- `min_dist`: nearest-neighbor separation
- `kl_divergence`: negative KL divergence against uniform bins

### Behavior Metrics

- `shadow_alignment`: agreement between persona primary axes and shadow-survey behavior axes
- `behavior_coverage`: coverage of simulated behavior-axis points
- `persona_behavior_mae`: per-axis absolute error between persona axes and behavior axes

### Experiment Score

The MVP score is a gated product (used by both batch experiments and evolution):

```text
experiment_score =
  schema_fitness
  * (0.5 + 0.5 * behavior_coverage)
  * (0.5 + 0.5 * shadow_alignment)
  * generation_rate
```

This makes invalid, duplicated, or behaviorally collapsed generations hard to
score highly.  Each gate floors at 0.5 so the optimization landscape retains
gradient for evolution.

The implementation lives in `src/mega_persona/experiment.compute_experiment_score`
and is reused by `src/mega_persona/evolution.genome_score`.

## Repeatable Batch Experiment

Run the offline baseline:

```bash
python scripts/run_mega_persona_experiment.py \
  --mode mock \
  --n 25 \
  --seeds 17,23,31 \
  --shadow-surveys 12 \
  --items-per-shadow-survey 12
```

Run the LLM generator:

```bash
python scripts/run_mega_persona_experiment.py \
  --mode llm \
  --n 25 \
  --seeds 17 \
  --model-key llm.persona_model
```

Outputs:

- `summary.json`: machine-readable results
- `summary.md`: human-readable experiment report

## Development Status

| Component | Status |
|---|---|
| Schema | Implemented |
| Validator | Implemented |
| Slot sampler | Implemented |
| Shadow surveys | Implemented |
| Rule-based shadow simulator | Implemented |
| Rule-based persona baseline | Implemented |
| Fixed multi-agent LLM generator | Implemented |
| Batch experiment runner | Implemented |
| Durable Open-Evolve optimization | Implemented MVP |
| Result visualization | Implemented MVP |

## Durable Open-Evolve MVP

The current MegaPersona evolution loop uses a restricted JSON genome instead of
free-form code mutation. This is intentional: the architecture should remain
stable while evolution tunes the experimental surface.

Current evolvable fields:

- quota bucket weights
- primary-axis bias
- primary-axis stretch
- shadow survey seed offset
- prompt profile for LLM mode

The score aggregation weights are fixed. This prevents a candidate from
improving by changing the evaluation ruler rather than improving the generated
population.

Fitness is computed with held-out shadow surveys. Training shadow surveys are
still persisted for analysis, but the final candidate score uses held-out
behavior coverage and held-out persona-behavior alignment.

Run:

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode mock \
  --n 25 \
  --seeds 17,23,31 \
  --generations 20 \
  --population-size 8 \
  --max-workers 4 \
  --heldout-shadow-surveys 4 \
  --output-dir data/results/mega_persona_evolution_run
```

Resume after interruption:

```bash
python scripts/run_mega_persona_evolution.py \
  --n 25 \
  --seeds 17,23,31 \
  --generations 20 \
  --population-size 8 \
  --max-workers 4 \
  --heldout-shadow-surveys 4 \
  --output-dir data/results/mega_persona_evolution_run \
  --resume
```

Run evolution with the LLM generator and evolved prompt profiles:

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode llm \
  --model-key llm.persona_model \
  --n 10 \
  --seeds 17 \
  --generations 5 \
  --population-size 4 \
  --max-workers 1 \
  --heldout-shadow-surveys 4 \
  --output-dir data/results/mega_persona_llm_evolution_run
```

Persistence layout:

```text
data/results/mega_persona_evolution_run/
  manifest.json
  checkpoint.json
  final_summary.json
  final_summary.md
  candidates/
    candidate_*.json
  generations/
    generation_0000.json
    generation_0001.json
    ...
  evaluations/
    eval_000001_candidate_x/
      result.json
    eval_000002_candidate_y/
      result.json
  figures/
    fitness_over_generations.png
    best_slot_axes.png
    best_persona_axes.png
    best_behavior_axes.png
    best_genome.png
    best_metrics.png
```

Each `result.json` stores the candidate genome, per-seed slots, generated
personas, schema metrics, train/held-out shadow behavior, train/held-out
behavior diversity, and shadow survey responses. The checkpoint is written after
every candidate evaluation, so the run can continue after process death, machine
sleep, or network interruption.

`manifest.json` stores the command, config, Python version, git commit, branch,
dirty flag, and `git status --short` output. Mock-mode evolution can safely use
`--max-workers > 1`; LLM mode should usually start with `--max-workers 1` to
avoid rate limits.

Visualize an evolution run, batch `summary.json`, or single generation JSON:

```bash
python scripts/visualize_mega_persona_results.py \
  --input data/results/mega_persona_evolution_run
```

The visualization script writes PNG figures for target slot coverage, generated
persona axes, shadow behavior axes, evolution fitness, best genome parameters,
and summary metrics.

## Next Experimental Extension

The genome now controls coarse prompt-profile fragments for LLM mode and the
current run artifacts can be visualized. The next extension is to let evolution
use more granular agent-specific prompt fragments, add a stronger statistical
report, and compare the rule-based shadow simulator with an LLM shadow simulator.
