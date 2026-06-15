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

The shadow-survey protocol follows the HACHIMI paper's external-evaluation
logic: personas are instantiated as agents and asked CEPS/PISA-style survey
items, then behavior is scored at the construct level. The code uses
construct-faithful local items tagged with scientific scale metadata rather
than redistributing verbatim CEPS/PISA questionnaire text.

Current non-academic scale families:

| Source | Scale IDs | Role in this project |
|---|---|---|
| PISA 2022 | `CURIOAGR`, `GROSAGR` | curiosity, growth mindset, autonomous motivation |
| PISA 2022 | `CREATEFF`, `CREATOP` | creative self-efficacy, openness to intellect |
| PISA 2022 | `RELATST`, `BELONG`, `BULLIED` | relationships, belonging, social threat |
| PISA 2022 | `PSYCHSYM`, `LIFESAT`, `WORKHOME` | distress, life satisfaction, workload/balance |
| CEPS | `CESD`, `TEACHREL`, `PEERREL`, `MISBEHAVIOR` | depressive symptoms, teacher/peer relations, behavioral regulation |

The official HACHIMI paper also uses academic PISA constructs such as
`MATHEFF` and `MATHEF21`; this project intentionally excludes those from the
main non-academic shadow-survey score.

Evolution uses frozen splits:

- `train`: persisted for analysis and future ablations
- `validation`: used for fitness and candidate selection
- `test`: sealed during evolution; evaluated only once for the final selected best candidate

The shadow simulator receives narrative/categorical persona evidence only. It
does not receive hidden numeric primary-axis scores, reducing circular
alignment inflation.

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
- prompt profile for LLM mode

The score aggregation weights are fixed. This prevents a candidate from
improving by changing the evaluation ruler rather than improving the generated
population.

Fitness is computed with frozen validation shadow surveys. Training shadow
surveys are persisted for analysis. Test shadow surveys are frozen and hashed at
run start, but they are not evaluated during candidate selection; after the best
candidate is selected, the system runs one sealed test evaluation and writes
`final_test_report.json`. The survey splits are independent of candidate
genomes, so evolution cannot improve by changing validation or test questions.

Run:

```bash
python scripts/run_mega_persona_evolution.py \
  --generator-mode mock \
  --n 25 \
  --seeds 17,23,31 \
  --generations 20 \
  --population-size 8 \
  --children-per-island 1 \
  --validation-shadow-surveys 4 \
  --test-shadow-surveys 4 \
  --output-dir data/results/mega_persona_evolution_run
```

Resume after interruption:

```bash
python scripts/run_mega_persona_evolution.py \
  --n 25 \
  --seeds 17,23,31 \
  --generations 20 \
  --population-size 8 \
  --children-per-island 1 \
  --validation-shadow-surveys 4 \
  --test-shadow-surveys 4 \
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
  --children-per-island 1 \
  --validation-shadow-surveys 4 \
  --test-shadow-surveys 4 \
  --output-dir data/results/mega_persona_llm_evolution_run
```

Persistence layout:

```text
data/results/mega_persona_evolution_run/
  manifest.json
  final_summary.json
  final_summary.md
  open_evolve/
    checkpoint.json
    checkpoint_gen_*.json
    elite_codes_gen_*/
  mega_eval/
    checkpoint.json
    final_summary.json
    final_test_report.json
    shadow_surveys/
      train.json
      validation.json
      test.json
      hashes.json
    candidates/
      candidate_*.json
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

Each candidate `result.json` stores the candidate genome, per-seed slots,
generated personas, schema metrics, frozen survey hashes, train/validation
shadow behavior, train/validation behavior diversity, and train/validation
shadow survey responses. It deliberately excludes test behavior. The selected
best candidate's sealed test behavior is stored only in `final_test_report.json`.
The checkpoint is written after every candidate evaluation, so the run can
continue after process death, machine sleep, or network interruption.

`manifest.json` stores the command, config, frozen survey hashes, Python
version, git commit, branch, dirty flag, and `git status --short` output.
OpenEvolve evaluates children inside each island. LLM mode should usually start
with `--children-per-island 1` and use `--shadow-max-workers` for parallelizing
shadow survey simulation inside each candidate.

Visualize an evolution run, batch `summary.json`, or single generation JSON:

```bash
python scripts/visualize_mega_persona_results.py \
  --input data/results/mega_persona_evolution_run
```

The visualization script writes PNG figures for target slot coverage, generated
persona axes, shadow behavior axes, evolution fitness, best genome parameters,
and summary metrics.

## Next Experimental Extension

The genome now controls prompt-profile fragments for LLM mode and records one
selected evolution operator from a 20-item mutation prompt bank on each mutated
candidate. Current run artifacts can be visualized. The next extension is to add
a stronger statistical report and compare operator families across repeated
formal runs.
