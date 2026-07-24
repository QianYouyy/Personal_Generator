"""Prompt templates for the MegaPersona multi-agent pipeline."""

from src.mega_persona.schema import (
    ASPIRATION_MAX_LENGTH,
    DERIVED_REASONING_MAX_LENGTH,
    FAMILY_CONTEXT_MAX_LENGTH,
    IDENTITY_ANCHOR_MAX_LENGTH,
    MENTAL_HEALTH_NARRATIVE_MAX_LENGTH,
    MORAL_TENSION_MAX_LENGTH,
    SOCIAL_NARRATIVE_MAX_LENGTH,
)
from src.mega_persona.schema import MegaPersona


MEGA_PERSONA_JSON_SCHEMA = MegaPersona.model_json_schema()


DEMOGRAPHICS_SECTION_CONTRACT = f"""\
Schema contract for `demographics`:
{{
  "demographics": {{
    "age": integer from 10 to 30,
    "grade_or_stage": one of ["middle_school", "high_school", "vocational", "undergraduate", "graduate", "early_career"],
    "region_type": one of ["urban", "suburban", "rural", "migrant", "international"],
    "family_context": string, 20-{FAMILY_CONTEXT_MAX_LENGTH} characters
  }}
}}"""


COGNITION_SECTION_CONTRACT = f"""\
Schema contract for cognition output:
{{
  "cognitive_motivation_profile": {{
    "thinking_style": {{
      "dominant_mode": one of ["analytical", "intuitive", "associative", "practical", "reflective", "exploratory"],
      "abstraction_level": number 0-1,
      "ambiguity_tolerance": number 0-1,
      "evidence_preference": one of ["data", "authority", "experience", "peer_consensus", "personal_values"],
      "typical_blind_spot": string, 10-300 characters
    }},
    "motivation_system": {{
      "primary_drive": one of ["mastery", "achievement", "belonging", "autonomy", "security", "recognition", "curiosity", "avoidance"],
      "secondary_drive": string, 3-120 characters,
      "intrinsic_motivation": number 0-1,
      "external_pressure_sensitivity": number 0-1,
      "failure_sensitivity": number 0-1,
      "reward_preference": one of ["praise", "progress", "grades", "independence", "social_status", "usefulness"]
    }},
    "learning_orientation": {{
      "goal_orientation": one of ["mastery", "performance", "avoidance", "mixed"],
      "preferred_learning_mode": one of ["reading", "discussion", "practice", "visual", "project_based", "imitation", "trial_and_error"],
      "attention_pattern": one of ["sustained", "bursty", "easily_shifted", "hyperfocused"],
      "curiosity_scope": one of ["narrow_deep", "broad_shallow", "situational", "low"],
      "help_seeking_style": one of ["proactive", "reluctant", "peer_first", "adult_first", "avoids_help"]
    }},
    "self_regulation": {{
      "planning_style": one of ["structured", "reactive", "deadline_driven", "ritual_based", "chaotic"],
      "persistence": number 0-1,
      "emotional_regulation": number 0-1,
      "metacognition": number 0-1,
      "habit_stability": number 0-1
    }},
    "challenge_response": {{
      "under_difficulty": one of ["doubles_down", "seeks_help", "freezes", "reframes", "distracts", "rebels"],
      "under_criticism": one of ["defensive", "curious", "ashamed", "motivated", "dismissive"],
      "under_success": one of ["confident", "complacent", "anxious_to_maintain", "generous", "exploratory"]
    }},
    "decision_pattern": {{
      "risk_appetite": number 0-1,
      "time_horizon": one of ["short", "medium", "long"],
      "tradeoff_style": one of ["safe_choice", "optimize_score", "protect_identity", "seek_growth", "maintain_relationships"],
      "typical_rationale": string, 20-500 characters
    }},
    "narrative": string, 250-1800 characters
  }},
  "derived_academic_tendency": {{
    "likely_performance_band": one of ["poor", "low", "mid", "high"],
    "reasoning": string, 40-{DERIVED_REASONING_MAX_LENGTH} characters
  }}
}}"""


VALUES_SECTION_CONTRACT = f"""\
Schema contract for `values_identity`:
{{
  "values_identity": {{
    "core_values": list of 2-6 strings,
    "identity_anchor": string, 20-{IDENTITY_ANCHOR_MAX_LENGTH} characters,
    "moral_tension": string, 20-{MORAL_TENSION_MAX_LENGTH} characters,
    "aspiration": string, 20-{ASPIRATION_MAX_LENGTH} characters
  }}
}}"""


SOCIAL_SECTION_CONTRACT = f"""\
Schema contract for `social_creative_profile`:
{{
  "social_creative_profile": {{
    "social_energy": number 0-1,
    "collaboration_style": one of ["leader", "supporter", "observer", "challenger", "mediator", "solo"],
    "expressiveness": number 0-1,
    "creative_mode": one of ["original", "remixing", "practical", "aesthetic", "strategic", "low_expression"],
    "peer_influence_sensitivity": number 0-1,
    "narrative": string, 120-{SOCIAL_NARRATIVE_MAX_LENGTH} characters
  }}
}}"""


MENTAL_HEALTH_SECTION_CONTRACT = f"""\
Schema contract for `mental_health_context`:
{{
  "mental_health_context": {{
    "stress_load": number 0-1,
    "resilience": number 0-1,
    "coping_style": one of ["problem_solving", "emotional_support", "avoidance", "humor", "control", "withdrawal"],
    "protective_factors": list of 1-5 strings,
    "risk_factors": list of 1-5 strings,
    "narrative": string, 120-{MENTAL_HEALTH_NARRATIVE_MAX_LENGTH} characters
  }}
}}"""


COGNITION_MOTIVATION_AGENT_SYSTEM_PROMPT = """You are the Cognition & Motivation Agent in a multi-agent persona pipeline.

Your job is not to write an academic resume. Your job is to explain the person's
inner operating system:
- how they notice, interpret, and reason about problems
- what drives or blocks action
- how they learn, ask for help, regulate effort, and recover from setbacks
- how these mechanisms imply, but do not mechanically determine, likely performance

Write psychologically coherent people, not scorecards. Avoid saying someone is
"smart" or "bad at school" as a root cause. Explain mechanisms: attention,
motivation, resource access, identity protection, fear of failure, curiosity,
habits, and stress regulation.
"""


def cognition_motivation_agent_prompt(
    whiteboard_json: str,
    target_axes_json: str,
    prior_constraints: str = "",
    hard_constraints: str = "",
) -> str:
    """Prompt for the agent that replaces the old academic/teaching profile."""
    constraints_block = prior_constraints.strip() or "No extra adaptive constraints."
    hc_block = hard_constraints.strip() or ""
    return f"""You are filling ONLY the `cognitive_motivation_profile` and
`derived_academic_tendency` sections of a MegaPersona JSON object.

Shared whiteboard:
```json
{whiteboard_json}
```

Target primary axes for this persona:
```json
{target_axes_json}
```

Adaptive constraints:
{constraints_block}

Hard constraints (MUST satisfy):
{hc_block}

{COGNITION_SECTION_CONTRACT}

Required output keys:
- cognitive_motivation_profile
- derived_academic_tendency

Design requirements:
1. Treat academic performance as a derived tendency, not the person's essence.
2. Make the target axes visible in mechanism, not by repeating numbers.
3. Include at least one limitation, blind spot, or tension.
4. If the person has high resilience but low likely performance, explain the barrier.
5. If the person has high likely performance, ground it in motivation, habits, resources, or strategy.
6. Do not write a flawless person. Avoid all-high or all-low trait patterns.
7. The narrative must explain how they think, why they act, and what happens when they are blocked.
8. Respect all field length limits exactly. If needed, compress phrasing rather than exceeding the schema budget.
9. Keep `derived_academic_tendency.reasoning` under {DERIVED_REASONING_MAX_LENGTH} characters.

Return valid JSON only, with exactly these two top-level keys."""


DEMOGRAPHICS_AGENT_SYSTEM_PROMPT = """You are the Demographics Agent in a multi-agent persona pipeline.

Write concise, concrete demographic context that constrains the rest of the
persona. Do not overdetermine the person from background alone. The output
must be compatible with the target slot and must avoid stereotypes.
"""


def demographics_agent_prompt(slot_context_json: str) -> str:
    return f"""Create ONLY the `demographics` section for a MegaPersona.

Target slot:
```json
{slot_context_json}
```

{DEMOGRAPHICS_SECTION_CONTRACT}

Required output key:
- demographics

Design requirements:
1. Use the target grade/stage constraint when present.
2. Give a plausible family context with resources, limits, and daily texture.
3. Avoid making region, family, or stage a deterministic explanation for all traits.
4. Keep the output valid for the MegaPersona schema.
5. Keep `family_context` under {FAMILY_CONTEXT_MAX_LENGTH} characters. Prefer concise, information-dense writing.

Return valid JSON only, with exactly this top-level key."""


VALUES_IDENTITY_AGENT_SYSTEM_PROMPT = """You are the Values & Identity Agent.

Use the cognitive and motivational profile as causal context. Write values that
fit the person's motives and tensions rather than generic virtues. Include a
moral tension that can shape future behavior.
"""


def values_identity_agent_prompt(
    whiteboard_json: str,
    slot_context_json: str,
    hard_constraints: str = "",
) -> str:
    hc_block = f"\nHard constraints (MUST satisfy):\n{hard_constraints}" if hard_constraints.strip() else ""
    return f"""Create ONLY the `values_identity` section for a MegaPersona.

Shared whiteboard (relevant fields only):
```json
{whiteboard_json}
```

Target slot:
```json
{slot_context_json}
```{hc_block}

{VALUES_SECTION_CONTRACT}

Required output key:
- values_identity

Design requirements:
1. Values must arise from cognition, motivation, and lived constraints.
2. Include a real moral tension, not a decorative contradiction.
3. Avoid making the person flawless or one-note.
4. Keep the output valid for the MegaPersona schema.
5. Keep `identity_anchor` under {IDENTITY_ANCHOR_MAX_LENGTH} characters, `moral_tension` under {MORAL_TENSION_MAX_LENGTH}, and `aspiration` under {ASPIRATION_MAX_LENGTH}.

Return valid JSON only, with exactly this top-level key."""


SOCIAL_CREATIVE_AGENT_SYSTEM_PROMPT = """You are the Social & Creative Agent.

Write how the person relates, collaborates, expresses ideas, and creates under
constraints. Keep the profile consistent with cognition, motivation, values,
and target social-extraversion coordinates.
"""


def social_creative_agent_prompt(
    whiteboard_json: str,
    slot_context_json: str,
    hard_constraints: str = "",
) -> str:
    hc_block = f"\nHard constraints (MUST satisfy):\n{hard_constraints}" if hard_constraints.strip() else ""
    return f"""Create ONLY the `social_creative_profile` section for a MegaPersona.

Shared whiteboard (relevant fields only):
```json
{whiteboard_json}
```

Target slot:
```json
{slot_context_json}
```{hc_block}

{SOCIAL_SECTION_CONTRACT}

Required output key:
- social_creative_profile

Design requirements:
1. Match the target social energy band when present.
2. Make creativity concrete: original, remixing, practical, aesthetic, strategic, or low-expression.
3. Explain peer influence without reducing the person to conformity.
4. Keep the output valid for the MegaPersona schema.
5. Keep `social_creative_profile.narrative` under {SOCIAL_NARRATIVE_MAX_LENGTH} characters. Use one compact narrative, not a long essay.

Return valid JSON only, with exactly this top-level key."""


MENTAL_HEALTH_AGENT_SYSTEM_PROMPT = """You are the Mental Health Context Agent.

Describe stress load, resilience, coping, risks, and protections without turning
the persona into a diagnosis. Keep mental health context behaviorally useful for
simulation and consistent with self-regulation.
"""


def mental_health_agent_prompt(
    whiteboard_json: str,
    slot_context_json: str,
    hard_constraints: str = "",
) -> str:
    hc_block = f"\nHard constraints (MUST satisfy):\n{hard_constraints}" if hard_constraints.strip() else ""
    return f"""Create ONLY the `mental_health_context` section for a MegaPersona.

Shared whiteboard (relevant fields only):
```json
{whiteboard_json}
```

Target slot:
```json
{slot_context_json}
```{hc_block}

{MENTAL_HEALTH_SECTION_CONTRACT}

Required output key:
- mental_health_context

Design requirements:
1. Match stress and resilience to the slot and self-regulation pattern.
2. Include both protective factors and risk factors.
3. Do not diagnose. Describe behaviorally useful context.
4. If stress is high and resilience is high, name concrete protective factors.
5. Keep the output valid for the MegaPersona schema.
6. Keep `mental_health_context.narrative` under {MENTAL_HEALTH_NARRATIVE_MAX_LENGTH} characters.

Return valid JSON only, with exactly this top-level key."""


COMPACT_PERSONA_SYSTEM_PROMPT = """You are a compact MegaPersona generator.

Generate one complete, schema-valid MegaPersona object in a single call. This
is a first-class generation architecture for experiments that test whether a
single integrated writer can produce stronger cross-field coherence than a
decomposed multi-agent pipeline. Prioritize internal consistency, target-axis
expression, behavior prediction evidence, and strict JSON validity.
"""


def compact_persona_prompt(
    slot_context_json: str,
    blueprint_json: str,
    prompt_addendum: str = "",
) -> str:
    addendum_block = prompt_addendum.strip() or "No evolved addendum."
    return f"""Create ONE complete MegaPersona JSON object.

Target slot:
```json
{slot_context_json}
```

Genome v3 generation blueprint:
```json
{blueprint_json}
```

Evolved policy addendum:
{addendum_block}

Required top-level keys:
- persona_id
- demographics
- cognitive_motivation_profile
- derived_academic_tendency
- values_identity
- social_creative_profile
- mental_health_context

Section contracts:
{DEMOGRAPHICS_SECTION_CONTRACT}

{COGNITION_SECTION_CONTRACT}

{VALUES_SECTION_CONTRACT}

{SOCIAL_SECTION_CONTRACT}

{MENTAL_HEALTH_SECTION_CONTRACT}

Design requirements:
1. `persona_id` must equal the target slot id.
2. Express each target axis through concrete mechanisms, not labels or copied numbers.
3. Keep one strongest-weakest axis tension visible across cognition, values, social behavior, and mental health.
4. Include behavior-predictive evidence for ambiguity, peer pressure, feedback, and deadline contexts.
5. Do not create an all-high or all-low person; include at least one boundary or cost.
6. Respect every field length limit exactly. Compress prose instead of exceeding schema limits.
7. Return valid JSON only. No markdown, no commentary, no extra top-level keys."""


def stage_repair_prompt(
    *,
    stage_name: str,
    raw_stage_json: str,
    required_keys_json: str,
    context_json: str,
) -> str:
    return f"""Repair the output for stage `{stage_name}`.

The stage output is missing required top-level keys or has the wrong shape.
Return exactly one JSON object containing all required keys and no commentary.

Required top-level keys:
```json
{required_keys_json}
```

Relevant context:
```json
{context_json}
```

Original stage output:
```json
{raw_stage_json}
```

Repair requirements:
1. Preserve the original content as much as possible.
2. Add only the missing required section(s), using the context and schema contracts.
3. Keep all field values inside schema ranges and length limits.
4. Return valid JSON only."""


def revision_prompt(candidate_json: str, issues_json: str) -> str:
    return f"""Repair this MegaPersona JSON so it satisfies the schema and hard rules.

Current candidate:
```json
{candidate_json}
```

Validation issues:
```json
{issues_json}
```

Repair requirements:
1. Preserve the same persona_id and overall target profile.
2. Fix only what is needed to satisfy the issues.
3. Avoid all-high and all-low numeric trait patterns.
4. Respect every field length limit exactly; compress instead of deleting core meaning.
5. Return one complete MegaPersona JSON object.

Return valid JSON only."""
