"""Prompt templates for the MegaPersona multi-agent pipeline."""

from src.mega_persona.schema import MegaPersona


MEGA_PERSONA_JSON_SCHEMA = MegaPersona.model_json_schema()


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
) -> str:
    """Prompt for the agent that replaces the old academic/teaching profile."""
    constraints_block = prior_constraints.strip() or "No extra adaptive constraints."
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

Required output key:
- demographics

Design requirements:
1. Use the target grade/stage constraint when present.
2. Give a plausible family context with resources, limits, and daily texture.
3. Avoid making region, family, or stage a deterministic explanation for all traits.
4. Keep the output valid for the MegaPersona schema.

Return valid JSON only, with exactly this top-level key."""


VALUES_IDENTITY_AGENT_SYSTEM_PROMPT = """You are the Values & Identity Agent.

Use the cognitive and motivational profile as causal context. Write values that
fit the person's motives and tensions rather than generic virtues. Include a
moral tension that can shape future behavior.
"""


def values_identity_agent_prompt(
    whiteboard_json: str,
    slot_context_json: str,
) -> str:
    return f"""Create ONLY the `values_identity` section for a MegaPersona.

Shared whiteboard:
```json
{whiteboard_json}
```

Target slot:
```json
{slot_context_json}
```

Required output key:
- values_identity

Design requirements:
1. Values must arise from cognition, motivation, and lived constraints.
2. Include a real moral tension, not a decorative contradiction.
3. Avoid making the person flawless or one-note.
4. Keep the output valid for the MegaPersona schema.

Return valid JSON only, with exactly this top-level key."""


SOCIAL_CREATIVE_AGENT_SYSTEM_PROMPT = """You are the Social & Creative Agent.

Write how the person relates, collaborates, expresses ideas, and creates under
constraints. Keep the profile consistent with cognition, motivation, values,
and target social-extraversion coordinates.
"""


def social_creative_agent_prompt(
    whiteboard_json: str,
    slot_context_json: str,
) -> str:
    return f"""Create ONLY the `social_creative_profile` section for a MegaPersona.

Shared whiteboard:
```json
{whiteboard_json}
```

Target slot:
```json
{slot_context_json}
```

Required output key:
- social_creative_profile

Design requirements:
1. Match the target social energy band when present.
2. Make creativity concrete: original, remixing, practical, aesthetic, strategic, or low-expression.
3. Explain peer influence without reducing the person to conformity.
4. Keep the output valid for the MegaPersona schema.

Return valid JSON only, with exactly this top-level key."""


MENTAL_HEALTH_AGENT_SYSTEM_PROMPT = """You are the Mental Health Context Agent.

Describe stress load, resilience, coping, risks, and protections without turning
the persona into a diagnosis. Keep mental health context behaviorally useful for
simulation and consistent with self-regulation.
"""


def mental_health_agent_prompt(
    whiteboard_json: str,
    slot_context_json: str,
) -> str:
    return f"""Create ONLY the `mental_health_context` section for a MegaPersona.

Shared whiteboard:
```json
{whiteboard_json}
```

Target slot:
```json
{slot_context_json}
```

Required output key:
- mental_health_context

Design requirements:
1. Match stress and resilience to the slot and self-regulation pattern.
2. Include both protective factors and risk factors.
3. Do not diagnose. Describe behaviorally useful context.
4. If stress is high and resilience is high, name concrete protective factors.
5. Keep the output valid for the MegaPersona schema.

Return valid JSON only, with exactly this top-level key."""


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
4. Return one complete MegaPersona JSON object.

Return valid JSON only."""
