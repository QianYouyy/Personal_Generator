"""Native Concordia adapter for MegaPersona shadow survey simulation.

This module keeps the MegaPersona experiment contract intact while routing the
simulation through Concordia's native EntityAgent/component lifecycle:

MegaPersona -> context components + memory component -> EntityAgent.observe()
-> EntityAgent.act() -> survey JSON.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from concordia.agents.entity_agent import EntityAgent
from concordia.language_model import language_model
from concordia.typing import entity
from concordia.typing import entity_component

from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_survey import ShadowSurvey
from src.mega_persona.slots import AXIS_NAMES, axis_roles_for_target_axes


CONCORDIA_NATIVE_SYSTEM_PROMPT = """\
You are the acting component inside a Concordia agent used for psychological
research. Use the agent's components, memories, observations, values, goals,
and current survey scene to decide how this person would answer.

Scientific calibration:
- Do not make the agent generically prosocial, resilient, curious, or growth
  minded unless the persona evidence supports it.
- Use weaknesses, blind spots, stress load, risk factors, peer pressure, and
  criticism reactions as strongly as strengths and aspirations.
- Similar survey items may receive different answers when the current scene
  activates different memories or pressures.
- Preserve between-person differences. A low-regulation or high-stress agent
  should sometimes disagree with positive self-regulation items.
- For motivation-related items, distinguish autonomous internal drive from
  effort caused by praise, grades, belonging needs, security needs, or fear of
  failure. Hard work is not automatically high autonomy.
- Avoid response collapse: use the full 1-5 scale when warranted by the
  persona, including 1/2 for clear mismatch and 5 for clear fit.

Return ONLY valid JSON mapping every item_id to an integer 1-5:
1=strongly disagree, 2=disagree, 3=neutral, 4=agree, 5=strongly agree.
Do not include explanations, markdown fences, or extra text."""


@dataclass(frozen=True)
class ConcordiaAgentBundle:
    """A constructed Concordia agent plus traceable metadata."""

    agent: EntityAgent
    component_names: tuple[str, ...]
    memories: tuple[str, ...]


class LLMClientLanguageModel(language_model.LanguageModel):
    """Adapter from the project's LLMClient shape to Concordia LanguageModel."""

    def __init__(
        self,
        llm_client,
        *,
        system_prompt: str = CONCORDIA_NATIVE_SYSTEM_PROMPT,
        temperature: float = 0.25,
    ):
        self._llm_client = llm_client
        self._system_prompt = system_prompt
        self._temperature = temperature

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators=language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        del terminators, top_p, top_k, timeout, seed
        effective_temperature = self._temperature if temperature is None else temperature
        return self._llm_client.generate(
            prompt,
            system_prompt=self._system_prompt,
            temperature=effective_temperature,
            max_tokens=max_tokens,
        )

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        del seed
        choice_prompt = (
            f"{prompt}\n\nChoose exactly one option from this list and return only the option text:\n"
            + "\n".join(f"- {response}" for response in responses)
        )
        raw = self.sample_text(choice_prompt, max_tokens=64).strip()
        for index, response in enumerate(responses):
            if raw == response or response in raw:
                return index, response, {"raw": raw}
        return 0, responses[0], {"raw": raw, "fallback": True}


class StaticContextComponent(entity_component.ContextComponent):
    """Static Concordia context component populated from MegaPersona fields."""

    def __init__(self, title: str, text: str):
        self._title = title
        self._text = text

    def pre_act(self, action_spec: entity.ActionSpec) -> str:
        del action_spec
        return f"{self._title}:\n{self._text}"

    def get_state(self) -> entity_component.ComponentState:
        return {"title": self._title, "text": self._text}

    def set_state(self, state: entity_component.ComponentState) -> None:
        self._title = str(state.get("title", self._title))
        self._text = str(state.get("text", self._text))


class EpisodicMemoryComponent(entity_component.ContextComponent):
    """Small native memory component with observations and persona memories."""

    def __init__(self, memories: Sequence[str], max_recent_observations: int = 3):
        self._memories = list(memories)
        self._observations: list[str] = []
        self._max_recent_observations = max_recent_observations

    def pre_act(self, action_spec: entity.ActionSpec) -> str:
        del action_spec
        recent = self._observations[-self._max_recent_observations :]
        sections = ["Relevant autobiographical memories:"]
        sections.extend(f"- {memory}" for memory in self._memories)
        if recent:
            sections.append("Recent observations:")
            sections.extend(f"- {observation}" for observation in recent)
        return "\n".join(sections)

    def pre_observe(self, observation: str) -> str:
        self._observations.append(observation)
        return "Observation stored in episodic memory."

    def get_state(self) -> entity_component.ComponentState:
        return {
            "memories": list(self._memories),
            "observations": list(self._observations),
            "max_recent_observations": self._max_recent_observations,
        }

    def set_state(self, state: entity_component.ComponentState) -> None:
        self._memories = [str(item) for item in state.get("memories", self._memories)]
        self._observations = [
            str(item) for item in state.get("observations", self._observations)
        ]
        self._max_recent_observations = int(
            state.get("max_recent_observations", self._max_recent_observations)
        )


class SurveyActingComponent(entity_component.ActingComponent):
    """Concordia acting component that answers a shadow survey as JSON."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        *,
        max_tokens: int = 1500,
        temperature: float = 0.25,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._last_prompt = ""
        self._last_action = ""

    def get_action_attempt(
        self,
        context: entity_component.ComponentContextMapping,
        action_spec: entity.ActionSpec,
    ) -> str:
        agent_name = self.get_entity().name
        context_text = "\n\n".join(
            f"[{name}]\n{value}" for name, value in context.items() if value
        )
        call_to_action = action_spec.call_to_action.format(name=agent_name, timedelta="")
        prompt = f"""\
CONCORDIA AGENT: {agent_name}

COMPONENT CONTEXT:
{context_text}

GAME MASTER REQUEST:
{call_to_action}

CALIBRATION RULES:
- First decide whether each item matches, conflicts with, or is only weakly
  supported by this exact agent's memories and components.
- Penalize idealized self-presentation. Answer as behavior under the survey
  context, not as a best possible version of the agent.
- Use negative evidence explicitly: blind spots, risk factors, high stress,
  poor habit stability, peer influence, and reactions to criticism can justify
  lower scores.
- For growth, curiosity, creativity, and effort items, check whether motivation
  is internally owned or externally pressured. Recognition/security/avoidance
  motives should produce more mixed autonomy answers.
- Keep answers differentiated across items. Do not reuse one general attitude
  for the whole survey.
- Return every item exactly once.

Return only the JSON object."""
        self._last_prompt = prompt
        self._last_action = self._model.sample_text(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return self._last_action

    def get_state(self) -> entity_component.ComponentState:
        return {
            "last_prompt": self._last_prompt,
            "last_action": self._last_action,
        }

    def set_state(self, state: entity_component.ComponentState) -> None:
        self._last_prompt = str(state.get("last_prompt", self._last_prompt))
        self._last_action = str(state.get("last_action", self._last_action))


def build_concordia_agent_bundle(
    persona: MegaPersona,
    llm_client,
    *,
    axis_names: tuple[str, ...] = AXIS_NAMES,
    axis_roles: dict[str, str] | None = None,
    temperature: float = 0.25,
    max_tokens: int = 1500,
) -> ConcordiaAgentBundle:
    """Build a native Concordia EntityAgent for one MegaPersona."""
    model = LLMClientLanguageModel(llm_client, temperature=temperature)
    memories = tuple(persona_to_concordia_memories(persona))
    context_components: dict[str, entity_component.ContextComponent] = {
        "identity": StaticContextComponent("Identity", _identity_context(persona)),
        "cognition": StaticContextComponent("Cognition", _cognition_context(persona)),
        "motivation": StaticContextComponent("Motivation", _motivation_context(persona)),
        "social_context": StaticContextComponent("Social context", _social_context(persona)),
        "health_context": StaticContextComponent("Health context", _health_context(persona)),
        "behavior_calibration": StaticContextComponent(
            "Behavior calibration",
            _behavior_calibration_context(persona, axis_names=axis_names, axis_roles=axis_roles),
        ),
        "memory": EpisodicMemoryComponent(memories),
    }
    act_component = SurveyActingComponent(
        model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    agent = EntityAgent(
        agent_name=persona.persona_id,
        act_component=act_component,
        context_components=context_components,
    )
    return ConcordiaAgentBundle(
        agent=agent,
        component_names=tuple(context_components.keys()),
        memories=memories,
    )


def survey_observation(survey: ShadowSurvey) -> str:
    return (
        f"Game master opens a shadow survey scene.\n"
        f"Survey id: {survey.survey_id}\n"
        f"Context: {survey.context}\n"
        "The agent should answer honestly as themselves, including the ways "
        "their habits, stressors, blind spots, and social pressures can pull "
        "them away from their ideals."
    )


def survey_action_spec(survey: ShadowSurvey) -> entity.ActionSpec:
    items_text = "\n".join(f'  "{item.item_id}": "{item.text}"' for item in survey.items)
    return entity.free_action_spec(
        call_to_action=f"""\
The game master asks {{name}} to answer this survey as themselves.

Use the agent's components and memories. Treat each item as a behavioral
claim under the current scene, not as a socially desirable identity statement.

For each item, silently calibrate:
- 1: clearly unlike this agent in this scene
- 2: somewhat unlike this agent
- 3: mixed, uncertain, or context-dependent
- 4: somewhat like this agent
- 5: clearly like this agent

Motivation-autonomy rule:
- High scores require internal interest or ownership.
- Effort driven mainly by praise, grades, belonging, security, recognition, or
  fear of failure should not be treated as high autonomy.
- If the item asks about growth, curiosity, creative confidence, or continuing
  effort, balance intrinsic motivation against external pressure sensitivity
  and failure sensitivity.

Return a JSON object mapping every item_id to an integer 1-5.

ITEMS:
{items_text}""",
        tag="shadow_survey",
    )


def persona_to_concordia_memories(persona: MegaPersona) -> list[str]:
    profile = persona.cognitive_motivation_profile
    values = persona.values_identity
    social = persona.social_creative_profile
    health = persona.mental_health_context
    return [
        f"Family context: {persona.demographics.family_context}",
        f"Identity anchor: {values.identity_anchor}",
        f"Moral tension: {values.moral_tension}",
        f"Aspiration: {values.aspiration}",
        f"Primary drive: {profile.motivation_system.primary_drive}",
        f"Intrinsic motivation: {profile.motivation_system.intrinsic_motivation:.2f}",
        f"External pressure sensitivity: {profile.motivation_system.external_pressure_sensitivity:.2f}",
        f"Failure sensitivity: {profile.motivation_system.failure_sensitivity:.2f}",
        f"Reward preference: {profile.motivation_system.reward_preference}",
        f"Typical blind spot: {profile.thinking_style.typical_blind_spot}",
        f"Attention pattern: {profile.learning_orientation.attention_pattern}",
        f"Help-seeking style: {profile.learning_orientation.help_seeking_style}",
        f"Under difficulty: {profile.challenge_response.under_difficulty}",
        f"Under criticism: {profile.challenge_response.under_criticism}",
        f"After success: {profile.challenge_response.under_success}",
        f"Cognitive-motivation narrative: {profile.narrative}",
        f"Social and creative narrative: {social.narrative}",
        f"Peer influence sensitivity: {social.peer_influence_sensitivity:.2f}",
        f"Stress load: {health.stress_load:.2f}",
        f"Resilience: {health.resilience:.2f}",
        f"Risk factors: {', '.join(health.risk_factors)}",
        f"Coping style: {health.coping_style}",
        f"Mental-health narrative: {health.narrative}",
        f"Derived academic tendency: {persona.derived_academic_tendency.reasoning}",
    ]


def _identity_context(persona: MegaPersona) -> str:
    values = persona.values_identity
    return "\n".join(
        [
            f"Age/stage: {persona.demographics.age}, {persona.demographics.grade_or_stage}",
            f"Region type: {persona.demographics.region_type}",
            f"Core values: {', '.join(values.core_values)}",
            f"Identity anchor: {values.identity_anchor}",
            f"Aspiration: {values.aspiration}",
        ]
    )


def _cognition_context(persona: MegaPersona) -> str:
    profile = persona.cognitive_motivation_profile
    thinking = profile.thinking_style
    learning = profile.learning_orientation
    decision = profile.decision_pattern
    challenge = profile.challenge_response
    return "\n".join(
        [
            f"Dominant thinking mode: {thinking.dominant_mode}",
            f"Evidence preference: {thinking.evidence_preference}",
            f"Typical blind spot: {thinking.typical_blind_spot}",
            f"Goal orientation: {learning.goal_orientation}",
            f"Learning mode: {learning.preferred_learning_mode}",
            f"Attention pattern: {learning.attention_pattern}",
            f"Decision time horizon: {decision.time_horizon}",
            f"Tradeoff style: {decision.tradeoff_style}",
            f"Typical rationale: {decision.typical_rationale}",
            f"Under difficulty: {challenge.under_difficulty}",
            f"Under criticism: {challenge.under_criticism}",
            f"After success: {challenge.under_success}",
        ]
    )


def _motivation_context(persona: MegaPersona) -> str:
    motivation = persona.cognitive_motivation_profile.motivation_system
    regulation = persona.cognitive_motivation_profile.self_regulation
    return "\n".join(
        [
            f"Primary drive: {motivation.primary_drive}",
            f"Secondary drive: {motivation.secondary_drive}",
            f"Intrinsic motivation: {motivation.intrinsic_motivation:.2f}",
            f"External pressure sensitivity: {motivation.external_pressure_sensitivity:.2f}",
            f"Failure sensitivity: {motivation.failure_sensitivity:.2f}",
            f"Reward preference: {motivation.reward_preference}",
            f"Planning style: {regulation.planning_style}",
            f"Persistence: {regulation.persistence:.2f}",
            f"Emotional regulation: {regulation.emotional_regulation:.2f}",
            f"Metacognition: {regulation.metacognition:.2f}",
            f"Habit stability: {regulation.habit_stability:.2f}",
        ]
    )


def _social_context(persona: MegaPersona) -> str:
    social = persona.social_creative_profile
    return "\n".join(
        [
            f"Social energy: {social.social_energy:.2f}",
            f"Collaboration style: {social.collaboration_style}",
            f"Expressiveness: {social.expressiveness:.2f}",
            f"Creative mode: {social.creative_mode}",
            f"Peer influence sensitivity: {social.peer_influence_sensitivity:.2f}",
            f"Narrative: {social.narrative}",
        ]
    )


def _health_context(persona: MegaPersona) -> str:
    health = persona.mental_health_context
    return "\n".join(
        [
            f"Stress load: {health.stress_load:.2f}",
            f"Resilience: {health.resilience:.2f}",
            f"Coping style: {health.coping_style}",
            f"Protective factors: {', '.join(health.protective_factors)}",
            f"Risk factors: {', '.join(health.risk_factors)}",
            f"Narrative: {health.narrative}",
        ]
    )


def _behavior_calibration_context(
    persona: MegaPersona,
    *,
    axis_names: tuple[str, ...] = AXIS_NAMES,
    axis_roles: dict[str, str] | None = None,
) -> str:
    profile = persona.cognitive_motivation_profile
    motivation = profile.motivation_system
    regulation = profile.self_regulation
    social = persona.social_creative_profile
    health = persona.mental_health_context
    roles = axis_roles or axis_roles_for_target_axes({axis: 0.5 for axis in axis_names})
    axes = persona.primary_axes(axis_names=axis_names, axis_roles=roles)
    cognitive_axis = roles.get("cognitive_core", axis_names[0])
    motivation_axis = roles.get("motivation_core", axis_names[min(1, len(axis_names) - 1)])
    regulation_axis = roles.get("regulation_core", axis_names[min(2, len(axis_names) - 1)])
    low_evidence: list[str] = []
    if axes[cognitive_axis] < 0.35:
        low_evidence.append(
            f"low {cognitive_axis}: avoid assuming broad curiosity or comfort with ambiguous methods"
        )
    if axes[motivation_axis] < 0.35:
        low_evidence.append(
            f"low {motivation_axis}: avoid assuming strong self-directed persistence"
        )
    if motivation.intrinsic_motivation < 0.40:
        low_evidence.append("low intrinsic motivation: growth/curiosity items need mixed or low answers unless context creates interest")
    if motivation.external_pressure_sensitivity > 0.60:
        low_evidence.append("high external pressure sensitivity: effort may reflect compliance, fear, or approval seeking rather than autonomy")
    if motivation.failure_sensitivity > 0.60:
        low_evidence.append("high failure sensitivity can turn challenge items into avoidance or defensive responses")
    if motivation.primary_drive in {"achievement", "belonging", "security", "recognition", "avoidance"}:
        low_evidence.append(
            f"primary drive is {motivation.primary_drive}: do not equate effort with internal ownership"
        )
    if axes[regulation_axis] < 0.35:
        low_evidence.append(
            f"low {regulation_axis}: pressure can lower follow-through and recovery"
        )
    if regulation.habit_stability < 0.45:
        low_evidence.append("unstable habits can reduce rule-following and planning consistency")
    if health.stress_load > 0.60:
        low_evidence.append("high stress load can reduce positive coping and life-satisfaction responses")
    if social.peer_influence_sensitivity > 0.60:
        low_evidence.append("peer pressure can shift behavior away from stated goals")

    evidence_text = "\n".join(f"- {item}" for item in low_evidence) or "- no strong low-end warning flags"
    return "\n".join(
        [
            "Use these numeric tendencies only to calibrate behavioral likelihood, not to copy scores.",
            "Primary axes: "
            + ", ".join(f"{axis}={axes[axis]:.2f}" for axis in axis_names),
            f"Motivation evidence: primary_drive={motivation.primary_drive}, "
            f"intrinsic_motivation={motivation.intrinsic_motivation:.2f}, "
            f"external_pressure_sensitivity={motivation.external_pressure_sensitivity:.2f}, "
            f"failure_sensitivity={motivation.failure_sensitivity:.2f}, "
            f"reward_preference={motivation.reward_preference}",
            f"Persistence={regulation.persistence:.2f}, emotional_regulation={regulation.emotional_regulation:.2f}, "
            f"metacognition={regulation.metacognition:.2f}, habit_stability={regulation.habit_stability:.2f}",
            f"Stress load={health.stress_load:.2f}, resilience={health.resilience:.2f}, "
            f"peer influence sensitivity={social.peer_influence_sensitivity:.2f}",
            "Low-score evidence and response anchors:",
            evidence_text,
            "For motivation-autonomy items, reward-driven effort is not enough for a high score; high scores require owned interest or self-endorsed goals.",
            "A 4 or 5 requires direct positive evidence. A 2 or 1 is appropriate when the item conflicts with these warnings.",
        ]
    )


def pretty_agent_trace(bundle: ConcordiaAgentBundle) -> str:
    """Return a compact trace for docs/debugging."""
    return json.dumps(
        {
            "agent": bundle.agent.name,
            "components": list(bundle.component_names),
            "memories": list(bundle.memories),
        },
        ensure_ascii=False,
        indent=2,
    )
