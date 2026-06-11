"""Fixed multi-agent MegaPersona generator.

This module is the non-evolved baseline for the new experiment path. Later,
Open-Evolve should mutate prompt text, constraints, axis extraction, and
sampling strategies around this stable pipeline rather than rewriting the
entire architecture.
"""

from dataclasses import dataclass, field
import json
from typing import Any

from src.mega_persona.prompts import (
    COGNITION_MOTIVATION_AGENT_SYSTEM_PROMPT,
    DEMOGRAPHICS_AGENT_SYSTEM_PROMPT,
    MENTAL_HEALTH_AGENT_SYSTEM_PROMPT,
    SOCIAL_CREATIVE_AGENT_SYSTEM_PROMPT,
    VALUES_IDENTITY_AGENT_SYSTEM_PROMPT,
    cognition_motivation_agent_prompt,
    demographics_agent_prompt,
    mental_health_agent_prompt,
    revision_prompt,
    social_creative_agent_prompt,
    values_identity_agent_prompt,
)
from src.mega_persona.schema import MegaPersona
from src.mega_persona.slots import MegaPersonaSlot, SlotSampler
from src.mega_persona.validator import ValidationReport, validate_mega_persona


class AgentOutputError(ValueError):
    """Raised when an agent response cannot be parsed as usable JSON."""


@dataclass
class MegaPersonaGenerationResult:
    slot: MegaPersonaSlot
    persona: MegaPersona | None
    validation_report: ValidationReport
    raw_outputs: dict[str, str] = field(default_factory=dict)
    candidate_json: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.persona is not None and self.validation_report.is_valid


class MegaPersonaGenerator:
    """HACHIMI-style fixed pipeline for structured MegaPersona generation."""

    def __init__(
        self,
        llm_client,
        temperature: float = 0.45,
        max_tokens: int = 3000,
        max_revisions: int = 1,
        prompt_addendum: str = "",
    ):
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_revisions = max_revisions
        self.prompt_addendum = prompt_addendum.strip()

    def generate(
        self,
        n: int,
        seed: int | None = None,
    ) -> list[MegaPersonaGenerationResult]:
        """Sample slots and generate a population."""
        slots = SlotSampler().sample(n=n, seed=seed)
        return self.generate_from_slots(slots)

    def generate_from_slots(
        self,
        slots: list[MegaPersonaSlot],
    ) -> list[MegaPersonaGenerationResult]:
        return [self.generate_one(slot) for slot in slots]

    def generate_one(self, slot: MegaPersonaSlot) -> MegaPersonaGenerationResult:
        """Generate one MegaPersona through staged whiteboard updates."""
        whiteboard: dict[str, Any] = {
            "persona_id": slot.slot_id,
            "slot": slot.prompt_context(),
        }
        candidate: dict[str, Any] = {"persona_id": slot.slot_id}
        raw_outputs: dict[str, str] = {}

        slot_json = _json_dumps(slot.prompt_context())

        demographics = self._call_agent(
            stage_name="demographics",
            prompt=demographics_agent_prompt(slot_json),
            system_prompt=self._system_prompt(DEMOGRAPHICS_AGENT_SYSTEM_PROMPT),
            raw_outputs=raw_outputs,
        )
        self._merge_stage(candidate, demographics, ("demographics",))
        whiteboard.update(demographics)

        cognition = self._call_agent(
            stage_name="cognition_motivation",
            prompt=cognition_motivation_agent_prompt(
                whiteboard_json=_json_dumps(whiteboard),
                target_axes_json=_json_dumps(slot.target_axes),
                prior_constraints="\n".join(slot.adaptive_constraints),
            ),
            system_prompt=self._system_prompt(COGNITION_MOTIVATION_AGENT_SYSTEM_PROMPT),
            raw_outputs=raw_outputs,
        )
        self._merge_stage(
            candidate,
            cognition,
            ("cognitive_motivation_profile", "derived_academic_tendency"),
        )
        whiteboard.update(cognition)

        values = self._call_agent(
            stage_name="values_identity",
            prompt=values_identity_agent_prompt(
                whiteboard_json=_json_dumps(whiteboard),
                slot_context_json=slot_json,
            ),
            system_prompt=self._system_prompt(VALUES_IDENTITY_AGENT_SYSTEM_PROMPT),
            raw_outputs=raw_outputs,
        )
        self._merge_stage(candidate, values, ("values_identity",))
        whiteboard.update(values)

        social = self._call_agent(
            stage_name="social_creative",
            prompt=social_creative_agent_prompt(
                whiteboard_json=_json_dumps(whiteboard),
                slot_context_json=slot_json,
            ),
            system_prompt=self._system_prompt(SOCIAL_CREATIVE_AGENT_SYSTEM_PROMPT),
            raw_outputs=raw_outputs,
        )
        self._merge_stage(candidate, social, ("social_creative_profile",))
        whiteboard.update(social)

        health = self._call_agent(
            stage_name="mental_health",
            prompt=mental_health_agent_prompt(
                whiteboard_json=_json_dumps(whiteboard),
                slot_context_json=slot_json,
            ),
            system_prompt=self._system_prompt(MENTAL_HEALTH_AGENT_SYSTEM_PROMPT),
            raw_outputs=raw_outputs,
        )
        self._merge_stage(candidate, health, ("mental_health_context",))

        return self._validate_or_revise(slot, candidate, raw_outputs)

    def _validate_or_revise(
        self,
        slot: MegaPersonaSlot,
        candidate: dict[str, Any],
        raw_outputs: dict[str, str],
    ) -> MegaPersonaGenerationResult:
        report = validate_mega_persona(candidate)
        if report.is_valid:
            return MegaPersonaGenerationResult(
                slot=slot,
                persona=MegaPersona.model_validate(candidate),
                validation_report=report,
                raw_outputs=raw_outputs,
                candidate_json=candidate,
            )

        revised = candidate
        for attempt in range(self.max_revisions):
            issues = [
                {
                    "rule_id": issue.rule_id,
                    "severity": issue.severity,
                    "message": issue.message,
                }
                for issue in report.issues
            ]
            raw = self.llm.generate(
                revision_prompt(
                    candidate_json=_json_dumps(revised),
                    issues_json=_json_dumps(issues),
                ),
                system_prompt="You repair structured JSON for a schema-constrained persona pipeline.",
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
            raw_outputs[f"revision_{attempt + 1}"] = raw
            revised = parse_json_object(raw)
            report = validate_mega_persona(revised)
            if report.is_valid:
                return MegaPersonaGenerationResult(
                    slot=slot,
                    persona=MegaPersona.model_validate(revised),
                    validation_report=report,
                    raw_outputs=raw_outputs,
                    candidate_json=revised,
                )

        return MegaPersonaGenerationResult(
            slot=slot,
            persona=None,
            validation_report=report,
            raw_outputs=raw_outputs,
            candidate_json=revised,
        )

    def _call_agent(
        self,
        stage_name: str,
        prompt: str,
        system_prompt: str,
        raw_outputs: dict[str, str],
    ) -> dict[str, Any]:
        raw = self.llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw_outputs[stage_name] = raw
        return parse_json_object(raw)

    def _system_prompt(self, base_prompt: str) -> str:
        if not self.prompt_addendum:
            return base_prompt
        return (
            f"{base_prompt.rstrip()}\n\n"
            "Evolved generation policy addendum:\n"
            f"{self.prompt_addendum}"
        )

    @staticmethod
    def _merge_stage(
        candidate: dict[str, Any],
        stage_payload: dict[str, Any],
        required_keys: tuple[str, ...],
    ) -> None:
        missing = [key for key in required_keys if key not in stage_payload]
        if missing:
            raise AgentOutputError(f"agent output missing keys: {missing}")
        for key in required_keys:
            candidate[key] = stage_payload[key]


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from plain text or a fenced model response."""
    text = raw.strip()
    if text.startswith("```"):
        text = _strip_fence(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(_extract_first_json_object(text))
    if not isinstance(parsed, dict):
        raise AgentOutputError("agent response must be a JSON object")
    return parsed


def _strip_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise AgentOutputError("agent response does not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise AgentOutputError("agent response contains incomplete JSON")


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
