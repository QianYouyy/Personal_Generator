"""MegaPersona LLM generator — decomposed and integrated pipeline modes.

Key improvements over the original sequential 5-agent design:
1. **Parallel agents 3-5**: Values, Social, and Mental Health run concurrently
   after Cognition finishes (they share no mutual dependencies).
2. **Slim whiteboards**: each agent receives only the fields it needs, cutting
   ~30 % of input tokens per later-stage call.
3. **Hard constraint injection**: concrete numeric targets and anti-collapse
   rules are injected directly into agent prompts, raising first-pass validity.
4. **Single-call architecture**: experiments can use an integrated generator
   when coherence/cost should be compared against staged decomposition.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Callable

from src.mega_persona.prompts import (
    COGNITION_MOTIVATION_AGENT_SYSTEM_PROMPT,
    COMPACT_PERSONA_SYSTEM_PROMPT,
    DEMOGRAPHICS_AGENT_SYSTEM_PROMPT,
    MENTAL_HEALTH_AGENT_SYSTEM_PROMPT,
    SOCIAL_CREATIVE_AGENT_SYSTEM_PROMPT,
    VALUES_IDENTITY_AGENT_SYSTEM_PROMPT,
    compact_persona_prompt,
    cognition_motivation_agent_prompt,
    demographics_agent_prompt,
    mental_health_agent_prompt,
    revision_prompt,
    social_creative_agent_prompt,
    stage_repair_prompt,
    values_identity_agent_prompt,
)
from src.mega_persona.schema import (
    ASPIRATION_MAX_LENGTH,
    DERIVED_REASONING_MAX_LENGTH,
    FAMILY_CONTEXT_MAX_LENGTH,
    IDENTITY_ANCHOR_MAX_LENGTH,
    MENTAL_HEALTH_NARRATIVE_MAX_LENGTH,
    MORAL_TENSION_MAX_LENGTH,
    SOCIAL_NARRATIVE_MAX_LENGTH,
    MegaPersona,
)
from src.mega_persona.slots import (
    AXIS_NAMES,
    MegaPersonaSlot,
    SlotSampler,
    axis_roles_for_target_axes,
)
from src.mega_persona.validator import ValidationIssue, ValidationReport, validate_mega_persona


logger = logging.getLogger(__name__)


_TEXT_LENGTH_BUDGETS = {
    ("demographics", "family_context"): FAMILY_CONTEXT_MAX_LENGTH,
    ("values_identity", "identity_anchor"): IDENTITY_ANCHOR_MAX_LENGTH,
    ("values_identity", "moral_tension"): MORAL_TENSION_MAX_LENGTH,
    ("values_identity", "aspiration"): ASPIRATION_MAX_LENGTH,
    ("social_creative_profile", "narrative"): SOCIAL_NARRATIVE_MAX_LENGTH,
    ("mental_health_context", "narrative"): MENTAL_HEALTH_NARRATIVE_MAX_LENGTH,
    ("derived_academic_tendency", "reasoning"): DERIVED_REASONING_MAX_LENGTH,
}


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
    """LLM pipeline for structured MegaPersona generation.

    `five_agent` runs a staged generator where agents 3-5 (Values, Social,
    Mental Health) run in parallel after cognition. `single_call` asks one
    integrated writer to produce the complete persona in one pass. Both modes
    share schema validation, revision, blueprint injection, and metrics.
    """

    def __init__(
        self,
        llm_client,
        temperature: float = 0.45,
        max_tokens: int = 3000,
        max_revisions: int = 1,
        prompt_addendum: str = "",
        blueprint_builder: Callable[[MegaPersonaSlot], dict[str, Any]] | None = None,
        pipeline_mode: str = "five_agent",
    ):
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_revisions = max_revisions
        self.prompt_addendum = prompt_addendum.strip()
        self.blueprint_builder = blueprint_builder
        if pipeline_mode == "compact":
            pipeline_mode = "single_call"
        if pipeline_mode not in {"five_agent", "single_call"}:
            raise ValueError("pipeline_mode must be 'five_agent' or 'single_call'")
        self.pipeline_mode = pipeline_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n: int,
        seed: int | None = None,
    ) -> list[MegaPersonaGenerationResult]:
        slots = SlotSampler().sample(n=n, seed=seed)
        return self.generate_from_slots(slots)

    def generate_from_slots(
        self,
        slots: list[MegaPersonaSlot],
        max_workers: int = 1,
    ) -> list[MegaPersonaGenerationResult]:
        max_workers = max(1, int(max_workers))
        if max_workers > 1 and len(slots) > 1:
            return self._generate_from_slots_parallel(slots, max_workers=max_workers)

        results: list[MegaPersonaGenerationResult] = []
        total = len(slots)
        for index, slot in enumerate(slots, start=1):
            results.append(self._generate_one_with_logging(slot, index, total))
        return results

    def _generate_from_slots_parallel(
        self,
        slots: list[MegaPersonaSlot],
        max_workers: int,
    ) -> list[MegaPersonaGenerationResult]:
        total = len(slots)
        logger.info(
            "MegaPersona generation parallel start slots=%s max_workers=%s",
            total,
            max_workers,
        )
        ordered: list[MegaPersonaGenerationResult | None] = [None] * total
        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
            futures = {
                executor.submit(self._generate_one_with_logging, slot, index, total): index
                for index, slot in enumerate(slots, start=1)
            }
            for future in as_completed(futures):
                index = futures[future]
                ordered[index - 1] = future.result()
                logger.info("MegaPersona generation parallel done %s/%s", index, total)
        return [result for result in ordered if result is not None]

    def _generate_one_with_logging(
        self,
        slot: MegaPersonaSlot,
        index: int,
        total: int,
    ) -> MegaPersonaGenerationResult:
        logger.info(
            "MegaPersona generation %s/%s slot=%s quota=%s",
            index,
            total,
            slot.slot_id,
            slot.quota_label,
        )
        try:
            result = self.generate_one(slot)
        except Exception as exc:
            logger.error(
                "MegaPersona generation %s/%s slot=%s failed (%s: %s)",
                index,
                total,
                slot.slot_id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            result = MegaPersonaGenerationResult(
                slot=slot,
                persona=None,
                validation_report=ValidationReport(
                    is_valid=False,
                    schema_valid=False,
                    issues=[
                        ValidationIssue(
                            rule_id="PIPELINE",
                            severity="error",
                            message=f"{type(exc).__name__}: {exc}",
                        )
                    ],
                ),
                raw_outputs={"pipeline_error": f"{type(exc).__name__}: {exc}"},
                candidate_json={"persona_id": slot.slot_id},
            )
        logger.info(
            "MegaPersona generation %s/%s slot=%s valid=%s issues=%s",
            index,
            total,
            slot.slot_id,
            result.is_valid,
            len(result.validation_report.issues),
        )
        return result

    # ------------------------------------------------------------------
    # Core pipeline (optimised)
    # ------------------------------------------------------------------

    def generate_one(self, slot: MegaPersonaSlot) -> MegaPersonaGenerationResult:
        if self.pipeline_mode == "single_call":
            return self._generate_one_compact(slot)

        blueprint = self._build_blueprint(slot)
        whiteboard: dict[str, Any] = {
            "persona_id": slot.slot_id,
            "slot": slot.prompt_context(),
        }
        if blueprint:
            whiteboard["generation_blueprint"] = blueprint
        candidate: dict[str, Any] = {"persona_id": slot.slot_id}
        raw_outputs: dict[str, str] = {}
        if blueprint:
            raw_outputs["generation_blueprint"] = _json_dumps(blueprint)
        slot_json = _json_dumps(slot.prompt_context())

        # ---- Agent 1: Demographics (must run alone — no prior context) ----
        logger.info("Slot=%s agent=demographics start", slot.slot_id)
        demographics, raw_demo = self._call_agent(
            stage_name="demographics",
            prompt=demographics_agent_prompt(slot_json),
            system_prompt=self._system_prompt(DEMOGRAPHICS_AGENT_SYSTEM_PROMPT),
        )
        logger.info("Slot=%s agent=demographics done", slot.slot_id)
        raw_outputs["demographics"] = raw_demo
        demographics = self._ensure_stage_keys(
            slot=slot,
            stage_name="demographics",
            stage_payload=demographics,
            required_keys=("demographics",),
            raw_text=raw_demo,
            context=whiteboard,
            raw_outputs=raw_outputs,
        )
        self._merge_stage(candidate, demographics, ("demographics",))
        whiteboard.update(demographics)

        # ---- Agent 2: Cognition & Motivation (needs demographics) ----
        hard_cc = _hard_constraints_for_cognition(slot)
        logger.info("Slot=%s agent=cognition_motivation start", slot.slot_id)
        cognition, raw_cog = self._call_agent(
            stage_name="cognition_motivation",
            prompt=cognition_motivation_agent_prompt(
                whiteboard_json=_json_dumps(whiteboard),
                target_axes_json=_json_dumps(slot.target_axes),
                prior_constraints="\n".join(
                    slot.adaptive_constraints + _blueprint_constraint_lines(blueprint, "cognition")
                ),
                hard_constraints=_join_constraint_blocks(
                    hard_cc,
                    _blueprint_hard_constraints(blueprint, "cognition"),
                ),
            ),
            system_prompt=self._system_prompt(COGNITION_MOTIVATION_AGENT_SYSTEM_PROMPT),
        )
        logger.info("Slot=%s agent=cognition_motivation done", slot.slot_id)
        raw_outputs["cognition_motivation"] = raw_cog
        cognition = self._ensure_stage_keys(
            slot=slot,
            stage_name="cognition_motivation",
            stage_payload=cognition,
            required_keys=("cognitive_motivation_profile", "derived_academic_tendency"),
            raw_text=raw_cog,
            context=whiteboard,
            raw_outputs=raw_outputs,
        )
        self._merge_stage(
            candidate,
            cognition,
            ("cognitive_motivation_profile", "derived_academic_tendency"),
        )
        whiteboard.update(cognition)

        # ---- Agents 3-5: Values, Social, Mental Health (parallel) ----
        # All three depend only on demographics + cognition; they are
        # mutually independent so we run them concurrently.
        results_3_5 = _run_parallel_agents(
            llm=self.llm,
            system_prompt_fn=self._system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            whiteboard=whiteboard,
            slot=slot,
            slot_json=slot_json,
            blueprint=blueprint,
        )

        for stage_name, parsed, raw_text in results_3_5:
            raw_outputs[stage_name] = raw_text
            key_map = {
                "values_identity": ("values_identity",),
                "social_creative": ("social_creative_profile",),
                "mental_health": ("mental_health_context",),
            }
            parsed = self._ensure_stage_keys(
                slot=slot,
                stage_name=stage_name,
                stage_payload=parsed,
                required_keys=key_map[stage_name],
                raw_text=raw_text,
                context=whiteboard,
                raw_outputs=raw_outputs,
            )
            self._merge_stage(candidate, parsed, key_map[stage_name])

        return self._validate_or_revise(slot, candidate, raw_outputs, blueprint=blueprint)

    def _generate_one_compact(self, slot: MegaPersonaSlot) -> MegaPersonaGenerationResult:
        blueprint = self._build_blueprint(slot)
        raw_outputs: dict[str, str] = {}
        if blueprint:
            raw_outputs["generation_blueprint"] = _json_dumps(blueprint)
        logger.info("Slot=%s compact_persona start", slot.slot_id)
        raw = _generate_with_retry(
            llm=self.llm,
            label="compact_persona",
            prompt=compact_persona_prompt(
                slot_context_json=_json_dumps(slot.prompt_context()),
                blueprint_json=_json_dumps(blueprint),
                prompt_addendum=self.prompt_addendum,
            ),
            system_prompt=self._system_prompt(COMPACT_PERSONA_SYSTEM_PROMPT),
            temperature=self.temperature,
            max_tokens=max(self.max_tokens, 4200),
        )
        logger.info("Slot=%s compact_persona done", slot.slot_id)
        raw_outputs["compact_persona"] = raw
        candidate = _parse_or_repair_json(
            llm=self.llm,
            raw=raw,
            stage_name="compact_persona",
            temperature=0.0,
            max_tokens=max(self.max_tokens, 4200),
        )
        candidate.setdefault("persona_id", slot.slot_id)
        return self._validate_or_revise(slot, candidate, raw_outputs, blueprint=blueprint)

    # ------------------------------------------------------------------
    # Validation & revision
    # ------------------------------------------------------------------

    def _validate_or_revise(
        self,
        slot: MegaPersonaSlot,
        candidate: dict[str, Any],
        raw_outputs: dict[str, str],
        blueprint: dict[str, Any] | None = None,
    ) -> MegaPersonaGenerationResult:
        trimmed_fields = _apply_text_length_budgets(candidate)
        _log_trimmed_fields(slot.slot_id, trimmed_fields, stage="pre-validate")
        report = validate_mega_persona(candidate)
        blueprint_issues = _blueprint_critic_issues(candidate, blueprint or {})
        if report.is_valid and not blueprint_issues:
            logger.info("Slot=%s validation passed", slot.slot_id)
            return MegaPersonaGenerationResult(
                slot=slot,
                persona=MegaPersona.model_validate(candidate),
                validation_report=report,
                raw_outputs=raw_outputs,
                candidate_json=candidate,
            )
        if report.is_valid and blueprint_issues:
            report = ValidationReport(
                is_valid=False,
                schema_valid=True,
                issues=blueprint_issues,
            )

        revised = candidate
        for attempt in range(self.max_revisions):
            logger.info(
                "Slot=%s validation failed; revision attempt=%s issues=%s",
                slot.slot_id,
                attempt + 1,
                len(report.issues),
            )
            _log_validation_issues(slot.slot_id, report)
            issues = [
                {
                    "rule_id": issue.rule_id,
                    "severity": issue.severity,
                    "message": issue.message,
                }
                for issue in report.issues
            ]
            raw = _generate_with_retry(
                llm=self.llm,
                label=f"revision_{attempt + 1}",
                prompt=revision_prompt(
                    candidate_json=_json_dumps(revised),
                    issues_json=_json_dumps(issues),
                ),
                system_prompt="You repair structured JSON for a schema-constrained persona pipeline.",
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
            raw_outputs[f"revision_{attempt + 1}"] = raw
            revised = _parse_or_repair_json(
                llm=self.llm,
                raw=raw,
                stage_name=f"revision_{attempt + 1}",
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            trimmed_fields = _apply_text_length_budgets(revised)
            _log_trimmed_fields(slot.slot_id, trimmed_fields, stage=f"post-revision-{attempt + 1}")
            report = validate_mega_persona(revised)
            if report.is_valid:
                logger.info("Slot=%s revision attempt=%s passed", slot.slot_id, attempt + 1)
                return MegaPersonaGenerationResult(
                    slot=slot,
                    persona=MegaPersona.model_validate(revised),
                    validation_report=report,
                    raw_outputs=raw_outputs,
                    candidate_json=revised,
                )

        logger.info("Slot=%s validation failed after revisions issues=%s", slot.slot_id, len(report.issues))
        _log_validation_issues(slot.slot_id, report)
        return MegaPersonaGenerationResult(
            slot=slot,
            persona=None,
            validation_report=report,
            raw_outputs=raw_outputs,
            candidate_json=revised,
        )

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_agent(
        self,
        stage_name: str,
        prompt: str,
        system_prompt: str,
    ) -> tuple[dict[str, Any], str]:
        """Call the LLM and return (parsed_json, raw_text)."""
        raw = _generate_with_retry(
            llm=self.llm,
            label=stage_name,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        parsed = _parse_or_repair_json(
            llm=self.llm,
            raw=raw,
            stage_name=stage_name,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return parsed, raw

    def _system_prompt(self, base_prompt: str) -> str:
        if not self.prompt_addendum:
            return base_prompt
        return (
            f"{base_prompt.rstrip()}\n\n"
            "Evolved generation policy addendum:\n"
            f"{self.prompt_addendum}"
        )

    def _build_blueprint(self, slot: MegaPersonaSlot) -> dict[str, Any]:
        if self.blueprint_builder is None:
            return {}
        try:
            blueprint = self.blueprint_builder(slot)
        except Exception as exc:
            logger.warning("Slot=%s blueprint builder failed: %s", slot.slot_id, exc)
            return {}
        return blueprint if isinstance(blueprint, dict) else {}

    def _ensure_stage_keys(
        self,
        *,
        slot: MegaPersonaSlot,
        stage_name: str,
        stage_payload: dict[str, Any],
        required_keys: tuple[str, ...],
        raw_text: str,
        context: dict[str, Any],
        raw_outputs: dict[str, str],
    ) -> dict[str, Any]:
        missing = [key for key in required_keys if key not in stage_payload]
        if not missing:
            return stage_payload
        logger.warning(
            "Slot=%s agent=%s missing keys=%s; requesting stage repair",
            slot.slot_id,
            stage_name,
            missing,
        )
        repair_raw = _generate_with_retry(
            llm=self.llm,
            label=f"{stage_name}_stage_repair",
            prompt=stage_repair_prompt(
                stage_name=stage_name,
                raw_stage_json=raw_text,
                required_keys_json=_json_dumps(list(required_keys)),
                context_json=_json_dumps(
                    {
                        "persona_id": slot.slot_id,
                        "slot": slot.prompt_context(),
                        "whiteboard": context,
                    }
                ),
            ),
            system_prompt=(
                "You repair one stage of a schema-constrained persona pipeline. "
                "Return valid JSON only with the required top-level keys."
            ),
            temperature=0.1,
            max_tokens=self.max_tokens,
        )
        raw_outputs[f"{stage_name}_stage_repair"] = repair_raw
        repaired = _parse_or_repair_json(
            llm=self.llm,
            raw=repair_raw,
            stage_name=f"{stage_name}_stage_repair",
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return repaired

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


# ======================================================================
# Parallel dispatch for agents 3-5
# ======================================================================

def _run_parallel_agents(
    *,
    llm,
    system_prompt_fn,
    temperature: float,
    max_tokens: int,
    whiteboard: dict[str, Any],
    slot: MegaPersonaSlot,
    slot_json: str,
    blueprint: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any], str]]:
    """Run Values, Social, and Mental Health agents concurrently.

    Each agent receives a **slim whiteboard** containing only the fields
    it actually needs, plus **hard constraints** derived from the slot.
    """

    def _call(agent_cfg: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        """Single-agent wrapper used inside the thread pool."""
        logger.info("Slot=%s agent=%s start", slot.slot_id, agent_cfg["stage_name"])
        raw = _generate_with_retry(
            llm=llm,
            label=agent_cfg["stage_name"],
            prompt=agent_cfg["prompt"],
            system_prompt=system_prompt_fn(agent_cfg["system_prompt"]),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("Slot=%s agent=%s done", slot.slot_id, agent_cfg["stage_name"])
        parsed = _parse_or_repair_json(
            llm=llm,
            raw=raw,
            stage_name=agent_cfg["stage_name"],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return agent_cfg["stage_name"], parsed, raw

    # Build per-agent configs with slim whiteboards
    configs: list[dict[str, Any]] = []

    # -- Values & Identity --
    wb_values = _slim_wb_for_values(whiteboard, slot)
    hc_values = _hard_constraints_for_values(whiteboard, slot)
    configs.append({
        "stage_name": "values_identity",
        "system_prompt": VALUES_IDENTITY_AGENT_SYSTEM_PROMPT,
        "prompt": values_identity_agent_prompt(
            whiteboard_json=_json_dumps(wb_values),
            slot_context_json=slot_json,
            hard_constraints=_join_constraint_blocks(
                hc_values,
                _blueprint_hard_constraints(blueprint or {}, "values_identity"),
            ),
        ),
    })

    # -- Social & Creative --
    wb_social = _slim_wb_for_social(whiteboard, slot)
    hc_social = _hard_constraints_for_social(whiteboard, slot)
    configs.append({
        "stage_name": "social_creative",
        "system_prompt": SOCIAL_CREATIVE_AGENT_SYSTEM_PROMPT,
        "prompt": social_creative_agent_prompt(
            whiteboard_json=_json_dumps(wb_social),
            slot_context_json=slot_json,
            hard_constraints=_join_constraint_blocks(
                hc_social,
                _blueprint_hard_constraints(blueprint or {}, "social_creative"),
            ),
        ),
    })

    # -- Mental Health --
    wb_health = _slim_wb_for_health(whiteboard, slot)
    hc_health = _hard_constraints_for_health(whiteboard, slot)
    configs.append({
        "stage_name": "mental_health",
        "system_prompt": MENTAL_HEALTH_AGENT_SYSTEM_PROMPT,
        "prompt": mental_health_agent_prompt(
            whiteboard_json=_json_dumps(wb_health),
            slot_context_json=slot_json,
            hard_constraints=_join_constraint_blocks(
                hc_health,
                _blueprint_hard_constraints(blueprint or {}, "mental_health"),
            ),
        ),
    })

    with ThreadPoolExecutor(max_workers=3) as executor:
        logger.info("Slot=%s agents=values/social/mental_health parallel start", slot.slot_id)
        futures = [executor.submit(_call, cfg) for cfg in configs]
        results = [future.result() for future in as_completed(futures)]
        logger.info("Slot=%s agents=values/social/mental_health parallel done", slot.slot_id)
        return results


# ======================================================================
# Whiteboard slimming — each agent only sees what it needs
# ======================================================================

def _slim_wb_for_values(wb: dict[str, Any], slot: MegaPersonaSlot) -> dict[str, Any]:
    """Values agent needs: persona context + cognition basics."""
    cog = _as_dict(wb.get("cognitive_motivation_profile", {}))
    demo = _as_dict(wb.get("demographics", {}))
    return {
        "persona_id": wb["persona_id"],
        "demographics": {
            "grade_or_stage": demo.get("grade_or_stage"),
            "family_context": _truncate(demo.get("family_context", ""), 300),
        },
        "cognitive_motivation_profile": {
            "thinking_style": _pick_keys(cog.get("thinking_style", {}), ("dominant_mode", "abstraction_level")),
            "motivation_system": _pick_keys(cog.get("motivation_system", {}), (
                "primary_drive", "intrinsic_motivation", "external_pressure_sensitivity",
                "reward_preference",
            )),
            "decision_pattern": _pick_keys(cog.get("decision_pattern", {}), ("risk_appetite", "time_horizon")),
        },
        "target_axes": slot.target_axes,
    }


def _slim_wb_for_social(wb: dict[str, Any], slot: MegaPersonaSlot) -> dict[str, Any]:
    """Social agent needs: demographics + cognition motivation/thinking."""
    cog = _as_dict(wb.get("cognitive_motivation_profile", {}))
    demo = _as_dict(wb.get("demographics", {}))
    return {
        "persona_id": wb["persona_id"],
        "demographics": {
            "age": demo.get("age"),
            "grade_or_stage": demo.get("grade_or_stage"),
            "region_type": demo.get("region_type"),
        },
        "cognitive_motivation_profile": {
            "thinking_style": _pick_keys(cog.get("thinking_style", {}), ("abstraction_level", "dominant_mode")),
            "motivation_system": _pick_keys(cog.get("motivation_system", {}), (
                "primary_drive", "intrinsic_motivation",
            )),
            "self_regulation": _pick_keys(cog.get("self_regulation", {}), ("emotional_regulation",)),
            "decision_pattern": _pick_keys(cog.get("decision_pattern", {}), ("risk_appetite",)),
        },
        "target_axes": slot.target_axes,
        "social_energy_band": slot.constraints.get("social_energy_band", "mid"),
    }


def _slim_wb_for_health(wb: dict[str, Any], slot: MegaPersonaSlot) -> dict[str, Any]:
    """Mental Health agent needs: self-regulation details + stress context."""
    cog = _as_dict(wb.get("cognitive_motivation_profile", {}))
    demo = _as_dict(wb.get("demographics", {}))
    return {
        "persona_id": wb["persona_id"],
        "demographics": {
            "family_context": _truncate(demo.get("family_context", ""), 250),
        },
        "cognitive_motivation_profile": {
            "motivation_system": _pick_keys(cog.get("motivation_system", {}), (
                "primary_drive", "failure_sensitivity", "external_pressure_sensitivity",
            )),
            "self_regulation": cog.get("self_regulation", {}),  # FULL — this is key input
            "challenge_response": cog.get("challenge_response", {}),
            "thinking_style": _pick_keys(cog.get("thinking_style", {}), ("ambiguity_tolerance",)),
        },
        "target_axes": slot.target_axes,
        "stress_band": slot.constraints.get("stress_band", "mid"),
    }


# ======================================================================
# Hard constraint injection — numeric targets embedded in prompts
# ======================================================================

# All 15 numeric trait paths used by the validator (R5/R6 all-high/all-low check)
_NUMERIC_TRAIT_LABELS = [
    "abstraction_level", "ambiguity_tolerance",
    "intrinsic_motivation", "external_pressure_sensitivity", "failure_sensitivity",
    "persistence", "emotional_regulation", "metacognition", "habit_stability",
    "risk_appetite", "social_energy", "expressiveness",
    "peer_influence_sensitivity", "stress_load", "resilience",
]


def _hard_constraints_for_cognition(slot: MegaPersonaSlot) -> str:
    """Numeric targets and anti-collapse rules for the cognition agent."""
    axes = slot.target_axes
    roles = axis_roles_for_target_axes(axes)
    lines = [
        "HARD TARGETS (embed these in mechanisms, not as repeated numbers):",
        f"  - {roles['cognitive_core']} ≈ {axes.get(roles['cognitive_core'], 0.5):.2f}",
        f"  - {roles['motivation_core']} ≈ {axes.get(roles['motivation_core'], 0.5):.2f}",
        f"  - {roles['regulation_core']} ≈ {axes.get(roles['regulation_core'], 0.5):.2f}",
        "",
        "ANTI-COLLAPSE RULES:",
        "  - At least 5 of these numeric traits MUST be in [0.25, 0.75]:",
    ]
    for label in _NUMERIC_TRAIT_LABELS:
        lines.append(f"      {label}")
    lines.append(
        "  - If a trait is extreme (≤0.2 or ≥0.8), explain WHY in the narrative."
    )
    lines.append(
        "  - Do NOT set all traits to mid-range values — keep the profile differentiated."
    )
    lines.append(
        f"  - The primary drive is '{slot.constraints.get('primary_drive', 'mastery')}'."
    )
    if slot.constraints.get("derived_performance_band") == "high":
        lines.append(
            "  - High likely performance → ground in habits/strategy/resources/support, not raw talent."
        )
    return "\n".join(lines)


def _hard_constraints_for_values(wb: dict[str, Any], slot: MegaPersonaSlot) -> str:
    """Alignment hints for values based on motivation profile."""
    cog = _as_dict(wb.get("cognitive_motivation_profile", {}))
    mot = _as_dict(cog.get("motivation_system", {}))
    primary = mot.get("primary_drive", "mastery")
    intrinsic = mot.get("intrinsic_motivation", 0.5)
    roles = axis_roles_for_target_axes(slot.target_axes)
    autonomy_key = roles["motivation_core"]
    autonomy = slot.target_axes.get(autonomy_key, 0.5)
    lines = [
        "HARD ALIGNMENT:",
        f"  - primary_drive={primary}, {autonomy_key}≈{autonomy:.2f}",
        "  - Core values must reflect the person's actual motivational structure.",
    ]
    if autonomy >= 0.65:
        lines.append("  - High autonomy → include at least one self-directed value.")
    if primary in ("belonging", "recognition"):
        lines.append("  - Socially-oriented primary drive → values should reflect relationships.")
    if primary in ("security", "avoidance"):
        lines.append("  - Security/avoidance drive → include stability or safety as a value.")
    lines.append("  - Include one genuine moral tension, not a decorative contradiction.")
    lines.append("  - The identity_anchor should connect to HOW they think, not just who they are.")
    return "\n".join(lines)


def _hard_constraints_for_social(wb: dict[str, Any], slot: MegaPersonaSlot) -> str:
    """Social energy targets from the slot constraint band."""
    band = slot.constraints.get("social_energy_band", "mid")
    band_map = {"low": (0.15, 0.30), "low_mid": (0.28, 0.45),
                 "mid": (0.40, 0.60), "mid_high": (0.55, 0.75),
                 "high": (0.70, 0.90)}
    lo, hi = band_map.get(band, (0.35, 0.65))
    lines = [
        f"HARD TARGET: social_energy should be in [{lo:.2f}, {hi:.2f}] (band: {band}).",
        "  - expressiveness should be consistent with social_energy.",
        "  - peer_influence_sensitivity should reflect their autonomy level.",
        "  - Make creativity mode concrete (not 'creative' as a generic label).",
        "  - collaboration_style must match both social_energy AND motivation type.",
    ]
    return "\n".join(lines)


def _hard_constraints_for_health(wb: dict[str, Any], slot: MegaPersonaSlot) -> str:
    """Stress-resilience consistency constraints."""
    band = slot.constraints.get("stress_band", "mid")
    roles = axis_roles_for_target_axes(slot.target_axes)
    regulation_key = roles["regulation_core"]
    regulation = slot.target_axes.get(regulation_key, 0.5)
    lines = [
        f"HARD TARGET: stress_load should reflect stress_band='{band}', "
        f"{regulation_key}≈{regulation:.2f}.",
    ]
    if band in ("high", "mid_high") and regulation >= 0.6:
        lines.append(
            "  - High stress + high resilience → MUST name ≥2 concrete protective factors."
        )
    if band in ("high", "mid_high") and regulation <= 0.35:
        lines.append(
            "  - High stress + low resilience → coping should show strain, not magic recovery."
        )
    if band in ("low", "low_mid") and regulation >= 0.7:
        lines.append(
            "  - Low stress + high resilience → explain why resilience is high (e.g. strong routines, support)."
        )
    lines.append("  - Do not diagnose. Describe behaviorally useful context.")
    lines.append("  - coping_style must make sense given their self-regulation pattern.")
    return "\n".join(lines)


# ======================================================================
# Genome v3 blueprint injection and lightweight critic
# ======================================================================

def _blueprint_constraint_lines(blueprint: dict[str, Any], stage_name: str) -> list[str]:
    if not blueprint:
        return []
    lines = []
    core_tension = blueprint.get("core_tension")
    if isinstance(core_tension, str) and core_tension.strip():
        lines.append(f"Blueprint core tension: {core_tension.strip()}")
    for binding in _blueprint_stage_bindings(blueprint, stage_name):
        lines.append(f"Blueprint binding: {binding}")
    return lines


def _blueprint_hard_constraints(blueprint: dict[str, Any], stage_name: str) -> str:
    if not blueprint:
        return ""
    lines = ["GENOME V3 BLUEPRINT CONSTRAINTS:"]
    axis_plan = blueprint.get("axis_expression_plan")
    if isinstance(axis_plan, dict):
        for axis, plan in axis_plan.items():
            if not isinstance(plan, dict):
                continue
            lines.append(
                "  - "
                f"{axis}: target={plan.get('target', 'n/a')} band={plan.get('band', 'mid')}; "
                f"evidence={plan.get('evidence', '')}; boundary={plan.get('boundary', '')}"
            )
    behavior_profile = blueprint.get("behavior_prediction_profile")
    if isinstance(behavior_profile, dict):
        stage_keys = {
            "cognition": ("ambiguous_task", "failure_feedback", "deadline"),
            "values_identity": ("peer_pressure", "failure_feedback"),
            "social_creative": ("peer_pressure", "ambiguous_task"),
            "mental_health": ("failure_feedback", "deadline"),
        }.get(stage_name, ())
        for key in stage_keys:
            value = behavior_profile.get(key)
            if isinstance(value, str) and value.strip():
                lines.append(f"  - Behavior prediction anchor `{key}`: {value.strip()}")
    for binding in _blueprint_stage_bindings(blueprint, stage_name):
        lines.append(f"  - Cross-agent binding: {binding}")
    lines.append(
        "  - The final text must echo these blueprint constraints through concrete behavior, "
        "not by naming the blueprint."
    )
    return "\n".join(lines)


def _blueprint_stage_bindings(blueprint: dict[str, Any], stage_name: str) -> list[str]:
    bindings = blueprint.get("cross_agent_binding")
    if not isinstance(bindings, list):
        return []
    stage_aliases = {
        "cognition": ("cognition", "cognitive", "motivation", "regulation"),
        "values_identity": ("values", "identity"),
        "social_creative": ("social", "creative", "peer"),
        "mental_health": ("health", "stress", "recovery"),
    }.get(stage_name, (stage_name,))
    selected = []
    for item in bindings:
        if not isinstance(item, str):
            continue
        lowered = item.lower()
        if any(alias in lowered for alias in stage_aliases):
            selected.append(item)
    return selected[:4]


def _blueprint_critic_issues(
    candidate: dict[str, Any],
    blueprint: dict[str, Any],
) -> list[ValidationIssue]:
    if not blueprint:
        return []
    text = _candidate_text_blob(candidate)
    if not text:
        return []
    issues: list[ValidationIssue] = []
    scenario_keywords = {
        "ambiguous_task": ("ambiguous", "uncertain", "unclear", "instruction", "task"),
        "peer_pressure": ("peer", "group", "classmate", "comparison", "social"),
        "failure_feedback": ("failure", "feedback", "mistake", "criticism", "setback"),
        "deadline": ("deadline", "schedule", "late", "time", "rush"),
    }
    missing_scenarios = [
        scenario
        for scenario, keywords in scenario_keywords.items()
        if not any(keyword in text for keyword in keywords)
    ]
    if len(missing_scenarios) >= 3:
        issues.append(
            ValidationIssue(
                rule_id="BLUEPRINT",
                severity="error",
                message=(
                    "Genome v3 blueprint behavior anchors are weakly reflected; missing scenarios: "
                    + ", ".join(missing_scenarios)
                ),
            )
        )
    return issues


def _candidate_text_blob(candidate: dict[str, Any]) -> str:
    chunks: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                _walk(nested)
        elif isinstance(value, list):
            for nested in value:
                _walk(nested)

    _walk(candidate)
    return " ".join(chunks).lower()


def _join_constraint_blocks(*blocks: str) -> str:
    return "\n\n".join(block.strip() for block in blocks if isinstance(block, str) and block.strip())


# ======================================================================
# Helpers
# ======================================================================

def _pick_keys(d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    return {k: d[k] for k in keys if k in d}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _truncate(text: Any, max_len: int) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _apply_text_length_budgets(candidate: dict[str, Any]) -> list[tuple[str, int, int]]:
    trimmed: list[tuple[str, int, int]] = []
    for path, max_len in _TEXT_LENGTH_BUDGETS.items():
        parent = candidate
        for key in path[:-1]:
            if not isinstance(parent, dict):
                parent = None
                break
            parent = parent.get(key)
        if not isinstance(parent, dict):
            continue
        leaf = path[-1]
        value = parent.get(leaf)
        if not isinstance(value, str):
            continue
        if len(value) <= max_len:
            continue
        parent[leaf] = _smart_trim_text(value, max_len)
        trimmed.append((".".join(path), len(value), len(parent[leaf])))
    return trimmed


def _smart_trim_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    hard_limit = max(0, max_len - 1)
    snippet = text[:hard_limit]
    boundary = max(
        snippet.rfind(". "),
        snippet.rfind("! "),
        snippet.rfind("? "),
        snippet.rfind("。"),
        snippet.rfind("！"),
        snippet.rfind("？"),
        snippet.rfind("; "),
        snippet.rfind("；"),
    )
    if boundary >= int(max_len * 0.6):
        snippet = snippet[: boundary + 1].rstrip()
    else:
        snippet = snippet.rstrip(" ,;:，；：")
    return snippet + "…"


def _log_trimmed_fields(slot_id: str, trimmed_fields: list[tuple[str, int, int]], stage: str) -> None:
    for field_path, old_len, new_len in trimmed_fields:
        logger.info(
            "Slot=%s text budget trim stage=%s field=%s old_len=%s new_len=%s",
            slot_id,
            stage,
            field_path,
            old_len,
            new_len,
        )


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


def _parse_or_repair_json(
    *,
    llm,
    raw: str,
    stage_name: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    try:
        return parse_json_object(raw)
    except Exception as first_error:
        logger.warning(
            "Malformed JSON from stage=%s; requesting one syntax repair (%s: %s)",
            stage_name,
            type(first_error).__name__,
            first_error,
        )
        repaired = _generate_with_retry(
            llm=llm,
            label=f"{stage_name}_json_repair",
            prompt=_json_repair_prompt(stage_name, raw),
            system_prompt=(
                "You repair malformed JSON from an LLM pipeline. Return ONLY one "
                "valid JSON object. Do not add markdown, commentary, or new fields."
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return parse_json_object(repaired)
        except Exception as second_error:
            logger.error(
                "JSON repair failed for stage=%s (%s: %s)",
                stage_name,
                type(second_error).__name__,
                second_error,
            )
            raise first_error


def _json_repair_prompt(stage_name: str, raw: str) -> str:
    return (
        f"Repair the malformed JSON object for stage `{stage_name}`.\n"
        "Rules:\n"
        "1. Return exactly one valid JSON object.\n"
        "2. Preserve the original keys, values, and language as much as possible.\n"
        "3. Fix only syntax problems such as missing colons, unescaped quotes, "
        "trailing commas, or markdown fences.\n"
        "4. Do not explain anything.\n\n"
        "Malformed JSON:\n"
        "```text\n"
        f"{raw}\n"
        "```"
    )


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


def _generate_with_retry(
    *,
    llm,
    label: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> str:
    transient_markers = (
        "timeout",
        "timed out",
        "connection",
        "rate limit",
        "temporarily",
        "server",
        "overloaded",
        "empty response",
        "missing choices",
        "missing message",
        "blank content",
        "response is none",
        "not subscriptable",
    )
    for attempt in range(max_retries + 1):
        try:
            return llm.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            message = str(exc).lower()
            transient = any(marker in message for marker in transient_markers)
            if attempt >= max_retries or not transient:
                raise
            wait_seconds = retry_backoff_seconds * (2 ** attempt)
            logger.warning(
                "MegaPersona LLM transient failure; retrying stage=%s attempt=%s/%s wait=%.1fs (%s: %s)",
                label,
                attempt + 1,
                max_retries,
                wait_seconds,
                type(exc).__name__,
                exc,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("unreachable retry state")


def _log_validation_issues(slot_id: str, report: ValidationReport) -> None:
    for issue in report.issues[:8]:
        logger.info(
            "Slot=%s validation issue rule=%s severity=%s message=%s",
            slot_id,
            issue.rule_id,
            issue.severity,
            issue.message.replace("\n", " ")[:500],
        )
