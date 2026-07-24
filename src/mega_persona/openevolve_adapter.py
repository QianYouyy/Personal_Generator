"""Adapter that runs MegaPersona genomes through the OpenEvolve engine."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Any

import numpy as np

from src.mega_persona.evolution import (
    EVOLUTION_PROMPT_OPERATORS,
    MegaEvolutionCandidate,
    MegaEvolutionConfig,
    MegaPersonaEvolver,
    PROMPT_POLICY_BANK,
    build_run_manifest,
    default_genome,
    default_genome_v4,
    mutate_genome,
    normalize_genome,
)
from src.mega_persona.generator import _generate_with_retry, parse_json_object
from src.mega_persona.mcts_policy import OperatorMCTSConfig, OperatorMCTSPolicy
from src.mega_persona.slots import (
    axis_names_for_binding,
    axis_roles_for_binding,
    quota_buckets_for_binding,
    schema_binding_for_genome,
)
from src.open_evolve.engine import OpenEvolve


logger = logging.getLogger(__name__)


def genome_to_code(genome: dict[str, Any]) -> str:
    """Serialize a MegaPersona genome as OpenEvolve's evolvable code string."""
    return json.dumps(genome, ensure_ascii=False, sort_keys=True, indent=2)


def genome_from_code(code: str) -> dict[str, Any]:
    """Parse OpenEvolve's code string back into a MegaPersona genome."""
    try:
        genome = json.loads(code)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenEvolve candidate is not a MegaPersona genome JSON: {exc}") from exc
    if not isinstance(genome, dict):
        raise ValueError("OpenEvolve candidate genome must be a JSON object")
    return genome


def genome_hash(genome: dict[str, Any]) -> str:
    encoded = genome_to_code(genome).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_LINEAGE_METADATA_KEYS = (
    "last_evolution_operator",
    "last_mutation",
    "openevolve_mutation",
)


def genome_phenotype_hash(genome: dict[str, Any]) -> str:
    """Hash only the genome fields that drive persona generation.

    Lineage/mutation metadata is excluded so children that differ only in how
    they were produced (operator tag, mutation timestamps) map to the same
    phenotype and can share one evaluation instead of being re-evaluated at
    full cost.
    """
    effective = {
        key: value
        for key, value in genome.items()
        if key not in _LINEAGE_METADATA_KEYS
    }
    encoded = json.dumps(effective, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MegaGenomeMutator:
    """OpenEvolve mutator for the fixed-architecture MegaPersona genome."""

    def __init__(
        self,
        random_seed: int = 1234,
        base_mutation_scale: float = 0.12,
        llm_client=None,
        enable_rule_fallback: bool = True,
        failure_dir: Path | None = None,
        fixed_operator_id: str | None = None,
        operator_family: str = "all",
        search_strategy: str = "openevolve",
        mcts_depth: int = 3,
        mcts_exploration_c: float = 1.4,
        mcts_progressive_widening: bool = False,
        mcts_reward_profile: str = "legacy",
        mcts_plateau_stagnation: int = 4,
        mcts_reward_weight_mode: str = "fixed",
    ):
        self.rng = np.random.default_rng(random_seed)
        self.base_mutation_scale = base_mutation_scale
        self.mutation_modes = ("prompt_only", "operator_only", "mixed", "numeric_only")
        self.llm_client = llm_client
        self.enable_rule_fallback = enable_rule_fallback
        self.failure_dir = failure_dir
        self.fixed_operator_id = fixed_operator_id
        self.operator_family = operator_family
        self._operator_pool = self._operators_for_family(operator_family)
        self.search_strategy = search_strategy
        self.mcts_config = OperatorMCTSConfig(
            max_depth=mcts_depth,
            exploration_c=mcts_exploration_c,
            progressive_widening=mcts_progressive_widening,
            reward_profile=mcts_reward_profile,
            plateau_stagnation=mcts_plateau_stagnation,
            reward_weight_mode=mcts_reward_weight_mode,
        )
        self.mcts_policy = self._build_mcts_policy()
        self._rng_lock = threading.Lock()
        self._failure_lock = threading.Lock()
        self._failure_count = 0
        if fixed_operator_id is not None:
            self._operator_by_id(fixed_operator_id)

    def mutate(
        self,
        parent_code: str,
        prompt: str | None = None,
        generation: int = 0,
        stagnation: int = 0,
        operator_id: str | None = None,
    ) -> str:
        parent = normalize_genome(genome_from_code(parent_code))
        with self._rng_lock:
            mode = str(self.rng.choice(self.mutation_modes))
        operator = self._operator_by_id(operator_id) if operator_id else self.choose_operator(generation, stagnation)
        stagnation_boost = 1.0 + min(stagnation, 4) * 0.15
        generation_boost = 1.0 + min(generation, 4) * 0.08
        mutation_scale = self.base_mutation_scale * stagnation_boost * generation_boost
        fallback_reason = None

        if int(parent.get("genome_version", 3)) == 4:
            with self._rng_lock:
                child = mutate_genome(
                    parent,
                    self.rng,
                    mutation_scale=mutation_scale,
                    mutation_mode="operator_only",
                    operator_id=operator["id"],
                )
            module = str(operator.get("v4_module", "unknown"))
            metadata = self._mutation_metadata(
                backend="structured_v4",
                mode="structured_v4",
                mutation_scale=mutation_scale,
                generation=generation,
                stagnation=stagnation,
                operator=operator,
            )
            metadata.update(
                {
                    "declared_edits": [module],
                    "actual_edits": [module],
                    "undeclared_edits": [],
                    "phantom_edits": [],
                    "noop_retries": 0,
                }
            )
            child["openevolve_mutation"] = metadata
            return genome_to_code(child)

        if self.llm_client is not None:
            try:
                child = self._mutate_with_llm(
                    parent=parent,
                    prompt=prompt,
                    generation=generation,
                    stagnation=stagnation,
                    mutation_mode=mode,
                    mutation_scale=mutation_scale,
                    operator=operator,
                )
                metadata = self._mutation_metadata(
                    backend="llm",
                    mode=mode,
                    mutation_scale=mutation_scale,
                    generation=generation,
                    stagnation=stagnation,
                    operator=operator,
                )
                edit_audit = child.pop("_mutation_edit_audit", None)
                if isinstance(edit_audit, dict):
                    metadata.update(edit_audit)
                child["openevolve_mutation"] = metadata
                return genome_to_code(child)
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "LLM mutator failed; falling back to rule mutation (%s)",
                    fallback_reason,
                )
                if not self.enable_rule_fallback:
                    raise

        with self._rng_lock:
            child = mutate_genome(
                parent,
                self.rng,
                mutation_scale=mutation_scale,
                mutation_mode=mode,
                operator_id=operator["id"],
            )
        child["openevolve_mutation"] = self._mutation_metadata(
            backend="rule",
            mode=mode,
            mutation_scale=mutation_scale,
            generation=generation,
            stagnation=stagnation,
            operator=operator,
            fallback_reason=fallback_reason,
        )
        return genome_to_code(child)

    def choose_operator(
        self,
        generation: int = 0,
        stagnation: int = 0,
        island_id: int | None = None,
        child_idx: int | None = None,
    ) -> dict[str, Any]:
        """Select one evolution operator.

        The signature accepts island/child metadata so the OpenEvolve engine can
        ask for an operator before parent selection.
        """
        del stagnation
        if self.fixed_operator_id:
            return self._operator_by_id(self.fixed_operator_id)
        if self.mcts_policy is not None:
            return self.mcts_policy.choose_operator(
                generation=generation,
                island_id=island_id,
                child_idx=child_idx,
            )
        return dict(self._operator_pool[int(self.rng.integers(0, len(self._operator_pool)))])

    def record_result(
        self,
        *,
        operator_id: str | None,
        parent_fitness: dict[str, float] | None,
        child_fitness: dict[str, float] | None,
        generation: int,
        island_id: int | None,
        child_idx: int | None,
        improved: bool,
        improved_metrics: list[str] | None = None,
    ) -> None:
        if self.mcts_policy is None or operator_id is None:
            return
        self.mcts_policy.record_result(
            operator_id=operator_id,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
            generation=generation,
            island_id=island_id,
            child_idx=child_idx,
            improved=improved,
            improved_metrics=improved_metrics or [],
        )

    def mcts_summary(self) -> dict[str, Any] | None:
        if self.mcts_policy is None:
            return None
        return self.mcts_policy.summary()

    def preferred_parent_metric(self, operator_id: str | None) -> str | None:
        if not operator_id:
            return None
        operator = self._operator_by_id(operator_id)
        metric = operator.get("preferred_parent_metric")
        return metric if isinstance(metric, str) else None

    def _operator_by_id(self, operator_id: str) -> dict[str, Any]:
        for operator in EVOLUTION_PROMPT_OPERATORS:
            if operator["id"] == operator_id:
                return dict(operator)
        raise ValueError(f"unknown evolution operator id: {operator_id}")

    def _operators_for_family(self, operator_family: str) -> tuple[dict[str, Any], ...]:
        family = operator_family.strip().lower()
        if family == "all":
            # Preserve the historical meaning of all: legacy + v3. Genome v4
            # is an explicit experimental surface and must be requested.
            operators = tuple(operator for operator in EVOLUTION_PROMPT_OPERATORS if "_v4_" not in operator["id"])
        elif family == "v3":
            operators = tuple(operator for operator in EVOLUTION_PROMPT_OPERATORS if "_v3_" in operator["id"])
        elif family == "v4":
            operators = tuple(operator for operator in EVOLUTION_PROMPT_OPERATORS if "_v4_" in operator["id"])
        elif family == "legacy":
            operators = tuple(
                operator
                for operator in EVOLUTION_PROMPT_OPERATORS
                if "_v3_" not in operator["id"] and "_v4_" not in operator["id"]
            )
        else:
            raise ValueError(f"unknown operator_family: {operator_family}")
        if not operators:
            raise ValueError(f"operator_family has no operators: {operator_family}")
        return tuple(dict(operator) for operator in operators)

    def operator_ids(self) -> list[str]:
        return [str(operator["id"]) for operator in self._operator_pool]

    def _build_mcts_policy(self) -> OperatorMCTSPolicy | None:
        strategy = self.search_strategy.strip().lower()
        if strategy == "openevolve":
            return None
        if strategy != "hybrid_mcts":
            raise ValueError(f"unknown search_strategy: {self.search_strategy}")
        if self.fixed_operator_id:
            logger.warning(
                "search_strategy=hybrid_mcts is ignored because fixed_operator_id=%s",
                self.fixed_operator_id,
            )
            return None
        return OperatorMCTSPolicy(
            operators=list(self._operator_pool),
            rng=self.rng,
            config=self.mcts_config,
        )

    def _mutate_with_llm(
        self,
        *,
        parent: dict[str, Any],
        prompt: str | None,
        generation: int,
        stagnation: int,
        mutation_mode: str,
        mutation_scale: float,
        operator: dict[str, Any],
    ) -> dict[str, Any]:
        base_prompt = self._mutation_prompt(
            parent=parent,
            prompt=prompt,
            generation=generation,
            stagnation=stagnation,
            mutation_mode=mutation_mode,
            mutation_scale=mutation_scale,
            operator=operator,
        )
        system_prompt = (
            "You are an OpenEvolve mutator for MegaPersona. Return only one valid JSON "
            "object. Prefer a compact patch object over a full genome. Preserve the fixed "
            "schema and make one coherent, fitness-seeking mutation rather than many random edits."
        )
        # Audit-driven retry: if a draft produces no effective change, ask once
        # more with explicit feedback. noop_retries counts discarded no-op
        # drafts (0 = clean first try, 2 = retry still produced nothing).
        noop_retries = 0
        feedback = ""
        normalized: dict[str, Any] = {}
        audit: dict[str, Any] = {}
        for _attempt in range(2):
            raw = _generate_with_retry(
                llm=self.llm_client,
                label="mutator",
                prompt=base_prompt + feedback,
                system_prompt=system_prompt,
                temperature=0.45,
                max_tokens=4200,
                max_retries=2,
                retry_backoff_seconds=1.5,
            )
            child = self._parse_or_repair_child(raw)
            declared_edits = _extract_declared_edits(child)
            normalized = self._normalize_child_genome(
                child=child,
                parent=parent,
                operator=operator,
                mutation_mode=mutation_mode,
                mutation_scale=mutation_scale,
            )
            audit = _build_edit_audit(
                parent=parent,
                child=normalized,
                declared_edits=declared_edits,
            )
            if audit["actual_edits"]:
                break
            noop_retries += 1
            feedback = _noop_retry_feedback(declared_edits)
        jittered = self._apply_numeric_jitter(normalized, mutation_scale=mutation_scale)
        if jittered["axis_bias"] or jittered["axis_stretch"]:
            audit["numeric_jitter"] = jittered
        audit["noop_retries"] = noop_retries
        normalized["_mutation_edit_audit"] = audit
        return normalized

    def _apply_numeric_jitter(
        self, genome: dict[str, Any], *, mutation_scale: float
    ) -> dict[str, list[str]]:
        """Keep the numeric surface exploring under LLM mutation.

        The LLM rarely edits axis_bias/axis_stretch (14/64 edits in the
        2026-07-20 audit), so every child gets a small Gaussian perturbation
        on the numeric layer at the rule-mutation magnitudes. This runs AFTER
        the edit audit so no-op detection stays mutator-facing; jittered axes
        are reported separately under ``numeric_jitter``.
        """
        changed: dict[str, list[str]] = {"axis_bias": [], "axis_stretch": []}
        with self._rng_lock:
            for axis, value in genome.get("axis_bias", {}).items():
                jitter = float(self.rng.normal(0.0, mutation_scale * 0.15))
                new_value = float(np.clip(value + jitter, -0.35, 0.35))
                if abs(new_value - value) > 1e-12:
                    genome["axis_bias"][axis] = new_value
                    changed["axis_bias"].append(axis)
            for axis, value in genome.get("axis_stretch", {}).items():
                jitter = float(self.rng.normal(0.0, mutation_scale * 0.25))
                new_value = float(np.clip(value + jitter, 0.55, 1.75))
                if abs(new_value - value) > 1e-12:
                    genome["axis_stretch"][axis] = new_value
                    changed["axis_stretch"].append(axis)
        return changed

    def _parse_or_repair_child(self, raw: str) -> dict[str, Any]:
        try:
            return parse_json_object(raw)
        except Exception as first_error:
            repaired = self.llm_client.generate(
                _json_repair_prompt(raw),
                system_prompt=(
                    "Repair malformed JSON. Return exactly one valid JSON object and nothing else."
                ),
                temperature=0.0,
                max_tokens=4200,
            )
            try:
                return parse_json_object(repaired)
            except Exception as second_error:
                self._write_mutator_failure(raw, repaired, first_error, second_error)
                raise ValueError(
                    f"mutator returned malformed genome JSON; repair failed "
                    f"({type(first_error).__name__}: {first_error}; "
                    f"{type(second_error).__name__}: {second_error})"
                ) from second_error

    def _normalize_child_genome(
        self,
        *,
        child: dict[str, Any],
        parent: dict[str, Any],
        operator: dict[str, Any],
        mutation_mode: str,
        mutation_scale: float,
    ) -> dict[str, Any]:
        base = default_genome()
        schema_binding = schema_binding_for_genome(parent)
        child = _child_from_patch_or_genome(parent, child)
        normalized = json.loads(json.dumps(base))
        normalized["schema_binding"] = schema_binding
        axis_names = axis_names_for_binding(schema_binding)
        axis_roles = axis_roles_for_binding(schema_binding)
        normalized["quota_weights"] = {}
        normalized["axis_bias"] = {}
        normalized["axis_stretch"] = {}

        parent_quota = parent.get("quota_weights", {})
        child_quota = child.get("quota_weights", {})
        for bucket in quota_buckets_for_binding(schema_binding):
            label = bucket.label
            default_weight = parent_quota.get(label, bucket.weight)
            value = child_quota.get(label, default_weight)
            normalized["quota_weights"][label] = _clip(_safe_float(value, default_weight), 0.02, 0.6)
        _normalize_weights(normalized["quota_weights"])

        parent_bias = parent.get("axis_bias", {})
        child_bias = child.get("axis_bias", {})
        for axis in axis_names:
            default_bias = parent_bias.get(axis, 0.0)
            value = child_bias.get(axis, default_bias)
            normalized["axis_bias"][axis] = _clip(_safe_float(value, default_bias), -0.35, 0.35)

        parent_stretch = parent.get("axis_stretch", {})
        child_stretch = child.get("axis_stretch", {})
        for axis in axis_names:
            default_stretch = parent_stretch.get(axis, 1.0)
            value = child_stretch.get(axis, default_stretch)
            normalized["axis_stretch"][axis] = _clip(_safe_float(value, default_stretch), 0.55, 1.75)

        parent_profile = parent.get("prompt_profile", {})
        child_profile = child.get("prompt_profile", {})
        for category, options in PROMPT_POLICY_BANK.items():
            default_choice = base["prompt_profile"][category]
            parent_choice = parent_profile.get(category, default_choice)
            child_choice = child_profile.get(category, parent_choice)
            normalized["prompt_profile"][category] = (
                child_choice if child_choice in options else parent_choice if parent_choice in options else default_choice
            )

        for key in (
            "agent_focus",
            "behavior_anchors",
            "repair_policy",
            "blueprint_policy",
            "axis_expression_policy",
            "cross_agent_binding_policy",
            "behavior_prediction_policy",
            "critic_policy",
        ):
            parent_value = parent.get(key, base[key])
            child_value = child.get(key, parent_value)
            if isinstance(child_value, dict):
                normalized[key].update(_clean_child_string_dict(child_value, normalized[key]))
            elif isinstance(parent_value, dict):
                normalized[key].update(_clean_child_string_dict(parent_value, normalized[key]))

        for key in ("field_requirements", "consistency_rules"):
            parent_value = parent.get(key, base[key])
            child_value = child.get(key, parent_value)
            normalized[key] = _clean_child_string_list(
                child_value if isinstance(child_value, list) else parent_value,
                fallback=base[key],
            )

        operator_meta = child.get("last_evolution_operator")
        if not isinstance(operator_meta, dict):
            operator_meta = {
                "id": operator["id"],
                "name": operator["name"],
                "instruction": operator["instruction"],
            }
        normalized["last_evolution_operator"] = operator_meta
        normalized["last_mutation"] = {
            "mode": mutation_mode,
            "scale": mutation_scale,
            "backend": "llm",
            "model": getattr(self.llm_client, "model", None),
        }
        return normalized

    def _write_mutator_failure(
        self,
        raw: str,
        repaired: str,
        first_error: Exception,
        second_error: Exception,
    ) -> None:
        if self.failure_dir is None:
            return
        try:
            with self._failure_lock:
                self._failure_count += 1
                index = self._failure_count
            self.failure_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "index": index,
                "timestamp": datetime.now().isoformat(),
                "mutator_model": getattr(self.llm_client, "model", None),
                "first_error": f"{type(first_error).__name__}: {first_error}",
                "second_error": f"{type(second_error).__name__}: {second_error}",
                "raw": raw,
                "repaired": repaired,
            }
            path = self.failure_dir / f"failure_{index:04d}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("failed to persist mutator failure artifact: %s", exc)

    def _mutation_prompt(
        self,
        *,
        parent: dict[str, Any],
        prompt: str | None,
        generation: int,
        stagnation: int,
        mutation_mode: str,
        mutation_scale: float,
        operator: dict[str, Any],
    ) -> str:
        objective = (
            "Maximize MegaPersona validation fitness by improving behavioral coverage, shadow-survey "
            "alignment, schema validity, and internal consistency at the same time."
        )
        optional_prompt = prompt.strip() if isinstance(prompt, str) and prompt.strip() else "None"
        return (
            "Mutate the following MegaPersona genome for the next OpenEvolve candidate.\n\n"
            f"Objective:\n- {objective}\n\n"
            "Return format:\n"
            "Return exactly this compact JSON shape, with no markdown and no commentary:\n"
            "{\"patch\": {\"prompt_profile\": {}, \"agent_focus\": {}, \"behavior_anchors\": {}, "
            "\"field_requirements\": [], \"consistency_rules\": [], \"repair_policy\": {}, "
            "\"blueprint_policy\": {}, \"axis_expression_policy\": {}, "
            "\"cross_agent_binding_policy\": {}, \"behavior_prediction_policy\": {}, \"critic_policy\": {}, "
            "\"axis_bias\": {}, \"axis_stretch\": {}, \"quota_weights\": {}}, "
            "\"declared_edits\": []}\n"
            "Only include fields that you actually change inside patch. Empty objects/lists are allowed.\n"
            "declared_edits must list every top-level genome field you intentionally changed "
            "(for example \"blueprint_policy\" or \"blueprint_policy.core_tension_rule\"). "
            "This audit trail is required even when you change only one field.\n\n"
            "Fixed schema rules:\n"
            "1. Return exactly one JSON object.\n"
            "2. Prefer the compact patch format above. Full child genomes are still accepted but discouraged.\n"
            "3. quota_weights must remain a full distribution over all buckets.\n"
            "4. axis_bias keys and axis_stretch keys must stay unchanged.\n"
            "5. prompt_profile values must be chosen from the existing option families.\n"
            "6. Genome v3 fields are the preferred mutation surface: blueprint_policy, "
            "axis_expression_policy, cross_agent_binding_policy, behavior_prediction_policy, critic_policy.\n"
            "7. Structural fields may be rewritten, but keep them concise and behavior-predictive.\n"
            "8. Make one coherent mutation, not random noise.\n"
            "9. Favor realistic, behavior-predictive, internally consistent personas.\n\n"
            f"Generation: {generation}\n"
            f"Stagnation: {stagnation}\n"
            f"Mutation mode: {mutation_mode}\n"
            f"Mutation scale: {mutation_scale:.4f}\n"
            f"Selected operator: {operator['id']} - {operator['name']}\n"
            f"Operator instruction: {operator['instruction']}\n"
            f"Engine prompt: {optional_prompt}\n\n"
            "Schema binding:\n"
            "```json\n"
            f"{json.dumps(schema_binding_for_genome(parent), ensure_ascii=False, indent=2)}\n"
            "```\n\n"
            "Prompt profile options:\n"
            f"{_prompt_policy_options_block()}\n\n"
            "Parent genome:\n"
            "```json\n"
            f"{genome_to_code(parent)}\n"
            "```\n\n"
            "Return only the mutated child genome JSON."
        )

    def _mutation_metadata(
        self,
        *,
        backend: str,
        mode: str,
        mutation_scale: float,
        generation: int,
        stagnation: int,
        operator: dict[str, Any],
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "backend": backend,
            "mode": mode,
            "scale": mutation_scale,
            "generation": generation,
            "stagnation": stagnation,
            "operator_id": operator.get("id"),
            "operator_name": operator.get("name"),
            "mutator_model": getattr(self.llm_client, "model", None),
            "timestamp": datetime.now().isoformat(),
        }
        if fallback_reason:
            payload["fallback_reason"] = fallback_reason
        return payload

    def get_state(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "base_mutation_scale": self.base_mutation_scale,
            "mutation_modes": list(self.mutation_modes),
            "llm_model": getattr(self.llm_client, "model", None),
            "fixed_operator_id": self.fixed_operator_id,
            "operator_family": self.operator_family,
            "search_strategy": self.search_strategy,
            "mcts_config": asdict(self.mcts_config),
            "mcts_policy_state": self.mcts_policy.get_state() if self.mcts_policy is not None else None,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.base_mutation_scale = state.get("base_mutation_scale", self.base_mutation_scale)
        self.mutation_modes = tuple(state.get("mutation_modes", self.mutation_modes))
        self.fixed_operator_id = state.get("fixed_operator_id", self.fixed_operator_id)
        self.operator_family = state.get("operator_family", self.operator_family)
        self.search_strategy = state.get("search_strategy", self.search_strategy)
        config = state.get("mcts_config", {})
        if isinstance(config, dict):
            self.mcts_config = OperatorMCTSConfig(
                max_depth=int(config.get("max_depth", self.mcts_config.max_depth)),
                exploration_c=float(config.get("exploration_c", self.mcts_config.exploration_c)),
                progressive_widening=bool(
                    config.get("progressive_widening", self.mcts_config.progressive_widening)
                ),
                reward_profile=str(config.get("reward_profile", self.mcts_config.reward_profile)),
                plateau_stagnation=int(
                    config.get("plateau_stagnation", self.mcts_config.plateau_stagnation)
                ),
                reward_weight_mode=str(
                    config.get("reward_weight_mode", self.mcts_config.reward_weight_mode)
                ),
            )
        self._operator_pool = self._operators_for_family(self.operator_family)
        if self.fixed_operator_id is not None:
            self._operator_by_id(self.fixed_operator_id)
        self.mcts_policy = self._build_mcts_policy()
        if self.mcts_policy is not None and isinstance(state.get("mcts_policy_state"), dict):
            self.mcts_policy.set_state(state["mcts_policy_state"])
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]


class MegaOpenEvolveEvaluator:
    """OpenEvolve evaluator that delegates scientific scoring to MegaPersona."""

    def __init__(self, backend: MegaPersonaEvolver):
        self.backend = backend
        self.num_personas = backend.config.n
        self.candidate_evaluation_repeats = max(
            1,
            int(getattr(backend.config, "candidate_evaluation_repeats", 1)),
        )
        self.elite_confirmation_repeats = max(
            self.candidate_evaluation_repeats,
            int(getattr(backend.config, "elite_confirmation_repeats", 1)),
        )
        self._code_to_candidate_id, self._phenotype_to_candidate_id = (
            self._load_existing_indexes()
        )
        self._lock = threading.Lock()
        self._phenotype_inflight: dict[str, tuple[str, threading.Event]] = {}
        self._candidate_sequence = backend.evaluation_count + 1

    def evaluate(self, code_str: str) -> dict[str, float]:
        return self.evaluate_with_context(code_str)

    def evaluate_with_context(
        self,
        code_str: str,
        *,
        parent_id: str | None = None,
    ) -> dict[str, float]:
        genome = genome_from_code(code_str)
        digest = genome_hash(genome)[:12]
        with self._lock:
            candidate_id = self._code_to_candidate_id.get(digest)
            if candidate_id is None:
                candidate_id = f"openevolve_{self._candidate_sequence:06d}_{digest}"
                self._candidate_sequence += 1
                self._code_to_candidate_id[digest] = candidate_id
        phenotype_digest = genome_phenotype_hash(genome)[:12]
        owner_event: threading.Event | None = None

        while owner_event is None:
            with self._lock:
                cached_payload = self.backend.store.find_candidate_result(candidate_id)
                phenotype_candidate_id = self._phenotype_to_candidate_id.get(phenotype_digest)
                inflight = self._phenotype_inflight.get(phenotype_digest)

            if (
                cached_payload is not None
                and not _is_zero_persona_cache_payload(cached_payload)
                and self._has_required_repeats(cached_payload)
            ):
                logger.info(
                    "OpenEvolve evaluator cache hit candidate=%s genome_hash=%s fitness=%.4f",
                    candidate_id,
                    digest,
                    cached_payload.get("fitness", 0.0),
                )
                return open_evolve_fitness_from_payload(cached_payload)
            if cached_payload is not None:
                logger.warning(
                    "OpenEvolve evaluator ignoring stale cache candidate=%s genome_hash=%s "
                    "cached_repeats=%s required_repeats=%s zero_persona=%s",
                    candidate_id,
                    digest,
                    _evaluation_repeat_count(cached_payload),
                    self.candidate_evaluation_repeats,
                    _is_zero_persona_cache_payload(cached_payload),
                )

            phenotype_payload = (
                self.backend.store.find_candidate_result(phenotype_candidate_id)
                if phenotype_candidate_id is not None
                else None
            )
            if (
                phenotype_payload is not None
                and not _is_zero_persona_cache_payload(phenotype_payload)
                and self._has_required_repeats(phenotype_payload)
            ):
                logger.info(
                    "OpenEvolve evaluator phenotype cache hit candidate=%s phenotype_hash=%s "
                    "source_candidate=%s; reusing prior evaluation",
                    candidate_id,
                    phenotype_digest,
                    phenotype_candidate_id,
                )
                alias_payload = self._write_phenotype_cache_alias(
                    candidate_id=candidate_id,
                    genome=genome,
                    genome_hash_digest=digest,
                    phenotype_hash_digest=phenotype_digest,
                    source_candidate_id=str(phenotype_candidate_id),
                    source_payload=phenotype_payload,
                    parent_id=parent_id,
                )
                return open_evolve_fitness_from_payload(alias_payload)
            if phenotype_candidate_id is not None:
                with self._lock:
                    if self._phenotype_to_candidate_id.get(phenotype_digest) == phenotype_candidate_id:
                        self._phenotype_to_candidate_id.pop(phenotype_digest, None)

            if inflight is None:
                with self._lock:
                    inflight = self._phenotype_inflight.get(phenotype_digest)
                    if inflight is None:
                        owner_event = threading.Event()
                        self._phenotype_inflight[phenotype_digest] = (candidate_id, owner_event)
                        break

            if inflight is not None:
                source_candidate_id, wait_event = inflight
                logger.info(
                    "OpenEvolve evaluator waiting for in-flight phenotype candidate=%s "
                    "phenotype_hash=%s source_candidate=%s",
                    candidate_id,
                    phenotype_digest,
                    source_candidate_id,
                )
                wait_event.wait()

        try:
            candidate = MegaEvolutionCandidate(
                candidate_id=candidate_id,
                genome=genome,
                generation=self._generation_for_genome(genome),
                parent_id=parent_id,
            )
            result = self._evaluate_candidate_repeats(candidate)
            candidate.fitness = result["fitness"]
            candidate.metrics = result["metrics"]
            candidate.evaluated = True
            candidate.metrics["openevolve_genome_hash"] = digest
            candidate.metrics["openevolve_phenotype_hash"] = phenotype_digest

            with self._lock:
                self.backend.evaluation_count += 1
                self.backend.store.write_evaluation(
                    evaluation_index=self.backend.evaluation_count,
                    candidate=candidate,
                    payload=result,
                )
                if not _is_zero_persona_cache_payload(result):
                    self._phenotype_to_candidate_id.setdefault(phenotype_digest, candidate_id)
                self.backend.best_candidate_id = self._best_candidate_id_after(candidate)
                self.backend._save_checkpoint()
            return open_evolve_fitness_from_payload(result)
        finally:
            with self._lock:
                inflight = self._phenotype_inflight.get(phenotype_digest)
                if inflight is not None and inflight[0] == candidate_id:
                    self._phenotype_inflight.pop(phenotype_digest, None)
                    inflight[1].set()

    def _evaluate_candidate_repeats(
        self,
        candidate: MegaEvolutionCandidate,
        *,
        repeat_count: int | None = None,
        start_index: int = 0,
    ) -> dict[str, Any]:
        repeat_count = self.candidate_evaluation_repeats if repeat_count is None else repeat_count
        repeat_payloads: list[dict[str, Any]] = []
        for offset in range(repeat_count):
            repeat_index = start_index + offset
            repeat_candidate = candidate
            if repeat_index > 0:
                repeat_candidate = MegaEvolutionCandidate(
                    candidate_id=f"{candidate.candidate_id}__repeat_{repeat_index + 1:02d}",
                    genome=candidate.genome,
                    generation=candidate.generation,
                    parent_id=candidate.parent_id,
                )
            logger.info(
                "OpenEvolve candidate=%s selection repeat=%s/%s",
                candidate.candidate_id,
                repeat_index + 1,
                start_index + repeat_count,
            )
            repeat_payloads.append(self.backend.evaluate_candidate(repeat_candidate))
        return _aggregate_candidate_evaluation_repeats(candidate, repeat_payloads)

    def confirm_evaluation(
        self,
        code_str: str,
        *,
        parent_id: str | None = None,
        required_repeats: int | None = None,
    ) -> dict[str, Any]:
        """Extend a cached candidate evaluation to the elite confirmation budget."""
        required = max(
            self.candidate_evaluation_repeats,
            int(required_repeats or self.elite_confirmation_repeats),
        )
        genome = genome_from_code(code_str)
        digest = genome_hash(genome)[:12]
        with self._lock:
            candidate_id = self._code_to_candidate_id.get(digest)
            cached_payload = (
                self.backend.store.find_candidate_result(candidate_id)
                if candidate_id is not None
                else None
            )
        if candidate_id is None or cached_payload is None:
            raise ValueError("candidate must receive its initial evaluation before confirmation")
        current_count = _evaluation_repeat_count(cached_payload)
        if current_count >= required:
            return open_evolve_fitness_from_payload(cached_payload)

        candidate = MegaEvolutionCandidate(
            candidate_id=candidate_id,
            genome=genome,
            generation=self._generation_for_genome(genome),
            parent_id=parent_id,
        )
        additional = self._evaluate_candidate_repeats(
            candidate,
            repeat_count=required - current_count,
            start_index=current_count,
        )
        repeat_payloads = _repeat_payloads_from_aggregate(cached_payload)
        repeat_payloads.extend(_repeat_payloads_from_aggregate(additional))
        result = _aggregate_candidate_evaluation_repeats(candidate, repeat_payloads)
        result["metrics"]["elite_confirmation"] = True
        result["metrics"]["elite_confirmation_repeats"] = required
        result["metrics"]["openevolve_genome_hash"] = digest
        result["metrics"]["openevolve_phenotype_hash"] = genome_phenotype_hash(genome)[:12]
        if bool(cached_payload.get("phenotype_cache_hit")):
            result["phenotype_cache_hit"] = True
            result["metrics"]["phenotype_cache_hit"] = True

        candidate.fitness = result["fitness"]
        candidate.metrics = result["metrics"]
        candidate.evaluated = True
        with self._lock:
            self.backend.evaluation_count += 1
            self.backend.store.write_evaluation(
                evaluation_index=self.backend.evaluation_count,
                candidate=candidate,
                payload=result,
            )
            self.backend.best_candidate_id = self._best_candidate_id_after(candidate)
            self.backend._save_checkpoint()
        logger.info(
            "OpenEvolve elite confirmation candidate=%s repeats=%s fitness=%.4f",
            candidate_id,
            required,
            candidate.fitness,
        )
        return open_evolve_fitness_from_payload(result)

    def _has_required_repeats(self, payload: dict[str, Any]) -> bool:
        return _evaluation_repeat_count(payload) >= self.candidate_evaluation_repeats

    def _write_phenotype_cache_alias(
        self,
        *,
        candidate_id: str,
        genome: dict[str, Any],
        genome_hash_digest: str,
        phenotype_hash_digest: str,
        source_candidate_id: str,
        source_payload: dict[str, Any],
        parent_id: str | None,
    ) -> dict[str, Any]:
        source_metrics = source_payload.get("metrics", {})
        if not isinstance(source_metrics, dict):
            source_metrics = {}
        metrics = dict(source_metrics)
        metrics["openevolve_genome_hash"] = genome_hash_digest
        metrics["openevolve_phenotype_hash"] = phenotype_hash_digest
        metrics["phenotype_cache_hit"] = True
        metrics["phenotype_cache_source_candidate_id"] = source_candidate_id

        candidate = MegaEvolutionCandidate(
            candidate_id=candidate_id,
            genome=genome,
            generation=self._generation_for_genome(genome),
            parent_id=parent_id,
            fitness=float(source_payload.get("fitness", 0.0) or 0.0),
            metrics=metrics,
            evaluated=True,
        )
        alias_payload = {
            **source_payload,
            "candidate": candidate.to_dict(),
            "fitness": candidate.fitness,
            "metrics": metrics,
            "phenotype_cache_hit": True,
            "phenotype_cache_source_candidate_id": source_candidate_id,
            "phenotype_cache_source_genome_hash": (
                source_payload.get("metrics", {}).get("openevolve_genome_hash")
                if isinstance(source_payload.get("metrics"), dict)
                else None
            ),
            "openevolve_genome_hash": genome_hash_digest,
            "openevolve_phenotype_hash": phenotype_hash_digest,
        }
        with self._lock:
            self.backend.store.write_cached_evaluation_alias(
                candidate=candidate,
                payload=alias_payload,
            )
        return alias_payload

    def candidate_id_for_code(self, code_str: str) -> str | None:
        digest = genome_hash(genome_from_code(code_str))[:12]
        with self._lock:
            return self._code_to_candidate_id.get(digest)

    @staticmethod
    def _generation_for_genome(genome: dict[str, Any]) -> int:
        mutation = genome.get("openevolve_mutation")
        if isinstance(mutation, dict):
            try:
                return max(0, int(mutation.get("generation", 0)))
            except (TypeError, ValueError):
                return 0
        return 0

    def _best_candidate_id_after(self, candidate: MegaEvolutionCandidate) -> str:
        current = self.backend.best_candidate_id
        if current is None:
            return candidate.candidate_id
        old_payload = self.backend.store.find_candidate_result(current)
        old_fitness = (
            old_payload.get("fitness", float("-inf"))
            if old_payload is not None
            else float("-inf")
        )
        if (candidate.fitness or 0.0) > old_fitness:
            return candidate.candidate_id
        return current

    def _load_existing_indexes(self) -> tuple[dict[str, str], dict[str, str]]:
        code_index: dict[str, str] = {}
        phenotype_index: dict[str, str] = {}
        candidates_dir = self.backend.store.candidates_dir
        if not candidates_dir.exists():
            return code_index, phenotype_index
        for path in sorted(candidates_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                genome = payload.get("genome")
                candidate_id = payload.get("candidate_id")
                if isinstance(genome, dict) and candidate_id:
                    code_index[genome_hash(genome)[:12]] = candidate_id
                    phenotype_index.setdefault(
                        genome_phenotype_hash(genome)[:12], candidate_id
                    )
            except Exception:
                continue
        return code_index, phenotype_index


class MegaPersonaOpenEvolveRunner:
    """Convenience wrapper that wires MegaPersona evaluation into OpenEvolve."""

    def __init__(
        self,
        config: MegaEvolutionConfig,
        output_dir: Path,
        resume: bool = False,
        mutator_llm_client=None,
        llm_client=None,
        simulator_llm_client=None,
        children_per_island: int = 1,
        base_mutation_scale: float = 0.12,
        fixed_operator_id: str | None = None,
        operator_family: str = "all",
        genome_version: int = 3,
        search_strategy: str = "openevolve",
        mcts_depth: int = 3,
        mcts_exploration_c: float = 1.4,
        mcts_progressive_widening: bool = False,
        mcts_reward_profile: str = "legacy",
        mcts_plateau_stagnation: int = 4,
        mcts_reward_weight_mode: str = "fixed",
        parent_selection: str = "operator_preferred",
        extinction_interval: int | None = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir = output_dir / "mega_eval"
        self.open_evolve_dir = output_dir / "open_evolve"
        self.open_evolve_dir.mkdir(parents=True, exist_ok=True)
        self.children_per_island = children_per_island
        self.resume = resume
        self.fixed_operator_id = fixed_operator_id
        self.operator_family = operator_family
        self.genome_version = int(genome_version)
        self.search_strategy = search_strategy
        self.mcts_depth = mcts_depth
        self.mcts_exploration_c = mcts_exploration_c
        self.mcts_progressive_widening = mcts_progressive_widening
        self.mcts_reward_profile = mcts_reward_profile
        self.mcts_plateau_stagnation = mcts_plateau_stagnation
        self.mcts_reward_weight_mode = mcts_reward_weight_mode
        self.parent_selection = parent_selection
        self.extinction_interval = extinction_interval

        if self.genome_version not in (3, 4):
            raise ValueError(f"unknown genome_version: {self.genome_version}")
        if self.genome_version == 4 and self.operator_family != "v4":
            raise ValueError("Genome v4 requires operator_family='v4'")
        if self.genome_version == 3 and self.operator_family == "v4":
            raise ValueError("operator_family='v4' requires genome_version=4")
        if self.fixed_operator_id:
            fixed_is_v4 = "_v4_" in self.fixed_operator_id
            if fixed_is_v4 != (self.genome_version == 4):
                raise ValueError(
                    "fixed operator genome family does not match genome_version"
                )

        if resume and not (self.open_evolve_dir / "checkpoint.json").exists():
            raise FileNotFoundError(
                f"OpenEvolve checkpoint not found: {self.open_evolve_dir / 'checkpoint.json'}"
            )

        backend_resume = resume and (self.eval_dir / "checkpoint.json").exists()
        initial_genome = default_genome_v4() if self.genome_version == 4 else default_genome()
        self.backend = MegaPersonaEvolver(
            config=config,
            output_dir=self.eval_dir,
            resume=backend_resume,
            llm_client=llm_client,
            simulator_llm_client=simulator_llm_client,
            initial_genome=initial_genome,
        )
        self.mutator = MegaGenomeMutator(
            random_seed=config.random_seed,
            base_mutation_scale=base_mutation_scale,
            llm_client=mutator_llm_client,
            failure_dir=self.open_evolve_dir / "mutator_failures",
            fixed_operator_id=fixed_operator_id,
            operator_family=operator_family,
            search_strategy=search_strategy,
            mcts_depth=mcts_depth,
            mcts_exploration_c=mcts_exploration_c,
            mcts_progressive_widening=mcts_progressive_widening,
            mcts_reward_profile=mcts_reward_profile,
            mcts_plateau_stagnation=mcts_plateau_stagnation,
            mcts_reward_weight_mode=mcts_reward_weight_mode,
        )
        self.evaluator = MegaOpenEvolveEvaluator(self.backend)
        self.engine = self._load_or_create_engine(resume=resume)
        if self.parent_selection not in ("operator_preferred", "objective_rotation"):
            raise ValueError(f"unknown parent_selection: {self.parent_selection}")
        self.engine.parent_selection = self.parent_selection
        if self.extinction_interval is not None:
            self._apply_extinction_interval(self.extinction_interval)

    def run(
        self,
        argv: list[str] | None = None,
        mutator_model_key: str | None = None,
        mutator_model: str | None = None,
        mutator_api_base: str | None = None,
        mutator_api_key_env: str | None = None,
        model_key: str | None = None,
        llm_provider: str | None = None,
        persona_model: str | None = None,
        persona_api_base: str | None = None,
        persona_api_key_env: str | None = None,
        simulator_model_key: str | None = None,
        simulator_model: str | None = None,
        simulator_api_base: str | None = None,
        simulator_api_key_env: str | None = None,
    ) -> MegaEvolutionCandidate:
        manifest = build_run_manifest(
            config=self.config,
            argv=argv,
            resume=self.resume,
            mutator_model_key=mutator_model_key,
            mutator_model=mutator_model,
            mutator_api_base=mutator_api_base,
            mutator_api_key_env=mutator_api_key_env,
            model_key=model_key,
            llm_provider=llm_provider,
            persona_model=persona_model,
            persona_api_base=persona_api_base,
            persona_api_key_env=persona_api_key_env,
            simulator_model_key=simulator_model_key,
            simulator_model=simulator_model,
            simulator_api_base=simulator_api_base,
            simulator_api_key_env=simulator_api_key_env,
        )
        manifest["engine"] = "src.open_evolve.engine.OpenEvolve"
        manifest["open_evolve_checkpoint_dir"] = str(self.open_evolve_dir)
        manifest["mega_eval_dir"] = str(self.eval_dir)
        manifest["shadow_survey_hashes"] = self.backend.survey_hashes
        manifest["fixed_operator_id"] = self.fixed_operator_id
        manifest["genome_version"] = self.genome_version
        manifest["operator_family"] = self.operator_family
        manifest["operator_pool"] = self.mutator.operator_ids()
        manifest["search_strategy"] = self.search_strategy
        manifest["parent_selection"] = self.parent_selection
        manifest["mcts_config"] = {
            "max_depth": self.mcts_depth,
            "exploration_c": self.mcts_exploration_c,
            "progressive_widening": self.mcts_progressive_widening,
            "reward_profile": self.mcts_reward_profile,
            "plateau_stagnation": self.mcts_plateau_stagnation,
            "reward_weight_mode": self.mcts_reward_weight_mode,
        }
        self._write_json(self.output_dir / "manifest.json", manifest)

        best = self.engine.run(
            max_generations=self.config.generations,
            children_per_island=self.children_per_island,
            max_workers=self.config.max_workers,
        )
        if best is None:
            raise RuntimeError("OpenEvolve finished without a best candidate")

        best_candidate = self._best_mega_candidate(best.code, best.fitness)
        final_test_report = self.backend.evaluate_final_test(best_candidate)
        self.backend.store.write_final_test_report(final_test_report)
        self.backend.store.write_final_summary(
            best_candidate,
            [best_candidate],
            self.config,
            final_test_report,
        )
        self._write_root_summary(best_candidate, best.fitness, final_test_report)
        return best_candidate

    def _load_or_create_engine(self, resume: bool) -> OpenEvolve:
        checkpoint = self.open_evolve_dir / "checkpoint.json"
        if resume and checkpoint.exists():
            engine = OpenEvolve.from_checkpoint(
                str(checkpoint),
                mutator=self.mutator,
                evaluator=self.evaluator,
                questionnaires=list(self.backend.survey_splits.validation),
            )
            engine.checkpoint_path = self.open_evolve_dir
            engine.max_workers = max(1, int(self.config.max_workers))
            # mutation 线程数跟随岛屿数（每岛 1 个子代时正好全并行）
            engine.mutation_max_workers = max(1, int(self.config.population_size))
            return engine

        seed_genome = default_genome_v4() if self.genome_version == 4 else default_genome()
        seed_codes = {"mega_default": genome_to_code(seed_genome)}
        engine = OpenEvolve(
            mutator=self.mutator,
            evaluator=self.evaluator,
            questionnaires=list(self.backend.survey_splits.validation),
            seed_codes=seed_codes,
            initial_seed_distribution={"mega_default": self.config.population_size},
            num_islands=self.config.population_size,
            max_workers=self.config.max_workers,
            checkpoint_path=self.open_evolve_dir,
        )
        # mutation 线程数跟随岛屿数（每岛 1 个子代时正好全并行）
        engine.mutation_max_workers = max(1, int(self.config.population_size))
        return engine

    def _apply_extinction_interval(self, extinction_interval: int) -> None:
        interval = max(2, int(extinction_interval))
        self.engine.extinction_interval = interval
        self.engine.extinction_stagnation_threshold = max(2, interval // 10)
        if hasattr(self.engine, "_effective_interval"):
            self.engine._effective_interval = interval
        logger.info("OpenEvolve extinction interval override: every %s generations", interval)

    def _best_mega_candidate(
        self,
        best_code: str,
        open_evolve_fitness: dict[str, float],
    ) -> MegaEvolutionCandidate:
        genome = genome_from_code(best_code)
        candidate_id = self.evaluator.candidate_id_for_code(best_code)
        if candidate_id is None:
            raise FileNotFoundError("Best OpenEvolve genome has no stored MegaPersona evaluation")
        payload = self.backend.store.find_candidate_result(candidate_id)
        if payload is None:
            raise FileNotFoundError(f"missing stored evaluation for best candidate {candidate_id}")
        stored_candidate = payload.get("candidate", {})
        candidate = MegaEvolutionCandidate(
            candidate_id=candidate_id,
            genome=genome,
            generation=int(stored_candidate.get("generation", 0)),
            parent_id=stored_candidate.get("parent_id"),
            fitness=payload["fitness"],
            metrics=payload["metrics"] | {"open_evolve_fitness": open_evolve_fitness},
            evaluated=True,
        )
        return candidate

    def _write_root_summary(
        self,
        best: MegaEvolutionCandidate,
        open_evolve_fitness: dict[str, float],
        final_test_report: dict[str, Any],
    ) -> None:
        payload = {
            "engine": "src.open_evolve.engine.OpenEvolve",
            "config": {**asdict(self.config), "seeds": list(self.config.seeds)},
            "best": best.to_dict(),
            "open_evolve_fitness": open_evolve_fitness,
            "multi_objective_best_candidates": multi_objective_best_candidates(
                self.backend.store.candidates_dir
            ),
            "final_test_report": final_test_report,
            "mega_eval_dir": str(self.eval_dir),
            "open_evolve_checkpoint_dir": str(self.open_evolve_dir),
            "search_strategy": self.search_strategy,
            "mcts_summary": self.mutator.mcts_summary(),
            "completed_at": datetime.now().isoformat(),
        }
        self._write_json(self.output_dir / "final_summary.json", payload)
        (self.output_dir / "final_summary.md").write_text(
            _root_summary_markdown(payload),
            encoding="utf-8",
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


_MULTI_OBJECTIVE_REPORT_METRICS = (
    "schema_fitness.mean",
    "validation_behavior_coverage.mean",
    "validation_behavior_balanced_diversity.mean",
    "validation_behavior_avg_dist.mean",
    "validation_shadow_alignment.mean",
    "validation_shadow_mae.mean",
    "internal_consistency.mean",
    "axis_alignment.mean",
    "axis_target_mae.mean",
    "consistency_issue_rate.mean",
    "strict_consistency_error.mean",
    "slot_coverage.mean",
    "near_duplicate_rate.mean",
)

_MULTI_OBJECTIVE_SELECTORS = (
    ("global_best", "fitness", "max"),
    ("research_score_v2_best", "open_evolve_fitness.research_score_v2", "max"),
    ("coverage_best", "validation_behavior_coverage.mean", "max"),
    ("diversity_best", "validation_behavior_balanced_diversity.mean", "max"),
    ("strict_consistency_best", "strict_consistency_error.mean", "min"),
    ("shadow_mae_best", "validation_shadow_mae.mean", "min"),
    ("axis_target_best", "axis_target_mae.mean", "min"),
    ("schema_best", "schema_fitness.mean", "max"),
)


def multi_objective_best_candidates(candidates_dir: Path) -> dict[str, Any]:
    """Return validation-only best candidates for several diagnostic objectives."""
    candidates = _load_candidate_report_rows(candidates_dir)
    roles: dict[str, Any] = {}
    for role, metric, direction in _MULTI_OBJECTIVE_SELECTORS:
        ranked = [
            row for row in candidates
            if _row_metric(row, metric) is not None
        ]
        if not ranked:
            continue
        if direction == "max":
            selected = max(
                ranked,
                key=lambda row: (
                    _row_metric(row, metric) or float("-inf"),
                    float(row.get("fitness", float("-inf"))),
                    -int(row.get("generation", 0)),
                ),
            )
        else:
            selected = min(
                ranked,
                key=lambda row: (
                    _row_metric(row, metric) or float("inf"),
                    -float(row.get("fitness", float("-inf"))),
                    int(row.get("generation", 0)),
                ),
            )
        roles[role] = {
            **selected,
            "selection_metric": metric,
            "selection_direction": direction,
            "selection_value": _row_metric(selected, metric),
        }
    return {
        "candidates_scanned": len(candidates),
        "roles": roles,
    }


def multi_objective_best_candidates_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown table for a multi-objective candidate report."""
    roles = report.get("roles", {})
    lines = [
        "# Multi-Objective Candidate Diagnostics",
        "",
        f"- candidates scanned: `{int(report.get('candidates_scanned', 0))}`",
        "",
        "| Role | Candidate | Gen | Operator | Selected metric | Value | Fitness | Coverage | Diversity | Strict err | Shadow MAE |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, item in roles.items():
        lines.append(
            "| "
            f"{role} | "
            f"`{item.get('candidate_id', '')}` | "
            f"{int(item.get('generation', 0))} | "
            f"`{item.get('operator_id', '')}` | "
            f"{item.get('selection_metric', '')} | "
            f"{_format_report_value(item.get('selection_value'))} | "
            f"{_format_report_value(item.get('fitness'))} | "
            f"{_format_report_value(item.get('validation_behavior_coverage.mean'))} | "
            f"{_format_report_value(item.get('validation_behavior_balanced_diversity.mean'))} | "
            f"{_format_report_value(item.get('strict_consistency_error.mean'))} | "
            f"{_format_report_value(item.get('validation_shadow_mae.mean'))} |"
        )
    return "\n".join(lines) + "\n"


def _load_candidate_report_rows(candidates_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not candidates_dir.exists():
        return rows
    for path in sorted(candidates_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable candidate report payload: %s", path)
            continue
        if not isinstance(payload, dict) or payload.get("fitness") is None:
            continue
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        genome = payload.get("genome", {})
        if not isinstance(genome, dict):
            genome = {}
        open_evolve_fitness = metrics.get("open_evolve_fitness")
        if not isinstance(open_evolve_fitness, dict):
            open_evolve_fitness = open_evolve_fitness_from_payload(payload)
        row: dict[str, Any] = {
            "candidate_id": str(payload.get("candidate_id", path.stem)),
            "generation": int(payload.get("generation", 0) or 0),
            "parent_id": payload.get("parent_id"),
            "operator_id": _candidate_operator_id(genome),
            "fitness": _safe_float(payload.get("fitness"), 0.0),
            "open_evolve_fitness": {
                key: _safe_float(value, 0.0)
                for key, value in open_evolve_fitness.items()
            },
        }
        for key in _MULTI_OBJECTIVE_REPORT_METRICS:
            if key in metrics:
                row[key] = _safe_float(metrics.get(key), 0.0)
        if "validation_behavior_balanced_diversity.mean" not in row:
            row["validation_behavior_balanced_diversity.mean"] = row.get(
                "validation_behavior_avg_dist.mean"
            )
        if "validation_shadow_mae.mean" not in row and "validation_shadow_alignment.mean" in row:
            row["validation_shadow_mae.mean"] = max(
                0.0,
                1.0 - _safe_float(row.get("validation_shadow_alignment.mean"), 0.0),
            )
        rows.append(row)
    return rows


def _candidate_operator_id(genome: dict[str, Any]) -> str:
    mutation_meta = genome.get("openevolve_mutation")
    if isinstance(mutation_meta, dict):
        operator_id = mutation_meta.get("operator_id")
        if operator_id:
            return str(operator_id)
    last_operator = genome.get("last_evolution_operator")
    if isinstance(last_operator, dict):
        operator_id = last_operator.get("id") or last_operator.get("operator_id")
        if operator_id:
            return str(operator_id)
    if isinstance(last_operator, str):
        return last_operator
    return ""


def _row_metric(row: dict[str, Any], key: str) -> float | None:
    value: Any = row
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            value = row.get(key)
            break
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_report_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def open_evolve_fitness_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Map MegaPersona validation metrics into OpenEvolve's elite slots."""
    metrics = payload.get("metrics", {})
    fitness = float(payload.get("fitness", 0.0))
    coverage = float(metrics.get("validation_behavior_coverage.mean", 0.0))
    shadow_mae = float(
        metrics.get(
            "validation_shadow_mae.mean",
            max(0.0, 1.0 - float(metrics.get("validation_shadow_alignment.mean", 0.0))),
        )
    )
    axis_target_mae = float(
        metrics.get(
            "axis_target_mae.mean",
            max(0.0, 1.0 - float(metrics.get("axis_alignment.mean", 0.0))),
        )
    )
    issue_rate = float(metrics.get("consistency_issue_rate.mean", 1.0))
    strict_error = float(metrics.get("strict_consistency_error.mean", axis_target_mae))
    shadow_mae_score = _clip01(1.0 - shadow_mae)
    axis_target_score = _clip01(1.0 - axis_target_mae)
    issue_rate_score = _clip01(1.0 - issue_rate)
    strict_consistency_score = _clip01(1.0 - strict_error)
    research_score_v2 = (
        fitness
        * (0.70 + 0.30 * strict_consistency_score)
        * (0.70 + 0.30 * shadow_mae_score)
    )
    diversity = float(
        metrics.get(
            "validation_behavior_balanced_diversity.mean",
            metrics.get("validation_behavior_avg_dist.mean", coverage),
        )
    )
    mapped: dict[str, Any] = {
        "global_best": fitness,
        "research_score_v2": research_score_v2,
        "coverage_elite": coverage,
        "alignment_elite": float(metrics.get("validation_shadow_alignment.mean", 0.0)),
        "shadow_mae_elite": shadow_mae_score,
        "consistency_elite": float(metrics.get("internal_consistency.mean", 0.0)),
        "axis_target_elite": axis_target_score,
        "issue_rate_elite": issue_rate_score,
        "strict_consistency_elite": strict_consistency_score,
        "diversity_elite": diversity,
        "schema_elite": float(metrics.get("schema_fitness.mean", 0.0)),
    }
    if bool(payload.get("phenotype_cache_hit")) or bool(metrics.get("phenotype_cache_hit")):
        mapped["phenotype_cache_hit"] = True
    return mapped


def _aggregate_candidate_evaluation_repeats(
    candidate: MegaEvolutionCandidate,
    repeat_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate independent full-pipeline evaluations for selection."""
    if not repeat_payloads:
        raise ValueError("candidate evaluation requires at least one repeat")

    repeat_count = len(repeat_payloads)
    fitness_values = np.asarray(
        [float(payload.get("fitness", 0.0) or 0.0) for payload in repeat_payloads],
        dtype=float,
    )
    fitness_std = float(np.std(fitness_values, ddof=1)) if repeat_count > 1 else 0.0
    fitness_sem = fitness_std / float(np.sqrt(repeat_count))

    first_metrics = repeat_payloads[0].get("metrics", {})
    metrics = dict(first_metrics) if isinstance(first_metrics, dict) else {}
    metric_keys: set[str] = set()
    for payload in repeat_payloads:
        payload_metrics = payload.get("metrics", {})
        if isinstance(payload_metrics, dict):
            metric_keys.update(payload_metrics)

    metric_stats: dict[str, dict[str, float | int]] = {}
    for key in sorted(metric_keys):
        values: list[float] = []
        for payload in repeat_payloads:
            payload_metrics = payload.get("metrics", {})
            value = payload_metrics.get(key) if isinstance(payload_metrics, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                values.append(numeric)
        if not values:
            continue
        values_array = np.asarray(values, dtype=float)
        metric_std = float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0
        metrics[key] = float(np.mean(values_array))
        metric_stats[key] = {
            "n": len(values),
            "values": [float(value) for value in values],
            "mean": metrics[key],
            "std": metric_std,
            "sem": metric_std / float(np.sqrt(len(values))),
        }

    metrics["evaluation_repeats"] = repeat_count
    metrics["selection_fitness_std"] = fitness_std
    metrics["selection_fitness_sem"] = fitness_sem
    return {
        "candidate": candidate.to_dict(),
        "fitness": float(np.mean(fitness_values)),
        "metrics": metrics,
        # The sealed test consumes one stored persona population. Repeat 1 is
        # fixed in advance so validation outcomes cannot choose that population.
        "per_seed": repeat_payloads[0].get("per_seed", []),
        "evaluation_repeat_per_seed": [
            payload.get("per_seed", []) for payload in repeat_payloads
        ],
        "evaluation_repeats": repeat_count,
        "selection_aggregation": "mean",
        "selection_representative_repeat": 1,
        "repeat_summary": {
            "count": repeat_count,
            "fitness": {
                "values": [float(value) for value in fitness_values],
                "mean": float(np.mean(fitness_values)),
                "std": fitness_std,
                "sem": fitness_sem,
            },
            "metrics": metric_stats,
        },
    }


def _repeat_payloads_from_aggregate(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rehydrate repeat-level numeric values needed to append confirmations."""
    count = _evaluation_repeat_count(payload)
    summary = payload.get("repeat_summary", {})
    fitness_summary = summary.get("fitness", {}) if isinstance(summary, dict) else {}
    fitness_values = fitness_summary.get("values") if isinstance(fitness_summary, dict) else None
    if not isinstance(fitness_values, list) or len(fitness_values) != count:
        fitness_values = [float(payload.get("fitness", 0.0) or 0.0)] * count

    metrics_summary = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    metrics_by_repeat: list[dict[str, float]] = [{} for _ in range(count)]
    if isinstance(metrics_summary, dict):
        for key, stat in metrics_summary.items():
            if not isinstance(stat, dict):
                continue
            values = stat.get("values")
            if not isinstance(values, list) or len(values) != count:
                mean = stat.get("mean")
                if not isinstance(mean, (int, float, np.number)):
                    continue
                values = [float(mean)] * count
            for index, value in enumerate(values):
                if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
                    metrics_by_repeat[index][key] = float(value)

    if count == 1 and not metrics_by_repeat[0]:
        raw_metrics = payload.get("metrics", {})
        if isinstance(raw_metrics, dict):
            metrics_by_repeat[0] = {
                key: float(value)
                for key, value in raw_metrics.items()
                if not isinstance(value, bool)
                and isinstance(value, (int, float, np.number))
                and np.isfinite(float(value))
            }

    per_seed_groups = payload.get("evaluation_repeat_per_seed")
    if not isinstance(per_seed_groups, list) or len(per_seed_groups) != count:
        per_seed_groups = [payload.get("per_seed", [])] + [[] for _ in range(count - 1)]
    return [
        {
            "fitness": float(fitness_values[index]),
            "metrics": metrics_by_repeat[index],
            "per_seed": per_seed_groups[index],
        }
        for index in range(count)
    ]


def _evaluation_repeat_count(payload: dict[str, Any]) -> int:
    value = payload.get("evaluation_repeats")
    if value is None:
        repeat_summary = payload.get("repeat_summary", {})
        if isinstance(repeat_summary, dict):
            value = repeat_summary.get("count")
    if value is None:
        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            value = metrics.get("evaluation_repeats")
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _root_summary_markdown(payload: dict[str, Any]) -> str:
    best = payload["best"]
    best_metrics = best.get("metrics", {})
    test_metrics = payload.get("final_test_report", {}).get("metrics", {})
    lines = [
        "# MegaPersona OpenEvolve Summary",
        "",
        f"- engine: `{payload['engine']}`",
        f"- best candidate: `{best['candidate_id']}`",
        f"- validation fitness: `{best.get('fitness', 0.0):.4f}`",
        f"- candidate evaluation repeats: `{int(best_metrics.get('evaluation_repeats', 1))}`",
        "- validation fitness uncertainty: "
        f"`std={float(best_metrics.get('selection_fitness_std', 0.0)):.4f}, "
        f"sem={float(best_metrics.get('selection_fitness_sem', 0.0)):.4f}`",
        f"- search strategy: `{payload.get('search_strategy', 'openevolve')}`",
        f"- mega eval dir: `{payload['mega_eval_dir']}`",
        f"- OpenEvolve checkpoint dir: `{payload['open_evolve_checkpoint_dir']}`",
        "",
        "## OpenEvolve Fitness",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in payload.get("open_evolve_fitness", {}).items():
        lines.append(f"| {key} | {value:.4f} |")
    multi_objective = payload.get("multi_objective_best_candidates")
    if isinstance(multi_objective, dict) and multi_objective.get("roles"):
        lines.extend(
            [
                "",
                "## Multi-Objective Best Candidates",
                "",
                "| Role | Candidate | Gen | Operator | Selected metric | Value | Fitness | Coverage | Diversity | Strict err | Shadow MAE |",
                "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for role, item in multi_objective.get("roles", {}).items():
            lines.append(
                "| "
                f"{role} | "
                f"`{item.get('candidate_id', '')}` | "
                f"{int(item.get('generation', 0))} | "
                f"`{item.get('operator_id', '')}` | "
                f"{item.get('selection_metric', '')} | "
                f"{_format_report_value(item.get('selection_value'))} | "
                f"{_format_report_value(item.get('fitness'))} | "
                f"{_format_report_value(item.get('validation_behavior_coverage.mean'))} | "
                f"{_format_report_value(item.get('validation_behavior_balanced_diversity.mean'))} | "
                f"{_format_report_value(item.get('strict_consistency_error.mean'))} | "
                f"{_format_report_value(item.get('validation_shadow_mae.mean'))} |"
            )
    mcts_summary = payload.get("mcts_summary")
    if isinstance(mcts_summary, dict):
        lines.extend(
            [
                "",
                "## Hybrid MCTS Operator Policy",
                "",
                f"- total policy updates: `{mcts_summary.get('total_results', 0)}`",
                "",
                "| Operator | Visits | Mean reward |",
                "|---|---:|---:|",
            ]
        )
        for item in mcts_summary.get("root_operator_stats", [])[:12]:
            lines.append(
                f"| {item.get('operator_id', '')} | "
                f"{int(item.get('visits', 0))} | "
                f"{float(item.get('mean_reward', 0.0)):.6f} |"
            )
        top_paths = mcts_summary.get("top_paths", [])[:5]
        if top_paths:
            lines.extend(["", "Top paths:", ""])
            for item in top_paths:
                path = " -> ".join(item.get("path", []))
                lines.append(
                    f"- `{path}`: visits={int(item.get('visits', 0))}, "
                    f"mean_reward={float(item.get('mean_reward', 0.0)):.6f}"
                )
    if test_metrics:
        lines.extend(
            [
                "",
                "## Sealed Test",
                "",
                f"- evaluation repeats: `{int(payload.get('final_test_report', {}).get('evaluation_repeats', 1))}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for key, value in test_metrics.items():
            lines.append(f"| {key} | {value:.4f} |")
    return "\n".join(lines) + "\n"


_PATCHABLE_DICT_KEYS = {
    "quota_weights",
    "axis_bias",
    "axis_stretch",
    "prompt_profile",
    "agent_focus",
    "behavior_anchors",
    "repair_policy",
    "blueprint_policy",
    "axis_expression_policy",
    "cross_agent_binding_policy",
    "behavior_prediction_policy",
    "critic_policy",
}
_PATCHABLE_LIST_KEYS = {"field_requirements", "consistency_rules"}


_EDIT_AUDIT_KEYS = (
    "quota_weights",
    "axis_bias",
    "axis_stretch",
    "prompt_profile",
    "agent_focus",
    "behavior_anchors",
    "field_requirements",
    "consistency_rules",
    "repair_policy",
    "blueprint_policy",
    "axis_expression_policy",
    "cross_agent_binding_policy",
    "behavior_prediction_policy",
    "critic_policy",
)


def _extract_declared_edits(raw_child: dict[str, Any]) -> list[str]:
    """Read the mutator's declared edit list from its raw response."""
    declared = raw_child.get("declared_edits")
    if not isinstance(declared, list):
        return []
    return [str(item).strip() for item in declared if isinstance(item, str) and item.strip()]


def _build_edit_audit(
    *,
    parent: dict[str, Any],
    child: dict[str, Any],
    declared_edits: list[str],
) -> dict[str, Any]:
    """Compare the mutator's declared edits against what actually changed.

    This makes LLM mutations auditable: ``actual_edits`` is the ground truth
    diff over the evolvable surface, ``undeclared_edits`` are changes the
    mutator made without declaring, and ``phantom_edits`` are declared fields
    that ended up identical to the parent.
    """
    actual = sorted(key for key in _EDIT_AUDIT_KEYS if parent.get(key) != child.get(key))
    declared_top = {item.split(".")[0] for item in declared_edits}
    undeclared = [key for key in actual if key not in declared_top]
    phantom = sorted(key for key in declared_top if key in _EDIT_AUDIT_KEYS and key not in actual)
    return {
        "declared_edits": list(declared_edits),
        "actual_edits": actual,
        "undeclared_edits": undeclared,
        "phantom_edits": phantom,
    }


def _noop_retry_feedback(declared_edits: list[str]) -> str:
    declared = ", ".join(declared_edits) if declared_edits else "(none declared)"
    return (
        "\n\nYour previous draft produced NO effective change: after normalization every "
        "genome field is identical to the parent (declared edits: "
        f"{declared}). Rewriting a field with equivalent wording counts as no change. "
        "You must materially alter at least one field's value inside patch."
    )


def _child_from_patch_or_genome(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Return a child genome, accepting either a full genome or a compact patch.

    The LLM mutator is asked to emit ``{"patch": ...}`` so the response remains
    short and JSON-safe. Older tests and checkpoints may still provide a full
    child genome; those continue to work unchanged.
    """
    patch = child.get("patch")
    if not isinstance(patch, dict):
        return child

    merged = json.loads(json.dumps(parent))
    for key in _PATCHABLE_DICT_KEYS:
        value = patch.get(key)
        if not isinstance(value, dict):
            continue
        target = merged.setdefault(key, {})
        if not isinstance(target, dict):
            target = {}
            merged[key] = target
        for item_key, item_value in value.items():
            target[item_key] = item_value

    for key in _PATCHABLE_LIST_KEYS:
        value = patch.get(key)
        if isinstance(value, list):
            merged[key] = value

    operator_meta = child.get("last_evolution_operator")
    if isinstance(operator_meta, dict):
        merged["last_evolution_operator"] = operator_meta
    return merged


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clean_child_string_dict(
    value: dict[str, Any],
    fallback: dict[str, str],
    *,
    max_chars: int = 260,
) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, default in fallback.items():
        raw = value.get(key, default)
        if not isinstance(raw, str):
            raw = str(default)
        cleaned[key] = " ".join(raw.split())[:max_chars].rstrip()
    return cleaned


def _clean_child_string_list(
    value: Any,
    *,
    fallback: list[str],
    max_items: int = 8,
    max_chars: int = 180,
) -> list[str]:
    if not isinstance(value, list):
        return list(fallback[:max_items])
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())[:max_chars].rstrip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned or list(fallback[:max_items])


def _is_zero_persona_cache_payload(payload: dict[str, Any]) -> bool:
    """Return true for stale failed evaluations caused by 0 generated personas."""
    if float(payload.get("fitness", 0.0) or 0.0) > 0.0:
        return False
    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, list) or not per_seed:
        return False
    return all(not seed_result.get("personas") for seed_result in per_seed if isinstance(seed_result, dict))


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _normalize_weights(weights: dict[str, float]) -> None:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        equal = 1.0 / max(1, len(weights))
        for key in weights:
            weights[key] = equal
        return
    for key in weights:
        weights[key] = max(weights[key], 0.0) / total


def _json_repair_prompt(raw: str) -> str:
    return (
        "Repair this malformed MegaPersona genome JSON.\n"
        "Rules:\n"
        "1. Return exactly one valid JSON object.\n"
        "2. Keep the original structure and values whenever possible.\n"
        "3. Fix syntax only.\n\n"
        "Malformed JSON:\n"
        "```text\n"
        f"{raw}\n"
        "```"
    )


def _prompt_policy_options_block() -> str:
    lines = []
    for category, options in PROMPT_POLICY_BANK.items():
        lines.append(f"- {category}: {', '.join(sorted(options.keys()))}")
    return "\n".join(lines)
