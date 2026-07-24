"""Target slot sampling for coverage-guided MegaPersona generation."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import qmc


AXIS_NAMES = (
    "cognitive_abstraction",
    "motivation_autonomy",
    "self_regulation_resilience",
)

AXIS_ROLE_MAP = {
    "cognitive_core": "cognitive_abstraction",
    "motivation_core": "motivation_autonomy",
    "regulation_core": "self_regulation_resilience",
}

LEGACY_AXIS_TO_ROLE = {
    "cognitive_abstraction": "cognitive_core",
    "motivation_autonomy": "motivation_core",
    "self_regulation_resilience": "regulation_core",
}


@dataclass(frozen=True)
class QuotaBucket:
    """A coarse population bucket used before continuous axis sampling."""

    label: str
    weight: float
    stage_options: tuple[str, ...]
    motivation_drives: tuple[str, ...]
    stress_band: str
    social_energy_band: str
    derived_performance_band: str


@dataclass(frozen=True)
class MegaPersonaSlot:
    """Generation target passed into the multi-agent MegaPersona pipeline."""

    slot_id: str
    quota_label: str
    target_axes: dict[str, float]
    constraints: dict[str, Any]
    adaptive_constraints: list[str] = field(default_factory=list)

    def axis_vector(self, axis_names: tuple[str, ...] = AXIS_NAMES) -> list[float]:
        return [self.target_axes[name] for name in axis_names]

    def prompt_context(self) -> dict[str, Any]:
        """Return a compact context suitable for prompt injection."""
        return {
            "slot_id": self.slot_id,
            "quota_label": self.quota_label,
            "target_axes": self.target_axes,
            "constraints": self.constraints,
            "adaptive_constraints": self.adaptive_constraints,
        }


DEFAULT_QUOTA_BUCKETS = (
    QuotaBucket(
        label="self_directed_builder",
        weight=0.18,
        stage_options=("high_school", "vocational", "undergraduate"),
        motivation_drives=("autonomy", "mastery", "curiosity"),
        stress_band="mid",
        social_energy_band="mid",
        derived_performance_band="mid",
    ),
    QuotaBucket(
        label="externally_driven_performer",
        weight=0.16,
        stage_options=("middle_school", "high_school", "undergraduate"),
        motivation_drives=("achievement", "recognition", "security"),
        stress_band="mid_high",
        social_energy_band="mid",
        derived_performance_band="high",
    ),
    QuotaBucket(
        label="anxious_high_effort",
        weight=0.16,
        stage_options=("middle_school", "high_school", "undergraduate"),
        motivation_drives=("security", "achievement", "avoidance"),
        stress_band="high",
        social_energy_band="low_mid",
        derived_performance_band="mid",
    ),
    QuotaBucket(
        label="curious_low_structure",
        weight=0.17,
        stage_options=("high_school", "undergraduate", "early_career"),
        motivation_drives=("curiosity", "autonomy", "mastery"),
        stress_band="mid",
        social_energy_band="mid_high",
        derived_performance_band="low",
    ),
    QuotaBucket(
        label="belonging_oriented_collaborator",
        weight=0.17,
        stage_options=("middle_school", "high_school", "vocational", "undergraduate"),
        motivation_drives=("belonging", "recognition", "autonomy"),
        stress_band="low_mid",
        social_energy_band="high",
        derived_performance_band="mid",
    ),
    QuotaBucket(
        label="reserved_resilient_observer",
        weight=0.16,
        stage_options=("high_school", "undergraduate", "graduate", "early_career"),
        motivation_drives=("mastery", "security", "curiosity"),
        stress_band="low_mid",
        social_energy_band="low",
        derived_performance_band="mid",
    ),
)


def default_schema_binding() -> dict[str, Any]:
    """Return the default schema-bound control space for genome evolution."""
    return {
        "axis_names": list(AXIS_NAMES),
        "axis_roles": dict(AXIS_ROLE_MAP),
        "quota_buckets": [
            {
                "label": bucket.label,
                "weight": bucket.weight,
                "stage_options": list(bucket.stage_options),
                "motivation_drives": list(bucket.motivation_drives),
                "stress_band": bucket.stress_band,
                "social_energy_band": bucket.social_energy_band,
                "derived_performance_band": bucket.derived_performance_band,
            }
            for bucket in DEFAULT_QUOTA_BUCKETS
        ],
    }


def axis_names_for_binding(binding: dict[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(binding, dict):
        return AXIS_NAMES
    axis_names = binding.get("axis_names")
    if not isinstance(axis_names, list) or not axis_names:
        return AXIS_NAMES
    cleaned = tuple(str(name) for name in axis_names if str(name).strip())
    return cleaned or AXIS_NAMES


def axis_roles_for_binding(binding: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(binding, dict):
        return dict(AXIS_ROLE_MAP)
    axis_names = set(axis_names_for_binding(binding))
    roles = dict(AXIS_ROLE_MAP)
    raw_roles = binding.get("axis_roles")
    if isinstance(raw_roles, dict):
        for role, axis_name in raw_roles.items():
            axis_name_str = str(axis_name)
            if axis_name_str in axis_names:
                roles[str(role)] = axis_name_str
    for legacy_axis, role in LEGACY_AXIS_TO_ROLE.items():
        if legacy_axis not in axis_names:
            continue
        roles.setdefault(role, legacy_axis)
    return roles


def axis_roles_for_target_axes(target_axes: dict[str, float] | None) -> dict[str, str]:
    """Infer axis-role bindings from a slot/item axis dictionary.

    This keeps downstream prompting and validation schema-aware even when only
    the renamed axis keys are available. Known legacy axes keep their canonical
    roles; otherwise we fall back to positional binding for the first three axes.
    """
    roles = dict(AXIS_ROLE_MAP)
    if not isinstance(target_axes, dict) or not target_axes:
        return roles

    axis_names = [str(name) for name in target_axes.keys() if str(name).strip()]
    for axis_name in axis_names:
        role = LEGACY_AXIS_TO_ROLE.get(axis_name)
        if role:
            roles[role] = axis_name

    fallback_roles = ("cognitive_core", "motivation_core", "regulation_core")
    for role, axis_name in zip(fallback_roles, axis_names):
        roles.setdefault(role, axis_name)
        if role not in roles or roles[role] not in axis_names:
            roles[role] = axis_name
    return roles


def quota_buckets_for_binding(binding: dict[str, Any] | None) -> tuple[QuotaBucket, ...]:
    if not isinstance(binding, dict):
        return DEFAULT_QUOTA_BUCKETS
    raw_buckets = binding.get("quota_buckets")
    if not isinstance(raw_buckets, list) or not raw_buckets:
        return DEFAULT_QUOTA_BUCKETS

    buckets: list[QuotaBucket] = []
    for raw in raw_buckets:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label", "")).strip()
        if not label:
            continue
        stage_options = tuple(str(item) for item in raw.get("stage_options", ()) if str(item).strip())
        motivation_drives = tuple(str(item) for item in raw.get("motivation_drives", ()) if str(item).strip())
        if not stage_options or not motivation_drives:
            continue
        try:
            weight = float(raw.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        buckets.append(
            QuotaBucket(
                label=label,
                weight=weight,
                stage_options=stage_options,
                motivation_drives=motivation_drives,
                stress_band=str(raw.get("stress_band", "mid")),
                social_energy_band=str(raw.get("social_energy_band", "mid")),
                derived_performance_band=str(raw.get("derived_performance_band", "mid")),
            )
        )
    return tuple(buckets) or DEFAULT_QUOTA_BUCKETS


def schema_binding_for_genome(genome: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(genome, dict):
        return default_schema_binding()
    binding = genome.get("schema_binding")
    if not isinstance(binding, dict):
        return default_schema_binding()
    return {
        "axis_names": list(axis_names_for_binding(binding)),
        "axis_roles": axis_roles_for_binding(binding),
        "quota_buckets": [
            {
                "label": bucket.label,
                "weight": bucket.weight,
                "stage_options": list(bucket.stage_options),
                "motivation_drives": list(bucket.motivation_drives),
                "stress_band": bucket.stress_band,
                "social_energy_band": bucket.social_energy_band,
                "derived_performance_band": bucket.derived_performance_band,
            }
            for bucket in quota_buckets_for_binding(binding)
        ],
    }


class SlotSampler:
    """Create quota-balanced targets with Sobol coverage inside the axis space."""

    def __init__(
        self,
        quota_buckets: tuple[QuotaBucket, ...] = DEFAULT_QUOTA_BUCKETS,
        axis_names: tuple[str, ...] = AXIS_NAMES,
    ):
        if not quota_buckets:
            raise ValueError("quota_buckets cannot be empty")
        if not axis_names:
            raise ValueError("axis_names cannot be empty")
        self.quota_buckets = quota_buckets
        self.axis_names = axis_names

    def sample(self, n: int, seed: int | None = None) -> list[MegaPersonaSlot]:
        if n <= 0:
            return []

        rng = np.random.default_rng(seed)
        points = self._sobol_points(n=n, dim=len(self.axis_names), seed=seed)
        bucket_indices = self._quota_indices(n, rng)

        slots: list[MegaPersonaSlot] = []
        for i, (point, bucket_idx) in enumerate(zip(points, bucket_indices), start=1):
            bucket = self.quota_buckets[bucket_idx]
            target_axes = {
                name: float(np.clip(value, 0.0, 1.0))
                for name, value in zip(self.axis_names, point)
            }
            constraints = self._constraints_for_bucket(bucket, rng)
            slots.append(
                MegaPersonaSlot(
                    slot_id=f"slot_{i:04d}",
                    quota_label=bucket.label,
                    target_axes=target_axes,
                    constraints=constraints,
                    adaptive_constraints=build_adaptive_constraints(target_axes, constraints),
                )
            )
        return slots

    def _quota_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        weights = np.array([bucket.weight for bucket in self.quota_buckets], dtype=float)
        weights = weights / weights.sum()
        raw_counts = weights * n
        counts = np.floor(raw_counts).astype(int)
        remainder = n - int(counts.sum())
        if remainder:
            order = np.argsort(raw_counts - counts)[::-1]
            counts[order[:remainder]] += 1

        indices = np.concatenate(
            [np.full(count, idx, dtype=int) for idx, count in enumerate(counts)]
        )
        rng.shuffle(indices)
        return indices

    @staticmethod
    def _sobol_points(n: int, dim: int, seed: int | None) -> np.ndarray:
        m = int(np.ceil(np.log2(n)))
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        return sampler.random_base2(m=m)[:n]

    @staticmethod
    def _constraints_for_bucket(
        bucket: QuotaBucket,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        return {
            "grade_or_stage": str(rng.choice(bucket.stage_options)),
            "primary_drive": str(rng.choice(bucket.motivation_drives)),
            "stress_band": bucket.stress_band,
            "social_energy_band": bucket.social_energy_band,
            "derived_performance_band": bucket.derived_performance_band,
            "avoid_all_high_profile": True,
            "avoid_all_low_profile": True,
        }


def build_adaptive_constraints(
    target_axes: dict[str, float],
    constraints: dict[str, Any],
    axis_roles: dict[str, str] | None = None,
) -> list[str]:
    """Translate target coordinates into natural-language consistency hints."""
    hints: list[str] = []
    roles = dict(AXIS_ROLE_MAP)
    if axis_roles:
        roles.update(axis_roles)

    abstraction_key = roles.get("cognitive_core", "cognitive_abstraction")
    autonomy_key = roles.get("motivation_core", "motivation_autonomy")
    regulation_key = roles.get("regulation_core", "self_regulation_resilience")

    abstraction = float(target_axes.get(abstraction_key, 0.5))
    autonomy = float(target_axes.get(autonomy_key, 0.5))
    regulation = float(target_axes.get(regulation_key, 0.5))

    if abstraction >= 0.75 and regulation <= 0.35:
        hints.append(
            "Explain why abstract thinking does not automatically become stable execution."
        )
    if autonomy >= 0.75 and constraints.get("primary_drive") in {"recognition", "security"}:
        hints.append(
            "Represent autonomy as a tension with external security or recognition needs."
        )
    if regulation >= 0.75 and constraints.get("stress_band") == "high":
        hints.append(
            "Ground high resilience in concrete protective factors, not generic toughness."
        )
    if regulation <= 0.25 and constraints.get("derived_performance_band") == "high":
        hints.append(
            "A high performance tendency with weak self-regulation needs external structure or support."
        )
    if abstraction <= 0.25 and autonomy >= 0.7:
        hints.append(
            "Show practical agency through concrete action rather than abstract self-description."
        )
    if constraints.get("social_energy_band") == "low" and autonomy <= 0.3:
        hints.append(
            "Avoid making reservedness identical to passivity; specify where agency still appears."
        )
    return hints
