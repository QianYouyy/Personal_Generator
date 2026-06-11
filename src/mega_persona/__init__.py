"""MegaPersona schema, sampling, survey, and validation utilities."""

from src.mega_persona.evaluation import (
    MegaPersonaEvaluation,
    evaluate_mega_personas,
    personas_to_axis_matrix,
)
from src.mega_persona.experiment import (
    MegaPersonaExperimentConfig,
    MegaPersonaExperimentRun,
    MegaPersonaExperimentRunner,
    MegaPersonaExperimentSummary,
    write_experiment_artifacts,
)
from src.mega_persona.evolution import (
    MegaEvolutionCandidate,
    MegaEvolutionConfig,
    MegaPersonaEvolver,
    build_run_manifest,
    candidate_slots,
    default_genome,
    mutate_genome,
    prompt_addendum_from_genome,
)
from src.mega_persona.generator import (
    AgentOutputError,
    MegaPersonaGenerationResult,
    MegaPersonaGenerator,
    parse_json_object,
)
from src.mega_persona.schema import MegaPersona
from src.mega_persona.shadow_survey import (
    ShadowSurvey,
    ShadowSurveyItem,
    build_initial_shadow_surveys,
    score_shadow_survey,
)
from src.mega_persona.shadow_simulator import (
    RuleBasedShadowSimulator,
    ShadowBehaviorReport,
    ShadowSurveySimulation,
    aggregate_shadow_behavior,
    shadow_behavior_axis_matrix,
)
from src.mega_persona.slots import (
    AXIS_NAMES,
    MegaPersonaSlot,
    QuotaBucket,
    SlotSampler,
    build_adaptive_constraints,
)
from src.mega_persona.template_generator import RuleBasedMegaPersonaBuilder
from src.mega_persona.validator import ValidationIssue, ValidationReport, validate_mega_persona

__all__ = [
    "AXIS_NAMES",
    "AgentOutputError",
    "MegaPersonaEvaluation",
    "MegaPersonaExperimentConfig",
    "MegaPersonaExperimentRun",
    "MegaPersonaExperimentRunner",
    "MegaPersonaExperimentSummary",
    "MegaEvolutionCandidate",
    "MegaEvolutionConfig",
    "MegaPersonaEvolver",
    "MegaPersonaGenerationResult",
    "MegaPersonaGenerator",
    "MegaPersona",
    "MegaPersonaSlot",
    "QuotaBucket",
    "RuleBasedMegaPersonaBuilder",
    "RuleBasedShadowSimulator",
    "ShadowBehaviorReport",
    "ShadowSurvey",
    "ShadowSurveyItem",
    "ShadowSurveySimulation",
    "SlotSampler",
    "aggregate_shadow_behavior",
    "ValidationIssue",
    "ValidationReport",
    "build_adaptive_constraints",
    "build_initial_shadow_surveys",
    "build_run_manifest",
    "candidate_slots",
    "default_genome",
    "evaluate_mega_personas",
    "mutate_genome",
    "prompt_addendum_from_genome",
    "parse_json_object",
    "personas_to_axis_matrix",
    "score_shadow_survey",
    "shadow_behavior_axis_matrix",
    "validate_mega_persona",
    "write_experiment_artifacts",
]
