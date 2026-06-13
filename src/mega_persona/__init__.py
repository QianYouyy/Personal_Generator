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
    compute_experiment_score,
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
    SCIENTIFIC_SCALE_REGISTRY,
    ShadowSurvey,
    ShadowSurveyItem,
    ShadowSurveySplit,
    build_initial_shadow_surveys,
    build_shadow_survey_splits,
    read_shadow_survey_splits,
    score_shadow_survey,
    shadow_survey_split_hashes,
    write_shadow_survey_splits,
)
from src.mega_persona.shadow_simulator import (
    LLMShadowSimulator,
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
from src.mega_persona.html_viz import generate_html_report
from src.mega_persona.template_generator import RuleBasedMegaPersonaBuilder
from src.mega_persona.validator import ValidationIssue, ValidationReport, validate_mega_persona
from src.mega_persona.visualization import visualize_result_path

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
    "LLMShadowSimulator",
    "MegaPersonaGenerator",
    "MegaPersona",
    "MegaPersonaSlot",
    "QuotaBucket",
    "RuleBasedMegaPersonaBuilder",
    "SCIENTIFIC_SCALE_REGISTRY",
    "ShadowBehaviorReport",
    "ShadowSurvey",
    "ShadowSurveyItem",
    "ShadowSurveySplit",
    "ShadowSurveySimulation",
    "SlotSampler",
    "aggregate_shadow_behavior",
    "ValidationIssue",
    "ValidationReport",
    "build_adaptive_constraints",
    "build_initial_shadow_surveys",
    "build_shadow_survey_splits",
    "read_shadow_survey_splits",
    "build_run_manifest",
    "candidate_slots",
    "compute_experiment_score",
    "default_genome",
    "evaluate_mega_personas",
    "generate_html_report",
    "mutate_genome",
    "prompt_addendum_from_genome",
    "parse_json_object",
    "personas_to_axis_matrix",
    "score_shadow_survey",
    "shadow_behavior_axis_matrix",
    "shadow_survey_split_hashes",
    "validate_mega_persona",
    "visualize_result_path",
    "write_shadow_survey_splits",
    "write_experiment_artifacts",
]
