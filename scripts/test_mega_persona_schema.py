"""Smoke tests for the MegaPersona schema and validator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona import MegaPersona, validate_mega_persona
from src.mega_persona.prompts import cognition_motivation_agent_prompt


def sample_persona() -> dict:
    return {
        "persona_id": "mp_001",
        "demographics": {
            "age": 16,
            "grade_or_stage": "high_school",
            "region_type": "suburban",
            "family_context": (
                "Lives with a parent who works long shifts and an older cousin who helps with routines. "
                "The home is stable but time, quiet space, and adult academic guidance are uneven."
            ),
        },
        "cognitive_motivation_profile": {
            "thinking_style": {
                "dominant_mode": "practical",
                "abstraction_level": 0.42,
                "ambiguity_tolerance": 0.63,
                "evidence_preference": "experience",
                "typical_blind_spot": "She can dismiss abstract explanations before checking whether they would help her later.",
            },
            "motivation_system": {
                "primary_drive": "autonomy",
                "secondary_drive": "being useful to people she respects",
                "intrinsic_motivation": 0.68,
                "external_pressure_sensitivity": 0.36,
                "failure_sensitivity": 0.48,
                "reward_preference": "usefulness",
            },
            "learning_orientation": {
                "goal_orientation": "mastery",
                "preferred_learning_mode": "practice",
                "attention_pattern": "bursty",
                "curiosity_scope": "situational",
                "help_seeking_style": "peer_first",
            },
            "self_regulation": {
                "planning_style": "deadline_driven",
                "persistence": 0.66,
                "emotional_regulation": 0.58,
                "metacognition": 0.52,
                "habit_stability": 0.43,
            },
            "challenge_response": {
                "under_difficulty": "seeks_help",
                "under_criticism": "curious",
                "under_success": "exploratory",
            },
            "decision_pattern": {
                "risk_appetite": 0.46,
                "time_horizon": "medium",
                "tradeoff_style": "seek_growth",
                "typical_rationale": "She chooses the path that lets her test a skill quickly while keeping a way to recover.",
            },
            "narrative": (
                "She thinks through concrete examples first, then slowly backs into the principle behind them. "
                "A difficult problem becomes interesting when it connects to something useful: fixing a device, "
                "helping a friend, or proving she can learn without being micromanaged. Her motivation is strongest "
                "when she feels ownership over the task, weaker when adults frame it as obedience. Under pressure "
                "she works in bursts and may procrastinate until the deadline gives the task shape. When blocked, "
                "she usually asks a peer to show one worked example, then tries to rebuild the method herself. "
                "Her blind spot is that she can underestimate abstract planning and overtrust improvisation, even "
                "though she is capable of reflection after the fact."
            ),
        },
        "values_identity": {
            "core_values": ["autonomy", "usefulness", "loyalty"],
            "identity_anchor": "She sees herself as someone who learns best by making things work for real people.",
            "moral_tension": "She wants independence, but she also feels responsible for not disappointing people who rely on her.",
            "aspiration": "She wants to become the kind of person who can solve practical problems without asking permission first.",
        },
        "social_creative_profile": {
            "social_energy": 0.54,
            "collaboration_style": "supporter",
            "expressiveness": 0.47,
            "creative_mode": "practical",
            "peer_influence_sensitivity": 0.42,
            "narrative": (
                "Socially, she prefers small working groups where everyone has a visible role. She is not a loud "
                "leader, but she often notices the missing practical step and quietly makes the group more functional."
            ),
        },
        "mental_health_context": {
            "stress_load": 0.57,
            "resilience": 0.64,
            "coping_style": "problem_solving",
            "protective_factors": ["trusted cousin", "hands-on competence", "one close peer"],
            "risk_factors": ["limited quiet study space", "deadline pressure"],
            "narrative": (
                "Stress shows up as irritability and avoidance before big tasks, but she recovers once the work is "
                "made concrete. She benefits from people who help her break vague demands into visible next steps."
            ),
        },
        "derived_academic_tendency": {
            "likely_performance_band": "mid",
            "reasoning": (
                "Her performance is likely uneven but functional because practical motivation, peer help, and deadline-driven "
                "persistence offset weaker planning habits and limited study resources."
            ),
        },
    }


def test_valid_sample():
    persona = MegaPersona.model_validate(sample_persona())
    axes = persona.primary_axes()
    assert set(axes) == {
        "cognitive_abstraction",
        "motivation_autonomy",
        "self_regulation_resilience",
    }
    assert all(0.0 <= value <= 1.0 for value in axes.values())
    report = validate_mega_persona(persona)
    assert report.schema_valid
    assert report.is_valid


def test_low_performance_cannot_be_fixed_deficit():
    data = sample_persona()
    data["derived_academic_tendency"] = {
        "likely_performance_band": "low",
        "reasoning": "She performs low because she is stupid and incapable rather than because of context or strategy.",
    }
    report = validate_mega_persona(data)
    assert not report.is_valid
    assert any(issue.rule_id == "R8" for issue in report.issues)


def test_prompt_contains_new_module_name():
    prompt = cognition_motivation_agent_prompt("{}", '{"cognitive_abstraction": 0.4}')
    assert "cognitive_motivation_profile" in prompt
    assert "derived_academic_tendency" in prompt


def main():
    test_valid_sample()
    test_low_performance_cannot_be_fixed_deficit()
    test_prompt_contains_new_module_name()
    print("MegaPersona schema tests passed.")


if __name__ == "__main__":
    main()
