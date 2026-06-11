"""Smoke tests for the MegaPersona experimental path."""

from copy import deepcopy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_mega_persona_schema import sample_persona
from src.mega_persona import (
    AXIS_NAMES,
    MegaPersona,
    RuleBasedMegaPersonaBuilder,
    RuleBasedShadowSimulator,
    SlotSampler,
    aggregate_shadow_behavior,
    build_initial_shadow_surveys,
    evaluate_mega_personas,
    personas_to_axis_matrix,
    score_shadow_survey,
)


def persona_variant(index: int, axes: dict[str, float]) -> dict:
    themes = [
        (
            "robotics workshop, repair logs, soldering practice, tool sharing",
            "prototype builder who learns through mechanical troubleshooting",
        ),
        (
            "debate club, public speaking, argument mapping, civic questions",
            "careful advocate who tests ideas through dialogue and rebuttal",
        ),
        (
            "family shop, inventory routines, customer stories, evening accounting",
            "practical organizer who notices patterns in everyday transactions",
        ),
        (
            "music production, rhythm sketches, remix folders, late-night editing",
            "sound experimenter who learns through layered iteration",
        ),
        (
            "community garden, weather notes, soil testing, neighbor coordination",
            "patient steward who connects growth with shared responsibility",
        ),
        (
            "game design, level balancing, player feedback, puzzle notebooks",
            "systems tinkerer who studies how rules shape behavior",
        ),
        (
            "caregiving schedule, medication reminders, hospital travel, quiet resilience",
            "reliable helper who builds competence through responsibility",
        ),
        (
            "language exchange, translation notes, cultural comparison, migration memory",
            "bridge builder who learns by moving between perspectives",
        ),
    ]
    setting, identity = themes[(index - 1) % len(themes)]
    data = deepcopy(sample_persona())
    data["persona_id"] = f"mp_variant_{index:03d}"
    data["demographics"]["family_context"] += f" A distinctive context involves {setting}."
    profile = data["cognitive_motivation_profile"]
    profile["thinking_style"]["abstraction_level"] = axes["cognitive_abstraction"]
    profile["motivation_system"]["intrinsic_motivation"] = axes["motivation_autonomy"]
    profile["motivation_system"]["external_pressure_sensitivity"] = (
        1.0 - axes["motivation_autonomy"]
    )
    regulation_value = axes["self_regulation_resilience"]
    profile["self_regulation"]["persistence"] = regulation_value
    profile["self_regulation"]["emotional_regulation"] = regulation_value
    profile["self_regulation"]["metacognition"] = regulation_value
    profile["self_regulation"]["habit_stability"] = regulation_value
    data["mental_health_context"]["resilience"] = regulation_value
    profile["narrative"] += (
        f" Variant {index} is shaped by {setting}. This makes the person a {identity}, "
        "with different examples, pressures, allies, and recovery routines."
    )
    data["values_identity"]["identity_anchor"] = (
        f"This persona sees themself as a {identity}."
    )
    data["social_creative_profile"]["narrative"] += (
        f" Social expression often appears through {setting}, giving this profile a distinct texture."
    )
    data["mental_health_context"]["narrative"] += (
        f" Coping is connected to the concrete rhythms of {setting}, rather than a generic routine."
    )
    return data


def test_slot_sampler():
    slots = SlotSampler().sample(16, seed=7)
    assert len(slots) == 16
    assert set(slots[0].target_axes) == set(AXIS_NAMES)
    assert all(0.0 <= value <= 1.0 for slot in slots for value in slot.target_axes.values())
    assert all(slot.constraints["avoid_all_high_profile"] for slot in slots)


def test_shadow_surveys():
    surveys = build_initial_shadow_surveys()
    assert len(surveys) == 12
    first = surveys[0]
    assert len(first.items) == 12
    responses = {item_id: 3 for item_id in first.item_ids()}
    scores = score_shadow_survey(first, responses)
    for axis_name in AXIS_NAMES:
        assert f"axis.{axis_name}" in scores
        assert 0.0 <= scores[f"axis.{axis_name}"] <= 1.0


def test_population_evaluation():
    slots = SlotSampler().sample(8, seed=11)
    personas = [
        persona_variant(index, slot.target_axes)
        for index, slot in enumerate(slots, start=1)
    ]
    matrix = personas_to_axis_matrix(personas)
    assert matrix.shape == (8, 3)

    report = evaluate_mega_personas(
        personas,
        coverage_radius=0.35,
        duplicate_threshold=0.95,
    )
    assert report.sample_size == 8
    assert report.valid_count == 8
    assert report.validity_rate == 1.0
    assert 0.0 <= report.near_duplicate_rate <= 1.0
    assert report.fitness > 0.0


def test_shadow_behavior_simulation():
    slots = SlotSampler().sample(5, seed=23)
    personas = [
        persona_variant(index, slot.target_axes)
        for index, slot in enumerate(slots, start=1)
    ]
    persona_objects = [
        MegaPersona.model_validate(persona)
        for persona in personas
    ]
    surveys = build_initial_shadow_surveys(num_surveys=3)
    simulations = RuleBasedShadowSimulator(seed=3).simulate_population(persona_objects, surveys)

    assert len(simulations) == 15
    assert all(1 <= response <= 5 for sim in simulations for response in sim.responses.values())

    report = aggregate_shadow_behavior(persona_objects, simulations)
    assert report.sample_size == 5
    assert report.survey_count == 3
    assert 0.0 <= report.overall_alignment <= 1.0
    for axis in AXIS_NAMES:
        assert axis in report.behavior_axis_mean
        assert axis in report.persona_behavior_mae


def test_rule_based_baseline_builder():
    slots = SlotSampler().sample(6, seed=31)
    personas = RuleBasedMegaPersonaBuilder().build_population(slots)
    assert len(personas) == 6
    assert all(persona.persona_id == slot.slot_id for persona, slot in zip(personas, slots))

    evaluation = evaluate_mega_personas(personas, duplicate_threshold=0.95)
    assert evaluation.validity_rate == 1.0

    surveys = build_initial_shadow_surveys(num_surveys=2)
    simulations = RuleBasedShadowSimulator(seed=9).simulate_population(personas, surveys)
    behavior = aggregate_shadow_behavior(personas, simulations)
    assert behavior.sample_size == 6
    assert behavior.survey_count == 2
    assert behavior.overall_alignment > 0.0


def main():
    test_slot_sampler()
    test_shadow_surveys()
    test_population_evaluation()
    test_shadow_behavior_simulation()
    test_rule_based_baseline_builder()
    print("MegaPersona experiment tests passed.")


if __name__ == "__main__":
    main()
