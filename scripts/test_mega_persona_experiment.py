"""Smoke tests for the MegaPersona experimental path."""

from copy import deepcopy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_mega_persona_schema import sample_persona
from src.mega_persona import (
    AXIS_NAMES,
    LLMShadowSimulator,
    MegaPersona,
    RuleBasedMegaPersonaBuilder,
    SlotSampler,
    aggregate_shadow_behavior,
    build_initial_shadow_surveys,
    build_shadow_survey_splits,
    evaluate_mega_personas,
    personas_to_axis_matrix,
    score_shadow_survey,
)


# ---------------------------------------------------------------------------
# Mock LLM client for smoke tests — returns neutral responses (3=neutral)
# so we can test the full pipeline without an actual API key.
# ---------------------------------------------------------------------------

class _MockLLMClient:
    """Returns a JSON blob of neutral responses for every simulate_persona call."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.0,
                 max_tokens: int = 1500) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        # We need to parse the prompt to figure out how many items to respond to,
        # then return {"item_id": 3, ...} for all of them.
        import re
        item_ids = re.findall(r'"([^"]+)":\s*"([^"]*)"', prompt.split("ITEMS:")[-1]) if "ITEMS:" in prompt else []
        ids = [m[0] for m in item_ids]
        responses = {iid: 3 for iid in ids}
        import json
        return json.dumps(responses)


_MOCK_LLM = _MockLLMClient()


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
    assert first.split == "train"
    assert all(item.scale_id for item in first.items)

    splits = build_shadow_survey_splits(
        train_surveys=2,
        validation_surveys=1,
        test_surveys=1,
        items_per_survey=6,
        seed=19,
    )
    assert len(splits.train) == 2
    assert len(splits.validation) == 1
    assert len(splits.test) == 1
    assert splits.train[0].survey_id.startswith("shadow_train")
    assert splits.validation[0].survey_id.startswith("shadow_validation")
    assert splits.test[0].survey_id.startswith("shadow_test")


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
    sim = LLMShadowSimulator(_MOCK_LLM)
    simulations = sim.simulate_population(persona_objects, surveys)

    assert len(simulations) == 15
    assert all(1 <= response <= 5 for sim_obj in simulations for response in sim_obj.responses.values())

    report = aggregate_shadow_behavior(persona_objects, simulations)
    assert report.sample_size == 5
    assert report.survey_count == 3
    assert 0.0 <= report.overall_alignment <= 1.0
    for axis in AXIS_NAMES:
        assert axis in report.behavior_axis_mean
        assert axis in report.persona_behavior_mae

    prompt = _MOCK_LLM.calls[-1]["prompt"]
    assert "Abstraction level (0=concrete" not in prompt
    assert "Intrinsic motivation (0-1)" not in prompt
    assert "Resilience:" not in prompt
    assert "DERIVED ACADEMIC TENDENCY" not in prompt


def test_shadow_simulator_malformed_json_fallback():
    class BadJsonLLM:
        def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.0,
                     max_tokens: int = 1500) -> str:
            import re
            ids = [
                match[0]
                for match in re.findall(
                    r'"([^"]+)":\s*"([^"]*)"',
                    prompt.split("ITEMS:")[-1],
                )
            ]
            if not ids:
                return "not json"
            return f'{{"{ids[0]}" 4, "{ids[1]}": 2}}'

    persona = MegaPersona.model_validate(sample_persona())
    survey = build_initial_shadow_surveys(num_surveys=1, items_per_survey=6)[0]
    simulation = LLMShadowSimulator(BadJsonLLM()).simulate_persona(persona, survey)

    assert set(simulation.responses) == set(survey.item_ids())
    assert simulation.responses[survey.item_ids()[0]] == 4
    assert simulation.responses[survey.item_ids()[1]] == 2
    assert all(1 <= value <= 5 for value in simulation.responses.values())


def test_shadow_simulator_retries_transient_failure():
    class TimeoutOnceLLM(_MockLLMClient):
        def __init__(self):
            super().__init__()
            self.failures = 0

        def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.0,
                     max_tokens: int = 1500) -> str:
            if self.failures == 0:
                self.failures += 1
                raise TimeoutError("Request timed out.")
            return super().generate(prompt, system_prompt, temperature, max_tokens)

    persona = MegaPersona.model_validate(sample_persona())
    survey = build_initial_shadow_surveys(num_surveys=1, items_per_survey=6)[0]
    llm = TimeoutOnceLLM()
    simulation = LLMShadowSimulator(
        llm,
        max_retries=1,
        retry_backoff_seconds=0.0,
    ).simulate_persona(persona, survey)

    assert llm.failures == 1
    assert set(simulation.responses) == set(survey.item_ids())


def test_rule_based_baseline_builder():
    slots = SlotSampler().sample(6, seed=31)
    personas = RuleBasedMegaPersonaBuilder().build_population(slots)
    assert len(personas) == 6
    assert all(persona.persona_id == slot.slot_id for persona, slot in zip(personas, slots))

    evaluation = evaluate_mega_personas(personas, duplicate_threshold=0.95)
    assert evaluation.validity_rate == 1.0

    surveys = build_initial_shadow_surveys(num_surveys=2)
    sim = LLMShadowSimulator(_MOCK_LLM)
    simulations = sim.simulate_population(personas, surveys)
    behavior = aggregate_shadow_behavior(personas, simulations)
    assert behavior.sample_size == 6
    assert behavior.survey_count == 2
    assert behavior.overall_alignment > 0.0


def main():
    test_slot_sampler()
    test_shadow_surveys()
    test_population_evaluation()
    test_shadow_behavior_simulation()
    test_shadow_simulator_malformed_json_fallback()
    test_shadow_simulator_retries_transient_failure()
    test_rule_based_baseline_builder()
    print("MegaPersona experiment tests passed.")


if __name__ == "__main__":
    main()
