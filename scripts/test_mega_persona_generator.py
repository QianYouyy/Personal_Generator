"""Smoke tests for the fixed MegaPersona multi-agent generator."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_mega_persona_schema import sample_persona
from src.mega_persona import (
    MegaPersonaGenerator,
    SlotSampler,
    evaluate_mega_personas,
    parse_json_object,
)


class MockMegaPersonaLLM:
    def __init__(self):
        self.calls: list[dict] = []
        self.sample = sample_persona()

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        if "Create ONLY the `demographics` section" in prompt:
            return self._payload("demographics")
        if "Create ONLY the `values_identity` section" in prompt:
            return self._payload("values_identity")
        if "Create ONLY the `social_creative_profile` section" in prompt:
            return self._payload("social_creative_profile")
        if "Create ONLY the `mental_health_context` section" in prompt:
            return self._payload("mental_health_context")
        if "You are filling ONLY the `cognitive_motivation_profile`" in prompt:
            return json.dumps(
                {
                    "cognitive_motivation_profile": self.sample["cognitive_motivation_profile"],
                    "derived_academic_tendency": self.sample["derived_academic_tendency"],
                }
            )
        if "Repair this MegaPersona JSON" in prompt:
            return json.dumps(self.sample)
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")

    def _payload(self, key: str) -> str:
        return "```json\n" + json.dumps({key: self.sample[key]}) + "\n```"


def test_parse_json_object_from_fence():
    parsed = parse_json_object('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}


def test_generate_one_valid_persona():
    slots = SlotSampler().sample(1, seed=3)
    llm = MockMegaPersonaLLM()
    generator = MegaPersonaGenerator(llm)
    result = generator.generate_one(slots[0])

    assert result.is_valid
    assert result.persona is not None
    assert result.persona.persona_id == slots[0].slot_id
    assert set(result.raw_outputs) == {
        "demographics",
        "cognition_motivation",
        "values_identity",
        "social_creative",
        "mental_health",
    }
    assert len(llm.calls) == 5


def test_prompt_addendum_is_injected():
    slots = SlotSampler().sample(1, seed=4)
    llm = MockMegaPersonaLLM()
    generator = MegaPersonaGenerator(
        llm,
        prompt_addendum="Favor behaviorally testable mechanisms.",
    )
    result = generator.generate_one(slots[0])

    assert result.is_valid
    assert "Favor behaviorally testable mechanisms." in llm.calls[0]["system_prompt"]
    assert "Evolved generation policy addendum" in llm.calls[0]["system_prompt"]


def test_generate_batch_and_evaluate():
    slots = SlotSampler().sample(3, seed=5)
    generator = MegaPersonaGenerator(MockMegaPersonaLLM())
    results = generator.generate_from_slots(slots)

    assert len(results) == 3
    assert all(result.is_valid for result in results)

    personas = [result.persona for result in results if result.persona is not None]
    report = evaluate_mega_personas(personas, duplicate_threshold=0.95)
    assert report.sample_size == 3
    assert report.validity_rate == 1.0


def main():
    test_parse_json_object_from_fence()
    test_generate_one_valid_persona()
    test_prompt_addendum_is_injected()
    test_generate_batch_and_evaluate()
    print("MegaPersona generator tests passed.")


if __name__ == "__main__":
    main()
