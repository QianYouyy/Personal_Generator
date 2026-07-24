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
        if "Create ONE complete MegaPersona JSON object" in prompt:
            return json.dumps(self.sample)
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
        if "Repair the output for stage `cognition_motivation`" in prompt:
            return json.dumps(
                {
                    "cognitive_motivation_profile": self.sample["cognitive_motivation_profile"],
                    "derived_academic_tendency": self.sample["derived_academic_tendency"],
                }
            )
        if "Repair the malformed JSON object" in prompt:
            return self._payload("demographics")
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


def test_compact_pipeline_generates_one_call_persona():
    slots = SlotSampler().sample(1, seed=31)
    llm = MockMegaPersonaLLM()
    generator = MegaPersonaGenerator(llm, pipeline_mode="single_call")
    result = generator.generate_one(slots[0])

    assert result.is_valid
    assert result.persona is not None
    assert "compact_persona" in result.raw_outputs
    assert len(llm.calls) == 1

    alias_llm = MockMegaPersonaLLM()
    alias_result = MegaPersonaGenerator(alias_llm, pipeline_mode="compact").generate_one(slots[0])
    assert alias_result.is_valid
    assert len(alias_llm.calls) == 1


def test_missing_stage_key_is_repaired():
    class MissingDerivedLLM(MockMegaPersonaLLM):
        def __init__(self):
            super().__init__()
            self.sent_missing = False

        def generate(self, prompt, system_prompt=None, **kwargs):
            if (
                "You are filling ONLY the `cognitive_motivation_profile`" in prompt
                and not self.sent_missing
            ):
                self.sent_missing = True
                self.calls.append(
                    {"prompt": prompt, "system_prompt": system_prompt, "kwargs": kwargs}
                )
                return json.dumps(
                    {
                        "cognitive_motivation_profile": self.sample[
                            "cognitive_motivation_profile"
                        ],
                    }
                )
            return super().generate(prompt, system_prompt=system_prompt, **kwargs)

    slots = SlotSampler().sample(1, seed=32)
    llm = MissingDerivedLLM()
    result = MegaPersonaGenerator(llm).generate_one(slots[0])

    assert result.is_valid
    assert "cognition_motivation_stage_repair" in result.raw_outputs


def test_malformed_agent_json_is_repaired():
    class MalformedOnceLLM(MockMegaPersonaLLM):
        def __init__(self):
            super().__init__()
            self.sent_bad_demographics = False

        def generate(self, prompt, system_prompt=None, **kwargs):
            if (
                "Create ONLY the `demographics` section" in prompt
                and not self.sent_bad_demographics
            ):
                self.sent_bad_demographics = True
                self.calls.append(
                    {"prompt": prompt, "system_prompt": system_prompt, "kwargs": kwargs}
                )
                return '{"demographics" {"age": 16}}'
            return super().generate(prompt, system_prompt=system_prompt, **kwargs)

    slots = SlotSampler().sample(1, seed=9)
    llm = MalformedOnceLLM()
    result = MegaPersonaGenerator(llm).generate_one(slots[0])

    assert result.is_valid
    assert any("Repair the malformed JSON object" in call["prompt"] for call in llm.calls)


def test_transient_agent_failure_is_retried():
    class TimeoutOnceLLM(MockMegaPersonaLLM):
        def __init__(self):
            super().__init__()
            self.failed_once = False

        def generate(self, prompt, system_prompt=None, **kwargs):
            if (
                "Create ONLY the `demographics` section" in prompt
                and not self.failed_once
            ):
                self.failed_once = True
                raise TimeoutError("Request timed out.")
            return super().generate(prompt, system_prompt=system_prompt, **kwargs)

    slots = SlotSampler().sample(1, seed=13)
    llm = TimeoutOnceLLM()
    result = MegaPersonaGenerator(llm).generate_one(slots[0])

    assert result.is_valid
    assert llm.failed_once


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


def test_overlong_fields_are_trimmed_before_validation():
    class OverlongNarrativeLLM(MockMegaPersonaLLM):
        def __init__(self):
            super().__init__()
            self.sample = sample_persona()
            self.sample["demographics"]["family_context"] = " ".join(
                ["The household rhythm is supportive but crowded and uneven."] * 30
            )
            self.sample["social_creative_profile"]["narrative"] = " ".join(
                ["She contributes by quietly improving group coordination and practical follow-through."] * 30
            )

    slots = SlotSampler().sample(1, seed=21)
    llm = OverlongNarrativeLLM()
    result = MegaPersonaGenerator(llm).generate_one(slots[0])

    assert result.is_valid
    assert result.persona is not None
    assert len(result.persona.demographics.family_context) <= 1000
    assert len(result.persona.social_creative_profile.narrative) <= 1500
    assert "revision_1" not in result.raw_outputs


def main():
    test_parse_json_object_from_fence()
    test_generate_one_valid_persona()
    test_compact_pipeline_generates_one_call_persona()
    test_missing_stage_key_is_repaired()
    test_malformed_agent_json_is_repaired()
    test_transient_agent_failure_is_retried()
    test_prompt_addendum_is_injected()
    test_generate_batch_and_evaluate()
    test_overlong_fields_are_trimmed_before_validation()
    print("MegaPersona generator tests passed.")


if __name__ == "__main__":
    main()
