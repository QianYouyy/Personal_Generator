"""Smoke tests for the end-to-end MegaPersona experiment runner."""

import json
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona import (
    MegaPersonaExperimentConfig,
    MegaPersonaExperimentRunner,
    write_experiment_artifacts,
)


class _MockSimLLM:
    """Returns neutral Likert responses for smoke tests."""

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        import json
        lines = prompt.split("\n")
        ids = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"') and '":' in stripped:
                item_id = stripped.split('"')[1] if len(stripped.split('"')) > 1 else None
                if item_id:
                    ids.append(item_id)
        return json.dumps({iid: 3 for iid in ids} if ids else {"unknown": 3})


_MOCK_SIM = _MockSimLLM()


def test_runner_summary():
    config = MegaPersonaExperimentConfig(
        n=5,
        seeds=(17, 23),
        mode="mock",
        num_shadow_surveys=3,
        items_per_shadow_survey=8,
    )
    summary = MegaPersonaExperimentRunner(config, simulator_llm_client=_MOCK_SIM).run()
    assert len(summary.runs) == 2
    aggregate = summary.aggregate_metrics()
    assert aggregate["experiment_score.mean"] > 0.0
    assert aggregate["validity_rate.mean"] == 1.0

    markdown = summary.to_markdown()
    assert "MegaPersona Experiment Summary" in markdown
    assert "experiment_score" in markdown


def test_write_artifacts():
    config = MegaPersonaExperimentConfig(
        n=4,
        seeds=(31,),
        mode="mock",
        num_shadow_surveys=2,
        items_per_shadow_survey=8,
    )
    summary = MegaPersonaExperimentRunner(config, simulator_llm_client=_MOCK_SIM).run()
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, markdown_path = write_experiment_artifacts(
            summary,
            Path(tmpdir),
            include_personas=False,
        )
        assert json_path.exists()
        assert markdown_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["aggregate"]["experiment_score.mean"] > 0.0
        assert "personas" not in data["runs"][0]


def main():
    test_runner_summary()
    test_write_artifacts()
    print("MegaPersona runner tests passed.")


if __name__ == "__main__":
    main()
