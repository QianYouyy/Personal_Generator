"""Smoke tests for OpenAI-compatible LLM client configuration."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_client import LLMClient


def test_deepseek_provider_configuration():
    old_key = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
    try:
        client = LLMClient.from_provider("deepseek", role="persona")
        assert client.model == "deepseek-v4-flash"
        assert client.base_url == "https://api.deepseek.com"
        assert client.api_key == "test-deepseek-key"
        assert client.api_key_env == "DEEPSEEK_API_KEY"
        assert client.timeout_seconds == 180.0

        simulator = LLMClient.from_provider(
            "deepseek",
            role="simulator",
            model="deepseek-v4-pro",
        )
        assert simulator.model == "deepseek-v4-pro"
        assert simulator.base_url == "https://api.deepseek.com"
    finally:
        if old_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_key


def test_deepseek_provider_requires_api_key():
    old_key = os.environ.get("DEEPSEEK_API_KEY")
    os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        try:
            LLMClient.from_provider("deepseek", role="persona")
        except ValueError as exc:
            assert "DeepSeek provider requires an API key" in str(exc)
        else:
            raise AssertionError("Expected missing DeepSeek API key to fail")
    finally:
        if old_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = old_key


def test_deepseek_provider_requires_base_url():
    old_key = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
    try:
        try:
            LLMClient.from_provider("deepseek", role="persona", base_url="")
        except ValueError as exc:
            assert "DeepSeek provider requires an explicit base URL" in str(exc)
        else:
            raise AssertionError("Expected missing DeepSeek base URL to fail")
    finally:
        if old_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_key


def main():
    test_deepseek_provider_configuration()
    test_deepseek_provider_requires_api_key()
    test_deepseek_provider_requires_base_url()
    print("LLM client tests passed.")


if __name__ == "__main__":
    main()
