import json
import os

import pytest

from utils.llm import LLMClient, parse_json_response


class TestLLMClient:
    def test_client_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key_env="TEST_API_KEY",
            model="test-model",
        )
        assert client.api_key == "sk-test-123"

    def test_client_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="MISSING_KEY"):
            LLMClient(
                base_url="https://api.example.com/v1",
                api_key_env="MISSING_KEY",
                model="test-model",
            )

    def test_client_default_retry(self):
        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="test-model",
        )
        assert client.max_retries == 3


class TestParseJsonResponse:
    def test_parse_clean_json(self):
        text = '{"clarity": 4, "feasibility": 5}'
        result = parse_json_response(text)
        assert result == {"clarity": 4, "feasibility": 5}

    def test_parse_json_in_markdown_block(self):
        text = '```json\n{"clarity": 4}\n```'
        result = parse_json_response(text)
        assert result == {"clarity": 4}

    def test_parse_json_with_leading_text(self):
        text = 'Here is the result:\n{"clarity": 4}'
        result = parse_json_response(text)
        assert result == {"clarity": 4}

    def test_parse_invalid_json_returns_none(self):
        text = "not json at all"
        result = parse_json_response(text)
        assert result is None
