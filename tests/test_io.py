import json
import tempfile
from pathlib import Path

from utils.io import read_jsonl, write_jsonl, load_config


class TestJsonlIO:
    def test_write_and_read_jsonl(self, tmp_path):
        data = [
            {"id": "1", "text": "hello"},
            {"id": "2", "text": "world"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(data, path)
        result = read_jsonl(path)
        assert result == data

    def test_read_jsonl_empty(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        result = read_jsonl(path)
        assert result == []

    def test_write_jsonl_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.jsonl"
        write_jsonl([{"a": 1}], path)
        assert path.exists()
        assert read_jsonl(path) == [{"a": 1}]


class TestLoadConfig:
    def test_load_config(self):
        config = load_config()
        assert "llm" in config
        assert "self_instruct" in config
        assert config["filter"]["keep_ratio"] == 0.6
