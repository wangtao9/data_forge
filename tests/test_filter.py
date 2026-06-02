from steps.filter_module import filter_by_score, compute_stats


class TestFilterByScore:
    def test_keeps_top_60_percent(self):
        data = [
            {"id": "1", "instruction": "a", "score": 4.0},
            {"id": "2", "instruction": "b", "score": 3.0},
            {"id": "3", "instruction": "c", "score": 2.0},
            {"id": "4", "instruction": "d", "score": 1.0},
            {"id": "5", "instruction": "e", "score": 5.0},
        ]
        result = filter_by_score(data, keep_ratio=0.6)
        assert len(result) == 3
        assert result[0]["id"] == "5"  # highest score first
        assert result[1]["id"] == "1"
        assert result[2]["id"] == "2"

    def test_keeps_all_if_small_dataset(self):
        data = [
            {"id": "1", "instruction": "a", "score": 4.0},
        ]
        result = filter_by_score(data, keep_ratio=0.6)
        assert len(result) == 1

    def test_empty_input(self):
        result = filter_by_score([], keep_ratio=0.6)
        assert result == []


class TestComputeStats:
    def test_stats(self):
        data = [
            {"id": "1", "score": 4.0},
            {"id": "2", "score": 3.0},
            {"id": "3", "score": 2.0},
        ]
        stats = compute_stats(data)
        assert stats["count"] == 3
        assert stats["min"] == 2.0
        assert stats["max"] == 4.0
        assert stats["mean"] == 3.0
