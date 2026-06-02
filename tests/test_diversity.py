from steps.diversity_metrics import ngram_dedup_rate, length_stats


class TestNgramDedupRate:
    def test_no_duplicates(self):
        texts = ["hello world", "foo bar baz", "unique text here"]
        rate = ngram_dedup_rate(texts, n=2)
        assert rate == 0.0

    def test_all_duplicates(self):
        texts = ["hello world", "hello world"]
        rate = ngram_dedup_rate(texts, n=2)
        assert rate == 1.0

    def test_partial_duplicates(self):
        texts = ["hello world foo", "hello world bar"]
        rate = ngram_dedup_rate(texts, n=2)
        assert 0.0 < rate < 1.0

    def test_empty_input(self):
        rate = ngram_dedup_rate([], n=2)
        assert rate == 0.0


class TestLengthStats:
    def test_basic_stats(self):
        texts = ["hello", "hello world", "hello world foo"]
        stats = length_stats(texts)
        assert stats["count"] == 3
        assert stats["mean"] == 2.0  # word counts: 1, 2, 3
        assert stats["min"] == 1
        assert stats["max"] == 3

    def test_empty_input(self):
        stats = length_stats([])
        assert stats["count"] == 0
