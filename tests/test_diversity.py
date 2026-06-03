from steps.diversity_metrics import ngram_dedup_rate, length_stats


class TestNgramDedupRate:
    def test_no_duplicates(self):
        texts = ["你好世界", "今天天气真好", "学习编程很有趣"]
        rate = ngram_dedup_rate(texts, n=2)
        assert rate == 0.0

    def test_all_duplicates(self):
        texts = ["你好世界", "你好世界"]
        rate = ngram_dedup_rate(texts, n=2)
        assert rate == 1.0

    def test_partial_duplicates(self):
        # "你好" 2-gram appears in both texts
        texts = ["你好世界", "你好朋友"]
        rate = ngram_dedup_rate(texts, n=2)
        assert 0.0 < rate < 1.0

    def test_empty_input(self):
        rate = ngram_dedup_rate([], n=2)
        assert rate == 0.0

    def test_short_text_skipped(self):
        # Single-char text can't produce 2-grams
        rate = ngram_dedup_rate(["好"], n=2)
        assert rate == 0.0


class TestLengthStats:
    def test_basic_stats(self):
        texts = ["你好", "你好世界", "你好世界编程"]
        stats = length_stats(texts)
        assert stats["count"] == 3
        assert stats["mean"] == 4.0  # char counts: 2, 4, 6
        assert stats["min"] == 2
        assert stats["max"] == 6

    def test_empty_input(self):
        stats = length_stats([])
        assert stats["count"] == 0
