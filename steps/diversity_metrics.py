"""Diversity metrics for Step 4, extracted for testability."""

from collections import Counter
from statistics import mean, median, stdev


def _get_ngrams(text: str, n: int) -> list[str]:
    """Extract n-grams from text (word-level)."""
    words = text.split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def ngram_dedup_rate(texts: list[str], n: int = 2) -> float:
    """Calculate the n-gram deduplication rate.

    Returns the fraction of unique n-grams that appear more than once.
    0.0 = all n-grams are unique (high diversity)
    1.0 = all n-grams are duplicates (low diversity)
    """
    if not texts:
        return 0.0
    all_ngrams: list[str] = []
    for text in texts:
        all_ngrams.extend(_get_ngrams(text, n))
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    duplicated = sum(1 for c in counts.values() if c > 1)
    return duplicated / len(counts)


def length_stats(texts: list[str]) -> dict:
    """Compute word-count statistics for a list of texts."""
    if not texts:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
    lengths = [len(text.split()) for text in texts]
    return {
        "count": len(lengths),
        "mean": round(mean(lengths), 2),
        "median": round(median(lengths), 2),
        "min": min(lengths),
        "max": max(lengths),
        "std": round(stdev(lengths), 2) if len(lengths) > 1 else 0,
    }
