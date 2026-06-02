"""Filter logic for Step 3, extracted for testability."""

from statistics import mean


def filter_by_score(data: list[dict], keep_ratio: float = 0.6) -> list[dict]:
    """Sort by score descending and keep top keep_ratio."""
    if not data:
        return []
    sorted_data = sorted(data, key=lambda x: x["score"], reverse=True)
    keep_count = max(1, int(len(sorted_data) * keep_ratio))
    return sorted_data[:keep_count]


def compute_stats(data: list[dict]) -> dict:
    """Compute basic statistics for scored data."""
    if not data:
        return {"count": 0, "min": 0, "max": 0, "mean": 0}
    scores = [item["score"] for item in data]
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": round(mean(scores), 3),
    }
