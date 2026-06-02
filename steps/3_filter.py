#!/usr/bin/env python3
"""Step 3: Filter — keep top-scoring instructions."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import load_config, read_jsonl, write_jsonl
from steps.filter_module import filter_by_score, compute_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Filter: keep top-scoring instructions")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--input", default="output/scored.jsonl", help="Input JSONL path")
    parser.add_argument("--output", default="output/filtered.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    config = load_config(args.config)
    keep_ratio = config["filter"]["keep_ratio"]

    data = read_jsonl(args.input)
    logger.info(f"Loaded {len(data)} scored instructions")

    before_stats = compute_stats(data)
    filtered = filter_by_score(data, keep_ratio)
    after_stats = compute_stats(filtered)

    logger.info(f"Filter stats (keep top {keep_ratio*100:.0f}%):")
    logger.info(f"  Before: {before_stats['count']} items, "
                f"score range [{before_stats['min']:.2f}, {before_stats['max']:.2f}], "
                f"mean={before_stats['mean']:.2f}")
    logger.info(f"  After:  {after_stats['count']} items, "
                f"score range [{after_stats['min']:.2f}, {after_stats['max']:.2f}], "
                f"mean={after_stats['mean']:.2f}")

    write_jsonl(filtered, Path(args.output))
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
