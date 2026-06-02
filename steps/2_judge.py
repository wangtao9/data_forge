#!/usr/bin/env python3
"""Step 2: LLM-as-Judge — score synthetic instructions for quality."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from utils.io import load_config, read_jsonl, write_jsonl
from utils.llm import LLMClient, parse_json_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """你是一个指令质量评估专家。你需要对给定的指令从三个维度进行评分（1-5分）：

1. **清晰度 (clarity)**：指令是否清晰明确、无歧义？
   - 1分：非常模糊，无法理解意图
   - 2分：有些模糊，需要猜测
   - 3分：基本清楚，但有些细节缺失
   - 4分：清晰明确
   - 5分：非常精确，没有任何歧义

2. **可执行性 (feasibility)**：指令是否可以被合理地执行？
   - 1分：完全无法执行
   - 2分：大部分无法执行
   - 3分：勉强可以执行，但需要大量假设
   - 4分：可以执行，只需少量补充信息
   - 5分：完全可执行，信息充分

3. **复杂度 (complexity)**：指令是否需要多步推理或有一定深度？
   - 1分：过于简单，无需思考
   - 2分：简单直接
   - 3分：需要一定思考
   - 4分：需要多步推理
   - 5分：高度复杂，需要深度分析

请以JSON格式返回评分：
{"clarity": X, "feasibility": Y, "complexity": Z, "reason": "简要理由"}"""


def score_instruction(client: LLMClient, instruction: str, config: dict) -> dict | None:
    """Score a single instruction using LLM-as-Judge."""
    judge_config = config["judge"]
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"请评估以下指令：\n\n{instruction}"},
    ]

    response = client.chat(
        messages=messages,
        temperature=judge_config["temperature"],
        max_tokens=512,
    )

    parsed = parse_json_response(response)
    if parsed is None:
        logger.warning(f"Failed to parse judge response for instruction: {instruction[:50]}...")
        return None

    if not all(k in parsed for k in ("clarity", "feasibility", "complexity")):
        logger.warning(f"Missing scoring fields in response: {parsed}")
        return None

    # Validate score ranges
    for key in ("clarity", "feasibility", "complexity"):
        val = parsed[key]
        if not isinstance(val, (int, float)) or val < 1 or val > 5:
            logger.warning(f"Invalid score for {key}: {val}")
            return None

    return parsed


def compute_composite_score(scores: dict, weights: dict) -> float:
    """Compute weighted composite score."""
    return (
        scores["clarity"] * weights["clarity"]
        + scores["feasibility"] * weights["feasibility"]
        + scores["complexity"] * weights["complexity"]
    )


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge: score instructions for quality")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--input", default="output/synthetic.jsonl", help="Input JSONL path")
    parser.add_argument("--output", default="output/scored.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    config = load_config(args.config)
    instructions = read_jsonl(args.input)
    logger.info(f"Loaded {len(instructions)} instructions to score")

    llm_config = config["llm"]
    client = LLMClient(
        base_url=llm_config["base_url"],
        api_key_env=llm_config["api_key_env"],
        model=config["judge"].get("model", llm_config["model"]),
        max_retries=llm_config.get("max_retries", 3),
    )

    weights = config["judge"]["weights"]
    results = []
    skipped = 0

    for item in tqdm(instructions, desc="Scoring instructions"):
        scores = score_instruction(client, item["instruction"], config)
        if scores is None:
            skipped += 1
            continue

        composite = compute_composite_score(scores, weights)
        result = {**item, "scores": scores, "score": round(composite, 3)}
        results.append(result)

    logger.info(f"Scored {len(results)} instructions, skipped {skipped}")

    write_jsonl(results, Path(args.output))
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
