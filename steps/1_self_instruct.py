#!/usr/bin/env python3
"""Step 1: Self-Instruct — generate synthetic instructions from seed data."""

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rouge_score import rouge_scorer
from tqdm import tqdm

from utils.io import load_config, read_jsonl, write_jsonl
from utils.llm import LLMClient, parse_json_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个指令生成器。你的任务是根据给定的示例指令，生成新的、多样化的指令。

要求：
1. 生成的指令应该与示例不同，但保持相似的质量和复杂度
2. 指令应该涵盖不同的主题和任务类型
3. 每条指令应该是清晰、具体、可执行的
4. 不要生成与已有指令过于相似的指令
5. 每条指令用中文撰写

请生成 {batch_size} 条新的指令。以JSON数组格式输出，每条指令是一个字符串。
格式：["指令1", "指令2", ...]"""


def build_few_shot_prompt(seeds: list[dict], k: int) -> str:
    """Build the few-shot prompt from randomly sampled seeds."""
    sampled = random.sample(seeds, min(k, len(seeds)))
    examples = "\n".join(f"{i+1}. {s['instruction']}" for i, s in enumerate(sampled))
    return f"以下是示例指令：\n\n{examples}"


def is_duplicate(new_instruction: str, existing: list[str], scorer: rouge_scorer.RougeScorer, threshold: float) -> bool:
    """Check if new_instruction is too similar to any existing instruction."""
    for existing_instr in existing:
        score = scorer.score(existing_instr, new_instruction)["rougeL"].fmeasure
        if score > threshold:
            return True
    return False


def generate_batch(
    client: LLMClient,
    seeds: list[dict],
    existing_instructions: list[str],
    config: dict,
    scorer: rouge_scorer.RougeScorer,
) -> list[str]:
    """Generate a batch of new instructions, filtering duplicates."""
    si_config = config["self_instruct"]
    few_shot_prompt = build_few_shot_prompt(seeds, si_config["few_shot_k"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(batch_size=si_config["batch_size"])},
        {"role": "user", "content": few_shot_prompt},
    ]

    response = client.chat(
        messages=messages,
        temperature=si_config["temperature"],
        max_tokens=2048,
    )

    parsed = parse_json_response(response)
    if parsed is None or not isinstance(parsed, list):
        logger.warning("Failed to parse LLM response as JSON list, skipping batch")
        return []

    new_instructions = []
    for instr in parsed:
        if not isinstance(instr, str) or len(instr.strip()) < 5:
            continue
        instr = instr.strip()
        if is_duplicate(instr, existing_instructions + new_instructions, scorer, si_config["rouge_threshold"]):
            logger.debug(f"Duplicate filtered: {instr[:50]}...")
            continue
        new_instructions.append(instr)

    return new_instructions


def main():
    parser = argparse.ArgumentParser(description="Self-Instruct: generate synthetic instructions")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--seeds", default="seeds/default_seeds.jsonl", help="Path to seed instructions")
    parser.add_argument("--output", default="output/synthetic.jsonl", help="Output path")
    parser.add_argument("--num", type=int, default=None, help="Override number of instructions to generate")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num is not None:
        config["self_instruct"]["num_instructions"] = args.num

    seeds = read_jsonl(args.seeds)
    logger.info(f"Loaded {len(seeds)} seed instructions")

    llm_config = config["llm"]
    client = LLMClient(
        base_url=llm_config["base_url"],
        api_key_env=llm_config["api_key_env"],
        model=config["self_instruct"].get("model", llm_config["model"]),
        max_retries=llm_config.get("max_retries", 3),
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    target = config["self_instruct"]["num_instructions"]
    existing_instructions = [s["instruction"] for s in seeds]
    results = []
    attempt = 0
    max_attempts = target * 5  # safety limit

    pbar = tqdm(total=target, desc="Generating instructions")
    while len(results) < target and attempt < max_attempts:
        attempt += 1
        new_batch = generate_batch(client, seeds, existing_instructions, config, scorer)
        for instr in new_batch:
            if len(results) >= target:
                break
            idx = len(results) + 1
            source_seeds = random.sample([s["id"] for s in seeds], min(3, len(seeds)))
            results.append({
                "id": f"syn_{idx:03d}",
                "instruction": instr,
                "source_seeds": source_seeds,
            })
            existing_instructions.append(instr)
            pbar.update(1)

    pbar.close()
    logger.info(f"Generated {len(results)} instructions in {attempt} attempts")

    write_jsonl(results, Path(args.output))
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
