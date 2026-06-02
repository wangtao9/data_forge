#!/usr/bin/env bash
set -euo pipefail

echo "=== Synthetic Data Pipeline ==="
echo ""

# Check API key
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "Error: DASHSCOPE_API_KEY environment variable is not set"
    exit 1
fi

CONFIG="${CONFIG:-config.yaml}"
SEEDS="${SEEDS:-seeds/default_seeds.jsonl}"

echo "[Step 1/4] Self-Instruct: generating synthetic instructions..."
python steps/1_self_instruct.py --config "$CONFIG" --seeds "$SEEDS" --output output/synthetic.jsonl
echo ""

echo "[Step 2/4] LLM-as-Judge: scoring instructions..."
python steps/2_judge.py --config "$CONFIG" --input output/synthetic.jsonl --output output/scored.jsonl
echo ""

echo "[Step 3/4] Filter: keeping top 60%..."
python steps/3_filter.py --config "$CONFIG" --input output/scored.jsonl --output output/filtered.jsonl
echo ""

echo "[Step 4/4] Diversity Analysis: comparing seed vs synthetic vs filtered..."
python steps/4_diversity.py --config "$CONFIG" --seeds "$SEEDS" --synthetic output/synthetic.jsonl --filtered output/filtered.jsonl --output-dir output
echo ""

echo "=== Pipeline Complete ==="
echo "Results:"
echo "  Synthetic:   output/synthetic.jsonl"
echo "  Scored:      output/scored.jsonl"
echo "  Filtered:    output/filtered.jsonl"
echo "  Report:      output/diversity_report.json"
echo "  Charts:      output/charts/"
