# Synthetic Data Pipeline Design

## Overview

Build a 4-step pipeline that uses open-source LLMs to generate, score, filter, and analyze synthetic instruction data from 175 seed instructions.

## Architecture

Four independent Python scripts, each reading/writing JSONL, orchestrated by a shell script:

```
seeds/default_seeds.jsonl
  → steps/1_self_instruct.py  → output/synthetic.jsonl
  → steps/2_judge.py          → output/scored.jsonl
  → steps/3_filter.py         → output/filtered.jsonl
  → steps/4_diversity.py      → output/diversity_report.json + output/charts/
```

## Project Structure

```
data_forge/
├── README.md
├── requirements.txt
├── run.sh
├── config.yaml
├── seeds/
│   └── default_seeds.jsonl       # 175 built-in seed instructions
├── steps/
│   ├── 1_self_instruct.py
│   ├── 2_judge.py
│   ├── 3_filter.py
│   └── 4_diversity.py
├── utils/
│   ├── __init__.py
│   ├── llm.py                    # OpenAI-compatible API client wrapper
│   └── io.py                     # JSONL read/write utilities
└── output/                       # Runtime artifacts (gitignored)
    ├── synthetic.jsonl
    ├── scored.jsonl
    ├── filtered.jsonl
    ├── diversity_report.json
    └── charts/
```

## Step 1: Self-Instruct Generation

**Algorithm** (classic Self-Instruct):

1. Randomly sample K seeds (default K=6) as few-shot examples
2. Construct prompt: system instruction + K seed examples
3. Call LLM to generate new instructions
4. Dedup check: discard if ROUGE-L overlap > threshold with existing instructions
5. Repeat until 1000 valid instructions collected

**Config:**

```yaml
self_instruct:
  num_instructions: 1000
  few_shot_k: 6
  temperature: 0.7
  rouge_threshold: 0.7
  batch_size: 5
  model: "qwen2.5-72b-instruct"
```

**Output** (`synthetic.jsonl`):

```json
{"id": "syn_001", "instruction": "...", "source_seeds": ["seed_003", "seed_042"]}
```

**Dedup strategy**: ROUGE-L for fast approximate dedup (no embedding model needed).

## Step 2: LLM-as-Judge Scoring

**Scoring dimensions** (1-5 scale):

| Dimension | Description |
|-----------|-------------|
| Clarity | Is the instruction clear and unambiguous? |
| Feasibility | Can it be reasonably executed? |
| Complexity | Does it require multi-step reasoning or depth? |

**Method**: Each instruction scored individually by LLM with rubric prompt. LLM returns structured JSON:

```json
{"clarity": 4, "feasibility": 5, "complexity": 3, "reason": "..."}
```

**Composite score**: `score = clarity * 0.3 + feasibility * 0.3 + complexity * 0.4`

**Config:**

```yaml
judge:
  model: "qwen2.5-72b-instruct"
  temperature: 0.1
  weights:
    clarity: 0.3
    feasibility: 0.3
    complexity: 0.4
```

**Output** (`scored.jsonl`): Input fields + scoring fields appended.

## Step 3: Filtering

- Sort by `score` descending, keep top 60%
- Output `filtered.jsonl`
- Print filter statistics (original count, retained count, score distribution)

## Step 4: Diversity Analysis

Three groups compared: original seeds, synthetic data, filtered data.

### Metrics

| Metric | Method | Description |
|--------|--------|-------------|
| Semantic embedding clustering | sentence-transformers → UMAP → KMeans | Cluster count + silhouette coefficient |
| n-gram dedup rate | 2-gram / 3-gram overlap ratio | Lower = more diverse |
| Instruction length distribution | Character/word count stats | Mean, median, std dev |

### Visualization (`charts/`)

1. `embedding_clusters.png` — UMAP scatter plot, 3 groups in different colors
2. `ngram_dedup.png` — Bar chart comparing n-gram dedup rates
3. `length_distribution.png` — Histogram of instruction lengths

### Embedding model

`paraphrase-multilingual-MiniLM-L12-v2` — supports Chinese, lightweight, no GPU required.

### Report

`diversity_report.json` — all metric values for programmatic consumption.

## LLM Backend

OpenAI-compatible API (`/v1/chat/completions`). Config:

```yaml
llm:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "${DASHSCOPE_API_KEY}"   # from env var
  model: "qwen2.5-72b-instruct"
```

## Dependencies

```
openai>=1.0
rouge-score
sentence-transformers
umap-learn
scikit-learn
matplotlib
pyyaml
tqdm
numpy
```

## Error Handling

- API rate limits: exponential backoff with retry (max 3 attempts)
- Malformed LLM output: log warning, skip item, continue
- Missing API key: fail fast with clear error message
