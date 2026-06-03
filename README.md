# Data Forge

合成数据 Pipeline：用开源 LLM 从种子指令生成高质量合成数据集。

## Pipeline 流程

1. **Self-Instruct** — 从 175 条种子指令生成 1000 条合成指令
2. **LLM-as-Judge** — 对合成指令进行质量评分（清晰度、可执行性、复杂度）
3. **Filter** — 保留评分 top 60% 的指令
4. **Diversity Analysis** — 对比种子/合成/过滤后数据的多样性

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API Key

```bash
export ZHIPUAI_API_KEY="your-api-key-here"
```

### 运行全流程

```bash
./run.sh
```

### 单步运行

```bash
# Step 1: 生成合成指令
python steps/1_self_instruct.py --num 1000

# Step 2: 质量评分
python steps/2_judge.py --input output/synthetic.jsonl

# Step 3: 过滤
python steps/3_filter.py --input output/scored.jsonl

# Step 4: 多样性分析
python steps/4_diversity.py --seeds seeds/default_seeds.jsonl
# Install & enable Chinese fonts if needed:
# sudo apt install fonts-wqy-microhei && rm -rf ~/.cache/matplotlib
```

## 配置

编辑 `config.yaml` 修改模型、温度、过滤比例等参数。

## 输出

| 文件 | 说明 |
|------|------|
| `output/synthetic.jsonl` | 生成的合成指令 |
| `output/scored.jsonl` | 带评分的指令 |
| `output/filtered.jsonl` | 过滤后的高质量指令 |
| `output/diversity_report.json` | 多样性分析报告 |
| `output/charts/` | 可视化图表 |

## 多样性指标

- **语义嵌入聚类**：使用 sentence-transformers 嵌入 + UMAP 降维 + KMeans 聚类，计算轮廓系数
- **N-gram 去重率**：计算 2-gram 和 3-gram 的重复比例，越低越多样
- **指令长度分布**：统计词数的均值、中位数、标准差

## 测试

```bash
python -m pytest tests/ -v
```
