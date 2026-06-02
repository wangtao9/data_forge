# Synthetic Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-step synthetic data pipeline that generates 1000 instructions from 175 seeds via Self-Instruct, scores them with LLM-as-Judge, filters to top 60%, and analyzes diversity.

**Architecture:** Four independent Python scripts in `steps/`, each reading/writing JSONL, orchestrated by `run.sh`. Shared utilities in `utils/` for LLM calls and JSONL I/O. Config in `config.yaml`.

**Tech Stack:** Python 3.10+, OpenAI Python SDK, rouge-score, sentence-transformers, umap-learn, scikit-learn, matplotlib, pyyaml, tqdm, numpy

---

## File Structure

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Python dependencies |
| `config.yaml` | All pipeline parameters |
| `.gitignore` | Ignore output/, __pycache__, .env |
| `seeds/default_seeds.jsonl` | 175 built-in seed instructions |
| `utils/__init__.py` | Package init |
| `utils/io.py` | JSONL read/write, config loading |
| `utils/llm.py` | OpenAI-compatible API client with retry |
| `steps/1_self_instruct.py` | Self-Instruct generation loop |
| `steps/2_judge.py` | LLM-as-Judge scoring |
| `steps/3_filter.py` | Top-60% filtering |
| `steps/4_diversity.py` | Diversity metrics + charts |
| `run.sh` | Orchestration script |
| `README.md` | Usage documentation |
| `tests/__init__.py` | Test package init |
| `tests/test_io.py` | Tests for utils/io.py |
| `tests/test_llm.py` | Tests for utils/llm.py |
| `tests/test_filter.py` | Tests for step 3 |
| `tests/test_diversity.py` | Tests for step 4 metrics |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `.gitignore`
- Create: `utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `output/.gitkeep`
- Create: `output/charts/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/wt/share/python/data_forge
mkdir -p utils tests steps seeds output/charts
touch utils/__init__.py tests/__init__.py output/.gitkeep output/charts/.gitkeep
```

- [ ] **Step 2: Create requirements.txt**

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
pytest
```

- [ ] **Step 3: Create config.yaml**

```yaml
llm:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key_env: "DASHSCOPE_API_KEY"
  model: "qwen2.5-72b-instruct"
  max_retries: 3

self_instruct:
  num_instructions: 1000
  few_shot_k: 6
  temperature: 0.7
  rouge_threshold: 0.7
  batch_size: 5
  model: "qwen2.5-72b-instruct"

judge:
  model: "qwen2.5-72b-instruct"
  temperature: 0.1
  weights:
    clarity: 0.3
    feasibility: 0.3
    complexity: 0.4

filter:
  keep_ratio: 0.6

diversity:
  embedding_model: "paraphrase-multilingual-MiniLM-L12-v2"
  n_clusters: 10
  umap_neighbors: 15
  umap_min_dist: 0.1
```

- [ ] **Step 4: Create .gitignore**

```
output/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.pytest_cache/
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config.yaml .gitignore utils/__init__.py tests/__init__.py output/.gitkeep output/charts/.gitkeep
git commit -m "chore: project scaffolding with config and dependencies"
```

---

### Task 2: utils/io.py — JSONL I/O and Config Loading

**Files:**
- Create: `utils/io.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io.py
import json
import tempfile
from pathlib import Path

from utils.io import read_jsonl, write_jsonl, load_config


class TestJsonlIO:
    def test_write_and_read_jsonl(self, tmp_path):
        data = [
            {"id": "1", "text": "hello"},
            {"id": "2", "text": "world"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(data, path)
        result = read_jsonl(path)
        assert result == data

    def test_read_jsonl_empty(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        result = read_jsonl(path)
        assert result == []

    def test_write_jsonl_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.jsonl"
        write_jsonl([{"a": 1}], path)
        assert path.exists()
        assert read_jsonl(path) == [{"a": 1}]


class TestLoadConfig:
    def test_load_config(self):
        config = load_config()
        assert "llm" in config
        assert "self_instruct" in config
        assert config["filter"]["keep_ratio"] == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/test_io.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.io'`

- [ ] **Step 3: Write implementation**

```python
# utils/io.py
import json
from pathlib import Path

import yaml


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(data: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/test_io.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/io.py tests/test_io.py
git commit -m "feat: add JSONL I/O utilities and config loading"
```

---

### Task 3: utils/llm.py — OpenAI-Compatible API Client

**Files:**
- Create: `utils/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import json
import os

import pytest

from utils.llm import LLMClient, parse_json_response


class TestLLMClient:
    def test_client_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key_env="TEST_API_KEY",
            model="test-model",
        )
        assert client.api_key == "sk-test-123"

    def test_client_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="MISSING_KEY"):
            LLMClient(
                base_url="https://api.example.com/v1",
                api_key_env="MISSING_KEY",
                model="test-model",
            )

    def test_client_default_retry(self):
        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="test-model",
        )
        assert client.max_retries == 3


class TestParseJsonResponse:
    def test_parse_clean_json(self):
        text = '{"clarity": 4, "feasibility": 5}'
        result = parse_json_response(text)
        assert result == {"clarity": 4, "feasibility": 5}

    def test_parse_json_in_markdown_block(self):
        text = '```json\n{"clarity": 4}\n```'
        result = parse_json_response(text)
        assert result == {"clarity": 4}

    def test_parse_json_with_leading_text(self):
        text = 'Here is the result:\n{"clarity": 4}'
        result = parse_json_response(text)
        assert result == {"clarity": 4}

    def test_parse_invalid_json_returns_none(self):
        text = "not json at all"
        result = parse_json_response(text)
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/test_llm.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.llm'`

- [ ] **Step 3: Write implementation**

```python
# utils/llm.py
import json
import os
import re
import time
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible API client with retry logic."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str = "DASHSCOPE_API_KEY",
        api_key: str | None = None,
        model: str = "qwen2.5-72b-instruct",
        max_retries: int = 3,
    ):
        if api_key is None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise EnvironmentError(
                    f"API key not found. Set the {api_key_env} environment variable."
                )
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {e}")
                    raise


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown blocks and leading text."""
    # Try parsing directly
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/test_llm.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/llm.py tests/test_llm.py
git commit -m "feat: add LLM client with retry and JSON response parsing"
```

---

### Task 4: Seed Data — 175 Built-in Instructions

**Files:**
- Create: `seeds/default_seeds.jsonl`

- [ ] **Step 1: Create the 175 seed instructions**

Each line is a JSON object with `id` and `instruction`. Seeds span 12 categories to ensure diversity.

Write `seeds/default_seeds.jsonl` with the following 175 instructions:

```
{"id": "seed_001", "instruction": "写一篇关于人工智能在医疗领域应用的科普文章"}
{"id": "seed_002", "instruction": "用Python实现一个二叉搜索树，包含插入、查找和删除操作"}
{"id": "seed_003", "instruction": "分析以下销售数据，找出增长最快的季度并说明原因"}
{"id": "seed_004", "instruction": "解释量子计算的基本原理，用通俗的语言描述"}
{"id": "seed_005", "instruction": "如果所有的猫都是动物，而有些动物是黑色的，能否推出有些猫是黑色的？请解释"}
{"id": "seed_006", "instruction": "将以下英文段落翻译成中文，保持专业术语的准确性"}
{"id": "seed_007", "instruction": "总结这篇论文的核心观点和主要贡献"}
{"id": "seed_008", "instruction": "将以下口语化的段落改写为正式的学术语言"}
{"id": "seed_009", "instruction": "制定一个为期三个月的健身计划，适合办公室白领"}
{"id": "seed_010", "instruction": "计算一个复利问题：本金10万，年利率5%，按月复利，10年后是多少"}
{"id": "seed_011", "instruction": "设计一个手机App的首页布局，适合老年人使用"}
{"id": "seed_012", "instruction": "写一首关于秋天的现代诗，不超过20行"}
{"id": "seed_013", "instruction": "用JavaScript实现一个防抖函数，并解释其工作原理"}
{"id": "seed_014", "instruction": "对以下用户反馈进行情感分析，分类为正面、负面或中性"}
{"id": "seed_015", "instruction": "什么是区块链技术？它和传统数据库有什么区别？"}
{"id": "seed_016", "instruction": "如果A比B高，B比C高，那么A和C谁高？这是什么类型的推理？"}
{"id": "seed_017", "instruction": "将以下中文技术文档翻译成英文，注意术语一致性"}
{"id": "seed_018", "instruction": "用三句话概括《三体》第一部的主要剧情"}
{"id": "seed_019", "instruction": "将以下长句拆分为几个短句，保持原意不变"}
{"id": "seed_020", "instruction": "为一家新开的咖啡店制定社交媒体营销策略"}
{"id": "seed_021", "instruction": "解方程 2x² + 5x - 3 = 0，写出详细步骤"}
{"id": "seed_022", "instruction": "设计一个电商网站的购物车交互流程"}
{"id": "seed_023", "instruction": "写一封商务邮件，邀请合作伙伴参加产品发布会"}
{"id": "seed_024", "instruction": "用Go语言实现一个简单的HTTP服务器，支持GET和POST请求"}
{"id": "seed_025", "instruction": "根据以下表格数据，计算各产品类别的同比增长率"}
{"id": "seed_026", "instruction": "解释深度学习中注意力机制的原理"}
{"id": "seed_027", "instruction": "判断以下论证是否有效：如果下雨地面会湿，地面湿了，所以下雨了"}
{"id": "seed_028", "instruction": "将以下日文邮件翻译成中文商务用语"}
{"id": "seed_029", "instruction": "提取以下文章中的关键论点和支撑论据"}
{"id": "seed_030", "instruction": "将以下被动语态的句子改写为主动语态"}
{"id": "seed_031", "instruction": "为一个小型团队制定远程办公管理规范"}
{"id": "seed_032", "instruction": "一个袋子里有3个红球和2个蓝球，随机取2个，都是红球的概率是多少"}
{"id": "seed_033", "instruction": "设计一个智能家居场景的语音交互对话流程"}
{"id": "seed_034", "instruction": "写一篇产品评测，对比两款主流无线耳机"}
{"id": "seed_035", "instruction": "用SQL查询出销售额前10的客户及其总消费金额"}
{"id": "seed_036", "instruction": "对以下客户评价数据进行聚类分析，识别主要问题类型"}
{"id": "seed_037", "instruction": "什么是微服务架构？它相比单体架构有哪些优缺点？"}
{"id": "seed_038", "instruction": "分析以下推理中的逻辑谬误：他每天都喝咖啡，他活了90岁，所以喝咖啡能长寿"}
{"id": "seed_039", "instruction": "将以下法语新闻标题翻译成中文"}
{"id": "seed_040", "instruction": "将这篇5000字的文章压缩为500字的摘要"}
{"id": "seed_041", "instruction": "将以下正式通知改写为更亲切友好的语气"}
{"id": "seed_042", "instruction": "制定一份家庭月度预算方案，收入2万元"}
{"id": "seed_043", "instruction": "证明根号2是无理数"}
{"id": "seed_044", "instruction": "设计一个儿童教育类App的奖励机制"}
{"id": "seed_045", "instruction": "写一个科幻短故事的开头，设定在2150年的火星基地"}
{"id": "seed_046", "instruction": "用Rust实现一个线程安全的计数器"}
{"id": "seed_047", "instruction": "根据以下时间序列数据，预测下个月的销售额"}
{"id": "seed_048", "instruction": "解释Docker容器和虚拟机的区别，各自适用场景是什么"}
{"id": "seed_049", "instruction": "以下三段论是否成立：所有鸟都会飞，企鹅是鸟，所以企鹅会飞"}
{"id": "seed_050", "instruction": "将以下德语技术手册的摘要翻译成中文"}
{"id": "seed_051", "instruction": "总结这个视频会议的讨论要点和行动项"}
{"id": "seed_052", "instruction": "将以下冗长的技术描述简化为非技术人员能理解的版本"}
{"id": "seed_053", "instruction": "为大学生制定一份考研复习时间表"}
{"id": "seed_054", "instruction": "用矩阵方法求解线性方程组 2x+y=5, x-3y=-8"}
{"id": "seed_055", "instruction": "设计一个在线教育平台的课程推荐算法思路"}
{"id": "seed_056", "instruction": "写一篇关于远程办公利弊的议论文"}
{"id": "seed_057", "instruction": "实现一个LRU缓存，支持get和put操作，时间复杂度O(1)"}
{"id": "seed_058", "instruction": "对以下电商评论数据做主题建模，提取主要话题"}
{"id": "seed_059", "instruction": "什么是梯度下降法？它在机器学习中如何使用？"}
{"id": "seed_060", "instruction": "分析以下论证的充分性和必要性：吸烟导致肺癌的证据"}
{"id": "seed_061", "instruction": "将以下韩语歌词翻译成中文，尽量保留韵律感"}
{"id": "seed_062", "instruction": "从以下研究论文中提取研究方法、样本量和主要发现"}
{"id": "seed_063", "instruction": "将以下技术文档中的长难句改写为简单易懂的表达"}
{"id": "seed_064", "instruction": "为一家初创公司制定三个月的产品路线图"}
{"id": "seed_065", "instruction": "使用贝叶斯定理计算：某疾病发病率0.1%，检测准确率95%，检测阳性时实际患病的概率"}
{"id": "seed_066", "instruction": "设计一个社区论坛的举报和审核机制"}
{"id": "seed_067", "instruction": "写一段对话，两个角色讨论是否应该禁止使用一次性塑料"}
{"id": "seed_068", "instruction": "用Python的pandas库对CSV数据进行清洗和预处理"}
{"id": "seed_069", "instruction": "根据以下用户行为日志，分析用户流失的关键节点"}
{"id": "seed_070", "instruction": "解释什么是RESTful API，并给出设计原则"}
{"id": "seed_071", "instruction": "评估以下因果推理的可靠性：冰激凌销量增加时溺水事故也增加"}
{"id": "seed_072", "instruction": "将以下西班牙语新闻翻译成中文"}
{"id": "seed_073", "instruction": "将这篇技术博客提炼为5个关键要点"}
{"id": "seed_074", "instruction": "将以下口语表达改写为书面语"}
{"id": "seed_075", "instruction": "为一场技术大会设计议程安排，为期两天"}
{"id": "seed_076", "instruction": "推导等比数列求和公式"}
{"id": "seed_077", "instruction": "设计一个密码管理器的核心功能列表"}
{"id": "seed_078", "instruction": "写一篇书评，评价最近读的一本非虚构类书籍"}
{"id": "seed_079", "instruction": "用TypeScript实现一个发布-订阅模式的事件系统"}
{"id": "seed_080", "instruction": "对以下A/B测试结果进行统计显著性分析"}
{"id": "seed_081", "instruction": "什么是函数式编程？它的核心概念有哪些？"}
{"id": "seed_082", "instruction": "判断以下归纳推理的强度：观察了100只天鹅都是白色的，所以所有天鹅都是白色的"}
{"id": "seed_083", "instruction": "将以下俄语技术规范翻译成中文"}
{"id": "seed_084", "instruction": "将这段1小时的播客内容整理为结构化笔记"}
{"id": "seed_085", "instruction": "将以下技术方案文档从第三人称改写为第一人称叙述"}
{"id": "seed_086", "instruction": "制定一份个人技能提升计划，目标是在一年内转型为数据工程师"}
{"id": "seed_087", "instruction": "用图论方法证明Königsberg七桥问题无解"}
{"id": "seed_088", "instruction": "设计一个外卖平台的骑手调度算法思路"}
{"id": "seed_089", "instruction": "写一封投诉信，反映网购商品质量问题"}
{"id": "seed_090", "instruction": "用Shell脚本实现日志文件的定时归档和清理"}
{"id": "seed_091", "instruction": "对以下问卷数据进行描述性统计分析"}
{"id": "seed_092", "instruction": "解释Kubernetes的核心概念：Pod、Service、Deployment"}
{"id": "seed_093", "instruction": "分析以下类比推理的有效性：医生之于医院如同教师之于学校"}
{"id": "seed_094", "instruction": "将以下葡萄牙语产品说明翻译成中文"}
{"id": "seed_095", "instruction": "将这篇长文的核心论点压缩为一条推文（280字以内）"}
{"id": "seed_096", "instruction": "将以下文言文翻译成现代白话文"}
{"id": "seed_097", "instruction": "为一家中型企业制定数据安全合规方案"}
{"id": "seed_098", "instruction": "证明斐波那契数列相邻两项互质"}
{"id": "seed_099", "instruction": "设计一个在线协作白板工具的核心交互"}
{"id": "seed_100", "instruction": "写一篇关于气候变化对农业影响的调研报告摘要"}
{"id": "seed_101", "instruction": "实现一个简单的正则表达式引擎，支持. * + ?"}
{"id": "seed_102", "instruction": "根据以下财务报表数据，计算流动比率和速动比率"}
{"id": "seed_103", "instruction": "解释Git的分支模型和常用合并策略"}
{"id": "seed_104", "instruction": "以下条件推理是否正确：如果温度降到0度以下水会结冰，现在水没有结冰，所以温度没有降到0度以下"}
{"id": "seed_105", "instruction": "将以下意大利语菜单翻译成中文"}
{"id": "seed_106", "instruction": "提取以下专利文档中的技术方案要点"}
{"id": "seed_107", "instruction": "将以下冗余的段落精简为不超过3句话"}
{"id": "seed_108", "instruction": "为一个开源项目制定贡献者指南"}
{"id": "seed_109", "instruction": "计算以下排列组合问题：10个人中选3人组成委员会，有多少种选法"}
{"id": "seed_110", "instruction": "设计一个健身App的每日打卡和社交功能"}
{"id": "seed_111", "instruction": "写一份项目复盘报告，总结上次迭代中的经验教训"}
{"id": "seed_112", "instruction": "用C++实现一个简单的内存池分配器"}
{"id": "seed_113", "instruction": "对以下A/B测试的转化率数据进行卡方检验"}
{"id": "seed_114", "instruction": "什么是事件驱动架构？它适用于哪些场景？"}
{"id": "seed_115", "instruction": "分析以下悖论：这句话是假的。该命题的真值是什么？"}
{"id": "seed_116", "instruction": "将以下阿拉伯语新闻摘要翻译成中文"}
{"id": "seed_117", "instruction": "将这篇访谈整理为Q&A格式，保留核心观点"}
{"id": "seed_118", "instruction": "将以下技术规范文档改写为用户友好的操作手册"}
{"id": "seed_119", "instruction": "制定一份新员工入职第一周的培训计划"}
{"id": "seed_120", "instruction": "用数学归纳法证明1+2+...+n=n(n+1)/2"}
{"id": "seed_121", "instruction": "设计一个内容平台的个性化推荐反馈机制"}
{"id": "seed_122", "instruction": "写一段产品发布会的开场白"}
{"id": "seed_123", "instruction": "用Python实现一个简单的Web爬虫，抓取网页标题和链接"}
{"id": "seed_124", "instruction": "根据以下用户画像数据，划分用户群体并描述各群体特征"}
{"id": "seed_125", "instruction": "解释TCP三次握手和四次挥手的过程"}
{"id": "seed_126", "instruction": "评估以下决策的质量：为了节省成本而取消所有安全培训"}
{"id": "seed_127", "instruction": "将以下泰语旅游指南翻译成中文"}
{"id": "seed_128", "instruction": "将这篇年度报告浓缩为执行摘要，不超过300字"}
{"id": "seed_129", "instruction": "将以下消极表达的反馈改写为建设性反馈"}
{"id": "seed_130", "instruction": "为一个10人团队制定敏捷开发Sprint计划模板"}
{"id": "seed_131", "instruction": "求解以下优化问题：在约束条件下求目标函数的最大值"}
{"id": "seed_132", "instruction": "设计一个在线考试的防作弊机制"}
{"id": "seed_133", "instruction": "写一篇关于数字货币利弊的分析文章"}
{"id": "seed_134", "instruction": "实现一个Trie树，支持插入、搜索和前缀匹配"}
{"id": "seed_135", "instruction": "对以下文本数据集进行去重和异常值检测"}
{"id": "seed_136", "instruction": "什么是设计模式中的观察者模式？给出一个实际应用示例"}
{"id": "seed_137", "instruction": "判断以下命题的逆否命题是否与原命题等价：如果x>5则x>3"}
{"id": "seed_138", "instruction": "将以下越南语商业邮件翻译成中文"}
{"id": "seed_139", "instruction": "将以下会议录音转写文本整理为会议纪要"}
{"id": "seed_140", "instruction": "将以下过于技术化的产品说明改写为面向普通消费者的版本"}
{"id": "seed_141", "instruction": "为一场线上活动制定应急预案"}
{"id": "seed_142", "instruction": "计算以下线性规划问题的最优解"}
{"id": "seed_143", "instruction": "设计一个打车软件的动态定价算法思路"}
{"id": "seed_144", "instruction": "写一封推荐信，推荐一位同事申请高级工程师职位"}
{"id": "seed_145", "instruction": "用Java实现一个生产者-消费者模式，使用BlockingQueue"}
{"id": "seed_146", "instruction": "对以下实验数据进行方差分析（ANOVA）"}
{"id": "seed_147", "instruction": "解释CI/CD的概念及其在现代软件开发中的作用"}
{"id": "seed_148", "instruction": "分析以下论证中是否包含滑坡谬误：如果允许学生用计算器，他们就不会心算，然后连基本算术都不会"}
{"id": "seed_149", "instruction": "将以下马来语产品介绍翻译成中文"}
{"id": "seed_150", "instruction": "将以下长篇技术规范总结为检查清单格式"}
{"id": "seed_151", "instruction": "将以下生硬的拒绝信改写为委婉得体的版本"}
{"id": "seed_152", "instruction": "制定一份小型团队的代码评审规范"}
{"id": "seed_153", "instruction": "证明log(ab) = log(a) + log(b)"}
{"id": "seed_154", "instruction": "设计一个智能家居场景的自动化规则引擎"}
{"id": "seed_155", "instruction": "写一篇关于远程教育发展趋势的评论"}
{"id": "seed_156", "instruction": "用Python实现一个简单的推荐系统，基于协同过滤"}
{"id": "seed_157", "instruction": "根据以下日志数据，分析系统性能瓶颈"}
{"id": "seed_158", "instruction": "解释什么是领域驱动设计（DDD），它的核心概念是什么"}
{"id": "seed_159", "instruction": "以下反证法是否正确：假设结论不成立，推导出矛盾，因此结论成立"}
{"id": "seed_160", "instruction": "将以下印尼语新闻翻译成中文"}
{"id": "seed_161", "instruction": "将以下冗长的技术讨论提炼为决策点和待办事项"}
{"id": "seed_162", "instruction": "将以下学术写作改写为博客风格"}
{"id": "seed_163", "instruction": "为一家SaaS公司制定客户成功团队的KPI体系"}
{"id": "seed_164", "instruction": "用欧拉方法数值求解dy/dx=x+y, y(0)=1在x=0.5处的近似值"}
{"id": "seed_165", "instruction": "设计一个社交平台的反垃圾信息策略"}
{"id": "seed_166", "instruction": "写一份技术方案的可行性分析报告"}
{"id": "seed_167", "instruction": "实现一个布隆过滤器，支持添加和查询操作"}
{"id": "seed_168", "instruction": "对以下多维数据集进行主成分分析（PCA）"}
{"id": "seed_169", "instruction": "什么是CAP定理？它对分布式系统设计有什么影响？"}
{"id": "seed_170", "instruction": "分析以下统计推断的可靠性：样本量30，p值0.04，结论说有显著差异"}
{"id": "seed_171", "instruction": "将以下波兰语产品手册翻译成中文"}
{"id": "seed_172", "instruction": "将这篇综述文章的各章节整理为思维导图格式"}
{"id": "seed_173", "instruction": "将以下过于简略的邮件扩写为更详细周到的版本"}
{"id": "seed_174", "instruction": "为技术团队制定一份On-Call值班轮转方案"}
{"id": "seed_175", "instruction": "用动态规划求解最长公共子序列问题"}
```

- [ ] **Step 2: Verify seed count**

```bash
cd /Users/wt/share/python/data_forge && wc -l seeds/default_seeds.jsonl
```

Expected: 175

- [ ] **Step 3: Commit**

```bash
git add seeds/default_seeds.jsonl
git commit -m "feat: add 175 built-in seed instructions across 12 categories"
```

---

### Task 5: Step 1 — Self-Instruct Generation

**Files:**
- Create: `steps/1_self_instruct.py`

- [ ] **Step 1: Write the Self-Instruct script**

```python
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
            # Track which seeds were used in the prompt (approximation: last sampled)
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
```

- [ ] **Step 2: Verify the script runs (dry test with --num 2)**

```bash
cd /Users/wt/share/python/data_forge && python steps/1_self_instruct.py --help
```

Expected: Shows argparse help text without errors.

- [ ] **Step 3: Commit**

```bash
git add steps/1_self_instruct.py
git commit -m "feat: add Self-Instruct generation step"
```

---

### Task 6: Step 2 — LLM-as-Judge Scoring

**Files:**
- Create: `steps/2_judge.py`

- [ ] **Step 1: Write the Judge script**

```python
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
```

- [ ] **Step 2: Verify the script runs (help check)**

```bash
cd /Users/wt/share/python/data_forge && python steps/2_judge.py --help
```

Expected: Shows argparse help text without errors.

- [ ] **Step 3: Commit**

```bash
git add steps/2_judge.py
git commit -m "feat: add LLM-as-Judge scoring step"
```

---

### Task 7: Step 3 — Filtering

**Files:**
- Create: `steps/3_filter.py`
- Create: `tests/test_filter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_filter.py
import json
from pathlib import Path

from steps.filter_module import filter_by_score, compute_stats


class TestFilterByScore:
    def test_keeps_top_60_percent(self):
        data = [
            {"id": "1", "instruction": "a", "score": 4.0},
            {"id": "2", "instruction": "b", "score": 3.0},
            {"id": "3", "instruction": "c", "score": 2.0},
            {"id": "4", "instruction": "d", "score": 1.0},
            {"id": "5", "instruction": "e", "score": 5.0},
        ]
        result = filter_by_score(data, keep_ratio=0.6)
        assert len(result) == 3
        assert result[0]["id"] == "5"  # highest score first
        assert result[1]["id"] == "1"
        assert result[2]["id"] == "2"

    def test_keeps_all_if_small_dataset(self):
        data = [
            {"id": "1", "instruction": "a", "score": 4.0},
        ]
        result = filter_by_score(data, keep_ratio=0.6)
        assert len(result) == 1

    def test_empty_input(self):
        result = filter_by_score([], keep_ratio=0.6)
        assert result == []


class TestComputeStats:
    def test_stats(self):
        data = [
            {"id": "1", "score": 4.0},
            {"id": "2", "score": 3.0},
            {"id": "3", "score": 2.0},
        ]
        stats = compute_stats(data)
        assert stats["count"] == 3
        assert stats["min"] == 2.0
        assert stats["max"] == 4.0
        assert stats["mean"] == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/test_filter.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'steps.filter_module'`

- [ ] **Step 3: Write the filter module**

```python
# steps/filter_module.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/test_filter.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Write the filter CLI script**

```python
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
```

- [ ] **Step 6: Verify the script runs (help check)**

```bash
cd /Users/wt/share/python/data_forge && python steps/3_filter.py --help
```

Expected: Shows argparse help text without errors.

- [ ] **Step 7: Commit**

```bash
git add steps/3_filter.py steps/filter_module.py tests/test_filter.py
git commit -m "feat: add filtering step with tests"
```

---

### Task 8: Step 4 — Diversity Analysis

**Files:**
- Create: `steps/4_diversity.py`
- Create: `steps/diversity_metrics.py`
- Create: `tests/test_diversity.py`

- [ ] **Step 1: Write the failing test for diversity metrics**

```python
# tests/test_diversity.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/test_diversity.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'steps.diversity_metrics'`

- [ ] **Step 3: Write the diversity metrics module**

```python
# steps/diversity_metrics.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/test_diversity.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Write the diversity analysis script**

```python
#!/usr/bin/env python3
"""Step 4: Diversity Analysis — compare seed, synthetic, and filtered data."""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import load_config, read_jsonl
from steps.diversity_metrics import ngram_dedup_rate, length_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    """Compute sentence embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def compute_clustering_metrics(embeddings: np.ndarray, n_clusters: int) -> dict:
    """Compute KMeans clustering and silhouette score."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, labels)
    return {"n_clusters": n_clusters, "silhouette_score": round(silhouette, 4), "labels": labels}


def plot_embedding_clusters(embeddings_dict: dict, labels_dict: dict, output_path: Path):
    """Plot UMAP-reduced embeddings for multiple groups."""
    from umap import UMAP

    all_embeddings = np.vstack(list(embeddings_dict.values()))
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    all_2d = reducer.fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"种子数据": "#2196F3", "合成数据": "#FF9800", "过滤后数据": "#4CAF50"}
    offset = 0
    for name, emb in embeddings_dict.items():
        count = len(emb)
        points = all_2d[offset : offset + count]
        ax.scatter(points[:, 0], points[:, 1], c=colors.get(name, "gray"), label=name, alpha=0.6, s=10)
        offset += count

    ax.set_title("语义嵌入聚类 (UMAP)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved embedding cluster plot to {output_path}")


def plot_ngram_dedup(rates_dict: dict, output_path: Path):
    """Plot n-gram dedup rate comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(rates_dict.keys())
    bigram_rates = [rates_dict[n]["bigram"] for n in names]
    trigram_rates = [rates_dict[n]["trigram"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, bigram_rates, width, label="2-gram", color="#2196F3")
    ax.bar(x + width / 2, trigram_rates, width, label="3-gram", color="#FF9800")

    ax.set_ylabel("去重率")
    ax.set_title("N-gram 去重率对比")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved n-gram dedup plot to {output_path}")


def plot_length_distribution(lengths_dict: dict, output_path: Path):
    """Plot instruction length distribution comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"种子数据": "#2196F3", "合成数据": "#FF9800", "过滤后数据": "#4CAF50"}

    for name, lengths in lengths_dict.items():
        ax.hist(lengths, bins=30, alpha=0.5, label=name, color=colors.get(name, "gray"), edgecolor="black")

    ax.set_xlabel("指令长度（词数）")
    ax.set_ylabel("频次")
    ax.set_title("指令长度分布对比")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved length distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diversity Analysis: compare seed, synthetic, and filtered data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--seeds", default="seeds/default_seeds.jsonl", help="Seed instructions path")
    parser.add_argument("--synthetic", default="output/synthetic.jsonl", help="Synthetic instructions path")
    parser.add_argument("--filtered", default="output/filtered.jsonl", help="Filtered instructions path")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    div_config = config["diversity"]
    output_dir = Path(args.output_dir)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    seeds = read_jsonl(args.seeds)
    synthetic = read_jsonl(args.synthetic)
    filtered = read_jsonl(args.filtered)

    seed_texts = [s["instruction"] for s in seeds]
    synthetic_texts = [s["instruction"] for s in synthetic]
    filtered_texts = [s["instruction"] for s in filtered]

    groups = {
        "种子数据": seed_texts,
        "合成数据": synthetic_texts,
        "过滤后数据": filtered_texts,
    }

    report = {}

    # 1. N-gram dedup rate
    logger.info("Computing n-gram dedup rates...")
    ngram_results = {}
    for name, texts in groups.items():
        ngram_results[name] = {
            "bigram": round(ngram_dedup_rate(texts, n=2), 4),
            "trigram": round(ngram_dedup_rate(texts, n=3), 4),
        }
    report["ngram_dedup"] = ngram_results
    plot_ngram_dedup(ngram_results, charts_dir / "ngram_dedup.png")

    # 2. Length distribution
    logger.info("Computing length statistics...")
    length_results = {}
    lengths_for_plot = {}
    for name, texts in groups.items():
        stats = length_stats(texts)
        length_results[name] = stats
        lengths_for_plot[name] = [len(t.split()) for t in texts]
    report["length_stats"] = length_results
    plot_length_distribution(lengths_for_plot, charts_dir / "length_distribution.png")

    # 3. Embedding clustering
    logger.info("Computing embeddings and clustering...")
    embedding_model = div_config["embedding_model"]
    n_clusters = div_config["n_clusters"]
    embeddings_dict = {}
    clustering_results = {}

    for name, texts in groups.items():
        emb = compute_embeddings(texts, embedding_model)
        embeddings_dict[name] = emb
        metrics = compute_clustering_metrics(emb, n_clusters)
        clustering_results[name] = {
            "n_clusters": metrics["n_clusters"],
            "silhouette_score": metrics["silhouette_score"],
        }

    report["clustering"] = clustering_results
    plot_embedding_clusters(embeddings_dict, clustering_results, charts_dir / "embedding_clusters.png")

    # Save report
    report_path = output_dir / "diversity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Diversity report saved to {report_path}")

    # Print summary
    logger.info("\n=== Diversity Report Summary ===")
    for name in groups:
        logger.info(f"\n{name}:")
        logger.info(f"  N-gram dedup: 2-gram={ngram_results[name]['bigram']:.4f}, "
                     f"3-gram={ngram_results[name]['trigram']:.4f}")
        logger.info(f"  Length: mean={length_results[name]['mean']:.1f}, "
                     f"median={length_results[name]['median']:.1f}, "
                     f"std={length_results[name]['std']:.1f}")
        logger.info(f"  Clustering: silhouette={clustering_results[name]['silhouette_score']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the script runs (help check)**

```bash
cd /Users/wt/share/python/data_forge && python steps/4_diversity.py --help
```

Expected: Shows argparse help text without errors.

- [ ] **Step 7: Commit**

```bash
git add steps/4_diversity.py steps/diversity_metrics.py tests/test_diversity.py
git commit -m "feat: add diversity analysis step with tests and visualizations"
```

---

### Task 9: Orchestration Script

**Files:**
- Create: `run.sh`

- [ ] **Step 1: Write run.sh**

```bash
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
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x run.sh
git add run.sh
git commit -m "feat: add pipeline orchestration script"
```

---

### Task 10: README and Final Integration

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
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
export DASHSCOPE_API_KEY="your-api-key-here"
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
```

- [ ] **Step 2: Run all tests**

```bash
cd /Users/wt/share/python/data_forge && python -m pytest tests/ -v
```

Expected: All tests pass (test_io, test_llm, test_filter, test_diversity)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## Self-Review

**1. Spec coverage:**
- Self-Instruct generation → Task 5 ✓
- LLM-as-Judge scoring → Task 6 ✓
- Filtering top 60% → Task 7 ✓
- Diversity analysis (embedding clustering, n-gram dedup, length distribution) → Task 8 ✓
- 175 seed instructions → Task 4 ✓
- config.yaml → Task 1 ✓
- run.sh → Task 9 ✓
- JSONL I/O → Task 2 ✓
- LLM client with retry → Task 3 ✓
- OpenAI-compatible API → Task 3 ✓
- Error handling (retry, malformed output, missing key) → Task 3 ✓

**2. Placeholder scan:** No TBD, TODO, or placeholder patterns found.

**3. Type consistency:** All function signatures and data formats are consistent across tasks. `filter_by_score` and `compute_stats` are defined in `steps/filter_module.py` and used in `steps/3_filter.py`. `ngram_dedup_rate` and `length_stats` are defined in `steps/diversity_metrics.py` and used in `steps/4_diversity.py`. JSONL format is consistent: `{id, instruction, source_seeds}` → `{..., scores, score}` → `{..., score}` (filtered).
