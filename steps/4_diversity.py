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
import matplotlib.font_manager as fm

def _setup_cjk_font():
    """Pick the first available CJK font for chart labels."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                  "SimHei", "PingFang SC", "Heiti TC", "STHeiti",
                  "Arial Unicode MS", "Songti SC"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams.get("font.sans-serif", [])
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_cjk_font()
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
