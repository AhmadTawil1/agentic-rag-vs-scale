"""
Visualization and Analysis Script
Generates charts and qualitative analysis of the RAG systems comparison.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def load_results(results_path: str) -> list[dict]:
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_comparison_chart(comparison_csv_path: str, output_path: str) -> pd.DataFrame:
    """Bar chart + improvement percentage comparing both systems on RAGAS metrics."""
    df = pd.read_csv(comparison_csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(df["Metric"]))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, df["System_A"], width,
                    label="System A (GPT-4o)", color="#3498db", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, df["System_B"], width,
                    label="System B (Llama-3.1-8B)", color="#2ecc71", alpha=0.8)

    ax1.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score (0–1)", fontsize=12, fontweight="bold")
    ax1.set_title("RAG Systems Comparison: GPT-4o vs Llama-3.1-8B",
                  fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Metric"], rotation=15, ha="right")
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    df["Improvement"] = (df["System_B"] - df["System_A"]) / df["System_A"] * 100
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["Improvement"]]
    bars3 = ax2.barh(df["Metric"], df["Improvement"], color=colors, alpha=0.8)
    ax2.set_xlabel("Improvement (%)", fontsize=12, fontweight="bold")
    ax2.set_title("System B Improvement Over System A", fontsize=14, fontweight="bold")
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars3, df["Improvement"]):
        ax2.text(
            val + 1 if val > 0 else val - 1, bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            ha="left" if val > 0 else "right", va="center",
            fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Comparison chart saved to: {output_path}")
    return df


def create_retrieval_attempts_chart(results_b_path: str, output_path: str) -> None:
    """Distribution of System B retrieval attempts, broken down by question type."""
    results_b = load_results(results_b_path)

    attempts = [r.get("retrieval_attempts", 1) for r in results_b]
    q_types = [r.get("type", "unknown") for r in results_b]

    df = pd.DataFrame({"attempts": attempts, "type": q_types})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    counts = pd.Series(attempts).value_counts().sort_index()
    bars = ax1.bar(counts.index, counts.values,
                   color="#2ecc71", alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Retrieval Attempts", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Questions", fontsize=12, fontweight="bold")
    ax1.set_title("System B: Retrieval Attempts Distribution",
                  fontsize=13, fontweight="bold")
    ax1.set_xticks(sorted(counts.index))
    ax1.grid(axis="y", alpha=0.3)
    total = sum(counts.values)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h,
                 f"{int(h)}\n({h/total*100:.0f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    pivot = df.groupby(["type", "attempts"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", ax=ax2, colormap="viridis", alpha=0.85, edgecolor="black")
    ax2.set_xlabel("Question Type", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Questions", fontsize=12, fontweight="bold")
    ax2.set_title("Retrieval Attempts by Question Type",
                  fontsize=13, fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.legend(title="Attempts", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Retrieval attempts chart saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise RAG systems comparison")
    parser.add_argument("--comparison-csv", required=True,
                        help="Path to comparison.csv from evaluate_ragas.py")
    parser.add_argument("--system-a", required=True,
                        help="Path to System A results JSON")
    parser.add_argument("--system-b", required=True,
                        help="Path to System B results JSON")
    parser.add_argument("--output-dir", default="evaluation/results",
                        help="Directory for output charts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    create_comparison_chart(args.comparison_csv,
                            str(output_dir / "comparison_chart.png"))
    create_retrieval_attempts_chart(args.system_b,
                                    str(output_dir / "retrieval_attempts.png"))

    print("\nAll visualisations complete.")
