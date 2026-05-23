"""
Publication-quality figures for the research paper.

Usage:
    python evaluation/create_professional_charts.py \
        --comparison-csv evaluation/results/comparison.csv \
        --output-dir evaluation/results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

COLOR_A = "#2E86AB"
COLOR_B = "#A23B72"


def make_comparison_chart(df: pd.DataFrame, output_path: str) -> None:
    """Side-by-side bar chart + absolute performance gap."""
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    metrics = df["Metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.35

    ax1 = fig.add_subplot(gs[0])
    bars1 = ax1.bar(x - width / 2, df["System_A"], width,
                    label="System A (GPT-4o)", color=COLOR_A,
                    alpha=0.85, edgecolor="black", linewidth=0.7)
    bars2 = ax1.bar(x + width / 2, df["System_B"], width,
                    label="System B (Llama-3.1-8B)", color=COLOR_B,
                    alpha=0.85, edgecolor="black", linewidth=0.7)

    ax1.set_ylabel("Score", fontweight="bold")
    ax1.set_xlabel("RAGAS Metrics", fontweight="bold")
    ax1.set_title("Performance Comparison: GPT-4o vs Llama-3.1-8B",
                  fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Faithfulness", "Answer\nRelevancy",
                          "Context\nPrecision", "Context\nRecall"])
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="black")
    ax1.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                     f"{h:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    df = df.copy()
    df["Gap_Pct"] = abs((df["System_B"] - df["System_A"]) / df["System_A"]) * 100
    df["Winner"] = df.apply(
        lambda r: "System A" if r["System_A"] > r["System_B"] else "System B", axis=1
    )
    colors = [COLOR_A if w == "System A" else COLOR_B for w in df["Winner"]]

    ax2 = fig.add_subplot(gs[1])
    bars = ax2.barh(metrics, df["Gap_Pct"], color=colors,
                    alpha=0.85, edgecolor="black", linewidth=0.7)
    ax2.set_xlabel("Performance Gap (%)", fontweight="bold")
    ax2.set_title("Absolute Performance Difference\n(Bar colour = winner)",
                  fontweight="bold", pad=15)
    ax2.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.set_axisbelow(True)

    for bar, val, winner in zip(bars, df["Gap_Pct"], df["Winner"]):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}% ({winner})",
                 ha="left", va="center", fontsize=9, fontweight="bold")

    a_wins = sum(1 for w in df["Winner"] if w == "System A")
    b_wins = sum(1 for w in df["Winner"] if w == "System B")
    textstr = f"Avg gap: {df['Gap_Pct'].mean():.1f}%\nSystem A: {a_wins}/4\nSystem B: {b_wins}/4"
    ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes, fontsize=9,
             va="bottom", ha="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3,
                       edgecolor="black", linewidth=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Comparison chart saved to: {output_path}")


def make_radar_chart(df: pd.DataFrame, output_path: str) -> None:
    """Radar/spider chart for multi-dimensional comparison."""
    categories = ["Faithfulness", "Answer\nRelevancy",
                  "Context\nPrecision", "Context\nRecall"]
    n = len(categories)
    angles = [i / n * 2 * np.pi for i in range(n)] + [0]

    values_a = df["System_A"].tolist() + [df["System_A"].iloc[0]]
    values_b = df["System_B"].tolist() + [df["System_B"].iloc[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.plot(angles, values_a, "o-", linewidth=2,
            label="System A (GPT-4o)", color=COLOR_A)
    ax.fill(angles, values_a, alpha=0.25, color=COLOR_A)
    ax.plot(angles, values_b, "o-", linewidth=2,
            label="System B (Llama-3.1-8B)", color=COLOR_B)
    ax.fill(angles, values_b, alpha=0.25, color=COLOR_B)

    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=9)
    ax.set_title("Multi-Dimensional Performance Comparison",
                 size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              framealpha=0.95, edgecolor="black")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Radar chart saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication-quality charts from RAGAS comparison CSV"
    )
    parser.add_argument("--comparison-csv", required=True,
                        help="Path to comparison.csv produced by evaluate_ragas.py")
    parser.add_argument("--output-dir", default="evaluation/results",
                        help="Directory for output figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.comparison_csv)
    make_comparison_chart(df, str(output_dir / "comparison_chart_professional.png"))
    make_radar_chart(df, str(output_dir / "radar_chart.png"))

    print("\nAll publication figures complete.")
