"""
Visualization: generates publication-quality charts from tournament + RAGAS results.

Usage:
    python evaluation/visualize_results.py \
        --system-a  evaluation/results/system_a_results_TIMESTAMP.json \
        --system-b  evaluation/results/system_b_results_TIMESTAMP.json \
        --comparison-csv evaluation/results/comparison.csv \
        --output-dir evaluation/results
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})

COLOR_A = "#2c7bb6"   # blue  — System A (Goliath)
COLOR_B = "#1a9641"   # green — System B (David)
COLOR_NA = "#cccccc"  # grey  — N/A bars


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Chart 1: RAGAS metrics (partial) ─────────────────────────────────────────
def create_ragas_chart(comparison_csv: str, output_path: str) -> None:
    df = pd.read_csv(comparison_csv)

    metrics   = df["Metric"].tolist()
    scores_a  = pd.to_numeric(df["System_A"], errors="coerce").tolist()
    scores_b  = pd.to_numeric(df["System_B"], errors="coerce").tolist()

    x     = np.arange(len(metrics))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (sa, sb) in enumerate(zip(scores_a, scores_b)):
        # System A bar
        ca = COLOR_A if not np.isnan(sa) else COLOR_NA
        va = sa if not np.isnan(sa) else 0.05
        ax.bar(x[i] - width / 2, va, width, color=ca, alpha=0.88, zorder=3)
        if not np.isnan(sa):
            ax.text(x[i] - width / 2, sa + 0.01, f"{sa:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(x[i] - width / 2, 0.07, "N/A",
                    ha="center", va="bottom", fontsize=9, color="#888888")

        # System B bar
        cb = COLOR_B if not np.isnan(sb) else COLOR_NA
        vb = sb if not np.isnan(sb) else 0.05
        ax.bar(x[i] + width / 2, vb, width, color=cb, alpha=0.88, zorder=3)
        if not np.isnan(sb):
            ax.text(x[i] + width / 2, sb + 0.01, f"{sb:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(x[i] + width / 2, 0.07, "N/A",
                    ha="center", va="bottom", fontsize=9, color="#888888")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=12, ha="right")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("RAGAS Evaluation: System A vs System B")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    legend = [
        mpatches.Patch(color=COLOR_A, alpha=0.88, label="System A — GPT-5.4-mini (Goliath)"),
        mpatches.Patch(color=COLOR_B, alpha=0.88, label="System B — Llama-3.1-8B (David)"),
        mpatches.Patch(color=COLOR_NA, alpha=0.88, label="N/A — metric not computable"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"RAGAS chart saved → {output_path}")


# ── Chart 2: Exact-match accuracy by question type ───────────────────────────
def create_exact_match_chart(results_a_path: str, results_b_path: str,
                              output_path: str) -> None:
    data_a = _load(results_a_path)
    data_b = [r for r in _load(results_b_path) if not r["answer"].startswith("ERROR")]

    def em(records):
        return sum(1 for r in records if r["ground_truth"].lower() in r["answer"].lower())

    cats = ["Overall", "Bridge", "Comparison"]

    def breakdown(data):
        bridge = [r for r in data if r["type"] == "bridge"]
        comp   = [r for r in data if r["type"] == "comparison"]
        return [
            (em(data)   / len(data)   * 100) if data   else 0,
            (em(bridge) / len(bridge) * 100) if bridge else 0,
            (em(comp)   / len(comp)   * 100) if comp   else 0,
        ]

    scores_a = breakdown(data_a)
    scores_b = breakdown(data_b)

    x     = np.arange(len(cats))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_a = ax.bar(x - width / 2, scores_a, width,
                    color=COLOR_A, alpha=0.88, label="System A — GPT-5.4-mini (Goliath)", zorder=3)
    bars_b = ax.bar(x + width / 2, scores_b, width,
                    color=COLOR_B, alpha=0.88, label="System B — Llama-3.1-8B (David)", zorder=3)

    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Exact-Match Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exact-Match Accuracy by Question Type")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    # Delta annotations above each pair
    for i, (sa, sb) in enumerate(zip(scores_a, scores_b)):
        delta = sb - sa
        color = "#1a9641" if delta > 0 else "#d73027"
        ax.annotate(f"{delta:+.1f}pp",
                    xy=(x[i], max(sa, sb) + 3),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Exact-match chart saved → {output_path}")


# ── Chart 3: System B loop behaviour ─────────────────────────────────────────
def create_loop_chart(results_b_path: str, output_path: str) -> None:
    data_b = [r for r in _load(results_b_path) if not r["answer"].startswith("ERROR")]

    ret_counts = pd.Series([r["retrieval_attempts"] for r in data_b]).value_counts().sort_index()
    gen_counts = pd.Series([r["generation_attempts"] for r in data_b]).value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # --- Retrieval attempts ---
    bars = ax1.bar(ret_counts.index, ret_counts.values,
                   color=COLOR_B, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
    total = len(data_b)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                 f"{int(h)}\n({h/total*100:.0f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(ret_counts.index)
    ax1.set_xticklabels([f"{i} attempt{'s' if i > 1 else ''}" for i in ret_counts.index])
    ax1.set_ylabel("Number of Questions")
    ax1.set_title("Retrieval Attempts per Question")
    ax1.set_ylim(0, max(ret_counts.values) * 1.22)
    ax1.yaxis.grid(True, alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)
    ax1.axhline(y=total * 0.5, color="#d73027", linestyle="--",
                linewidth=1.2, alpha=0.7, label="50% threshold")
    ax1.legend(fontsize=9)

    # --- Generation attempts ---
    bars2 = ax2.bar(gen_counts.index, gen_counts.values,
                    color="#756bb1", alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                 f"{int(h)}\n({h/total*100:.0f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xticks(gen_counts.index)
    ax2.set_xticklabels([f"{i} attempt{'s' if i > 1 else ''}" for i in gen_counts.index])
    ax2.set_ylabel("Number of Questions")
    ax2.set_title("Generation Attempts per Question")
    ax2.set_ylim(0, max(gen_counts.values) * 1.22)
    ax2.yaxis.grid(True, alpha=0.35, zorder=0)
    ax2.set_axisbelow(True)

    fig.suptitle("System B (David) — Agentic Loop Behaviour  ($n=139$ valid questions)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loop behaviour chart saved → {output_path}")


# ── Chart 4: Completion rate ──────────────────────────────────────────────────
def create_completion_chart(results_b_path: str, output_path: str) -> None:
    data_b = _load(results_b_path)
    ok  = sum(1 for r in data_b if not r["answer"].startswith("ERROR"))
    err = len(data_b) - ok

    _, ax = plt.subplots(figsize=(5, 5))
    _, _, autotexts = ax.pie(
        [ok, err],
        labels=["Completed\n(139)", "Failed — Rate Limit\n(61)"],
        colors=[COLOR_B, "#d73027"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("System B Completion Rate\n(200 HotpotQA Questions)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Completion chart saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication charts")
    parser.add_argument("--system-a",        required=True, help="System A results JSON")
    parser.add_argument("--system-b",        required=True, help="System B results JSON")
    parser.add_argument("--comparison-csv",  required=True, help="RAGAS comparison.csv")
    parser.add_argument("--output-dir",      default="evaluation/results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    create_ragas_chart(args.comparison_csv,
                       str(out / "chart_ragas.png"))
    create_exact_match_chart(args.system_a, args.system_b,
                             str(out / "chart_exact_match.png"))
    create_loop_chart(args.system_b,
                      str(out / "chart_loop_behaviour.png"))
    create_completion_chart(args.system_b,
                            str(out / "chart_completion.png"))

    print("\nAll charts generated.")
