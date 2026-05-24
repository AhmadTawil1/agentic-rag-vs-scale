"""
Visualization: generates publication-quality charts for the 2x2 RAG study.

Usage:
    python evaluation/visualize_results.py \
        --system-a  evaluation/results/system_a_results_TIMESTAMP.json \
        --system-b  evaluation/results/system_b_results_TIMESTAMP.json \
        --system-c  evaluation/results/system_c_results_TIMESTAMP.json \
        --system-d  evaluation/results/system_d_results_TIMESTAMP.json \
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

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

COLOR_A = "#2c7bb6"   # blue   — Naive GPT
COLOR_B = "#1a9641"   # green  — Agentic Llama
COLOR_C = "#f46d43"   # orange — Naive Llama
COLOR_D = "#7b2d8b"   # purple — Agentic GPT

SYSTEMS = {
    "A": ("System A\nNaive GPT",    COLOR_A),
    "B": ("System B\nAgentic Llama", COLOR_B),
    "C": ("System C\nNaive Llama",  COLOR_C),
    "D": ("System D\nAgentic GPT",  COLOR_D),
}


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Chart 1: RAGAS metrics (4 systems) ───────────────────────────────────────
def create_ragas_chart(comparison_csv: str, output_path: str) -> None:
    df = pd.read_csv(comparison_csv)
    metrics = df["Metric"].tolist()

    col_a = [c for c in df.columns if "Goliath" in c or "System A" in c][0]
    col_b = [c for c in df.columns if "David"   in c or "System B" in c][0]
    col_c = [c for c in df.columns if "Hermes"  in c or "System C" in c][0]
    col_d = [c for c in df.columns if "Titan"   in c or "System D" in c][0]

    scores = {
        "A": pd.to_numeric(df[col_a], errors="coerce").tolist(),
        "B": pd.to_numeric(df[col_b], errors="coerce").tolist(),
        "C": pd.to_numeric(df[col_c], errors="coerce").tolist(),
        "D": pd.to_numeric(df[col_d], errors="coerce").tolist(),
    }

    x     = np.arange(len(metrics))
    n     = 4
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(11, 5))

    for idx, (key, offset) in enumerate(zip("ABCD", offsets)):
        label, color = SYSTEMS[key]
        vals = scores[key]
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.bar(x[i] + offset * width, v, width, color=color, alpha=0.88, zorder=3)
                ax.text(x[i] + offset * width, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=10)
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.1)
    ax.set_title("RAGAS Evaluation — All Four Systems")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=SYSTEMS[k][1], alpha=0.88, label=SYSTEMS[k][0].replace("\n", " — "))
               for k in "ABCD"]
    ax.legend(handles=patches, fontsize=9, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"RAGAS chart saved -> {output_path}")


# ── Chart 2: Exact-match accuracy by question type (4 systems) ───────────────
def create_exact_match_chart(paths: dict, output_path: str) -> None:
    def em(records):
        return sum(1 for r in records if r["ground_truth"].lower() in r["answer"].lower())

    def breakdown(data):
        valid  = [r for r in data if not r["answer"].startswith("ERROR")]
        bridge = [r for r in valid if r["type"] == "bridge"]
        comp   = [r for r in valid if r["type"] == "comparison"]
        return [
            (em(valid)  / len(valid)  * 100) if valid  else 0,
            (em(bridge) / len(bridge) * 100) if bridge else 0,
            (em(comp)   / len(comp)   * 100) if comp   else 0,
        ]

    cats   = ["Overall", "Bridge", "Comparison"]
    scores = {k: breakdown(_load(paths[k])) for k in "ABCD"}

    x      = np.arange(len(cats))
    n      = 4
    width  = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(10, 5))

    for key, offset in zip("ABCD", offsets):
        label, color = SYSTEMS[key]
        bars = ax.bar(x + offset * width, scores[key], width,
                      color=color, alpha=0.88, label=label.replace("\n", " — "), zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("Exact-Match Accuracy (%)")
    ax.set_ylim(0, 95)
    ax.set_title("Exact-Match Accuracy by Question Type — 2x2 Design")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Exact-match chart saved -> {output_path}")


# ── Chart 3: Agentic loop behaviour (Systems B and D) ────────────────────────
def create_loop_chart(path_b: str, path_d: str, output_path: str) -> None:
    def loop_series(path):
        data = [r for r in _load(path) if not r["answer"].startswith("ERROR")]
        ret = pd.Series([r.get("retrieval_attempts", 0) for r in data]).value_counts().sort_index()
        gen = pd.Series([r.get("generation_attempts", 0) for r in data]).value_counts().sort_index()
        return data, ret, gen

    data_b, ret_b, gen_b = loop_series(path_b)
    data_d, ret_d, gen_d = loop_series(path_d)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Agentic Loop Behaviour — System B (Llama) vs System D (GPT)",
                 fontsize=13, fontweight="bold")

    def _bar(ax, counts, color, total, title, xlabel):
        bars = ax.bar(counts.index, counts.values, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{int(h)}\n({h/total*100:.0f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(counts.index)
        ax.set_xticklabels([f"{i}" for i in counts.index])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Questions")
        ax.set_title(title)
        ax.set_ylim(0, max(counts.values) * 1.28)
        ax.yaxis.grid(True, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

    _bar(axes[0][0], ret_b, COLOR_B, len(data_b),
         f"System B — Retrieval Attempts (n={len(data_b)})", "Attempts")
    _bar(axes[0][1], gen_b, "#756bb1", len(data_b),
         f"System B — Generation Attempts (n={len(data_b)})", "Attempts")
    _bar(axes[1][0], ret_d, COLOR_D, len(data_d),
         f"System D — Retrieval Attempts (n={len(data_d)})", "Attempts")
    _bar(axes[1][1], gen_d, "#de77ae", len(data_d),
         f"System D — Generation Attempts (n={len(data_d)})", "Attempts")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loop chart saved -> {output_path}")


# ── Chart 4: 2x2 interaction heatmap ─────────────────────────────────────────
def create_interaction_chart(paths: dict, comparison_csv: str, output_path: str) -> None:
    def em_overall(path):
        data = [r for r in _load(path) if not r["answer"].startswith("ERROR")]
        return sum(1 for r in data if r["ground_truth"].lower() in r["answer"].lower()) / len(data) * 100

    df = pd.read_csv(comparison_csv)
    col_a = [c for c in df.columns if "Goliath" in c or "System A" in c][0]
    col_b = [c for c in df.columns if "David"   in c or "System B" in c][0]
    col_c = [c for c in df.columns if "Hermes"  in c or "System C" in c][0]
    col_d = [c for c in df.columns if "Titan"   in c or "System D" in c][0]

    def ragas_avg(col):
        return pd.to_numeric(df[col], errors="coerce").mean()

    # 2x2 grids: rows = model (GPT / Llama), cols = pipeline (Naive / Agentic)
    em_grid = np.array([
        [em_overall(paths["A"]), em_overall(paths["D"])],   # GPT row
        [em_overall(paths["C"]), em_overall(paths["B"])],   # Llama row
    ])
    ragas_grid = np.array([
        [ragas_avg(col_a), ragas_avg(col_d)],
        [ragas_avg(col_c), ragas_avg(col_b)],
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    def _heatmap(ax, data, title, fmt, vmin, vmax, cmap, show_ylabels=True):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Naive Pipeline", "Agentic Pipeline"], fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["GPT-5.4-mini", "Llama-3.1-8B"] if show_ylabels else ["", ""],
                           fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i in range(2):
            for j in range(2):
                val = data[i, j]
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if abs(val - vmin) / (vmax - vmin) > 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, fraction=0.04, pad=0.04)
        # Arrows in axes-fraction coords — row 0 (GPT, top) ≈ 0.75, row 1 (Llama, bottom) ≈ 0.25
        delta_row = data[:, 1] - data[:, 0]
        row_y_frac = [0.75, 0.25]
        for i, d in enumerate(delta_row):
            sym = f"{'+'if d>=0 else ''}{fmt.format(d)}"
            color = "#1a9641" if d >= 0 else "#d73027"
            ax.annotate("", xy=(1.22, row_y_frac[i]), xytext=(1.14, row_y_frac[i]),
                        xycoords="axes fraction", textcoords="axes fraction",
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
            ax.text(1.24, row_y_frac[i], sym, va="center", ha="left", fontsize=10,
                    color=color, fontweight="bold", transform=ax.transAxes,
                    clip_on=False)

    _heatmap(ax1, em_grid,    "Exact-Match Accuracy (%)",  "{:.1f}%",
             em_grid.min()-2, em_grid.max()+2, "Blues", show_ylabels=True)
    _heatmap(ax2, ragas_grid, "Mean RAGAS Score",           "{:.3f}",
             ragas_grid.min()-0.02, ragas_grid.max()+0.02, "Purples", show_ylabels=False)

    fig.suptitle("2x2 Interaction: Model Scale vs Pipeline Architecture",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.92, wspace=0.38, top=0.88, bottom=0.10)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"2x2 interaction chart saved -> {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 2x2 publication charts")
    parser.add_argument("--system-a",       required=True)
    parser.add_argument("--system-b",       required=True)
    parser.add_argument("--system-c",       required=True)
    parser.add_argument("--system-d",       required=True)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--output-dir",     default="evaluation/results")
    args = parser.parse_args()

    paths = {"A": args.system_a, "B": args.system_b,
             "C": args.system_c, "D": args.system_d}
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    create_ragas_chart(args.comparison_csv,       str(out / "chart_ragas.png"))
    create_exact_match_chart(paths,               str(out / "chart_exact_match.png"))
    create_loop_chart(args.system_b, args.system_d, str(out / "chart_loop_behaviour.png"))
    create_interaction_chart(paths, args.comparison_csv, str(out / "chart_2x2_interaction.png"))

    print("\nAll 4 charts generated.")
