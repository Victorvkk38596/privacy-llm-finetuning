"""Figure: step-1 warmup gate (seed 42, eps = 5). No warmup vs. public IMDb head warmup.

Reads the gate records (exp names *__w-none and *__w-public_imdb, seed 42) and plots each model's
validation accuracy, so the collapse without warmup is visible next to the learned models.

  python scripts/plot_warmup_gate.py --records results/grid/records --out results/grid/analysis
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grid_records import load_records, index_records  # noqa: E402
from grid_spec import GATE_CONFIGS, CONFIGS, SST2_VAL_MAJORITY_ACC, exp_name  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
VARIANTS = [("none", "No warmup", "#eb6834", "s"), ("public_imdb", "IMDb head warmup", "#2a78d6", "o")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/grid/records")
    ap.add_argument("--out", default="results/grid/analysis")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    idx = index_records(load_records(a.records))
    fig, ax = plt.subplots(figsize=(4.6, 2.8), constrained_layout=True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(SST2_VAL_MAJORITY_ACC, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
    import matplotlib.transforms as mtransforms
    ax.text(0.5, SST2_VAL_MAJORITY_ACC + 0.6, "majority class (50.9%)",
            transform=mtransforms.blended_transform_factory(ax.transData, ax.transData),
            fontsize=7, color=MUTED, ha="center", va="bottom")
    for vi, (w, label, color, marker) in enumerate(VARIANTS):
        xs, ys = [], []
        for j, c in enumerate(GATE_CONFIGS):
            for k, role in enumerate(("M_A", "M_B")):
                r = idx.get((exp_name(c, w), a.seed, role))
                if r is None:
                    continue
                xs.append(j + (vi - 0.5) * 0.3 + (k - 0.5) * 0.08)
                ys.append(r["metrics"]["val_accuracy"])
        ax.scatter(xs, ys, s=34, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   label=label)
    ax.set_xticks(range(len(GATE_CONFIGS)))
    ax.set_xticklabels([CONFIGS[c][3] for c in GATE_CONFIGS])
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(45, 92)
    ax.set_title(f"Warmup gate, seed {a.seed}, ε = 5 (one point per model)", loc="left", fontsize=9, color=INK)
    ax.legend(loc="center right", frameon=False, fontsize=7.5)
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_warmup_gate.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(a.out, "fig_warmup_gate.{pdf,png}"))


if __name__ == "__main__":
    main()
