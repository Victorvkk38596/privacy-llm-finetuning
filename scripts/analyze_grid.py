"""Aggregate results/grid/records.jsonl into seed-level results, summaries, paired
differences, LaTeX tables, and figures. Everything is regenerated from the saved records.

Aggregation (per the protocol):
  1. Per (config, seed): average the two swap-pair models (M_A on D_A, M_B on D_B).
  2. Per config: mean and sample SD (ddof=1) over the three seed-level values.
     A 95% t-interval (df = n_seeds - 1; t = 4.303 for 3 seeds) is reported as well; with
     three seeds it is wide and should be read as a rough uncertainty statement.
  3. Comparisons use matched seeds: the per-seed difference between two configs, and
     whether its sign is the same in every seed.

Usage:  python scripts/analyze_grid.py [--records results/grid/records.jsonl]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/grid"
FIG = ROOT / "figures"

LABELS = {
    "early3_eps5": "Early-3 (0-2)", "mid3_eps5": "Mid-3 (4-6)", "late3_eps5": "Late-3 (9-11)",
    "all12_eps5": "All-12", "clean_all12": "All-12, no DP",
    "late3_eps1": "Late-3, eps=1", "late3_eps8": "Late-3, eps=8",
    "last1_eps5": "Last-1 (11)", "last6_eps5": "Last-6 (6-11)",
}
COMPARISONS = {
    "location": ["early3_eps5", "mid3_eps5", "late3_eps5", "all12_eps5"],
    "budget": ["late3_eps1", "late3_eps5", "late3_eps8"],
    "count": ["last1_eps5", "late3_eps5", "last6_eps5", "all12_eps5"],
}
PAIRS = [  # (comparison, A, B): reports A - B per seed
    ("location", "mid3_eps5", "early3_eps5"),
    ("location", "late3_eps5", "mid3_eps5"),
    ("location", "late3_eps5", "early3_eps5"),
    ("location", "late3_eps5", "all12_eps5"),
    ("location", "mid3_eps5", "all12_eps5"),
    ("location", "early3_eps5", "all12_eps5"),
    ("budget", "late3_eps5", "late3_eps1"),
    ("budget", "late3_eps8", "late3_eps5"),
    ("count", "late3_eps5", "last1_eps5"),
    ("count", "last6_eps5", "late3_eps5"),
    ("count", "all12_eps5", "last6_eps5"),
    ("context", "clean_all12", "all12_eps5"),
]
METRICS = ["val_accuracy", "rmia_auc", "rmia_tpr_at_1fpr", "epsilon_spent",
           "loss_auc", "rmia_tpr_at_5fpr", "final_train_loss", "train_time_sec"]

BLUE, GRAY, INK, INK2, SURF = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    failed = df[df["status"] != "ok"].copy()
    ok = df[df["status"] == "ok"].copy()
    # keep the latest record per (exp, seed, role) in case a pair was re-run
    ok = ok.drop_duplicates(subset=["exp_name", "seed", "role"], keep="last")
    # a failure is resolved if the pair later completed
    if len(failed):
        done = set(zip(ok.exp_name, ok.seed))
        failed["resolved_by_rerun"] = [(e, s) in done for e, s in zip(failed.exp_name, failed.seed)]
    return ok, failed


def seed_level(ok):
    agg = {m: "mean" for m in METRICS if m in ok}
    g = ok.groupby(["exp_name", "seed"])
    sl = g.agg(agg)
    sl["n_models"] = g.size()
    first = g.first()
    for c in ["train_layers", "trainable_params", "param_fraction_pct", "use_dp", "target_epsilon",
              "target_delta", "noise_multiplier", "partition_hash", "head_warmup_source"]:
        if c in first:
            sl[c] = first[c]
    # per-role values kept for transparency
    for role in ("M_A", "M_B"):
        r = ok[ok.role == role].set_index(["exp_name", "seed"])
        for m in ("val_accuracy", "rmia_auc", "rmia_tpr_at_1fpr"):
            sl[f"{m}_{role}"] = r[m]
    return sl.reset_index()


def summarize(sl):
    out = []
    for exp, d in sl.groupby("exp_name"):
        row = {"exp_name": exp, "label": LABELS.get(exp, exp), "n_seeds": len(d),
               "seeds": ",".join(str(s) for s in sorted(d.seed)),
               "train_layers": d.train_layers.iloc[0], "trainable_params": int(d.trainable_params.iloc[0]),
               "param_fraction_pct": d.param_fraction_pct.iloc[0],
               "target_epsilon": d.target_epsilon.iloc[0]}
        for m in ("val_accuracy", "rmia_auc", "rmia_tpr_at_1fpr", "epsilon_spent", "loss_auc"):
            v = d[m].dropna().to_numpy(float)
            row[f"{m}_mean"] = v.mean() if len(v) else np.nan
            row[f"{m}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            if len(v) > 1:
                h = stats.t.ppf(0.975, len(v) - 1) * row[f"{m}_sd"] / math.sqrt(len(v))
                row[f"{m}_ci95_lo"], row[f"{m}_ci95_hi"] = row[f"{m}_mean"] - h, row[f"{m}_mean"] + h
            for s, val in zip(d.seed, d[m]):
                row[f"{m}_s{s}"] = val
        out.append(row)
    return pd.DataFrame(out)


def paired(sl):
    rows = []
    piv = {m: sl.pivot(index="seed", columns="exp_name", values=m)
           for m in ("val_accuracy", "rmia_auc", "rmia_tpr_at_1fpr")}
    for comp, a, b in PAIRS:
        for m, p in piv.items():
            if a not in p or b not in p:
                continue
            d = (p[a] - p[b]).dropna()
            if d.empty:
                continue
            signs = np.sign(d.to_numpy())
            rows.append({
                "comparison": comp, "A": a, "B": b, "metric": m, "n_seeds": len(d),
                **{f"diff_s{s}": v for s, v in d.items()},
                "mean_diff": d.mean(), "sd_diff": d.std(ddof=1) if len(d) > 1 else np.nan,
                "same_sign_all_seeds": bool(len(d) > 1 and (np.all(signs > 0) or np.all(signs < 0))),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- tables
def fmt(m, sd, digits):
    if pd.isna(m):
        return "--"
    return f"{m:.{digits}f}" + ("" if pd.isna(sd) else f" $\\pm$ {sd:.{digits}f}")


def latex_table(summ, path):
    order = ["clean_all12", "early3_eps5", "mid3_eps5", "late3_eps5", "all12_eps5",
             "late3_eps1", "late3_eps8", "last1_eps5", "last6_eps5"]
    groups = {"clean_all12": "Context", "early3_eps5": "Location", "late3_eps1": "Budget",
              "last1_eps5": "Count"}
    s = summ.set_index("exp_name")
    lines = [r"\begin{tabular}{@{}llrrcccc@{}}", r"\toprule",
             r"Config & Blocks & Trainable & Target $\varepsilon$ & Seeds & Val.\ acc.\ (\%) & RMIA AUC & TPR@1\%FPR (\%) \\",
             r"\midrule"]
    for e in order:
        if e not in s.index:
            continue
        r = s.loc[e]
        if e in groups and e != "clean_all12":
            lines.append(r"\midrule")
        eps = "$\\infty$" if pd.isna(r.target_epsilon) else f"{r.target_epsilon:g}"
        blocks = {"all": "0--11"}.get(str(r.train_layers), str(r.train_layers).replace(",", ", "))
        lines.append(" & ".join([
            LABELS.get(e, e).replace("eps=", "$\\varepsilon$="), blocks,
            f"{r.trainable_params/1e6:.1f}M ({r.param_fraction_pct:.1f}\\%)", eps, str(int(r.n_seeds)),
            fmt(r.val_accuracy_mean, r.val_accuracy_sd, 1),
            fmt(r.rmia_auc_mean, r.rmia_auc_sd, 3),
            fmt(100 * r.rmia_tpr_at_1fpr_mean, 100 * r.rmia_tpr_at_1fpr_sd, 1),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def latex_seed_table(sl, path):
    order = ["clean_all12", "early3_eps5", "mid3_eps5", "late3_eps5", "all12_eps5",
             "late3_eps1", "late3_eps8", "last1_eps5", "last6_eps5"]
    seeds = sorted(sl.seed.unique())
    p_acc = sl.pivot(index="exp_name", columns="seed", values="val_accuracy")
    p_auc = sl.pivot(index="exp_name", columns="seed", values="rmia_auc")
    p_eps = sl.pivot(index="exp_name", columns="seed", values="epsilon_spent")
    cols = "l" + "r" * len(seeds) + "r" * len(seeds) + "r"
    head = (" & ".join(["Config"] + [f"Acc s{s}" for s in seeds] + [f"AUC s{s}" for s in seeds]
                       + [r"Accounted $\varepsilon$"]) + r" \\")
    lines = [rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule", head, r"\midrule"]
    for e in order:
        if e not in p_acc.index:
            continue
        acc = [("--" if pd.isna(p_acc.loc[e].get(s)) else f"{p_acc.loc[e][s]:.1f}") for s in seeds]
        au = [("--" if pd.isna(p_auc.loc[e].get(s)) else f"{p_auc.loc[e][s]:.3f}") for s in seeds]
        ev = p_eps.loc[e].dropna()
        eps = "--" if ev.empty else f"{ev.max():.3f}"
        lines.append(" & ".join([LABELS.get(e, e).replace("eps=", "$\\varepsilon$=")] + acc + au + [eps]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def _panel(ax, xs, sl, exps, metric, scale=1.0, xticklabels=None, logx=False):
    for x, e in zip(xs, exps):
        v = sl[sl.exp_name == e][metric].to_numpy(float) * scale
        if len(v) == 0:
            continue
        jitter = np.linspace(-0.06, 0.06, len(v)) * (x if logx else 1)
        ax.scatter(np.asarray(x) + (0 if logx else jitter), v, s=14, color=GRAY, zorder=3,
                   linewidths=0, label=None)
    means = [sl[sl.exp_name == e][metric].mean() * scale for e in exps]
    sds = [sl[sl.exp_name == e][metric].std(ddof=1) * scale for e in exps]
    ax.errorbar(xs, means, yerr=[0 if pd.isna(s) else s for s in sds], color=BLUE, lw=2,
                marker="o", ms=6, capsize=3, zorder=4, mec=SURF, mew=1.5)
    if logx:
        ax.set_xscale("log")
    ax.set_xticks(xs)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    _style(ax)


def figures(sl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelcolor": INK2,
                         "text.color": INK, "figure.facecolor": SURF, "savefig.facecolor": SURF})
    clean_acc = sl[sl.exp_name == "clean_all12"].val_accuracy.mean()
    clean_auc = sl[sl.exp_name == "clean_all12"].rmia_auc.mean()
    specs = [
        ("location", list(range(4)), COMPARISONS["location"],
         ["Early-3\n(0-2)", "Mid-3\n(4-6)", "Late-3\n(9-11)", "All-12"], False, "Layer location, target $\\varepsilon$=5"),
        ("budget", [1, 5, 8], COMPARISONS["budget"], ["1", "5", "8"], False,
         "Privacy budget, late-3 blocks"),
        ("count", None, COMPARISONS["count"], ["Last-1", "Last-3", "Last-6", "All-12"], True,
         "Blocks trained from the output end, target $\\varepsilon$=5"),
    ]
    for name, xs, exps, xt, logx, title in specs:
        exps_present = [e for e in exps if e in set(sl.exp_name)]
        if len(exps_present) < 2:
            continue
        if name == "count":
            xs = [float(sl[sl.exp_name == e].trainable_params.iloc[0]) / 1e6 for e in exps_present]
            xt = [f"{lab}\n{x:.1f}M" for lab, x, e in zip(xt, xs, exps) if e in exps_present] \
                if len(exps_present) == len(exps) else [f"{x:.1f}M" for x in xs]
        else:
            idx = [exps.index(e) for e in exps_present]
            xs = [xs[i] for i in idx]
            xt = [xt[i] for i in idx]
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
        for ax, metric, scale, ylab in zip(
                axes, ("val_accuracy", "rmia_auc", "rmia_tpr_at_1fpr"), (1, 1, 100),
                ("Validation accuracy (%)", "RMIA ROC-AUC", "TPR at 1% FPR (%)")):
            _panel(ax, xs, sl, exps_present, metric, scale, xt, logx)
            ax.set_ylabel(ylab)
            if metric == "val_accuracy":
                ax.axhline(50, color=INK2, lw=0.8, ls=":")
                ax.text(ax.get_xlim()[0], 50.5, " majority class", fontsize=7, color=INK2, va="bottom")
                if not pd.isna(clean_acc):
                    ax.axhline(clean_acc, color=INK2, lw=0.8, ls="--")
                    ax.text(ax.get_xlim()[0], clean_acc + 0.5, f" no-DP all-12: {clean_acc:.1f}",
                            fontsize=7, color=INK2, va="bottom")
            elif metric == "rmia_auc":
                ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
                ax.text(ax.get_xlim()[0], 0.5, " chance", fontsize=7, color=INK2, va="bottom")
            else:
                ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
                ax.text(ax.get_xlim()[0], 1.0, " chance", fontsize=7, color=INK2, va="bottom")
            if name == "budget":
                ax.set_xlabel("Target $\\varepsilon$ ($\\delta=10^{-5}$)")
            elif name == "count":
                ax.set_xlabel("Trainable parameters (log scale)")
        fig.suptitle(title + "  (blue: mean $\\pm$ SD over seeds; gray: individual seeds)",
                     fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"grid_{name}.{ext}", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(OUT / "records.jsonl"))
    args = ap.parse_args()
    ok, failed = load(args.records)
    ok.to_csv(OUT / "per_model.csv", index=False)
    sl = seed_level(ok)
    sl.to_csv(OUT / "seed_level.csv", index=False)
    summ = summarize(sl)
    summ.to_csv(OUT / "summary_by_config.csv", index=False)
    pr = paired(sl)
    pr.to_csv(OUT / "paired_differences.csv", index=False)
    latex_table(summ, OUT / "table_summary.tex")
    latex_seed_table(sl, OUT / "table_seeds.tex")
    if len(failed):
        failed.to_csv(OUT / "failed_runs.csv", index=False)
    figures(sl)

    pd.set_option("display.width", 200)
    cols = ["exp_name", "n_seeds", "val_accuracy_mean", "val_accuracy_sd", "rmia_auc_mean",
            "rmia_auc_sd", "rmia_tpr_at_1fpr_mean", "epsilon_spent_mean"]
    print(summ[cols].round(4).to_string(index=False))
    if len(pr):
        print()
        print(pr[pr.metric != "rmia_tpr_at_1fpr"][["A", "B", "metric", "n_seeds", "mean_diff", "sd_diff",
                                                   "same_sign_all_seeds"]].round(4).to_string(index=False))
    if len(failed):
        print(f"\n{len(failed)} failed-run records (see results/grid/failed_runs.csv)")


if __name__ == "__main__":
    main()
