"""Aggregate per-model grid records into tables and figures (reproducible from results/grid/records).

Procedure:
  1. One row per model record (models.csv), including failed / privacy-check-failed records.
  2. Seed level: average the two swap-pair models (M_A, M_B) within each (config, seed)  -> seed_level.csv
     A (config, seed) counts only if both models have status "ok".
  3. Config level: mean and sample SD (ddof=1) across seed-level values, plus each seed's value -> summary.csv
     A 95% t-interval (df = n_seeds - 1; t = 4.30 for 3 seeds) is reported for completeness; with three
     seeds it is very wide and reflects seed-to-seed variation only (partition + init + DP noise + order).
  4. Matched-seed differences vs. each comparison's reference arm, on seeds where both arms are ok,
     with the number of seeds whose difference has the same sign -> matched_diffs.csv
  5. Compact LaTeX table (table_core.tex) and one figure per comparison (fig_<comparison>.pdf/.png).

  python scripts/aggregate_grid.py --records results/grid/records --out results/grid/analysis --warmup none
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from grid_spec import CONFIGS, COMPARISONS, SEEDS, SST2_VAL_MAJORITY_ACC, TOTAL_PARAMS, expected_trainable, exp_name  # noqa: E402
from grid_records import load_records, model_collapsed, index_records, grid_pair_states, MANIFESTS_DIR  # noqa: E402
from src.record_validation import validate_model  # noqa: E402

METRICS = [("val_accuracy", "Validation accuracy (%)"), ("rmia_auc", "RMIA ROC-AUC"),
           ("rmia_tpr_at_1fpr", "RMIA TPR at 1% FPR")]
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}

# Audit resolution: each RMIA audit scores 1,024 members (D_A or D_B) against 1,024 non-members (D_out).
# Under no leakage, one audit's AUC has SE sqrt((n1+n2+1)/(12 n1 n2)) ~ 0.0128 and TPR at 1% FPR has
# SE sqrt(0.01*0.99/n1) ~ 0.0031. The shaded band in the figures is chance +- 2 SE of ONE audit: the range
# a single model's metric would fall in about 95% of the time if the attack had no signal.
N_MEMBERS = N_NONMEMBERS = 1024
AUC_SE = math.sqrt((N_MEMBERS + N_NONMEMBERS + 1) / (12 * N_MEMBERS * N_NONMEMBERS))
TPR_SE = math.sqrt(0.01 * 0.99 / N_MEMBERS)
CHANCE_BAND = {"rmia_auc": (0.5, AUC_SE), "rmia_tpr_at_1fpr": (0.01, TPR_SE)}

# Validated categorical slots 1-3 (all-pairs safe) for the three seeds; markers give a second encoding.
SEED_STYLE = {42: ("#2a78d6", "o"), 43: ("#eb6834", "s"), 44: ("#1baf7a", "^")}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


# ----------------------------------------------------------------------------- tables
def model_table(recs):
    rows = []
    for r in recs:
        m, p, t = r.get("metrics") or {}, r.get("privacy") or {}, r.get("training") or {}
        led = p.get("accounting_ledger") or {}
        rows.append({
            "exp_name": r.get("exp_name"), "config_name": r.get("config_name"), "seed": r.get("seed"),
            "role": r.get("role"), "status": r.get("status"), "partition_hash": r.get("partition_hash"),
            "train_layers": r.get("train_layers_spec"),
            "trainable_blocks": ",".join(str(b) for b in (r.get("trainable_blocks") or [])),
            "trainable_params": r.get("trainable_params"), "trainable_fraction_pct": r.get("trainable_fraction_pct"),
            "use_dp": t.get("use_dp"), "head_warmup": (t.get("head_warmup") or {}).get("source"),
            "lr_backbone": t.get("lr_backbone"), "max_steps": t.get("max_steps"),
            "steps_completed": t.get("steps_completed"), "logical_batch_size": t.get("logical_batch_size"),
            "max_grad_norm": t.get("max_grad_norm"),
            "target_epsilon": p.get("target_epsilon"), "target_delta": p.get("target_delta"),
            "accounted_epsilon": p.get("accounted_epsilon"), "noise_multiplier": p.get("noise_multiplier"),
            "accounting_order_ok": led.get("ordering_ok"), "accountant_steps_match": led.get("steps_match_ok"),
            "val_accuracy": m.get("val_accuracy"), "val_pred_pos_rate": m.get("val_pred_pos_rate"),
            "collapsed": model_collapsed(r) if (r.get("metrics") or {}).get("val_accuracy") is not None else None,
            "invalid_reasons": "; ".join(validate_model(r)),
            "rmia_auc": m.get("rmia_auc"), "rmia_tpr_at_1fpr": m.get("rmia_tpr_at_1fpr"),
            "loss_auc": m.get("loss_auc"), "clip_fraction": r.get("clip_fraction"),
            "final_train_loss": r.get("final_train_loss"), "train_time_sec": r.get("train_time_sec"),
            "wall_time_sec": r.get("wall_time_sec"), "error": (r.get("error") or "")[:200],
        })
    return pd.DataFrame(rows)


def wanted_names(warmup):
    return {exp_name(c, warmup): c for c in CONFIGS}


def _n_superseded(records_dir, name, seed):
    import glob
    if not records_dir:
        return 0
    return len(glob.glob(os.path.join(records_dir, "superseded", "*", f"{name}__seed{seed}__M_*.json")))


def seed_table(recs, warmup, manifests_dir=MANIFESTS_DIR, records_dir=None):
    """One row per (config, seed). Acceptance uses the shared rule (src/record_validation.py, via
    grid_pair_states): both models valid, matching nonempty partition hash equal to the seed's
    reference, DP accounting checks passed, accounted eps finite and <= target, grid step counts.
    Only 'ok' pairs carry metrics; every other status stays listed with its reasons."""
    idx = index_records(recs)
    states = grid_pair_states(idx, warmup, manifests_dir=manifests_dir)
    rows = []
    for cfg in CONFIGS:
        name = exp_name(cfg, warmup)
        for s in SEEDS:
            st = states[(cfg, s)]
            row = {"config_name": cfg, "exp_name": name, "seed": s, "status": st["status"],
                   "reasons": "; ".join(st["reasons"]), "partition_hash": st["hash"],
                   "reference_hash": st["reference_hash"], "reference_source": st["reference_source"],
                   "n_superseded_records": _n_superseded(records_dir, name, s)}
            if st["status"] == "ok":
                pair = {x: idx[(name, s, x)] for x in ("M_A", "M_B")}
                for k in ("val_accuracy", "rmia_auc", "rmia_tpr_at_1fpr", "val_pred_pos_rate"):
                    vals = [(pair[x].get("metrics") or {}).get(k) for x in ("M_A", "M_B")]
                    for x, v in zip(("M_A", "M_B"), vals):
                        row[f"{k}_{x}"] = v
                    # swap-pair average within the seed (the unit for across-seed statistics)
                    row[k] = float(np.mean(vals)) if all(v is not None for v in vals) else None
                eps = [(pair[x].get("privacy") or {}).get("accounted_epsilon") for x in ("M_A", "M_B")]
                row["accounted_epsilon"] = float(np.mean(eps)) if all(e is not None for e in eps) else None
                row["n_collapsed_models"] = int(sum(model_collapsed(pair[x]) for x in ("M_A", "M_B")))
                walls = [pair[x].get("wall_time_sec") for x in ("M_A", "M_B") if pair[x].get("wall_time_sec") is not None]
                row["pair_wall_time_sec"] = float(max(walls)) if walls else None
            rows.append(row)
    cols = ["val_accuracy", "rmia_auc", "rmia_tpr_at_1fpr", "val_pred_pos_rate", "accounted_epsilon",
            "n_collapsed_models", "pair_wall_time_sec"]
    df = pd.DataFrame(rows)
    for c in cols:  # present even when no pair is ok yet
        if c not in df.columns:
            df[c] = np.nan
    return df


def _ci(vals):
    n = len(vals)
    if n < 2:
        return (None, None)
    half = T_975[n - 1] * float(np.std(vals, ddof=1)) / math.sqrt(n)
    return (float(np.mean(vals)) - half, float(np.mean(vals)) + half)


def summary_table(seeds):
    rows = []
    for cfg, (layers, eps, comps, label) in CONFIGS.items():
        sub = seeds[(seeds.config_name == cfg)]
        ok = sub[sub.status == "ok"].sort_values("seed")
        row = {"config_name": cfg, "label": label, "train_layers": layers, "target_epsilon": eps,
               "trainable_params": expected_trainable(layers),
               "trainable_fraction_pct": 100 * expected_trainable(layers) / TOTAL_PARAMS,
               "n_seeds_ok": len(ok), "seeds_ok": ",".join(str(s) for s in ok.seed),
               "seeds_not_ok": ";".join(f"{r.seed}:{r.status}" for r in sub[sub.status != "ok"].itertuples()),
               "n_collapsed_models": int(ok.n_collapsed_models.sum()) if len(ok) else 0}
        for k, _ in METRICS + [("accounted_epsilon", "")]:
            vals = ok[k].astype(float).tolist() if len(ok) else []
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{k}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            lo, hi = _ci(vals)
            row[f"{k}_ci95_lo"], row[f"{k}_ci95_hi"] = lo, hi
            for s, v in zip(ok.seed, vals):
                row[f"{k}_seed{s}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def matched_diffs(seeds):
    rows = []
    for comp, spec in COMPARISONS.items():
        ref = spec["reference"]
        for arm in spec["arms"] + spec["context"]:
            if arm == ref:
                continue
            a = seeds[(seeds.config_name == arm) & (seeds.status == "ok")].set_index("seed")
            b = seeds[(seeds.config_name == ref) & (seeds.status == "ok")].set_index("seed")
            # matched seeds: both arms valid and on the same partition
            common = sorted(s for s in set(a.index) & set(b.index)
                            if a.loc[s, "partition_hash"] == b.loc[s, "partition_hash"])
            row = {"comparison": comp, "arm": arm, "reference": ref, "matched_seeds": ",".join(map(str, common)),
                   "n_matched": len(common)}
            for k, _ in METRICS:
                d = [float(a.loc[s, k] - b.loc[s, k]) for s in common]
                row[f"{k}_diff_mean"] = float(np.mean(d)) if d else None
                row[f"{k}_diff_sd"] = float(np.std(d, ddof=1)) if len(d) > 1 else None
                lo, hi = _ci(d)
                row[f"{k}_diff_ci95_lo"], row[f"{k}_diff_ci95_hi"] = lo, hi
                sign = np.sign(np.mean(d)) if d else 0
                row[f"{k}_n_same_sign"] = int(sum(np.sign(x) == sign and x != 0 for x in d)) if d else 0
                for s, x in zip(common, d):
                    row[f"{k}_diff_seed{s}"] = x
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- latex
def _fmt(m, sd, nd):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "--"
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} $\\pm$ {sd:.{nd}f}"


def latex_table(summary, seeds, warmup):
    lines = [r"\begin{tabular}{@{}llrrrlll@{}}", r"\toprule",
             r"Comparison & Arm & Train.\,\% & $\varepsilon$ & $\hat\varepsilon$ & Val.\ acc.\ (\%) & RMIA AUC & TPR@1\%FPR \\",
             r"\midrule"]
    s_by = summary.set_index("config_name")
    for comp, spec in COMPARISONS.items():
        arms = spec["arms"] + spec["context"]
        for i, arm in enumerate(arms):
            r = s_by.loc[arm]
            n = int(r.n_seeds_ok)
            note = ""
            if n < len(SEEDS):
                bad = seeds[(seeds.config_name == arm) & (seeds.status != "ok")]
                kinds = sorted(set(bad.status))
                note = f" ({n}/{len(SEEDS)}; {', '.join(kinds)})".replace("_", r"\_")
            if r.n_collapsed_models:
                note += r" $^\dagger$"
            eps = "$\\infty$" if r.target_epsilon is None or (isinstance(r.target_epsilon, float) and math.isnan(r.target_epsilon)) else f"{r.target_epsilon:g}"
            acc_eps = "--" if r.accounted_epsilon_mean is None or pd.isna(r.accounted_epsilon_mean) else f"{r.accounted_epsilon_mean:.2f}"
            lines.append(" & ".join([
                comp if i == 0 else "", r.label.replace("eps=", r"$\varepsilon$=") + note,
                f"{r.trainable_fraction_pct:.1f}", eps, acc_eps,
                _fmt(r.val_accuracy_mean, r.val_accuracy_sd, 1),
                _fmt(r.rmia_auc_mean, r.rmia_auc_sd, 3),
                _fmt(r.rmia_tpr_at_1fpr_mean, r.rmia_tpr_at_1fpr_sd, 3)]) + r" \\")
        lines.append(r"\midrule" if comp != list(COMPARISONS)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    header = (f"% Generated by scripts/aggregate_grid.py from per-model records (DP head warmup: {warmup}).\n"
              f"% Mean $\\pm$ SD across seed-level values; each seed value = mean of the two swap-pair models.\n")
    return header + "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- figures
def figures(seeds, out_dir, warmup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    titles = {"location": "Layer location at ε = 5 (3 blocks each; all-12 = full backbone)",
              "budget": "Privacy budget, late-3 (blocks 9–11), δ = 1e-5",
              "count": "Trainable block count at ε = 5, anchored at the output end"}
    paths = []
    for comp, spec in COMPARISONS.items():
        arms = spec["arms"] + spec["context"]
        if comp == "budget":
            labels = [f"ε = {CONFIGS[a][1]:g}" for a in arms]
        elif comp == "count":
            labels = [{"dp_late3_e5": "last-3"}.get(a, CONFIGS[a][3]) for a in arms]
        else:
            labels = [CONFIGS[a][3] for a in arms]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
        for ax, (k, ylab) in zip(axes, METRICS):
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ref = {"val_accuracy": SST2_VAL_MAJORITY_ACC, "rmia_auc": 0.5, "rmia_tpr_at_1fpr": 0.01}[k]
            ax.axhline(ref, color=MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
            if k in CHANCE_BAND:
                c, se = CHANCE_BAND[k]
                ax.axhspan(c - 2 * se, c + 2 * se, color=GRID, alpha=0.6, linewidth=0, zorder=0)
            ax.text(0.99, ref, {"val_accuracy": "majority class", "rmia_auc": "chance", "rmia_tpr_at_1fpr": "chance"}[k],
                    transform=ax.get_yaxis_transform(), fontsize=7, color=MUTED, ha="right", va="bottom")
            # paired seed lines (matched seeds) across the comparison's main arms
            for s in SEEDS:
                color, marker = SEED_STYLE[s]
                xs, ys = [], []
                for j, a in enumerate(arms):
                    row = seeds[(seeds.config_name == a) & (seeds.seed == s)]
                    if len(row) and row.iloc[0].status == "ok":
                        xs.append(j + (s - 43) * 0.09)
                        ys.append(float(row.iloc[0][k]))
                    else:
                        xs.append(np.nan)
                        ys.append(np.nan)
                n_main = len(spec["arms"])
                ax.plot(xs[:n_main], ys[:n_main], color=color, linewidth=0.8, alpha=0.45, zorder=2)
                ax.scatter(xs, ys, s=32, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.2,
                           zorder=3, label=f"seed {s}")
            # mean +- SD (black) and not-ok annotations (muted text at the top of the panel)
            for j, a in enumerate(arms):
                ok = seeds[(seeds.config_name == a) & (seeds.status == "ok")]
                if len(ok):
                    m = ok[k].astype(float).mean()
                    sd = ok[k].astype(float).std(ddof=1) if len(ok) > 1 else 0.0
                    ax.vlines(j + 0.3, m - sd, m + sd, color=INK, linewidth=1.2, zorder=4)
                    ax.hlines(m, j + 0.2, j + 0.4, color=INK, linewidth=2, zorder=4)
                nbad = len(SEEDS) - len(ok)
                if nbad:
                    st = sorted(set(seeds[(seeds.config_name == a) & (seeds.status != "ok")].status))
                    ax.text(j, 0.985, f"{nbad}/{len(SEEDS)}\n{'/'.join(st)}", transform=ax.get_xaxis_transform(),
                            fontsize=6, color=INK2, ha="center", va="top", linespacing=0.9)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.12 * (hi - lo))   # headroom for the status notes
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(labels, rotation=25, ha="right")
            if spec["context"]:
                ax.axvline(len(spec["arms"]) - 0.5, color=GRID, linewidth=1)
            ax.set_ylabel(ylab)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h[:3], l[:3], loc="outside upper right", ncol=3, frameon=False, fontsize=7.5)
        fig.suptitle(titles[comp], x=0.01, ha="left", fontsize=9, color=INK)
        fig.text(0.01, -0.02, f"Points: seed-level means of the two swap-pair models (lines join matched seeds). "
                 f"Black bar: mean ± SD over seeds. Grey band: chance ± 2 SE of one audit (1,024 vs 1,024). "
                 f"DP head warmup: {warmup}.", fontsize=6.5, color=INK2)
        for ext in ("pdf", "png"):
            p = os.path.join(out_dir, f"fig_{comp}.{ext}")
            fig.savefig(p, dpi=200, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- Pareto
def pareto_table(summary):
    """Eligibility for the accuracy-vs-RMIA-AUC plot: all three seeds valid and no collapsed model.
    Frontier: no other eligible config has accuracy >= and AUC <= with at least one strict."""
    rows = []
    for r in summary.itertuples():
        reasons = []
        if r.n_seeds_ok < len(SEEDS):
            reasons.append(f"{r.n_seeds_ok}/{len(SEEDS)} seeds valid" + (f" ({r.seeds_not_ok})" if r.seeds_not_ok else ""))
        if r.n_collapsed_models:
            reasons.append(f"{r.n_collapsed_models} collapsed model(s)")
        rows.append({"config_name": r.config_name, "label": r.label, "eligible": not reasons,
                     "excluded_because": "; ".join(reasons),
                     "val_accuracy_mean": r.val_accuracy_mean, "val_accuracy_sd": r.val_accuracy_sd,
                     "rmia_auc_mean": r.rmia_auc_mean, "rmia_auc_sd": r.rmia_auc_sd,
                     "is_dp": r.target_epsilon is not None and not (isinstance(r.target_epsilon, float) and math.isnan(r.target_epsilon))})
    df = pd.DataFrame(rows)
    el = df[df.eligible]
    front = []
    for r in el.itertuples():
        dominated = any((o.val_accuracy_mean >= r.val_accuracy_mean and o.rmia_auc_mean <= r.rmia_auc_mean and
                         (o.val_accuracy_mean > r.val_accuracy_mean or o.rmia_auc_mean < r.rmia_auc_mean))
                        for o in el.itertuples() if o.config_name != r.config_name)
        front.append(not dominated)
    df["on_frontier"] = False
    df.loc[el.index, "on_frontier"] = front
    return df


def pareto_figure(ptab, out_dir, warmup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    el = ptab[ptab.eligible]
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
    ax.axhline(SST2_VAL_MAJORITY_ACC, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(0.5, 0.99, " chance AUC", transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED, va="top")
    ax.text(0.99, SST2_VAL_MAJORITY_ACC, "majority class", transform=ax.get_yaxis_transform(), fontsize=7,
            color=MUTED, ha="right", va="bottom")
    c, se = CHANCE_BAND["rmia_auc"]
    ax.axvspan(c - 2 * se, c + 2 * se, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    ax.text(c + 2 * se, 0.14, " chance ± 2 SE\n of one audit", transform=ax.get_xaxis_transform(),
            fontsize=6.5, color=INK2, va="bottom")
    if len(el):
        for r in el.itertuples():
            dp = bool(r.is_dp)
            ax.errorbar(r.rmia_auc_mean, r.val_accuracy_mean, xerr=r.rmia_auc_sd or 0, yerr=r.val_accuracy_sd or 0,
                        fmt="o", markersize=6, color=SEED_STYLE[42][0] if dp else INK2,
                        mfc=SEED_STYLE[42][0] if dp else SURFACE, mec=SEED_STYLE[42][0] if dp else INK2,
                        elinewidth=1, capsize=0, zorder=3)
        # labels in a column to the right of the DP cluster, with leader lines (no overlaps)
        dp = el[el.is_dp].sort_values("val_accuracy_mean", ascending=False)
        x_lab = float((el[el.is_dp].rmia_auc_mean + el[el.is_dp].rmia_auc_sd.fillna(0)).max()) + 0.02 if len(dp) else 0.55
        prev = None
        for r in dp.itertuples():
            y = r.val_accuracy_mean if prev is None else min(r.val_accuracy_mean, prev - 2.6)
            prev = y
            ax.annotate(r.label, xy=(r.rmia_auc_mean, r.val_accuracy_mean), xytext=(x_lab, y),
                        fontsize=7, color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=0, shrinkB=4))
        for r in el[~el.is_dp].itertuples():
            ax.annotate(r.label + " (non-DP)", (r.rmia_auc_mean, r.val_accuracy_mean), xytext=(0, 8),
                        textcoords="offset points", fontsize=7, color=INK, ha="center")
        lo = min(c - 2 * se - 0.005, float((el.rmia_auc_mean - el.rmia_auc_sd.fillna(0)).min()) - 0.005)
        hi = max(x_lab + 0.04, float((el.rmia_auc_mean + el.rmia_auc_sd.fillna(0)).max()) + 0.02)
        ax.set_xlim(lo, hi)
    else:
        ax.text(0.5, 0.5, "no configuration has three valid,\nnon-collapsed seeds yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("RMIA ROC-AUC (mean ± SD over 3 seeds; lower = less measured leakage)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Accuracy vs. empirical membership leakage", loc="left", fontsize=9, color=INK)
    excl = ptab[~ptab.eligible]
    note = "Excluded: " + "; ".join(f"{r.label} ({r.excluded_because})" for r in excl.itertuples()) if len(excl) else "Excluded: none"
    import textwrap
    fig.text(0.01, -0.01, textwrap.fill(note + f". DP head warmup: {warmup}. Differences in RMIA AUC smaller than "
             f"the grey band are within one audit's sampling error.", 120), fontsize=6.3, color=INK2, va="top")
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"fig_pareto.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- runtime estimate
def runtime_estimate(seeds, warmup):
    lines = ["Runtime estimate from MEASURED completed swap pairs only (wall time per pair, this hardware):"]
    measured = seeds[(seeds.status == "ok") & seeds.pair_wall_time_sec.notna()] if "pair_wall_time_sec" in seeds else seeds.iloc[0:0]
    if not len(measured):
        return "No completed swap pairs with timing yet; run the gate stage first, then re-run with --estimate.\n"
    per_cfg = measured.groupby("config_name").pair_wall_time_sec.mean()
    pending = seeds[seeds.status != "ok"]
    total, unmeasured = 0.0, []
    for r in pending.itertuples():
        if r.config_name in per_cfg:
            total += per_cfg[r.config_name]
        else:
            unmeasured.append(f"{r.config_name}/seed{r.seed}")
    for c, v in per_cfg.items():
        lines.append(f"  {c:14s} {v/60:6.1f} min per swap pair (n={int((measured.config_name == c).sum())})")
    lines.append(f"Pending pairs with a measured config: {len(pending) - len(unmeasured)} -> ~{total/3600:.2f} h")
    if unmeasured:
        lines.append(f"Pending pairs whose config has no measured run yet ({len(unmeasured)}): " + ", ".join(unmeasured))
        lines.append("  (no estimate given for these; validate one run of each first)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/grid/records")
    ap.add_argument("--out", default="results/grid/analysis")
    ap.add_argument("--warmup", default="none", choices=["none", "public_imdb"])
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--manifests", default=MANIFESTS_DIR,
                    help="manifest dir: a seed's manifest hash is the reference for its partition check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = load_records(args.records) if os.path.isdir(args.records) else []
    models = model_table(recs) if recs else pd.DataFrame(columns=["exp_name", "seed", "role", "status", "partition_hash"])
    seeds = seed_table(recs, args.warmup, manifests_dir=args.manifests, records_dir=args.records)
    # Partition consistency is enforced per pair (status 'partition_mismatch'); a valid seed has one hash.
    for s in SEEDS:
        hs = set(seeds[(seeds.seed == s) & (seeds.status == "ok")].partition_hash.dropna())
        assert len(hs) <= 1, f"internal error: seed {s} has several hashes among valid pairs: {hs}"
    summary = summary_table(seeds)
    ptab = pareto_table(summary)
    ptab.to_csv(os.path.join(args.out, "pareto_status.csv"), index=False)
    diffs = matched_diffs(seeds)
    models.to_csv(os.path.join(args.out, "models.csv"), index=False)
    seeds.to_csv(os.path.join(args.out, "seed_level.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    diffs.to_csv(os.path.join(args.out, "matched_diffs.csv"), index=False)
    with open(os.path.join(args.out, "table_core.tex"), "w") as f:
        f.write(latex_table(summary, seeds, args.warmup))
    if not args.no_figures:
        figures(seeds, args.out, args.warmup)
        pareto_figure(ptab, args.out, args.warmup)
    if args.estimate:
        est = runtime_estimate(seeds, args.warmup)
        with open(os.path.join(args.out, "runtime_estimate.txt"), "w") as f:
            f.write(est)
        print(est)
    n_ok = int((seeds.status == "ok").sum())
    print(f"{len(recs)} records -> {n_ok}/{len(seeds)} swap pairs ok. Outputs in {args.out}")
    for r in seeds[seeds.status != "ok"].itertuples():
        if r.status != "pending":
            print(f"  {r.exp_name} seed {r.seed}: {r.status} {('- ' + r.reasons) if r.reasons else ''}")


if __name__ == "__main__":
    main()
