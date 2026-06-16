#!/usr/bin/env python3
"""
analyze_experiments.py
======================

Comprehensive analysis of the hyperparameter sweep produced by run_experiments.py.

It reads
  * experiment_results.csv         one row per run (hyperparameters + best_train_accuracy)
  * a per-run TRAIN log  (log.jsonl)        -> learning dynamics
  * a per-run TEST  log  (testing-log.jsonl) -> generalisation + per-class accuracy

and produces
  * enriched_results.csv   original data + architecture features + train/test metrics
  * analysis_plots/*.png    importance, overfitting, per-class, interactions, curves ...
  * analysis_report.md      a data-driven written summary of the findings
  * console output          the same headline findings

WHY TEST ACCURACY MATTERS
-------------------------
The runner's summary CSV only records *train* accuracy. Train accuracy is a poor yard-
stick for choosing hyperparameters: bigger nets, more data and less dropout almost always
raise train accuracy while often *hurting* generalisation. Now that you log a test set
(testing-log.jsonl), this analyzer makes **test accuracy** the primary metric and reports
the train-test gap so you can see overfitting directly.

LOG DISCOVERY (no need to re-run anything)
------------------------------------------
For each run_id the analyzer looks for the two logs in this order:
  TRAIN: logs/<run_id>.json  ->  logs/<run_id>.train.json  ->  training-data/<net>_*/log.jsonl
  TEST : logs/<run_id>.test.json -> logs/<run_id>.test.jsonl -> training-data/<net>_*/testing-log.jsonl
where <net> is NETWORK_TEMPLATE.format(run_id=run_id) (default "neuro_{run_id}.wgt"),
matching the naming used by run_experiments.py.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                 # headless / no display needed
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")     # keep the console output readable

# Optional: sklearn is only used for a cross-check of importance. Degrades gracefully.
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


# ----------------------------------------------------------------------------------
# CONFIGURATION (overridable via command line)
# ----------------------------------------------------------------------------------
DEFAULTS = dict(
    summary_csv="experiment_results.csv",
    logs_dir="logs",
    training_data_dir="training-data",
    network_template="neuro_{run_id}.wgt",
    out_csv="enriched_results.csv",
    plot_dir="analysis_plots",
    report="analysis_report.md",
    max_epochs=500,           # constant from the sweep; used to flag "never early-stopped"
    top_n=8,                  # how many top runs to draw learning curves / per-class heatmaps for
)

# Candidate hyperparameter columns (only those present AND varied are actually analysed).
FACTOR_COLS = ["layers", "initialLearningRate", "dropoutRate",
               "patience", "learningRateDecay", "datasetLimitPerLabel"]


# ==================================================================================
# 1. LOADING + ENRICHMENT
# ==================================================================================
def load_summary(csv_path):
    df = pd.read_csv(csv_path)
    for col in ["initialLearningRate", "dropoutRate", "patience", "learningRateDecay",
                "datasetLimitPerLabel", "best_train_accuracy", "best_epoch"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "run_id" not in df.columns:
        df["run_id"] = [f"run_{i:04d}" for i in range(len(df))]
    return df


# ---- architecture features parsed from the "layers" string ----------------------
def parse_layers(layers_str):
    try:
        sizes = [int(x) for x in str(layers_str).split(",") if x.strip() != ""]
    except ValueError:
        return {}
    if len(sizes) < 2:
        return {}
    hidden = sizes[1:-1]
    # weights + biases between consecutive layers = a good capacity proxy
    n_params = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    return {
        "n_weight_layers": len(sizes) - 1,
        "n_hidden_layers": len(hidden),
        "total_hidden_units": int(sum(hidden)) if hidden else 0,
        "widest_hidden": int(max(hidden)) if hidden else 0,
        "n_params": int(n_params),
    }


def enrich_with_architecture(df):
    feats = df["layers"].apply(parse_layers).apply(pd.Series) if "layers" in df.columns else pd.DataFrame()
    return pd.concat([df, feats], axis=1)


# ---- file discovery -------------------------------------------------------------
def _resolve_log(run_id, cfg, kind):
    """kind in {'train','test'} -> return a Path to the log file or None."""
    logs = Path(cfg["logs_dir"])
    net = cfg["network_template"].format(run_id=run_id)
    td = Path(cfg["training_data_dir"])

    if kind == "train":
        candidates = [logs / f"{run_id}.json", logs / f"{run_id}.train.json",
                      logs / f"{run_id}.train.jsonl"]
        td_name = "log.jsonl"
    else:
        candidates = [logs / f"{run_id}.test.json", logs / f"{run_id}.test.jsonl"]
        td_name = "testing-log.jsonl"

    for c in candidates:
        if c.exists():
            return c
    # fallback: original training-data run directory (timestamped name)
    if td.exists():
        matches = sorted(td.glob(f"{net}_*/{td_name}"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---- TRAIN dynamics from log.jsonl ----------------------------------------------
def train_dynamics(run_id, cfg):
    path = _resolve_log(run_id, cfg, "train")
    if path is None:
        return {}
    rows = _read_jsonl(path)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    if "epochNumber" not in df or "bestTrainAccuracy" not in df:
        return {}
    df = df.sort_values("epochNumber")
    best = df["bestTrainAccuracy"].astype(float).to_numpy()
    ep = df["epochNumber"].astype(int).to_numpy()
    lr = df["learningRate"].astype(float).to_numpy() if "learningRate" in df else np.array([])
    final_best = float(best.max())

    def epochs_to(frac):
        target = frac * final_best
        hit = ep[best >= target]
        return int(hit.min()) if len(hit) else np.nan

    out = {
        "train_best_acc": final_best,
        "epochs_run": int(ep.max()) + 1,
        "reached_max_epochs": bool(ep.max() >= cfg["max_epochs"] - 1),
        "mean_train_acc": float(df["trainAccuracy"].astype(float).mean()) if "trainAccuracy" in df else np.nan,
        "epochs_to_90pct_of_best": epochs_to(0.90),
        "epochs_to_99pct_of_best": epochs_to(0.99),
    }
    if lr.size:
        out["n_lr_decays"] = int((np.diff(lr) < 0).sum())
        out["final_lr"] = float(lr[-1])
    return out


# ---- TEST metrics from testing-log.jsonl ----------------------------------------
def _entry_to_acc(entry):
    """Return (overall_micro, macro, per_class_list) for one testing-log line."""
    stats = entry.get("stats", [])
    passed = np.array([s.get("passedTests", 0) for s in stats], dtype=float)
    total = np.array([s.get("totalTests", 0) for s in stats], dtype=float)
    if total.sum() == 0:
        return np.nan, np.nan, []
    per_class = np.divide(passed, total, out=np.full_like(passed, np.nan), where=total > 0)
    micro = passed.sum() / total.sum()
    macro = float(np.nanmean(per_class))
    return float(micro), macro, per_class.tolist()


def test_metrics(run_id, cfg):
    path = _resolve_log(run_id, cfg, "test")
    if path is None:
        return {}
    rows = _read_jsonl(path)
    if not rows:
        return {}

    checkpoints = []          # (epoch, micro, macro, per_class)  for epoch >= 0
    final = None              # the epoch == -1 entry (model after training finished)
    for e in rows:
        micro, macro, pc = _entry_to_acc(e)
        if np.isnan(micro):
            continue
        if e.get("epoch", -1) == -1:
            final = (e["epoch"], micro, macro, pc)
        else:
            checkpoints.append((int(e["epoch"]), micro, macro, pc))

    checkpoints.sort(key=lambda t: t[0])
    # if no explicit final marker, treat the last checkpoint as final
    if final is None and checkpoints:
        final = checkpoints[-1]
    if final is None:
        return {}

    all_entries = checkpoints + ([final] if final not in checkpoints else [])
    micros = [m for (_, m, _, _) in all_entries]
    best_idx = int(np.argmax(micros))
    best_epoch, best_micro, best_macro, _ = all_entries[best_idx]

    _, final_micro, final_macro, final_pc = final
    final_pc = np.array(final_pc, dtype=float)

    out = {
        "test_acc": final_micro,                         # headline: accuracy of the final model
        "test_acc_macro": final_macro,                   # class-balanced accuracy
        "best_test_acc": float(max(micros)),             # best across periodic checkpoints
        "best_test_epoch": int(best_epoch),
        "overfit_drop": float(max(micros) - final_micro),  # how much test acc fell from its peak
        "class_acc_min": float(np.nanmin(final_pc)) if final_pc.size else np.nan,
        "class_acc_max": float(np.nanmax(final_pc)) if final_pc.size else np.nan,
        "class_acc_std": float(np.nanstd(final_pc)) if final_pc.size else np.nan,
        "worst_class": int(np.nanargmin(final_pc)) if final_pc.size else -1,
        "best_class": int(np.nanargmax(final_pc)) if final_pc.size else -1,
        "n_classes": int(final_pc.size),
    }
    # expose per-class accuracy as classacc_<i> columns
    for i, v in enumerate(final_pc):
        out[f"classacc_{i}"] = float(v)
    # keep the trajectory for the learning-curve plot
    out["_test_traj"] = [(ep, m) for (ep, m, _, _) in checkpoints]
    out["_test_final_pt"] = final_micro
    return out


def enrich_runs(df, cfg):
    train_rows, test_rows, traj = [], [], {}
    for run_id in df["run_id"]:
        td = train_dynamics(run_id, cfg)
        tm = test_metrics(run_id, cfg)
        traj[run_id] = (tm.pop("_test_traj", None), tm.pop("_test_final_pt", None))
        train_rows.append(td)
        test_rows.append(tm)
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(train_rows), pd.DataFrame(test_rows)], axis=1)

    # generalisation gap: best the model fit on train minus what it scores on test
    base_train = "train_best_acc" if "train_best_acc" in df else "best_train_accuracy"
    if base_train in df and "test_acc" in df:
        df["gen_gap"] = df[base_train] - df["test_acc"]
    return df, traj


# ==================================================================================
# 2. STATISTICS
# ==================================================================================
def active_factors(df):
    return [c for c in FACTOR_COLS if c in df.columns and df[c].nunique(dropna=True) > 1]


def eta_squared(df, factor, target):
    """Fraction of variance in `target` explained by `factor` (one-way).
    On a balanced full-factorial grid these are unconfounded and comparable."""
    d = df[[factor, target]].dropna()
    if d[factor].nunique() < 2 or len(d) < 3:
        return np.nan
    grand = d[target].mean()
    ss_total = ((d[target] - grand) ** 2).sum()
    if ss_total == 0:
        return np.nan
    ss_between = sum(len(g) * (g[target].mean() - grand) ** 2 for _, g in d.groupby(factor))
    return ss_between / ss_total


def importance_table(df, factors, target, maximize=True):
    """Per-factor: variance explained, best level, and spread of the level means.
    Set maximize=False for metrics where lower is better (e.g. the train-test gap)."""
    rows = []
    for f in factors:
        d = df[[f, target]].dropna()
        if d.empty:
            continue
        means = d.groupby(f)[target].mean()
        best_level = means.idxmax() if maximize else means.idxmin()
        best_mean = means.max() if maximize else means.min()
        worst_mean = means.min() if maximize else means.max()
        rows.append({
            "factor": f,
            "var_explained_%": 100 * eta_squared(df, f, target),
            "best_level": best_level,
            "best_level_mean": best_mean,
            "worst_level_mean": worst_mean,
            "spread": means.max() - means.min(),
        })
    out = pd.DataFrame(rows).sort_values("var_explained_%", ascending=False)
    return out.reset_index(drop=True)


def rf_importance(df, factors, target):
    """Optional cross-check that captures interactions/non-linearities."""
    if not _HAVE_SKLEARN:
        return None
    d = df[factors + [target]].dropna()
    if len(d) < 20:
        return None
    X = pd.DataFrame({f: d[f].astype("category").cat.codes for f in factors})
    y = d[target].to_numpy()
    rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(X, y)
    r = permutation_importance(rf, X, y, n_repeats=10, random_state=0, n_jobs=-1)
    imp = pd.DataFrame({"factor": factors, "rf_importance": r.importances_mean})
    return imp.sort_values("rf_importance", ascending=False).reset_index(drop=True)


def interaction_strength(df, f1, f2, target):
    """How much the two factors interact beyond their additive effects (0..1)."""
    d = df[[f1, f2, target]].dropna()
    if d.empty or d[f1].nunique() < 2 or d[f2].nunique() < 2:
        return np.nan
    grand = d[target].mean()
    ss_total = ((d[target] - grand) ** 2).sum()
    if ss_total == 0:
        return np.nan
    r_eff = d.groupby(f1)[target].mean() - grand
    c_eff = d.groupby(f2)[target].mean() - grand
    cell = d.groupby([f1, f2])[target].agg(["mean", "count"])
    inter_ss = 0.0
    for (a, b), row in cell.iterrows():
        pred = grand + r_eff[a] + c_eff[b]
        inter_ss += row["count"] * (row["mean"] - pred) ** 2
    return inter_ss / ss_total


def best_per_architecture(df, target):
    if "layers" not in df.columns:
        return pd.DataFrame()
    idx = df.dropna(subset=[target]).groupby("layers")[target].idxmax()
    cols = ["layers", target] + [c for c in
            ["initialLearningRate", "dropoutRate", "patience", "learningRateDecay",
             "datasetLimitPerLabel", "gen_gap", "n_params"] if c in df.columns]
    return df.loc[idx, cols].sort_values(target, ascending=False).reset_index(drop=True)


def class_difficulty(df):
    cols = sorted([c for c in df.columns if c.startswith("classacc_")],
                  key=lambda c: int(c.split("_")[1]))
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame({
        "class": [int(c.split("_")[1]) for c in cols],
        "mean_acc": [df[c].mean() for c in cols],
        "std_acc": [df[c].std() for c in cols],
        "min_acc": [df[c].min() for c in cols],
    }).sort_values("mean_acc").reset_index(drop=True)
    return out


# ==================================================================================
# 3. PLOTS  (each wrapped in try/except by the caller)
# ==================================================================================
def _save(fig, plot_dir, name):
    p = Path(plot_dir) / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def plot_importance(df, factors, target, plot_dir):
    imp = importance_table(df, factors, target)
    rf = rf_importance(df, factors, target)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(imp) + 2))
    ax.barh(imp["factor"][::-1], imp["var_explained_%"][::-1], color="#4C72B0")
    ax.set_xlabel(f"variance of {target} explained (%)")
    ax.set_title(f"Which hyperparameters drive {target}?")
    for y, v in zip(range(len(imp))[::-1], imp["var_explained_%"][::-1]):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=9)
    _save(fig, plot_dir, f"importance_{target}.png")
    return imp, rf


def plot_marginal_effects(df, factors, target, plot_dir):
    n = len(factors)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    for i, f in enumerate(factors):
        ax = axes[i]
        order = sorted(df[f].dropna().unique(), key=lambda x: (isinstance(x, str), x))
        sns.boxplot(data=df, x=f, y=target, order=order, ax=ax, color="#A6CEE3", fliersize=2)
        means = df.groupby(f)[target].mean().reindex(order)
        ax.plot(range(len(order)), means.values, "o-", color="crimson", lw=1.5, label="mean")
        ax.set_title(f"{target} vs {f}")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Marginal effect of each hyperparameter on {target}", fontsize=14)
    _save(fig, plot_dir, f"marginal_effects_{target}.png")


def plot_overfitting(df, plot_dir):
    base_train = "train_best_acc" if "train_best_acc" in df else "best_train_accuracy"
    if base_train not in df or "test_acc" not in df:
        return
    d = df.dropna(subset=[base_train, "test_acc"])
    color_by = "dropoutRate" if "dropoutRate" in d else None
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(d[base_train], d["test_acc"],
                    c=(d[color_by] if color_by else "#4C72B0"),
                    cmap="viridis", s=18, alpha=0.7)
    lim = [min(d[base_train].min(), d["test_acc"].min()), 1.0]
    ax.plot(lim, lim, "k--", lw=1, label="train = test (no overfit)")
    ax.set_xlabel("train accuracy (best)")
    ax.set_ylabel("test accuracy (final)")
    ax.set_title("Overfitting: every run's train vs test accuracy")
    if color_by:
        fig.colorbar(sc, ax=ax, label=color_by)
    ax.legend()
    _save(fig, plot_dir, "overfitting_train_vs_test.png")


def plot_architecture(df, target, plot_dir):
    if "layers" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    order = df.groupby("layers")[target].mean().sort_values().index
    sns.boxplot(data=df, x="layers", y=target, order=order, ax=axes[0], color="#B2DF8A")
    axes[0].set_title(f"{target} by architecture")
    axes[0].tick_params(axis="x", rotation=45)
    for lbl in axes[0].get_xticklabels():
        lbl.set_ha("right")
    if "n_params" in df.columns:
        d = df.dropna(subset=["n_params", target])
        axes[1].scatter(d["n_params"], d[target], s=18, alpha=0.6, color="#1F78B4")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("parameter count (log)")
        axes[1].set_ylabel(target)
        axes[1].set_title(f"{target} vs network capacity")
    _save(fig, plot_dir, f"architecture_{target}.png")


def _arch_order_by_capacity(df):
    """Architectures ordered small -> large (by parameter count), so the x-axis
    means the same thing in every facet. Falls back to alphabetical."""
    if "n_params" in df.columns and df["n_params"].notna().any():
        return (df.dropna(subset=["n_params"])
                  .groupby("layers")["n_params"].first()
                  .sort_values().index.tolist())
    return sorted(df["layers"].dropna().unique())


def plot_architecture_faceted(df, target, plot_dir, facet_by="datasetLimitPerLabel"):
    """One architecture boxplot panel per `facet_by` level (default: dataset size).

    Plotting `target` by architecture while averaging over every other setting hides
    the fact that a confound (usually how much data each run saw) drives most of the
    variance. Faceting holds that confound fixed inside each panel, so the *true*
    architecture effect is what's left. The x-axis is ordered by capacity (small ->
    large) and the y-axis is shared, so panels are directly comparable.
    """
    if "layers" not in df.columns or facet_by not in df.columns:
        return
    levels = sorted(df[facet_by].dropna().unique())
    if len(levels) < 2:
        return  # nothing to separate
    arch_order = _arch_order_by_capacity(df)

    n = len(levels)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, sharey=True,
                             figsize=(cols * (0.7 * len(arch_order) + 2.5), rows * 4.6),
                             squeeze=False)
    axes = axes.flatten()
    for i, lvl in enumerate(levels):
        ax = axes[i]
        sub = df[df[facet_by] == lvl]
        sns.boxplot(data=sub, x="layers", y=target, order=arch_order,
                    ax=ax, color="#B2DF8A", fliersize=2)
        means = sub.groupby("layers")[target].mean().reindex(arch_order)
        ax.plot(range(len(arch_order)), means.values, "o-", color="crimson",
                lw=1.4, markersize=4, label="mean")
        ax.set_title(f"{facet_by} = {lvl}   (n={len(sub)})")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"{target} by architecture, held within each {facet_by} level "
                 f"(x-axis: smaller \u2192 larger network)", fontsize=13)
    _save(fig, plot_dir, f"architecture_faceted_by_{facet_by}.png")


def plot_arch_vs_facet_heatmap(df, target, plot_dir, facet_by="datasetLimitPerLabel"):
    """Mean `target` for every architecture x `facet_by` cell on one heatmap.

    Reading across a row (architecture fixed, data varying) vs down a column
    (data fixed, architecture varying) shows at a glance which lever actually moves
    the score. Row range and column range are annotated for exactly that comparison.
    """
    if "layers" not in df.columns or facet_by not in df.columns:
        return
    if df[facet_by].nunique(dropna=True) < 2:
        return
    arch_order = _arch_order_by_capacity(df)
    piv = (df.pivot_table(index="layers", columns=facet_by, values=target,
                          aggfunc="mean").reindex(arch_order))
    # how much each lever moves the mean, on average
    row_span = (piv.max(axis=1) - piv.min(axis=1)).mean()   # vary data, fix arch
    col_span = (piv.max(axis=0) - piv.min(axis=0)).mean()   # vary arch, fix data
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 4, 0.6 * piv.shape[0] + 2))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                cbar_kws={"label": f"mean {target}"})
    ax.set_title(f"mean {target}: architecture (rows) x {facet_by} (cols)\n"
                 f"avg swing from {facet_by}: {row_span:.3f}   |   "
                 f"avg swing from architecture: {col_span:.3f}")
    ax.set_ylabel("architecture (small \u2192 large, top \u2192 bottom)")
    _save(fig, plot_dir, f"architecture_vs_{facet_by}_heatmap.png")
    return row_span, col_span


def plot_interactions(df, factors, target, plot_dir, k=3):
    pairs = []
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            s = interaction_strength(df, factors[i], factors[j], target)
            if not np.isnan(s):
                pairs.append((s, factors[i], factors[j]))
    pairs.sort(reverse=True)
    pairs = pairs[:k]
    if not pairs:
        return pairs
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)
    axes = axes[0]
    for ax, (s, f1, f2) in zip(axes, pairs):
        piv = df.pivot_table(index=f1, columns=f2, values=target, aggfunc="mean")
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar=False)
        ax.set_title(f"{f1} x {f2}\n(interaction {100*s:.1f}% of var)")
    fig.suptitle(f"Strongest two-way interactions for {target}", fontsize=13)
    _save(fig, plot_dir, f"interactions_{target}.png")
    return pairs


def plot_class_difficulty(df, plot_dir):
    cd = class_difficulty(df)
    if cd.empty:
        return cd
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cd["class"].astype(str), cd["mean_acc"], yerr=cd["std_acc"],
           color="#FB9A99", capsize=3)
    ax.set_xlabel("class")
    ax.set_ylabel("mean test accuracy across all runs")
    ax.set_title("Per-class difficulty (lower = harder, averaged over every run)")
    _save(fig, plot_dir, "class_difficulty.png")
    return cd


def plot_class_heatmap_top(df, target, plot_dir, top_n):
    cols = sorted([c for c in df.columns if c.startswith("classacc_")],
                  key=lambda c: int(c.split("_")[1]))
    if not cols:
        return
    top = df.dropna(subset=[target]).nlargest(top_n, target)
    mat = top[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 3, 0.5 * len(top) + 2))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                xticklabels=[c.split("_")[1] for c in cols],
                yticklabels=top["run_id"].tolist(), ax=ax)
    ax.set_xlabel("class")
    ax.set_title(f"Per-class test accuracy of the top {len(top)} runs (by {target})")
    _save(fig, plot_dir, "class_accuracy_top_runs.png")


def plot_learning_curves(df, traj, target, plot_dir, top_n):
    top = df.dropna(subset=[target]).nlargest(top_n, target)
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab10
    for n, (_, row) in enumerate(top.iterrows()):
        rid = row["run_id"]
        test_traj, final_pt = traj.get(rid, (None, None))
        col = cmap(n % 10)
        # train curve
        tpath = _resolve_log(rid, CFG, "train")
        if tpath:
            tr = pd.DataFrame(_read_jsonl(tpath))
            if "epochNumber" in tr and "bestTrainAccuracy" in tr:
                tr = tr.sort_values("epochNumber")
                ax.plot(tr["epochNumber"], tr["bestTrainAccuracy"],
                        color=col, lw=1.2, alpha=0.5)
        # test checkpoints
        if test_traj:
            xs = [e for e, _ in test_traj]
            ys = [m for _, m in test_traj]
            ax.plot(xs, ys, "o--", color=col, lw=1.6,
                    label=f"{rid} (test={row[target]:.3f})")
            if final_pt is not None and xs:
                ax.scatter([max(xs)], [final_pt], color=col, marker="*", s=120, zorder=5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Top {len(top)} runs: train (faint solid) vs test (dashed, ★=final)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _save(fig, plot_dir, "learning_curves_top.png")


def plot_correlations(df, target, plot_dir):
    num = df.select_dtypes("number")
    keep = [c for c in num.columns if df[c].nunique() > 1 and not c.startswith("classacc_")]
    if target not in keep or len(keep) < 3:
        return
    corr = num[keep].corr()
    order = corr[target].abs().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(1.0 * len(order) + 2, 0.9 * len(order) + 1))
    sns.heatmap(corr.loc[order, order], annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation between numeric metrics")
    _save(fig, plot_dir, "correlation_heatmap.png")


# ==================================================================================
# 4. REPORT
# ==================================================================================
def _md_table(df, floatfmt=".4f"):
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.to_string(index=False)


def build_report(df, traj, target, factors, cfg, importances, interactions, cd):
    L = []
    add = L.append
    n = len(df)
    n_test = df["test_acc"].notna().sum() if "test_acc" in df else 0

    add(f"# Hyperparameter sweep analysis\n")
    add(f"- Runs in summary: **{n}**")
    add(f"- Runs with a test log parsed: **{n_test}**")
    add(f"- Primary metric: **`{target}`**")
    if target == "test_acc":
        add("  (final-model accuracy on the held-out test set — the metric you should "
            "actually optimise).")
    add("")

    base_train = "train_best_acc" if "train_best_acc" in df else "best_train_accuracy"
    if "test_acc" in df and base_train in df:
        gap = df["gen_gap"].mean() if "gen_gap" in df else np.nan
        add("## Train vs test (overfitting)\n")
        add(f"- Mean train accuracy: **{df[base_train].mean():.4f}**")
        add(f"- Mean test accuracy:  **{df['test_acc'].mean():.4f}**")
        add(f"- Mean generalisation gap (train − test): **{gap:.4f}**")
        if "overfit_drop" in df:
            add(f"- Mean drop from best-checkpoint test acc to final test acc: "
                f"**{df['overfit_drop'].mean():.4f}** "
                f"(large values ⇒ training past the test-accuracy peak).")
        add("")

    add("## Best configurations (by test accuracy)\n")
    show = [c for c in ["run_id", target, base_train, "gen_gap"] + factors if c in df.columns]
    top = df.dropna(subset=[target]).nlargest(10, target)[show]
    add(_md_table(top))
    add("")

    if not df.dropna(subset=[target]).empty:
        best = df.loc[df[target].idxmax()]
        add("### Headline winner\n")
        for f in factors:
            add(f"- **{f}** = `{best[f]}`")
        add(f"- → test accuracy **{best[target]:.4f}**"
            + (f", train accuracy {best[base_train]:.4f}, gap {best.get('gen_gap', float('nan')):.4f}"
               if base_train in df else ""))
        add("")

    add("## Which hyperparameters matter (variance explained)\n")
    add("Computed on your balanced grid, so the factors are unconfounded and the "
        "percentages are directly comparable.\n")
    add(_md_table(importances.round(4)))
    add("")
    if not importances.empty:
        top_f = importances.iloc[0]
        add(f"➡ **{top_f['factor']}** explains the most variance "
            f"({top_f['var_explained_%']:.1f}%); its best level is "
            f"`{top_f['best_level']}` (mean {target} {top_f['best_level_mean']:.4f}).\n")

    add("## Best setting for each hyperparameter (marginal)\n")
    rows = []
    for f in factors:
        means = df.groupby(f)[target].mean()
        rows.append({"factor": f, "best_level": means.idxmax(),
                     "mean_at_best": means.max(),
                     "mean_at_worst": means.min(),
                     "delta": means.max() - means.min()})
    add(_md_table(pd.DataFrame(rows).round(4)))
    add("\n*Marginal = averaged over all other settings. Check the interactions below "
        "before treating these as independent.*\n")

    if interactions:
        add("## Strongest interactions\n")
        for s, f1, f2 in interactions:
            add(f"- **{f1} × {f2}**: {100*s:.1f}% of variance "
                f"(the best {f1} depends on {f2}).")
        add("")

    if "gen_gap" in df and df["gen_gap"].notna().any():
        add("## What controls overfitting (drivers of the train−test gap)\n")
        gap_imp = importance_table(df, factors, "gen_gap", maximize=False)
        add(_md_table(gap_imp.round(4)))
        add("\n*Lower gap is better. Look for the factor/level that shrinks it — "
            "typically more dropout, more data, or a smaller network.*\n")

    if not cd.empty:
        add("## Per-class difficulty (averaged across all runs)\n")
        add(_md_table(cd.round(4)))
        hardest = cd.iloc[0]
        add(f"\n➡ Hardest class on average: **{int(hardest['class'])}** "
            f"(mean acc {hardest['mean_acc']:.3f}). Worth inspecting its training images.\n")

    bpa = best_per_architecture(df, target)
    if not bpa.empty:
        add("## Best result achievable by each architecture\n")
        add(_md_table(bpa.round(4)))
        add("")

    add("## Suggested next steps\n")
    add("- Re-centre the grid around the winning levels and **narrow** it; the full "
        "factorial here is very large, so a coarse-then-fine search saves compute.")
    add("- If `overfit_drop` is large, consider early-stopping on **test** accuracy "
        "rather than train accuracy, or lowering `maxEpochs`.")
    add("- Focus extra data/augmentation on the hardest classes identified above.")
    add("")

    Path(cfg["report"]).write_text("\n".join(L))
    print(f"  saved {cfg['report']}")
    return top


# ==================================================================================
# MAIN
# ==================================================================================
CFG = DEFAULTS.copy()   # module-level so plot_learning_curves can resolve logs


def main():
    global CFG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    ap.add_argument("--target", default=None,
                    help="metric to optimise (default: test_acc if available, else best_train_accuracy)")
    args = ap.parse_args()
    CFG = {k: getattr(args, k) for k in DEFAULTS}

    plot_dir = Path(CFG["plot_dir"])
    plot_dir.mkdir(exist_ok=True)

    print("Loading summary ...")
    df = load_summary(CFG["summary_csv"])
    if df.empty:
        print("No data found. Run the experiments first.")
        return

    df = enrich_with_architecture(df)
    print(f"Enriching {len(df)} runs from train/test logs ...")
    df, traj = enrich_runs(df, CFG)

    # choose target
    target = args.target
    if target is None:
        target = "test_acc" if df.get("test_acc", pd.Series(dtype=float)).notna().any() \
                 else "best_train_accuracy"
    if target not in df.columns or df[target].notna().sum() == 0:
        print(f"Target '{target}' has no data; falling back to best_train_accuracy.")
        target = "best_train_accuracy"
    print(f"Primary metric: {target}")

    df.to_csv(CFG["out_csv"], index=False)
    print(f"  saved {CFG['out_csv']}")

    factors = active_factors(df)
    print(f"Varied hyperparameters: {factors}")

    print("\nGenerating plots ...")
    importances = importance_table(df, factors, target)
    interactions = []
    cd = class_difficulty(df)

    def safe(fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            print(f"  [skipped {fn.__name__}: {e}]")
            return None

    safe(plot_importance, df, factors, target, plot_dir)
    safe(plot_marginal_effects, df, factors, target, plot_dir)
    safe(plot_overfitting, df, plot_dir)
    safe(plot_architecture, df, target, plot_dir)
    safe(plot_architecture_faceted, df, target, plot_dir)
    safe(plot_arch_vs_facet_heatmap, df, target, plot_dir)
    interactions = safe(plot_interactions, df, factors, target, plot_dir) or []
    cd = safe(plot_class_difficulty, df, plot_dir)
    cd = cd if cd is not None else pd.DataFrame()
    safe(plot_class_heatmap_top, df, target, plot_dir, CFG["top_n"])
    safe(plot_learning_curves, df, traj, target, plot_dir, CFG["top_n"])
    safe(plot_correlations, df, target, plot_dir)

    print("\nWriting report ...")
    safe(build_report, df, traj, target, factors, CFG, importances, interactions, cd)

    # console headline
    print("\n" + "=" * 60)
    print("HEADLINE FINDINGS")
    print("=" * 60)
    print("\nHyperparameter importance (variance explained %):")
    print(importances.to_string(index=False))
    if not df.dropna(subset=[target]).empty:
        best = df.loc[df[target].idxmax()]
        print(f"\nBest run: {best['run_id']}  ->  {target} = {best[target]:.4f}")
        for f in factors:
            print(f"    {f} = {best[f]}")
    print(f"\nAll outputs in: {plot_dir.resolve().parent}")


if __name__ == "__main__":
    main()
