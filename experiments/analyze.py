#!/usr/bin/env python3
"""
Analyse the experiment results.
Reads the summary CSV and per‑run log files to generate plots and statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# ---------- CONFIGURATION ----------
SUMMARY_CSV = "experiment_results.csv"
LOGS_DIR = Path("logs")               # where run_experiments.py saved the per‑run logs
PLOT_DIR = Path("analysis_plots")
PLOT_DIR.mkdir(exist_ok=True)

# Parameters to plot individually
HYPERPARAMS = ["initialLearningRate", "dropoutRate", "patience",
               "learningRateDecay", "datasetLimitPerLabel"]

# ---------- DATA LOADING ----------
def load_summary(csv_path):
    df = pd.read_csv(csv_path)
    # Convert numeric columns (coerce errors to NaN)
    numeric_cols = ["initialLearningRate", "dropoutRate", "patience",
                    "learningRateDecay", "datasetLimitPerLabel",
                    "best_train_accuracy", "best_epoch"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_epoch_log(run_id):
    """Load the epoch log for a single run. Returns lists of epochs and accuracies."""
    log_file = LOGS_DIR / f"{run_id}.json"
    if not log_file.exists():
        return None, None
    epochs = []
    accs = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            epochs.append(data["epochNumber"])
            accs.append(data["bestTrainAccuracy"])
    return epochs, accs

# ---------- PLOTTING FUNCTIONS ----------
def plot_accuracy_vs_hyperparams(df):
    """Stripplots of best accuracy vs each hyperparameter."""
    n = len(HYPERPARAMS)
    # Arrange subplots: 2 rows, flexible columns
    cols = min(3, n)
    rows = -(-n // cols)  # ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, hp in enumerate(HYPERPARAMS):
        ax = axes[i]
        if hp in df.columns:
            sns.stripplot(data=df, x=hp, y="best_train_accuracy", ax=ax, jitter=True)
            ax.set_title(f"Best accuracy vs {hp}")
            ax.tick_params(axis='x', rotation=30)
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Best train accuracy vs hyperparameters", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "accuracy_vs_hyperparams.png", dpi=150)
    plt.close()
    print("Saved accuracy_vs_hyperparams.png")

def plot_accuracy_by_layers(df):
    """Bar plot of mean best accuracy per network architecture."""
    if "layers" not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    # Group and order by mean accuracy
    means = df.groupby("layers")["best_train_accuracy"].mean().sort_values()
    sns.barplot(x=means.index, y=means.values, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Mean best accuracy by network architecture")
    plt.xlabel("Layer configuration")
    plt.ylabel("Mean best accuracy")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "accuracy_by_layers.png", dpi=150)
    plt.close()
    print("Saved accuracy_by_layers.png")

def plot_learning_curves(df, top_n=5):
    """
    For the top N runs (by best_train_accuracy), plot their learning curves
    (bestTrainAccuracy vs epoch) together with a legend showing hyperparameters.
    """
    top_runs = df.nlargest(top_n, "best_train_accuracy")
    plt.figure(figsize=(12, 7))

    for _, row in top_runs.iterrows():
        run_id = row["run_id"]
        epochs, accs = load_epoch_log(run_id)
        if epochs is None:
            print(f"Warning: no log found for {run_id}, skipping.")
            continue
        # Build a short label
        label_parts = []
        for hp in HYPERPARAMS + ["layers"]:
            if hp in row and not pd.isna(row[hp]):
                val = row[hp]
                if isinstance(val, float):
                    val = f"{val:.4g}" if hp != "layers" else val
                label_parts.append(f"{hp}={val}")
        label = ", ".join(label_parts[:5])  # keep label manageable
        plt.plot(epochs, accs, label=f"{run_id}: {label}", linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel("Best Train Accuracy")
    plt.title(f"Learning curves of top {top_n} configurations")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "learning_curves_top.png", dpi=150)
    plt.close()
    print(f"Saved learning_curves_top.png (top {top_n})")

def print_statistics(df):
    """Print a statistical summary and best configuration."""
    print("\n===== Summary Statistics =====")
    print(df[["best_train_accuracy", "best_epoch"]].describe())
    print("\n===== Best Configuration =====")
    best = df.loc[df["best_train_accuracy"].idxmax()]
    for col in df.columns:
        print(f"  {col}: {best[col]}")
    print(f"  Best accuracy: {best['best_train_accuracy']:.4f}")

# ---------- MAIN ----------
def main():
    df = load_summary(SUMMARY_CSV)
    if df.empty:
        print("No data found. Make sure the experiments have been run and the CSV exists.")
        return

    plot_accuracy_vs_hyperparams(df)
    plot_accuracy_by_layers(df)
    plot_learning_curves(df, top_n=5)   # you can change the number here

    print_statistics(df)
    print(f"\nAll plots saved to {PLOT_DIR.resolve()}")

if __name__ == "__main__":
    main()
