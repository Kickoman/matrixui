#!/usr/bin/env python3
"""
Run the headless training multiple times with different hyperparameters.
Each run uses a unique network file so training always starts from scratch.
Saves a summary CSV and copies each run’s epoch log for later analysis.
"""

import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import csv

# ---------- CONFIGURATION ----------
# Base command – NOTE: no --network flag here, it will be added per run
BASE_CMD = [
    "./MatrixGui_headless", "train",
    "--train-dataset", os.path.expanduser("~/Documents/projects/mnist-pngs/train"),
    "--test-dataset", os.path.expanduser("~/Documents/projects/mnist-pngs/test"),
]

# Default values for the learning config JSON (will be overridden where listed in PARAM_GRID)
BASE_CONFIG = {
    "datasetLimitPerLabel": 1000,
    "dropoutRate": 0.1,
    "initialLearningRate": 0.005,
    "innerEpochs": 1,
    "learningRateDecay": 0.8,
    "maxEpochs": 500,           # kept constant (as requested)
    "minLearningRate": 0.0001,
    "patience": 5
}

# Hyperparameters to explore – add/remove keys as needed.
# "layers" and "datasetLimitPerLabel" are command‑line overrides.
PARAM_GRID = {
    "layers": [
        "784,10",
        "784,50,10",
        "784,100,10",
        "784,256,10",
        "784,256,70,10",
        "784,512,128,10",
        "784,256,128,64,10"
    ],
    "initialLearningRate": [0.005, 0.01, 0.001],
    "dropoutRate": [0.0, 0.1, 0.2],
    "patience": [5, 10],
    "learningRateDecay": [0.5, 0.8, 0.9],
    "datasetLimitPerLabel": [100, 500, 1000, 4000]
}

# Output summary
SUMMARY_CSV = "experiment_results.csv"

# Directory where the training program puts its results
TRAINING_DATA_DIR = "training-data"

# Name of the epoch log file inside each run directory (adjust to match your actual filename)
LOG_FILE_NAME = "log.jsonl"   # or "log.json" etc.

# Where to store per‑run logs for later analysis
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ---------- HELPER FUNCTIONS ----------
def generate_config_file(params, run_id):
    """Create a temporary learning-config JSON with the desired hyperparameters."""
    config = BASE_CONFIG.copy()
    # Override with any key that belongs in the JSON
    for k in ["initialLearningRate", "dropoutRate", "patience",
              "learningRateDecay", "datasetLimitPerLabel", "minLearningRate",
              "innerEpochs"]:
        if k in params:
            config[k] = params[k]
    config_path = f"temp_config_{run_id}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path

def build_command(params, config_path, run_id):
    """Assemble the full command line with a unique network name."""
    network_name = f"neuro_{run_id}.wgt"
    cmd = BASE_CMD.copy()
    # Insert the unique network name
    cmd += ["--network", network_name]
    cmd += ["--learning-config", config_path]
    if "layers" in params:
        cmd += ["--layers", params["layers"]]
    if "datasetLimitPerLabel" in params:
        cmd += ["--dataset-limit-per-label", str(params["datasetLimitPerLabel"])]
    return cmd, network_name

def find_run_dir(network_name):
    """
    Return the training-data directory for the given network name.
    Assumes exactly one directory starting with that name exists.
    """
    base = Path(TRAINING_DATA_DIR)
    if not base.exists():
        raise FileNotFoundError(f"'{TRAINING_DATA_DIR}' directory not found")
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(network_name)]
    if not dirs:
        raise FileNotFoundError(f"No directory found starting with '{network_name}'")
    if len(dirs) > 1:
        # If multiple exist (unlikely due to unique naming), use the latest
        return max(dirs, key=lambda d: d.stat().st_mtime)
    return dirs[0]

def extract_best_metrics(run_dir):
    """Parse the epoch log and return (best_train_accuracy, best_epoch_number)."""
    log_path = run_dir / LOG_FILE_NAME
    if not log_path.exists():
        # Try to find any JSON file
        json_files = list(run_dir.glob("*.json"))
        if json_files:
            log_path = json_files[0]
        else:
            raise FileNotFoundError(f"No log file found in {run_dir}")
    best_acc = 0.0
    best_epoch = -1
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            acc = data.get("bestTrainAccuracy")
            ep = data.get("epochNumber")
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_epoch = ep
    return best_acc, best_epoch, log_path

def run_single_experiment(params, run_id):
    """Execute one training run and return (best_accuracy, best_epoch)."""
    config_path = generate_config_file(params, run_id)
    cmd, network_name = build_command(params, config_path, run_id)
    print(f"[{run_id}] Starting: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    t1 = time.time()
    if result.returncode != 0:
        print(f"[{run_id}] WARNING: training returned exit code {result.returncode}")
    # Clean up temporary config
    try:
        os.remove(config_path)
    except OSError:
        pass

    # Locate the run directory for this network name
    try:
        run_dir = find_run_dir(network_name)
    except Exception as e:
        print(f"[{run_id}] Error finding run directory: {e}")
        return None, None

    try:
        best_acc, best_ep, log_path = extract_best_metrics(run_dir)
    except Exception as e:
        print(f"[{run_id}] Error extracting metrics: {e}")
        best_acc, best_ep = None, None
        log_path = None

    # Copy the log for later analysis
    if log_path and log_path.exists():
        dest = LOGS_DIR / f"{run_id}.json"
        shutil.copy2(log_path, dest)
        print(f"[{run_id}] Log saved to {dest}")

    test_src = run_dir / "testing-log.jsonl"
    if test_src.exists():
        shutil.copy2(test_src, LOGS_DIR / f"{run_id}.test.json")

    elapsed = t1 - t0
    print(f"[{run_id}] Best accuracy: {best_acc:.4f} at epoch {best_ep}   ({elapsed:.1f}s)")
    return best_acc, best_ep

# ---------- MAIN EXPERIMENT LOOP ----------
def main():
    # Build all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    print(f"Total experiments: {len(combinations)}")

    fieldnames = ["run_id"] + keys + ["best_train_accuracy", "best_epoch"]
    with open(SUMMARY_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, params in enumerate(combinations):
            run_id = f"run_{i:04d}"
            # Small delay to ensure distinct directory timestamps (optional)
            time.sleep(1)
            best_acc, best_ep = run_single_experiment(params, run_id)
            row = {"run_id": run_id}
            row.update(params)
            row["best_train_accuracy"] = best_acc if best_acc is not None else ""
            row["best_epoch"] = best_ep if best_ep is not None else ""
            writer.writerow(row)
            csvfile.flush()   # ensure data persists if the script is interrupted

    print(f"All done. Summary written to {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
