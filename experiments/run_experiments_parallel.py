#!/usr/bin/env python3
"""
Run the headless training many times with different hyperparameters, IN PARALLEL.

Each run uses a unique network file so training always starts from scratch.
A pool of worker threads launches several `MatrixGui_headless` processes at once
(each training process is an external subprocess, so threads are the right tool:
they sit in subprocess.run() releasing the GIL while the C/C++ trainer does the work).

Saves a summary CSV and copies each run's train + test logs for later analysis.

USAGE
-----
  python run_experiments.py                 # parallel over all CPU cores
  python run_experiments.py -j 4            # at most 4 training processes at once
  python run_experiments.py -j 8 --threads-per-job 2   # 8 procs x 2 threads each
  python run_experiments.py --resume        # skip runs already present in the CSV
  python run_experiments.py --binary ./MatrixGui_headless

CHOOSING -j
-----------
* If a single training run currently pins ONE core, the trainer is single-threaded,
  so -j = number of cores (the default) is ideal.
* If a single run already uses many cores, the trainer is multi-threaded; running N
  copies that each grab every core will oversubscribe and may be SLOWER. In that case
  lower -j and/or use --threads-per-job to partition cores (only works if the binary
  honours OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS).
* Each concurrent run also loads its own dataset into RAM, so pick -j with memory in
  mind too (large datasetLimitPerLabel x many workers = a lot of memory).
"""

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------- CONFIGURATION ----------
# Path to the training binary (overridable with --binary).
BINARY = "./MatrixGui_headless"

# Base arguments AFTER the binary (no --network here; it is added per run).
BASE_ARGS = [
    "train",
    "--train-dataset", os.path.expanduser("~/Documents/Personal/fashion-png-numeric/train"),
    "--test-dataset", os.path.expanduser("~/Documents/Personal/fashion-png-numeric/test"),
]

# Default values for the learning config JSON (overridden where listed in PARAM_GRID)
BASE_CONFIG = {
    "datasetLimitPerLabel": 1000,
    "dropoutRate": 0.0,
    "initialLearningRate": 0.001,
    "innerEpochs": 1,
    "learningRateDecay": 0.8,
    "maxEpochs": 100,           # kept constant (as requested)
    "minLearningRate": 0.0001,
    "patience": 5
}

# Hyperparameters to explore – add/remove keys as needed.
PARAM_GRID = {
    "layers": [
        "784,10",
        "784,50,10",
        "784,256,10",
        "784,256,70,10",
        "784,512,128,10",
        "784,256,128,64,10",
        "784,256,500,256,10",
    ],
    # "initialLearningRate": [0.005, 0.01, 0.001],
    # "dropoutRate": [0.0, 0.1, 0.2],
    # "patience": [5, 10],
    # "learningRateDecay": [0.5, 0.8, 0.9],
    # "datasetLimitPerLabel": [100, 500, 1000, 4000]
}

SUMMARY_CSV = "experiment_results.csv"
TRAINING_DATA_DIR = "training-data"
LOG_FILE_NAME = "log.jsonl"
TEST_LOG_FILE_NAME = "testing-log.jsonl"

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# A lock so worker threads can write to the shared CSV safely.
_csv_lock = threading.Lock()
# Per-run subprocess environment (thread caps). Filled in main().
_RUN_ENV = None


# ---------- HELPER FUNCTIONS ----------
def generate_config_file(params, run_id):
    """Create a temporary learning-config JSON with the desired hyperparameters."""
    config = BASE_CONFIG.copy()
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
    cmd = [BINARY] + BASE_ARGS
    cmd += ["--network", network_name]
    cmd += ["--learning-config", config_path]
    if "layers" in params:
        cmd += ["--layers", params["layers"]]
    if "datasetLimitPerLabel" in params:
        cmd += ["--dataset-limit-per-label", str(params["datasetLimitPerLabel"])]
    return cmd, network_name


def find_run_dir(network_name):
    """Return the training-data directory for the given (unique) network name.
    The unique per-run network name means only this run's dir(s) match, so this is
    safe under parallelism. If several exist (e.g. a re-run), use the newest."""
    base = Path(TRAINING_DATA_DIR)
    if not base.exists():
        raise FileNotFoundError(f"'{TRAINING_DATA_DIR}' directory not found")
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(network_name)]
    if not dirs:
        raise FileNotFoundError(f"No directory found starting with '{network_name}'")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def extract_best_metrics(run_dir):
    """Parse the train log and return (best_train_accuracy, best_epoch_number, log_path)."""
    log_path = run_dir / LOG_FILE_NAME
    if not log_path.exists():
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
    """Execute one training run and return a CSV row dict.
    Runs entirely inside a worker thread; touches only per-run unique files."""
    row = {"run_id": run_id}
    row.update(params)
    row["best_train_accuracy"] = ""
    row["best_epoch"] = ""

    config_path = generate_config_file(params, run_id)
    cmd, network_name = build_command(params, config_path, run_id)
    layers = params.get('layers')
    initialLr = params.get('initialLearningRate', BASE_CONFIG['initialLearningRate'])
    drop = params.get('dropoutRate', BASE_CONFIG['dropoutRate'])
    print(f"[{run_id}] start  ({layers}, lr={initialLr}, drop={drop})")

    # Each run's stdout/stderr goes to its own file so parallel output isn't interleaved.
    out_path = LOGS_DIR / f"{run_id}.out"
    t0 = time.time()
    try:
        with open(out_path, "w") as out_f:
            result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT,
                                    env=_RUN_ENV)
        rc = result.returncode
    except Exception as e:
        print(f"[{run_id}] ERROR launching trainer: {e}")
        rc = -999
    t1 = time.time()
    if rc != 0:
        print(f"[{run_id}] WARNING: training exit code {rc} (see {out_path})")

    # Clean up the temporary config
    try:
        os.remove(config_path)
    except OSError:
        pass

    # Locate this run's output directory
    try:
        run_dir = find_run_dir(network_name)
    except Exception as e:
        print(f"[{run_id}] error finding run directory: {e}")
        return row

    # Train metrics + copy train log
    try:
        best_acc, best_ep, log_path = extract_best_metrics(run_dir)
        row["best_train_accuracy"] = best_acc
        row["best_epoch"] = best_ep
        if log_path and log_path.exists():
            shutil.copy2(log_path, LOGS_DIR / f"{run_id}.json")
    except Exception as e:
        print(f"[{run_id}] error extracting train metrics: {e}")
        best_acc, best_ep = None, None

    # Copy test log (for the analyzer's generalisation / per-class analysis)
    test_src = run_dir / TEST_LOG_FILE_NAME
    if test_src.exists():
        shutil.copy2(test_src, LOGS_DIR / f"{run_id}.test.json")

    elapsed = t1 - t0
    acc_str = f"{best_acc:.4f}" if best_acc is not None else "n/a"
    print(f"[{run_id}] done   best_train_acc={acc_str} epoch={best_ep}  ({elapsed:.1f}s)")
    return row


# ---------- MAIN EXPERIMENT LOOP ----------
def build_combinations():
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return keys, combos


def load_done_run_ids(csv_path):
    """Return the set of run_ids already recorded (for --resume)."""
    done = set()
    p = Path(csv_path)
    if not p.exists():
        return done
    try:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                rid = r.get("run_id", "").strip()
                if rid:
                    done.add(rid)
    except Exception:
        pass
    return done


def main(argv=None):
    global BINARY, _RUN_ENV

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="max training processes to run concurrently (default: all cores)")
    ap.add_argument("--threads-per-job", type=int, default=None,
                    help="cap threads per training process via OMP/OPENBLAS/MKL env vars "
                         "(only effective if the binary honours them)")
    ap.add_argument("--resume", action="store_true",
                    help="skip run_ids already present in the summary CSV")
    ap.add_argument("--binary", default=BINARY, help="path to the training binary")
    ap.add_argument("--summary-csv", default=SUMMARY_CSV)
    args = ap.parse_args(argv)

    BINARY = args.binary
    summary_csv = args.summary_csv
    jobs = max(1, args.jobs or 1)

    # Build the per-subprocess environment (optional thread cap)
    _RUN_ENV = os.environ.copy()
    if args.threads_per_job:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            _RUN_ENV[var] = str(args.threads_per_job)

    keys, combos = build_combinations()
    fieldnames = ["run_id"] + keys + ["best_train_accuracy", "best_epoch"]

    done = load_done_run_ids(summary_csv) if args.resume else set()
    todo = [(f"run_{i:04d}", p) for i, p in enumerate(combos)
            if f"run_{i:04d}" not in done]

    print(f"Total experiments: {len(combos)}   already done: {len(done)}   "
          f"to run: {len(todo)}")
    print(f"Parallelism: {jobs} concurrent process(es)"
          + (f", {args.threads_per_job} thread(s) each" if args.threads_per_job else "")
          + f"  (detected {os.cpu_count()} cores)")
    if not todo:
        print("Nothing to do.")
        return

    # Append when resuming an existing CSV, otherwise start fresh with a header.
    new_file = not (args.resume and Path(summary_csv).exists())
    mode = "w" if new_file else "a"

    t_start = time.time()
    completed = 0
    with open(summary_csv, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
            csvfile.flush()

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(run_single_experiment, p, rid): rid for rid, p in todo}
            try:
                for fut in as_completed(futures):
                    rid = futures[fut]
                    try:
                        row = fut.result()
                    except Exception as e:
                        print(f"[{rid}] crashed: {e}")
                        row = {"run_id": rid, "best_train_accuracy": "", "best_epoch": ""}
                    # Only the main thread writes to the CSV; lock is belt-and-suspenders.
                    with _csv_lock:
                        writer.writerow(row)
                        csvfile.flush()
                    completed += 1
                    if completed % max(1, len(todo) // 100 or 1) == 0 or completed == len(todo):
                        rate = completed / (time.time() - t_start)
                        eta = (len(todo) - completed) / rate if rate > 0 else 0
                        print(f"  progress: {completed}/{len(todo)}  "
                              f"({100*completed/len(todo):.1f}%)  eta ~{eta/60:.1f} min")
            except KeyboardInterrupt:
                print("\nInterrupted — cancelling pending runs "
                      "(in-flight ones finish). Re-run with --resume to continue.")
                ex.shutdown(wait=False, cancel_futures=True)
                raise

    print(f"All done. {completed} runs written to {summary_csv} "
          f"in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
