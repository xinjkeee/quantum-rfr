import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.qrf import QRF, repetitions_for_delta
from src.utils import (absolute_metrics, available_cpu_count, load_metrics,
                       preprocess, process_single_sample, worker_init)

N_PRECISION_QUBITS = 9
SEED = 42
DELTA = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision-qubits", type=int, default=N_PRECISION_QUBITS)
    parser.add_argument("--jobs", type=int, default=max(1, available_cpu_count() // 2))
    parser.add_argument("--samples", type=int, default=154, help="First N test objects per target")
    parser.add_argument("--targets", nargs="+", choices=["Y1", "Y2"], default=["Y1", "Y2"])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if not 1 <= args.precision_qubits <= 12 or args.jobs < 1 or not 1 <= args.samples <= 154:
        parser.error("precision-qubits must be 1..12, jobs >=1 and samples 1..154")
    if len(args.targets) != len(set(args.targets)):
        parser.error("Duplicate targets")
    return args


def main(args):
    data = pd.read_csv(Path(__file__).resolve().parent / "data/ENB2012_data.csv")
    if len(data) != 768 or data.isna().any().any():
        raise ValueError("Expected complete ENB2012 dataset with 768 observations")
    X = data.drop(columns=["Y1", "Y2"])
    args.out.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")

    for target in args.targets:
        prefix = f"qrf_{target}_m{args.precision_qubits}"
        csv_path = args.out / f"{prefix}_predictions.csv"
        json_path = args.out / f"{prefix}_metrics.json"
        if csv_path.exists() or json_path.exists():
            raise FileExistsError(f"Results for {prefix} already exist. Use a new --out directory.")
        print(f"{target}: split/preprocessing/training RF", flush=True)
        a, b, ya, yb = train_test_split(X, data[target], train_size=0.8, random_state=SEED)
        a, b = preprocess(a, b, target)
        rf = RandomForestRegressor(n_estimators=128, max_depth=6, criterion="squared_error",
                                   random_state=SEED, n_jobs=1).fit(a, ya)
        low, high = float(ya.min()), float(ya.max())
        qrf = QRF(rf, low, high, args.precision_qubits)
        workers = min(args.jobs, args.samples)
        print(f"{target}: qubits={qrf.total_wires}, m={qrf.n_amp}, "
              f"controlled Grover calls={2 ** qrf.n_amp - 1}, workers={workers}, "
              f"median repetitions={repetitions_for_delta(DELTA)}", flush=True)
        tasks = [(i, b[i], float(yb.iloc[i]), DELTA, SEED, int(target[-1]))
                 for i in range(args.samples)]
        rows = []
        started = perf_counter()
        with context.Pool(processes=workers, initializer=worker_init,
                          initargs=(rf, low, high, args.precision_qubits)) as pool:
            for row in pool.imap_unordered(process_single_sample, tasks, chunksize=1):
                rows.append(row)
                print(f"{target}: {len(rows)}/{len(tasks)} finished; "
                      f"elapsed={perf_counter() - started:.1f}s", flush=True)
        wall = perf_counter() - started
        frame = pd.DataFrame(rows).sort_values("idx")
        metrics = {
            "R_classic_vs_true": load_metrics(frame.y_true, frame.y_classic),
            "R_quantum_vs_true": load_metrics(frame.y_true, frame.y_quantum),
            "R_quantum_vs_classic": load_metrics(frame.y_classic, frame.y_quantum),
            "beta_classic_vs_true": absolute_metrics(frame.beta_true, frame.beta_exact),
            "beta_quantum_vs_true": absolute_metrics(frame.beta_true, frame.beta_quantum),
            "beta_quantum_vs_classic": absolute_metrics(frame.beta_exact, frame.beta_quantum),
        }
        summary = {
            "target": target, "samples": args.samples, "train_samples": len(ya),
            "test_samples": len(yb), "seed": SEED, "delta": DELTA,
            "precision_qubits": args.precision_qubits, "total_qubits": qrf.total_wires,
            "register_mode": "compact_without_path_register", "backend": "lightning.qubit",
            "repetitions": repetitions_for_delta(DELTA), "workers": workers,
            "rf_parameters": rf.get_params(), "y_min_train": low, "y_max_train": high,
            "preprocessing": {"onehot": ["X6", "X8"] if target == "Y1" else [],
                              "scaler": "standard_numeric_only" if target == "Y1" else "none"},
            "wall_seconds": wall, "mean_object_seconds": float(frame.duration_seconds.mean()),
            "within_bound_fraction_empirical": float(frame.within_bound.mean()),
            "metrics": metrics,
        }
        frame.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        print(f"Saved {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main(parse_args())
