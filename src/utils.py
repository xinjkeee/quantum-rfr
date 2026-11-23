"""Preprocessing, metrics and independent per-object worker helpers."""
from pathlib import Path
import os
from time import perf_counter

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .qrf import QRF


def available_cpu_count():
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    for quota_path, period_path in [
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ]:
        try:
            if period_path is None:
                quota, period = Path(quota_path).read_text().split()
                if quota == "max":
                    continue
            else:
                quota = Path(quota_path).read_text().strip()
                period = Path(period_path).read_text().strip()
                if int(quota) < 0:
                    continue
            count = min(count, max(1, int(quota) // int(period)))
        except (OSError, ValueError):
            pass
    return max(1, count)


def preprocess(X_train, X_test, target):
    """Fit transforms only on training data, matching the completed experiment."""
    if target == "Y2":
        return X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float)
    if target != "Y1":
        raise ValueError("target must be Y1 or Y2")
    categorical = ["X6", "X8"]
    numeric = [column for column in X_train.columns if column not in categorical]
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    train = np.hstack([scaler.fit_transform(X_train[numeric]),
                       encoder.fit_transform(X_train[categorical])])
    test = np.hstack([scaler.transform(X_test[numeric]),
                      encoder.transform(X_test[categorical])])
    return train, test


def absolute_metrics(reference, prediction):
    error = np.asarray(prediction, dtype=float) - np.asarray(reference, dtype=float)
    return {"MAE": float(np.mean(abs(error))), "MSE": float(np.mean(error ** 2)),
            "RMSE": float(np.sqrt(np.mean(error ** 2)))}


def load_metrics(reference, prediction):
    """Percentage metrics are only for physical load R, never normalized beta."""
    a, b = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    error = abs(a - b)
    den = abs(a) + abs(b)
    smape = float(np.mean(np.divide(error, den, out=np.zeros_like(error), where=den != 0)) * 100)
    return {**absolute_metrics(a, b),
            "MAPE_percent": float(np.mean(error / abs(a)) * 100) if np.all(a != 0) else None,
            "wMAPE_percent": float(error.sum() / abs(a).sum() * 100) if abs(a).sum() else None,
            "sMAPE_paper_percent": smape,
            "sMAPE_factor2_percent": 2 * smape}


_qrf = None


def worker_init(rf_model, y_min, y_max, n_precision_qubits):
    global _qrf
    _qrf = QRF(rf_model, y_min, y_max, n_precision_qubits)


def process_single_sample(args):
    idx, x_sample, y_true, delta, seed, target_number = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, target_number, idx]))
    start = perf_counter()
    y_quantum, repetitions = _qrf.predict(x_sample, delta=delta, rng=rng)
    y_classic = float(_qrf.rf.predict(x_sample.reshape(1, -1))[0])
    span = _qrf.y_max - _qrf.y_min
    beta_exact = (y_classic - _qrf.y_min) / span
    beta_quantum = (y_quantum - _qrf.y_min) / span
    t = 2 ** _qrf.n_amp
    bound = 2 * np.pi * np.sqrt(beta_exact * (1 - beta_exact)) / t + np.pi ** 2 / t ** 2
    return {
        "idx": idx, "y_true": float(y_true), "y_classic": y_classic,
        "y_quantum": float(y_quantum),
        "beta_true": float((y_true - _qrf.y_min) / span),
        "beta_exact": float(beta_exact), "beta_quantum": float(beta_quantum),
        "y_min_train": _qrf.y_min, "y_max_train": _qrf.y_max,
        "bound_beta": float(bound), "bound_load": float(bound * span),
        "within_bound": bool(abs(beta_quantum - beta_exact) <= bound),
        "repetitions": repetitions, "duration_seconds": perf_counter() - start,
    }
