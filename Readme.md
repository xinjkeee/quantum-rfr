# Quantum Random Forest for Regression

This repository contains a Quantum Random Forest (QRF) implementation for regression. The classical Random Forest is trained with scikit-learn; PennyLane is used to simulate the Quantum Amplitude Estimation (QAE) aggregation step.

## Repository contents

- `main.py` — experiment entry point;
- `src/qrf.py` — QRF circuit and QAE prediction logic;
- `src/utils.py` — multiprocessing helpers;
- `data/ENB2012_data.csv` — Energy Efficiency dataset;
- `requirements.txt` — Python dependencies.

## Dataset

The ENB2012 dataset contains 768 observations, 8 input features, and two targets:

- `Y1` — heating load;
- `Y2` — cooling load.

The standard experiment uses an 80/20 split: 614 training and 154 test observations, with `random_state=42`.

## Classical Random Forest configuration

- `n_estimators=128`;
- `max_depth=6`;
- `criterion='squared_error'`;
- `random_state=42`;
- bootstrap sampling enabled by the scikit-learn default.

The quantum model uses the same trained forest. QAE is applied during inference to approximate the ensemble prediction.

## Quantum simulation

The implementation uses:

- PennyLane;
- `pennylane-lightning`;
- `lightning.qubit` CPU state-vector simulation;

The default `N_PRECISION_QUBITS=9` is the QAE precision register size: `t=512` grid points and 511 controlled Grover applications. The current implementation uses the compact circuit without a tree-path register: 7 tree-index qubits, 1 output qubit and 9 precision qubits, 17 in total. Tree traversal and state-preparation inputs are computed classically; the circuit explicitly enumerates all trees.

For Y1, OneHotEncoder is fitted on `X6`/`X8`, and StandardScaler is fitted on the remaining numeric features. For Y2, neither transformation is applied. All transformers are fitted on training data only.

## Reproducibility

The dataset split and classical forest use seed `42`. Each prediction uses `default_rng(SeedSequence([42, target_number, test_index]))`, independently of worker count and scheduling. Median amplification uses 7 samples for `delta=0.05`; delta is a failure probability, not a 5% forecasting-error guarantee. The simulator computes the probability distribution once and samples it repeatedly. Dependencies are pinned in `requirements.txt`; Python 3.10–3.12 is recommended. Generated results, logs, virtual environments, server scripts, and figures are excluded from Git.

## Hardware used for the latest experiment

The VAST instance used for the latest experiment was:

- CPU: AMD EPYC 7B12 64-Core Processor;
- allocation shown by VAST: `64/128 CPU`;
- RAM shown by VAST: `64.3 GB`;
- GPU: NVIDIA GeForce RTX 5080;
- VRAM shown by VAST: `15.9 GB`;

The RTX 5080 was not used: the selected `lightning.qubit` backend
performs CPU state-vector simulation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This runs both targets on all 154 test objects per target. The default worker count is half the CPU quota/affinity, with one thread per process; adjust `--jobs` for available RAM. Full m=9 simulations can take hours.

```bash
# One test object (still potentially slow at m=9)
python main.py --targets Y1 --samples 1 --jobs 1 --out results/one

# Explicit parallelism / precision; choose a new output directory per experiment
python main.py --precision-qubits 9 --jobs 30 --out results/m9
```

The minimal runner writes prediction CSVs and metric/configuration JSONs without overwriting existing results. The separate experimental ZIP contains server-specific checkpoint/resume, heartbeat and report utilities; these are not part of this minimal repository.

## Citation and limitations

This implementation is a hybrid CPU simulation intended to study QAE-based aggregation of a classical Random Forest. Its simulator wall time should not be interpreted as a hardware quantum speedup. Percentage metrics (MAPE, wMAPE and sMAPE) are evaluated only on the physical load R. Normalized beta is evaluated with MAE/MSE/RMSE; beta_exact is the normalized classical forecast, while beta_true is the normalized true load. Both sMAPE conventions are explicitly named in the JSON output. A single simulation seed does not establish statistical superiority over the classical forest.
