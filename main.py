import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from time import time
import multiprocessing as mp

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.qrf import QRF
from src.utils import worker_init, process_single_sample

if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)

    df = pd.read_csv("./data/ENB2012_data.csv")
    np.random.seed(42)

    for target in ['Y1', 'Y2']:
        X = df.drop(columns=['Y1', 'Y2'])
        y = df[target] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rf = RandomForestRegressor(
            n_estimators=128,
            max_depth=6,
            criterion='squared_error',
            random_state=42
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        print("R2:", r2_score(y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
        print("MAE:", mean_absolute_error(y_test, y_pred))

        Y_MIN = np.min(y_train)
        Y_MAX = np.max(y_train)
        Y_RANGE = Y_MAX - Y_MIN
        DELTA = 0.05
        
        NUM_SAMPLES = len(X_test)
        N_PRECISION_QUBITS = 5
        N_JOBS = 40 #adjust to the number of cores
        
        print(f"Parallel jobs: {N_JOBS}")

        temp_qrf = QRF(rf, Y_MIN, Y_MAX, N_PRECISION_QUBITS)
        ctx = mp.get_context("spawn")

        t = 2 ** N_PRECISION_QUBITS

        print(f"trees: {temp_qrf.n_trees}, depth: {temp_qrf.tree_depth}, qbits: {temp_qrf.total_wires}, grover steps: {t}") 

        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_test), NUM_SAMPLES, replace=False)
        
        task_args =[
            (int(idx), X_test[idx], y_test.iloc[idx], DELTA)
            for idx in indices
        ]

        time_start = time()

        with ctx.Pool(
            processes=N_JOBS,
            initializer=worker_init,
            initargs=(rf, Y_MIN, Y_MAX, N_PRECISION_QUBITS)
        ) as pool:
            results = pool.map(process_single_sample, task_args)
        
        total_time = time() - time_start

        avg_sample_time = np.mean([res['duration'] for res in results])
        
        print(f"\nFull time: {total_time:.2f}s")
        print(f"Average time for 1 sample: {avg_sample_time:.2f}s")
        print(f"Parallel speed-up: {total_time / NUM_SAMPLES:.2f}s/sample\n")

        y_trues, y_classics, y_quantums = [], [],[]
        beta_exacts, beta_quantums = [],[]

        results.sort(key=lambda x: x['idx'])

        for res in results:
            y_trues.append(res['y_true'])
            y_classics.append(res['y_classic'])
            y_quantums.append(res['y_quantum'])
            beta_exacts.append(res['beta_exact'])
            beta_quantums.append(res['beta_quantum'])

        y_trues = np.array(y_trues)
        y_classics = np.array(y_classics)
        y_quantums = np.array(y_quantums)
        beta_exacts = np.array(beta_exacts)
        beta_quantums = np.array(beta_quantums)

        mae_qt = np.mean(np.abs(y_quantums - y_trues))
        rmse_qt = np.sqrt(np.mean((y_quantums - y_trues) ** 2))
        mae_ct = np.mean(np.abs(y_classics - y_trues))
        rmse_ct = np.sqrt(np.mean((y_classics - y_trues) ** 2))
        mae_qc = np.mean(np.abs(y_quantums - y_classics))
        mae_beta = np.mean(np.abs(beta_quantums - beta_exacts))

        mape_qt = np.mean(np.abs((y_quantums - y_trues) / y_trues)) * 100
        mape_ct = np.mean(np.abs((y_classics - y_trues) / y_trues)) * 100
        mape_qc = np.mean(np.abs((y_quantums - y_classics) / y_classics)) * 100

        print(f"{'':<20} | {'Classic vs True':<18} | {'Quantum vs True':<18} | {'Quantum vs Classic':<18}")
        print("-" * 83)
        print(f"{'MAE':<20} | {mae_ct:<18.4f} | {mae_qt:<18.4f} | {mae_qc:<18.4f}")
        print(f"{'RMSE':<20} | {rmse_ct:<18.4f} | {rmse_qt:<18.4f} | {'—':<18}")
        print(f"{'MAPE (%)':<20} | {mape_ct:<18.2f} | {mape_qt:<18.2f} | {mape_qc:<18.2f}")
        print(f"\nlemma2 bound |Δβ| for t={t}: {2*np.pi*0.5/t + np.pi**2/t**2:.6f} (at β=0.5)")
        print(f"actual |Δβ|: {mae_beta:.6f}\n")

        df_results = pd.DataFrame(results)
        
        csv_filename = os.path.join("results", f"qrf_results_{target}_qbits{N_PRECISION_QUBITS}_trees{rf.n_estimators}_depth{rf.max_depth}.csv")
        df_results.to_csv(csv_filename, index=False)

        txt_filename = os.path.join("results", f"qrf_metrics_{target}_qbits{N_PRECISION_QUBITS}_trees{rf.n_estimators}.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"QRF Configuration:\n")
            f.write(f"Target: {target}\n")
            f.write(f"Trees: {temp_qrf.n_trees}, Depth: {temp_qrf.tree_depth}, Qubits: {temp_qrf.total_wires}, Grover steps: {t}\n")
            f.write(f"Samples tested: {NUM_SAMPLES}\n")
            f.write(f"Parallel: {N_JOBS}\n")
            f.write(f"Total Wall Time: {total_time:.2f}s\n")
            f.write(f"Avg Compute Time per sample: {avg_sample_time:.2f}s\n\n")
            
            header_file = f"{'Metric':<20} | {'Classic vs True':<18} | {'Quantum vs True':<18} | {'Quantum vs Classic':<18}\n"
            f.write(header_file)
            f.write("-" * 81 + "\n")
            f.write(f"{'MAE':<20} | {mae_ct:<18.4f} | {mae_qt:<18.4f} | {mae_qc:<18.4f}\n")
            f.write(f"{'RMSE':<20} | {rmse_ct:<18.4f} | {rmse_qt:<18.4f} | {'—':<18}\n")
            f.write(f"{'MAPE (%)':<20} | {mape_ct:<18.2f} | {mape_qt:<18.2f} | {mape_qc:<18.2f}\n\n")
            
            f.write(f"Lemma 2 Bound |Δβ| for t={t}: {2*np.pi*0.5/t + np.pi**2/t**2:.6f} (at β=0.5)\n")
            f.write(f"actual |Δβ|: {mae_beta:.6f}\n")