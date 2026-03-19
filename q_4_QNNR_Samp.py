"""
QNNR - Quantum Neural Network Regressor
(Sampler)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED
 
CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
NUM_LAYERS = 2
MAXITER = 200


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def parity(bitstring_int, n_qubits):
    return bin(bitstring_int).count("1") % 2


class QuantumSamplerRegressor:
    def __init__(self):
        self.feature_map = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)
        self.ansatz = TwoLocal(
            num_qubits=NUM_QUBITS,
            rotation_blocks='ry',
            entanglement_blocks='cz',
            entanglement='linear',
            reps=NUM_LAYERS
        )
        self.circuit = self.feature_map.compose(self.ansatz)
        self.num_params = len(self.ansatz.parameters)
        self.theta = np.zeros(self.num_params, dtype=float)

    def _predict_single(self, x, theta):
        param_dict = {}
        for p, val in zip(self.feature_map.parameters, x):
            param_dict[p] = float(val)
        for p, val in zip(self.ansatz.parameters, theta):
            param_dict[p] = float(val)
        bound = self.circuit.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound)
        probs = sv.probabilities()
        parity_1_prob = sum(
            p for i, p in enumerate(probs)
            if bin(i).count("1") % 2 == 1
        )
        return float(parity_1_prob)

    def predict(self, X):
        return np.array([self._predict_single(x, self.theta) for x in X])

    def fit(self, X, y):
        def loss(theta):
            preds = np.array([self._predict_single(x, theta) for x in X])
            return float(np.mean((preds - y) ** 2))

        res = scipy_minimize(loss, self.theta, method='COBYLA',
                             options={'maxiter': MAXITER, 'rhobeg': 0.3})
        self.theta = res.x
        return res.fun


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS
    X_all = np.array([value_to_features(v) for v in range(n_states)])

    print(f"\n--- QNNR SamplerQNN ({NUM_QUBITS}q, parity, "
          f"COBYLA {MAXITER} iter) ---")
    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        y = build_empirical(draws, pos)

        qreg = QuantumSamplerRegressor()
        final_loss = qreg.fit(X_all, y)

        pred = qreg.predict(X_all)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"loss={final_loss:.6f}  top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QNNR-Sampler, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- QNNR SamplerQNN (5q, parity, COBYLA 200 iter) ---
  Poz 1... loss=0.138300  top: 4:0.065 | 7:0.064 | 25:0.062
  Poz 2... loss=0.135933  top: 31:0.067 | 28:0.055 | 32:0.052
  Poz 3... loss=0.133539  top: 32:0.080 | 33:0.060 | 29:0.054
  Poz 4... loss=0.134424  top: 33:0.076 | 30:0.065 | 34:0.059
  Poz 5... loss=0.136206  top: 31:0.065 | 34:0.061 | 17:0.060
  Poz 6... loss=0.134705  top: 35:0.072 | 32:0.066 | 18:0.056
  Poz 7... loss=0.140747  top: 35:0.065 | 19:0.061 | 33:0.059

==================================================
Predikcija (QNNR-Sampler, deterministicki, seed=39):
[4, 31, x, y, z, 35, 36]
==================================================
"""



"""
QNNR - Quantum Neural Network Regressor
(Sampler)

Isto kolo kao Estimator verzija (ZZFeatureMap + TwoLocal)
Razlika: umesto Pauli Z ekspektacije, koristi parity interpret 
- sabira verovatnoce svih stanja sa neparnim brojem jedinica u bitstringu
parity_1_prob = P(odd parity) je izlaz za svaki ulaz
Egzaktno preko Statevector.probabilities() - deterministicki
COBYLA 200 iteracija po poziciji
"""
