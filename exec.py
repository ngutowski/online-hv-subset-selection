# -*- coding: utf-8 -*-
"""
Usage:
    python exec.py <front> <d> <n> <n_front> <k> <sigma> <eps> <T> <eta>

Example:
    python exec.py convex 2 36 12 3 0.05 0.05 2000 0.01
    python exec.py clusters 5 100 50 6 0.02 0.10 5000 1.0

Parameters:
    front    : linear | convex | concave | clusters
    d        : number of objectives
    n        : total number of arms
    n_front  : number of arms on the Pareto front
    k        : slate size
    sigma    : observation noise
    eps      : covering tolerance
    T        : horizon
    eta      : eta value
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m
import math
import numpy as np
import sys

# ---------- Parse arguments ----------
if len(sys.argv) != 10:
    print(__doc__)
    sys.exit(1)

front   = str(sys.argv[1])
d       = int(sys.argv[2])
n       = int(sys.argv[3])
n_front = int(sys.argv[4])
k       = int(sys.argv[5])
sigma   = float(sys.argv[6])
eps     = float(sys.argv[7])
T       = int(sys.argv[8])
eta     = float(sys.argv[9])
seeds = 10
alpha_approx = 1 - 1/math.e

# ---------- Environment ----------
env = m.make_env_front(n=n, front=front, sigma=sigma, seed=0, d=d, n_front=n_front)

P_idx = m.pareto_front_indices(env.mu)
print(f"Config: front={front}, d={d}, n={n}, n_front={n_front}, k={k}, "
      f"sigma={sigma}, eps={eps}, T={T}")
print(f"|Pareto front| = {len(P_idx)}")
print(f"Front mean per coord: {env.mu[P_idx].mean(axis=0).round(3)}")

from math import comb
c = comb(n, k)
print(f"C({n},{k}) = {c:,}")

algos= ["THV-UCB","ParetoUCB_DR","ParetoUCB","ParetoUCB-Div", "ParetoUCB-Crowd",
          "ParetoTS_Faithful_TopK","ParetoTSPlus", "ChebyshevUCB_M","ChebyshevUCB","HVScalarUCB", "HVScalarUCB_Zhang2024", "ScalarUCB", "ScalarUCB-RandW",  "RandomK"]


# ---------- Compute V_star once ----------
print("Computing V_star...", flush=True)
S_star_global, V_star_val = m.best_eps_cover_hv(env.mu, k=k, eps=eps, ref=env.ref)
print(f"V* = {V_star_val:.4f},  alpha*V* = {alpha_approx*V_star_val:.4f}")


# ---------- Algorithms ----------
results = {}
for algo in algos:
    eta_algo = eta
    print(f"Running {algo}...", flush=True)

    runs = [m.run_once(env, algo, T=T, k=k, eps=eps,
                       eta=eta_algo if eta_algo is not None else eta,
                       seed=1234+s, alpha_approx=alpha_approx,
                       V_star_precomputed=V_star_val) for s in range(seeds)]
    
    results[algo] = m.summarize_runs(runs)



# ---------- Summary table ----------
def _summary_lines(algos, results, seeds, eta):
    lines = []
    hdr = f"\n{'Algorithm':<25} {'HV_last100':>12} {'±95%CI':>10} {'Final Reg':>10} {'±95%CI':>10} {'eta':>6}"
    sep = "-" * 78
    lines += [hdr, sep]
    for algo in algos:
        hv_last     = results[algo]["hv_mean"][-100:].mean()
        hv_std      = results[algo]["hv_std"][-100:].mean()
        ci_hv       = 1.96 * hv_std / math.sqrt(seeds)
        cum_reg     = np.cumsum(results[algo]["aregret_mean"])[-1]
        cum_reg_std = np.cumsum(results[algo]["aregret_std"])[-1]
        ci_reg      = 1.96 * cum_reg_std / math.sqrt(seeds)
        eta_str     = f"{eta}" if eta is not None else "-"
        lines.append(f"{algo:<25} {hv_last:>12.4f} {ci_hv:>10.4f} {cum_reg:>10.2f} {ci_reg:>10.2f} {eta_str:>6}")
    return lines
 
summary = _summary_lines(algos, results, seeds, eta)
print("\n".join(summary))

import pickle
import os



# ---------- Summary table ----------
print(f"\n{'Algorithm':<25} {'HV_last100':>12} {'±95%CI':>10} {'Final Reg':>10} {'±95%CI':>10} {'eta':>6}")
print("-" * 78)
for algo in algos:
    hv_last = results[algo]["hv_mean"][-100:].mean()
    hv_std  = results[algo]["hv_std"][-100:].mean()
    ci_hv   = 1.96 * hv_std / math.sqrt(seeds)

    cum_reg     = np.cumsum(results[algo]["aregret_mean"])[-1]
    cum_reg_std = np.cumsum(results[algo]["aregret_std"])[-1]
    ci_reg      = 1.96 * cum_reg_std / math.sqrt(seeds)

    eta_val = eta #eta_map.get(algo, eta)
    eta_str = f"{eta_val}" if eta_val is not None else "-"

    print(f"{algo:<25} {hv_last:>12.4f} {ci_hv:>10.4f} {cum_reg:>10.2f} {ci_reg:>10.2f} {eta_str:>6}")


'''
results = {}
for algo in algos:
    print(f"Running {algo}...", flush=True)
    runs = [m.run_once(env, algo, T=T, k=k, eps=eps, eta=eta,
                       seed=1234+s, alpha_approx=alpha_approx) for s in range(seeds)]
    results[algo] = m.summarize_runs(runs)
'''
Vstar = V_star_val

# ---------- Tag for filenames ----------
tag = f"{front}_d{d}_n{n}_nf{n_front}_k{k}_s{sigma}_e{eps}_T{T}"

# ---------- Save results ----------
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"results_{tag}.pkl")
with open(save_path, "wb") as f:
    pickle.dump({
        "results": results,
        "algos": algos,
        "tag": tag,
        "front": front, "d": d, "n": n, "n_front": n_front,
        "k": k, "sigma": sigma, "eps": eps, "T": T, "seeds": seeds,
        "alpha_approx": alpha_approx,
    }, f)
print(f"Saved results to {save_path}")


x = np.arange(1, T+1)

# ---------- Cumulative clipped alpha-regret ----------
plt.figure(figsize=(9, 5))
for algo in algos:
    plt.plot(x, np.cumsum(results[algo]["aregret_mean"]), label=algo)
plt.xlabel("Round t")
plt.ylabel("Cumulative clipped α-regret")
plt.title(f"front={front}, d={d}, n={n}, n_front={n_front}, k={k}, "
          f"σ={sigma}, ε={eps}")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(f"Regret_{tag}.pdf")
print(f"Saved Regret_{tag}.pdf")

# ---------- HV trajectory ----------
w = 50
plt.figure(figsize=(9, 5))
for algo in algos:
    my = results[algo]["hv_mean"]
    s = results[algo]["hv_std"]
    ci = 1.96 * s / math.sqrt(seeds)
    ms = m.moving_average(my, w=w)
    los = m.moving_average(my - ci, w=w)
    his = m.moving_average(my + ci, w=w)
    plt.plot(x, ms, label=algo)
    plt.fill_between(x, los, his, alpha=0.15)
plt.xlabel("Round t")
plt.ylabel("HV(S_t) (moving avg)")
plt.title(f"front={front}, d={d}, n={n}, n_front={n_front}, k={k}, "
          f"σ={sigma}, ε={eps}, w={w}")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(f"HV_{tag}.pdf")
print(f"Saved HV_{tag}.pdf")

# ---------- Summary table ----------
print(f"\n{'Algorithm':<20} {'HV_last100':>12} {'±95%CI':>10} {'Final Reg':>10} {'±95%CI':>10}")
print("-" * 66)
for algo in algos:
    hv_last = results[algo]["hv_mean"][-100:].mean()
    hv_std  = results[algo]["hv_std"][-100:].mean()
    ci_hv = 1.96 * hv_std / math.sqrt(seeds)
    
    cum_reg = np.cumsum(results[algo]["aregret_mean"])[-1]
    cum_reg_std = np.cumsum(results[algo]["aregret_std"])[-1]
    ci_reg = 1.96 * cum_reg_std / math.sqrt(seeds)
    
    print(f"{algo:<25} {hv_last:>12.4f} {ci_hv:>10.4f} {cum_reg:>10.4f} {ci_reg:>10.4f}")
    
    
# ---------- Summary table (fin) + save text file ----------
summary = _summary_lines(algos, results, seeds, eta)
print("\n".join(summary))
 
summary_path = os.path.join(save_dir, f"summary_{tag}_eta{eta}.txt")
with open(summary_path, "w") as f:
    f.write(f"front={front}, d={d}, n={n}, n_front={n_front}, k={k}, "
            f"sigma={sigma}, eps={eps}, T={T}, eta={eta}, seeds={seeds}\n")
    f.write("\n".join(summary) + "\n")
print(f"Saved summary to {summary_path}")
