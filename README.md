# Top-k Pareto Bandits: Hypervolume Regret for Multi-Objective Slate Selection

Code for the paper *"Top-k Pareto Bandits: Hypervolume Regret for Multi-Objective Slate Selection"*, submitted to NeurIPS 2026.

---

## Overview

This repository implements **THV-UCB**, an optimistic algorithm for the Top-k Pareto bandit problem, together with 13 baselines adapted to the top-k semi-bandit setting. The agent selects a slate of k arms per round and is evaluated by the dominated hypervolume of the selected subset relative to the best ε-cover of the Pareto frontier.

---

## Files

| File | Description |
|---|---|
| `model.py` | All algorithm classes, environment generation, hypervolume utilities, and runner |
| `exec.py` | Main experiment script — parse args, run all algorithms, save results and plots |

---

## Requirements

```
numpy
matplotlib
```

No other dependencies. The hypervolume is computed exactly via inclusion-exclusion (tractable for small k).

---

## Usage

```bash
python exec.py <front> <d> <n> <n_front> <k> <sigma> <eps> <T> <eta>
```

**Parameters:**

| Argument | Description |
|---|---|
| `front` | Pareto front geometry: `linear`, `convex`, `concave`, `clusters` |
| `d` | Number of objectives |
| `n` | Total number of arms |
| `n_front` | Number of arms on the Pareto front |
| `k` | Slate size |
| `sigma` | Observation noise (Gaussian) |
| `eps` | ε-covering tolerance |
| `T` | Horizon (number of rounds) |
| `eta` | Confidence parameter η (grid-searched over {0.01, 0.1, 0.3, 1.0}) |

**Examples:**

```bash
# d=2, paper default config
python exec.py concave 2 36 12 3 0.05 0.05 2000 0.01

# d=5
python exec.py clusters 5 150 70 6 0.02 0.2 5000 0.01
```

**Outputs** (saved in `results/`):
- `results_{tag}.pkl` — full results dict (pickle)
- `summary_{tag}_eta{eta}.txt` — HV_last100 ± CI and cumulative α-regret ± CI for all algorithms
- `Regret_{tag}.pdf` — cumulative α-regret curves
- `HV_{tag}.pdf` — hypervolume trajectory (moving average, w=50)

---

## Experimental configurations

| d | n | n_front | k | σ | ε | T |
|---|---|---|---|---|---|---|
| 2 | 36 | 12 | 3 | 0.05 | 0.05 | 2000 |
| 3 | 60 | 30 | 4 | 0.035 | 0.1 | 3000 |
| 4 | 100 | 50 | 5 | 0.025 | 0.15 | 5000 |
| 5 | 150 | 70 | 6 | 0.02 | 0.2 | 5000 |

All results are averaged over 10 random seeds. η is selected per algorithm and front geometry via grid search.

---

## Algorithms

### THV-UCB (ours)
Coordinate-wise UCB boxes + safe pruning (LCB/UCB dominance) + greedy selection by optimistic marginal hypervolume gain.

### Baselines

| Class in `model.py` | Paper name | Fidelity | Reference |
|---|---|---|---|
| `ParetoUCB_TopK_DR` | ParetoUCB | • faithful | [6] Drugan & Nowé, IJCNN 2013 |
| `ParetoUCB_TopK` | ParetoUCB+ | ◦ extension | [6] Drugan & Nowé, IJCNN 2013 |
| `ParetoUCB_DiverseTopK` | ParetoUCB-Div | ◦ extension | [6] Drugan & Nowé, IJCNN 2013 |
| `ParetoUCB_CrowdingTopK` | ParetoUCB-Crowd | ◦ extension | [6, 5] Drugan & Nowé + Deb et al., 2002 |
| `ParetoTS_Faithful_TopK` | ParetoTS | • faithful | [34] Yahyaa & Manderick, ESANN 2015 |
| `ParetoTSPlus_TopK` | ParetoTS+ | ◦ extension | [34] Yahyaa & Manderick, ESANN 2015 |
| `ChebyshevUCB_Mandow` | ChebyshevUCB | • faithful | [18] Mandow et al., MODeM@ECAI 2023 |
| `ChebyshevUCB` | ChebyshevUCB+ | ◦ extension | [18] Mandow et al., MODeM@ECAI 2023 |
| `HVScalarUCB` | HVScalarUCB | • faithful | [37] Zhang & Golovin, ICML 2020 |
| `HVScalarUCB_Zhang2024` | HVScalarUCB+ | • faithful | [36] Zhang, NeurIPS 2024 |
| `ScalarUCB` | ScalarUCB | ◦ extension | [2] Auer et al., 2002 |
| `ScalarUCB_RandW` | ScalarUCB-RandW | ◦ extension | [22] Paria et al., UAI 2020 |
| `RandomK` | RandomK | — | — |

**Fidelity:** • = faithful top-k adaptation (core mechanism unchanged); ◦ = our extension (modified confidence term, tie-break, or diversity mechanism).



