# Real-data case studies (KuaiRand-Pure, MovieLens-1M, Goodbooks-10k, Criteo, ASlib SAT12 and QBF-2014)

Semi-synthetic **log-bootstrap** environments: arms are the 100 most exposed items (recommendation),
the 100 most exposed campaigns (Criteo) or all the solvers of an ASlib scenario (SAT12: 31, QBF-2014: 14);
objectives are binary rates measured on the logs, and pulling an arm resamples a real record
(KuaiRand / Criteo: one logged impression, keeping the correlations between signals; MovieLens /
Goodbooks: one draw per objective in the corresponding population; SAT12 / QBF-2014: one instance of
each category, reward = solved within the cutoff). True means are the exact population rates,
so the benchmark V* is computed exactly by enumerating the k-subsets of the true Pareto front.

The algorithms are those of `../model.py`, **unchanged**. `fastpatch.py` only vectorises the
hypervolume and Pareto-dominance helpers; trajectories are identical to the original code (≤ 1e-16).

| Setting | Objectives | n | k | |Pareto front| |
|---|---|---|---|---|
| `kuai-d2` | long view, profile visit | 100 | 3 | 9 |
| `kuai-d3` | long view, like, profile visit | 100 | 4 | 16 |
| `ml-d2` | like (rating ≥ 4), discovery | 100 | 3 | 6 |
| `ml-d3` | like (women), like (men), discovery | 100 | 4 | 12 |
| `gb-d2` | 5-star, to-read | 100 | 3 | 5 |
| `gb-d3` | 5-star, discovery, to-read | 100 | 4 | 15 |
| `sat-d2` | solved rate on industrial, random instances | 31 | 3 | 5 |
| `sat-d3` | solved rate on crafted, industrial, random instances | 31 | 4 | 5 |
| `crit-d2` | click, conversion | 100 | 3 | 5 |
| `crit-d3` | click, conversion, low cost (cost ≤ global median) | 100 | 4 | 13 |
| `qbf-d2` | solved rate on applications, eval2012 instances | 14 | 3 | 4 |
| `qbf-d3` | solved rate on applications, eval2012, preprocessing instances | 14 | 4 | 5 |

## Data
The data are not included (the MovieLens license forbids redistribution). Run

```bash
./download_data.sh
```

| Dataset | Source | Files used |
|---|---|---|
| KuaiRand-Pure | https://kuairand.com (mirror: https://github.com/Applied-Machine-Learning-Lab/KuaiSim) | `log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv` |
| MovieLens-1M | https://grouplens.org/datasets/movielens/1m/ | `ratings.dat`, `users.dat` |
| Goodbooks-10k | https://github.com/zygmuntz/goodbooks-10k | `ratings.csv`, `to_read.csv` |
| ASlib SAT12 | https://github.com/coseal/aslib_data | `SAT12-{ALL,HAND,INDU,RAND}/algorithm_runs.arff` |
| ASlib QBF-2014 | https://github.com/coseal/aslib_data | `QBF-2014/algorithm_runs.arff` |
| Criteo Attribution | https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/ (also on Kaggle) | `pcb_dataset_final.tsv`, reduced by `reduce_criteo.py` to `data/criteo/criteo_agg.csv.gz` |

Criteo cannot be fetched by the script (license acceptance). Download `pcb_dataset_final.tsv` manually, then run
`python3 reduce_criteo.py /path/to/pcb_dataset_final.tsv` (16,468,027 impressions, 675 campaigns).

## Run
```bash
pip install numpy pandas scipy matplotlib
python run_real.py                 # 12 settings x 14 algorithms x eta in {0.001,0.003,0.01,0.1,0.3,1} x 10 seeds, T=3000
python run_real.py kuai-d3         # a single setting
python run_norm.py                 # ScalarUCB-Norm (online min-max normalization of the objectives), same protocol
python analysis_real.py            # LaTeX tables (out/tables) and figures (out/figures); eta* = argmax of the final HV (last 100 rounds)
python diversity_real.py           # diversity of selection (Gutowski et al., SAC 2019) at the selected eta
python pruning_real.py             # active-set size and timing of safe pruning at the selected eta
python lockstep.py ml-d2 0.001     # pruning vs. no pruning from identical states
```
The full grid takes several hours on 2 CPU cores (`run_real.py` uses 2 processes).
