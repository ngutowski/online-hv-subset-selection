import sys, os, numpy as np
import paths
from multiprocessing import Pool
OUT = paths.RUNS; T, SEEDS, ETAS = 3000, 10, [0.001, 0.003, 0.01, 0.1, 0.3, 1.0]
def task(args):
    import fastpatch; fastpatch.apply()
    import model as m, realenv
    from extra_baselines import ScalarUCBNorm
    cfg, eta = args; f = f"{OUT}{cfg}__ScalarUCB-norm__{eta}.npz"
    if os.path.exists(f): return
    e0 = realenv.get_env(cfg); mu = e0.mu; n, d = mu.shape; hv, arms = [], []
    for s in range(SEEDS):
        e = e0.fresh(); rng = np.random.default_rng(1234 + s)
        alg = ScalarUCBNorm(n, d, e0.k, T, eta=eta, seed=1234 + s, min_pulls=2); h = np.zeros(T)
        for t in range(1, T + 1):
            a = alg.select(t); alg.update(a, e.sample(a, rng)); h[t - 1] = m.hv(mu[a], e0.ref)
        hv.append(h); arms.append(np.array(e.history, dtype=np.int16))
    np.savez_compressed(f, hv=np.array(hv), arms=np.array(arms))
if __name__ == '__main__':
    import realenv
    with Pool(2) as p: p.map(task, [(c, eta) for c in realenv.CONFIGS for eta in ETAS])
    print('done')
