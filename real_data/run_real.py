import sys, os, time, numpy as np
import paths
from multiprocessing import Pool
OUT = paths.RUNS
ALGOS = ["THV-UCB","ParetoUCB_DR","ParetoUCB","ParetoUCB-Div","ParetoUCB-Crowd","ParetoTS_Faithful_TopK","ParetoTSPlus",
         "ChebyshevUCB_M","ChebyshevUCB","HVScalarUCB","HVScalarUCB_Zhang2024","ScalarUCB","ScalarUCB-RandW","RandomK"]
NO_ETA = {"ParetoTS_Faithful_TopK", "ParetoTSPlus", "RandomK"}
ETAS = [0.001, 0.003, 0.01, 0.1, 0.3, 1.0]
T, SEEDS = 3000, 10

def task(args):
    cfg, algo, eta = args
    f = f"{OUT}{cfg}__{algo}__{eta}.npz"
    if os.path.exists(f): return f
    import fastpatch; fastpatch.apply()
    import realenv, model as m
    e0 = realenv.get_env(cfg); t0 = time.time()
    hv, arms = [], []
    for s in range(SEEDS):
        e = e0.fresh()
        r = m.run_once(e, algo, T=T, k=e0.k, eps=0.0, eta=eta if eta is not None else 0.1,
                       seed=1234 + s, alpha_approx=1 - 1 / np.e, V_star_precomputed=e0.V_star)
        hv.append(r['hv'].astype(np.float64)); arms.append(np.array(e.history, dtype=np.int16))
    np.savez_compressed(f, hv=np.array(hv), arms=np.array(arms))
    print(f"{cfg} {algo} {eta} {time.time()-t0:.0f}s", flush=True)
    return f

if __name__ == '__main__':
    import realenv
    cfgs = sys.argv[1:] or list(realenv.CONFIGS)
    tasks = []
    for c in cfgs:
        for a in ALGOS:
            for eta in ([None] if a in NO_ETA else ETAS):
                tasks.append((c, a, eta))
    tasks.sort(key=lambda x: (x[1] != 'THV-UCB', x[1] not in ('ParetoUCB_DR',)))  # long ones first
    with Pool(2) as p:
        for _ in p.imap_unordered(task, tasks): pass
    print("ALL DONE", flush=True)
