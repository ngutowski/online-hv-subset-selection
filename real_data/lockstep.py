"""Lockstep check: at every round, from the SAME algorithm state, compare the slate chosen with and
without safe pruning, and the optimistic hypervolume of both slates."""
import sys, numpy as np
import paths
import fastpatch; fastpatch.apply()
import model as m, realenv
from pruning_real import THVInstr
cfg = sys.argv[1]; ETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01; e0 = realenv.get_env(cfg); T = 3000
for seed in range(1234, 1237):
    e = e0.fresh(); rng = np.random.default_rng(seed)
    alg = THVInstr(n=e0.mu.shape[0], d=e0.mu.shape[1], k=e0.k, ref=e0.ref, T=T, eta=ETA, seed=seed, min_pulls=2)
    alg.active_log = []
    ndiff, maxgap, zero_gain_only = 0, 0.0, True
    for t in range(1, T + 1):
        alg.prune = True; a1 = alg.select(t)
        alg.prune = False; a2 = alg.select(t)
        if a1 != a2:
            ndiff += 1
            b = np.sqrt(alg.eta * alg._logterm / (2.0 * np.maximum(1, alg.counts)))[:, None]; U = alg.means + b
            maxgap = max(maxgap, abs(m.hv(U[a1], e0.ref) - m.hv(U[a2], e0.ref)))
            # which items differ, and do they add HV?
        alg.update(a1, e.sample(a1, rng))
    print(cfg, seed, 'rounds with different slates:', ndiff, '/', T, ' max |optimistic HV gap|: %.2e' % maxgap, flush=True)
