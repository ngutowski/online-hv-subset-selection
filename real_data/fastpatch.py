"""Speed patches for model.py: same mathematics, vectorised (verified to give identical traces)."""
import numpy as np
import model as m

_SUB = {}
def _subsets(k):
    if k not in _SUB:
        S = np.array([[(s >> i) & 1 for i in range(k)] for s in range(1, 1 << k)], dtype=bool)
        sign = np.where(S.sum(1) % 2 == 1, 1.0, -1.0)
        _SUB[k] = (S, sign)
    return _SUB[k]

def hv_fast(points, ref):
    if points is None or len(points) == 0:
        return 0.0
    P = np.asarray(points, dtype=float); r = np.asarray(ref, dtype=float)
    P = P[np.all(P > r[None, :], axis=1)]
    if P.shape[0] == 0:
        return 0.0
    S, sign = _subsets(P.shape[0])
    upper = np.where(S[:, :, None], P[None, :, :], np.inf).min(axis=1)
    vol = np.prod(np.maximum(0.0, upper - r), axis=1)
    return float(max(0.0, float(np.sum(sign * vol))))

def _front(V, cand):
    X = V[cand]
    dom = (X[:, None, :] >= X[None, :, :]).all(2) & (X[:, None, :] > X[None, :, :]).any(2)
    nd = ~dom.any(axis=0)
    return [cand[t] for t in range(len(cand)) if nd[t]]

def _pareto_layers(self, V):
    remaining = set(range(self.n)); layers = []
    while remaining:
        rem = list(remaining)
        front = _front(V, rem)
        for i in front:
            remaining.remove(i)
        layers.append(front)
    return layers

def _pareto_front(self, U, candidates):
    return _front(U, list(candidates))

def apply():
    m.hv = hv_fast
    for cls in [m.ParetoUCB_TopK, m.ParetoTSPlus_TopK, m.ParetoTS_TopK, m.ParetoTS_Faithful_TopK]:
        cls._pareto_layers = _pareto_layers
    m.ParetoUCB_TopK_DR._pareto_front = _pareto_front
