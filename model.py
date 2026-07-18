# -*- coding: utf-8 -*-
"""
Created on 2026

@author: anonymous
"""

import math
import itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np

# ---------- Hypervolume (exact for anchored boxes, any d, small |S|=k) ----------

def hv_exact_anchored(points: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact dominated hypervolume for a set of points (maximization) with common reference point ref,
    when HV is defined as volume of union of axis-aligned boxes [ref, point].
    Works in any dimension d, but exponential in m = number of points (2^m).
    Perfect when m=k is small (e.g., k<=10; in your experiments k=3).
    """
    if points is None or len(points) == 0:
        return 0.0

    P = np.asarray(points, dtype=float)
    r = np.asarray(ref, dtype=float)
    assert P.ndim == 2 and r.ndim == 1 and P.shape[1] == r.shape[0]

    # keep only points that dominate ref strictly (optional strictness; keeps HV correct anyway)
    mask = np.all(P > r[None, :], axis=1)
    P = P[mask]
    if P.shape[0] == 0:
        return 0.0

    m, d = P.shape
    hv = 0.0

    # inclusion–exclusion over non-empty subsets J of {0..m-1}
    # intersection box upper corner is coord-wise min over points in J
    for s in range(1, 1 << m):
        idx = [i for i in range(m) if (s >> i) & 1]
        upper = np.min(P[idx, :], axis=0)
        side = np.maximum(0.0, upper - r)
        vol = float(np.prod(side))
        sign = 1.0 if (len(idx) % 2 == 1) else -1.0
        hv += sign * vol

    return float(max(0.0, hv))

def hv(points: np.ndarray, ref: np.ndarray) -> float:
    return hv_exact_anchored(points, ref)


# ---------- Pareto front, eps-cover ----------

def pareto_front_indices(mu: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-undominated points (maximize). O(n^2 d)."""
    mu = np.asarray(mu, dtype=float)
    n = mu.shape[0]
    is_dom = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dom[i]:
            continue
        for j in range(n):
            if i == j or is_dom[i]:
                continue
            if np.all(mu[j] >= mu[i]) and np.any(mu[j] > mu[i]):
                is_dom[i] = True
    return np.where(~is_dom)[0]

def eps_covers(mu: np.ndarray, S: List[int], eps: float, P_idx: np.ndarray) -> bool:
    """Check if subset S eps-covers Pareto front points indexed by P_idx."""
    S = list(S)
    for p in P_idx:
        ok = False
        for i in S:
            if np.all(mu[p] <= mu[i] + eps):
                ok = True
                break
        if not ok:
            return False
    return True

def best_eps_cover_hv(mu: np.ndarray, k: int, eps: float, ref: np.ndarray,
                     brute_force_limit: int = 200000) -> Tuple[List[int], float]:
    """
    Return S*_eps maximizing HV(S) s.t. eps-covers Pareto front.
    Brute force if C(n,k) <= brute_force_limit, else fallback to greedy on true means (no guarantee).
    """
    mu = np.asarray(mu, dtype=float)
    n = mu.shape[0]
    P_idx = pareto_front_indices(mu)

    # brute force if feasible
    from math import comb
    if comb(n, k) <= brute_force_limit:
        best_S = None
        best_V = -1.0
        for S in itertools.combinations(range(n), k):
            if eps_covers(mu, S, eps, P_idx):
                V = hv(mu[list(S)], ref)
                if V > best_V:
                    best_V = V
                    best_S = list(S)
        if best_S is None:
            # if infeasible, drop cover constraint (fallback)
            best_S = max(itertools.combinations(range(n), k),
                         key=lambda S: hv(mu[list(S)], ref))
            best_S = list(best_S)
            best_V = hv(mu[best_S], ref)
        return best_S, float(best_V)

    # fallback greedy on true means (no cover constraint)
    chosen = []
    cur = np.empty((0, mu.shape[1]))
    cur_hv = 0.0
    for _ in range(k):
        best_i, best_gain = None, -1e18
        for i in range(n):
            if i in chosen:
                continue
            cand = np.vstack([cur, mu[i:i+1]])
            gain = hv(cand, ref) - cur_hv
            if gain > best_gain:
                best_gain, best_i = gain, i
        chosen.append(best_i)
        cur = np.vstack([cur, mu[best_i:best_i+1]])
        cur_hv = hv(cur, ref)
    return chosen, float(cur_hv)

# ---------- Synthetic environment ----------

@dataclass
class SyntheticEnv:
    mu: np.ndarray          # (n,2)
    sigma: float
    ref: np.ndarray         # (2,)

    def sample(self, arms: List[int], rng: np.random.Generator) -> np.ndarray:
        means = self.mu[arms]
        noise = rng.normal(0.0, self.sigma, size=means.shape)
        return np.clip(means + noise, 0.0, 1.0)



def make_env_front(n: int, front: str, sigma: float, seed: int = 0,
                   ref: Optional[np.ndarray] = None, d: int = 2,
                   n_front: Optional[int] = None) -> SyntheticEnv:
    """
    Synthetic MO environment.
    - front in {"linear","convex","concave","clusters"}.
    - n_front: number of arms on the Pareto front (default: max(10, 0.35*n)).
    """
    rng = np.random.default_rng(seed)
    if ref is None:
        ref = np.zeros(d, dtype=float)
    else:
        ref = np.asarray(ref, float)

    if n_front is None:
        n_front = max(10, int(0.35 * n))
    n_front = min(n_front, n)
    n_dom = n - n_front
    

    def dtlz_points(m, alpha, angle_ranges=None):
        """Sample m points on a DTLZ front with shape exponent alpha."""
        if angle_ranges is None:
            angle_ranges = [(0.0, np.pi / 2)] * (d - 1)
        angles = np.column_stack(
            [rng.uniform(lo, hi, size=m) for lo, hi in angle_ranges]
        )
        pts = np.zeros((m, d))
        for i in range(d):
            p = np.ones(m)
            for j in range(i):                       # sin^alpha for each preceding angle
                p *= np.sin(angles[:, j]) ** alpha
            if i < d - 1:                            # cos^alpha for own angle
                p *= np.cos(angles[:, i]) ** alpha
            pts[:, i] = p
        return pts

    if front == "concave":
        # Spherical / DTLZ2 : sum(x_i^2) = 1
        front_mu = dtlz_points(n_front, alpha=1.0)

    elif front == "convex":
        # Outward-bulging : sum(x_i^4) = 1
        front_mu = dtlz_points(n_front, alpha=0.5)

    elif front == "linear":
        # Simplex : sum(x_i) = 1.  Scale up and clip to [0,1] so that
        # individual coordinates (and thus HV) don't collapse with d.
        scale = min(d * 0.45, 2.5)
        #scale = min(d * 0.65, 4.0)
        front_mu = dtlz_points(n_front, alpha=2.0) * scale
        front_mu = np.clip(front_mu, 0.01, 1.0)

    elif front == "clusters":
        # Two groups on the spherical front with separated angle ranges
        n1 = n_front // 2
        n2 = n_front - n1
        # cluster 1 : theta_1 small  -> x_d small, x_1 large
        ranges1 = [(0.0, np.pi / 5)] + [(0.0, np.pi / 2)] * (d - 2)
        # cluster 2 : theta_1 large  -> x_d large, x_1 small
        ranges2 = [(3 * np.pi / 10, np.pi / 2)] + [(0.0, np.pi / 2)] * (d - 2)
        front_mu = np.vstack([
            dtlz_points(n1, alpha=1.0, angle_ranges=ranges1),
            dtlz_points(n2, alpha=1.0, angle_ranges=ranges2),
        ])
    else:
        raise ValueError(f"Unknown front: {front}")

    front_mu = np.clip(front_mu, 0.01, 1.0)
    dom = rng.random((n_dom, d)) * 0.30
    mu = np.vstack([front_mu, dom])
    return SyntheticEnv(mu=mu, sigma=float(sigma), ref=ref)




# ---------- Algorithms ----------

class RandomK:
    def __init__(self, n: int, k: int, seed: int = 0):
        self.n, self.k = n, k
        self.rng = np.random.default_rng(seed)

    def select(self, t: int) -> List[int]:
        return self.rng.choice(self.n, size=self.k, replace=False).tolist()

    def update(self, arms: List[int], rewards: np.ndarray):
        pass

class ScalarUCB:
    """
	ScalarUCB – our extension of Auer et al. (Machine Learning 2002) [2] to top-k.
	- Fixed uniform linear scalarization: w = 1/d
	- Scalar UCB index: mu_hat_i + beta_i, with beta_i = sqrt(η * log(n * T²) / (2 * N_i))
	- Select top-k arms by scalar UCB index.
    """
    def __init__(self, n: int, d: int, k: int, T: int, w: np.ndarray,
                 eta: float = 1.0, seed: int = 0, min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.w = np.asarray(w, float)
        self.eta = float(eta)
        self.min_pulls = int(min_pulls)

        self.counts = np.zeros(n, dtype=int)
        self.mean_scalar = np.zeros(n, dtype=float)
        self._logterm = math.log(self.n * (self.T**2) + 1.0)

    def beta(self, i: int) -> float:
        denom = max(1, int(self.counts[i]))
        return math.sqrt(self.eta * self._logterm / (2.0 * denom))

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        ucb = self.mean_scalar + np.array([self.beta(i) for i in range(self.n)])
        idx = np.argsort(-ucb)[:self.k]
        return idx.tolist()

    def update(self, arms: List[int], rewards: np.ndarray):
        # rewards are vectors; scalarize
        scal = rewards @ self.w
        for a, r in zip(arms, scal):
            self.counts[a] += 1
            n_a = self.counts[a]
            self.mean_scalar[a] += (r - self.mean_scalar[a]) / n_a

class ScalarUCB_RandW:
    """
	ScalarUCB-RandW – our extension of Paria et al. (UAI 2020) [22] to top-k.
	- Fresh weight vector w ~ Dirichlet(1) drawn each round
	- Compute UCB vectors U_i = mu_hat_i + beta_i * 1 coordinate-wise
	- Select top-k arms by score w^T U_i.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 eta: float = 1.0, seed: int = 0, min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.eta = float(eta)
        self.min_pulls = int(min_pulls)
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(n, dtype=int)
        self.means = np.zeros((n, d), dtype=float)
        self._logterm = math.log(self.n * self.d * (self.T**2) + 1.0)

    def beta(self, i: int) -> float:
        denom = max(1, int(self.counts[i]))
        return math.sqrt(self.eta * self._logterm / (2.0 * denom))

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # random weights on simplex (Dirichlet)
        w = self.rng.dirichlet(np.ones(self.d))

        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i)
            U[i] = np.clip(self.means[i] + b, 0.0, 1.0)

        scores = U @ w
        idx = np.argsort(-scores)[:self.k]
        return idx.tolist()

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i

class ParetoUCB_TopK_DR:
    """
    Pareto UCB1 (Drugan & Nowé, IJCNN 2013) adapted to top-k.
    - Confidence term: sqrt(2 * ln(n * (D*K)^{1/4}) / n_i)
      using the empirical variant (K instead of unknown |A*|).
    - Non-dominated set A' computed on UCB vectors.
    - If |A'| >= k: select k arms uniformly at random from A'.
    - If |A'| < k: take all of A', then compute next Pareto layer
      on remaining arms, sample uniformly from that layer, repeat.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 eta: float = 1.0, seed: int = 0, min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.eta = float(eta)  # kept for interface compatibility but not used in beta
        self.min_pulls = int(min_pulls)
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(n, dtype=int)
        self.means = np.zeros((n, d), dtype=float)
        self.total_pulls = 0

    def beta(self, i: int, front_size: int) -> float:
        n_total = max(1, self.total_pulls)
        n_i = max(1, int(self.counts[i]))
        return math.sqrt(2.0 * math.log(n_total * (self.d * front_size) ** 0.25) / n_i)

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # Only once per round
        front_size = max(1, len(pareto_front_indices(self.means)))

        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i, front_size)
            U[i] = self.means[i] + b

        
	# Fill k slots by successive Pareto layers, uniform random within each
        chosen = []
        remaining = list(range(self.n))
        while len(chosen) < self.k and remaining:
            front = self._pareto_front(U, remaining)
            need = self.k - len(chosen)
            if len(front) <= need:
                chosen.extend(front)
            else:
                picked = self.rng.choice(front, size=need, replace=False).tolist()
                chosen.extend(picked)
            for i in front:
                if i in remaining:
                    remaining.remove(i)
        return chosen[:self.k]
        
    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a >= b) and np.any(a > b)

    def _pareto_front(self, U: np.ndarray, candidates: List[int]) -> List[int]:
        """Return non-dominated set among candidates based on UCB vectors U."""
        front = []
        for i in candidates:
            dominated = False
            for j in candidates:
                if j == i:
                    continue
                if self._dominates(U[j], U[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        return front


    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            self.total_pulls += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i

class HVScalarUCB_Zhang2024:
    """
	  HVScalarUCB+ – our top-k extension of Zhang (NeurIPS 2024) [36].

	Fidelity to the original paper:
	- Scalarization: s_λ(y) = min_i(y_i / λ_i)  (Section 5, practical version)
	- λ ~ S^{d-1}_+ uniform (distribution D from the paper)
	- k distinct λ per round: each slate slot targets a different
	  direction of the Pareto front (Lemma 5)

	Top-k adaptation: our own extension of the directional intuition of Lemma 5.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 ref: Optional[np.ndarray] = None,
                 eta: float = 1.0,
                 seed: int = 0,
                 min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.eta = float(eta)
        self.min_pulls = int(min_pulls)
        self.rng = np.random.default_rng(seed)
        self.ref = np.zeros(d, dtype=float) if ref is None else np.asarray(ref, float)

        self.counts = np.zeros(n, dtype=int)
        self.means = np.zeros((n, d), dtype=float)
        self._logterm = math.log(self.n * self.d * (self.T ** 2) + 1.0)

    def beta(self, i: int) -> float:
        denom = max(1, int(self.counts[i]))
        return math.sqrt(self.eta * self._logterm / (2.0 * denom))

    def _sample_lambda(self) -> np.ndarray:
        """λ ~ S^{d-1}_+: positive unit sphere."""
        lam = np.abs(self.rng.standard_normal(self.d))
        norm = np.linalg.norm(lam)
        if norm < 1e-12:
            lam = np.ones(self.d)
            norm = np.linalg.norm(lam)
        return lam / norm

    def _hv_scalarization(self, y: np.ndarray, lam: np.ndarray) -> float:
        """
        s_λ(y) = min_i(y_i / λ_i)
        The maximizer is the Pareto point in direction λ (Lemma 5).
        """
        ratio = np.maximum(0.0, (y - self.ref) / lam)
        return float(np.min(ratio))

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # UCB vectors: add the bonus to the reward vector before scalarization.
        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i)
            U[i] = np.clip(self.means[i] + b, 0.0, 1.0)

        # k distinct λ's → k distinct directions on the Pareto frontier.
        chosen = []
        for _ in range(self.k):
            lam = self._sample_lambda()
            scores = np.array([
                self._hv_scalarization(U[i], lam)
                for i in range(self.n)
            ])
            # Exclude already selected arms.
            for j in chosen:
                scores[j] = -np.inf
            best = int(np.argmax(scores))
            chosen.append(best)

        return chosen

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i

class ParetoUCB_TopK:
    """
	ParetoUCB+ – our extension of Drugan & Nowé (IJCNN 2013) to top-k:
	- Confidence term: sqrt(η * log(n^d * T²) / (2 * n_i))
	  standard UCB bonus parameterised by η (grid-searched).
	- Non-dominated set A' computed on UCB vectors.
	- If |A'| >= k: select k arms by sum(U) tie-break within A'.
	- If |A'| < k: take all of A', then compute next Pareto layer
	  on remaining arms, pick by sum(U), repeat.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 eta: float = 1.0, seed: int = 0, min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.eta = float(eta)
        self.min_pulls = int(min_pulls)

        self.counts = np.zeros(n, dtype=int)
        self.means = np.zeros((n, d), dtype=float)
        self._logterm = math.log(self.n * self.d * (self.T**2) + 1.0)

    def beta(self, i: int) -> float:
        denom = max(1, int(self.counts[i]))
        return math.sqrt(self.eta * self._logterm / (2.0 * denom))

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a >= b) and np.any(a > b)

    def _pareto_layers(self, U: np.ndarray) -> List[List[int]]:
        remaining = set(range(self.n))
        layers = []
        while remaining:
            rem = list(remaining)
            front = []
            for i in rem:
                dominated = False
                for j in rem:
                    if j == i:
                        continue
                    if self._dominates(U[j], U[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                remaining.remove(i)
            layers.append(front)
        return layers

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i)
            U[i] = np.clip(self.means[i] + b, 0.0, 1.0)

        layers = self._pareto_layers(U)

        chosen = []
        for layer in layers:
            if len(chosen) >= self.k:
                break
            layer_sorted = sorted(layer, key=lambda i: float(U[i].sum()), reverse=True)
            for i in layer_sorted:
                chosen.append(i)
                if len(chosen) >= self.k:
                    break
        return chosen[:self.k]

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i

class ParetoUCB_DiverseTopK(ParetoUCB_TopK):
    """
    Pareto-UCB Drugan & Nowé (IJCNN 2013) + diversity tie-break (farthest-point sampling).
    Same UCB vectors and Pareto layers, but fills k with a maximin distance rule
    to encourage coverage of the front.

    distance: "linf" or "l2"
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 eta: float = 1.0, seed: int = 0, min_pulls: int = 1,
                 distance: str = "linf"):
        super().__init__(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls)
        self.distance = distance

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.distance == "linf":
            return float(np.max(np.abs(a - b)))
        elif self.distance == "l2":
            return float(np.linalg.norm(a - b))
        else:
            raise ValueError("distance must be 'linf' or 'l2'")

    def _farthest_fill(self, candidates: List[int], U: np.ndarray, chosen: List[int], need: int) -> List[int]:
        """
        Farthest-point (maximin): iteratively pick candidate maximizing min distance to chosen.
        If chosen is empty, start from the candidate with largest sum(U).
        """
        picked = []
        cand = candidates.copy()

        # start point if none chosen yet
        if len(chosen) == 0 and len(cand) > 0:
            start = max(cand, key=lambda i: float(U[i].sum()))
            picked.append(start)
            cand.remove(start)

        while len(picked) < need and len(cand) > 0:
            best_i, best_score = None, -1e18
            for i in cand:
                # distance to already selected (chosen + picked)
                S = chosen + picked
                if len(S) == 0:
                    score = float(U[i].sum())
                else:
                    score = min(self._dist(U[i], U[j]) for j in S)
                if score > best_score:
                    best_score, best_i = score, i
            picked.append(best_i)
            cand.remove(best_i)

        return picked

    def select(self, t: int) -> List[int]:
        # forced exploration
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # UCB vectors
        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i)
            U[i] = np.clip(self.means[i] + b, 0.0, 1.0)

        # Pareto layers on U
        layers = self._pareto_layers(U)

        chosen = []
        for layer in layers:
            if len(chosen) >= self.k:
                break
            remaining = self.k - len(chosen)

            if len(layer) <= remaining:
                # if fits, still use diverse fill to order it nicely
                add = self._farthest_fill(layer, U, chosen, need=len(layer))
                chosen.extend(add)
            else:
                add = self._farthest_fill(layer, U, chosen, need=remaining)
                chosen.extend(add)

        return chosen[:self.k]

class ChebyshevUCB_Mandow:
    """
	ChebyshevUCB – faithful top-k adaptation of Mandow et al. (MODeM@ECAI 2023), Algorithm C2.

	Fidelity to the original paper:
	- S scalarization functions with varied weights (uniform spread over the simplex)
	- Shared estimates across all functions (C1/C2 variant, not C0)
	- Bonus added to mean vectors BEFORE scalarization (C2, not C1)
	- Function drawn uniformly at random each round
	- Score: f^j(U_i) = min_d { w_d * (U[i,d] - z_d) }  with z estimated nadir point

	Top-k adaptation:
	- Select the k arms with the highest score under the drawn function.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 n_scal: int = 11,
                 eta: float = 1.0,
                 eps_nadir: float = 0.01,
                 seed: int = 0,
                 min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.eta = float(eta)
        self.eps_nadir = float(eps_nadir)
        self.min_pulls = int(min_pulls)
        self.rng = np.random.default_rng(seed)

        # S scalarization functions: weights are uniformly spread over the simplex.
        # For d=2: (1,0),(0.9,0.1),…,(0,1), generalized to d dimensions.
        self.weights = self._make_weights(d, n_scal)  # (S, d)
        self.S = len(self.weights)

        # Shared estimates (C2: a single mean estimate per arm, not per scalarization function S).
        self.counts = np.zeros(n, dtype=int)
        self.means = np.zeros((n, d), dtype=float)
        self._logterm_base = math.log(max(1, T))  # log t approché par log T

    @staticmethod
    def _make_weights(d: int, n_scal: int) -> np.ndarray:
        """
	Generate n_scal weight vectors uniformly spread over the simplex.
	For d=2, reproduces exactly the 11 weights of Mandow (1,0),(0.9,0.1),...
	For d>2, uses regular sampling by recursion.
        """
        if d == 2:
            alphas = np.linspace(0.0, 1.0, n_scal)
            return np.column_stack([1.0 - alphas, alphas])
        else:
            # For d>2: a uniform simplex grid (recursive, approximate).
            from itertools import combinations_with_replacement
            step = 1.0 / (n_scal - 1) if n_scal > 1 else 1.0
            pts = []
            # Simplex grid: all weight combinations whose components sum to 1.
            n_div = n_scal - 1
            for combo in combinations_with_replacement(range(n_div + 1), d - 1):
                breaks = [0] + list(combo) + [n_div]
                w = np.diff(breaks) / n_div
                if len(w) == d:
                    pts.append(w)
            if len(pts) == 0:
                pts = [np.ones(d) / d]
            return np.array(pts, dtype=float)

    def beta(self, i: int, t: int) -> float:
        denom = max(1, int(self.counts[i]))
        # log(t) as in original UCB1 of Mandow (no log(ndT²))
        logterm = math.log(max(1, t))
        return math.sqrt(self.eta * 2.0 * logterm / denom)

    def _nadir(self) -> np.ndarray:
        """
	Estimated nadir point: coordinate-wise minimum over current empirical means,
	minus eps (as in the paper).
	For unobserved arms, 0 is used as a lower bound.
        """
        if np.all(self.counts == 0):
            return np.zeros(self.d) - self.eps_nadir
        # min sur les bras ayant été tirés au moins une fois
        pulled = self.counts > 0
        z = self.means[pulled].min(axis=0) - self.eps_nadir
        return z

    def select(self, t: int) -> List[int]:
        # Initial forced exploration.
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # UCB vectors: the bonus is added to the reward vector before scalarization (C2).
        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i, t)
            U[i] = np.clip(self.means[i] + b, 0.0, 1.0)

        # Estimated nadir point.
        z = self._nadir()

        # Randomly sample a scalarization function (as in C0/C1/C2)
        j = self.rng.integers(0, self.S)
        w = self.weights[j]  # (d,)

        # Score C2 : f^j(U_i) = min_d { w_d * (U[i,d] - z_d) }
        scores = np.min((U - z[None, :]) * w[None, :], axis=1)  # (n,)

        # Select the top-k arms according to their scores. 
        idx = np.argsort(-scores)[:self.k]
        return idx.tolist()

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i


class ParetoTSPlus_TopK:
    """
    Pareto Thompson Sampling+ - – our extension of Yahyaa & Manderick (ESANN 2015):
	- Heuristic posterior: std = sqrt((sigma_prior² + sigma_obs²) / N)
	  (slower decay than conjugate, encourages longer exploration)
	- Sample theta_i ~ N(mean_i, std_i² I) per coordinate
	- Select k arms via Pareto layers on sampled vectors, tie-break by sum(theta)

	Differs from faithful ParetoTS: slower posterior + deterministic tie-break.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 prior_mean: float = 0.5, prior_var: float = 1.0,
                 sigma_obs: float = 0.05,
                 seed: int = 0, min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.min_pulls = int(min_pulls)
        self.rng = np.random.default_rng(seed)

        # simple conjugate-ish heuristic: posterior variance shrinks as 1/N
        self.counts = np.zeros(n, dtype=int)
        self.means = np.full((n, d), float(prior_mean))
        self.prior_var = float(prior_var)
        self.sigma_obs = float(sigma_obs)

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a >= b) and np.any(a > b)

    def _pareto_layers(self, V: np.ndarray) -> List[List[int]]:
        remaining = set(range(self.n))
        layers = []
        while remaining:
            rem = list(remaining)
            front = []
            for i in rem:
                dominated = False
                for j in rem:
                    if j == i:
                        continue
                    if self._dominates(V[j], V[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                remaining.remove(i)
            layers.append(front)
        return layers

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # posterior std per arm (heuristic)
        std = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            # shrink with N; keep a floor to avoid collapse too fast
            denom = max(1, int(self.counts[i]))
            post_var = self.prior_var / denom + (self.sigma_obs**2) / denom
            std[i] = math.sqrt(post_var)

        # sample theta vectors
        theta = self.means + self.rng.normal(0.0, std[:, None], size=self.means.shape)
        theta = np.clip(theta, 0.0, 1.0)

        layers = self._pareto_layers(theta)

        chosen = []
        for layer in layers:
            if len(chosen) >= self.k:
                break
            layer_sorted = sorted(layer, key=lambda i: float(theta[i].sum()), reverse=True)
            for i in layer_sorted:
                chosen.append(i)
                if len(chosen) >= self.k:
                    break
        return chosen[:self.k]

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i

class ParetoTS_TopK:
    """
    Pareto Thompson Sampling baseline adapted to top-k:
    - Gaussian conjugate posterior: post_var = sigma_obs² / N
    - sample theta_i ~ N(mean_i, post_var * I)
    - select k arms via Pareto layers on sampled vectors, tie-break by sum(theta)
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 prior_mean: float = 0.5, prior_var: float = 1.0,
                 sigma_obs: float = 0.05,
                 seed: int = 0, min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.min_pulls = int(min_pulls)
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(n, dtype=int)
        self.means = np.full((n, d), float(prior_mean))
        self.prior_var = float(prior_var)
        self.sigma_obs = float(sigma_obs)

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a >= b) and np.any(a > b)

    def _pareto_layers(self, V: np.ndarray) -> List[List[int]]:
        remaining = set(range(self.n))
        layers = []
        while remaining:
            rem = list(remaining)
            front = []
            for i in rem:
                dominated = False
                for j in rem:
                    if j == i:
                        continue
                    if self._dominates(V[j], V[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                remaining.remove(i)
            layers.append(front)
        return layers

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # Conjugate Gaussian posterior standard deviation : sigma_obs / sqrt(N)
        # For arms that have not yet been pulled (N=0), use prior_var as a fallback.
        std = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            n_i = int(self.counts[i])
            if n_i == 0:
                std[i] = math.sqrt(self.prior_var)
            else:
                std[i] = self.sigma_obs / math.sqrt(n_i)

        # sample theta vectors
        theta = self.means + self.rng.normal(0.0, std[:, None], size=self.means.shape)
        theta = np.clip(theta, 0.0, 1.0)

        layers = self._pareto_layers(theta)

        chosen = []
        for layer in layers:
            if len(chosen) >= self.k:
                break
            layer_sorted = sorted(layer, key=lambda i: float(theta[i].sum()), reverse=True)
            for i in layer_sorted:
                chosen.append(i)
                if len(chosen) >= self.k:
                    break
        return chosen[:self.k]

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i

class ParetoTS_Faithful_TopK:
    """
	Pareto Thompson Sampling, faithful top-k adaptation of Yahyaa & Manderick (ESANN 2015).

	Fidelity to the original paper:
	- Conjugate Gaussian posterior: std = sigma_obs / sqrt(N)
	- UNIFORM random selection within the Pareto front (Step 8, Fig. 1)
	- Top-k: complete with next Pareto layers if |front| < k

	Only change from Y&M: top-k instead of single arm.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 prior_mean: float = 0.5, prior_var: float = 1.0,
                 sigma_obs: float = 0.05,
                 seed: int = 0, min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.min_pulls = int(min_pulls)
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(n, dtype=int)
        self.means = np.full((n, d), float(prior_mean))
        self.prior_var = float(prior_var)
        self.sigma_obs = float(sigma_obs)

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a >= b) and np.any(a > b)

    def _pareto_layers(self, V: np.ndarray) -> List[List[int]]:
        remaining = set(range(self.n))
        layers = []
        while remaining:
            rem = list(remaining)
            front = []
            for i in rem:
                dominated = False
                for j in rem:
                    if j == i:
                        continue
                    if self._dominates(V[j], V[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                remaining.remove(i)
            layers.append(front)
        return layers

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # Conjugate Gaussian posterior
        std = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            n_i = int(self.counts[i])
            if n_i == 0:
                std[i] = math.sqrt(self.prior_var)
            else:
                std[i] = self.sigma_obs / math.sqrt(n_i)

        # Sample theta vectors
        theta = self.means + self.rng.normal(0.0, std[:, None], size=self.means.shape)
        theta = np.clip(theta, 0.0, 1.0)

        layers = self._pareto_layers(theta)

        # Uniform sampling within each layer (Y&M Step 8, Fig. 1)
        chosen = []
        for layer in layers:
            if len(chosen) >= self.k:
                break
            need = self.k - len(chosen)
            if len(layer) <= need:
                chosen.extend(layer)
            else:
                picked = self.rng.choice(layer, size=need, replace=False).tolist()
                chosen.extend(picked)

        return chosen[:self.k]

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i            
            
class ParetoUCB_CrowdingTopK(ParetoUCB_TopK):
    """
    Pareto-UCB Drugan & Nowé (IJCNN 2013) + crowding distance (NSGA-II style) within each Pareto layer.
    - Compute UCB vectors U
    - Non-dominated sorting into layers
    - Fill k by layers; when a layer is too large, select those with largest crowding distance
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 eta: float = 1.0, seed: int = 0, min_pulls: int = 1):
        super().__init__(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls)

    @staticmethod
    def _crowding_distance(layer_idx: List[int], U: np.ndarray) -> Dict[int, float]:
        """
        NSGA-II crowding distance on points U[layer_idx] in objective space (maximize).
        Returns dict {i: crowding_distance}.
        """
        if len(layer_idx) == 0:
            return {}
        if len(layer_idx) <= 2:
            return {i: float("inf") for i in layer_idx}

        d = U.shape[1]
        dist = {i: 0.0 for i in layer_idx}

        # for each objective
        for j in range(d):
            # sort by objective value
            sorted_idx = sorted(layer_idx, key=lambda i: float(U[i, j]))
            vmin = float(U[sorted_idx[0], j])
            vmax = float(U[sorted_idx[-1], j])

            # boundaries get infinite distance
            dist[sorted_idx[0]] = float("inf")
            dist[sorted_idx[-1]] = float("inf")

            if vmax <= vmin + 1e-12:
                # all equal on this objective, skip contribution
                continue

            # internal points: normalized gap to neighbors
            for pos in range(1, len(sorted_idx) - 1):
                i_prev = sorted_idx[pos - 1]
                i_next = sorted_idx[pos + 1]
                i_cur  = sorted_idx[pos]

                if math.isinf(dist[i_cur]):
                    continue

                gap = (float(U[i_next, j]) - float(U[i_prev, j])) / (vmax - vmin)
                dist[i_cur] += gap

        return dist

    def select(self, t: int) -> List[int]:
        # forced exploration
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # UCB vectors
        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i)
            U[i] = np.clip(self.means[i] + b, 0.0, 1.0)

        layers = self._pareto_layers(U)

        chosen = []
        for layer in layers:
            if len(chosen) >= self.k:
                break
            remaining = self.k - len(chosen)

            if len(layer) <= remaining:
                # still apply crowding ordering for deterministic behavior
                cd = self._crowding_distance(layer, U)
                layer_sorted = sorted(layer, key=lambda i: (cd[i], float(U[i].sum())), reverse=True)
                chosen.extend(layer_sorted)
            else:
                cd = self._crowding_distance(layer, U)
                # pick top by crowding distance, tie-break by sum(U)
                layer_sorted = sorted(layer, key=lambda i: (cd[i], float(U[i].sum())), reverse=True)
                chosen.extend(layer_sorted[:remaining])

        return chosen[:self.k]


class ChebyshevUCB:
    """
	   ChebyshevUCB+ – our extension of Mandow et al. (MODeM@ECAI 2023) [18].
	- Fixed uniform weight w = 1/d instead of random draw each round
	- Standard UCB bonus: beta_i = sqrt(η * log(n^d * T²) / (2 * N_i))
	- Score: f(U_i) = min_j { w_j * (U[i,j] - z_j) } with z estimated nadir point
	- Select top-k arms by score

	Differs from faithful ChebyshevUCB: deterministic weight + η-parameterised bonus.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 w: Optional[np.ndarray] = None,
                 eta: float = 1.0, seed: int = 0, min_pulls: int = 1,
                 eps_w: float = 1e-8):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.eta = float(eta)
        self.min_pulls = int(min_pulls)
        self.eps_w = float(eps_w)

        self.counts = np.zeros(n, dtype=int)
        self.means = np.zeros((n, d), dtype=float)
        self._logterm = math.log(self.n * self.d * (self.T**2) + 1.0)

        if w is None:
            w = np.ones(d) / d
        w = np.asarray(w, float)
        w = np.maximum(w, self.eps_w)
        self.w = w / w.sum()

    def beta(self, i: int) -> float:
        denom = max(1, int(self.counts[i]))
        return math.sqrt(self.eta * self._logterm / (2.0 * denom))

    def select(self, t: int) -> List[int]:
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i)
            U[i] = np.clip(self.means[i] + b, 0.0, 1.0)

        # Chebyshev scalarization (maximize): min_j U_{ij} / w_j
        scores = np.min(U / self.w[None, :], axis=1)
        idx = np.argsort(-scores)[:self.k]
        return idx.tolist()

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i
            
class HVScalarUCB:
    """
	   HVScalarUCB – faithful top-k adaptation of Zhang & Golovin (ICML 2020) [37].

	Fidelity to the original paper:
	- Hypervolume scalarization: s_λ(y) = min_i ( max(0, y_i / λ_i) )^d
	  (Lemma 5, exact equation)
	- λ drawn uniformly on S^{d-1}_+ each round
	  (positive unit sphere, not the simplex)
	- UCB bonus added to mean vectors BEFORE scalarization
	  (consistent with their Algorithm 1 / UCB acquisition)

	Top-k adaptation:
	- Select the k arms with the highest score under the drawn λ,
	  natural extension to the semi-bandit setting.
    """
    def __init__(self, n: int, d: int, k: int, T: int,
                 ref: Optional[np.ndarray] = None,
                 eta: float = 1.0,
                 seed: int = 0,
                 min_pulls: int = 1):
        self.n, self.d, self.k = n, d, k
        self.T = int(T)
        self.eta = float(eta)
        self.min_pulls = int(min_pulls)
        self.rng = np.random.default_rng(seed)

        # Reference point
        self.ref = np.zeros(d, dtype=float) if ref is None else np.asarray(ref, float)

        self.counts = np.zeros(n, dtype=int)
        self.means = np.zeros((n, d), dtype=float)
        self._logterm = math.log(self.n * self.d * (self.T ** 2) + 1.0)

    def beta(self, i: int) -> float:
        denom = max(1, int(self.counts[i]))
        return math.sqrt(self.eta * self._logterm / (2.0 * denom))

    def _sample_lambda(self) -> np.ndarray:
        """
	Draw λ uniformly on S^{d-1}_+ (positive unit sphere).
	Method: sample normal, take abs, then L2-normalize.
	This is the distribution D_λ from the paper (notation S^{k-1}_+).
        """
        lam = np.abs(self.rng.standard_normal(self.d))
        norm = np.linalg.norm(lam)
        if norm < 1e-12:
            lam = np.ones(self.d)
            norm = np.linalg.norm(lam)
        return lam / norm

    def _hv_scalarization(self, y: np.ndarray, lam: np.ndarray) -> float:
        """
	s_λ(y) = min_i ( max(0, y_i / λ_i) )^d
	y = U[i] - ref  (shifted with respect to the reference point)
	Lemma 5 of Zhang & Golovin (ICML 2020).
        """
        ratio = np.maximum(0.0, y / lam)          # max(0, y_i / λ_i)
        return float(np.min(ratio) ** self.d)      # min_i(...)^d

    def select(self, t: int) -> List[int]:
        # Forced initial exploration
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        # UCB vectors: bonus added to mean vector before scalarization (Algorithm 1)
        U = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i)
            U[i] = np.clip(self.means[i] + b, 0.0, 1.0)

        # Sample λ ~ S^{d-1}_+
        lam = self._sample_lambda()

        # Hypervolume score for each arm
        scores = np.array([
            self._hv_scalarization(U[i] - self.ref, lam)
            for i in range(self.n)
        ])

        # Top-k by score
        idx = np.argsort(-scores)[:self.k]
        return idx.tolist()

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i


class THV_UCB_Practical:
    """
    THV-UCB+ (practical):
    - horizon-based confidence
    - forced exploration (min_pulls)
    - safe pruning: discard i if exists j with L[j] >= U[i] componentwise
    - greedy selection by optimistic HV marginal gain
    """
    def __init__(self, n: int, d: int, k: int, ref: np.ndarray, T: int,
                 eta: float = 1.0, seed: int = 0, min_pulls: int = 2):
        
        self.n, self.d, self.k = n, d, k
        self.ref = np.asarray(ref, float)
        self.T = int(T)
        self.eta = float(eta)
        self.min_pulls = int(min_pulls)

        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n, dtype=int)
        self.means = np.zeros((n, d), dtype=float)
        self._logterm = math.log(self.n * self.d * (self.T**2) + 1.0)

    def beta(self, i: int) -> float:
        denom = max(1, int(self.counts[i]))
        return math.sqrt(self.eta * self._logterm / (2.0 * denom))

    def select(self, t: int) -> List[int]:
        # forced exploration
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            idx = np.argsort(self.counts)[:self.k]
            return idx.tolist()

        U = np.empty_like(self.means)
        L = np.empty_like(self.means)
        for i in range(self.n):
            b = self.beta(i)
            U[i] = self.means[i] + b
            L[i] = self.means[i] - b

        # safe pruning: vectorized — U[i] dominated by some L[j] componentwise
        # dom_matrix[j, i] = True si L[j] >= U[i] for all objectives.
        dom_matrix = np.all(L[:, None, :] >= U[None, :, :], axis=2)  # (n, n)
        np.fill_diagonal(dom_matrix, False)
        dominated = dom_matrix.any(axis=0)  # Arm i is dominated by at least one arm j
        active = np.where(~dominated)[0].tolist()
        if len(active) < self.k:
            active = list(range(self.n))

        # greedy optimistic HV gain
        selected = []
        current = np.empty((0, self.d))
        cur_hv = 0.0
        for _ in range(self.k):
            best_i, best_gain = None, -1e18
            for i in active:
                if i in selected:
                    continue
                cand = np.vstack([current, U[i:i+1]])
                gain = hv(cand, self.ref) - cur_hv
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i is None:
                rest = [i for i in range(self.n) if i not in selected]
                rest = sorted(rest, key=lambda x: self.counts[x])
                best_i = rest[0]
            selected.append(best_i)
            current = np.vstack([current, U[best_i:best_i+1]])
            cur_hv = hv(current, self.ref)
        return selected

    def update(self, arms: List[int], rewards: np.ndarray):
        for idx, i in enumerate(arms):
            self.counts[i] += 1
            n_i = self.counts[i]
            self.means[i] += (rewards[idx] - self.means[i]) / n_i







# ---------- Runner / aggregation ----------

def run_once(env: SyntheticEnv, algo_name: str,
             T: int, k: int, eps: float, eta: float,
             seed: int, alpha_approx: float,
             brute_force_limit: int = 200000,
             min_pulls: int = 2,
             V_star_precomputed: Optional[float] = None) -> Dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)
    mu = env.mu
    n, d = mu.shape
    ref = env.ref

    if V_star_precomputed is not None:
        V_star = V_star_precomputed
        S_star = []
    else:
        S_star, V_star = best_eps_cover_hv(mu, k=k, eps=eps, ref=ref,
                                           brute_force_limit=brute_force_limit)
    
    # instantiate
    if algo_name == "THV-UCB":
        alg = THV_UCB_Practical(n=n, d=d, k=k, ref=ref, T=T, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "RandomK":
        alg = RandomK(n=n, k=k, seed=seed)
    elif algo_name == "ScalarUCB":
        w = np.ones(d) / d
        alg = ScalarUCB(n=n, d=d, k=k, T=T, w=w, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "ScalarUCB-RandW":
        alg = ScalarUCB_RandW(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "ParetoUCB":
        alg = ParetoUCB_TopK(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "ParetoUCB_DR":
        alg = ParetoUCB_TopK_DR(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "ParetoTS":
        # sigma_obs: use env.sigma for consistency
        alg = ParetoTS_TopK(n=n, d=d, k=k, T=T, sigma_obs=float(env.sigma), seed=seed, min_pulls=min_pulls)
    elif algo_name == "ParetoTSPlus":
        # sigma_obs: use env.sigma for consistency
        alg = ParetoTSPlus_TopK(n=n, d=d, k=k, T=T, sigma_obs=float(env.sigma), seed=seed, min_pulls=min_pulls)
    elif algo_name == "ParetoTS_Faithful_TopK":
        # sigma_obs: use env.sigma for consistency
        alg = ParetoTS_Faithful_TopK(n=n, d=d, k=k, T=T, sigma_obs=float(env.sigma), seed=seed, min_pulls=min_pulls)
    elif algo_name == "ChebyshevUCB":
        w = np.ones(d) / d
        alg = ChebyshevUCB(n=n, d=d, k=k, T=T, w=w, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "ChebyshevUCB_M":
        alg = ChebyshevUCB_Mandow(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "ParetoUCB-Div":
        alg = ParetoUCB_DiverseTopK(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls, distance="linf")
    elif algo_name == "ParetoUCB-Crowd":
        alg = ParetoUCB_CrowdingTopK(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "HVScalarUCB":
        alg = HVScalarUCB(n=n, d=d, k=k, T=T, ref=ref, eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "HVScalarUCB_Zhang2024":
        alg = HVScalarUCB_Zhang2024(n=n, d=d, k=k, T=T, ref=ref,                                eta=eta, seed=seed, min_pulls=min_pulls)
    elif algo_name == "ParetoUCB":
        alg = ParetoUCB_TopK(n=n, d=d, k=k, T=T, eta=eta, seed=seed, min_pulls=min_pulls)
        
        
        
    else:
        raise ValueError("Unknown algo_name")

    hv_trace = np.zeros(T, float)
    aregret_trace = np.zeros(T, float)

    for t in range(1, T + 1):
        arms = alg.select(t)
        rewards = env.sample(arms, rng=rng)
        alg.update(arms, rewards)

        hv_t = hv(mu[arms], ref)  # pseudo-evaluation on true means
        hv_trace[t - 1] = hv_t

        r = alpha_approx * V_star - hv_t
        aregret_trace[t - 1] = max(0.0, r)  # clipped alpha-regret

    return {
        "hv": hv_trace,
        "aregret": aregret_trace,
        "V_star": np.array([V_star], float),
        "S_star": np.array(S_star, int),
    }

def summarize_runs(runs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    hv_stack = np.stack([r["hv"] for r in runs], axis=0)
    ar_stack = np.stack([r["aregret"] for r in runs], axis=0)
    return {
        "hv_mean": hv_stack.mean(axis=0),
        "hv_std": hv_stack.std(axis=0),
        "aregret_mean": ar_stack.mean(axis=0),
        "aregret_std": ar_stack.std(axis=0),
        "V_star": runs[0]["V_star"],
        "S_star": runs[0]["S_star"],
    }

def moving_average(y, w=50):
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y
    kernel = np.ones(w) / w
    y_pad = np.pad(y, (w-1, 0), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")



