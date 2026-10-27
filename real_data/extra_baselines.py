"""Additional baseline for the real-data study: ScalarUCB with online min-max normalization
of the objectives (weights w_j(t) proportional to 1/range_j of the current empirical means)."""
import math, numpy as np

class ScalarUCBNorm:
    def __init__(self, n, d, k, T, eta=1.0, seed=0, min_pulls=2):
        self.n, self.d, self.k, self.eta, self.min_pulls = n, d, k, float(eta), int(min_pulls)
        self.counts = np.zeros(n, int); self.means = np.zeros((n, d))
        self._logterm = math.log(n * T ** 2 + 1.0)          # same radius as ScalarUCB

    def select(self, t):
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            return np.argsort(self.counts)[:self.k].tolist()
        rng = np.maximum(self.means.max(0) - self.means.min(0), 1e-6)
        w = (1.0 / rng) / (1.0 / rng).sum()
        ucb = self.means @ w + np.sqrt(self.eta * self._logterm / (2.0 * np.maximum(1, self.counts)))
        return np.argsort(-ucb)[:self.k].tolist()

    def update(self, arms, rewards):
        for a, r in zip(arms, rewards):
            self.counts[a] += 1
            self.means[a] += (r - self.means[a]) / self.counts[a]
