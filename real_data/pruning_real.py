"""Instrumented THV-UCB runs: size of the active set A_t after safe pruning, and
check that pruning is lossless (identical slates with and without pruning)."""
import sys, json, time, numpy as np
from paths import OUT
import fastpatch; fastpatch.apply()
import model as m, realenv



class THVInstr(m.THV_UCB_Practical):
    prune = True

    def select(self, t):
        if self.min_pulls > 0 and np.any(self.counts < self.min_pulls):
            self.active_log.append(self.n)
            return np.argsort(self.counts)[:self.k].tolist()
        b = np.sqrt(self.eta * self._logterm / (2.0 * np.maximum(1, self.counts)))[:, None]
        U, L = self.means + b, self.means - b
        if self.prune:
            dom = np.all(L[:, None, :] >= U[None, :, :], axis=2); np.fill_diagonal(dom, False)
            active = np.where(~dom.any(axis=0))[0].tolist()
            if len(active) < self.k:
                active = list(range(self.n))
        else:
            active = list(range(self.n))
        self.active_log.append(len(active))
        selected, current, cur_hv = [], np.empty((0, self.d)), 0.0
        for _ in range(self.k):
            best_i, best_gain = None, -1e18
            for i in active:
                if i in selected: continue
                gain = m.hv(np.vstack([current, U[i:i + 1]]), self.ref) - cur_hv
                if gain > best_gain: best_gain, best_i = gain, i
            selected.append(best_i); current = np.vstack([current, U[best_i:best_i + 1]]); cur_hv = m.hv(current, self.ref)
        return selected


def run(cfg, eta, prune, seed, T=3000):
    e0 = realenv.get_env(cfg); e = e0.fresh(); rng = np.random.default_rng(seed)
    alg = THVInstr(n=e0.mu.shape[0], d=e0.mu.shape[1], k=e0.k, ref=e0.ref, T=T, eta=eta, seed=seed, min_pulls=2)
    alg.prune, alg.active_log = prune, []
    t0 = time.time(); arms = []
    for t in range(1, T + 1):
        a = alg.select(t); arms.append(a); alg.update(a, e.sample(a, rng))
    return np.array(alg.active_log), np.array(arms), time.time() - t0


if __name__ == '__main__':
    etas = json.load(open(OUT + 'tables/summary.json'))
    res = {}
    for cfg in realenv.CONFIGS:
        eta = float(etas[cfg]['THV-UCB']['eta'])
        act, same, tp, tn = [], True, 0, 0
        for s in range(10):
            a1, s1, t1 = run(cfg, eta, True, 1234 + s)
            act.append(a1 / realenv.get_env(cfg).mu.shape[0]); tp += t1
            if s < 3:
                a2, s2, t2 = run(cfg, eta, False, 1234 + s); tn += t2
                same &= bool(np.array_equal(s1, s2))
        act = np.array(act)
        res[cfg] = dict(eta=eta, active_frac_mean=act.mean(0).tolist(), final_active=float(act[:, -100:].mean()),
                        identical_slates=same, time_prune_per_run=tp / 10, time_noprune_per_run=tn / 3)
        print(cfg, 'eta', eta, 'final active frac %.3f' % res[cfg]['final_active'], 'identical', same,
              't prune %.1fs  no-prune %.1fs' % (tp / 10, tn / 3), flush=True)
    json.dump(res, open(OUT + 'tables/pruning.json', 'w'))
    # ---- LaTeX table
    DS = {'kuai': 'KuaiRand-Pure', 'ml': 'MovieLens-1M', 'gb': 'Goodbooks-10k', 'crit': 'Criteo', 'sat': 'SAT12', 'qbf': 'QBF-2014'}
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Setting & $\eta^\star$ & Active arms (avg.\ over $t$) & Active arms (last 100 rounds) & Time with / without pruning (s/run) & Speed-up\\", r"\midrule"]
    for c in [c for c in ['kuai-d2', 'kuai-d3', 'ml-d2', 'ml-d3', 'gb-d2', 'gb-d3', 'crit-d2', 'crit-d3', 'sat-d2', 'sat-d3', 'qbf-d2', 'qbf-d3'] if c in res]:
        r = res[c]; a = np.array(r['active_frac_mean'])
        L.append(f"{DS[c.split('-')[0]]}, $d={c[-1]}$ & {r['eta']:g} & {100*a.mean():.0f}\\% & {100*r['final_active']:.0f}\\% & "
                 f"{r['time_prune_per_run']:.1f} / {r['time_noprune_per_run']:.1f} & {r['time_noprune_per_run']/r['time_prune_per_run']:.1f}$\\times$\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(OUT + 'tables/table_pruning.tex', 'w').write("\n".join(L) + "\n")
