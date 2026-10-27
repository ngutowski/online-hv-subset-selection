"""Diversity of selection (Gutowski et al., SAC 2019): Div = 1 - c_v(N) / sqrt(n),
where N = (N_1, ..., N_n) counts how many times each of the n arms was included in a slate.
Computed for each algorithm at its selected eta (out/tables/summary.json), over the whole
horizon and over the last 100 rounds, averaged over seeds.  Run after analysis_real.py."""
import json, numpy as np
from paths import RUNS, OUT
from analysis_real import NAMES, CFGS, DSNAME
import realenv


def div(counts):
    counts = np.asarray(counts, float)
    return 1.0 - (counts.std() / counts.mean()) / np.sqrt(len(counts))


if __name__ == '__main__':
    S = json.load(open(OUT + 'tables/summary.json'))
    res = {}
    for c in CFGS:
        k = 4 if c.endswith('d3') else 3; n = realenv.get_env(c).mu.shape[0]
        fixed = np.zeros(n); fixed[:k] = 1
        rows = []
        for a in NAMES:
            z = np.load(f"{RUNS}{c}__{a}__{S[c][a]['eta']}.npz")['arms']
            full = np.mean([div(np.bincount(z[s].ravel(), minlength=n)) for s in range(len(z))])
            last = np.mean([div(np.bincount(z[s, -100:].ravel(), minlength=n)) for s in range(len(z))])
            rows.append((a, full, last, S[c][a]['hv']))
        res[c] = {'oracle': div(fixed), 'rows': rows}
    json.dump(res, open(OUT + 'tables/diversity.json', 'w'))
    ds = list(dict.fromkeys(c.split('-')[0] for c in CFGS))
    L = [r"\begin{tabular}{ll" + "c" * len(CFGS) + "}", r"\toprule",
         " & & " + " & ".join(r"\multicolumn{2}{c}{%s}" % DSNAME[x] for x in ds) + r"\\",
         "".join(r"\cmidrule(lr){%d-%d}" % (3 + 2 * i, 4 + 2 * i) for i in range(len(ds))),
         "Algorithm & Fid. & " + " & ".join(["$d=2$ & $d=3$"] * len(ds)) + r"\\", r"\midrule",
         "Fixed slate (e.g.\\ $S^\\star$) & & " + " & ".join(f"{res[c]['oracle']:.2f} / {res[c]['oracle']:.2f}" for c in CFGS) + r"\\", r"\midrule"]
    for a, (pn, fid) in NAMES.items():
        L.append(f"{pn} & {fid} & " + " & ".join(f"{r[1]:.2f} / {r[2]:.2f}" for c in CFGS for r in res[c]['rows'] if r[0] == a) + r"\\")
        if a == 'THV-UCB':
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(OUT + 'tables/table_diversity.tex', 'w').write("\n".join(L) + "\n")
    print("\n".join(L))
