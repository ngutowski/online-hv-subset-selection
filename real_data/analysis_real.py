import sys, os, glob, json, collections
import numpy as np
from paths import RUNS, OUT
import fastpatch; fastpatch.apply()
import model as m, realenv
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHA = 1 - 1 / np.e
CFGS = ['kuai-d2', 'kuai-d3', 'ml-d2', 'ml-d3', 'gb-d2', 'gb-d3', 'crit-d2', 'crit-d3', 'sat-d2', 'sat-d3', 'qbf-d2', 'qbf-d3']
DSNAME = {'kuai': 'KuaiRand-Pure', 'ml': 'MovieLens-1M', 'gb': 'Goodbooks-10k', 'crit': 'Criteo', 'sat': 'SAT12', 'qbf': 'QBF-2014'}
# code name -> (paper name, fidelity)
NAMES = collections.OrderedDict([
    ("THV-UCB", ("THV-UCB (ours)", "")),
    ("ParetoUCB_DR", ("ParetoUCB", r"$\bullet$")), ("ParetoUCB", ("ParetoUCB+", r"$\circ$")),
    ("ParetoUCB-Div", ("ParetoUCB-Div", r"$\circ$")), ("ParetoUCB-Crowd", ("ParetoUCB-Crowd", r"$\circ$")),
    ("ParetoTS_Faithful_TopK", ("ParetoTS", r"$\bullet$")), ("ParetoTSPlus", ("ParetoTS+", r"$\circ$")),
    ("ChebyshevUCB_M", ("ChebyshevUCB", r"$\bullet$")), ("ChebyshevUCB", ("ChebyshevUCB+", r"$\circ$")),
    ("HVScalarUCB", ("HVScalarUCB", r"$\bullet$")), ("HVScalarUCB_Zhang2024", ("HVScalarUCB+", r"$\bullet$")),
    ("ScalarUCB", ("ScalarUCB", r"$\circ$")), ("ScalarUCB-norm", ("ScalarUCB-Norm", r"$\circ$")), ("ScalarUCB-RandW", ("ScalarUCB-RandW", r"$\circ$")),
    ("RandomK", ("RandomK", "--")),
])
PLOT_SET = ["THV-UCB", "ScalarUCB-norm", "ParetoUCB-Crowd", "ChebyshevUCB", "HVScalarUCB_Zhang2024", "ScalarUCB", "RandomK"]
COLORS = dict(zip(PLOT_SET, ['#c0392b', '#2471a3', '#17a589', '#d68910', '#7d3c98', '#5d6d7e', '#aab7b8']))

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9,
                     'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


PRETTY = {'long_view': 'long view', 'is_like': 'like', 'is_profile_enter': 'profile visit'}

def load(cfg):
    env = realenv.get_env(cfg); V = env.V_star
    env.obj_names = [PRETTY.get(o, o) for o in env.obj_names]
    res = {}
    for a in NAMES:
        best = None
        for f in glob.glob(f"{RUNS}{cfg}__{a}__*.npz"):
            eta = f.split('__')[-1][:-4]
            z = np.load(f); hv = z['hv']
            areg = np.maximum(0, ALPHA * V - hv).sum(1) / V          # per-seed cumulative alpha-regret / V*
            reg1 = (V - hv).sum(1) / V                                  # per-seed cumulative exact regret / V*
            rec = dict(eta=eta, hv=hv / V, areg=areg, reg1=reg1, arms=z['arms'])
            hvl_ = (hv / V)[:, -100:].mean()
            if best is None or (-hvl_, areg.mean()) < (-best['hv'][:, -100:].mean(), best['areg'].mean()):
                best = rec
        if best is not None:
            res[a] = best
    return env, res


def ci(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def slate_stats(env, arms, last=100):
    """mean objective values of exposed slates and #distinct Pareto-optimal items exposed (last rounds, per seed)."""
    A = arms[:, -last:, :]
    obj = env.mu[A].mean(axis=(1, 2))                 # seeds x d
    P = set(env.front.tolist())
    npar = np.array([len(set(A[s].ravel().tolist()) & P) for s in range(A.shape[0])])
    return obj, npar


def main_table(rows_by_cfg):
    """Single LaTeX table: per config, HV last-100 (% V*) and normalized cumulative alpha-regret."""
    L = []
    L.append(r"\begin{tabular}{ll" + "rr" * 6 + "}")
    L.append(r"\toprule")
    L.append(r" & & \multicolumn{2}{c}{KuaiRand-Pure} & \multicolumn{2}{c}{MovieLens-1M} & \multicolumn{2}{c}{Goodbooks-10k} & \multicolumn{2}{c}{Criteo} & \multicolumn{2}{c}{SAT12} & \multicolumn{2}{c}{QBF-2014}\\")
    L.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}")
    L.append(r"Algorithm & Fid. & $d=2$ & $d=3$ & $d=2$ & $d=3$ & $d=2$ & $d=3$ & $d=2$ & $d=3$ & $d=2$ & $d=3$ & $d=2$ & $d=3$\\")
    L.append(r"\midrule")
    return L


def fmt_pm(x, e, nd=1):
    return f"{x:.{nd}f}{{\\scriptsize$\\pm${e:.{nd}f}}}"


def build():
    summary = {}
    data = {}
    for cfg in CFGS:
        env, res = load(cfg)
        if not res:
            continue
        data[cfg] = (env, res)
    # ---------- per-config rows
    for cfg, (env, res) in data.items():
        rows = {}
        for a, r in res.items():
            hvl = r['hv'][:, -100:].mean(1) * 100
            obj, npar = slate_stats(env, r['arms'])
            rows[a] = dict(eta=r['eta'], hv=hvl.mean(), hv_ci=ci(hvl), areg=r['areg'].mean(), areg_ci=ci(r['areg']),
                           reg1=r['reg1'].mean(), reg1_ci=ci(r['reg1']), obj=obj.mean(0).tolist(), npar=npar.mean(),
                           areg_seeds=r['areg'].tolist(), hv_seeds=hvl.tolist())
        summary[cfg] = rows
    with open(OUT + 'tables/summary.json', 'w') as f:
        json.dump({c: {a: {k: v for k, v in d.items()} for a, d in r.items()} for c, r in summary.items()}, f, indent=1)
    return data, summary


def latex_tables(data, summary):
    cfgs = [c for c in CFGS if c in summary]
    # ---------- Table 1: HV (% of V*) and cumulative alpha-regret (x V*)
    for metric, title, fn, nd, better in [('hv', 'HV', 'table_hv.tex', 1, max), ('areg', 'AREG', 'table_regret.tex', 1, min), ('reg1', 'REG1', 'table_regret_exact.tex', 0, min)]:
        L = main_table(None)
        best = {c: better(summary[c][a][metric] for a in summary[c]) for c in cfgs}
        for a, (pname, fid) in NAMES.items():
            cells = []
            for c in cfgs:
                if a not in summary[c]:
                    cells.append('--'); continue
                r = summary[c][a]
                s = fmt_pm(r[metric], r[metric + '_ci'], nd)
                if abs(r[metric] - best[c]) < 1e-9:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            L.append(f"{pname} & {fid} & " + " & ".join(cells) + r"\\")
            if a == 'THV-UCB':
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(OUT + 'tables/' + fn, 'w').write("\n".join(L) + "\n")
    # ---------- Table: configurations
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Dataset & Objectives ($d$) & $n$ & $k$ & $|\mathcal{P}^\star|$ & $V^\star$ & Log rows / arm & Noise\\", r"\midrule"]
    for c in cfgs:
        env = data[c][0]
        pops = [len(p) for objs, pop in env.groups if not isinstance(pop, str) for p in pop]
        L.append(f"{DSNAME[c.split('-')[0]]} & {', '.join(env.obj_names).replace('_', ' ')} ({env.mu.shape[1]}) & {env.mu.shape[0]} & {env.k} & "
                 f"{len(env.front)} & {env.V_star:.4f} & {min(pops):,}--{max(pops):,} & Bernoulli\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(OUT + 'tables/table_configs.tex', 'w').write("\n".join(L) + "\n")
    # ---------- Table: eta*
    L = [r"\begin{tabular}{l" + "c" * len(cfgs) + "}", r"\toprule",
         "Algorithm & " + " & ".join(f"{c.split('-')[0].upper()} {c.split('-')[1]}" for c in cfgs) + r"\\", r"\midrule"]
    for a, (pname, _) in NAMES.items():
        L.append(pname + " & " + " & ".join(summary[c][a]['eta'].replace('None', '--') if a in summary[c] else '--' for c in cfgs) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(OUT + 'tables/table_eta.tex', 'w').write("\n".join(L) + "\n")
    # ---------- Table: statistical tests (one-sided Mann-Whitney U over the 10 seeds)
    L = [r"\begin{tabular}{l lrrl lrrl}", r"\toprule",
         r" & \multicolumn{4}{c}{Final HV (\% of $V^\star$), $H_1$: THV-UCB $>$ baseline} & \multicolumn{4}{c}{Cum. $\alpha$-regret$/V^\star$, $H_1$: THV-UCB $<$ baseline}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Setting & Best baseline & THV & Base. & $p$ & Best baseline & THV & Base. & $p$\\", r"\midrule"]
    tests = {}
    for c in cfgs:
        S = summary[c]; row = f"{DSNAME[c.split('-')[0]]}, $d={c[-1]}$"
        for metric, alt, pick in [('hv', 'greater', max), ('areg', 'less', min)]:
            seeds = 'hv_seeds' if metric == 'hv' else 'areg_seeds'
            base = pick((a for a in S if a != 'THV-UCB'), key=lambda a: S[a][metric])
            x, y = np.array(S['THV-UCB'][seeds]), np.array(S[base][seeds])
            p = 1.0 if np.allclose(x, y) else wilcoxon(x, y, alternative=alt).pvalue   # paired over seeds
            tests[(c, metric)] = (base, p)
            row += f" & {NAMES[base][0]} & {x.mean():.1f} & {y.mean():.1f} & {p:.2g}"
        L.append(row + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(OUT + 'tables/table_tests.tex', 'w').write("\n".join(L) + "\n")
    # ---------- Table: slate composition (per-objective means of exposed items, last 100 rounds)
    L = [r"\begin{tabular}{ll l r}", r"\toprule",
         r"Setting & Algorithm & Mean objective values of exposed items & Pareto items exposed\\", r"\midrule"]
    show = ["THV-UCB", "ScalarUCB-norm", "HVScalarUCB_Zhang2024", "ScalarUCB", "RandomK"]
    for c in cfgs:
        env = data[c][0]
        oS = env.mu[env.S_star].mean(0)
        L.append(f"{DSNAME[c.split('-')[0]]}, $d={c[-1]}$ & $S^\\star$ (oracle) & " +
                 ", ".join(f"{nm.replace('_',' ')} {v:.3f}" for nm, v in zip(env.obj_names, oS)) + f" & {env.k}\\\\")
        for a in show:
            r = summary[c][a]
            L.append(f" & {NAMES[a][0]} & " + ", ".join(f"{v:.3f}" for v in r['obj']) + f" & {r['npar']:.1f}\\\\")
        L.append(r"\midrule" if c != cfgs[-1] else r"\bottomrule")
    L += [r"\end{tabular}"]
    open(OUT + 'tables/table_slates.tex', 'w').write("\n".join(L) + "\n")
    return tests


def ma(y, w=50):
    return m.moving_average(y, w)


def figures(data, summary):
    cfgs = [c for c in CFGS if c in summary]
    T = next(iter(data.values()))[1]['THV-UCB']['hv'].shape[1]; x = np.arange(1, T + 1)
    # ---- HV trajectories
    for kind in ['hv', 'regret']:
        fig, axes = plt.subplots(2, 6, figsize=(13.5, 4.6), sharex=True)
        for j, c in enumerate(cfgs):
            ax = axes[int(c[-1]) - 2, ['kuai', 'ml', 'gb', 'crit', 'sat', 'qbf'].index(c.split('-')[0])]
            env, res = data[c]
            for a in PLOT_SET[::-1]:
                r = res[a]
                if kind == 'hv':
                    y = r['hv']; mu_ = ma(y.mean(0)); e = ma(1.96 * y.std(0, ddof=1) / np.sqrt(len(y)))
                    ax.plot(x[::5], mu_[::5], color=COLORS[a], lw=1.6 if a == 'THV-UCB' else 1.0, label=NAMES[a][0])
                    ax.fill_between(x[::5], (mu_ - e)[::5], (mu_ + e)[::5], color=COLORS[a], alpha=0.15, lw=0)
                else:
                    y = np.cumsum(np.maximum(0, ALPHA - r['hv']), axis=1)
                    mu_ = y.mean(0); e = 1.96 * y.std(0, ddof=1) / np.sqrt(len(y))
                    ax.plot(x[::5], mu_[::5], color=COLORS[a], lw=1.6 if a == 'THV-UCB' else 1.0, label=NAMES[a][0])
                    ax.fill_between(x[::5], (mu_ - e)[::5], (mu_ + e)[::5], color=COLORS[a], alpha=0.15, lw=0)
            ax.set_title(f"{DSNAME[c.split('-')[0]]}, $d={c[-1]}$, $k={env.k}$")
            if kind == 'hv':
                ax.axhline(1.0, color='k', lw=0.6, ls=':'); ax.axhline(ALPHA, color='k', lw=0.6, ls='--')
            ax.grid(alpha=0.25, lw=0.5)
        for ax in axes[1]: ax.set_xlabel('Round $t$')
        for ax in axes[:, 0]:
            ax.set_ylabel(r'$\mathrm{HV}(S_t)/V^\star$ (moving avg.)' if kind == 'hv' else r'Cum. $\alpha$-regret $/V^\star$')
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h[::-1], l[::-1], loc='lower center', ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(OUT + f"figures/real_{kind}.pdf", bbox_inches='tight'); fig.savefig(OUT + f"figures/real_{kind}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
    # ---- objective space (d=2 configs): items, front, S*, most frequent final slate (seed 0)
    d2 = [c for c in cfgs if c.endswith('d2')]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4)); axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(np.atleast_1d(axes), d2):
        env, res = data[c]; mu = env.mu
        ax.scatter(mu[:, 0], mu[:, 1], s=7, color='#bfc9ca', lw=0, label='Items')
        P = env.front[np.argsort(mu[env.front, 0])]
        ax.plot(mu[P, 0], mu[P, 1], color='k', lw=0.6, ls='--', marker='o', ms=3, mfc='white', label='Pareto front')
        ax.scatter(mu[env.S_star, 0], mu[env.S_star, 1], s=110, marker='*', facecolor='none', edgecolor='k', lw=0.8, zorder=5, label='Optimal slate $S^\\star$')
        for a, mk, sz in [('THV-UCB', 'o', 75), ('ScalarUCB-norm', 'D', 30), ('ScalarUCB', 's', 34), ('ChebyshevUCB', '^', 34)]:
            A = res[a]['arms'][:, -100:, :].reshape(-1)
            cnt = collections.Counter(A.tolist()); top = [i for i, _ in cnt.most_common(env.k)]
            ax.scatter(mu[top, 0], mu[top, 1], s=sz, marker=mk, color=COLORS[a], alpha=0.9 if a != 'THV-UCB' else 0.55,
                       lw=0, zorder=1 if a == 'THV-UCB' else 4, label=NAMES[a][0] + ' (most exposed items)' if a == 'THV-UCB' else NAMES[a][0])
        ax.set_xlabel(env.obj_names[0].replace('_', ' ')); ax.set_ylabel(env.obj_names[1].replace('_', ' '))
        ax.set_title(DSNAME[c.split('-')[0]]); ax.grid(alpha=0.25, lw=0.5)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT + "figures/real_slates_d2.pdf", bbox_inches='tight'); fig.savefig(OUT + "figures/real_slates_d2.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------- Table: final HV with average rank and objective correlations
def table_hv_ranked(data, summary):
    import itertools
    from scipy.stats import rankdata, pearsonr
    cfgs = [c for c in CFGS if c in summary]
    algos = list(NAMES)
    ranks = {a: [] for a in algos}
    for c in cfgs:
        v = np.array([summary[c][a]['hv'] for a in algos])
        for a, r in zip(algos, rankdata(-v)):
            ranks[a].append(r)
    best = {c: max(summary[c][a]['hv'] for a in algos) for c in cfgs}
    L = [r"\begin{tabular}{ll" + "r" * len(cfgs) + "r}", r"\toprule",
         r" & & \multicolumn{2}{c}{KuaiRand-Pure} & \multicolumn{2}{c}{MovieLens-1M} & \multicolumn{2}{c}{Goodbooks-10k} & \multicolumn{2}{c}{Criteo} & \multicolumn{2}{c}{SAT12} & \multicolumn{2}{c}{QBF-2014} & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
         r"Algorithm & Fid. & $d=2$ & $d=3$ & $d=2$ & $d=3$ & $d=2$ & $d=3$ & $d=2$ & $d=3$ & $d=2$ & $d=3$ & $d=2$ & $d=3$ & Avg.\ rank\\", r"\midrule"]
    cells = []
    for c in cfgs:
        mu = data[c][0].mu; d = mu.shape[1]
        cells.append(" / ".join(f"{pearsonr(mu[:, i], mu[:, j])[0]:.2f}".replace('-', '$-$')
                                for i, j in itertools.combinations(range(d), 2)))
    L.append(r"\textit{Pairwise correlation $\rho$} & & " + " & ".join(r"{\scriptsize " + x + "}" for x in cells) + r" & \\")
    L.append(r"\midrule")
    for a in sorted(algos, key=lambda a: (a != 'THV-UCB', np.mean(ranks[a]))):
        pname, fid = NAMES[a]
        row = []
        for c in cfgs:
            r = summary[c][a]
            s = fmt_pm(r['hv'], r['hv_ci'], 1)
            if abs(r['hv'] - best[c]) < 1e-9:
                s = r"\textbf{" + s + "}"
            row.append(s)
        L.append(f"{pname} & {fid} & " + " & ".join(row) + f" & {np.mean(ranks[a]):.2f}" + r"\\")
        if a == 'THV-UCB':
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(OUT + 'tables/table_hv.tex', 'w').write("\n".join(L) + "\n")

if __name__ == '__main__':
    data, summary = build()
    tests = latex_tables(data, summary)
    table_hv_ranked(data, summary)
    figures(data, summary)
    for c in summary:
        print('==', c, 'V*=%.4g' % data[c][0].V_star)
        for a in sorted(summary[c], key=lambda a: summary[c][a]['areg']):
            r = summary[c][a]
            print(f"  {NAMES[a][0]:18s} eta={r['eta']:>5} HV%={r['hv']:6.2f}±{r['hv_ci']:.2f}  aReg/V*={r['areg']:8.1f}±{r['areg_ci']:.1f}  Reg1/V*={r['reg1']:8.1f}  nPar={r['npar']:.1f}")
        print('  test HV', tests.get((c,'hv')), ' test reg', tests.get((c,'areg')))
