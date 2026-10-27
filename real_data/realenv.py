"""Real-data (log-bootstrap) environments for Top-k Pareto bandits.

Each arm is an item; each objective is an engagement/satisfaction rate measured
on a real interaction log.  Pulling an arm returns a d-dimensional binary reward
obtained by resampling the log:
  * KuaiRand-Pure : one logged impression of the video is drawn uniformly at
                    random; the reward is its vector of binary feedback signals
                    (objectives are therefore correlated, as in the real log).
  * MovieLens-1M  : for each objective an independent draw is made in the
                    corresponding population (a rating of the movie by a user of
                    the group -> 1{rating>=4}; a random user -> 1{has not rated
                    the movie} for "discovery").
  * Goodbooks-10k : same mechanism (rating -> 1{rating==5}; random user ->
                    1{book on to-read list}; random user -> 1{not rated}).
  * Criteo        : one logged impression of the campaign is drawn; joint (click, conversion, low cost).
  * SAT12 / QBF-2014 (ASlib): for each objective (instance category) an instance of that category is
                    drawn and the reward is 1{solver solved it within the cutoff}.
The true mean vectors mu are the exact population rates, so V* is computed
exactly.  Rewards lie in {0,1}^d, hence are 1/2-sub-Gaussian.
"""
import os, itertools, pickle
import numpy as np
import pandas as pd
import model as m

from paths import DATA as D, CACHE


class BootstrapEnv:
    """Same interface as model.SyntheticEnv (mu, sigma, ref, sample)."""

    def __init__(self, name, mu, groups, obj_names, item_ids):
        # groups: list of (obj_indices, per_item_list_of_uint8_arrays (rows x len(obj_indices)))
        #         or (obj_indices, 'bern') for objectives drawn as Bernoulli(mu) (random-user draws)
        self.name, self.mu = name, np.asarray(mu, float)
        self.groups, self.obj_names, self.item_ids = groups, obj_names, item_ids
        n, d = self.mu.shape
        self.ref = np.zeros(d)
        # pooled reward standard deviation (used only by the Thompson-sampling baselines)
        self.sigma = float(np.sqrt(np.mean(self.mu * (1 - self.mu))))
        self.history = []

    def fresh(self):
        e = BootstrapEnv(self.name, self.mu, self.groups, self.obj_names, self.item_ids)
        return e

    def sample(self, arms, rng):
        arms = list(arms)
        self.history.append(arms)
        out = np.empty((len(arms), self.mu.shape[1]))
        for objs, pop in self.groups:
            if isinstance(pop, str):  # Bernoulli draw of a random user
                out[:, objs] = (rng.random((len(arms), len(objs))) < self.mu[arms][:, objs]).astype(float)
            else:
                for r, a in enumerate(arms):
                    rows = pop[a]
                    out[r, objs] = rows[rng.integers(len(rows))]
        return out


# ---------------------------------------------------------------- loaders
def _kuai():
    p = D + 'kuairand-pure/'
    df = pd.concat([pd.read_csv(p + 'log_standard_4_08_to_4_21_pure.csv'),
                    pd.read_csv(p + 'log_standard_4_22_to_5_08_pure.csv')])
    return df


def kuairand(cols, n=100):
    df = _kuai()
    top = df.video_id.value_counts().index[:n]
    s = df[df.video_id.isin(top)]
    items = list(top)
    pops = [s.loc[s.video_id == v, cols].to_numpy(np.uint8) for v in items]
    mu = np.array([p.mean(0) for p in pops])
    return mu, [(list(range(len(cols))), pops)], cols, items


def movielens(spec, n=100):
    r = pd.read_csv(D + 'ml-1m/ratings.dat', sep='::', engine='python',
                    names=['u', 'i', 'r', 't'])
    u = pd.read_csv(D + 'ml-1m/users.dat', sep='::', engine='python',
                    names=['u', 'g', 'a', 'o', 'z'])
    r = r.merge(u, on='u')
    nu = u.u.nunique()
    items = list(r.i.value_counts().index[:n])
    s = r[r.i.isin(items)]
    groups, mus, names = [], [], []
    for j, obj in enumerate(spec):
        if obj == 'discovery':
            cnt = s.groupby('i').size().reindex(items).to_numpy()
            mus.append(1 - cnt / nu); groups.append(([j], 'bern')); names.append('discovery')
            continue
        sub = s if obj == 'like' else s[s.g == obj]
        pops = [(sub.loc[sub.i == it, 'r'].to_numpy() >= 4).astype(np.uint8)[:, None] for it in items]
        mus.append(np.array([p.mean() for p in pops])); groups.append(([j], pops))
        names.append({'like': 'like', 'F': 'like (women)', 'M': 'like (men)'}[obj])
    return np.c_[tuple(mus)], groups, names, items


def goodbooks(spec, n=100):
    gr = pd.read_csv(D + 'goodbooks-10k/ratings.csv')
    tr = pd.read_csv(D + 'goodbooks-10k/to_read.csv')
    nu = 53424
    items = list(gr.book_id.value_counts().index[:n])
    s = gr[gr.book_id.isin(items)]
    groups, mus, names = [], [], []
    for j, obj in enumerate(spec):
        if obj == 'five':
            pops = [(s.loc[s.book_id == it, 'rating'].to_numpy() == 5).astype(np.uint8)[:, None] for it in items]
            mus.append(np.array([p.mean() for p in pops])); groups.append(([j], pops)); names.append('5-star')
        elif obj == 'toread':
            c = tr[tr.book_id.isin(items)].groupby('book_id').size().reindex(items).fillna(0).to_numpy()
            mus.append(c / nu); groups.append(([j], 'bern')); names.append('to-read')
        elif obj == 'discovery':
            c = s.groupby('book_id').size().reindex(items).to_numpy()
            mus.append(1 - c / nu); groups.append(([j], 'bern')); names.append('discovery')
    return np.c_[tuple(mus)], groups, names, items



def sat12(families):
    """ASlib SAT12-ALL (SATzilla 2012): arms = 31 SAT solvers; objective f = solved within the 1200 s
    cutoff on an instance of competition category f (HAND / INDU / RAND). A pull draws one instance
    of each category uniformly at random."""
    base = D + 'aslib/'
    def load(sc):
        rows, data = [], False
        for line in open(base + sc + '/algorithm_runs.arff'):
            l = line.strip()
            if l.upper() == '@DATA': data = True; continue
            if data and l and not l.startswith('%'): rows.append(l.split(',')[:5])
        return pd.DataFrame(rows, columns=['inst', 'rep', 'algo', 'rt', 'st'])
    df = load('SAT12-ALL')
    fam = {}
    for f in ['HAND', 'INDU', 'RAND']:
        for i in set(load('SAT12-' + f).inst): fam[i] = f
    df['fam'] = df.inst.map(fam); df = df[df.fam.notna()]
    df['ok'] = (df.st == 'ok').astype(np.uint8)
    items = sorted(df.algo.unique())
    groups, mus = [], []
    for j, f in enumerate(families):
        sub = df[df.fam == f]
        pops = [sub.loc[sub.algo == a, 'ok'].to_numpy(np.uint8)[:, None] for a in items]
        mus.append(np.array([p.mean() for p in pops])); groups.append(([j], pops))
    names = [{'HAND': 'crafted', 'INDU': 'industrial', 'RAND': 'random'}[f] for f in families]
    return np.c_[tuple(mus)], groups, names, items


def criteo(cols, n=100):
    """Criteo Attribution Modeling for Bidding: arms = the n most exposed campaigns; a pull draws one
    logged impression of the campaign and returns (click, conversion, cheap) jointly, where
    cheap = 1{cost <= global median cost}."""
    d = pd.read_csv(D + 'criteo/criteo_agg.csv.gz')
    w = d['count'].to_numpy()
    med = np.median(np.repeat(d.cost.to_numpy(), w))
    d['cheap'] = (d.cost <= med).astype(np.uint8)
    tot = d.groupby('campaign')['count'].sum().sort_values(ascending=False)
    items = list(tot.index[:n])
    pops = []
    for c in items:
        x = d[d.campaign == c]
        pops.append(np.repeat(x[cols].to_numpy(np.uint8), x['count'].to_numpy(), axis=0))
    mu = np.array([p.mean(0) for p in pops])
    names = {'click': 'click', 'conversion': 'conversion', 'cheap': 'low cost'}
    return mu, [(list(range(len(cols))), pops)], [names[c] for c in cols], items


def qbf14(families):
    """ASlib QBF-2014 (QBF Gallery 2014): arms = 14 QBF solvers; objective f = solved within the
    900 s cutoff on an instance of benchmark set f (applications / eval2012r2 / preproc_track_set).
    A pull draws one instance of each set uniformly at random."""
    rows, data = [], False
    for line in open(D + 'aslib/QBF-2014/algorithm_runs.arff'):
        l = line.strip()
        if l.upper() == '@DATA': data = True; continue
        if data and l and not l.startswith('%'): rows.append(l.split(',')[:5])
    df = pd.DataFrame(rows, columns=['inst', 'rep', 'algo', 'rt', 'st'])
    df['fam'] = df.inst.str.replace("'", "").str.split('/').str[0]
    df['ok'] = (df.st == 'ok').astype(np.uint8)
    items = sorted(df.algo.unique())
    groups, mus = [], []
    for j, f in enumerate(families):
        sub = df[df.fam == f]
        pops = [sub.loc[sub.algo == a, 'ok'].to_numpy(np.uint8)[:, None] for a in items]
        mus.append(np.array([p.mean() for p in pops])); groups.append(([j], pops))
    names = [{'applications': 'applications', 'eval2012r2': 'eval2012', 'preproc_track_set': 'preprocessing'}[f] for f in families]
    return np.c_[tuple(mus)], groups, names, items

CONFIGS = {
    # name: (builder, args, k)
    'ml-d2':    (movielens, (['like', 'discovery'],), 3),
    'ml-d3':    (movielens, (['F', 'M', 'discovery'],), 4),
    'kuai-d2':  (kuairand, (['long_view', 'is_profile_enter'],), 3),
    'kuai-d3':  (kuairand, (['long_view', 'is_like', 'is_profile_enter'],), 4),
    'gb-d2':    (goodbooks, (['five', 'toread'],), 3),
    'gb-d3':    (goodbooks, (['five', 'discovery', 'toread'],), 4),
    'sat-d2':   (sat12, (['INDU', 'RAND'],), 3),
    'sat-d3':   (sat12, (['HAND', 'INDU', 'RAND'],), 4),
    'crit-d2':  (criteo, (['click', 'conversion'],), 3),
    'crit-d3':  (criteo, (['click', 'conversion', 'cheap'],), 4),
    'qbf-d2':   (qbf14, (['applications', 'eval2012r2'],), 3),
    'qbf-d3':   (qbf14, (['applications', 'eval2012r2', 'preproc_track_set'],), 4),
}


def exact_vstar(mu, k, ref):
    P = m.pareto_front_indices(mu)
    if len(P) <= k:
        S = list(P) + [i for i in range(len(mu)) if i not in P][:k - len(P)]
        return S, m.hv(mu[S], ref), P
    best, bS = -1, None
    for S in itertools.combinations(P, k):
        v = m.hv(mu[list(S)], ref)
        if v > best:
            best, bS = v, list(S)
    return bS, best, P


def get_env(name):
    f = CACHE + name + '.pkl'
    if os.path.exists(f):
        with open(f, 'rb') as fh:
            return pickle.load(fh)
    builder, args, k = CONFIGS[name]
    mu, groups, names, items = builder(*args)
    env = BootstrapEnv(name, mu, groups, names, items)
    S, V, P = exact_vstar(mu, k, env.ref)
    env.k, env.S_star, env.V_star, env.front = k, S, V, P
    with open(f, 'wb') as fh:
        pickle.dump(env, fh)
    return env
