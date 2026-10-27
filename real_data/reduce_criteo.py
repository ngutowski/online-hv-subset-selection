"""Reduce the Criteo Attribution Modeling for Bidding log (pcb_dataset_final.tsv, ~2.4 GB, 16.5M rows)
to the aggregated file used by realenv.criteo():  data/criteo/criteo_agg.csv.gz with columns
campaign, click, conversion, cost, count  (count = number of impressions with these four values).

Usage:  python3 reduce_criteo.py path/to/pcb_dataset_final.tsv[.gz]
Cost values are kept as written in the log (no float re-formatting), so the result does not depend
on the locale of the machine."""
import sys, csv, gzip, os, collections
from paths import DATA

src = sys.argv[1]
op = gzip.open if src.endswith('.gz') else open
cnt = collections.Counter()
with op(src, 'rt', newline='') as f:
    r = csv.reader(f, delimiter='\t')
    h = next(r)
    ic, ik, iv, ico = h.index('campaign'), h.index('click'), h.index('conversion'), h.index('cost')
    for row in r:
        cnt[(row[ic], row[ik], row[iv], row[ico])] += 1
os.makedirs(DATA + 'criteo', exist_ok=True)
out = DATA + 'criteo/criteo_agg.csv.gz'
with gzip.open(out, 'wt', newline='') as g:
    w = csv.writer(g); w.writerow(['campaign', 'click', 'conversion', 'cost', 'count'])
    for k, v in sorted(cnt.items()): w.writerow(list(k) + [v])
print(f"{sum(cnt.values())} impressions, {len({k[0] for k in cnt})} campaigns -> {out}")
