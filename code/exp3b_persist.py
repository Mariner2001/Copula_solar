"""Timing-induced deficit as a function of the persistence of the driver's extremes (Fig. 3b).

Drivers share the benchmark margin and almost its normal-score autocorrelation; the persistence of their
extremes grows with (kappa_V, tau_V). Timing errors follow the published laws, with no magnitude error.
The random numbers are those of exp1_budget.py, so the first and last settings reproduce the budgets
'timing only' with the benchmark and with the persistent driver exactly.
Usage (from results/):  python ../code/exp3b_persist.py [n_batches]     (default 32)
"""
import numpy as np, json, sys
from sim_core import *

edges = np.concatenate([np.arange(0, 12, 0.5), np.arange(12, 27, 1.0)])
cen = 0.5 * (edges[1:] + edges[:-1])
timing_pub = lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 32
res = []
for kap, tv in [(0.0, 720.), (0.9, 720.), (1.6, 1440.), (2.4, 1440.), (3.2, 2880.), (4.5, 5760.), (6.0, 5760.)]:
    S, C = [], []; e60 = []; e30 = []
    for b in range(NB):
        rng = np.random.default_rng(1000 + b)
        X = driver(256, rng, **({} if kap == 0 else dict(kind='mixture', mix_kappa=kap, mix_tau=tv)))
        if b < 4:                                   # extremograms of the true driver at its 0.999 quantile
            q = np.quantile(X, 0.999)
            for L, lst in [(60, e60), (30, e30)]:
                a = X[:, :-L] > q; bb = X[:, L:] > q; lst.append((a & bb).sum() / a.sum())
        Xs, Xt = measure(X, rng, dt_fn=timing_pub, eps_scale=0.0)
        s, _, c = binned_mean(Xs, Xt, edges); S.append(s); C.append(c)
        del X, Xs, Xt
    S, C = np.array(S), np.array(C)
    r = dict(kappa=kap, tauV=tv, ext30=float(np.mean(e30)), ext60=float(np.mean(e60)))
    for x in (15.0, 20.0, 25.0):
        v, e = interp_se(S, C, cen, x)
        r[f'm{x:.0f}'] = v; r[f'se{x:.0f}'] = e
    res.append(r)
    print({k_: round(v_, 3) for k_, v_ in r.items()},
          f"deficits 20, 25: {1 - r['m20'] / 20:.3f}+-{r['se20'] / 20:.3f}, {1 - r['m25'] / 25:.3f}+-{r['se25'] / 25:.3f}",
          flush=True)
json.dump(res, open('exp3b_persist_final.json', 'w'))
