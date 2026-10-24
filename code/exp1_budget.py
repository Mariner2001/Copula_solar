"""Error budgets of the benchmark model (Fig. 2c, Fig. 5a, Extended Data Table 1).

Each budget changes one ingredient of the reconstructed error model. All budgets share the random numbers
of the driver (common random numbers). The conditional mean <X | X*> is accumulated in bins for every batch,
so that Monte Carlo standard errors follow from the spread between batches. Values at X* = 10, 15, 20 and
25 mV/m are linear interpolations between bin centres.
Usage (from results/):  python ../code/exp1_budget.py [n_batches]     (default 32 batches of 256 segments)
"""
import numpy as np, json, time, sys
from scipy import special
from sim_core import *

edges = np.concatenate([np.arange(0, 12, 0.5), np.arange(12, 27, 1.0)])
k = 1.3; g1 = special.gamma(1 + 1/k); g2 = special.gamma(1 + 2/k)
lam_mean17 = 17/g1; lam_sd25 = 25/np.sqrt(g2 - g1**2); mean_sd25 = lam_sd25*g1

variants = {
 'V0_published':  dict(dt=lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0), eps=1.0),
 'V1_dt2mean17':  dict(dt=lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=lam_mean17, shift=17.0), eps=1.0),
 'V2_dt2sd25':    dict(dt=lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=lam_sd25, shift=mean_sd25), eps=1.0),
 'V3_finitevar':  dict(dt=lambda n, r: dt1_normal(n, r, 9.0) + dt2_weibull(n, r, lam=lam_mean17, shift=17.0), eps=1.0),
 'V4_nodouble':   dict(dt=lambda n, r: dt2_weibull(n, r, lam=lam_mean17, shift=17.0), eps=1.0),
 'V5_magonly':    dict(dt=None, eps=1.0),
 'V6_timeonly':   dict(dt=lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0), eps=0.0),
 # the same budgets with a driver whose extremes persist as observed (extremogram about 0.6 at 60 min)
 'V7_persistent':        dict(dt=lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0), eps=1.0, persist=True),
 'V8_persistent_once':   dict(dt=lambda n, r: dt2_weibull(n, r, lam=lam_mean17, shift=17.0), eps=1.0, persist=True),
 'V9_persistent_timing': dict(dt=lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0), eps=0.0, persist=True),
}
PERSIST = dict(kind='mixture', mix_kappa=6.0, mix_tau=5760.0)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 32
M = 256
XQ = (10.0, 15.0, 20.0, 25.0)
out = {}
for name, v in variants.items():
    t0 = time.time()
    S, S2, C = [], [], []
    for b in range(NB):
        rng = np.random.default_rng(1000 + b)          # common random numbers across variants
        X = driver(M, rng, **(PERSIST if v.get('persist') else {}))
        Xs, Xt = measure(X, rng, dt_fn=v['dt'], eps_scale=v['eps'])
        s, s2, c = binned_mean(Xs, Xt, edges)
        S.append(s); S2.append(s2); C.append(c)
        del X, Xs, Xt
    S, S2, C = np.array(S), np.array(S2), np.array(C)
    cen = 0.5 * (edges[1:] + edges[:-1])
    tot, cnt = S.sum(0), C.sum(0)
    m = np.where(cnt >= 30, tot / np.maximum(cnt, 1), np.nan)
    at = {str(x): interp_se(S, C, cen, x) for x in XQ}
    out[name] = dict(centers=cen.tolist(), mean=[None if not np.isfinite(q) else float(q) for q in m],
                     counts=cnt.tolist(), n_batches=NB, minutes=int(NB * M * SEG),
                     at={x: [a, e] for x, (a, e) in at.items()},
                     deficit={x: [1 - a / float(x), e / float(x)] for x, (a, e) in at.items()},
                     batch_sums=S.tolist(), batch_counts=C.tolist())
    print(f"{name:21s} " + "  ".join(f"<X|X*={float(x):.0f}>={a:5.2f}+-{e:.2f}" for x, (a, e) in at.items())
          + "  deficits 20, 25: " + ", ".join(f"{1 - a / float(x):.3f}+-{e / float(x):.3f}"
                                             for x, (a, e) in at.items() if float(x) >= 20)
          + f"  [{time.time() - t0:.0f}s]", flush=True)
json.dump(out, open('exp1_budget.json', 'w'))
