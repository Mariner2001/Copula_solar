"""Serial dependence of the model drivers (Fig. 3a, Fig. 5b).

Extremogram at the 0.999 quantile and normal-score autocorrelation of the true driver for the benchmark
log-Gaussian process, a driver with moderately persistent extremes and the persistent driver whose
extremogram matches the observed one; the extremogram of the measured benchmark driver (published error
budget) shows how measurement error changes persistence. Random numbers as in exp1_budget.py.
"""
import numpy as np, json
from scipy import stats
from sim_core import *
from copula_tools import *

lags = np.array([10, 20, 30, 60, 90, 120, 180, 240, 480, 1440])
timing_pub = lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0)
NB = 4
out = {}


def extremogram(A, q, L):
    a = A[:, :-L] > q; b = A[:, L:] > q
    return (a & b).sum() / a.sum()


for kind, kw in [('gauss', {}), ('mixture', dict(kind='mixture', mix_kappa=0.9, mix_tau=720.0)),
                 ('persistent', dict(kind='mixture', mix_kappa=6.0, mix_tau=5760.0))]:
    ext = np.zeros(len(lags)); acfN = np.zeros(len(lags)); extm = np.zeros(len(lags))
    for b in range(NB):
        rng = np.random.default_rng(1000 + b)
        X = driver(256, rng, **kw)
        z = stats.norm.ppf(ranks01(X)).reshape(X.shape)
        q = np.quantile(X, 0.999)
        for i, L in enumerate(lags):
            ext[i] += extremogram(X, q, L) / NB
            acfN[i] += np.mean(z[:, :-L] * z[:, L:]) / NB
        del z
        Xs, _ = measure(X, rng, dt_fn=timing_pub, eps_scale=1.0)
        qs = np.quantile(Xs, 0.999)
        for i, L in enumerate(lags):
            extm[i] += extremogram(Xs, qs, L) / NB
        del X, Xs
    print(f"{kind:10s} extremogram (true driver) at lags {lags.tolist()}: {np.round(ext, 3).tolist()}")
    print(f"{'':10s} extremogram (measured, published budget):  {np.round(extm, 3).tolist()}")
    print(f"{'':10s} normal-score ACF (true driver):            {np.round(acfN, 3).tolist()}", flush=True)
    out[kind] = dict(lags=lags.tolist(), ext=ext.tolist(), ext_measured=extm.tolist(), acfN=acfN.tolist())
json.dump(out, open('exp3_serial.json', 'w'))
