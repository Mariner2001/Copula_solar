"""Checks behind the Matters Arising on Sivadas et al. (Nature 655, 1143, 2026).

1. Margins. Under a linear response of unit gain with conditionally unbiased noise (PCI = X + e, E[e | X] = 0),
   the index is a mean-preserving spread of the true driver, so its stop-loss transform E[(PCI - t)+] can
   only exceed that of X for every t (convex order). We compare the observed index with the log-normal
   assumed for X by the published model (mu = -0.2518, sigma = 0.85), and report exceedance probabilities.
2. Persistence of measured extremes in the published model: extremogram at 60 min of the measured driver
   for magnitude errors that are white, or autocorrelated as the observed difference E_m* - PCI (as in the
   published code), with timing errors drawn independently each minute (as in the published code).
3. Deficits of the published error budget with a driver whose extremes persist as observed, as a function
   of the scale k of the magnitude errors.
Writes results/ma_checks.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import special, optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_core import (driver, gauss_scores, binned_mean, interp_se, rel_err_profile, dt1_t, dt2_weibull,
                      measure, PAD, SEG, MU_LN, SIG_LN)

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / 'results', ROOT / 'data' / 'processed'
BLOCK, NBOOT = 2880, 1000
RNG = np.random.default_rng(20261022)
out = {}
t0 = time.time()

# ------------------------------------------------------------------ 1. margins
df = pd.read_parquet(PROC / 'polarcap_1min.parquet', columns=['time', 'Em_star', 'PCI'])
t = df.time.values
keep = (t >= np.datetime64('1995-01-01')) & (t < np.datetime64('2020-01-01')) & df.Em_star.notna().values & df.PCI.notna().values
x, y = df.Em_star.values[keep].astype(float), df.PCI.values[keep].astype(float)
blk = ((t[keep] - np.datetime64('1995-01-01T00:00')) / np.timedelta64(1, 'm')).astype(np.int64) // BLOCK
ub, inv = np.unique(blk, return_inverse=True)
W = RNG.multinomial(ub.size, np.full(ub.size, 1.0 / ub.size), size=NBOOT).astype(float)
nb = np.bincount(inv, minlength=ub.size).astype(float)
del df
ts = np.array([5.0, 10.0, 15.0, 20.0, 25.0])


def lognormal_stoploss(tt, mu=MU_LN, s=SIG_LN):
    d1 = (mu + s ** 2 - np.log(tt)) / s
    return np.exp(mu + s ** 2 / 2) * special.ndtr(d1) - tt * special.ndtr(d1 - s)


def lognormal_sf(tt, mu=MU_LN, s=SIG_LN):
    return special.ndtr(-(np.log(tt) - mu) / s)


marg = {'t': ts.tolist(), 'n_minutes': int(x.size)}
for name, v in (('PCI', y), ('Em_star', x)):
    SL = np.stack([np.bincount(inv, weights=np.maximum(v - q, 0), minlength=ub.size) for q in ts], 1)
    SF = np.stack([np.bincount(inv, weights=(v > q).astype(float), minlength=ub.size) for q in ts], 1)
    bN = W @ nb
    sl, sf = SL.sum(0) / nb.sum(), SF.sum(0) / nb.sum()
    slb, sfb = (W @ SL) / bN[:, None], (W @ SF) / bN[:, None]
    marg[name] = {'stoploss': sl.tolist(), 'stoploss_ci': [np.percentile(slb, 2.5, 0).tolist(), np.percentile(slb, 97.5, 0).tolist()],
                  'sf': sf.tolist(), 'sf_ci': [np.percentile(sfb, 2.5, 0).tolist(), np.percentile(sfb, 97.5, 0).tolist()]}
marg['lognormal'] = {'stoploss': lognormal_stoploss(ts).tolist(), 'sf': lognormal_sf(ts).tolist()}
marg['ratio_stoploss_lognormal_to_PCI'] = (lognormal_stoploss(ts) / np.array(marg['PCI']['stoploss'])).tolist()
marg['ratio_sf_lognormal_to_PCI'] = (lognormal_sf(ts) / np.array(marg['PCI']['sf'])).tolist()
# stop-loss and survival curves on a grid for the figure
gsl = np.round(np.arange(2.0, 25.01, 0.5), 2)
for name, v in (('PCI', y), ('Em_star', x)):
    SLg = np.stack([np.bincount(inv, weights=np.maximum(v - q, 0), minlength=ub.size) for q in gsl], 1)
    bN = W @ nb
    marg[name]['stoploss_grid'] = (SLg.sum(0) / nb.sum()).tolist()
    bb = (W @ SLg) / bN[:, None]
    marg[name]['stoploss_grid_ci'] = [np.percentile(bb, 2.5, 0).tolist(), np.percentile(bb, 97.5, 0).tolist()]
marg['stoploss_grid'] = gsl.tolist()
marg['lognormal']['stoploss_grid'] = lognormal_stoploss(gsl).tolist()
grid = np.round(np.arange(1.0, 30.01, 0.5), 2)
xs, ys = np.sort(x), np.sort(y)
marg['grid'] = grid.tolist()
marg['sf_grid_Em_star'] = (1 - np.searchsorted(xs, grid, side='right') / xs.size).tolist()
marg['sf_grid_PCI'] = (1 - np.searchsorted(ys, grid, side='right') / ys.size).tolist()
marg['sf_grid_lognormal'] = lognormal_sf(grid).tolist()
out['margins'] = marg
print('margins:', {k: np.round(v, 4).tolist() for k, v in marg.items() if k.startswith('ratio')}, f'[{time.time() - t0:.0f}s]', flush=True)

# ACF of the observed difference E_m* - PCI (normal scores), fitted by two exponentials
df = pd.read_parquet(PROC / 'polarcap_1min.parquet', columns=['time', 'Em_star', 'PCI'])
tt_ = df.time.values
k2 = (tt_ >= np.datetime64('1995-01-01')) & (tt_ < np.datetime64('2020-01-01'))
d = (df.Em_star.values[k2] - df.PCI.values[k2]).astype(float)
del df
ok = np.isfinite(d)
z = np.full(d.size, np.nan)
r = np.empty(ok.sum()); r[np.argsort(d[ok], kind='stable')] = np.arange(1, ok.sum() + 1)
z[ok] = special.ndtri(r / (ok.sum() + 1.0))
lags = np.unique(np.round(np.logspace(0, np.log10(3000), 30)).astype(int))
acf = []
for h in lags:
    p = np.isfinite(z[:-h]) & np.isfinite(z[h:])
    acf.append(float(np.mean(z[:-h][p] * z[h:][p])))
acf = np.array(acf)


def two_exp(l, w, t1, t2):
    return w * np.exp(-l / t1) + (1 - w) * np.exp(-l / t2)


(w_, t1_, t2_), _ = optimize.curve_fit(two_exp, lags, acf, p0=(0.5, 30, 1000), bounds=([0, 1, 10], [1, 500, 20000]))
out['diff_acf'] = {'lags': lags.tolist(), 'acf': acf.tolist(), 'fit': [float(w_), float(t1_), float(t2_)]}
print(f'ACF of E_m* - PCI: fit w={w_:.3f}, tau1={t1_:.1f} min, tau2={t2_:.0f} min; acf(60)={np.interp(60, lags, acf):.3f}', flush=True)
del d, z, x, y

# ------------------------------------------------------------------ 2. measured extremogram of the published model
timing_pub = lambda n, rr: dt1_t(n, rr) + dt2_weibull(n, rr, lam=25.0, shift=17.0)
LAG = 60


def extremogram(A, q, L=LAG):
    a = A[:, :-L] > q; b = A[:, L:] > q
    return float((a & b).sum() / a.sum())


ext = {k: [] for k in ('true', 'timing_only', 'white_eps', 'acf_eps')}
for b in range(4):
    rng = np.random.default_rng(1000 + b)
    X = driver(256, rng)
    M_, L_ = X.shape
    ext['true'].append(extremogram(X, np.quantile(X, 0.999)))
    tg = np.arange(PAD, PAD + SEG)
    dt = np.rint(timing_pub(M_ * SEG, rng)).reshape(M_, SEG)
    idx = np.clip(tg[None, :] + dt, 0, L_ - 1).astype(np.int64)
    Xp = np.take_along_axis(X, idx, axis=1)
    ext['timing_only'].append(extremogram(Xp, np.quantile(Xp, 0.999)))
    Xw = np.maximum(Xp + rel_err_profile(Xp) * Xp * rng.standard_normal(Xp.shape), 0)
    ext['white_eps'].append(extremogram(Xw, np.quantile(Xw, 0.999)))
    # magnitude error autocorrelated as E_m* - PCI, defined on the driver's time grid and sampled at t' (published code)
    N = gauss_scores(M_, L_, rng, w=(w_, 1 - w_), tau=(t1_, t2_))
    temp = np.maximum(X + rel_err_profile(X) * X * N, 0)
    Xa = np.take_along_axis(temp, idx, axis=1)
    ext['acf_eps'].append(extremogram(Xa, np.quantile(Xa, 0.999)))
    del X, Xp, Xw, N, temp, Xa
out['extremogram60_published_model'] = {k: float(np.mean(v)) for k, v in ext.items()}
SLm = np.zeros(gsl.size); nm = 0
for b in range(4):
    rng = np.random.default_rng(1000 + b)
    Xs, _ = measure(driver(256, rng), rng, dt_fn=timing_pub)
    v = np.sort(Xs.ravel()); cs = np.concatenate([[0.0], np.cumsum(v[::-1])])[::-1]
    i = np.searchsorted(v, gsl, side='right')
    SLm += cs[i] - gsl * (v.size - i); nm += v.size
    del Xs, v, cs
out['margins']['published_model_Xstar_stoploss_grid'] = (SLm / nm).tolist()
print('extremogram at 60 min, published model:', {k: round(float(np.mean(v)), 3) for k, v in ext.items()},
      f'[{time.time() - t0:.0f}s]', flush=True)

# ------------------------------------------------------------------ 3. deficits with observed persistence vs error scale
edges = np.concatenate([np.arange(0, 12, 0.5), np.arange(12, 27, 1.0)])
cen = 0.5 * (edges[1:] + edges[:-1])
PERSIST = dict(kind='mixture', mix_kappa=6.0, mix_tau=5760.0)
K13 = 1.3; LAM17 = 17 / special.gamma(1 + 1 / K13)
TIMING = {'V0': timing_pub, 'V4': lambda n, rr: dt2_weibull(n, rr, lam=LAM17, shift=17.0)}
scan = {}
for tv, tfun in TIMING.items():
    for k in (1.0, 1.25, 1.5, 2.0, 2.5):
        S, C = [], []
        for b in range(16):
            rng = np.random.default_rng(1000 + b)
            X = driver(256, rng, **PERSIST)
            Xs, Xt = measure(X, rng, dt_fn=tfun, eps_scale=k)
            s, _, c = binned_mean(Xs, Xt, edges); S.append(s); C.append(c)
            del X, Xs, Xt
        S, C = np.array(S), np.array(C)
        res = {}
        for xq in (15.0, 20.0):
            v, e = interp_se(S, C, cen, xq)
            res[str(xq)] = [1 - v / xq, e / xq]
        scan[f'{tv}_k{k}'] = res
        print(f'  persistent driver, timing {tv}, k={k}: deficit 15 = {res["15.0"][0]:.3f}, 20 = {res["20.0"][0]:.3f} [{time.time() - t0:.0f}s]', flush=True)
out['persistent_deficit_scan'] = scan
json.dump(out, open(RES / 'ma_checks.json', 'w'))
print('done', flush=True)
