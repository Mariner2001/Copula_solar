"""Empirical copula analysis of the measured solar wind driver and the ionospheric response.

Uses data/processed/*.parquet built by obs_prepare.py and the benchmark error model of sim_core.py.
Writes results/obs_analysis.json (Fig. 5 and Extended Data Figs 4-6 of the manuscript).

Uncertainties: block bootstrap with 2-day blocks of calendar time (storms are kept together); 1,000
replicates computed from per-block sufficient statistics. Monte Carlo uncertainties of benchmark curves
come from the spread over 8 independent simulation batches.
"""
import json, gc, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, special

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_core import (driver, measure, dt1_t, dt2_weibull, rel_err_profile, MU_LN, SIG_LN, PAD, SEG,
                      ACF_W, ACF_TAU)

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / 'data' / 'processed', ROOT / 'results'
T0 = np.datetime64('1995-01-01T00:00')
YEARS = (1995, 2019)
EDGES = np.concatenate([np.arange(0, 12, 0.5), np.arange(12, 27, 1.0)])     # as in the simulations
CEN = 0.5 * (EDGES[1:] + EDGES[:-1])
CEDGES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30.])          # coarse bins for classes
US = 1 - np.logspace(-1, -4, 13)                                           # 0.9 ... 0.9999
LAGS = np.array([10, 20, 30, 60, 90, 120, 180, 240])
BLOCK, NBOOT = 2880, 1000
K13 = 1.3
LAM17 = 17 / special.gamma(1 + 1 / K13)
VARIANTS = {'V0': lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0),
            'V4': lambda n, r: dt2_weibull(n, r, lam=LAM17, shift=17.0)}
RNG = np.random.default_rng(20261020)


def tolist(a):
    return [None if (v is None or not np.isfinite(v)) else float(v) for v in np.asarray(a, float).ravel()]


def ranks01(a, rng=RNG):
    """Ranks scaled to (0,1) with random tie-breaking (PCI has 10% exact zeros)."""
    o = np.lexsort((rng.random(a.size), a))
    r = np.empty(a.size)
    r[o] = np.arange(1, a.size + 1)
    return r / (a.size + 1.0)


# ------------------------------------------------------------------ data
def load_polarcap():
    cols = ['time', 'Em_star', 'PCI', 'SML', 'MA']
    df = pd.read_parquet(PROC / 'polarcap_1min.parquet', columns=cols)
    t = df.time.values
    keep = (t >= np.datetime64(f'{YEARS[0]}-01-01')) & (t < np.datetime64(f'{YEARS[1] + 1}-01-01'))
    out = {c: df[c].values[keep].astype(np.float64) for c in cols[1:]}
    out['imin'] = ((t[keep] - T0) / np.timedelta64(1, 'm')).astype(np.int64)
    del df; gc.collect()
    ms = pd.read_parquet(PROC / 'nose_multi_sc_1min.parquet', columns=['time', 'MA_ace'])
    im = ((ms.time.values - T0) / np.timedelta64(1, 'm')).astype(np.int64) + 17     # nose -> polar-cap clock
    ma_ace = np.full(out['imin'][-1] + 1, np.nan)
    ok = im < ma_ace.size
    ma_ace[im[ok]] = ms.MA_ace.values[ok]
    out['MA_ace'] = ma_ace[out['imin']]
    del ms; gc.collect()
    return out


def running_mean(a, w):
    """Centred running mean over w minutes, requiring at least half of the minutes to be valid."""
    v = np.isfinite(a)
    c = np.concatenate([[0.0], np.cumsum(np.where(v, a, 0.0))])
    n = np.concatenate([[0], np.cumsum(v)])
    h = w // 2
    i = np.arange(a.size)
    lo, hi = np.clip(i - h, 0, a.size), np.clip(i + h + 1, 0, a.size)
    cnt = n[hi] - n[lo]
    return np.where(cnt >= w // 2, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


# ------------------------------------------------------------------ block bootstrap machinery
class Blocks:
    def __init__(self, imin):
        b = imin // BLOCK
        self.ub, self.inv = np.unique(b, return_inverse=True)
        self.K = self.ub.size
        self.W = RNG.multinomial(self.K, np.full(self.K, 1.0 / self.K), size=NBOOT).astype(np.float64)

    def sums(self, weights, cell=None, ncell=1):
        """Per-block sums of weights (optionally split into cells): returns (K, ncell)."""
        if cell is None:
            w = np.full(self.inv.size, float(weights)) if np.ndim(weights) == 0 else weights
            return np.bincount(self.inv, weights=w, minlength=self.K)[:, None]
        ok = (cell >= 0) & (cell < ncell)
        key = self.inv[ok] * ncell + cell[ok]
        w = weights[ok] if np.ndim(weights) else np.full(ok.sum(), float(weights))
        return np.bincount(key, weights=w, minlength=self.K * ncell).reshape(self.K, ncell)

    def boot(self, M):
        return self.W @ M


def ci(a, axis=0):
    return np.nanpercentile(a, 2.5, axis=axis), np.nanpercentile(a, 97.5, axis=axis)


# ------------------------------------------------------------------ benchmark simulations
def simulate_pairs(variant, nbatch=8, M=256, seed0=1000):
    """Benchmark measurement pairs (X*, X) with the seeds of exp1_budget.py; returns arrays and batch ids."""
    Xs_all, X_all, bid = [], [], []
    for b in range(nbatch):
        rng = np.random.default_rng(seed0 + b)
        X = driver(M, rng)
        Xs, Xt = measure(X, rng, dt_fn=VARIANTS[variant])
        Xs_all.append(Xs.ravel().astype(np.float32)); X_all.append(Xt.ravel().astype(np.float32))
        bid.append(np.full(Xs.size, b, np.int8))
    return np.concatenate(Xs_all), np.concatenate(X_all), np.concatenate(bid)


def benchmark_with_observed_margins(variant, xobs_sorted, yobs_sorted):
    """Curve E_A[F_Y^{-1}(T) | U in bin] of the benchmark measurement copula A, evaluated with the observed
    margins of the driver (bins mapped to ranks) and of the response; chi_A(u); MC s.e. over batches."""
    Xs, X, bid = simulate_pairs(variant)
    n = Xs.size
    U = np.empty(n); U[np.argsort(Xs, kind='stable')] = np.arange(1, n + 1); U /= (n + 1.0)
    T = special.ndtr((np.log(X.astype(np.float64)) - MU_LN) / SIG_LN)
    del Xs, X; gc.collect()
    ny = yobs_sorted.size
    FinvY = yobs_sorted[np.minimum((T * ny).astype(np.int64), ny - 1)]
    u_edges = np.searchsorted(xobs_sorted, EDGES) / (xobs_sorted.size + 1.0)
    cell = np.digitize(U, u_edges) - 1
    nb = EDGES.size - 1
    out = {}
    per_batch = np.zeros((8, nb)); cnt_b = np.zeros((8, nb))
    for b in range(8):
        s = bid == b
        c = cell[s]; ok = (c >= 0) & (c < nb)
        per_batch[b] = np.bincount(c[ok], weights=FinvY[s][ok], minlength=nb)
        cnt_b[b] = np.bincount(c[ok], minlength=nb)
    tot = per_batch.sum(0) / np.maximum(cnt_b.sum(0), 1)
    mb = per_batch / np.maximum(cnt_b, 1)
    out['curve'] = np.where(cnt_b.sum(0) >= 30, tot, np.nan)
    out['curve_se'] = np.where(cnt_b.min(0) >= 5, mb.std(0, ddof=1) / np.sqrt(8), np.nan)
    chi_b = np.array([[np.mean(T[(bid == b) & (U > u)] > u) for u in US] for b in range(8)])
    out['chi'] = np.array([np.mean(T[U > u] > u) for u in US])
    out['chi_se'] = chi_b.std(0, ddof=1) / np.sqrt(8)
    zU, zT = special.ndtri(U), special.ndtri(T)
    out['rhoN'] = float(np.corrcoef(zU, zT)[0, 1])
    # check of the stochastic-increasing condition used by Proposition 6: P(U > u | T in decile) increasing
    dec = np.minimum((T * 10).astype(int), 9)
    out['si_check'] = [[float(np.mean(U[dec == d] > u)) for d in range(10)] for u in (0.9, 0.99, 0.999)]
    del U, T, FinvY, zU, zT, cell; gc.collect()
    return out


# ------------------------------------------------------------------ response analysis
def response_analysis(D, resp, blocks_cache):
    x, y, imin = D['Em_star'], D[resp], D['imin']
    if resp == 'SML':
        y = -y                                      # westward electrojet strength, positive
    m = np.isfinite(x) & np.isfinite(y)
    x, y, imin = x[m], y[m], imin[m]
    n = x.size
    B = Blocks(imin)
    blocks_cache[resp] = (B, m)
    res = {'n_minutes': int(n), 'n_blocks': int(B.K)}
    # averaged curve
    cell = np.digitize(x, EDGES) - 1
    nb = EDGES.size - 1
    S = B.sums(y, cell, nb); C = B.sums(1.0, cell, nb)
    curve = S.sum(0) / np.maximum(C.sum(0), 1)
    bc = B.boot(S) / np.maximum(B.boot(C), 1)
    lo, hi = ci(np.where(B.boot(C) >= 10, bc, np.nan))
    res.update(centers=tolist(CEN), counts=tolist(C.sum(0)), curve=tolist(np.where(C.sum(0) >= 30, curve, np.nan)),
               curve_lo=tolist(lo), curve_hi=tolist(hi))
    # quantile map at bin centres and its bootstrap via fine histograms
    xs_sorted, ys_sorted = np.sort(x), np.sort(y)
    Fx = np.searchsorted(xs_sorted, CEN) / (n + 1.0)
    Q = ys_sorted[np.minimum((Fx * n).astype(np.int64), n - 1)]
    fine = np.concatenate([[-1e9, 0.0], np.logspace(-2, np.log10(400), 1500), [1e12]])
    nf = fine.size - 1
    Hx = B.sums(1.0, np.digitize(x, fine) - 1, nf); Hy = B.sums(1.0, np.digitize(y, fine) - 1, nf)
    bHx, bHy = B.boot(Hx), B.boot(Hy)
    cFx = np.cumsum(bHx, 1) / bHx.sum(1, keepdims=True); cFy = np.cumsum(bHy, 1) / bHy.sum(1, keepdims=True)
    right = fine[1:]
    XV = np.array([10.0, 15.0, 20.0])
    Qb = np.empty((NBOOT, CEN.size)); Qbx = np.empty((NBOOT, XV.size))
    for r in range(NBOOT):
        Qb[r] = np.interp(np.interp(CEN, right, cFx[r]), cFy[r], right)
        Qbx[r] = np.interp(np.interp(XV, right, cFx[r]), cFy[r], right)
    qlo, qhi = ci(Qb)
    Qx = ys_sorted[np.minimum((np.searchsorted(xs_sorted, XV) / (n + 1.0) * n).astype(np.int64), n - 1)]
    res.update(qmap=tolist(Q), qmap_lo=tolist(qlo), qmap_hi=tolist(qhi))
    res['x_at_us'] = tolist(xs_sorted[np.minimum((US * n).astype(np.int64), n - 1)])
    # copula summaries
    U, V = ranks01(x), ranks01(y)
    zU, zV = special.ndtri(U), special.ndtri(V)
    rho = float(np.corrcoef(zU, zV)[0, 1])
    res['rhoN'] = rho
    Sz = B.sums(zU * zV); Su = B.sums(zU); Sv = B.sums(zV); Suu = B.sums(zU ** 2); Svv = B.sums(zV ** 2); Nn = B.sums(1.0)
    bs = [B.boot(a)[:, 0] for a in (Sz, Su, Sv, Suu, Svv, Nn)]
    cov = bs[0] / bs[5] - bs[1] * bs[2] / bs[5] ** 2
    rb = cov / np.sqrt((bs[3] / bs[5] - (bs[1] / bs[5]) ** 2) * (bs[4] / bs[5] - (bs[2] / bs[5]) ** 2))
    res['rhoN_ci'] = tolist(ci(rb))
    EU = np.stack([B.sums((U > u).astype(float))[:, 0] for u in US], 1)
    J = np.stack([B.sums(((U > u) & (V > u)).astype(float))[:, 0] for u in US], 1)
    chi = J.sum(0) / EU.sum(0)
    bchi = B.boot(J) / np.maximum(B.boot(EU), 1)
    clo, chi_hi = ci(bchi)
    chig = []
    for u in US:
        z = special.ndtri(u)
        chig.append(stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([-z, -z]) / (1 - u))
    chibar = [2 * np.log(1 - u) / np.log(j / n) - 1 for u, j in zip(US, J.sum(0))]
    res.update(us=tolist(US), chi=tolist(chi), chi_lo=tolist(clo), chi_hi=tolist(chi_hi), chi_gauss=tolist(chig),
               chibar=tolist(chibar))
    # Gaussian-copula counterfactual with the observed margins
    gx, gw = np.polynomial.hermite_e.hermegauss(80); gw = gw / gw.sum()
    EG = []
    for F in Fx:
        z = special.ndtri(np.clip(F, 1e-12, 1 - 1e-12))
        v = special.ndtr(rho * z + np.sqrt(1 - rho ** 2) * gx)
        EG.append(np.sum(gw * ys_sorted[np.minimum((v * n).astype(np.int64), n - 1)]))
    res['gauss_counterfactual'] = tolist(EG)
    del U, V, zU, zV, Hx, Hy, bHx, bHy, cFx, cFy; gc.collect()
    # benchmark measurement copula with the observed margins (Proposition 6 bound)
    for v in ('V0', 'V4'):
        bm = benchmark_with_observed_margins(v, xs_sorted, ys_sorted)
        res[f'bench_{v}'] = {'curve': tolist(bm['curve']), 'curve_se': tolist(bm['curve_se']), 'chi': tolist(bm['chi']),
                            'chi_se': tolist(bm['chi_se']), 'rhoN': bm['rhoN'], 'si_check': bm['si_check']}
        gc.collect()
    # deficits and channel split at exactly 10, 15 and 20 mV/m (linear interpolation between bin centres)
    if resp != 'PCI':
        return res
    split = {}
    bcm = np.where(B.boot(C) >= 10, bc, np.nan)
    for i, xv in enumerate(XV):
        j = int(np.searchsorted(CEN, xv)) - 1
        w = (xv - CEN[j]) / (CEN[j + 1] - CEN[j])
        cur = (1 - w) * curve[j] + w * curve[j + 1]
        cb = (1 - w) * bcm[:, j] + w * bcm[:, j + 1]
        tot_b, marg_b = xv - cb, xv - Qbx[:, i]
        split[str(xv)] = {'x': float(xv), 'curve': float(cur), 'curve_ci': tolist(ci(cb)),
                          'deficit': float((xv - cur) / xv), 'deficit_ci': tolist(ci(tot_b / xv)),
                          'qmap': float(Qx[i]), 'marginal_share': float((xv - Qx[i]) / (xv - cur)),
                          'marginal_share_ci': tolist(ci(marg_b / tot_b))}
    res['split'] = split
    return res


# ------------------------------------------------------------------ margins: index tail versus driver tail
def margins_test(D):
    x, y, imin = D['Em_star'], D['PCI'], D['imin']
    m = np.isfinite(x) & np.isfinite(y)
    x, y, imin = x[m], y[m], imin[m]
    B = Blocks(imin)
    grid = np.array([5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25.])
    Sx = np.stack([B.sums((x > g).astype(float))[:, 0] for g in grid], 1)
    Sy = np.stack([B.sums((y > g).astype(float))[:, 0] for g in grid], 1)
    N = B.sums(1.0)[:, 0]
    bN = B.W @ N
    sx, sy = Sx.sum(0) / N.sum(), Sy.sum(0) / N.sum()
    bsx, bsy = B.boot(Sx) / bN[:, None], B.boot(Sy) / bN[:, None]
    # convolution of the index distribution with the published heteroskedastic magnitude error, scaled by k
    cells = np.concatenate([np.arange(0, 40, 0.02), [400.]])
    cid = np.digitize(y, cells) - 1
    Hy = B.sums(1.0, cid, cells.size - 1)
    mid = 0.5 * (cells[1:] + cells[:-1]); mid[0] = 0.0
    ks = np.round(np.arange(0.0, 6.01, 0.05), 3)
    Kmat = np.zeros((mid.size, len(ks), grid.size))
    sd0 = rel_err_profile(mid) * mid
    for i, k in enumerate(ks):
        sd = np.maximum(k * sd0, 1e-9)[:, None]
        Kmat[:, i, :] = special.ndtr((mid[:, None] - grid[None, :]) / sd)       # P(max(X + eps, 0) > g), g > 0
    Kmat[0, :, :] = 0.0                                                      # X = 0 stays 0
    conv = np.einsum('c,cig->ig', Hy.sum(0), Kmat) / Hy.sum()
    bconv = np.einsum('bc,cig->big', B.boot(Hy), Kmat) / bN[:, None, None]

    def kreq(conv_k, target):
        out = []
        for g in range(grid.size):
            c = conv_k[:, g]
            if target[g] <= c[0]:
                out.append(0.0); continue
            if target[g] > c[-1]:
                out.append(np.nan); continue
            out.append(float(np.interp(target[g], c, ks)))
        return np.array(out)
    k_hat = kreq(conv, sx)
    k_b = np.array([kreq(bconv[r], bsx[r]) for r in range(NBOOT)])
    # relative error needed by a linear response if the relative s.d. were the same at all levels (flat profile)
    rs = np.round(np.arange(0.0, 1.201, 0.01), 3)
    Kf = np.zeros((mid.size, rs.size, grid.size))
    for i, rr in enumerate(rs):
        sd = np.maximum(rr * mid, 1e-9)[:, None]
        Kf[:, i, :] = special.ndtr((mid[:, None] - grid[None, :]) / sd)
    Kf[0] = 0.0
    convf = np.einsum('c,cig->ig', Hy.sum(0), Kf) / Hy.sum()
    bconvf = np.einsum('bc,cig->big', B.boot(Hy), Kf) / bN[:, None, None]

    def rreq(conv_r, target):
        out = []
        for g in range(grid.size):
            c = conv_r[:, g]
            if target[g] <= c[0]:
                out.append(0.0); continue
            if target[g] > c[-1]:
                out.append(np.nan); continue
            out.append(float(np.interp(target[g], c, rs)))
        return np.array(out)
    r_hat = rreq(convf, sx)
    r_b = np.array([rreq(bconvf[r], bsx[r]) for r in range(NBOOT)])
    i1 = int(np.where(ks == 1.0)[0][0]); i2 = int(np.where(ks == 2.0)[0][0])
    res = {'grid': tolist(grid), 'S_driver': tolist(sx), 'S_driver_ci': [tolist(a) for a in ci(bsx)],
           'S_index': tolist(sy), 'S_index_ci': [tolist(a) for a in ci(bsy)],
           'S_conv_published': tolist(conv[i1]), 'S_conv_published_ci': [tolist(a) for a in ci(bconv[:, i1, :])],
           'S_conv_double': tolist(conv[i2]),
           'k_required': tolist(k_hat), 'k_required_ci': [tolist(a) for a in ci(k_b)],
           'k_required_nan_frac': tolist(np.mean(~np.isfinite(k_b), 0)),
           'rel_err_published_high': 0.30,
           'r_required_flat': tolist(r_hat), 'r_required_flat_ci': [tolist(a) for a in ci(r_b)],
           'r_published_at_grid': tolist(rel_err_profile(grid))}
    # gain check: least-squares slope of <PCI|E*> for E* < 2 mV/m
    s = x < 2
    res['bulk_slope'] = float(np.sum(x[s] * y[s]) / np.sum(x[s] ** 2))
    return res


# ------------------------------------------------------------------ serial dependence and tails
def serial_and_tails(D):
    out = {}
    x = D['Em_star']; imin = D['imin']
    full = np.full(imin[-1] - imin[0] + 1, np.nan)
    full[imin - imin[0]] = x
    v = np.isfinite(full)
    q = np.nanquantile(full, 0.999)
    ext, ext_lo, ext_hi = [], [], []
    blk = (np.arange(full.size) + imin[0]) // BLOCK
    ub, inv = np.unique(blk, return_inverse=True)
    W = RNG.multinomial(ub.size, np.full(ub.size, 1.0 / ub.size), size=NBOOT).astype(float)
    for h in LAGS:
        a = (full[:-h] > q) & v[h:]
        j = a & (full[h:] > q)
        A = np.bincount(inv[:-h], weights=a, minlength=ub.size); Jb = np.bincount(inv[:-h], weights=j, minlength=ub.size)
        ext.append(Jb.sum() / A.sum())
        b = (W @ Jb) / np.maximum(W @ A, 1)
        lo, hi = ci(b); ext_lo.append(lo); ext_hi.append(hi)
    out['extremogram'] = {'lags': tolist(LAGS), 'Em_star': tolist(ext), 'lo': tolist(ext_lo), 'hi': tolist(ext_hi),
                          'threshold': float(q)}
    # benchmark: extremogram of the true driver (exp3) is reused in the figure; here the measured benchmark series
    # normal-score autocorrelation of the driver and of the index
    acf = {}
    lags_acf = np.unique(np.concatenate([np.round(np.logspace(0, np.log10(3000), 30)).astype(int), [30, 60, 120]]))
    for name in ('Em_star', 'PCI'):
        f = np.full(full.size, np.nan)
        f[imin - imin[0]] = D[name]
        ok = np.isfinite(f)
        z = np.full(f.size, np.nan)
        z[ok] = special.ndtri(ranks01(f[ok]))
        r = []
        for h in lags_acf:
            p = np.isfinite(z[:-h]) & np.isfinite(z[h:])
            r.append(np.mean(z[:-h][p] * z[h:][p]))
        acf[name] = tolist(r)
    acf['lags'] = tolist(lags_acf)
    acf['benchmark_fit'] = tolist(sum(w * np.exp(-lags_acf / t) for w, t in zip(ACF_W, ACF_TAU)))
    out['acf'] = acf
    # Hill estimates of the upper-tail index (top 1 % of values; block bootstrap with 200 replicates)
    ks = np.unique(np.round(np.logspace(2, np.log10(30000), 25)).astype(int))
    hill = {'k': tolist(ks)}
    for name in ('Em_star', 'PCI'):
        vals = D[name][np.isfinite(D[name])]; bl = D['imin'][np.isfinite(D[name])] // BLOCK
        thr = np.quantile(vals, 0.99)
        sel = vals > thr
        tv, tb = vals[sel], bl[sel]
        srt = np.sort(tv)[::-1]
        alpha = [1.0 / np.mean(np.log(srt[:k] / srt[k])) for k in ks]
        hill[name + '_threshold'] = tolist([srt[k] for k in ks])             # level of the k-th largest value
        ub2, inv2 = np.unique(bl, return_inverse=True)
        # map top values to block positions
        pos = np.searchsorted(ub2, tb)
        boots = []
        for r in range(200):
            w = RNG.multinomial(ub2.size, np.full(ub2.size, 1.0 / ub2.size))
            rep = np.repeat(tv, w[pos])
            s2 = np.sort(rep)[::-1]
            boots.append([1.0 / np.mean(np.log(s2[:k] / s2[k])) if s2.size > k else np.nan for k in ks])
        lo, hi = ci(np.array(boots))
        hill[name] = tolist(alpha); hill[name + '_lo'] = tolist(lo); hill[name + '_hi'] = tolist(hi)
    # benchmark measured driver X* (V0) and true driver
    Xs, X, _ = simulate_pairs('V0', nbatch=4)
    for name, vals in (('bench_Xstar', Xs.astype(float)), ('bench_X', X.astype(float))):
        srt = np.sort(vals[vals > np.quantile(vals, 0.99)])[::-1]
        hill[name] = tolist([1.0 / np.mean(np.log(srt[:k] / srt[k])) for k in ks])
    out['hill'] = hill
    return out


# ------------------------------------------------------------------ two upstream monitors
def monitors():
    cols = ['time', 'Em_wind', 'Em_ace', 'Em_geotail']
    ms = pd.read_parquet(PROC / 'nose_multi_sc_1min.parquet', columns=cols)
    imin = ((ms.time.values - T0) / np.timedelta64(1, 'm')).astype(np.int64)
    out = {}
    bins = np.array([0.5, 1, 2, 3, 5, 7, 10, 14, 20, 30])
    for a, b in (('Em_wind', 'Em_geotail'), ('Em_wind', 'Em_ace'), ('Em_ace', 'Em_geotail')):
        s = np.isfinite(ms[a].values) & np.isfinite(ms[b].values)
        xa, xb, im = ms[a].values[s].astype(float), ms[b].values[s].astype(float), imin[s]
        U, V = ranks01(xa), ranks01(xb)
        rho = float(np.corrcoef(special.ndtri(U), special.ndtri(V))[0, 1])
        B = Blocks(im)
        EU = np.stack([B.sums((U > u).astype(float))[:, 0] for u in US], 1)
        J = np.stack([B.sums(((U > u) & (V > u)).astype(float))[:, 0] for u in US], 1)
        chi = J.sum(0) / EU.sum(0); lo, hi = ci(B.boot(J) / np.maximum(B.boot(EU), 1))
        mean = 0.5 * (xa + xb); rel = (xa - xb) / np.maximum(mean, 1e-9)
        c = np.digitize(mean, bins) - 1
        nb = bins.size - 1
        S1 = B.sums(rel, c, nb); S2 = B.sums(rel ** 2, c, nb); Cn = B.sums(1.0, c, nb)
        sd = np.sqrt(S2.sum(0) / np.maximum(Cn.sum(0), 1) - (S1.sum(0) / np.maximum(Cn.sum(0), 1)) ** 2)
        bS1, bS2, bC = B.boot(S1), B.boot(S2), B.boot(Cn)
        bsd = np.sqrt(np.maximum(bS2 / np.maximum(bC, 1) - (bS1 / np.maximum(bC, 1)) ** 2, 0))
        slo, shi = ci(np.where(bC >= 10, bsd, np.nan))
        out[f'{a[3:]}_{b[3:]}'] = {'n': int(s.sum()), 'rhoN': rho, 'us': tolist(US), 'chi': tolist(chi), 'chi_lo': tolist(lo),
                                    'chi_hi': tolist(hi), 'bins': tolist(bins), 'bin_counts': tolist(Cn.sum(0)),
                                    'rel_sd': tolist(np.where(Cn.sum(0) >= 30, sd, np.nan)), 'rel_sd_lo': tolist(slo),
                                    'rel_sd_hi': tolist(shi)}
    del ms; gc.collect()
    # benchmark prediction for an L1 monitor against a near-Earth one: timing error dt1 only
    ch, ch_b, rho = [], [], []
    for b in range(4):
        rng = np.random.default_rng(3000 + b)
        X = driver(256, rng)
        Xs, Xt = measure(X, rng, dt_fn=lambda n, r: dt1_t(n, r), eps_scale=0.0)
        U, V = ranks01(Xs.ravel()), ranks01(Xt.ravel())
        ch_b.append([np.mean(V[U > u] > u) for u in US])
        rho.append(np.corrcoef(special.ndtri(U), special.ndtri(V))[0, 1])
    ch_b = np.array(ch_b)
    out['bench_dt1_only'] = {'chi': tolist(ch_b.mean(0)), 'chi_se': tolist(ch_b.std(0, ddof=1) / 2), 'rhoN': float(np.mean(rho))}
    ch_b, rho = [], []
    for b in range(4):
        rng = np.random.default_rng(3000 + b)
        X = driver(256, rng, kind='mixture', mix_kappa=6.0, mix_tau=5760.0)
        Xs, Xt = measure(X, rng, dt_fn=lambda n, r: dt1_t(n, r), eps_scale=0.0)
        U, V = ranks01(Xs.ravel()), ranks01(Xt.ravel())
        ch_b.append([np.mean(V[U > u] > u) for u in US])
        rho.append(np.corrcoef(special.ndtri(U), special.ndtri(V))[0, 1])
    ch_b = np.array(ch_b)
    out['bench_dt1_only_persistent'] = {'chi': tolist(ch_b.mean(0)), 'chi_se': tolist(ch_b.std(0, ddof=1) / 2),
                                        'rhoN': float(np.mean(rho))}
    return out


# ------------------------------------------------------------------ hemispheric indices: noise of the response
def hemispheres():
    df = pd.read_parquet(PROC / 'polarcap_1min.parquet', columns=['time', 'Em_star', 'PCN', 'PCS', 'PCI'])
    t = df.time.values
    m = ((t >= np.datetime64(f'{YEARS[0]}-01-01')) & (t < np.datetime64(f'{YEARS[1] + 1}-01-01'))
         & df.Em_star.notna().values & df.PCN.notna().values & df.PCS.notna().values)
    im = ((t[m] - T0) / np.timedelta64(1, 'm')).astype(np.int64)
    N = np.maximum(df.PCN.values[m].astype(float), 0); S = np.maximum(df.PCS.values[m].astype(float), 0)
    E = df.Em_star.values[m].astype(float); P = df.PCI.values[m].astype(float)
    del df; gc.collect()
    uN, uS, uE, uP = ranks01(N), ranks01(S), ranks01(E), ranks01(P)
    z = {k: special.ndtri(v) for k, v in dict(N=uN, S=uS, E=uE, P=uP).items()}
    c = lambda a, b: float(np.corrcoef(z[a], z[b])[0, 1])
    rNS, rEN, rES, rEP = c('N', 'S'), c('E', 'N'), c('E', 'S'), c('E', 'P')
    B = Blocks(im)
    EU = np.stack([B.sums((uN > u).astype(float))[:, 0] for u in US], 1)
    J = np.stack([B.sums(((uN > u) & (uS > u)).astype(float))[:, 0] for u in US], 1)
    chi = J.sum(0) / EU.sum(0); lo, hi = ci(B.boot(J) / np.maximum(B.boot(EU), 1))
    bins = np.array([2, 4, 6, 8, 10, 15, 25.])
    rel = []
    for a_, b_ in zip(bins[:-1], bins[1:]):
        s_ = (P >= a_) & (P < b_)
        d = (N[s_] - S[s_]) / P[s_]
        rel.append([float(a_), float(b_), int(s_.sum()), float(d.std()), float(np.median(np.abs(d)))])
    lamE = np.sqrt(rEN * rES / rNS)
    out = {'n': int(E.size), 'rhoN_NS': rNS, 'rhoN_EN': rEN, 'rhoN_ES': rES, 'rhoN_EP': rEP,
           'lambda_E': float(lamE), 'lambda_N': float(np.sqrt(rEN * rNS / rES)), 'lambda_S': float(np.sqrt(rES * rNS / rEN)),
           'us': tolist(US), 'chi_NS': tolist(chi), 'chi_NS_lo': tolist(lo), 'chi_NS_hi': tolist(hi),
           'chi_EN': tolist([np.mean(uN[uE > u] > u) for u in US]), 'chi_ES': tolist([np.mean(uS[uE > u] > u) for u in US]),
           'rel_disagreement': rel}
    return out


# ------------------------------------------------------------------ Mach-number classes
def classes(D, blocks_cache):
    B, m = blocks_cache['PCI']
    x, y = D['Em_star'][m], D['PCI'][m]
    out = {'edges': tolist(CEDGES), 'thresholds': [3.3, 7.2]}
    ma_2h = running_mean(D['MA'], 120)
    defs = {'wind_1min': D['MA'][m], 'wind_2h': ma_2h[m], 'ace_1min': D['MA_ace'][m]}
    nb = CEDGES.size - 1
    cell = np.digitize(x, CEDGES) - 1
    for name, ma in defs.items():
        res = {}
        for lab, sel in (('low', ma < 3.3), ('mid', (ma >= 3.3) & (ma < 7.2)), ('high', ma >= 7.2)):
            w = sel.astype(float)
            S = B.sums(y * w, cell, nb); C = B.sums(w, cell, nb)
            cur = S.sum(0) / np.maximum(C.sum(0), 1)
            bc = B.boot(S) / np.maximum(B.boot(C), 1)
            lo, hi = ci(np.where(B.boot(C) >= 10, bc, np.nan))
            xs_, ys_ = x[sel], y[sel]
            U, V = ranks01(xs_), ranks01(ys_)
            rho = float(np.corrcoef(special.ndtri(U), special.ndtri(V))[0, 1])
            hi_sel = xs_ > 5
            rho_hi = float(np.corrcoef(np.log(xs_[hi_sel]), np.log(np.maximum(ys_[hi_sel], 0.05)))[0, 1]) if hi_sel.sum() > 100 else None
            res[lab] = {'curve': tolist(np.where(C.sum(0) >= 50, cur, np.nan)), 'lo': tolist(lo), 'hi': tolist(hi),
                        'counts': tolist(C.sum(0)), 'rhoN': rho, 'corr_log_above5': rho_hi, 'n': int(sel.sum())}
        # ordering contrast at 8-12 mV/m
        j0, j1 = int(np.searchsorted(CEDGES, 8)), int(np.searchsorted(CEDGES, 12))
        wl, wh = (ma < 3.3).astype(float), (ma >= 7.2).astype(float)
        cs = (cell >= j0) & (cell < j1)
        Sl = B.sums(y * wl * cs); Cl = B.sums(wl * cs); Sh = B.sums(y * wh * cs); Ch = B.sums(wh * cs)
        d = Sh.sum() / Ch.sum() - Sl.sum() / Cl.sum()
        db = (B.boot(Sh) / B.boot(Ch) - B.boot(Sl) / B.boot(Cl))[:, 0]
        res['contrast_8_12'] = {'high_minus_low': float(d), 'ci': tolist(ci(db)),
                                'mean_x_low': float(np.sum(x[(ma < 3.3) & (x >= 8) & (x < 12)]) / max(1, np.sum((ma < 3.3) & (x >= 8) & (x < 12)))),
                                'mean_x_high': float(np.sum(x[(ma >= 7.2) & (x >= 8) & (x < 12)]) / max(1, np.sum((ma >= 7.2) & (x >= 8) & (x < 12))))}
        out[name] = res
    return out


def main(sections=None):
    t0 = time.time()
    path = RES / 'obs_analysis.json'
    out = json.load(open(path)) if (sections and path.exists()) else {}
    out['meta'] = {'years': list(YEARS), 'block_minutes': BLOCK, 'n_boot': NBOOT, 'delay_nose_to_polar_cap_min': 17}
    want = lambda k: sections is None or k in sections
    D = None
    if any(want(k) for k in ('PCI', 'SML', 'margins', 'classes', 'serial')):
        D = load_polarcap()
        print(f'loaded {D["imin"].size:,} minutes [{time.time() - t0:.0f}s]', flush=True)
    cache = {}
    for resp in ('PCI', 'SML'):
        if want(resp) or (resp == 'PCI' and want('classes')):
            r = response_analysis(D, resp, cache)
            if want(resp):
                out[resp] = r
            print(f'{resp}: n={r["n_minutes"]:,} rhoN={r["rhoN"]:.3f} [{time.time() - t0:.0f}s]', flush=True)
            json.dump(out, open(path, 'w'))
    for name, fn in (('margins', lambda: margins_test(D)), ('classes', lambda: classes(D, cache)),
                     ('serial', lambda: serial_and_tails(D))):
        if want(name):
            out[name] = fn()
            print(f'{name} done [{time.time() - t0:.0f}s]', flush=True)
            json.dump(out, open(path, 'w'))
    del D; gc.collect()
    for name, fn in (('hemispheres', hemispheres), ('monitors', monitors)):
        if want(name):
            out[name] = fn()
            print(f'{name} done [{time.time() - t0:.0f}s]', flush=True)
            json.dump(out, open(path, 'w'))


if __name__ == '__main__':
    main(sys.argv[1:] or None)
