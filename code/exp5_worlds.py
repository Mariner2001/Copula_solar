"""Two-world identifiability experiment.
World L: linear response, large (published) error budget.
World S: saturating response g(x)=x/(1+x/xs) with response noise, smaller and finite-variance error budget.
Both worlds: noisy magnetosheath reference at conjunctions (point-measurement / definition noise).
Applied to each world:
  (i)  conditional-mean regression calibration with an error budget estimated from conjunctions against the
       sheath reference plus separately imposed propagation-time errors (published-style pipeline);
  (ii) factor-copula (triad) estimator of the response function.
"""
import numpy as np, json, sys
from scipy import stats, special
from sim_core import *
from copula_tools import ranks01

K = 1.3; G1 = special.gamma(1 + 1/K); LAM17 = 17/G1
EDGES = np.concatenate([np.arange(0, 12, 0.5), np.arange(12, 27, 1.0)])
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80); GH_W = GH_W / GH_W.sum()

WORLDS = {
    'L': dict(dt1=lambda n, r: dt1_t(n, r), dt2=lambda n, r: dt2_weibull(n, r, lam=25.0, shift=17.0),
              rel=rel_err_profile, g=lambda x: x, s_eta=0.10),
    'S': dict(dt1=lambda n, r: dt1_normal(n, r, 9.0), dt2=lambda n, r: dt2_weibull(n, r, lam=LAM17, shift=17.0),
              rel=lambda x: 0.20 + 0 * x, g=None, s_eta=0.20),
}
S_SH = 0.30          # log-s.d. of the magnetosheath reference noise
CONJ_FRAC = 0.01     # fraction of minutes with conjunctions (in 2-h blocks)


def make_world(name, xs, nbatch, M, seed):
    w = WORLDS[name]
    g = w['g'] if w['g'] is not None else (lambda x: x / (1 + x / xs))
    out = {k: [] for k in ['Xs', 'Y', 'cXs', 'cXsh', 'cY', 'cX']}
    accObs = Accum(EDGES)
    for b in range(nbatch):
        rng = np.random.default_rng(seed + b)
        X = driver(M, rng)
        Mx, L = X.shape
        t = np.arange(PAD, PAD + SEG)
        n = Mx * SEG
        d1 = np.rint(w['dt1'](n, rng)).reshape(Mx, SEG)
        d2 = np.rint(w['dt2'](n, rng)).reshape(Mx, SEG)
        Xt = X[:, PAD:PAD + SEG]
        Xl1 = np.take_along_axis(X, np.clip(t[None] + d1, 0, L - 1).astype(np.int64), 1)
        Xs = np.maximum(Xl1 + w['rel'](Xl1) * Xl1 * rng.standard_normal(Xt.shape), 0)   # L1 estimate (nose time)
        Xpc = np.take_along_axis(X, np.clip(t[None] - d2, 0, L - 1).astype(np.int64), 1)  # driver felt at polar cap
        Y = g(Xpc) * np.exp(w['s_eta'] * rng.standard_normal(Xt.shape) - w['s_eta'] ** 2 / 2)
        Xsh = Xt * np.exp(S_SH * rng.standard_normal(Xt.shape) - S_SH ** 2 / 2)
        # conjunction blocks of 120 min
        nb = SEG // 120
        pick = rng.random((Mx, nb)) < CONJ_FRAC
        mask = np.repeat(pick, 120, axis=1)
        mask = np.concatenate([mask, np.zeros((Mx, SEG - mask.shape[1]), bool)], axis=1)
        out['Xs'].append(Xs.ravel()); out['Y'].append(Y.ravel())
        out['cXs'].append(Xs[mask]); out['cXsh'].append(Xsh[mask]); out['cY'].append(Y[mask]); out['cX'].append(Xt[mask])
        accObs.add(Xs, Y)
    for k in out:
        out[k] = np.concatenate(out[k])
    out['obs_curve'] = accObs.mean(30)
    out['g'] = g
    return out


def pipeline(world, nbatch_mc, M, seed):
    """Published-style regression calibration."""
    Y = world['Y']; ly = np.log(Y[Y > 0]); muY, sY = ly.mean(), ly.std()
    # magnitude uncertainty from conjunctions: s.d. of L1 estimate given the sheath value
    e = np.concatenate([np.arange(0, 12, 1.0), [12, 14, 17, 21, 26]])
    idx = np.digitize(world['cXsh'], e) - 1
    rel = []; cen = []
    for i in range(len(e) - 1):
        m = idx == i
        if m.sum() >= 30:
            xc = np.median(world['cXsh'][m]); rel.append(np.std(world['cXs'][m]) / xc); cen.append(xc)
    rel = np.array(rel); cen = np.array(cen)
    relf = lambda x: np.interp(x, cen, rel)                      # held constant beyond the last bin
    acc = Accum(EDGES)
    for b in range(nbatch_mc):
        rng = np.random.default_rng(seed + b)
        Xm = driver(M, rng, mu=muY, sig=sY)
        Xsm, Xtm = measure(Xm, rng, dt_fn=lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0),
                           eps_fn=relf)
        acc.add(Xsm, Xtm)
    c = acc.centers(); fr = acc.mean(30)
    ok = np.isfinite(fr)
    frm = np.maximum.accumulate(np.where(ok, fr, -np.inf))[ok]
    xc = np.interp(world['Xs'], c[ok], frm)
    accC = Accum(EDGES); accC.add(xc, world['Y'])
    return dict(muY=muY, sY=sY, rel_cen=cen.tolist(), rel=rel.tolist(), fr=fr.tolist(),
                calib=accC.mean(30).tolist(), calib_counts=accC.c.tolist())


def nscore_corr(a, b):
    za = stats.norm.ppf(ranks01(a)); zb = stats.norm.ppf(ranks01(b))
    return np.corrcoef(za, zb)[0, 1]


def copula_estimator(world, dt2_law, seed):
    rng = np.random.default_rng(seed)
    r12 = nscore_corr(world['cXs'], world['cXsh'])
    r13 = nscore_corr(world['Xs'], world['Y'])                   # long record
    r23 = nscore_corr(world['cXsh'], world['cY'])
    l1 = np.sqrt(r12 * r13 / r23); l2 = np.sqrt(r12 * r23 / r13); l3 = np.sqrt(r13 * r23 / r12)
    lsh = np.log(world['cXsh']); sd_sh = lsh.std(); sX = l2 * sd_sh
    muX = lsh.mean() + (sd_sh ** 2 - sX ** 2) / 2
    # serial dependence of the true driver at propagation lags, from the sheath reference inside
    # conjunction intervals (white reference noise divides out through l2^2)
    zsh = stats.norm.ppf(ranks01(world['cXsh'])).reshape(-1, 120)
    lags = np.arange(1, 111)
    acf = np.array([np.mean(zsh[:, :-L] * zsh[:, L:]) for L in lags])
    rZ = np.minimum(acf / l2 ** 2, 1.0)
    d2 = np.abs(np.rint(dt2_law(200000, rng))).astype(int)
    rz_full = np.concatenate([[1.0], rZ])
    lag_lam = np.mean(rz_full[np.clip(d2, 0, 110)])
    lres = min(l3 / lag_lam, 0.9999)
    Yq = np.sort(world['Y'])
    xg = np.linspace(0.5, 25, 50)
    zx = (np.log(xg) - muX) / sX
    v = stats.norm.cdf(lres * zx[:, None] + np.sqrt(1 - lres ** 2) * GH_X[None, :])
    ghat = (np.quantile(Yq, v.ravel()).reshape(v.shape) * GH_W[None, :]).sum(1)
    gqm = np.quantile(Yq, stats.norm.cdf(zx))                      # deterministic-response limit
    # direct (oracle) truth sigma for reference
    sX_true = np.log(world['cX']).std()
    return dict(r12=r12, r13=r13, r23=r23, l1=l1, l2=l2, l3=l3, lag_lam=lag_lam, lres=lres,
                muX=muX, sX=sX, sX_true=sX_true, xg=xg.tolist(), ghat=ghat.tolist(), gqm=gqm.tolist())


if __name__ == '__main__':
    # usage (from results/):  python ../code/exp5_worlds.py scan        -> observable curves of both worlds for several xs
    #                         python ../code/exp5_worlds.py full XS NB  -> full experiment with saturation scale XS
    mode = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    if mode == 'scan':
        WL = make_world('L', None, 3, 256, 100)
        cL = WL['obs_curve']; cen = Accum(EDGES).centers()
        for xs in [25, 30, 35, 40, 45, 50, 60]:
            WS = make_world('S', xs, 3, 256, 200)
            cS = WS['obs_curve']
            sel = [np.argmin(abs(cen - q)) for q in [5, 10, 15, 20, 25]]
            print(f"xs={xs}: obs L {np.round(cL[sel],2)}  obs S {np.round(cS[sel],2)}", flush=True)
    else:
        xs = float(sys.argv[2]); nb = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        res = {}
        for name in ['L', 'S']:
            W = make_world(name, xs, nb, 256, 100 if name == 'L' else 200)
            P = pipeline(W, 8, 256, 900)
            C = copula_estimator(W, lambda n, r: dt2_weibull(n, r, lam=LAM17, shift=17.0), 5)
            Ct = copula_estimator(W, WORLDS[name]['dt2'], 5)
            xg = np.array(C['xg']); C['gtrue'] = W['g'](xg).tolist()
            C['ghat_truelag'] = Ct['ghat']; C['lres_truelag'] = Ct['lres']; C['lag_truelag'] = Ct['lag_lam']
            res[name] = dict(obs=[None if not np.isfinite(v) else float(v) for v in W['obs_curve']],
                             centers=Accum(EDGES).centers().tolist(), pipeline=P, copula=C)
            cen = np.array(res[name]['centers']); cal = np.array(P['calib'], dtype=float)
            sel = [np.argmin(abs(cen - q)) for q in [5, 8, 10, 12, 14]]
            print(f"World {name}: rel-profile(pipeline) {np.round(P['rel'],2)}")
            print(f"   calibrated <Y|c> at c=5,8,10,12,14: {np.round(cal[sel],2)}")
            print(f"   copula: l1={C['l1']:.3f} l2={C['l2']:.3f} l3={C['l3']:.3f} lag={C['lag_lam']:.3f} lres={C['lres']:.3f} "
                  f"sX={C['sX']:.3f} (true {C['sX_true']:.3f}) muX={C['muX']:.3f}")
            gi = [np.argmin(abs(xg - q)) for q in [5, 10, 15, 20, 25]]
            print(f"   g_true {np.round(np.array(C['gtrue'])[gi],2)}  g_hat {np.round(np.array(C['ghat'])[gi],2)}  g_QM {np.round(np.array(C['gqm'])[gi],2)}", flush=True)
            print(f"   with true lag law: lag={C['lag_truelag']:.3f} lres={C['lres_truelag']:.3f} g_hat {np.round(np.array(C['ghat_truelag'])[gi],2)}", flush=True)
        res['xs'] = xs
        json.dump(res, open('exp5_worlds.json', 'w'))
