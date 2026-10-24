import numpy as np, json
from scipy import special, stats
from sim_core import *
from copula_tools import *
k = 1.3; g1 = special.gamma(1 + 1/k); lam17 = 17/g1
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(80); gh_w = gh_w/gh_w.sum()
out = {}
for name, dtf in [('V0_published', lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0)),
                  ('V4_nodouble', lambda n, r: dt2_weibull(n, r, lam=lam17, shift=17.0))]:
    rng = np.random.default_rng(4242)
    XS = []; XT = []
    for b in range(2):
        X = driver(512, rng); Xs, Xt = measure(X, rng, dt_fn=dtf); XS.append(Xs.ravel()); XT.append(Xt.ravel())
    Xs = np.concatenate(XS); Xt = np.concatenate(XT)
    U = ranks01(Xs); V = ranks01(Xt); zU = stats.norm.ppf(U); zV = stats.norm.ppf(V)
    rho = np.corrcoef(zU, zV)[0, 1]
    us = 1 - np.logspace(-1, -4, 16)
    chi = lambda_u(U, V, us); chig = lambda_u_gauss(rho, us)
    # chi-bar (Coles et al. 1999)
    chibar = np.array([2*np.log(1-u)/np.log(np.mean((U > u) & (V > u))) - 1 for u in us])
    grid = np.linspace(0.5, 25, 50)
    srt = np.sort(Xs); Fs = np.searchsorted(srt, grid)/(Xs.size + 1.0)
    Xt_sorted = np.sort(Xt)
    EG = []
    for F in Fs:
        z = stats.norm.ppf(F); v = stats.norm.cdf(rho*z + np.sqrt(1-rho**2)*gh_x)
        EG.append(np.sum(gh_w*np.quantile(Xt_sorted, v)))
    out[name] = dict(rho=rho, us=us.tolist(), chi=chi.tolist(), chig=chig.tolist(), chibar=chibar.tolist(),
                     grid=grid.tolist(), EG=EG)
    print(name, 'rho', round(rho, 3), 'chi(0.999)', round(float(np.interp(0.999, us, chi)), 3), 'chibar tail', np.round(chibar[-4:], 3))
json.dump(out, open('figdata_extra.json', 'w'))
