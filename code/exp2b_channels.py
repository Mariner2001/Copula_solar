import numpy as np, json
from scipy import special, stats
from sim_core import *
from copula_tools import *
k = 1.3; g1 = special.gamma(1 + 1/k); lam17 = 17/g1
cases = {
 'V0_published': lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0),
 'V4_nodouble':  lambda n, r: dt2_weibull(n, r, lam=lam17, shift=17.0),
 'V5_magonly': None,
 'V6_timeonly': 'time',
}
e1 = json.load(open('exp1_budget.json'))
xq = np.array([5., 10., 15., 20., 25.])
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60); gh_w = gh_w / gh_w.sum()
out = {}
for name, dtf in cases.items():
    rng = np.random.default_rng(91)
    Xall = []; Sall = []
    for b in range(2):
        X = driver(512, rng)
        if dtf == 'time':
            Xs, Xt = measure(X, rng, dt_fn=lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0), eps_scale=0.0)
        else:
            Xs, Xt = measure(X, rng, dt_fn=dtf)
        Xall.append(Xt.ravel()); Sall.append(Xs.ravel())
    Xt = np.concatenate(Xall); Xs = np.concatenate(Sall)
    srt_s = np.sort(Xs); srt_x = np.sort(Xt); n = Xs.size
    Fs = np.searchsorted(srt_s, xq) / (n + 1.0)
    Q = np.quantile(Xt, Fs)                     # marginal map F_X^{-1}(F_*(x))
    zU = stats.norm.ppf(ranks01(Xs)); zV = stats.norm.ppf(ranks01(Xt)); rho = np.corrcoef(zU, zV)[0, 1]
    # Gaussian-copula counterfactual with same margins
    EG = []
    for F in Fs:
        z = stats.norm.ppf(F)
        v = stats.norm.cdf(rho * z + np.sqrt(1 - rho**2) * gh_x)
        EG.append(np.sum(gh_w * np.quantile(Xt, v)))
    EG = np.array(EG)
    c = np.array(e1[name]['centers']); m = np.array([np.nan if q is None else q for q in e1[name]['mean']])
    tot = np.array([interp_curve(x, c, m) for x in xq])     # linear interpolation between bin centres
    print(f"{name}: rho_N={rho:.3f}")
    for x, a, qq, g in zip(xq, tot, Q, EG):
        d = x - a
        print(f"   x*={x:4.0f}  <X|X*>={a:6.2f}  Q={qq:6.2f}  GaussCop={g:6.2f}  | deficit={d:5.2f}: marginal={x-qq:5.2f} ({100*(x-qq)/d:4.0f}%), copula={qq-a:5.2f} ({100*(qq-a)/d:4.0f}%)")
    out[name] = dict(xq=xq.tolist(), total=tot.tolist(), Q=Q.tolist(), gauss=EG.tolist(), rho=rho,
                     marginal_share=((xq - Q) / (xq - tot)).tolist())
    # full Q curve on grid for plotting
    grid = np.linspace(0.5, 25, 50)
    out[name]['grid'] = grid.tolist()
    out[name]['Qgrid'] = np.quantile(Xt, np.searchsorted(srt_s, grid) / (n + 1.0)).tolist()
json.dump(out, open('exp2b_channels.json', 'w'))
