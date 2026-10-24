import numpy as np, json
from scipy import special
from sim_core import *
K = 1.3; G1 = special.gamma(1 + 1/K); LAM17 = 17/G1
edges = np.concatenate([np.arange(0, 12, 0.5), np.arange(12, 27, 1.0)])
classes = {
 'turbulent': dict(w=(0.6, 0.4), tau=(30.0, 360.0), r=0.50),
 'coherent':  dict(w=(0.6, 0.4), tau=(360.0, 1800.0), r=0.25),
}
timing = lambda n, r: dt1_normal(n, r, 9.0) + dt2_weibull(n, r, lam=LAM17, shift=17.0)
out = {}
for name, c in classes.items():
    acc = Accum(edges)
    for b in range(8):
        rng = np.random.default_rng(1200 + b)
        X = driver(256, rng, w=c['w'], tau=c['tau'])
        Xs, Xt = measure(X, rng, dt_fn=timing, eps_fn=lambda x, rr=c['r']: rr + 0 * x)
        acc.add(Xs, Xt)
    cen = acc.centers(); m = acc.mean(30)
    ratio = m / cen
    ok = np.isfinite(ratio) & (cen > 1)
    # knee: first X* where the deficit exceeds 20 %
    knee = cen[ok][np.argmax(ratio[ok] < 0.8)] if np.any(ratio[ok] < 0.8) else np.nan
    sel = [np.argmin(abs(cen - q)) for q in [2, 3, 5, 8, 10, 15]]
    print(f"{name:9s} ratio <X|X*>/X* at 2,3,5,8,10,15: {np.round(ratio[sel],3)}  knee(20% deficit) = {knee}")
    out[name] = dict(centers=cen.tolist(), mean=[None if not np.isfinite(v) else float(v) for v in m], knee=float(knee))
json.dump(out, open('exp6_structure.json', 'w'))
