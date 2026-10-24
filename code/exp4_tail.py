import numpy as np, json
from scipy import stats
MU, SIG = -0.2518, 0.85
s = 0.39                                    # log-s.d. of multiplicative error (relative s.d. ~0.40)
u0 = 0.99; q0 = np.exp(MU + SIG*stats.norm.ppf(u0)); f0 = stats.lognorm(SIG, scale=np.exp(MU)).pdf(q0)
xi = 0.25; beta = (1-u0)/f0
def q_lognorm(u): return np.exp(MU + SIG*stats.norm.ppf(u))
def q_spliced(u):
    out = q_lognorm(np.minimum(u, u0))
    m = u > u0
    out[m] = q0 + beta/xi*(((1-u0)/(1-u[m]))**xi - 1)
    return out
alpha = 1/xi
kappa_th = np.exp(-(alpha-1)*s**2)
grid = np.exp(np.linspace(np.log(2), np.log(400), 28))
res = {}
rng = np.random.default_rng(3)
for name, qf in [('lognormal', q_lognorm), ('pareto_tail', q_spliced)]:
    S = np.zeros(len(grid)-1); C = np.zeros(len(grid)-1)
    for b in range(40):
        u = rng.random(5_000_000)
        X = qf(u)
        W = np.exp(s*rng.standard_normal(u.size) - s**2/2)
        Xs = X*W
        idx = np.digitize(Xs, grid) - 1; ok = (idx >= 0) & (idx < len(grid)-1)
        S += np.bincount(idx[ok], weights=X[ok], minlength=len(grid)-1); C += np.bincount(idx[ok], minlength=len(grid)-1)
    cen = np.sqrt(grid[1:]*grid[:-1]); m = np.where(C > 200, S/np.maximum(C,1), np.nan)
    res[name] = dict(centers=cen.tolist(), ratio=(m/cen).tolist(), counts=C.tolist())
    print(name, [ (round(c,1), round(r,3), int(n)) for c, r, n in zip(cen, m/cen, C) if np.isfinite(r)][::3])
# analytic lognormal ratio: E[X|X*=x]/x = exp(MU + b(ln x - mu*) + v/2)/x
mus = MU - s**2/2; b = SIG**2/(SIG**2+s**2); v = SIG**2*s**2/(SIG**2+s**2)
res['lognormal_analytic'] = (np.exp(MU + b*(np.log(np.array(res['lognormal']['centers'])) - mus) + v/2)/np.array(res['lognormal']['centers'])).tolist()
res['kappa_th'] = kappa_th; res['b_lognormal'] = b
print('kappa (Pareto asymptote) =', round(kappa_th, 3), ' lognormal log-log slope b =', round(b, 3))
json.dump(res, open('exp4_tail.json', 'w'))
