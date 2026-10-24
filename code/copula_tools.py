import numpy as np
from scipy import stats

def ranks01(a):
    a = np.asarray(a).ravel()
    r = np.empty(a.size); r[np.argsort(a, kind='stable')] = np.arange(1, a.size + 1)
    return r / (a.size + 1.0)

def nscore(a):
    return stats.norm.ppf(ranks01(a))

def lambda_u(U, V, us):
    return np.array([np.mean((U > u) & (V > u)) / (1 - u) for u in us])

def lambda_u_gauss(rho, us):
    out = []
    for u in us:
        z = stats.norm.ppf(u)
        p = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([-z, -z])
        out.append(p / (1 - u))
    return np.array(out)

def eta_hill(U, V, ks):
    T = np.minimum(1 / (1 - U), 1 / (1 - V))
    Ts = np.sort(T)[::-1]
    return np.array([np.mean(np.log(Ts[:k] / Ts[k])) for k in ks])

def rank_regression(U, V, edges):
    idx = np.digitize(U, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    s = np.bincount(idx[ok], weights=V[ok], minlength=len(edges) - 1)
    c = np.bincount(idx[ok], minlength=len(edges) - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s / c, c

def m_gauss(rho, u):
    return stats.norm.cdf(rho * stats.norm.ppf(u) / np.sqrt(2 - rho ** 2))

def kendall_sub(x, y, n, rng):
    i = rng.choice(x.size, size=min(n, x.size), replace=False)
    return stats.kendalltau(x.ravel()[i], y.ravel()[i]).statistic
