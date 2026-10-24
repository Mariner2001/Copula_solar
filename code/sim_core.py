"""Core Monte Carlo machinery for the copula study.

Reconstruction of the benchmark error model of Sivadas et al. (Nature 655, 1143, 2026), following its Methods,
Supplementary Methods 2b,d and the code that accompanies it (github.com/nithinsivadas/polar-cap-saturation):
  X  : stationary log-normal process, log-space mean -0.2518, log-space s.d. 0.85,
       autocorrelation with a ~100 min fast component and a slow component (ED Fig. 2b),
  X*(t) = max{X(t') + eps(t'), 0},  t' = t + dt1 + dt2  (the measured parcel and its error, as in the code),
  dt1: Student t, shape (nu) 1.3, scale 8 min, mean 0,
  dt2: Weibull, shape 1.3 (scale chosen per variant), shifted by the nominal 17 min delay,
  eps: zero-mean Gaussian, s.d. = r(X) X, with the relative error r(X) digitized from Fig. 2c of that paper.
All times in minutes, fields in mV/m.
"""
import numpy as np
from scipy import signal, stats, special

MU_LN, SIG_LN = -0.2518, 0.85
# two-exponential normal-score autocorrelation fitted to ED Fig. 2b (see verify notes)
ACF_W = (0.408, 0.592)
ACF_TAU = (106.0, 1578.0)          # minutes

PAD = 2048                          # padding (min) on each side of a segment
SEG = 16384                         # usable minutes per segment


def ar1(M, L, tau, rng):
    phi = np.exp(-1.0 / tau)
    e = rng.standard_normal((M, L))
    e[:, 1:] *= np.sqrt(1.0 - phi ** 2)
    return signal.lfilter([1.0], [1.0, -phi], e, axis=1)


def gauss_scores(M, L, rng, w=ACF_W, tau=ACF_TAU):
    z = np.zeros((M, L))
    for wi, ti in zip(w, tau):
        z += np.sqrt(wi) * ar1(M, L, ti, rng)
    return z


def normal_score_rank(a):
    """Rank-transform an array (any shape) to standard normal scores."""
    flat = a.ravel()
    r = np.empty(flat.size)
    r[np.argsort(flat, kind='stable')] = np.arange(1, flat.size + 1)
    return stats.norm.ppf(r / (flat.size + 1.0)).reshape(a.shape)


def driver(M, rng, mu=MU_LN, sig=SIG_LN, kind='gauss', mix_kappa=0.9, mix_tau=720.0,
           w=ACF_W, tau=ACF_TAU):
    """Return (M, SEG+2*PAD) array of the true driver X.
    kind='gauss'  : log-Gaussian process (Gaussian serial copula)
    kind='mixture': Gaussian scores multiplied by a slowly varying scale, then mapped back to the
                    same log-normal margin (tail-dependent serial copula, same margin)."""
    L = SEG + 2 * PAD
    z = gauss_scores(M, L, rng, w, tau)
    if kind == 'mixture':
        v = np.exp(mix_kappa * ar1(M, L, mix_tau, rng))
        z = normal_score_rank(z * np.sqrt(v))
    return np.exp(mu + sig * z)


# ---------------- error components ----------------
# relative magnitude error r(X) = s.d.(eps)/X, digitized at 0.5 mV/m steps from Fig. 2c of Sivadas et al. (2026),
# held constant below 0.5 and above 25 mV/m
_R_X = np.concatenate([[0.0], np.arange(0.5, 25.01, 0.5), [1000.0]])
_R_V = np.array([84.5, 84.5, 54.5, 44.2, 39.6, 36.8, 34.2, 32.5, 32.1, 32.6, 33.7, 35.3, 37.0, 38.1, 38.3, 38.5,
                 39.5, 42.0, 45.8, 49.8, 52.8, 54.5, 54.7, 53.6, 51.8, 50.2, 49.5, 49.0, 47.9, 46.3, 43.9, 39.7,
                 34.1, 31.5, 34.8, 38.1, 36.3, 31.6, 27.3, 25.4, 26.2, 26.4, 23.7, 22.0, 24.2, 27.0, 27.6, 26.7,
                 26.1, 27.2, 31.1, 31.1]) / 100.0


def rel_err_profile(x):
    """Relative magnitude uncertainty r(X) of the published error model (Fig. 2c of Sivadas et al. 2026)."""
    return np.interp(x, _R_X, _R_V)


def dt1_t(n, rng, nu=1.3, scale=8.0):
    return stats.t.rvs(nu, scale=scale, size=n, random_state=rng)


def dt1_normal(n, rng, sd=9.0):
    return rng.normal(0.0, sd, size=n)


def dt2_weibull(n, rng, k=1.3, lam=25.0, shift=17.0):
    return lam * rng.weibull(k, size=n) - shift


def measure(X, rng, dt_fn=None, eps_fn=rel_err_profile, eps_scale=1.0, scale_at='propagated'):
    """X*(t) = max{X(t') + eps(t'), 0} with t' = t + dt on the central window of each padded segment.
    dt_fn(n, rng) returns timing errors in minutes (None: no timing error); eps has s.d. eps_scale*eps_fn(x)*x
    evaluated at the measured parcel x = X(t') as in the published code (scale_at='propagated'), or at the
    true value X(t) (scale_at='reference'); eps_scale=0 switches the magnitude error off.
    Returns (X*, X) on the central window."""
    M, L = X.shape
    t = np.arange(PAD, PAD + SEG)
    Xt = X[:, PAD:PAD + SEG]
    n = M * SEG
    if dt_fn is None:
        Xp = Xt.copy()
    else:
        dt = np.rint(dt_fn(n, rng)).reshape(M, SEG)
        idx = np.clip(t[None, :] + dt, 0, L - 1).astype(np.int64)
        Xp = np.take_along_axis(X, idx, axis=1)
    if eps_scale > 0:
        base = Xp if scale_at == 'propagated' else Xt
        Xs = Xp + eps_scale * eps_fn(base) * base * rng.standard_normal(Xp.shape)
        np.maximum(Xs, 0.0, out=Xs)
    else:
        Xs = Xp
    return Xs, Xt


def binned_mean(x, y, edges):
    idx = np.digitize(x.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    s = np.bincount(idx[ok], weights=y.ravel()[ok], minlength=len(edges) - 1)
    s2 = np.bincount(idx[ok], weights=(y.ravel()[ok]) ** 2, minlength=len(edges) - 1)
    c = np.bincount(idx[ok], minlength=len(edges) - 1)
    return s, s2, c


class Accum:
    """Accumulate binned conditional means over batches."""
    def __init__(self, edges):
        self.edges = np.asarray(edges)
        self.s = np.zeros(len(edges) - 1)
        self.s2 = np.zeros(len(edges) - 1)
        self.c = np.zeros(len(edges) - 1)

    def add(self, x, y):
        s, s2, c = binned_mean(x, y, self.edges)
        self.s += s; self.s2 += s2; self.c += c

    def mean(self, min_count=50):
        m = np.full(self.s.shape, np.nan)
        ok = self.c >= min_count
        m[ok] = self.s[ok] / self.c[ok]
        return m

    def centers(self):
        return 0.5 * (self.edges[1:] + self.edges[:-1])


def interp_curve(xq, centers, curve):
    ok = np.isfinite(curve)
    return np.interp(xq, centers[ok], curve[ok])


def interp_se(S, C, centers, xq, min_count=30):
    """Conditional mean at xq, interpolated linearly between the centres of the valid bins that surround
    it, from per-batch bin sums S[b, j] and counts C[b, j]; the Monte Carlo standard error follows from
    the spread between batches (ratio estimator linearized, batches as independent clusters)."""
    S, C = np.asarray(S, float), np.asarray(C, float)
    s, c = S.sum(0), C.sum(0)
    ok = np.where(c >= min_count)[0]
    cv = centers[ok]
    i = int(np.searchsorted(cv, xq)) - 1
    if i < 0 or i + 1 >= cv.size:
        return float('nan'), float('nan')
    jl, jr = ok[i], ok[i + 1]
    a = (xq - centers[jl]) / (centers[jr] - centers[jl])
    ml, mr = s[jl] / c[jl], s[jr] / c[jr]
    e = (1 - a) * (S[:, jl] - ml * C[:, jl]) / c[jl] + a * (S[:, jr] - mr * C[:, jr]) / c[jr]
    nb = S.shape[0]
    return float((1 - a) * ml + a * mr), float(np.sqrt(nb / (nb - 1) * np.sum(e ** 2)))
