"""Can regression to the mean with a linear response reproduce the observed copula?

Linear response with the index noise implied by the two hemispheric polar cap indices:
PCN = X nu_N, PCS = X nu_S, PCI = (PCN + PCS)/2, with nu log-normal (mean 1) and log-s.d. s_h chosen so
that the simulated PCN-PCS tail association at u = 0.999 equals the observed one (a tail calibration that
implies less noise than the bulk dependence of PCN and PCS, so it favours regression to the mean). The
benchmark measurement model is applied with its magnitude-error profile multiplied by k. The tail
association chi(X*, PCI_sim)(u) must reach the observed chi(E_m*, PCI)(u) (Proposition 6), which bounds k
from above. Drivers: the benchmark log-Gaussian process and a driver with the same margin whose extremes
persist as observed (extremogram about 0.6 at 60 min). Timing laws: as specified (V0) and counted once (V4).
Writes results/obs_consistency.json.
"""
import json, gc, sys, time
from pathlib import Path
import numpy as np
from scipy import special, optimize
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_core import driver, measure, dt1_t, dt2_weibull, SIG_LN, MU_LN

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
K13 = 1.3
LAM17 = 17 / special.gamma(1 + 1 / K13)
TIMING = {'V0': lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0),
          'V4': lambda n, r: dt2_weibull(n, r, lam=LAM17, shift=17.0)}
DRIVERS = {'benchmark': dict(kind='gauss'), 'persistent': dict(kind='mixture', mix_kappa=6.0, mix_tau=5760.0)}
KS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
NB, M = 4, 256


def chi_gauss(rho, u):
    z = special.ndtri(u)
    return multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([-z, -z]) / (1 - u)


def rank_fast(a):
    """Ranks scaled to (0,1); the simulated variables are continuous, so ties do not occur."""
    r = np.empty(a.size); r[np.argsort(a, kind='stable')] = np.arange(1, a.size + 1)
    return r / (a.size + 1.0)


def copula_constraint(obs, hemi):
    us = np.array(obs['PCI']['us'])
    i999 = int(np.argmin(abs(us - 0.999)))
    target = hemi['chi_NS'][i999]
    rho_h = optimize.brentq(lambda r: chi_gauss(r, 0.999) - target, 0.3, 0.999)
    s_h = SIG_LN * np.sqrt(1 / rho_h - 1)
    out = {'s_h': float(s_h), 'rho_h_implied': float(rho_h), 'rho_NS_observed': hemi['rhoN_NS'],
           'chi_NS_target': target, 'us': us.tolist(), 'ks': KS,
           'chi_obs': obs['PCI']['chi'], 'chi_obs_lo': obs['PCI']['chi_lo'], 'chi_obs_hi': obs['PCI']['chi_hi'],
           'rhoN_obs': obs['PCI']['rhoN']}
    for dn, dkw in DRIVERS.items():
        for tv, tfun in TIMING.items():
            res = {}
            for k in KS:
                Xs_all, Y_all, X_all = [], [], []
                for b in range(NB):
                    rng = np.random.default_rng(6000 + b)            # common random numbers across k
                    X = driver(M, rng, **dkw)
                    Xs, Xt = measure(X, rng, dt_fn=tfun, eps_scale=k)
                    nu = np.exp(s_h * rng.standard_normal((2,) + Xt.shape) - s_h ** 2 / 2)
                    Y = Xt * 0.5 * (nu[0] + nu[1])
                    Xs_all.append(Xs.ravel().astype(np.float32)); Y_all.append(Y.ravel().astype(np.float32))
                    X_all.append(Xt.ravel().astype(np.float32))
                    del X, nu
                Xs_all, Y_all, X_all = np.concatenate(Xs_all), np.concatenate(Y_all), np.concatenate(X_all)
                U, V = rank_fast(Xs_all), rank_fast(Y_all)
                T = special.ndtr((np.log(X_all.astype(np.float64)) - MU_LN) / SIG_LN)       # exact log-normal ranks
                chi = [float(np.mean(V[U > u] > u)) for u in us]
                chiA = [float(np.mean(T[U > u] > u)) for u in us]
                rho = float(np.corrcoef(special.ndtri(U), special.ndtri(V))[0, 1])
                curve = {}
                for xv in (10.0, 15.0, 20.0):
                    s_ = np.abs(Xs_all - xv) < 0.5
                    curve[str(xv)] = [float(X_all[s_].mean()), float(Y_all[s_].mean()), int(s_.sum())]
                res[str(k)] = {'chi': chi, 'chi_A': chiA, 'rhoN': rho, 'curve': curve}
                print(f'  {dn:10s} {tv} k={k:4.2f}: chi(0.999)={chi[i999]:.3f} chi_A(0.999)={chiA[i999]:.3f} rhoN={rho:.3f}',
                      flush=True)
                del Xs_all, Y_all, X_all, U, V, T; gc.collect()
            def kmax(cvals, thr):
                """Largest k whose predicted chi still reaches thr (chi decreases with k); NaN if beyond the scan."""
                ok = cvals >= thr
                if not ok.any():
                    return 0.0
                if ok.all():
                    return float('nan')
                j = int(np.where(~ok)[0][0])
                return float(np.interp(thr, [cvals[j], cvals[j - 1]], [KS[j], KS[j - 1]]))
            kp, kl = [], []
            for i in range(us.size):
                ci_ = np.array([res[str(k)]['chi'][i] for k in KS])
                kp.append(kmax(ci_, obs['PCI']['chi'][i])); kl.append(kmax(ci_, obs['PCI']['chi_lo'][i]))
            res['k_cop_point_u'] = kp; res['k_cop_lower_u'] = kl
            res['k_cop_point'] = kp[i999]; res['k_cop_lower'] = kl[i999]
            out[f'{dn}_{tv}'] = res
            print(f'  -> {dn} {tv}: k_cop (point) = {res["k_cop_point"]:.2f}, (lower 95%) = {res["k_cop_lower"]:.2f}', flush=True)
    return out


def main(parts=None):
    t0 = time.time()
    obs = json.load(open(RES / 'obs_analysis.json'))
    path = RES / 'obs_consistency.json'
    out = {}
    if parts is None or 'copula' in parts:
        out['copula'] = copula_constraint(obs, obs['hemispheres'])
        json.dump(out, open(path, 'w'))
        print(f'copula constraint done [{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
