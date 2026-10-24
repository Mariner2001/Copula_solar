import numpy as np, json
from scipy import special, stats
from sim_core import *
from copula_tools import *
k = 1.3; g1 = special.gamma(1 + 1/k); lam17 = 17/g1
cases = {
 'V0_published': lambda n, r: dt1_t(n, r) + dt2_weibull(n, r, lam=25.0, shift=17.0),
 'V4_nodouble':  lambda n, r: dt2_weibull(n, r, lam=lam17, shift=17.0),
}
us = np.array([0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999])
ks = [500, 1000, 2000, 5000, 10000]
uedges = np.concatenate([np.linspace(0, 0.9, 19), np.linspace(0.9, 0.99, 10)[1:], np.linspace(0.99, 0.9999, 12)[1:], [1.0]])
out = {}
for name, dtf in cases.items():
    rng = np.random.default_rng(77)
    X = driver(512, rng)
    Xs, Xt = measure(X, rng, dt_fn=dtf)
    U = ranks01(Xs); V = ranks01(Xt)
    zU = stats.norm.ppf(U); zV = stats.norm.ppf(V)
    rhoN = np.corrcoef(zU, zV)[0, 1]
    rhoS = np.corrcoef(U, V)[0, 1]
    tau = kendall_sub(Xs, Xt, 300000, rng)
    lu = lambda_u(U, V, us); lg = lambda_u_gauss(rhoN, us)
    eta = eta_hill(U, V, ks)
    mr, mc = rank_regression(U, V, uedges)
    uc = 0.5 * (uedges[1:] + uedges[:-1])
    # log-log slope of <X|X*> from exp1 not here; compute Gaussian-copula implied tau
    tau_g = 2/np.pi*np.arcsin(rhoN); rhoS_g = 6/np.pi*np.arcsin(rhoN/2)
    print(f"{name}: rho_N={rhoN:.4f} rho_S={rhoS:.4f} (Gauss {rhoS_g:.4f}) tau={tau:.4f} (Gauss {tau_g:.4f})")
    print("   u      lambda_U(emp)  lambda_U(Gauss)")
    for u, a, b in zip(us, lu, lg): print(f"   {u:.4f}   {a:.4f}        {b:.4f}")
    print("   eta Hill (k):", dict(zip(ks, np.round(eta, 3))), " Gauss eta=", round((1 + rhoN)/2, 3))
    print("   rank regression at top bins:", [(round(a,4), round(b,4), round(float(m_gauss(rhoN, a)),4)) for a, b in zip(uc[-6:], mr[-6:])])
    out[name] = dict(rhoN=rhoN, rhoS=rhoS, tau=tau, us=us.tolist(), lu=lu.tolist(), lg=lg.tolist(),
                     ks=ks, eta=eta.tolist(), uc=uc.tolist(), mr=mr.tolist(), mc=mc.tolist())
    # subsample of normal scores for plotting
    i = rng.choice(zU.size, 20000, replace=False)
    out[name]['zU'] = zU[i].tolist(); out[name]['zV'] = zV[i].tolist()
json.dump(out, open('exp2_copula.json', 'w'))
