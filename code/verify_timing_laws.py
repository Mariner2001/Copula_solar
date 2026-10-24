import numpy as np
from scipy import stats, special, optimize
print("=== Weibull dt2 consistency (shape k=1.3) ===")
k=1.3
g1=special.gamma(1+1/k); g2=special.gamma(1+2/k)
cv=np.sqrt(g2/g1**2-1)
print(f"CV(k=1.3) = {cv:.4f}")
lam_mean17=17/g1; sd_mean17=lam_mean17*np.sqrt(g2-g1**2)
print(f"mean=17 -> scale={lam_mean17:.2f}, SD={sd_mean17:.2f} min")
lam_sd25=25/np.sqrt(g2-g1**2); print(f"SD=25 -> scale={lam_sd25:.2f}, mean={lam_sd25*g1:.2f} min")
for lam in [lam_mean17,25.0,lam_sd25]:
    d=stats.weibull_min(k,scale=lam); mode=lam*((k-1)/k)**(1/k)
    print(f" scale={lam:6.2f}: mean={d.mean():6.2f} sd={d.std():6.2f} mode={mode:5.2f} peak pdf={d.pdf(mode):.4f}; dt2=T2-17 mean={d.mean()-17:+.2f}")
# shape needed for mean 17 and sd 25
f=lambda kk: special.gamma(1+2/kk)/special.gamma(1+1/kk)**2-1-(25/17)**2
kk=optimize.brentq(f,0.3,1.3); print(f"shape giving mean17/sd25: k={kk:.3f}")
print("=== Student t, nu=1.3, scale 8 min ===")
t=stats.t(1.3,scale=8)
print("variance:", t.var())
for thr in [8,16,30,60,120,360]:
    print(f" P(|dt1|>{thr:4d} min) = {2*t.sf(thr):.4f}")
print(" peak pdf:", t.pdf(0))
print(" IQR (min):", t.ppf(0.75)-t.ppf(0.25), " normal(sd=8) IQR:", 2*0.6745*8)
print("=== ED Fig 6: r=0.61, n=9 (and n=8 w/o Boyle) ===")
for n in [9,8]:
    r=0.61; tt=r*np.sqrt(n-2)/np.sqrt(1-r**2); p=2*stats.t.sf(tt,n-2)
    z=np.arctanh(r); se=1/np.sqrt(n-3); lo,hi=np.tanh(z-1.96*se),np.tanh(z+1.96*se)
    print(f" n={n}: t={tt:.3f}, p={p:.3f}, 95% CI for r = [{lo:.2f},{hi:.2f}]")
# with actual points from ED Fig 6a
yrs=np.array([1981,1983,1990,1997,2001,2002,2002,2002,2005]); lim=np.array([3,0.5,4,8,3,10,3,5,10])
r,p=stats.pearsonr(yrs,lim); print(f" from table values: r={r:.3f}, p={p:.3f}")
m=yrs!=1997; r2,p2=stats.pearsonr(yrs[m],lim[m]); print(f" without Boyle: r={r2:.3f}, p={p2:.3f}")
print("=== lognormal X used by authors: mu=-0.2518, s=0.85 ===")
mu,s=-0.2518,0.85
for x in [5,10,15,20,25]:
    print(f" P(X>{x}) = {stats.norm.sf((np.log(x)-mu)/s):.2e}")
