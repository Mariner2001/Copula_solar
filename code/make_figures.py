"""All figures of the manuscript, from the JSON files in results/. Writes PDFs to manuscript/figures/."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, special

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / 'results', ROOT / 'manuscript' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_core import rel_err_profile, ACF_W, ACF_TAU
from copula_tools import m_gauss

B, O, A, GR, INK, INK2, K = '#2a78d6', '#eb6834', '#1baf7a', '#9a9893', '#0b0b0b', '#52514e', '#3d3c38'
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.0, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 300, 'lines.linewidth': 1.3,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic',
})
MM = 1 / 25.4
UNIT = r'(mV m$^{-1}$)'


def J(name):
    return json.load(open(RES / name))


def nan(a):
    return np.array([np.nan if v is None else v for v in a], float)


def label(ax, s, dx=-0.2):
    ax.text(dx, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')


def interp_at(centers, means, x):
    c = np.asarray(centers, float); m = nan(means); ok = np.isfinite(m)
    return float(np.interp(x, c[ok], m[ok]))


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches='tight'); plt.close(fig)


e1, ch, fx = J('exp1_budget.json'), J('exp2b_channels.json'), J('figdata_extra.json')
obs, cons = J('obs_analysis.json'), J('obs_consistency.json')

# ------------------------------------------------------------------ Fig. 2
fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 60 * MM), gridspec_kw=dict(wspace=0.62, width_ratios=[1, 1, 1.15]))
ax = axs[0]
c = np.array(e1['V0_published']['centers']); m = nan(e1['V0_published']['mean'])
g = np.array(ch['V0_published']['grid']); Q = np.array(ch['V0_published']['Qgrid'])
okm = np.isfinite(m) & (c <= 25.5)
mi = np.interp(g, c[okm], m[okm])
ax.fill_between(g, Q, g, color=O, alpha=0.16, lw=0)
ax.fill_between(g, mi, Q, color=B, alpha=0.16, lw=0)
ax.plot([0, 25], [0, 25], color=GR, lw=0.9, ls=(0, (4, 2)), label='Identity')
ax.plot(g, Q, color=O, lw=1.3, label=r'Quantile map $F_X^{-1}\!\circ F_{*}$')
ax.plot(c[okm], m[okm], color=B, lw=1.6, label=r'$\langle X\,|\,X^*\rangle$')
ax.plot(g, fx['V0_published']['EG'], color=A, lw=1.1, ls=(0, (2.5, 1.5)), label='Gaussian copula, same margins')
ax.legend(loc='upper left', borderaxespad=0.1, handlelength=1.8)
xl = 23.0; ql = float(np.interp(xl, g, Q)); cl_ = float(np.interp(xl, c[okm], m[okm]))
ax.text(xl, 0.5 * (xl + ql) + 0.3, 'Marginal', color=INK2, fontsize=5.6, ha='center', va='center', rotation=36)
ax.text(xl, 0.5 * (ql + cl_), 'Copula', color=INK2, fontsize=5.6, ha='center', va='center', rotation=24)
ax.set_xlim(0, 25.5); ax.set_ylim(0, 25.5)
ax.set_xlabel('Measured driver $X^*$ ' + UNIT); ax.set_ylabel('True driver ' + UNIT)
label(ax, 'a', -0.22)
ax = axs[1]
us = np.array(fx['V0_published']['us'])
ax.plot(1 - us, fx['V0_published']['chi'], color=B, lw=1.6, marker='o', ms=2.6, label='Benchmark copula')
ax.plot(1 - us, fx['V0_published']['chig'], color=A, lw=1.1, ls=(0, (2.5, 1.5)), label='Gaussian copula,\nsame normal-score correlation')
ax.set_xscale('log'); ax.invert_xaxis(); ax.set_ylim(0, 1)
ax.set_xlabel(r'Exceedance probability $1-u$'); ax.set_ylabel(r'Tail association $\chi(u)$')
ax.legend(loc='lower left', handlelength=2.0, borderaxespad=0.2)
label(ax, 'b', -0.22)
ax = axs[2]
order = [('V0_published', 'As specified'), ('V1_dt2mean17', 'Delay, stated mean'),
         ('V2_dt2sd25', 'Delay, stated s.d.'), ('V3_finitevar', 'Finite-variance timing'),
         ('V4_nodouble', 'Timing counted once'), ('V5_magonly', 'Magnitude only'), ('V6_timeonly', 'Timing only')]
vals = [1 - interp_at(e1[k]['centers'], e1[k]['mean'], 20.0) / 20.0 for k, _ in order]
y = np.arange(len(order))[::-1]
sp = obs['PCI']['split']['20.0']
ax.axvspan(sp['deficit_ci'][0], sp['deficit_ci'][1], color=O, alpha=0.14, lw=0)
ax.axvline(sp['deficit'], color=O, lw=1.1, ls=(0, (3, 1.5)))
ax.barh(y, vals, height=0.62, color=B, edgecolor='white', linewidth=0.8)
for yi, v in zip(y, vals):
    ax.text(v - 0.012, yi, f'{v:.2f}', va='center', ha='right', fontsize=6.0, color='white', fontweight='bold')
ax.text(sp['deficit'] + 0.01, -0.45, 'Observed', color=INK, fontsize=5.6, ha='left', va='bottom')
ax.set_yticks(y); ax.set_yticklabels([o[1] for o in order], fontsize=6.0)
ax.set_xlim(0, 0.62); ax.set_ylim(-0.6, len(order) + 0.2)
ax.set_xlabel(r'Deficit $1-\langle X|X^*\rangle/X^*$ at 20 mV m$^{-1}$')
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
label(ax, 'c', -0.62)
save(fig, 'fig2_channels.pdf')

# ------------------------------------------------------------------ Fig. 3
s3, p3, t4 = J('exp3_serial.json'), J('exp3b_persist_final.json'), J('exp4_tail.json')
fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM), gridspec_kw=dict(wspace=0.5))
ax = axs[0]
lags = np.array(s3['gauss']['lags'])
ax.plot(lags, s3['gauss']['ext'], color=B, lw=1.6, marker='o', ms=2.8, label='Benchmark driver')
ax.plot(lags, s3['persistent']['ext'], color=O, lw=1.6, marker='o', ms=2.8, label='Persistent driver')
ax.plot(lags, s3['gauss']['acfN'], color=B, lw=0.9, ls=(0, (2.5, 1.5)), label='Normal-score ACF (dashed)')
ax.plot(lags, s3['persistent']['acfN'], color=O, lw=0.9, ls=(0, (2.5, 1.5)))
ax.set_xscale('log'); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel('Lag (min)'); ax.set_ylabel('Extremogram or autocorrelation')
ax.legend(loc='upper right', handlelength=2.0, borderaxespad=0.0, fontsize=5.6)
label(ax, 'a')
ax = axs[1]
e60 = np.array([p['ext60'] for p in p3]); d25 = np.array([1 - p['m25'] / 25 for p in p3]); d20 = np.array([1 - p['m20'] / 20 for p in p3])
ex = obs['serial']['extremogram']; j60 = list(ex['lags']).index(60.0)
ax.axvspan(ex['lo'][j60], ex['hi'][j60], color=GR, alpha=0.18, lw=0)
ax.axvline(ex['Em_star'][j60], color=GR, lw=0.9, ls=(0, (3, 1.5)))
ax.text(ex['Em_star'][j60] - 0.01, 0.235, 'Observed\ndriver', fontsize=5.6, color=INK2, ha='right', va='top')
s25 = np.array([p['se25'] / 25 for p in p3]); s20 = np.array([p['se20'] / 20 for p in p3])
ax.errorbar(e60, d25, yerr=s25, color=B, lw=1.4, marker='o', ms=3.2, elinewidth=0.7, capsize=1.5, label=r'$X^*=25$ mV m$^{-1}$')
ax.errorbar(e60, d20, yerr=s20, color=O, lw=1.4, marker='s', ms=3.0, elinewidth=0.7, capsize=1.5, label=r'$X^*=20$ mV m$^{-1}$')
ax.set_xlabel('Extremogram at 60-min lag'); ax.set_ylabel('Timing-induced deficit')
ax.set_ylim(0, 0.25); ax.set_xlim(0.2, 0.72); ax.legend(loc='lower left', borderaxespad=0.1)
label(ax, 'b')
ax = axs[2]
cl = np.array(t4['lognormal']['centers']); rl = np.array(t4['lognormal']['ratio'], float)
cp = np.array(t4['pareto_tail']['centers']); rp = np.array(t4['pareto_tail']['ratio'], float)
ax.plot(cl, t4['lognormal_analytic'], color=B, lw=0.9, ls=(0, (2.5, 1.5)))
ax.plot(cl, rl, color=B, lw=1.6, marker='o', ms=2.4, label='Log-normal truth')
ax.plot(cp, rp, color=O, lw=1.6, marker='o', ms=2.4, label='Pareto-tailed truth')
ax.axhline(t4['kappa_th'], color=GR, lw=0.9, ls=(0, (1, 1.5)))
ax.text(2.3, t4['kappa_th'] - 0.012, r'Pareto limit $\kappa$', ha='left', va='top', fontsize=6.0, color=INK2)
ax.set_xscale('log'); ax.set_xlim(2, 320); ax.set_ylim(0.45, 1.0)
ax.set_xlabel('Measured driver $X^*$ ' + UNIT); ax.set_ylabel(r'$\langle X|X^*\rangle/X^*$')
ax.legend(loc='upper right', borderaxespad=0.1)
label(ax, 'c')
save(fig, 'fig3_persistence_tails.pdf')

# ------------------------------------------------------------------ Fig. 4
w5 = J('exp5_worlds.json'); XS = w5['xs']
fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM), gridspec_kw=dict(wspace=0.42))
ax = axs[0]
for wn, col, lab in [('L', B, 'Linear world'), ('S', O, 'Saturating world')]:
    c = np.array(w5[wn]['centers']); o = nan(w5[wn]['obs']); ok = np.isfinite(o) & (c <= 25.5)
    ax.plot(c[ok], o[ok], color=col, lw=1.6 if wn == 'L' else 1.3, ls='-' if wn == 'L' else (0, (3, 1.2)), label=lab)
ax.plot([0, 25], [0, 25], color=GR, lw=0.9, ls=(0, (4, 2)))
ax.set_xlim(0, 25.5); ax.set_ylim(0, 25.5)
ax.set_xlabel('Measured driver $X^*$ ' + UNIT); ax.set_ylabel(r'Observed $\langle Y|X^*\rangle$ ' + UNIT)
ax.legend(loc='upper left', borderaxespad=0.1)
label(ax, 'a')
ax = axs[1]
xx = np.linspace(0, 25, 100)
for wn, col, lab in [('L', B, 'Linear world'), ('S', O, 'Saturating world')]:
    c = np.array(w5[wn]['centers']); cal = nan(w5[wn]['pipeline']['calib']); cnt = np.array(w5[wn]['pipeline']['calib_counts'])
    ok = np.isfinite(cal) & (cnt >= 2000)
    ax.plot(c[ok], cal[ok], color=col, lw=1.6, label=lab)
gt = lambda x: x / (1 + x / XS)
ax.plot(xx, xx, color=B, lw=0.9, ls=(0, (1, 1.3)))
ax.plot(xx, gt(xx), color=O, lw=0.9, ls=(0, (1, 1.3)))
ax.text(16.5, 16.5 + 0.7, 'True, linear', fontsize=5.9, color=INK2, ha='center', va='bottom', rotation=42)
ax.text(16.5, gt(16.5) - 0.7, 'True, saturating', fontsize=5.9, color=INK2, ha='center', va='top', rotation=26)
ax.set_xlim(0, 20); ax.set_ylim(0, 21.5)
ax.set_xlabel(r'Calibrated driver $X^{c}$ ' + UNIT); ax.set_ylabel(r'$\langle Y|X^{c}\rangle$ ' + UNIT)
ax.legend(loc='upper left', borderaxespad=0.1)
label(ax, 'b')
ax = axs[2]
for wn, col, lab in [('L', B, 'Linear world'), ('S', O, 'Saturating world')]:
    C = w5[wn]['copula']; xg = np.array(C['xg'])
    ax.fill_between(xg, C['ghat'], C['gqm'], color=col, alpha=0.18, lw=0)
    ax.plot(xg, C['ghat'], color=col, lw=1.6, label=lab + ', copula')
    ax.plot(xg, C['gtrue'], color=col, lw=0.9, ls=(0, (1, 1.3)))
ax.text(25.0, 26.0, 'True', fontsize=5.9, color=INK2, ha='right', va='bottom')
ax.text(25.0, gt(25.0) - 1.9, 'True', fontsize=5.9, color=INK2, ha='right', va='top')
ax.set_xlim(0, 25.5); ax.set_ylim(0, 27)
ax.set_xlabel('True driver $X$ ' + UNIT); ax.set_ylabel(r'Response function $E[Y|X]$ ' + UNIT)
ax.legend(loc='upper left', borderaxespad=0.1, fontsize=5.8)
label(ax, 'c')
save(fig, 'fig4_worlds.pdf')

# ------------------------------------------------------------------ Fig. 5 (observations)
P = obs['PCI']
fig, axs = plt.subplots(2, 2, figsize=(183 * MM, 118 * MM), gridspec_kw=dict(wspace=0.32, hspace=0.48))
ax = axs[0, 0]
c = np.array(P['centers']); cnt = np.array(P['counts'])
ok = (cnt >= 100) & (c <= 22.5)
ax.plot([0, 22.5], [0, 22.5], color=GR, lw=0.9, ls=(0, (4, 2)), label='Identity')
ax.fill_between(c[ok], nan(P['qmap_lo'])[ok], nan(P['qmap_hi'])[ok], color=O, alpha=0.18, lw=0)
ax.plot(c[ok], nan(P['qmap'])[ok], color=O, lw=1.3, label=r'Quantile map $F_{\mathrm{PCI}}^{-1}\!\circ F_{*}$')
ax.fill_between(c[ok], nan(P['curve_lo'])[ok], nan(P['curve_hi'])[ok], color=B, alpha=0.2, lw=0)
ax.plot(c[ok], nan(P['curve'])[ok], color=B, lw=1.7, label=r'Observed $\langle\mathrm{PCI}\,|\,E_m^*\rangle$')
for k, col, ls, lab in [('V0_published', K, (0, (4, 1.5)), 'Error model, published'),
                        ('V7_persistent', INK2, (0, (1.2, 1.2)), 'Error model, observed persistence')]:
    cc = np.array(e1[k]['centers']); mm = nan(e1[k]['mean']); s_ = np.isfinite(mm) & (cc <= 22.5)
    ax.plot(cc[s_], mm[s_], color=col, lw=1.1, ls=ls, label=lab)
ax.set_xlim(0, 22.5); ax.set_ylim(0, 22.5)
ax.set_xlabel(r'Measured driver $E_m^*$ ' + UNIT); ax.set_ylabel('Polar cap index ' + UNIT)
ax.legend(loc='upper left', borderaxespad=0.1, handlelength=2.0)
label(ax, 'a', -0.16)
ax = axs[0, 1]
ex = obs['serial']['extremogram']
lg = np.array(ex['lags'])
ax.fill_between(lg, ex['lo'], ex['hi'], color=B, alpha=0.2, lw=0)
ax.plot(lg, ex['Em_star'], color=B, lw=1.7, marker='o', ms=2.8, label=r'Observed driver $E_m^*$')
ls3 = np.array(s3['gauss']['lags']); sel = ls3 <= 240
ax.plot(ls3[sel], np.array(s3['gauss']['ext'])[sel], color=K, lw=1.1, ls=(0, (4, 1.5)), marker='s', ms=2.4, label='Model driver, published')
ax.plot(ls3[sel], np.array(s3['persistent']['ext'])[sel], color=INK2, lw=1.1, ls=(0, (1.2, 1.2)), marker='^', ms=2.6, label='Model driver, observed persistence')
ax.set_xscale('log'); ax.set_ylim(0, 1); ax.set_xlim(9, 260)
ax.set_xlabel('Lag (min)'); ax.set_ylabel(r'Extremogram at the 0.999 quantile')
ax.legend(loc='lower left', borderaxespad=0.1)
label(ax, 'b', -0.16)
ax = axs[1, 0]
us = np.array(P['us']); xu = 1 - us
H, Mo, cc_ = obs['hemispheres'], obs['monitors'], cons['copula']
ax.fill_between(xu, P['chi_lo'], P['chi_hi'], color=B, alpha=0.2, lw=0)
ax.plot(xu, P['chi'], color=B, lw=1.7, marker='o', ms=2.6, label=r'Observed, $E_m^*$ and PCI')
ax.plot(xu, cc_['benchmark_V0']['1.0']['chi'], color=K, lw=1.1, ls=(0, (4, 1.5)), label='Model + index noise, published')
ax.plot(xu, cc_['persistent_V0']['1.0']['chi'], color=INK2, lw=1.1, ls=(0, (1.2, 1.2)), label='Model + index noise, persistent')
ax.plot(xu, H['chi_NS'], color=A, lw=1.1, marker='d', ms=2.4, label='Observed, PCN and PCS')
ax.plot(xu, Mo['wind_geotail']['chi'], color=O, lw=1.1, marker='v', ms=2.4, label='Observed, Wind and Geotail')
ax.set_xscale('log'); ax.invert_xaxis(); ax.set_ylim(0, 1)
ax.set_xlabel(r'Exceedance probability $1-u$'); ax.set_ylabel(r'Tail association $\chi(u)$')
ax.legend(loc='lower center', borderaxespad=0.1, fontsize=5.3, ncol=2, columnspacing=0.8, handlelength=1.8)
label(ax, 'c', -0.16)
ax = axs[1, 1]
mg = obs['margins']; gx = np.array(mg['grid']); sel = gx >= 10
kr, krlo, krhi = np.array(mg['k_required']), np.array(mg['k_required_ci'][0]), np.array(mg['k_required_ci'][1])
xu_ = np.array(P['x_at_us']); su = (np.array(P['us']) >= 0.99)
kc = np.array([cc_[key]['k_cop_point_u'] for key in ('benchmark_V0', 'benchmark_V4', 'persistent_V0', 'persistent_V4')], float)
ax.fill_between(xu_[su], np.nanmin(kc, 0)[su], np.nanmax(kc, 0)[su], color=B, alpha=0.22, lw=0, label='Largest scale allowed by the copula')
ax.plot(xu_[su], kc[0][su], color=B, lw=0.9, ls=(0, (4, 1.5)))
ax.plot(xu_[su], kc[3][su], color=B, lw=0.9, ls=(0, (1.2, 1.2)))
ax.fill_between(gx[sel], krlo[sel], krhi[sel], color=O, alpha=0.2, lw=0)
ax.plot(gx[sel], kr[sel], color=O, lw=1.7, marker='o', ms=2.8, label='Scale needed by a linear response')
ax.axhline(1.0, color=K, lw=0.9, ls=(0, (3, 1.5)))
ax.text(24.8, 0.97, 'Published errors', color=INK2, fontsize=5.6, ha='right', va='top')
ax.set_xlim(5, 25); ax.set_ylim(0, 2.0)
ax.set_xlabel(r'Driver level ' + UNIT); ax.set_ylabel('Scale of the magnitude errors\n(relative to published)')
ax.legend(loc='upper left', borderaxespad=0.1)
label(ax, 'd', -0.16)
save(fig, 'fig5_observations.pdf')

# ------------------------------------------------------------------ Extended Data Fig. 1 (inputs)
fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 55 * MM), gridspec_kw=dict(wspace=0.45))
ax = axs[0]
tt = np.linspace(-60, 120, 901)
ax.plot(tt, stats.t(1.3, scale=8).pdf(tt), color=B, lw=1.5, label=r'$dt_1$ ($t$, $\nu=1.3$, scale 8 min)')
k_ = 1.3; g1 = special.gamma(1 + 1 / k_)
for lam, col, lab, ls in [(25.0, O, r'$dt_2$ (Weibull, scale 25 min)', '-'),
                          (17 / g1, A, r'$dt_2$ (Weibull, mean 17 min)', (0, (3, 1.2)))]:
    ax.plot(tt, stats.weibull_min(k_, scale=lam).pdf(tt + 17), color=col, lw=1.3, ls=ls, label=lab)
ax.set_xlabel('Timing error (min)'); ax.set_ylabel('Probability density (min$^{-1}$)'); ax.set_ylim(0, 0.068)
ax.set_xlim(-60, 120); ax.legend(loc='upper right', fontsize=5.6, borderaxespad=0.1, handlelength=1.6)
label(ax, 'a')
ax = axs[1]
xg = np.linspace(0.1, 25, 300)
ax.plot(xg, 100 * rel_err_profile(xg), color=B, lw=1.5)
ax.set_xlabel('True driver $X$ ' + UNIT); ax.set_ylabel('Relative magnitude error (%)'); ax.set_ylim(0, 90); ax.set_xlim(0, 25)
label(ax, 'b')
ax = axs[2]
lag = np.logspace(0, np.log10(3000), 200)
r = sum(w * np.exp(-lag / t) for w, t in zip(ACF_W, ACF_TAU))
ax.plot(lag, r, color=B, lw=1.5)
ax.set_xscale('log'); ax.set_ylim(0, 1.02)
ax.set_xlabel('Lag (min)'); ax.set_ylabel('Normal-score autocorrelation')
label(ax, 'c')
save(fig, 'ed_fig1_inputs.pdf')

# ------------------------------------------------------------------ Extended Data Fig. 2 (structure)
s6 = J('exp6_structure.json')
fig, ax = plt.subplots(figsize=(89 * MM, 58 * MM))
for nm, col, lab in [('coherent', B, 'Coherent intervals (magnetic-cloud-like)'), ('turbulent', O, 'Turbulent intervals (sheath-like)')]:
    c = np.array(s6[nm]['centers']); m = nan(s6[nm]['mean']); ok = np.isfinite(m) & (c > 0.6) & (c <= 25.5)
    ax.plot(c[ok], m[ok] / c[ok], color=col, lw=1.6, label=lab)
ax.axhline(0.8, color=GR, lw=0.9, ls=(0, (3, 1.5)))
ax.text(24.8, 0.81, '20% deficit', fontsize=5.9, color=INK2, ha='right', va='bottom')
ax.set_ylim(0.4, 1.02); ax.set_xlim(0, 25.5)
ax.set_xlabel('Measured driver $X^*$ ' + UNIT); ax.set_ylabel(r'$\langle X|X^*\rangle/X^*$')
ax.legend(loc='lower left', borderaxespad=0.1)
save(fig, 'ed_fig2_structure.pdf')

# ------------------------------------------------------------------ Extended Data Fig. 3 (copula diagnostics)
c2 = J('exp2_copula.json')
fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM), gridspec_kw=dict(wspace=0.45))
ax = axs[0]
zU = np.array(c2['V0_published']['zU']); zV = np.array(c2['V0_published']['zV'])
ax.scatter(zU[:8000], zV[:8000], s=1.0, color=B, alpha=0.25, lw=0, rasterized=True)
ax.plot([-4.5, 4.5], [-4.5, 4.5], color=GR, lw=0.8, ls=(0, (4, 2)))
ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')
ax.set_xlabel('Normal score of $X^*$'); ax.set_ylabel('Normal score of $X$')
label(ax, 'a')
ax = axs[1]
uc = np.array(c2['V0_published']['uc']); mr = np.array(c2['V0_published']['mr'], float)
sel = uc > 0.9
ax.plot(1 - uc[sel], 1 - mr[sel], color=B, lw=1.6, marker='o', ms=2.6, label='Benchmark copula')
ax.plot(1 - uc[sel], 1 - m_gauss(c2['V0_published']['rhoN'], uc[sel]), color=A, lw=1.1, ls=(0, (2.5, 1.5)), label='Gaussian copula')
ax.plot([1e-4, 0.1], [1e-4, 0.1], color=GR, lw=0.8, ls=(0, (4, 2)), label='No regression')
ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
ax.set_xlabel(r'$1-u$ (measured rank)'); ax.set_ylabel(r'$1-E[T\,|\,U=u]$ (true rank)')
ax.legend(loc='upper right', borderaxespad=0.1)
label(ax, 'b')
ax = axs[2]
us_ = np.array(fx['V0_published']['us'])
ax.plot(1 - us_, fx['V0_published']['chibar'], color=B, lw=1.6, marker='o', ms=2.6, label=r'Benchmark copula, $\bar\chi(u)$')
ax.axhline(fx['V0_published']['rho'], color=A, lw=1.1, ls=(0, (2.5, 1.5)), label=r'Gaussian copula ($\bar\chi=\rho$)')
ax.axhline(1.0, color=GR, lw=0.8, ls=(0, (4, 2)), label='Asymptotic dependence')
ax.set_xscale('log'); ax.invert_xaxis(); ax.set_ylim(0.5, 1.05)
ax.set_xlabel(r'Exceedance probability $1-u$'); ax.set_ylabel(r'$\bar\chi(u)$')
ax.legend(loc='lower left', borderaxespad=0.1)
label(ax, 'c')
save(fig, 'ed_fig3_copula.pdf')

# ------------------------------------------------------------------ Extended Data Fig. 4 (observations: serial, monitors, noise, tails)
fig, axs = plt.subplots(2, 2, figsize=(183 * MM, 112 * MM), gridspec_kw=dict(wspace=0.3, hspace=0.45))
ax = axs[0, 0]
ac = obs['serial']['acf']; la = np.array(ac['lags'])
ax.plot(la, ac['Em_star'], color=B, lw=1.6, label=r'Observed $E_m^*$')
ax.plot(la, ac['PCI'], color=O, lw=1.6, label='Observed PCI')
ax.plot(la, ac['benchmark_fit'], color=K, lw=1.1, ls=(0, (4, 1.5)), label='Model driver')
ax.set_xscale('log'); ax.set_ylim(0, 1.02)
ax.set_xlabel('Lag (min)'); ax.set_ylabel('Normal-score autocorrelation')
ax.legend(loc='lower left', borderaxespad=0.1)
label(ax, 'a', -0.16)
ax = axs[0, 1]
for key, col, mk, lab in [('wind_geotail', O, 'v', 'Wind and Geotail'), ('wind_ace', B, 'o', 'Wind and ACE')]:
    d = Mo[key]
    ax.fill_between(xu, d['chi_lo'], d['chi_hi'], color=col, alpha=0.15, lw=0)
    ax.plot(xu, d['chi'], color=col, lw=1.5, marker=mk, ms=2.4, label=lab)
ax.plot(xu, Mo['bench_dt1_only']['chi'], color=K, lw=1.1, ls=(0, (4, 1.5)), label='Timing law $dt_1$, model driver')
ax.plot(xu, Mo['bench_dt1_only_persistent']['chi'], color=INK2, lw=1.1, ls=(0, (1.2, 1.2)), label='Timing law $dt_1$, persistent driver')
ax.set_xscale('log'); ax.invert_xaxis(); ax.set_ylim(0, 1)
ax.set_xlabel(r'Exceedance probability $1-u$'); ax.set_ylabel(r'Tail association $\chi(u)$')
ax.legend(loc='lower left', borderaxespad=0.1)
label(ax, 'b', -0.16)
ax = axs[1, 0]
for key, col, mk, lab in [('wind_geotail', O, 'v', 'Wind and Geotail'), ('wind_ace', B, 'o', 'Wind and ACE')]:
    d = Mo[key]; bn = np.array(d['bins']); cen = np.sqrt(bn[1:] * bn[:-1]); rs = nan(d['rel_sd']); ok = np.isfinite(rs)
    ax.plot(cen[ok], rs[ok] / np.sqrt(2), color=col, lw=1.5, marker=mk, ms=2.6, label=lab)
hd = np.array(H['rel_disagreement'])
ax.plot(np.sqrt(hd[:, 0] * hd[:, 1]), hd[:, 3] / np.sqrt(2), color=A, lw=1.5, marker='d', ms=2.6, label='PCN and PCS')
xg = np.linspace(0.5, 25, 300)
ax.plot(xg, rel_err_profile(xg), color=K, lw=1.1, ls=(0, (4, 1.5)), label='Published magnitude error')
ax.set_xscale('log'); ax.set_ylim(0, 0.9); ax.set_xlim(0.5, 25)
ax.set_xlabel('Level of the pair ' + UNIT); ax.set_ylabel('Relative error per instrument')
ax.legend(loc='upper right', borderaxespad=0.1)
label(ax, 'c', -0.16)
ax = axs[1, 1]
hl = obs['serial']['hill']; kk = np.array(hl['k'])
ax.fill_between(kk, hl['Em_star_lo'], hl['Em_star_hi'], color=B, alpha=0.15, lw=0)
ax.plot(kk, hl['Em_star'], color=B, lw=1.5, label=r'Observed $E_m^*$')
ax.fill_between(kk, hl['PCI_lo'], hl['PCI_hi'], color=O, alpha=0.15, lw=0)
ax.plot(kk, hl['PCI'], color=O, lw=1.5, label='Observed PCI')
ax.plot(kk, hl['bench_Xstar'], color=K, lw=1.1, ls=(0, (4, 1.5)), label='Model, measured driver')
ax.set_xscale('log'); ax.set_ylim(0, 14)
ax.set_xlabel('Number of upper order statistics'); ax.set_ylabel(r'Hill estimate of the tail index $\alpha$')
ax.legend(loc='upper right', borderaxespad=0.1)
label(ax, 'd', -0.16)
save(fig, 'ed_fig4_observations.pdf')

# ------------------------------------------------------------------ Extended Data Fig. 5 (Mach-number classes)
CL = obs['classes']; ce = np.array(CL['edges']); cc = 0.5 * (ce[1:] + ce[:-1])
fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 56 * MM), gridspec_kw=dict(wspace=0.32), sharey=True)
for ax, (name, ttl), lb in zip(axs, [('wind_1min', 'Wind, 1-min Mach number'), ('wind_2h', 'Wind, 2-h mean Mach number'),
                                      ('ace_1min', 'ACE, 1-min Mach number')], 'abc'):
    for cls, col, lab in [('low', O, r'$M_A<3.3$'), ('mid', A, r'$3.3\leq M_A<7.2$'), ('high', B, r'$M_A\geq7.2$')]:
        d = CL[name][cls]; cu = nan(d['curve']); ok = np.isfinite(cu)
        ax.fill_between(cc[ok], nan(d['lo'])[ok], nan(d['hi'])[ok], color=col, alpha=0.15, lw=0)
        ax.plot(cc[ok], cu[ok], color=col, lw=1.5, marker='o', ms=2.2, label=lab)
    ax.plot([0, 25], [0, 25], color=GR, lw=0.8, ls=(0, (4, 2)))
    ax.set_xlim(0, 25); ax.set_ylim(0, 16)
    ax.set_title(ttl, fontsize=6.5, color=INK2)
    ax.set_xlabel(r'Measured driver $E_m^*$ ' + UNIT)
    label(ax, lb, -0.12)
axs[0].set_ylabel(r'$\langle\mathrm{PCI}\,|\,E_m^*\rangle$ ' + UNIT)
axs[0].legend(loc='upper left', borderaxespad=0.1)
save(fig, 'ed_fig5_mach.pdf')

# ------------------------------------------------------------------ Extended Data Fig. 6 (westward electrojet)
S = obs['SML']
fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 56 * MM), gridspec_kw=dict(wspace=0.38))
ax = axs[0]
c = np.array(S['centers']); cnt = np.array(S['counts']); ok = (cnt >= 100) & (c <= 22.5)
ax.fill_between(c[ok], nan(S['qmap_lo'])[ok], nan(S['qmap_hi'])[ok], color=O, alpha=0.18, lw=0)
ax.plot(c[ok], nan(S['qmap'])[ok], color=O, lw=1.3, label=r'Quantile map $F_{-\mathrm{SML}}^{-1}\!\circ F_{*}$')
ax.fill_between(c[ok], nan(S['curve_lo'])[ok], nan(S['curve_hi'])[ok], color=B, alpha=0.2, lw=0)
ax.plot(c[ok], nan(S['curve'])[ok], color=B, lw=1.7, label=r'Observed $\langle-\mathrm{SML}\,|\,E_m^*\rangle$')
ax.plot(c[ok], nan(S['bench_V0']['curve'])[ok], color=K, lw=1.1, ls=(0, (4, 1.5)), label='Benchmark measurement copula')
ax.set_xlim(0, 22.5); ax.set_ylim(0, 2600)
ax.set_xlabel(r'Measured driver $E_m^*$ ' + UNIT); ax.set_ylabel('Westward electrojet $-$SML (nT)')
ax.legend(loc='upper left', borderaxespad=0.1)
label(ax, 'a', -0.2)
ax = axs[1]
us = np.array(S['us']); xu = 1 - us
ax.fill_between(xu, S['chi_lo'], S['chi_hi'], color=B, alpha=0.2, lw=0)
ax.plot(xu, S['chi'], color=B, lw=1.6, marker='o', ms=2.4, label=r'Observed, $E_m^*$ and $-$SML')
ax.plot(xu, S['bench_V0']['chi'], color=K, lw=1.1, ls=(0, (4, 1.5)), label='Benchmark measurement copula')
ax.plot(xu, S['chi_gauss'], color=A, lw=1.0, ls=(0, (2.5, 1.5)), label='Gaussian copula, same correlation')
ax.set_xscale('log'); ax.invert_xaxis(); ax.set_ylim(0, 1)
ax.set_xlabel(r'Exceedance probability $1-u$'); ax.set_ylabel(r'Tail association $\chi(u)$')
ax.legend(loc='upper right', borderaxespad=0.1)
label(ax, 'b', -0.2)
save(fig, 'ed_fig6_sml.pdf')

print('figures written to', FIG)
