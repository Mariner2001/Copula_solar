"""Figures of the Matters Arising (manuscript/matters_arising/figures/), from the JSON files in results/."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / 'results', ROOT / 'manuscript' / 'matters_arising' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
B, O, A, GR, INK, INK2, K = '#2a78d6', '#eb6834', '#1baf7a', '#9a9893', '#0b0b0b', '#52514e', '#3d3c38'
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.0, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'savefig.dpi': 400, 'lines.linewidth': 1.3, 'mathtext.fontset': 'custom',
    'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic'})
MM = 1 / 25.4
UNIT = r'(mV m$^{-1}$)'
J = lambda n: json.load(open(RES / n))
nan = lambda a: np.array([np.nan if v is None else v for v in a], float)
obs, s3, e1, cons, ma = J('obs_analysis.json'), J('exp3_serial.json'), J('exp1_budget.json'), J('obs_consistency.json'), J('ma_checks.json')
# published model curve <X|X*>, digitized from Fig. 2b of Sivadas et al. (2026)
PUB = np.array([[10, 6.80], [12, 7.87], [14, 8.90], [15, 9.41], [16, 10.06], [18, 10.78], [20, 11.45], [22, 12.19], [24, 12.66]])


def label(ax, s, dx=-0.17):
    ax.text(dx, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.jpg', bbox_inches='tight', dpi=400, pil_kwargs={'quality': 95})
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1: persistence and the averaged curve
fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 66 * MM), gridspec_kw=dict(wspace=0.3))
ax = axs[0]
ex = obs['serial']['extremogram']; lg = np.array(ex['lags'])
ax.fill_between(lg, ex['lo'], ex['hi'], color=B, alpha=0.2, lw=0)
ax.plot(lg, ex['Em_star'], color=B, lw=1.7, marker='o', ms=2.8, label=r'Observed driver $E_m^*$')
l3 = np.array(s3['gauss']['lags']); sel = l3 <= 240
ax.plot(l3[sel], np.array(s3['gauss']['ext'])[sel], color=K, lw=1.1, ls=(0, (4, 1.5)), marker='s', ms=2.3,
        label='Published model, true driver')
ax.plot(l3[sel], np.array(s3['gauss']['ext_measured'])[sel], color=GR, lw=1.1, ls=(0, (4, 1.5)), marker='s', ms=2.3,
        label='Published model, measured driver')
ax.plot(l3[sel], np.array(s3['persistent']['ext'])[sel], color=O, lw=1.1, ls=(0, (1.2, 1.2)), marker='^', ms=2.5,
        label='Driver with observed persistence')
ax.set_xscale('log'); ax.set_xlim(9, 260); ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel('Lag (min)'); ax.set_ylabel('Probability of remaining above the\n99.9th percentile (extremogram)')
ax.legend(loc='upper right', borderaxespad=0.1)
label(ax, 'a')
ax = axs[1]
P = obs['PCI']; c = np.array(P['centers']); cnt = np.array(P['counts']); ok = (cnt >= 100) & (c <= 22.5)
ax.plot([0, 22.5], [0, 22.5], color=GR, lw=0.9, ls=(0, (4, 2)), label='Identity')
ax.fill_between(c[ok], nan(P['curve_lo'])[ok], nan(P['curve_hi'])[ok], color=B, alpha=0.2, lw=0)
ax.plot(c[ok], nan(P['curve'])[ok], color=B, lw=1.7, label=r'Observed $\langle\mathrm{PCI}\,|\,E_m^*\rangle$')
for k, col, ls, lab in [('V0_published', K, (0, (4, 1.5)), r'Published model $\langle X|X^*\rangle$ (reconstructed)'),
                        ('V7_persistent', O, (0, (1.2, 1.2)), 'Same errors, observed persistence')]:
    cc = np.array(e1[k]['centers']); mm = nan(e1[k]['mean']); s_ = np.isfinite(mm) & (cc <= 22.5)
    ax.plot(cc[s_], mm[s_], color=col, lw=1.2, ls=ls, label=lab)
ax.plot(PUB[:, 0], PUB[:, 1], ls='none', marker='x', ms=3.2, mew=0.8, color=INK2, label='Published model (ref. 1, digitized)')
ax.set_xlim(0, 22.5); ax.set_ylim(0, 22.5)
ax.set_xlabel(r'Measured driver $E_m^*$, $X^*$ ' + UNIT); ax.set_ylabel('Polar cap index or true driver ' + UNIT)
ax.legend(loc='upper left', borderaxespad=0.1, handlelength=2.0)
label(ax, 'b')
save(fig, 'ma_fig1')

# ------------------------------------------------------------------ Fig. 2: margins and error scales
fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 66 * MM), gridspec_kw=dict(wspace=0.3))
ax = axs[0]
m = ma['margins']; g = np.array(m['stoploss_grid'])
for key, col, lab in [('Em_star', B, r'Observed driver $E_m^*$'), ('PCI', O, 'Observed polar cap index')]:
    lo, hi = np.array(m[key]['stoploss_grid_ci'][0]), np.array(m[key]['stoploss_grid_ci'][1])
    v = np.array(m[key]['stoploss_grid']); okk = v > 0
    ax.fill_between(g[okk], np.maximum(lo[okk], 1e-7), hi[okk], color=col, alpha=0.18, lw=0)
    ax.plot(g[okk], v[okk], color=col, lw=1.6, label=lab)
ax.plot(g, m['lognormal']['stoploss_grid'], color=K, lw=1.2, ls=(0, (4, 1.5)), label='Published model, true driver')
ax.plot(g, m['published_model_Xstar_stoploss_grid'], color=GR, lw=1.2, ls=(0, (1.2, 1.2)), label='Published model, measured driver')
ax.set_yscale('log'); ax.set_xlim(2, 25); ax.set_ylim(1e-6, 0.1)
ax.set_xlabel('Level $t$ ' + UNIT); ax.set_ylabel(r'Expected excess above $t$, $E[(Z-t)_+]$ ' + UNIT)
ax.legend(loc='lower left', borderaxespad=0.1)
label(ax, 'a')
ax = axs[1]
mg = obs['margins']; gx = np.array(mg['grid']); sel = gx >= 10
kr, krlo, krhi = np.array(mg['k_required']), np.array(mg['k_required_ci'][0]), np.array(mg['k_required_ci'][1])
cc_ = cons['copula']; xu_ = np.array(obs['PCI']['x_at_us']); su = np.array(obs['PCI']['us']) >= 0.99
kc = np.array([cc_[key]['k_cop_point_u'] for key in ('benchmark_V0', 'benchmark_V4', 'persistent_V0', 'persistent_V4')], float)
ax.fill_between(xu_[su], np.nanmin(kc, 0)[su], np.nanmax(kc, 0)[su], color=B, alpha=0.22, lw=0,
                label='Largest scale allowed by the tail association')
ax.fill_between(gx[sel], krlo[sel], krhi[sel], color=O, alpha=0.2, lw=0)
ax.plot(gx[sel], kr[sel], color=O, lw=1.7, marker='o', ms=2.8, label='Scale needed by the margins')
# scale needed to reproduce the observed bending with the observed persistence (interpolated from the scan)
sc = ma['persistent_deficit_scan']; ks = np.array([1.0, 1.25, 1.5, 2.0, 2.5])
split = obs['PCI']['split']
for xq, dx in ((15.0, -0.25), (20.0, 0.25)):
    d = np.array([sc[f'V0_k{k}'][str(xq)][0] for k in ks])
    target = [split[str(xq)]['deficit'], *split[str(xq)]['deficit_ci']]
    kn = [float(np.interp(tg, d, ks)) for tg in target]
    ax.errorbar([xq + dx], [kn[0]], yerr=[[kn[0] - kn[1]], [kn[2] - kn[0]]], color=A, marker='D', ms=3.2, lw=1.1,
                capsize=1.8, label='Scale needed by the bending, observed persistence' if xq == 15.0 else None)
ax.axhline(1.0, color=K, lw=0.9, ls=(0, (3, 1.5)))
ax.text(24.8, 0.97, 'Published errors', color=INK2, fontsize=5.6, ha='right', va='top')
ax.set_xlim(5, 25); ax.set_ylim(0, 2.4)
ax.set_xlabel('Driver level ' + UNIT); ax.set_ylabel('Scale of the magnitude errors\n(relative to published)')
ax.legend(loc='upper left', borderaxespad=0.1)
label(ax, 'b')
save(fig, 'ma_fig2')
print('figures written to', FIG)
