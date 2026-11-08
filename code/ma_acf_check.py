"""Autocorrelation checks quoted in the Matters Arising (Methods) and in the note on the authors' reply.

Compares the autocorrelation of the values and of the normal scores of the observed PCI (1995-2019) with those of
the benchmark driver X = exp(mu + sigma Z) and of the persistent driver. Writes results/ma_acf_check.json and
results/ma_acf_check.pdf.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sim_core as sc

ROOT = Path(__file__).resolve().parents[1]
L = np.arange(0, 841)


def acf_gappy(x):
    m = np.isfinite(x); xc = np.where(m, x - np.nanmean(x), 0.0)
    n = 1 << int(np.ceil(np.log2(2 * x.size)))
    F, G = np.fft.rfft(xc, n), np.fft.rfft(m.astype(float), n)
    num, cnt = np.fft.irfft(F * np.conj(F), n)[:L.size], np.fft.irfft(G * np.conj(G), n)[:L.size]
    return num / np.round(cnt) / (np.sum(xc ** 2) / m.sum())


def nscore(x):
    m = np.isfinite(x); z = np.full(x.size, np.nan)
    r = stats.rankdata(x[m] + 1e-9 * np.random.default_rng(0).random(m.sum()))
    z[m] = stats.norm.ppf(r / (m.sum() + 1)); return z


df = pd.read_parquet(ROOT / 'data' / 'processed' / 'polarcap_1min.parquet', columns=['time', 'PCI'])
pci = df.loc[df.time < '2020-01-01', 'PCI'].to_numpy(float)
obs, obs_ns = acf_gappy(pci), acf_gappy(nscore(pci))
rz = sc.ACF_W[0] * np.exp(-L / sc.ACF_TAU[0]) + sc.ACF_W[1] * np.exp(-L / sc.ACF_TAU[1])
rx = (np.exp(sc.SIG_LN ** 2 * rz) - 1) / (np.exp(sc.SIG_LN ** 2) - 1)
lags = [60, 240, 840]
per = []
for seed in range(4):
    X = sc.driver(128, np.random.default_rng(100 + seed), kind='mixture', mix_kappa=6.0, mix_tau=5760.0)[:, sc.PAD:-sc.PAD]
    Z = (np.log(X) - sc.MU_LN) / sc.SIG_LN
    f = lambda A: [float(np.mean((A - A.mean())[:, :A.shape[1] - k] * (A - A.mean())[:, k:]) / A.var()) for k in lags]
    per.append(f(X) + f(Z))
per = np.array(per)
out = {'lags_min': lags,
       'PCI_values': [float(obs[k]) for k in lags], 'PCI_normal_scores': [float(obs_ns[k]) for k in lags],
       'benchmark_values': [float(rx[k]) for k in lags], 'benchmark_normal_scores': [float(rz[k]) for k in lags],
       'max_abs_diff_benchmark_vs_PCI_values_up_to_14h': float(np.abs(rx - obs)[1:].max()),
       'persistent_values_mean_sd': [per[:, :3].mean(0).tolist(), per[:, :3].std(0).tolist()],
       'persistent_normal_scores_mean_sd': [per[:, 3:].mean(0).tolist(), per[:, 3:].std(0).tolist()]}
json.dump(out, open(ROOT / 'results' / 'ma_acf_check.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
h = L / 60
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
ax[0].plot(h, obs, 'k', lw=1.8, label='Data: PCI')
ax[0].plot(h, rx, color='#2a78d6', lw=1.4, ls=(0, (4, 1.5)), label=r'Error model: $X=\exp(\mu+\sigma Z)$')
ax[1].plot(h, obs_ns, 'k', lw=1.8, label='Data: PCI, normal scores')
ax[1].plot(h, rz, color='#d62728', lw=1.4, label=r'$\rho_Z=0.408e^{-\tau/106}+0.592e^{-\tau/1578}$')
for a, t in zip(ax, ('a  Autocorrelation of the values', 'b  Autocorrelation of the normal scores')):
    a.set_title(t, loc='left', fontsize=8.5, fontweight='bold'); a.set_xlim(0, 14); a.set_ylim(0, 1)
    a.set_xlabel(r'Lag $\tau$ (hours)'); a.legend(loc='upper right', fontsize=7.5, frameon=False)
    a.spines[['top', 'right']].set_visible(False)
ax[0].set_ylabel('Autocorrelation'); fig.tight_layout(); fig.savefig(ROOT / 'results' / 'ma_acf_check.pdf')
