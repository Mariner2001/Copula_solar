"""Build the 1-min observational tables used in the empirical part of the study.

Inputs (not redistributed; see data/README.md and data/download_data.sh):
  data/raw/uncertainty_Data.zip      Git-LFS archive of github.com/nithinsivadas/uncertainty
        Data/omni_components/wind_min_bYYYY.txt   Wind 1-min, shifted to the bow-shock nose (OMNI HRO
                                                  spacecraft-specific files, SPDF), 1995-2021
        Data/pc_index/PCI/pcnpcsYYYY.txt          PCN and PCS 1-min (pcindex.org), 1995-2019
        Data/20210927-20-45-supermag.txt          SuperMAG SML and SMU 1-min
  data/raw/regression_bias_Data.zip  Git-LFS archive of github.com/nithinsivadas/regression-bias
        Data/omni_components/ace_min_bYYYY.txt     ACE 1-min at the nose, 1998-2021
        Data/omni_components/geotail_min_bYYYY.txt Geotail 1-min at the nose, 1995-2006

Outputs (data/processed/):
  polarcap_1min.parquet   grid in polar-cap time T (UTC); driver quantities are the Wind values at nose
                          time T - 17 min, as in Sivadas et al. (2026); PCN, PCS, PCI, SML, SMU at T.
  nose_multi_sc_1min.parquet  grid in nose time; merging field from Wind, ACE and Geotail with their
                          GSE positions, for the two-monitor comparison.
The archives are read in place (never extracted, nothing inside them is executed).
"""
import argparse, gc, io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DELAY_NOSE_TO_PC = 17          # minutes, constant nose-to-polar-cap delay used by Sivadas et al.
T0, T1 = np.datetime64('1995-01-01T00:00'), np.datetime64('2022-01-01T00:00')

# column indices of the OMNI spacecraft-specific 1-min format (format.txt, 37 words)
COLS = dict(year=0, doy=1, hour=2, minute=3, pct_interp=5, timeshift=7, B=11, BxGSE=12, ByGSM=15, BzGSM=16,
            rms_timeshift=17, V=22, Vx=23, n=26, T=27, Xsc=28, Ysc=29, Zsc=30, rms_target=34)
FILL = dict(timeshift=999999, B=9999.99, BxGSE=9999.99, ByGSM=9999.99, BzGSM=9999.99, rms_timeshift=999999,
            V=99999.9, Vx=99999.9, n=999.99, T=9999999., Xsc=9999.99, Ysc=9999.99, Zsc=9999.99,
            rms_target=9999.99)


def minute_index(t):
    """Minutes since T0 (int64) for datetime64[m] values."""
    return (t.astype('datetime64[m]') - T0).astype(np.int64)


NGRID = int((T1 - T0) / np.timedelta64(1, 'm'))


def read_sc_file(zf, name):
    with zf.open(name) as fh:
        df = pd.read_csv(io.TextIOWrapper(fh, 'ascii'), sep=r'\s+', header=None, usecols=list(COLS.values()),
                         engine='c', dtype=np.float64)
    df.columns = list(COLS.keys())
    for k, f in FILL.items():
        df.loc[df[k] >= f * 0.999, k] = np.nan
    year = int(df.year.iloc[0])
    base = np.datetime64(f'{year}-01-01T00:00')
    offs = ((df.doy.values - 1) * 1440 + df.hour.values * 60 + df.minute.values).astype('int64')
    t = base + offs.astype('timedelta64[m]')
    df['imin'] = minute_index(t)
    return df


def merging_field(V, By, Bz):
    """Kan-Lee merging electric field V B_T sin^2(theta/2) in mV/m (V in km/s, B in nT, GSM)."""
    BT = np.hypot(By, Bz)
    theta = np.arctan2(By, Bz)
    return 1e-3 * V * BT * np.sin(theta / 2.0) ** 2


def spacecraft_series(zf, prefix, years):
    names = set(zf.namelist())
    cols = {k: np.full(NGRID, np.nan, np.float32) for k in
            ['Em', 'V', 'n', 'B', 'By', 'Bz', 'T', 'Xsc', 'Ysc', 'Zsc', 'timeshift', 'rms_timeshift', 'pct_interp']}
    for y in years:
        nm = f'Data/omni_components/{prefix}_min_b{y}.txt'
        if nm not in names:
            continue
        df = read_sc_file(zf, nm)
        i = df.imin.values
        ok = (i >= 0) & (i < NGRID)
        i = i[ok]; df = df[ok]
        cols['Em'][i] = merging_field(df.V.values, df.ByGSM.values, df.BzGSM.values)
        for k, c in [('V', 'V'), ('n', 'n'), ('B', 'B'), ('By', 'ByGSM'), ('Bz', 'BzGSM'), ('T', 'T'), ('Xsc', 'Xsc'),
                     ('Ysc', 'Ysc'), ('Zsc', 'Zsc'), ('timeshift', 'timeshift'), ('rms_timeshift', 'rms_timeshift'),
                     ('pct_interp', 'pct_interp')]:
            cols[k][i] = df[c].values
        print(f'  {prefix} {y}: {np.isfinite(cols["Em"][i]).sum():,} valid minutes', flush=True)
    return cols


def pc_series(zf, years):
    pcn = np.full(NGRID, np.nan, np.float32); pcs = np.full(NGRID, np.nan, np.float32)
    names = set(zf.namelist())
    for y in years:
        nm = f'Data/pc_index/PCI/pcnpcs{y}.txt'
        if nm not in names:
            continue
        with zf.open(nm) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, 'ascii'), sep=r'\s+', comment='#', header=None,
                             names=['date', 'hm', 'PCN', 'PCS'], dtype={'date': str, 'hm': str})
        t = pd.to_datetime(df.date + ' ' + df.hm, format='%Y-%m-%d %H:%M').values.astype('datetime64[m]')
        i = minute_index(t)
        ok = (i >= 0) & (i < NGRID)
        a = df.PCN.values.astype(float); b = df.PCS.values.astype(float)
        a[a >= 999] = np.nan; b[b >= 999] = np.nan
        pcn[i[ok]] = a[ok]; pcs[i[ok]] = b[ok]
        print(f'  PC {y}: {np.isfinite(a).sum():,} PCN, {np.isfinite(b).sum():,} PCS', flush=True)
    return pcn, pcs


def supermag_series(zf):
    nm = 'Data/20210927-20-45-supermag.txt'
    with zf.open(nm) as fh:
        txt = io.TextIOWrapper(fh, 'utf-8', errors='replace')
        skip = 0
        for line in txt:
            skip += 1
            if line.startswith('<year>'):
                break
    with zf.open(nm) as fh:
        df = pd.read_csv(io.TextIOWrapper(fh, 'utf-8', errors='replace'), sep=r'\s+', skiprows=skip, header=None,
                         names=['y', 'mo', 'd', 'h', 'mi', 's', 'SML', 'SMU'], engine='c')
    t = pd.to_datetime(dict(year=df.y, month=df.mo, day=df.d, hour=df.h, minute=df.mi)).values.astype('datetime64[m]')
    i = minute_index(t)
    ok = (i >= 0) & (i < NGRID)
    sml = np.full(NGRID, np.nan, np.float32); smu = np.full(NGRID, np.nan, np.float32)
    a = df.SML.values.astype(float); b = df.SMU.values.astype(float)
    a[np.abs(a) >= 999999] = np.nan; b[np.abs(b) >= 999999] = np.nan
    sml[i[ok]] = a[ok]; smu[i[ok]] = b[ok]
    print(f'  SuperMAG: {np.isfinite(sml).sum():,} SML minutes', flush=True)
    return sml, smu


def shift(a, k):
    """Value at index i of the output equals a[i - k] (k minutes later on the output clock)."""
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=str(ROOT / 'data' / 'raw'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'processed'))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zu = zipfile.ZipFile(raw / 'uncertainty_Data.zip')
    zr = zipfile.ZipFile(raw / 'regression_bias_Data.zip')
    years = range(1995, 2022)
    print('Wind:', flush=True)
    W = spacecraft_series(zu, 'wind', years)
    print('PC indices:', flush=True)
    pcn, pcs = pc_series(zu, years)
    print('SuperMAG:', flush=True)
    sml, smu = supermag_series(zu)
    pci = np.where(np.isfinite(pcn) & np.isfinite(pcs), 0.5 * (np.maximum(pcn, 0) + np.maximum(pcs, 0)), np.nan)
    tgrid = T0 + np.arange(NGRID).astype('timedelta64[m]')
    # polar-cap clock: driver at nose time T-17 min
    MA = W['V'] / (21.81 * W['B'] / np.sqrt(W['n']))
    Pdyn = 1.6726e-6 * W['n'] * W['V'] ** 2
    pc = pd.DataFrame({'time': tgrid,
                       'Em_star': shift(W['Em'], DELAY_NOSE_TO_PC),
                       'V': shift(W['V'], DELAY_NOSE_TO_PC), 'n': shift(W['n'], DELAY_NOSE_TO_PC),
                       'B': shift(W['B'], DELAY_NOSE_TO_PC), 'By': shift(W['By'], DELAY_NOSE_TO_PC),
                       'Bz': shift(W['Bz'], DELAY_NOSE_TO_PC), 'MA': shift(MA.astype(np.float32), DELAY_NOSE_TO_PC),
                       'Pdyn': shift(Pdyn.astype(np.float32), DELAY_NOSE_TO_PC),
                       'Xsc': shift(W['Xsc'], DELAY_NOSE_TO_PC), 'Ysc': shift(W['Ysc'], DELAY_NOSE_TO_PC),
                       'Zsc': shift(W['Zsc'], DELAY_NOSE_TO_PC),
                       'timeshift': shift(W['timeshift'], DELAY_NOSE_TO_PC),
                       'rms_timeshift': shift(W['rms_timeshift'], DELAY_NOSE_TO_PC),
                       'PCN': pcn, 'PCS': pcs, 'PCI': pci.astype(np.float32), 'SML': sml, 'SMU': smu})
    pc.to_parquet(out / 'polarcap_1min.parquet', index=False)
    print('written', out / 'polarcap_1min.parquet', pc.shape, flush=True)
    del pc, pcn, pcs, pci, sml, smu, MA, Pdyn
    for k in list(W):
        if k not in ('Em', 'Xsc', 'Ysc', 'Zsc', 'V'):
            del W[k]
    gc.collect()
    print('ACE and Geotail:', flush=True)
    A = spacecraft_series(zr, 'ace', years)
    G = spacecraft_series(zr, 'geotail', years)
    MA_ace = (A['V'] / (21.81 * A['B'] / np.sqrt(A['n']))).astype(np.float32)
    ms = pd.DataFrame({'time': tgrid, 'Em_wind': W['Em'], 'Em_ace': A['Em'], 'Em_geotail': G['Em'],
                       'X_wind': W['Xsc'], 'Y_wind': W['Ysc'], 'Z_wind': W['Zsc'],
                       'X_ace': A['Xsc'], 'Y_ace': A['Ysc'], 'Z_ace': A['Zsc'],
                       'X_geotail': G['Xsc'], 'Y_geotail': G['Ysc'], 'Z_geotail': G['Zsc'],
                       'V_wind': W['V'], 'V_geotail': G['V'], 'MA_ace': MA_ace})
    keep = np.isfinite(ms.Em_wind) | np.isfinite(ms.Em_ace) | np.isfinite(ms.Em_geotail)
    ms[keep].to_parquet(out / 'nose_multi_sc_1min.parquet', index=False)
    print('written', out / 'nose_multi_sc_1min.parquet', int(keep.sum()), flush=True)


if __name__ == '__main__':
    main()
