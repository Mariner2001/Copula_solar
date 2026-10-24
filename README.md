# Copulas separate apparent from physical saturation of geomagnetic storms

Code, derived results and LaTeX sources of the article and of a Matters Arising on Sivadas et al., *Nature*
(2026). Every number, figure and table is produced by the scripts in `code/` from public data and seeded
simulations.

## Quick start

```bash
pip install -r requirements.txt          # Python >= 3.10
bash run_all.sh                          # full reproduction, about 1 h on two cores
```

`run_all.sh` downloads the data (`data/download_data.sh`, 2.0 GB, checked by SHA-256), builds the
one-minute tables, runs the simulations and the observational analysis, draws the figures and writes
`manuscript/main.tex` (and `main.pdf` if `latexmk` is installed). The JSON files in `results/` are
included, so figures and manuscript can be rebuilt in a minute without data or simulations:

```bash
python3 code/make_figures.py
python3 manuscript/source/build_manuscript.py
cd manuscript && latexmk -pdf main.tex
```

## Layout

```
README.md  LICENSE  CITATION.cff  requirements.txt  .gitignore
run_all.sh                 whole pipeline
run_simulations.sh         Monte Carlo part only
code/
  sim_core.py              benchmark error model (driver, timing and magnitude errors), binning helpers
  copula_tools.py          ranks, normal scores, tail association, Hill estimator, rank regression
  verify_timing_laws.py    moments of the published timing laws
  exp1_budget.py           error budgets                        -> Fig. 2c, Fig. 5a, Extended Data Table 1
  exp2b_channels.py        marginal and copula channels         -> Fig. 2a
  exp2_copula.py           measurement-copula diagnostics       -> Extended Data Fig. 3
  figdata_extra.py         chi(u), chi-bar(u), Gaussian-copula counterfactual -> Fig. 2a,b
  exp3_serial.py           extremograms of the model drivers    -> Fig. 3a, Fig. 5b
  exp3b_persist.py         persistence scan                     -> Fig. 3b
  exp4_tail.py             log-normal versus Pareto-tailed truth -> Fig. 3c
  exp5_worlds.py           two synthetic worlds, calibration versus factor copula -> Fig. 4
  exp6_structure.py        coherent and turbulent intervals     -> Extended Data Fig. 2
  obs_prepare.py           one-minute tables from the data archives -> data/processed/
  obs_analysis.py          observational statistics, block bootstrap -> Fig. 5, Extended Data Figs 4-6, Table 2
  obs_consistency.py       copula bound on the magnitude-error scale -> Fig. 5c,d
  make_figures.py          all figures                          -> manuscript/figures/
  ma_checks.py             checks behind the Matters Arising    -> results/ma_checks.json
  make_ma_figures.py       figures of the Matters Arising       -> manuscript/matters_arising/figures/
data/
  README.md                sources, checksums, processed tables
  download_data.sh         download and verification
results/                   JSON outputs and logs of every script
manuscript/
  main.tex                 manuscript (generated; upload with figures/ to Overleaf)
  main.pdf                 compiled manuscript
  figures/                 figure PDFs
  source/main_src.tex      manuscript source with placeholders for references and tables
  source/build_manuscript.py  numbers references in Nature order, fills the tables, counts words
  matters_arising/         comment on Sivadas et al. (Nature, 2026): ma_main.tex, ma_main.pdf, figures/
                           (PDF and JPEG for submission)
  serra/                   version for Stochastic Environmental Research and Risk Assessment:
                           build_serra.py -> serra_main.tex, serra_main.pdf, figures/
```

The benchmark error model in `sim_core.py` reconstructs the model of Sivadas et al., *Nature* **655**,
1143–1147 (2026), from its Methods and the code that accompanies it
(github.com/nithinsivadas/polar-cap-saturation): a log-normal driver, Student-*t* and Weibull timing errors,
and a heteroskedastic magnitude error whose relative size is digitized from Fig. 2c of that paper.

Memory: `obs_analysis.py` needs about 4 GB. Figures use the Liberation Sans font if available.

## Guía rápida (español)

1. Instale las dependencias: `pip install -r requirements.txt`.
2. Reproducción completa: `bash run_all.sh` (descarga y verifica los datos, ejecuta simulaciones y análisis,
   genera figuras y `manuscript/main.tex`).
3. Solo figuras y manuscrito (sin datos ni simulaciones): `python3 code/make_figures.py` y
   `python3 manuscript/source/build_manuscript.py`.
4. Para Overleaf basta con subir `manuscript/main.tex` y la carpeta `manuscript/figures/`.
5. Matters Arising: `manuscript/matters_arising/`.
6. Versión para SERRA: `manuscript/serra/` (se regenera con `python3 manuscript/serra/build_serra.py`).

## Licence

Code under the MIT licence (`LICENSE`). The data are not redistributed; see `data/README.md` for their
sources and acknowledgement rules.
