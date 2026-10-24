#!/usr/bin/env bash
# Reproduces the whole study: data download, preparation, simulations, observational analysis, figures and
# manuscript. About 1 h on two cores; needs about 7 GB of memory and 4 GB of free disk.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p results
bash data/download_data.sh                                          # 2 archives (2.0 GB), SHA-256 verified
python3 -u code/obs_prepare.py      > results/obs_prepare.log       # 1-min tables in data/processed/
bash run_simulations.sh                                             # Monte Carlo experiments
python3 -u code/obs_analysis.py     > results/obs_analysis.log      # observational statistics
python3 -u code/obs_consistency.py  > results/obs_consistency.log   # copula bound on the error scale
python3 -u code/ma_checks.py        > results/ma_checks.log         # checks for the Matters Arising
python3 code/make_figures.py                                        # manuscript/figures/*.pdf
python3 code/make_ma_figures.py                                     # manuscript/matters_arising/figures/
python3 manuscript/source/build_manuscript.py                       # manuscript/main.tex, word counts
python3 manuscript/serra/build_serra.py                             # SERRA version (author-year)
if command -v latexmk > /dev/null; then
  (cd manuscript && latexmk -pdf -interaction=nonstopmode -quiet main.tex)
  (cd manuscript/matters_arising && latexmk -pdf -interaction=nonstopmode -quiet ma_main.tex)
  (cd manuscript/serra && latexmk -pdf -interaction=nonstopmode -quiet serra_main.tex)
fi
echo "done: manuscript/main.tex and manuscript/matters_arising/ma_main.tex (and PDFs if LaTeX is installed)"
