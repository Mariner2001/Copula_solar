#!/usr/bin/env bash
# Monte Carlo part of the study: regenerates every simulation output in results/ (about 30 min on 2 cores).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/results"
PY="python3 -u"
$PY ../code/verify_timing_laws.py      > verify_timing_laws.txt
$PY ../code/exp1_budget.py 32          > exp1_budget.log       # error budgets (Fig. 2c, 5a; ED Table 1)
$PY ../code/exp2b_channels.py          > exp2b_channels.log    # marginal and copula channels (Fig. 2a)
$PY ../code/exp2_copula.py             > exp2_copula.log       # measurement-copula diagnostics (ED Fig. 3)
$PY ../code/figdata_extra.py           > figdata_extra.log     # chi, chi-bar, Gaussian counterfactual (Fig. 2a,b)
$PY ../code/exp3_serial.py             > exp3_serial.log       # extremograms of the model drivers (Fig. 3a, 5b)
$PY ../code/exp3b_persist.py 32        > exp3b_persist.log     # persistence scan (Fig. 3b)
$PY ../code/exp4_tail.py               > exp4_tail.log         # tail class (Fig. 3c)
$PY ../code/exp5_worlds.py scan        > exp5_scan.log         # saturation scale that matches the two worlds
$PY ../code/exp5_worlds.py full 45 6   > exp5_worlds.log       # two synthetic worlds (Fig. 4)
$PY ../code/exp6_structure.py          > exp6_structure.log    # coherent and turbulent intervals (ED Fig. 2)
echo "simulations done"
