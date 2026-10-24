# Data

The observational part of the study uses public one-minute records compiled by N. Sivadas for earlier
studies. They are not redistributed here; `download_data.sh` fetches them into `data/raw/` and checks
their SHA-256 checksums.

| File in `data/raw/` | Size | Source | Contents used |
|---|---|---|---|
| `uncertainty_Data.zip` | 777 MB | Git LFS file of [github.com/nithinsivadas/uncertainty](https://github.com/nithinsivadas/uncertainty), commit `39a5696`; the repository is cited as Zenodo [10.5281/zenodo.5559102](https://doi.org/10.5281/zenodo.5559102) | Wind 1-min plasma and field shifted to the bow-shock nose (OMNI HRO spacecraft-specific files, NASA/GSFC SPDF), 1995-2021; PCN and PCS 1-min (pcindex.org), 1995-2019; SuperMAG SML 1-min |
| `regression_bias_Data.zip` | 1.27 GB | Git LFS file of [github.com/nithinsivadas/regression-bias](https://github.com/nithinsivadas/regression-bias), commit `c18a859`; data and code of Sivadas & Sibeck (2022), Zenodo [10.5281/zenodo.6604150](https://doi.org/10.5281/zenodo.6604150) | ACE (1998-2021) and Geotail (1995-2006) 1-min data shifted to the bow-shock nose |

SHA-256:

```
d50c31ea0b4c858235a77452206933a21790036562885130fc7b4a704516d049  uncertainty_Data.zip
5efef53ef7a0fc19d8ab0700a970bfb7f27ee1e51f8ccf42a44c2b72d3355cbc  regression_bias_Data.zip
```

`code/obs_prepare.py` reads the archives in place and writes two tables to `data/processed/`
(about 760 MB together):

* `polarcap_1min.parquet`: one-minute grid in polar-cap time, 1995-2021. The driver quantities
  (merging field E_m*, Alfvén Mach number, speed, density, field) are the Wind values at the bow-shock nose
  17 min earlier, as in Sivadas et al. (Nature 655, 1143, 2026); PCN, PCS, PCI = (max(PCN,0)+max(PCS,0))/2
  and SML at the same minute.
* `nose_multi_sc_1min.parquet`: one-minute grid in nose time with the merging field from Wind, ACE and
  Geotail and the spacecraft positions, for the two-monitor comparison.

The original sources are OMNIWeb (https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/sc_specific/),
the PC index archive (https://pcindex.org/archive) and SuperMAG (https://supermag.jhuapl.edu/indices/).
Users of SuperMAG data should follow its acknowledgement rules
(https://supermag.jhuapl.edu/info/?page=acknowledgement).

The magnetosheath conjunctions of Sivadas et al. (2026) (Zenodo
[10.5281/zenodo.17546718](https://doi.org/10.5281/zenodo.17546718)) are not used; they would allow the
factor-copula triad (Proposition 3) to be applied to observations.
