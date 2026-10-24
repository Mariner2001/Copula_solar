#!/usr/bin/env bash
# Downloads the two public data compilations used by the observational part of the study and verifies them.
#
#   uncertainty_Data.zip      776,637,520 bytes   github.com/nithinsivadas/uncertainty (Git LFS), cited as
#                                                 Zenodo 10.5281/zenodo.5559102 (Sivadas 2021)
#       Wind 1-min data shifted to the bow-shock nose (OMNI HRO spacecraft-specific files, NASA/GSFC SPDF),
#       PCN and PCS 1-min indices (pcindex.org), SuperMAG SML/SMU 1-min indices.
#   regression_bias_Data.zip  1,274,732,523 bytes github.com/nithinsivadas/regression-bias (Git LFS); data of
#                                                 Sivadas & Sibeck (2022), Zenodo 10.5281/zenodo.6604150
#       ACE and Geotail 1-min data shifted to the bow-shock nose (OMNI HRO spacecraft-specific files).
#
# The archives are only read by code/obs_prepare.py (in place, nothing inside them is executed).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)/raw"
mkdir -p "$DIR"

get() {   # name  url  sha256
  local name="$1" url="$2" sum="$3"
  if [ -f "$DIR/$name" ] && echo "$sum  $DIR/$name" | sha256sum -c --status; then
    echo "$name: present and verified"; return
  fi
  echo "downloading $name ..."
  curl -L --fail --retry 3 -o "$DIR/$name.part" "$url"
  mv "$DIR/$name.part" "$DIR/$name"
  echo "$sum  $DIR/$name" | sha256sum -c
}

get uncertainty_Data.zip \
  https://media.githubusercontent.com/media/nithinsivadas/uncertainty/39a5696da2fb68ab2e00e57b1c46c724f7b04711/Data.zip \
  d50c31ea0b4c858235a77452206933a21790036562885130fc7b4a704516d049
get regression_bias_Data.zip \
  https://media.githubusercontent.com/media/nithinsivadas/regression-bias/c18a85941041ac837cbccbf8c0f4313c3087a3c8/Data.zip \
  5efef53ef7a0fc19d8ab0700a970bfb7f27ee1e51f8ccf42a44c2b72d3355cbc
echo "data ready in $DIR"
