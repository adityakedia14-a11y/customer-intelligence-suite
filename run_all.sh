#!/usr/bin/env bash
# Reproduces every result: download data -> SQL features -> models -> Power BI tables.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p data/raw

if [ ! -f data/raw/online_retail_ii.zip ]; then
  echo "Downloading UCI Online Retail II..."
  curl -sSL -o data/raw/online_retail_ii.zip https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
fi
if [ ! -f data/raw/hillstrom.csv ]; then
  echo "Downloading Hillstrom email experiment..."
  curl -sSL -o data/raw/hillstrom.csv \
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
fi

cd src
../$PY convert_raw.py
echo "== SQL features ==";        ../$PY build_features.py
echo "== Segmentation ==";        ../$PY segment.py
echo "== Lifetime value ==";      ../$PY clv.py
echo "== Churn ==";               ../$PY churn.py
echo "== Uplift ==";              ../$PY uplift.py
echo "== Power BI export ==";     ../$PY export_powerbi.py
