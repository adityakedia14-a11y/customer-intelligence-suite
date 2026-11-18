"""
Converts the UCI Online Retail II Excel file (two yearly sheets) into one CSV.
Reading Excel is slow, so we do it once and use the CSV from then on.

Run: python src/convert_raw.py
"""
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    out = RAW / "online_retail_ii.csv"
    if out.exists():
        print(f"{out.name} already exists")
        return
    xlsx = RAW / "online_retail_II.xlsx"
    if not xlsx.exists():
        with zipfile.ZipFile(RAW / "online_retail_ii.zip") as z:
            z.extract("online_retail_II.xlsx", RAW)
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows from {len(sheets)} sheets -> {out.name}")


if __name__ == "__main__":
    main()
