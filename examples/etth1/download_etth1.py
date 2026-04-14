"""
Download and prepare ETTh1 dataset for UPN experiments.

The ETTh1 dataset contains 17,420 hourly electricity transformer
temperature measurements. We select oil temperature and two
high-load features (HUFL, HULL) as described in Section 5.4.

Source: https://github.com/zhouhaoyi/Informer2020

Usage:
    python examples/etth1/download_etth1.py
"""

import os
import urllib.request
import pandas as pd

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
RAW_PATH = "data/etth1/ETTh1_raw.csv"
OUT_PATH = "data/etth1/etth1_simple.csv"


def main():
    os.makedirs("data/etth1", exist_ok=True)

    # Download raw data
    if not os.path.exists(RAW_PATH):
        print(f"Downloading ETTh1 from {URL} ...")
        urllib.request.urlretrieve(URL, RAW_PATH)
        print(f"  Saved to {RAW_PATH}")
    else:
        print(f"Raw data already exists at {RAW_PATH}")

    # Load and extract 3 features used in the paper
    df = pd.read_csv(RAW_PATH)
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Paper uses: oil temperature (OT) + 2 high-load features (HUFL, HULL)
    selected = df[["OT", "HUFL", "HULL"]].copy()
    selected.columns = ["temperature", "feature_1", "feature_2"]

    selected.to_csv(OUT_PATH, index=False)
    print(f"\nPrepared dataset: {selected.shape}")
    print(f"  Saved to {OUT_PATH}")
    print(f"  Columns: {selected.columns.tolist()}")
    print(selected.head(3))


if __name__ == "__main__":
    main()
