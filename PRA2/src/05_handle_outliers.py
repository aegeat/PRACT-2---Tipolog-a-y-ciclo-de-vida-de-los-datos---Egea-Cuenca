# -*- coding: utf-8 -*-
"""
@author: Andres Tripiana
"""

from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "data" / "credit_clean_step2.csv"
OUT_PATH = BASE_DIR / "data" / "credit_clean_step3.csv"

BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

LOW_Q = 0.01
HIGH_Q = 0.99


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def cap_outliers(df: pd.DataFrame, cols, q_low=0.01, q_high=0.99) -> pd.DataFrame:
    """
    Cap values outside [q_low, q_high] quantiles.
    """
    out = df.copy()

    for col in cols:
        low = out[col].quantile(q_low)
        high = out[col].quantile(q_high)
        out[col] = out[col].clip(lower=low, upper=high)

    return out


def report_caps(df_before: pd.DataFrame, df_after: pd.DataFrame, cols) -> None:
    print("=== STEP 3.3: OUTLIER CAPPING ===\n")

    for col in cols:
        min_before, max_before = df_before[col].min(), df_before[col].max()
        min_after, max_after = df_after[col].min(), df_after[col].max()

        print(f"{col}:")
        print(f"  before -> min: {min_before:.2f}, max: {max_before:.2f}")
        print(f"  after  -> min: {min_after:.2f}, max: {max_after:.2f}")
    print()


def main() -> None:
    df = load_data(IN_PATH)

    cols_to_cap = BILL_COLS + PAY_COLS

    df_capped = cap_outliers(df, cols_to_cap, LOW_Q, HIGH_Q)
    report_caps(df, df_capped, cols_to_cap)

    df_capped.to_csv(OUT_PATH, index=False)
    print(f"Saved dataset with capped outliers to: {OUT_PATH}")


if __name__ == "__main__":
    main()
