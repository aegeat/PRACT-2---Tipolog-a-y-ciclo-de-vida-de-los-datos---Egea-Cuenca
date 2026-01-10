# -*- coding: utf-8 -*-
"""
@author: Andres Tripiana
"""
from pathlib import Path
import pandas as pd
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "data" / "credit_clean_step3.csv"

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def check_age(df: pd.DataFrame) -> None:
    invalid = df[df["AGE"] <= 0]
    print("AGE check:")
    print(f"  Invalid (AGE <= 0): {len(invalid)}")


def check_limit_bal(df: pd.DataFrame) -> None:
    invalid = df[df["LIMIT_BAL"] <= 0]
    print("LIMIT_BAL check:")
    print(f"  Invalid (<= 0): {len(invalid)}")


def check_pay_ranges(df: pd.DataFrame) -> None:
    print("PAY_* range checks:")
    for col in PAY_COLS:
        invalid = df[(df[col] < -2) | (df[col] > 9)]
        print(f"  {col}: out of range [-2, 9] -> {len(invalid)}")


def check_infinite_values(df: pd.DataFrame) -> None:
    inf_count = np.isinf(df.select_dtypes(include="number")).sum().sum()
    print("Infinite values check:")
    print(f"  Total infinite values: {inf_count}")


def check_nan_values(df: pd.DataFrame) -> None:
    nan_count = np.isnan(df.select_dtypes(include="number")).sum().sum()
    print("NaNs check:")
    print(f"  Total NaNs values: {nan_count}")


def main() -> None:
    df = load_data(IN_PATH)

    print("=== STEP 3.4: FINAL SANITY CHECKS ===\n")

    check_age(df)
    check_limit_bal(df)
    check_pay_ranges(df)
    check_infinite_values(df)
    check_nan_values(df)




if __name__ == "__main__":
    main()

