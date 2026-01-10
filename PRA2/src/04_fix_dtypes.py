# -*- coding: utf-8 -*-
"""
@author: Andres Tripiana
"""

from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "data" / "credit_clean_step1.csv"
OUT_PATH = BASE_DIR / "data" / "credit_clean_step2.csv"

CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set correct data types:
      - Categorical variables as 'category'
      - All others kept numeric
    """
    out = df.copy()

    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out


def report_dtypes(df: pd.DataFrame) -> None:
    print("=== STEP 3.2: DATA TYPES ===\n")

    print("Categorical variables:")
    for col in CATEGORICAL_COLS:
        print(f"  {col}")
    print()

    print("Numeric variables (sample):")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        if "default" not in col:
            print(f"  {col}")
    print()


def main() -> None:
    df = load_data(IN_PATH)
    df = fix_dtypes(df)

    report_dtypes(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved dataset with fixed dtypes to: {OUT_PATH}")


if __name__ == "__main__":
    main()