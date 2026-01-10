# -*- coding: utf-8 -*-
"""
@author: Andres Tripiana
"""

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "UCI_Credit_Card.csv"

TARGET_COL = "default.payment.next.month"


def load_data(path: Path) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def overview(df: pd.DataFrame, target_col: str) -> None:
    """Print a compact overview useful for the project report."""
    print("=== SHAPE ===")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}\n")

    print("=== COLUMNS ===")
    print(df.columns.tolist(), "\n")

    print("=== DATA TYPES ===")
    print(df.dtypes, "\n")

    print("=== MISSING VALUES (NaN) ===")
    na_counts = df.isna().sum()
    if (na_counts > 0).any():
        print(na_counts[na_counts > 0], "\n")
    else:
        print("No NaN values found.\n")

    print("=== TARGET DISTRIBUTION ===")
    counts = df[target_col].value_counts().sort_index()
    ratios = (counts / counts.sum()).round(4)
    print(pd.DataFrame({"count": counts, "ratio": ratios}), "\n")

    print("=== CATEGORICAL CODES CHECK ===")
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        print(f"{col}: {sorted(df[col].unique())}")
    print()

    print("=== PAY STATUS CHECK ===")
    pay_cols = [c for c in df.columns if c.startswith("PAY_")]
    for col in pay_cols:
        vals = sorted(df[col].unique())
        print(f"{col}: min={min(vals)}, max={max(vals)}")
    print()
    
    print("=== BILL AMOUNT CHECK ===")
    pay_cols = [c for c in df.columns if c.startswith("BILL_")]
    for col in pay_cols:
        vals = sorted(df[col].unique())
        print(f"{col}: min={min(vals)}, max={max(vals)}")
    print()


def main() -> None:
    df = load_data(DATA_PATH)
    overview(df, TARGET_COL)


if __name__ == "__main__":
    main()

