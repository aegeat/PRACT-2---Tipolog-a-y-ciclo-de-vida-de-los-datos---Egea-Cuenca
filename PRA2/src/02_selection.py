# -*- coding: utf-8 -*-
"""
@author: Andres Tripiana
"""
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "UCI_Credit_Card.csv"
OUT_PATH = BASE_DIR / "data" / "credit_selected.csv"

ID_COL = "ID"
TARGET_COL = "default.payment.next.month"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Drop the identifier column if present."""
    if id_col in df.columns:
        return df.drop(columns=[id_col]).copy()
    return df.copy()


def basic_checks(df: pd.DataFrame, target_col: str) -> None:
    """Minimal checks to ensure downstream steps behave as expected."""
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    uniques = sorted(df[target_col].unique())
    print("=== TARGET UNIQUE VALUES ===")
    print(uniques, "\n")

    if set(uniques) - {0, 1}:
        print("WARNING: Target is not strictly binary {0,1}. Please verify.\n")


def main() -> None:
    df = load_data(RAW_PATH)

    print("=== RAW SHAPE ===")
    print(df.shape, "\n")

    df = drop_id(df, ID_COL)

    print("=== AFTER DROPPING ID ===")
    print(df.shape)
    print(f"ID column present? {'YES' if ID_COL in df.columns else 'NO'}\n")

    basic_checks(df, TARGET_COL)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved selected dataset to: {OUT_PATH}")


if __name__ == "__main__":
    main()

