# -*- coding: utf-8 -*-
"""
@author: Andres Tripiana
"""

from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "data" / "credit_selected.csv"
OUT_PATH = BASE_DIR / "data" / "credit_clean_step1.csv"

# Domain rules (from dataset conventions)
INVALID_EDUCATION = {0}
INVALID_MARRIAGE = {0}


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def remove_invalid_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with invalid/unknown category codes:
      - EDUCATION in {0,5,6}
      - MARRIAGE in {0}
    """
    mask = (~df["EDUCATION"].isin(INVALID_EDUCATION)) & (~df["MARRIAGE"].isin(INVALID_MARRIAGE))
    return df.loc[mask].copy()


def report_removals(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    removed = len(df_before) - len(df_after)
    pct = removed / max(len(df_before), 1) * 100

    print("=== REMOVE INVALID CATEGORY CODES ===")
    print(f"Rows before: {len(df_before):,}")
    print(f"Rows after : {len(df_after):,}")
    print(f"Removed    : {removed:,} ({pct:.2f}%)\n")




def main() -> None:
    df = load_data(IN_PATH)

    # Basic guardrails
    for col in ["EDUCATION", "MARRIAGE"]:
        if col not in df.columns:
            raise ValueError(f"Expected column not found: {col}")

    df_clean = remove_invalid_codes(df)
    report_removals(df, df_clean)

    df_clean.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {OUT_PATH}")


if __name__ == "__main__":
    main()
