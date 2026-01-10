# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 10:11:35 2026

@author: Andres Tripiana
"""

from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency

# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_with_clusters.csv"

TARGET_COL = "default.payment.next.month"
CLUSTER_COL = "cluster"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    df = load_data(DATA_PATH)

    # Contingency table
    contingency = pd.crosstab(df[CLUSTER_COL], df[TARGET_COL])

    print("=== CONTINGENCY TABLE (Cluster x Default) ===")
    print(contingency)
    print()

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    print("=== CHI-SQUARE TEST OF INDEPENDENCE ===")
    print(f"Chi-square statistic : {chi2:.4f}")
    print(f"Degrees of freedom  : {dof}")
    print(f"P-value             : {p_value:.6f}")
    print()

    print("=== EXPECTED FREQUENCIES (H0) ===")
    print(pd.DataFrame(expected,
                       index=contingency.index,
                       columns=contingency.columns).round(2))
    print()

    # Effect size: Cramér's V
    n = contingency.values.sum()
    k = min(contingency.shape)
    cramer_v = (chi2 / (n * (k - 1))) ** 0.5

    print("=== EFFECT SIZE ===")
    print(f"Cramér's V : {cramer_v:.4f}")


if __name__ == "__main__":
    main()
