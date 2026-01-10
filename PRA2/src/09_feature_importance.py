# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 08:43:47 2026

@author: Andres Tripiana
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_clean_step3.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PLOT_PATH = REPORTS_DIR / "feature_importance.png"

TARGET_COL = "default.payment.next.month"
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

N_SPLITS = 5
RANDOM_STATE = 42

# Best params from your grid search
BEST_PARAMS = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.01,
    "max_depth": 6,
    "n_estimators": 500,
    "subsample": 0.6,
}


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def encode_categories_as_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Encode categorical columns as integer codes (stable and simple).
    This avoids one-hot and keeps feature importance readable per original variable.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes.astype(int)
    return out


def main() -> None:
    # --- Load
    df = load_data(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    df = encode_categories_as_int(df, CAT_COLS)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).values

    # --- Class imbalance weight
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    # --- Model
    from xgboost import XGBClassifier  # assumes xgboost installed

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **BEST_PARAMS,
    )

    # --- OOF AP sanity check (k=5)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    ap_oof = average_precision_score(y, y_prob_oof)
    print(f"OOF Average Precision (k=5): {ap_oof:.4f}")

    # Fit on full data for importance
    model.fit(X, y)

    # --- Permutation importance (on original columns)
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print("\n=== TOP 15 FEATURES (Permutation Importance, original variables) ===")
    print(importances.head(15).to_string(index=False, formatters={
        "importance_mean": "{:.6f}".format,
        "importance_std": "{:.6f}".format
    }))

    # Plot top 15
    top = importances.head(15).iloc[::-1]

    plt.figure()
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance (mean decrease in Average Precision)")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances (Permutation Importance)")
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
