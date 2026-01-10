# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 09:42:09 2026

@author: Andres Tripiana
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score

# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_with_clusters.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PLOT_BAR_PATH = REPORTS_DIR / "default_rate_by_cluster.png"
PLOT_BOX_PATH = REPORTS_DIR / "predicted_default_by_cluster.png"

TARGET_COL = "default.payment.next.month"
CLUSTER_COL = "cluster"
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

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
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes.astype(int)
    return out


def plot_default_rate_by_cluster(df: pd.DataFrame) -> None:
    rates = df.groupby(CLUSTER_COL)[TARGET_COL].mean().sort_index()
    counts = df.groupby(CLUSTER_COL)[TARGET_COL].size().sort_index()

    x = np.arange(len(rates))
    y = rates.values

    plt.figure()
    plt.bar(x, y)
    plt.xticks(x, [f"Cluster {i}\n(n={counts[i]})" for i in rates.index])
    plt.ylabel("Default rate")
    plt.title("Default rate by cluster (not used for clustering)")
    plt.ylim(0, max(y) * 1.25)
    plt.savefig(PLOT_BAR_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def plot_predicted_default_by_cluster(df: pd.DataFrame) -> None:
    # Train XGBoost on full dataset (simple) and predict probabilities
    from xgboost import XGBClassifier  # assumes xgboost installed

    features = df.drop(columns=[TARGET_COL, CLUSTER_COL])
    y = df[TARGET_COL].astype(int).values

    # Encode categorical as int codes (same convention as feature importance)
    tmp = df.drop(columns=[CLUSTER_COL]).copy()
    tmp = encode_categories_as_int(tmp, CAT_COLS)

    X = tmp.drop(columns=[TARGET_COL]).copy()
    y = tmp[TARGET_COL].astype(int).values

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **BEST_PARAMS,
    )
    model.fit(X, y)

    p_default = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["p_default"] = p_default

    # Quick sanity metric (not CV; just to ensure model runs)
    ap = average_precision_score(y, p_default)
    print(f"Average Precision on full data (sanity check, not CV): {ap:.4f}")

    # Boxplot of predicted probabilities by cluster
    clusters = sorted(df[CLUSTER_COL].unique())
    data = [df.loc[df[CLUSTER_COL] == c, "p_default"].values for c in clusters]

    plt.figure()
    plt.boxplot(data, labels=[f"Cluster {c}" for c in clusters], showfliers=False)
    plt.ylabel("Predicted probability of default")
    plt.title("Predicted default probability by cluster (XGBoost score)")
    plt.savefig(PLOT_BOX_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_data(DATA_PATH)

    # Plot 1: default rate by cluster
    plot_default_rate_by_cluster(df)

    # Plot 2: predicted probability by cluster
    plot_predicted_default_by_cluster(df)

    print(f"Saved plots to:\n - {PLOT_BAR_PATH}\n - {PLOT_BOX_PATH}")


if __name__ == "__main__":
    main()
