# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 07:50:53 2026

@author: Andres Tripiana
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)

# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_clean_step3.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PLOT_PATH = REPORTS_DIR / "pr_curve_xgb.png"

TARGET_COL = "default.payment.next.month"
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

N_SPLITS = 5
RANDOM_STATE = 42

# You can change this if you want a different operating point
TARGET_RECALL = 0.80


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )


def find_threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float) -> float:
    """
    Choose the HIGHEST threshold that still achieves recall >= min_recall.
    This gives the best precision you can get while meeting the recall constraint.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # thresholds has length len(recall)-1. For each threshold t_i, the corresponding recall is recall[i+1]
    # (because recall includes the point at threshold -> +inf at index 0).
    recall_for_thresholds = recall[1:]

    valid = thresholds[recall_for_thresholds >= min_recall]
    if len(valid) == 0:
        return 0.5  # not achievable; fallback

    return float(valid.max())



def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC uses probabilities (threshold-independent)
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    out["average_precision"] = average_precision_score(y_true, y_prob)
    return out


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    """
    Return TN, FP, FN, TP for a given threshold.
    """
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"threshold": thr, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def print_threshold_table(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float]) -> None:
    rows = []
    for thr in thresholds:
        counts = confusion_counts(y_true, y_prob, thr)
        m = metrics_at_threshold(y_true, y_prob, thr)
        rows.append({
            **counts,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })

    df_tab = pd.DataFrame(rows)
    print("\n=== THRESHOLD IMPACT (OOF) ===")
    print(df_tab.to_string(index=False, formatters={
        "precision": "{:.3f}".format,
        "recall": "{:.3f}".format,
        "f1": "{:.3f}".format,
        "threshold": "{:.2f}".format
    }))


def main() -> None:
    # --- Load
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # --- Compute class imbalance weight
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    # --- XGBoost model (simple, stable defaults)
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception as e:
        raise ImportError(
            "xgboost is not installed. Install it (pip/conda) to run this script."
        ) from e

    pre = build_preprocessor(X)

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        # keep these fixed; we'll tune the rest
        objective="binary:logistic",
    )

    pipe = Pipeline([("pre", pre), ("xgb", xgb)])

    # --- CV and Grid
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        "xgb__n_estimators": [200, 500, 700],
        "xgb__max_depth": [3, 4, 6],
        "xgb__learning_rate": [0.01, 0.05, 0.1],
        "xgb__subsample": [0.6, 0.8, 1.0],
        "xgb__colsample_bytree": [0.6, 0.8, 1.0],
    }
    


    # Optimize PR-AUC (Average Precision) for imbalanced classification
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    gs.fit(X, y)

    print("\n=== XGBOOST GRID SEARCH (CV=5) ===")
    print(f"Best CV Average Precision: {gs.best_score_:.4f}")
    print("Best params:")
    for k, v in gs.best_params_.items():
        print(f"  {k}: {v}")

    best_model = gs.best_estimator_

    # --- Out-of-fold probabilities for an honest PR curve
    y_prob_oof = cross_val_predict(
        best_model, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    # --- Metrics at default threshold 0.5
    m_05 = metrics_at_threshold(y.values, y_prob_oof, thr=0.5)

    # --- Threshold tuned for target recall
    thr_recall = find_threshold_for_recall(y.values, y_prob_oof, min_recall=TARGET_RECALL)
    m_thr = metrics_at_threshold(y.values, y_prob_oof, thr=thr_recall)

    print("\n=== OOF METRICS (out-of-fold) ===")
    print("Threshold = 0.50")
    for k, v in m_05.items():
        print(f"  {k:16s}: {v:.4f}")
        
    # --- Confusion matrix counts across thresholds (OOF)
    thresholds_to_check = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print_threshold_table(y.values, y_prob_oof, thresholds_to_check)

    print(f"\nThreshold tuned for recall >= {TARGET_RECALL:.2f}  -> thr={thr_recall:.4f}")
    for k, v in m_thr.items():
        print(f"  {k:16s}: {v:.4f}")

    # --- Plot: Precision-Recall curve with markers
    precision, recall, thresholds = precision_recall_curve(y.values, y_prob_oof)
    ap = average_precision_score(y.values, y_prob_oof)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (OOF) | AP={ap:.3f}")

    # Mark threshold 0.5 point (approx by computing metrics)
    p05 = m_05["precision"]
    r05 = m_05["recall"]
    plt.scatter([r05], [p05], marker="o")
    plt.annotate("thr=0.50", (r05, p05), textcoords="offset points", xytext=(8, 6))

    # Mark recall-target threshold point
    pT = m_thr["precision"]
    rT = m_thr["recall"]
    plt.scatter([rT], [pT], marker="o")
    plt.annotate(f"thr={thr_recall:.3f}", (rT, pT), textcoords="offset points", xytext=(8, -12))

    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
