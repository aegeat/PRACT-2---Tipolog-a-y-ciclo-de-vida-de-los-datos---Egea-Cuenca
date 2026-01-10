# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:11:23 2026

@author: Andres Tripiana
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_clean_step3.csv"

TARGET_COL = "default.payment.next.month"
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

N_SPLITS = 5
RANDOM_STATE = 42


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


def get_models():
    models = {}
    

    # Random Forest (simple baseline)
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    # HistGradientBoosting (strong sklearn baseline)
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_depth=4,
        learning_rate=0.05,
    )
    

    # XGBoost (if available)
    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="error",
        )

    return models


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC AUC only if probabilities exist and both classes are present
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    return out


def evaluate_model_cv(name: str, model, X: pd.DataFrame, y: pd.Series) -> None:
    print(f"\n==============================")
    print(f"MODEL: {name}")
    print(f"==============================")

    pre = build_preprocessor(X)
    pipe = Pipeline([("pre", pre), ("clf", model)])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    train_metrics = []
    test_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        pipe.fit(X_tr, y_tr)

        # Train
        y_tr_pred = pipe.predict(X_tr)
        y_tr_prob = pipe.predict_proba(X_tr)[:, 1] if hasattr(pipe, "predict_proba") else None
        tr_m = compute_metrics(y_tr, y_tr_pred, y_tr_prob) if y_tr_prob is not None else compute_metrics(y_tr, y_tr_pred, np.zeros_like(y_tr, dtype=float))

        # Test (validation fold)
        y_te_pred = pipe.predict(X_te)
        y_te_prob = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None
        te_m = compute_metrics(y_te, y_te_pred, y_te_prob) if y_te_prob is not None else compute_metrics(y_te, y_te_pred, np.zeros_like(y_te, dtype=float))

        train_metrics.append(tr_m)
        test_metrics.append(te_m)

        print(
            f"Fold {fold}: "
            f"TRAIN recall={tr_m['recall']:.3f} f1={tr_m['f1']:.3f} auc={tr_m['roc_auc']:.3f} | "
            f"TEST  recall={te_m['recall']:.3f} f1={te_m['f1']:.3f} auc={te_m['roc_auc']:.3f}"
        )

    def summarize(metrics_list, label: str):
        dfm = pd.DataFrame(metrics_list)
        mean = dfm.mean(numeric_only=True)
        std = dfm.std(numeric_only=True)
        print(f"\n{label} (mean ± std over {N_SPLITS} folds):")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            print(f"  {k:9s}: {mean[k]:.4f} ± {std[k]:.4f}")

    summarize(train_metrics, "TRAIN")
    summarize(test_metrics, "TEST")


def main() -> None:
    df = load_data(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    models = get_models()
    for name, model in models.items():
        evaluate_model_cv(name, model, X, y)


if __name__ == "__main__":
    main()
