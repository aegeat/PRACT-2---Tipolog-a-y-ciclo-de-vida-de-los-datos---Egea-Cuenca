# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 09:06:22 2026

@author: Andres Tripiana
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_clean_step3.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PLOT_SIL_PATH = REPORTS_DIR / "clustering_silhouette.png"
PLOT_PCA_PATH = REPORTS_DIR / "clustering_pca.png"

TARGET_COL = "default.payment.next.month"

RANDOM_STATE = 42
K_RANGE = range(2, 9)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only behavioural / financial variables for clustering.
    """
    feature_cols = (
        ["LIMIT_BAL", "AGE"] +
        [c for c in df.columns if c.startswith("PAY_")] +
        [c for c in df.columns if c.startswith("BILL_AMT")] +
        [c for c in df.columns if c.startswith("PAY_AMT")]
    )
    return df[feature_cols].copy()


def main() -> None:
    # --- Load
    df = load_data(DATA_PATH)

    # --- Feature selection (no target!)
    X = select_clustering_features(df)

    # --- Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Choose k via silhouette
    silhouettes = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"k={k}: silhouette={sil:.4f}")

    # --- Plot silhouette vs k
    plt.figure()
    plt.plot(list(K_RANGE), silhouettes, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score vs number of clusters")
    plt.savefig(PLOT_SIL_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    # --- Choose best k
    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"\nChosen number of clusters: k={best_k}")

    # --- Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # --- PCA for visualization only
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means clusters (PCA projection)")
    plt.colorbar(scatter, label="Cluster")
    plt.savefig(PLOT_PCA_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    # --- Cluster profiles (means)
    profile = df.groupby("cluster")[X.columns.tolist()].mean()

    print("\n=== CLUSTER PROFILES (mean values) ===")
    print(profile.round(2))

    # --- Default rate per cluster (interpretation only)
    default_rate = df.groupby("cluster")[TARGET_COL].mean()
    print("\n=== DEFAULT RATE PER CLUSTER (not used for clustering) ===")
    print(default_rate.round(3))

    # Save enriched dataset
    out_path = BASE_DIR / "data" / "credit_with_clusters.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved dataset with clusters to: {out_path}")
    print(f"Saved plots to:\n - {PLOT_SIL_PATH}\n - {PLOT_PCA_PATH}")


if __name__ == "__main__":
    main()
