"""Clustering Feature Preprocessing.

Reads reviewer_profiles.csv (produced by the L1 ETL pipeline) and prepares a
scaled feature matrix suitable for distance-based clustering (K-Means, DBSCAN,
etc.).

Outputs:
    outputs/clustering_features_scaled.csv  — StandardScaler-transformed features
    outputs/clustering_features_raw.csv     — log-transformed (unscaled) features
                                              with user_id and spam_rate
    plots/l4_feature_correlation.png        — Pearson correlation heatmap

Usage:
    python L4_Clustering/01_preprocessing.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "L1_ETL_OLAP", "output_csv", "reviewer_profiles.csv"),
    os.path.join(PROJECT_ROOT, "reviewer_features.csv"),
]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "L4_Clustering", "outputs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# The 10 behavioural features used as clustering inputs.
# NOTE: The user-facing spec listed unique_sellers twice; deduplicated here.
CLUSTERING_FEATURES = [
    "review_count",
    "reviews_per_week",
    "tenure_days",
    "avg_rating",
    "rating_std",
    "unique_sellers",
    "max_seller_fraction",
    "burst_score",
    "rating_entropy",
    "avg_review_length",
]

# Features that follow power-law distributions (median reviewer has 1 review)
# and benefit from a log1p compression before scaling.
LOG_TRANSFORM_FEATURES = [
    "review_count",
    "reviews_per_week",
    "tenure_days",
    "burst_score",
    "unique_sellers",
]

# spam_rate is deliberately EXCLUDED from clustering inputs.
# Including it would cause data leakage: we want behavioural clusters to
# emerge organically from reviewer activity patterns.  spam_rate is a label
# (derived from ground-truth spam annotations), so folding it into the
# feature space would let the algorithm "see" the answer we later want to
# compare clusters against.  We keep it alongside the raw output for
# post-hoc analysis only.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- 1. Locate and load input CSV ----
    input_path = None
    for candidate in INPUT_CANDIDATES:
        if os.path.exists(candidate):
            input_path = candidate
            break

    if input_path is None:
        raise FileNotFoundError(
            "Could not find reviewer feature data.  Looked in:\n"
            + "\n".join(f"  - {p}" for p in INPUT_CANDIDATES)
            + "\nRun the L1 ETL pipeline first (L1_ETL_OLAP/main.py)."
        )

    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    missing_cols = [c for c in CLUSTERING_FEATURES + ["user_id", "spam_rate"] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # ---- 2. Drop rows where ALL clustering features are null ----
    rows_before = len(df)
    df = df.dropna(subset=CLUSTERING_FEATURES, how="all")
    rows_dropped = rows_before - len(df)
    print(f"  Dropped {rows_dropped:,} rows where all clustering features were null")

    # Separate spam_rate for post-hoc use (NOT a clustering input — see note above)
    spam_rate = df[["user_id", "spam_rate"]].copy()

    # ---- 3. Log1p transform on heavily right-skewed features ----
    df_features = df[CLUSTERING_FEATURES].copy()
    for col in LOG_TRANSFORM_FEATURES:
        df_features[col] = np.log1p(df_features[col])

    # ---- 4. Correlation heatmap (before scaling, after log transform) ----
    os.makedirs(PLOTS_DIR, exist_ok=True)

    corr = df_features.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation — Clustering Features (log-transformed)", fontsize=13)
    plt.tight_layout()

    heatmap_path = os.path.join(PLOTS_DIR, "l4_feature_correlation.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved correlation heatmap -> {heatmap_path}")

    # ---- 5. StandardScaler ----
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_array, columns=CLUSTERING_FEATURES, index=df.index)

    # ---- 6. Save outputs ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Scaled matrix with user_id preserved
    df_scaled.insert(0, "user_id", df["user_id"].values)
    scaled_path = os.path.join(OUTPUT_DIR, "clustering_features_scaled.csv")
    df_scaled.to_csv(scaled_path, index=False)
    print(f"  Saved scaled features   -> {scaled_path}")

    # Raw (log-transformed, unscaled) with user_id + spam_rate for analysis
    df_raw = df_features.copy()
    df_raw.insert(0, "user_id", df["user_id"].values)
    df_raw["spam_rate"] = spam_rate["spam_rate"].values
    raw_path = os.path.join(OUTPUT_DIR, "clustering_features_raw.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"  Saved raw features      -> {raw_path}")

    # ---- 7. Diagnostics ----
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Output shape           : {df_scaled.shape[0]:,} rows × {len(CLUSTERING_FEATURES)} features")
    print(f"  Rows dropped (all-null): {rows_dropped:,}")

    # Flag highly correlated pairs (|r| > 0.85) — redundancy warning for
    # distance-based methods where correlated dimensions inflate distances.
    high_corr_pairs = []
    for i in range(len(CLUSTERING_FEATURES)):
        for j in range(i + 1, len(CLUSTERING_FEATURES)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append((CLUSTERING_FEATURES[i], CLUSTERING_FEATURES[j], r))

    if high_corr_pairs:
        print("\n  WARNING: Highly correlated feature pairs (|r| > 0.85):")
        for f1, f2, r in high_corr_pairs:
            print(f"     {f1}  <->  {f2}  (r = {r:.3f})")
        print("     Consider PCA or dropping one of each pair to reduce redundancy.")
    else:
        print("\n  OK: No feature pairs with |r| > 0.85 -- no redundancy concern.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
