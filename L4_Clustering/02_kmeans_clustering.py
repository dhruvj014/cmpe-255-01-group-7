"""K-Means Clustering on reviewer behavioural features.

Reads the scaled feature matrix produced by 01_preprocessing.py, selects an
optimal K via the elbow method and silhouette analysis, then runs the final
K-Means model and produces cluster-level summaries.

Outputs:
    outputs/reviewer_clusters.csv              — per-reviewer cluster assignments
    outputs/kmeans_cluster_summary.csv         — aggregate stats per cluster
    plots/l4_kmeans_elbow_silhouette.png       — K selection diagnostics
    plots/l4_kmeans_spam_rate_by_cluster.png   — spam rate by cluster bar chart

Usage:
    python L4_Clustering/02_kmeans_clustering.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALED_PATH = os.path.join(PROJECT_ROOT, "L4_Clustering", "outputs", "clustering_features_scaled.csv")
RAW_PATH = os.path.join(PROJECT_ROOT, "L4_Clustering", "outputs", "clustering_features_raw.csv")

SPAM_RATE_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "reviewer_features.csv"),
    os.path.join(PROJECT_ROOT, "L1_ETL_OLAP", "output_csv", "reviewer_profiles.csv"),
    RAW_PATH,
]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "L4_Clustering", "outputs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

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

K_RANGE = range(2, 13)
SILHOUETTE_SAMPLE = 20_000
RANDOM_STATE = 42
DATASET_SPAM_MEAN = 0.132


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_spam_rate() -> pd.DataFrame:
    """Return a DataFrame with columns [user_id, spam_rate]."""
    for path in SPAM_RATE_CANDIDATES:
        if os.path.exists(path):
            print(f"  spam_rate source: {path}")
            df = pd.read_csv(path, usecols=["user_id", "spam_rate"])
            return df
    raise FileNotFoundError(
        "Could not locate spam_rate data. Looked in:\n"
        + "\n".join(f"  - {p}" for p in SPAM_RATE_CANDIDATES)
    )


def _detect_elbow(inertias: list[float]) -> int:
    """Find the elbow point using the maximum-distance-to-line heuristic."""
    coords = np.array(list(zip(range(len(inertias)), inertias)))
    line_start, line_end = coords[0], coords[-1]
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len

    distances = []
    for pt in coords:
        vec = pt - line_start
        proj_len = np.dot(vec, line_unit)
        proj = line_start + proj_len * line_unit
        distances.append(np.linalg.norm(pt - proj))

    return int(np.argmax(distances))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- 1. Load scaled features ----
    print(f"Reading {SCALED_PATH} ...")
    df_scaled = pd.read_csv(SCALED_PATH)
    print(f"  Loaded {len(df_scaled):,} rows")

    user_ids = df_scaled["user_id"]
    X = df_scaled[CLUSTERING_FEATURES].values

    # ---- 2. Load spam_rate for post-hoc join ----
    print("Locating spam_rate ...")
    df_spam = _find_spam_rate()

    # ---- 3. K selection: elbow + silhouette ----
    print("\nRunning K-Means for K = 2..12 ...")
    inertias: list[float] = []
    silhouettes: list[float] = []

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    X_sample = X[sample_idx]

    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        km.fit(X)
        inertias.append(km.inertia_)

        labels_sample = km.labels_[sample_idx]
        sil = silhouette_score(X_sample, labels_sample)
        silhouettes.append(sil)
        print(f"  K={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    ks = list(K_RANGE)
    elbow_idx = _detect_elbow(inertias)
    elbow_k = ks[elbow_idx]
    sil_peak_idx = int(np.argmax(silhouettes))
    sil_peak_k = ks[sil_peak_idx]

    print(f"\n  Elbow point   : K = {elbow_k}")
    print(f"  Silhouette peak: K = {sil_peak_k}")

    # ---- 4. Save elbow / silhouette plot ----
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ks, inertias, "o-", color="#2563eb", linewidth=2, markersize=6)
    ax1.axvline(elbow_k, color="red", linestyle="--", linewidth=1.2, label=f"Elbow K={elbow_k}")
    ax1.set_xlabel("K (number of clusters)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Method")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ks, silhouettes, "o-", color="#059669", linewidth=2, markersize=6)
    ax2.axvline(sil_peak_k, color="red", linestyle="--", linewidth=1.2, label=f"Peak K={sil_peak_k}")
    ax2.set_xlabel("K (number of clusters)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    elbow_path = os.path.join(PLOTS_DIR, "l4_kmeans_elbow_silhouette.png")
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved K-selection plot -> {elbow_path}")

    # ---- 5. Choose recommended K ----
    # Default to silhouette peak; fall back to K=5 if methods disagree heavily
    recommended_k = sil_peak_k
    if abs(elbow_k - sil_peak_k) > 2:
        recommended_k = 5  # ambiguous — conservative default
        print(f"  Elbow and silhouette disagree (K={elbow_k} vs K={sil_peak_k}); defaulting to K={recommended_k}")
    print(f"\n  ** Recommended K = {recommended_k} **\n")

    # ---- 6. Final K-Means ----
    print(f"Fitting final K-Means with K={recommended_k} ...")
    km_final = KMeans(n_clusters=recommended_k, n_init=20, random_state=RANDOM_STATE)
    km_final.fit(X)
    labels = km_final.labels_

    # ---- 7. Build reviewer-level output ----
    df_result = pd.DataFrame({"user_id": user_ids, "cluster_id": labels})
    df_result = df_result.merge(df_spam, on="user_id", how="left")

    # ---- 8. Load raw features for summary stats ----
    df_raw = pd.read_csv(RAW_PATH)

    df_summary_input = df_result.merge(df_raw[["user_id"] + CLUSTERING_FEATURES], on="user_id", how="left")

    summary = (
        df_summary_input.groupby("cluster_id")
        .agg(
            size=("user_id", "count"),
            spam_rate_mean=("spam_rate", "mean"),
            spam_rate_std=("spam_rate", "std"),
            avg_review_count=("review_count", "mean"),
            avg_tenure_days=("tenure_days", "mean"),
            avg_burst_score=("burst_score", "mean"),
            avg_max_seller_fraction=("max_seller_fraction", "mean"),
            avg_rating_entropy=("rating_entropy", "mean"),
        )
        .reset_index()
    )

    # ---- 9. Save outputs ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clusters_path = os.path.join(OUTPUT_DIR, "reviewer_clusters.csv")
    df_result[["user_id", "cluster_id", "spam_rate"]].to_csv(clusters_path, index=False)
    print(f"  Saved cluster assignments -> {clusters_path}")

    summary_path = os.path.join(OUTPUT_DIR, "kmeans_cluster_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Saved cluster summary     -> {summary_path}")

    # ---- 10. Print summary ----
    print("\n" + "=" * 90)
    print("CLUSTER SUMMARY")
    print("=" * 90)
    with pd.option_context("display.max_columns", None, "display.width", 120, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    print("=" * 90)

    # ---- 11. Spam-rate bar chart ----
    summary_sorted = summary.sort_values("spam_rate_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 0.8 * len(summary_sorted) + 2))
    bars = ax.barh(
        summary_sorted["cluster_id"].astype(str),
        summary_sorted["spam_rate_mean"],
        xerr=summary_sorted["spam_rate_std"],
        color="#6366f1",
        edgecolor="white",
        capsize=4,
        height=0.6,
    )
    ax.axvline(DATASET_SPAM_MEAN, color="red", linestyle="--", linewidth=1.5, label=f"Dataset mean ({DATASET_SPAM_MEAN})")
    ax.set_xlabel("Mean Spam Rate")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Spam Rate by K-Means Cluster")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    bar_path = os.path.join(PLOTS_DIR, "l4_kmeans_spam_rate_by_cluster.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved spam-rate bar chart -> {bar_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
