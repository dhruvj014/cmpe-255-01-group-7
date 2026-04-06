"""DBSCAN Clustering on reviewer behavioural features.

Reads the scaled feature matrix produced by 01_preprocessing.py, tunes eps via
k-distance analysis, runs DBSCAN, and examines noise points as potential
spammer candidates.

Outputs:
    outputs/dbscan_results.csv              — per-reviewer DBSCAN assignments
    outputs/dbscan_cluster_summary.csv      — aggregate stats per cluster
    plots/l4_dbscan_kdistance.png           — k-distance graph for eps selection
    plots/l4_dbscan_noise_vs_cluster.png    — noise vs cluster feature comparison

Usage:
    python L4_Clustering/03_dbscan_clustering.py
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

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

DATASET_SPAM_MEAN = 0.132
MIN_SAMPLES = 10
EPS_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7]
TARGET_NOISE_RANGE = (0.01, 0.05)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    """Print with immediate flush so output always appears on Windows."""
    print(msg, flush=True)


def _find_spam_rate() -> pd.DataFrame:
    """Return a DataFrame with columns [user_id, spam_rate]."""
    for path in SPAM_RATE_CANDIDATES:
        if os.path.exists(path):
            log(f"  spam_rate source: {path}")
            df = pd.read_csv(path, usecols=["user_id", "spam_rate"])
            return df
    raise FileNotFoundError(
        "Could not locate spam_rate data. Looked in:\n"
        + "\n".join(f"  - {p}" for p in SPAM_RATE_CANDIDATES)
    )


def _select_best_eps(results: list[dict]) -> float:
    """Pick the eps that yields a noise fraction closest to the 1-5% target."""
    in_range = [r for r in results if TARGET_NOISE_RANGE[0] <= r["noise_frac"] <= TARGET_NOISE_RANGE[1]]
    if in_range:
        return min(in_range, key=lambda r: abs(r["noise_frac"] - 0.03))["eps"]
    # Nothing in range — fall back to closest to 3%
    best = min(results, key=lambda r: abs(r["noise_frac"] - 0.03))
    if best["noise_frac"] > 0:
        return best["eps"]
    return 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=" * 70)
    log("  DBSCAN CLUSTERING PIPELINE")
    log("=" * 70)
    log(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    log(f"  SCALED_PATH  : {SCALED_PATH}")
    log(f"  RAW_PATH     : {RAW_PATH}")
    log(f"  OUTPUT_DIR   : {OUTPUT_DIR}")
    log(f"  PLOTS_DIR    : {PLOTS_DIR}")
    log()

    # ---- 0. Verify input files exist ----
    log("[Step 0] Checking input files ...")
    if not os.path.exists(SCALED_PATH):
        log(f"  ERROR: Scaled features not found at {SCALED_PATH}")
        log("  Run 'python L4_Clustering/01_preprocessing.py' first.")
        sys.exit(1)
    log(f"  OK: {SCALED_PATH}")

    if not os.path.exists(RAW_PATH):
        log(f"  ERROR: Raw features not found at {RAW_PATH}")
        log("  Run 'python L4_Clustering/01_preprocessing.py' first.")
        sys.exit(1)
    log(f"  OK: {RAW_PATH}")

    spam_found = False
    for p in SPAM_RATE_CANDIDATES:
        if os.path.exists(p):
            log(f"  OK: spam_rate source -> {p}")
            spam_found = True
            break
    if not spam_found:
        log("  ERROR: No spam_rate source found.")
        sys.exit(1)
    log()

    # ---- 1. Load scaled features ----
    log("[Step 1] Loading scaled features ...")
    df_scaled = pd.read_csv(SCALED_PATH)
    log(f"  Loaded {len(df_scaled):,} rows, {len(df_scaled.columns)} columns")
    log(f"  Columns: {list(df_scaled.columns)}")

    user_ids = df_scaled["user_id"]
    X = df_scaled[CLUSTERING_FEATURES].values
    log(f"  Feature matrix shape: {X.shape}")
    log()

    # ---- 2. Load spam_rate for post-hoc analysis ----
    log("[Step 2] Loading spam_rate ...")
    df_spam = _find_spam_rate()
    log(f"  spam_rate rows: {len(df_spam):,}")
    log()

    # ---- 3. k-distance graph for eps tuning ----
    log("[Step 3] Computing k-distance graph (k=5) ...")
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    log("  NearestNeighbors fitted.")
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])[::-1]
    log(f"  k-distances computed. Range: [{k_distances[-1]:.4f}, {k_distances[0]:.4f}]")
    log(f"  Median 5-NN distance: {np.median(k_distances):.4f}")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(k_distances)), k_distances, color="#2563eb", linewidth=1.2)
    ax.set_xlabel("Points (sorted by distance)")
    ax.set_ylabel("5th Nearest Neighbor Distance")
    ax.set_title("k-Distance Graph (k=5) — Look for the Knee")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    kdist_path = os.path.join(PLOTS_DIR, "l4_dbscan_kdistance.png")
    fig.savefig(kdist_path, dpi=150)
    plt.close(fig)
    log(f"  Saved k-distance plot -> {kdist_path}")
    log()

    # ---- 4. Eps sweep ----
    log("[Step 4] Epsilon sweep ...")
    log(f"  eps candidates: {EPS_CANDIDATES}")
    log(f"  min_samples   : {MIN_SAMPLES}")
    log()
    log(f"  {'eps':>6s}  {'clusters':>8s}  {'noise_pts':>10s}  {'noise_frac':>10s}")
    log("  " + "-" * 40)

    sweep_results: list[dict] = []
    for eps in EPS_CANDIDATES:
        try:
            db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            noise_frac = n_noise / len(labels)
            sweep_results.append({
                "eps": eps,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_frac": noise_frac,
            })
            log(f"  {eps:6.1f}  {n_clusters:8d}  {n_noise:10,d}  {noise_frac:10.4f}")
        except MemoryError:
            log(f"  {eps:6.1f}  ** MemoryError — eps too large for dataset, skipping **")
            break

    log()

    # ---- 5. Select best eps and run final DBSCAN ----
    best_eps = _select_best_eps(sweep_results)
    log(f"  ** Selected eps = {best_eps} (targeting {TARGET_NOISE_RANGE[0]*100:.0f}-{TARGET_NOISE_RANGE[1]*100:.0f}% noise) **")
    log()

    log(f"[Step 5] Running final DBSCAN with eps={best_eps}, min_samples={MIN_SAMPLES} ...")
    db_final = DBSCAN(eps=best_eps, min_samples=MIN_SAMPLES)
    final_labels = db_final.fit_predict(X)

    n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
    n_noise = int(np.sum(final_labels == -1))
    noise_pct = n_noise / len(final_labels) * 100
    log(f"  Clusters found : {n_clusters}")
    log(f"  Noise points   : {n_noise:,} ({noise_pct:.2f}%)")
    log()

    # ---- 6. Join cluster labels with spam_rate ----
    log("[Step 6] Joining cluster labels with spam_rate ...")
    df_result = pd.DataFrame({"user_id": user_ids, "dbscan_cluster": final_labels})
    df_result = df_result.merge(df_spam, on="user_id", how="left")
    log(f"  Result shape: {df_result.shape}")
    log()

    # ---- 7. Cluster summary ----
    log("[Step 7] Computing cluster summary ...")
    summary = (
        df_result.groupby("dbscan_cluster")
        .agg(size=("user_id", "count"), spam_rate_mean=("spam_rate", "mean"))
        .reset_index()
    )
    log(f"  Summary rows: {len(summary)}")
    log()

    # ---- 8. Save outputs ----
    log("[Step 8] Saving output files ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_path = os.path.join(OUTPUT_DIR, "dbscan_results.csv")
    df_result[["user_id", "dbscan_cluster", "spam_rate"]].to_csv(results_path, index=False)
    log(f"  Saved assignments -> {results_path}")

    summary_path = os.path.join(OUTPUT_DIR, "dbscan_cluster_summary.csv")
    summary.to_csv(summary_path, index=False)
    log(f"  Saved summary     -> {summary_path}")
    log()

    # ---- 9. Print summary table ----
    log("=" * 60)
    log("DBSCAN CLUSTER SUMMARY")
    log("=" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 120, "display.float_format", "{:.4f}".format):
        log(summary.to_string(index=False))
    log("=" * 60)

    noise_spam = df_result.loc[df_result["dbscan_cluster"] == -1, "spam_rate"].mean()
    noise_spam_pct = noise_spam * 100 if not np.isnan(noise_spam) else 0.0

    log(
        f"\nDBSCAN found {n_clusters} clusters and {n_noise:,} noise reviewers "
        f"({noise_pct:.1f}%). Noise group spam rate: {noise_spam_pct:.1f}% "
        f"vs dataset average {DATASET_SPAM_MEAN * 100:.1f}%"
    )
    log()

    # ---- 10. Noise analysis: comparison boxplots ----
    log("[Step 10] Noise analysis — feature comparison ...")

    df_raw = pd.read_csv(RAW_PATH)
    log(f"  Loaded raw features: {df_raw.shape}")

    df_analysis = df_result[["user_id", "dbscan_cluster"]].merge(df_raw, on="user_id", how="left")
    df_analysis["group"] = np.where(df_analysis["dbscan_cluster"] == -1, "Noise (-1)", "Clustered")

    noise_stats = df_analysis.loc[df_analysis["group"] == "Noise (-1)"]
    cluster_stats = df_analysis.loc[df_analysis["group"] == "Clustered"]

    compare_features = ["burst_score", "tenure_days", "max_seller_fraction"]

    log(f"  Noise reviewers    : {len(noise_stats):,}")
    log(f"  Clustered reviewers: {len(cluster_stats):,}")
    log()
    log(f"  {'feature':>22s}  {'noise_mean':>12s}  {'cluster_mean':>12s}")
    log("  " + "-" * 50)
    for feat in compare_features:
        n_mean = noise_stats[feat].mean()
        c_mean = cluster_stats[feat].mean()
        log(f"  {feat:>22s}  {n_mean:12.4f}  {c_mean:12.4f}")
    log()

    log("  Generating 3-panel boxplot ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, compare_features):
        data = [
            cluster_stats[feat].dropna().values,
            noise_stats[feat].dropna().values,
        ]
        bp = ax.boxplot(data, tick_labels=["Clustered", "Noise (-1)"], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#6366f1")
        bp["boxes"][1].set_facecolor("#ef4444")
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        ax.set_title(feat, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("DBSCAN Noise vs Clustered Reviewers", fontsize=14, y=1.02)
    plt.tight_layout()

    boxplot_path = os.path.join(PLOTS_DIR, "l4_dbscan_noise_vs_cluster.png")
    fig.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved noise comparison plot -> {boxplot_path}")

    log()
    log("=" * 70)
    log("  DBSCAN PIPELINE COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
