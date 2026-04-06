"""Unified Cluster Characterization Report.

Reads K-Means and DBSCAN cluster assignments alongside raw reviewer features,
then produces a consolidated cluster-spam summary CSV, a K-Means cluster
heatmap, and a side-by-side bar chart comparing spam rates across methods.

Outputs:
    L4_Clustering/outputs/cluster_spam_summary.csv  — consolidated characterization
    plots/l4_kmeans_cluster_heatmap.png             — feature heatmap by K-Means cluster
    plots/l4_cluster_comparison.png                 — K-Means vs DBSCAN spam bar charts

Usage:
    python L4_Clustering/04_cluster_analysis.py
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KMEANS_PATH = os.path.join(PROJECT_ROOT, "L4_Clustering", "outputs", "reviewer_clusters.csv")
DBSCAN_PATH = os.path.join(PROJECT_ROOT, "L4_Clustering", "outputs", "dbscan_results.csv")

RAW_FEATURE_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "L4_Clustering", "outputs", "clustering_features_raw.csv"),
    os.path.join(PROJECT_ROOT, "L1_ETL_OLAP", "output_csv", "reviewer_profiles.csv"),
    os.path.join(PROJECT_ROOT, "reviewer_features.csv"),
]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "L4_Clustering", "outputs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

HEATMAP_FEATURES = [
    "review_count",
    "tenure_days",
    "burst_score",
    "max_seller_fraction",
    "rating_entropy",
    "avg_rating",
    "rating_std",
    "unique_sellers",
]

DATASET_SPAM_MEAN = 0.132


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(msg, flush=True)


def _load_raw_features() -> pd.DataFrame:
    """Return a DataFrame with user_id, the 8 heatmap features, and spam_rate."""
    for path in RAW_FEATURE_CANDIDATES:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        missing = [c for c in HEATMAP_FEATURES + ["user_id", "spam_rate"] if c not in df.columns]
        if missing:
            log(f"  Skipping {path} — missing columns: {missing}")
            continue
        log(f"  Feature source: {path}")
        return df
    raise FileNotFoundError(
        "Could not locate raw features with required columns. Looked in:\n"
        + "\n".join(f"  - {p}" for p in RAW_FEATURE_CANDIDATES)
    )


def _classify_profile(means: pd.Series, global_means: pd.Series) -> str:
    """Generate a short dominant-profile tag string from cluster-level means."""
    tags: list[str] = []

    ratio = means / global_means.replace(0, np.nan)

    if ratio.get("tenure_days", 1) < 0.7:
        tags.append("New")
    elif ratio.get("tenure_days", 1) > 1.5:
        tags.append("Veteran")

    if ratio.get("burst_score", 1) > 1.4:
        tags.append("bursty")

    if ratio.get("max_seller_fraction", 1) > 1.3:
        tags.append("concentrated")
    elif ratio.get("max_seller_fraction", 1) < 0.7:
        tags.append("diverse")

    if ratio.get("review_count", 1) > 1.8:
        tags.append("active")
    elif ratio.get("review_count", 1) < 0.5:
        tags.append("low-activity")

    if ratio.get("rating_entropy", 1) < 0.6:
        tags.append("uniform-rating")
    elif ratio.get("rating_entropy", 1) > 1.4:
        tags.append("varied-rating")

    if ratio.get("unique_sellers", 1) > 1.5:
        tags.append("multi-seller")

    spam = means.get("spam_rate_mean", DATASET_SPAM_MEAN)
    if spam > DATASET_SPAM_MEAN * 1.5:
        tags.append("high-spam")
    elif spam < DATASET_SPAM_MEAN * 0.5:
        tags.append("low-spam")

    return " ".join(t.capitalize() if i == 0 else t for i, t in enumerate(tags)) if tags else "Average"


def _generate_notes(means: pd.Series) -> str:
    """Return a 1-sentence interpretation of spam suspicion for this cluster."""
    spam = means.get("spam_rate_mean", DATASET_SPAM_MEAN)
    ratio = spam / DATASET_SPAM_MEAN if DATASET_SPAM_MEAN else 1.0

    if ratio > 2.0:
        return (
            f"Spam rate {spam:.1%} is {ratio:.1f}x the dataset average; "
            "behavioural features indicate suspicious review patterns."
        )
    if ratio > 1.3:
        return (
            f"Spam rate {spam:.1%} is moderately elevated; "
            "mix of suspicious and borderline reviewers."
        )
    if ratio < 0.5:
        return (
            f"Spam rate {spam:.1%} is well below average; "
            "likely organic reviewers with normal purchasing behaviour."
        )
    return (
        f"Spam rate {spam:.1%} is near the dataset average ({DATASET_SPAM_MEAN:.1%}); "
        "no strong signal of spam or organic dominance."
    )


def _print_markdown_table(df: pd.DataFrame) -> None:
    """Print a DataFrame as a GitHub-flavoured markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    log(header)
    log(sep)
    for _, row in df.iterrows():
        cells: list[str] = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        log("| " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=" * 70)
    log("  CLUSTER CHARACTERIZATION REPORT")
    log("=" * 70)

    # ---- 0. Verify inputs ----
    log("\n[Step 0] Checking input files ...")
    for tag, path in [("K-Means", KMEANS_PATH), ("DBSCAN", DBSCAN_PATH)]:
        if not os.path.exists(path):
            log(f"  ERROR: {tag} results not found at {path}")
            log("  Run the upstream clustering scripts first.")
            sys.exit(1)
        log(f"  OK: {tag} -> {path}")

    # ---- 1. Load data ----
    log("\n[Step 1] Loading data ...")
    df_km = pd.read_csv(KMEANS_PATH)
    log(f"  K-Means assignments : {len(df_km):,} rows  columns={list(df_km.columns)}")

    df_db = pd.read_csv(DBSCAN_PATH)
    log(f"  DBSCAN assignments  : {len(df_db):,} rows  columns={list(df_db.columns)}")

    df_feat = _load_raw_features()
    log(f"  Raw features        : {len(df_feat):,} rows")

    # ---- 2. Merge features with cluster labels ----
    log("\n[Step 2] Merging features with cluster assignments ...")
    keep_cols = ["user_id"] + HEATMAP_FEATURES + ["spam_rate"]
    df_feat_sub = df_feat[[c for c in keep_cols if c in df_feat.columns]]

    df_km_merged = df_km[["user_id", "cluster_id"]].merge(df_feat_sub, on="user_id", how="left")
    df_db_merged = df_db[["user_id", "dbscan_cluster"]].merge(df_feat_sub, on="user_id", how="left")
    log(f"  K-Means merged shape: {df_km_merged.shape}")
    log(f"  DBSCAN  merged shape: {df_db_merged.shape}")

    # Global means for relative profiling
    global_means = df_feat_sub[HEATMAP_FEATURES].mean()

    # ---- 3. K-Means cluster heatmap ----
    log("\n[Step 3] Generating K-Means cluster heatmap ...")
    km_cluster_means = df_km_merged.groupby("cluster_id")[HEATMAP_FEATURES].mean()

    fig, ax = plt.subplots(figsize=(12, max(3, 0.8 * len(km_cluster_means) + 1)))
    sns.heatmap(
        km_cluster_means,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("K-Means Cluster Means — Raw Feature Values", fontsize=13)
    ax.set_ylabel("Cluster ID")
    ax.set_xlabel("")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    heatmap_path = os.path.join(PLOTS_DIR, "l4_kmeans_cluster_heatmap.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    log(f"  Saved -> {heatmap_path}")

    # ---- 4. Build consolidated cluster_spam_summary ----
    log("\n[Step 4] Building cluster_spam_summary ...")
    rows: list[dict] = []

    # K-Means rows
    km_summary = (
        df_km_merged.groupby("cluster_id")
        .agg(size=("user_id", "count"), spam_rate_mean=("spam_rate", "mean"))
        .reset_index()
    )
    for _, r in km_summary.iterrows():
        feat_means = km_cluster_means.loc[r["cluster_id"]]
        combined = pd.concat([feat_means, pd.Series({"spam_rate_mean": r["spam_rate_mean"]})])
        rows.append({
            "method": "kmeans",
            "cluster_id": int(r["cluster_id"]),
            "size": int(r["size"]),
            "spam_rate_mean": round(r["spam_rate_mean"], 4),
            "dominant_profile": _classify_profile(combined, global_means),
            "notes": _generate_notes(combined),
        })

    # DBSCAN rows (cluster -1 = noise)
    db_cluster_means = df_db_merged.groupby("dbscan_cluster")[HEATMAP_FEATURES].mean()
    db_summary = (
        df_db_merged.groupby("dbscan_cluster")
        .agg(size=("user_id", "count"), spam_rate_mean=("spam_rate", "mean"))
        .reset_index()
    )
    for _, r in db_summary.iterrows():
        cid = int(r["dbscan_cluster"])
        feat_means = db_cluster_means.loc[r["dbscan_cluster"]]
        combined = pd.concat([feat_means, pd.Series({"spam_rate_mean": r["spam_rate_mean"]})])
        label = "noise" if cid == -1 else str(cid)
        rows.append({
            "method": "dbscan",
            "cluster_id": label,
            "size": int(r["size"]),
            "spam_rate_mean": round(r["spam_rate_mean"], 4),
            "dominant_profile": _classify_profile(combined, global_means),
            "notes": _generate_notes(combined),
        })

    df_summary = pd.DataFrame(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "cluster_spam_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    log(f"  Saved -> {summary_path}")

    # ---- 5. Print markdown table ----
    log("\n" + "=" * 90)
    log("CLUSTER SPAM SUMMARY")
    log("=" * 90)
    _print_markdown_table(df_summary)
    log("=" * 90)

    # ---- 6. Comparison bar chart: K-Means vs DBSCAN ----
    log("\n[Step 6] Generating comparison bar chart ...")

    fig, (ax_km, ax_db) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left: K-Means — sorted descending by spam rate
    km_plot = km_summary.sort_values("spam_rate_mean", ascending=False).reset_index(drop=True)
    bars_km = ax_km.bar(
        km_plot["cluster_id"].astype(str),
        km_plot["spam_rate_mean"],
        color="#6366f1",
        edgecolor="white",
        width=0.6,
    )
    ax_km.axhline(DATASET_SPAM_MEAN, color="red", linestyle="--", linewidth=1.5,
                  label=f"Dataset avg ({DATASET_SPAM_MEAN})")
    ax_km.set_xlabel("K-Means Cluster")
    ax_km.set_ylabel("Mean Spam Rate")
    ax_km.set_title("K-Means")
    ax_km.legend(fontsize=9)
    ax_km.set_ylim(0.0, 0.5)
    ax_km.grid(axis="y", alpha=0.3)

    # Right: DBSCAN — noise bar coloured red
    db_plot = db_summary.sort_values("dbscan_cluster").reset_index(drop=True)
    labels_db = [("noise" if c == -1 else str(c)) for c in db_plot["dbscan_cluster"]]
    colors_db = ["#ef4444" if c == -1 else "#6366f1" for c in db_plot["dbscan_cluster"]]

    bars_db = ax_db.bar(
        labels_db,
        db_plot["spam_rate_mean"],
        color=colors_db,
        edgecolor="white",
        width=0.6,
    )
    ax_db.axhline(DATASET_SPAM_MEAN, color="red", linestyle="--", linewidth=1.5,
                  label=f"Dataset avg ({DATASET_SPAM_MEAN})")
    ax_db.set_xlabel("DBSCAN Cluster")
    ax_db.set_title("DBSCAN")
    ax_db.legend(fontsize=9)
    ax_db.set_ylim(0.0, 0.5)
    ax_db.grid(axis="y", alpha=0.3)

    fig.suptitle("Spam Rate by Cluster: K-Means vs DBSCAN", fontsize=14, y=1.02)
    plt.tight_layout()

    comparison_path = os.path.join(PLOTS_DIR, "l4_cluster_comparison.png")
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved -> {comparison_path}")

    log("\n" + "=" * 70)
    log("  CLUSTER ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
