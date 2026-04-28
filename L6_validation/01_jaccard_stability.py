"""Cross-signal Jaccard stability analysis (L6).

For several values of K, extract the top-K most-suspicious reviewer user_id
sets from each signal source and compute pairwise Jaccard similarity.

Signal sources (all restricted to the L5 holdout population):
    L1-behavioral  : reviewer_profiles  sorted by spam_rate desc
    L2-rules       : l5_feature_table   sorted by l2_rule_max_weight desc
    L4-kmeans      : reviewer_clusters  joined with kmeans_cluster_summary,
                     ranked by cluster spam_rate_mean desc
    L4-dbscan      : dbscan_results     joined with dbscan_cluster_summary,
                     ranked by cluster spam_rate_mean desc (noise = -1 at top)
    L5-supervised  : supervised_holdout_predictions  aggregated to reviewer-level
                     (max score per user_id), sorted by score desc
    L5-anomaly     : anomaly_holdout_scores  aggregated to reviewer-level
                     (max score per user_id), sorted by IF score desc

Outputs:
    outputs/jaccard_matrix.csv    K x signal-pair table
    outputs/topk_overlap_table.csv  raw overlap counts and union sizes
    plots/jaccard_heatmap.png      heatmap at K=2000
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    L4_DB_SUMMARY,
    L4_DBSCAN,
    L4_KM_SUMMARY,
    L4_KMEANS,
    L5_ANOM_SCORES,
    L5_FEATURE_TABLE,
    L5_SUP_PRED,
    OUTPUT_DIR,
    PLOT_DIR,
    aggregate_review_scores_to_reviewer,
    load_best_supervised_name,
    load_holdout_user_ids,
    load_reviewer_profiles,
)

K_VALUES = [500, 1000, 2000, 5000, 10000]
HEATMAP_K = 2000

SIGNALS = [
    "L1-behavioral",
    "L2-rules",
    "L4-kmeans",
    "L4-dbscan",
    "L5-supervised",
    "L5-anomaly",
]


def _rank_filtered(df: pd.DataFrame, holdout: set, sort_cols: list[str],
                   ascending: list[bool]) -> pd.Series:
    """Return a holdout-restricted series of user_ids sorted by sort_cols."""
    df = df[df["user_id"].isin(holdout)].copy()
    df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return df["user_id"].reset_index(drop=True)


def _load_rankings(holdout: set) -> dict[str, pd.Series]:
    """Load every signal's ranked user_id series (restricted to holdout)."""
    rankings: dict[str, pd.Series] = {}

    # --- L1-behavioral: spam_rate desc, review_count desc as tiebreaker ---
    profiles = load_reviewer_profiles()[["user_id", "spam_rate", "review_count"]]
    rankings["L1-behavioral"] = _rank_filtered(
        profiles, holdout,
        sort_cols=["spam_rate", "review_count"], ascending=[False, False],
    )

    # --- L2-rules: l2_rule_max_weight desc, partial_max as tiebreaker ---
    # L5 feature table is now review-level; L2 features are per-reviewer
    # (identical for all reviews of a user), so just deduplicate.
    feat = pd.read_csv(
        L5_FEATURE_TABLE,
        usecols=["user_id", "l2_rule_max_weight", "l2_rule_partial_max"],
    )
    feat_dedup = feat.drop_duplicates(subset="user_id")
    rankings["L2-rules"] = _rank_filtered(
        feat_dedup, holdout,
        sort_cols=["l2_rule_max_weight", "l2_rule_partial_max"],
        ascending=[False, False],
    )

    # --- L4-kmeans: join cluster id -> cluster spam_rate_mean ---
    km = pd.read_csv(L4_KMEANS, usecols=["user_id", "cluster_id"])
    km_sum = pd.read_csv(L4_KM_SUMMARY, usecols=["cluster_id", "spam_rate_mean"])
    km = km.merge(km_sum, on="cluster_id", how="left")
    rankings["L4-kmeans"] = _rank_filtered(
        km, holdout,
        sort_cols=["spam_rate_mean", "user_id"], ascending=[False, True],
    )

    # --- L4-dbscan: noise points first, then by cluster spam rate ---
    db = pd.read_csv(L4_DBSCAN, usecols=["user_id", "dbscan_cluster"])
    db_sum = pd.read_csv(L4_DB_SUMMARY, usecols=["dbscan_cluster", "spam_rate_mean"])
    db = db.merge(db_sum, on="dbscan_cluster", how="left")
    rankings["L4-dbscan"] = _rank_filtered(
        db, holdout,
        sort_cols=["spam_rate_mean", "user_id"], ascending=[False, True],
    )

    # --- L5-supervised: aggregate review-level scores to reviewer-level ---
    best_name = load_best_supervised_name()
    score_col = f"{best_name}_score"
    sup = pd.read_csv(L5_SUP_PRED, usecols=["user_id", score_col])
    # Aggregate: max(score) per reviewer
    sup_agg = aggregate_review_scores_to_reviewer(sup, score_col, agg="max")
    rankings["L5-supervised"] = _rank_filtered(
        sup_agg, holdout,
        sort_cols=[score_col, "user_id"], ascending=[False, True],
    )

    # --- L5-anomaly: aggregate review-level scores to reviewer-level ---
    anom = pd.read_csv(L5_ANOM_SCORES, usecols=["user_id", "isolation_forest_score"])
    anom_agg = aggregate_review_scores_to_reviewer(anom, "isolation_forest_score", agg="max")
    rankings["L5-anomaly"] = _rank_filtered(
        anom_agg, holdout,
        sort_cols=["isolation_forest_score", "user_id"], ascending=[False, True],
    )

    return rankings


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    holdout = load_holdout_user_ids()
    print(f"Holdout population: {len(holdout):,} reviewers")

    rankings = _load_rankings(holdout)
    for name, series in rankings.items():
        print(f"  {name:15s} ranked {len(series):,} reviewers")

    pairs = list(combinations(SIGNALS, 2))

    # --- 1. Jaccard matrix at each K ---
    matrix_rows = []
    overlap_rows = []
    for K in K_VALUES:
        topk = {name: set(rankings[name].head(K).values) for name in SIGNALS}
        row = {"K": K}
        for a, b in pairs:
            j = _jaccard(topk[a], topk[b])
            row[f"{a}__vs__{b}"] = j
            overlap_rows.append({
                "K": K,
                "signal_a": a,
                "signal_b": b,
                "intersection": len(topk[a] & topk[b]),
                "union": len(topk[a] | topk[b]),
                "jaccard": j,
            })
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_path = OUTPUT_DIR / "jaccard_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False)
    print(f"\nSaved {matrix_path}")

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_path = OUTPUT_DIR / "topk_overlap_table.csv"
    overlap_df.to_csv(overlap_path, index=False)
    print(f"Saved {overlap_path}")

    # --- 2. Heatmap at HEATMAP_K ---
    topk = {name: set(rankings[name].head(HEATMAP_K).values) for name in SIGNALS}
    heat = pd.DataFrame(index=SIGNALS, columns=SIGNALS, dtype=float)
    for a in SIGNALS:
        for b in SIGNALS:
            heat.loc[a, b] = _jaccard(topk[a], topk[b])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        heat.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title(f"Cross-Signal Jaccard Similarity  (top-K = {HEATMAP_K:,}, L5 holdout)")
    plt.tight_layout()
    heatmap_path = PLOT_DIR / "jaccard_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Saved {heatmap_path}")

    # --- 3. Console summary ---
    print("\n" + "=" * 70)
    print(f"JACCARD SUMMARY AT K={HEATMAP_K}")
    print("=" * 70)
    pair_scores = [(a, b, heat.loc[a, b]) for a, b in pairs]
    pair_scores.sort(key=lambda t: t[2], reverse=True)
    for a, b, j in pair_scores:
        print(f"  {a:15s}  vs  {b:15s}  J = {j:.4f}")


if __name__ == "__main__":
    main()
