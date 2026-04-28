"""Build Layer-5 feature table at REVIEW-LEVEL by fusing L1, L2, and L4 outputs.

Each row = one review with its own features + its reviewer's aggregated
features joined by user_id.

Inputs (auto-detected):
    - L1 reviews_enriched.csv  (608K reviews, per-review features + is_spam)
    - L1 reviewer_profiles.csv (260K reviewers, aggregated features)
    - L2_FPGrowth/outputs/spam_correlated_rules.csv
    - L4_Clustering/outputs/reviewer_clusters.csv
    - L4_Clustering/outputs/dbscan_results.csv

Outputs:
    - L5_Classification/outputs/l5_feature_table.csv
    - L5_Classification/outputs/l5_feature_columns.txt
    - L5_Classification/outputs/l5_build_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L5_OUTPUT_DIR = PROJECT_ROOT / "L5_Classification" / "outputs"

# Review-level data (608K rows)
REVIEWS_ENRICHED_CANDIDATES = [
    PROJECT_ROOT / "L1_ETL_OLAP" / "output_csv" / "reviews_enriched.csv",
    PROJECT_ROOT / "reviews_enriched.csv",
]

# Reviewer-level aggregated profiles (260K rows)
REVIEWER_PROFILE_CANDIDATES = [
    PROJECT_ROOT / "L1_ETL_OLAP" / "output_csv" / "reviewer_profiles.csv",
    PROJECT_ROOT / "reviewer_profiles.csv",
    PROJECT_ROOT / "reviewer_features.csv",
]

L2_RULES_PATH = PROJECT_ROOT / "L2_FPGrowth" / "outputs" / "spam_correlated_rules.csv"
L4_KMEANS_PATH = PROJECT_ROOT / "L4_Clustering" / "outputs" / "reviewer_clusters.csv"
L4_DBSCAN_PATH = PROJECT_ROOT / "L4_Clustering" / "outputs" / "dbscan_results.csv"

TOP_L2_RULES = 50

# Review-level columns to keep from reviews_enriched.csv
REVIEW_LEVEL_COLS = [
    "user_id",
    "prod_id",
    "rating",
    "is_spam",
    "review_length",
    "word_count",
    "exclamation_count",
    "question_count",
    "capital_ratio",
    "avg_word_length",
    "day_of_week",
    "month",
]

# Reviewer-aggregate columns to keep from reviewer_profiles.csv
REVIEWER_AGG_COLS = [
    "user_id",
    "review_count",
    "avg_rating",
    "rating_std",
    "avg_review_length",
    "avg_word_count",
    "unique_sellers",
    "tenure_days",
    "reviews_per_week",
    "max_seller_fraction",
    "avg_days_between_reviews",
    "burst_score",
    "rating_entropy",
]


def _pick_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in paths]}")


def _normalize_profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "total_reviews": "review_count",
        "mean_rating": "avg_rating",
        "mean_review_length": "avg_review_length",
        "mean_caps_ratio": "capital_ratio",
        "mean_exclamations": "avg_exclamations",
        "concentration": "max_seller_fraction",
    }
    applicable = {old: new for old, new in rename_map.items()
                  if old in df.columns and new not in df.columns}
    df = df.rename(columns=applicable)
    return df


# ---------------------------------------------------------------------------
# L2 rule discretisation helpers (same as before)
# ---------------------------------------------------------------------------

def _parse_frozenset(raw: str) -> frozenset[str]:
    try:
        return eval(raw, {"__builtins__": {}}, {"frozenset": frozenset})
    except Exception:
        return frozenset()


def _rc_bin(value: float) -> str:
    if value <= 2:
        return "review_count=Low"
    if value <= 10:
        return "review_count=Medium"
    return "review_count=High"


def _tenure_bin(value: float) -> str:
    if value < 30:
        return "tenure=new"
    if value <= 180:
        return "tenure=moderate"
    if value <= 365:
        return "tenure=established"
    return "tenure=veteran"


def _seller_bin(value: float) -> str:
    return "seller_conc=High" if value >= 0.5 else "seller_conc=Low"


def _burst_bin(value: float) -> str:
    return "burst=Bursty" if value > 2 else "burst=Normal"


def _build_item_set(row: pd.Series) -> set[str]:
    items: set[str] = set()

    if pd.notna(row.get("review_count", np.nan)):
        items.add(_rc_bin(float(row["review_count"])))
    if pd.notna(row.get("tenure_days", np.nan)):
        items.add(_tenure_bin(float(row["tenure_days"])))
    if pd.notna(row.get("max_seller_fraction", np.nan)):
        items.add(_seller_bin(float(row["max_seller_fraction"])))
    if pd.notna(row.get("burst_score", np.nan)):
        items.add(_burst_bin(float(row["burst_score"])))

    return items


def _load_l2_rules() -> pd.DataFrame:
    if not L2_RULES_PATH.exists():
        return pd.DataFrame(columns=["antecedents", "support", "confidence", "lift", "antecedent_spam_rate"])

    rules = pd.read_csv(L2_RULES_PATH)
    required = {"antecedents", "support", "confidence", "lift", "antecedent_spam_rate"}
    missing = sorted(required - set(rules.columns))
    if missing:
        raise KeyError(f"Missing columns in {L2_RULES_PATH}: {missing}")

    rules["antecedents"] = rules["antecedents"].astype(str).apply(_parse_frozenset)
    rules = rules[rules["antecedents"].apply(len) > 0].copy()
    rules["rule_weight"] = (
        rules["support"].astype(float)
        * rules["confidence"].astype(float)
        * rules["lift"].astype(float)
        * rules["antecedent_spam_rate"].astype(float)
    )
    rules = rules.sort_values("rule_weight", ascending=False).head(TOP_L2_RULES).reset_index(drop=True)
    return rules


def _compute_l2_features(base: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Compute L2 rule-match features per reviewer (operates on reviewer_profiles)."""
    if rules.empty:
        return pd.DataFrame(
            {
                "user_id": base["user_id"],
                "l2_rule_match_count": 0,
                "l2_rule_max_weight": 0.0,
                "l2_rule_partial_mean": 0.0,
                "l2_rule_partial_max": 0.0,
            }
        )

    antecedents = rules["antecedents"].tolist()
    weights = rules["rule_weight"].to_numpy(dtype=float)

    match_count = np.zeros(len(base), dtype=int)
    max_weight = np.zeros(len(base), dtype=float)
    partial_mean = np.zeros(len(base), dtype=float)
    partial_max = np.zeros(len(base), dtype=float)

    for idx, row in base.iterrows():
        items = _build_item_set(row)
        if not items:
            continue

        partial_scores: list[float] = []
        full_hits = 0
        max_hit_weight = 0.0
        max_partial_val = 0.0

        for ant, wt in zip(antecedents, weights):
            overlap = len(items.intersection(ant))
            if overlap == 0:
                partial_scores.append(0.0)
                continue

            frac = overlap / max(1, len(ant))
            pscore = float(frac * wt)
            partial_scores.append(pscore)
            if pscore > max_partial_val:
                max_partial_val = pscore

            if ant.issubset(items):
                full_hits += 1
                if wt > max_hit_weight:
                    max_hit_weight = float(wt)

        if partial_scores:
            partial_mean[idx] = float(np.mean(partial_scores))
            partial_max[idx] = max_partial_val
        match_count[idx] = full_hits
        max_weight[idx] = max_hit_weight

    return pd.DataFrame(
        {
            "user_id": base["user_id"],
            "l2_rule_match_count": match_count,
            "l2_rule_max_weight": max_weight,
            "l2_rule_partial_mean": partial_mean,
            "l2_rule_partial_max": partial_max,
        }
    )


def _load_l4_features(user_ids: pd.Series) -> pd.DataFrame:
    """Load L4 cluster assignments — only kmeans_cluster_id and dbscan_is_noise."""
    out = pd.DataFrame({"user_id": user_ids})

    if L4_KMEANS_PATH.exists():
        km = pd.read_csv(L4_KMEANS_PATH, usecols=["user_id", "cluster_id"])
        out = out.merge(km, on="user_id", how="left")
        out.rename(columns={"cluster_id": "kmeans_cluster_id"}, inplace=True)

    if L4_DBSCAN_PATH.exists():
        db = pd.read_csv(L4_DBSCAN_PATH, usecols=["user_id", "dbscan_cluster"])
        out = out.merge(db, on="user_id", how="left")
        out["dbscan_is_noise"] = (out["dbscan_cluster"] == -1).astype(int)
        # Drop the raw dbscan_cluster — only keep the binary noise flag
        out.drop(columns=["dbscan_cluster"], inplace=True)

    return out


def main() -> None:
    L5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load reviews_enriched.csv (review-level, ~608K rows)
    # Use python engine + on_bad_lines='skip' because the CSV has unescaped
    # commas inside the review_text column that confuse the C parser.
    reviews_path = _pick_existing(REVIEWS_ENRICHED_CANDIDATES)
    reviews = pd.read_csv(reviews_path, engine="python", on_bad_lines="skip")
    # Keep only the columns we need
    available_review_cols = [c for c in REVIEW_LEVEL_COLS if c in reviews.columns]
    reviews = reviews[available_review_cols].copy()
    print(f"Reviews source: {reviews_path}")
    print(f"  Reviews loaded: {len(reviews):,}")

    # Step 2: Load reviewer_profiles.csv (reviewer-level, ~260K rows)
    profiles_path = _pick_existing(REVIEWER_PROFILE_CANDIDATES)
    profiles = pd.read_csv(profiles_path)
    profiles = _normalize_profile_columns(profiles)
    available_agg_cols = [c for c in REVIEWER_AGG_COLS if c in profiles.columns]
    profiles = profiles[available_agg_cols].copy()
    print(f"Profiles source: {profiles_path}")
    print(f"  Profiles loaded: {len(profiles):,}")

    # Step 3: Left join reviews onto reviewer_profiles by user_id
    merged = reviews.merge(profiles, on="user_id", how="left")
    print(f"  After join: {len(merged):,} rows")

    # Step 4: Compute L2 rule features per-reviewer, then join back
    l2_rules = _load_l2_rules()
    l2_features = _compute_l2_features(profiles, l2_rules)
    merged = merged.merge(l2_features, on="user_id", how="left")

    # Step 5: Join L4 cluster assignments by user_id (kmeans_cluster_id + dbscan_is_noise only)
    l4_features = _load_l4_features(profiles["user_id"])
    merged = merged.merge(l4_features, on="user_id", how="left")

    # Step 6: Fill nulls from optional merges
    if "kmeans_cluster_id" in merged.columns:
        merged["kmeans_cluster_id"] = merged["kmeans_cluster_id"].fillna(-1)
    if "dbscan_is_noise" in merged.columns:
        merged["dbscan_is_noise"] = merged["dbscan_is_noise"].fillna(0)

    for col in ["l2_rule_match_count", "l2_rule_max_weight", "l2_rule_partial_mean", "l2_rule_partial_max"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Save
    output_path = L5_OUTPUT_DIR / "l5_feature_table.csv"
    merged.to_csv(output_path, index=False)

    # Feature columns = everything except identifiers and label
    exclude = {
        "user_id",
        "prod_id",
        "is_spam",
        "spam_label",
        "spam_rate",
        "spam_count",
        "is_spam_reviewer",
        "label",
        "first_review_date",
        "last_review_date",
    }
    feature_columns = [c for c in merged.columns if c not in exclude]
    (L5_OUTPUT_DIR / "l5_feature_columns.txt").write_text("\n".join(feature_columns), encoding="utf-8")

    positive_rate = float(merged["is_spam"].mean()) if "is_spam" in merged.columns else 0.0
    metadata = {
        "base_source": str(reviews_path.relative_to(PROJECT_ROOT)),
        "profile_source": str(profiles_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(merged)),
        "columns": int(len(merged.columns)),
        "positive_rate": positive_rate,
        "l2_rules_used": int(len(l2_rules)),
        "has_kmeans": bool(L4_KMEANS_PATH.exists()),
        "has_dbscan": bool(L4_DBSCAN_PATH.exists()),
    }
    (L5_OUTPUT_DIR / "l5_build_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nRows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Label (is_spam) positive rate: {positive_rate:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
