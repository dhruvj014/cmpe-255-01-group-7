"""Shared helpers for Layer-6 validation scripts.

Centralises input paths, L2 discretisation logic (mirrors L2_FPGrowth/01),
L4 preprocessing logic (mirrors L4_Clustering/01) and small loaders so the
three L6 scripts stay focused on their own analysis.

Updated for review-level L5 outputs: predictions now have multiple rows per
user_id and must be aggregated to reviewer-level for Jaccard comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

L1_PROFILES = PROJECT_ROOT / "L1_ETL_OLAP" / "output_csv" / "reviewer_profiles.csv"
L1_PROFILES_FALLBACK = PROJECT_ROOT / "reviewer_features.csv"

L2_RULES = PROJECT_ROOT / "L2_FPGrowth" / "outputs" / "spam_correlated_rules.csv"

L4_KMEANS = PROJECT_ROOT / "L4_Clustering" / "outputs" / "reviewer_clusters.csv"
L4_DBSCAN = PROJECT_ROOT / "L4_Clustering" / "outputs" / "dbscan_results.csv"
L4_KM_SUMMARY = PROJECT_ROOT / "L4_Clustering" / "outputs" / "kmeans_cluster_summary.csv"
L4_DB_SUMMARY = PROJECT_ROOT / "L4_Clustering" / "outputs" / "dbscan_cluster_summary.csv"
L4_RAW = PROJECT_ROOT / "L4_Clustering" / "outputs" / "clustering_features_raw.csv"

L5_FEATURE_TABLE = PROJECT_ROOT / "L5_Classification" / "outputs" / "l5_feature_table.csv"
L5_MODEL = PROJECT_ROOT / "L5_Classification" / "outputs" / "supervised_best_model.joblib"
L5_MODEL_NAME = PROJECT_ROOT / "L5_Classification" / "outputs" / "supervised_best_model_name.txt"
L5_SUP_PRED = PROJECT_ROOT / "L5_Classification" / "outputs" / "supervised_holdout_predictions.csv"
L5_ANOM_SCORES = PROJECT_ROOT / "L5_Classification" / "outputs" / "anomaly_holdout_scores.csv"
L5_FEATURE_COLS = PROJECT_ROOT / "L5_Classification" / "outputs" / "l5_feature_columns.txt"

L6_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = L6_DIR / "outputs"
PLOT_DIR = L6_DIR / "plots"

# ---------------------------------------------------------------------------
# L4 / L5 feature definitions (must stay in sync with the layer scripts)
# ---------------------------------------------------------------------------
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

LOG_TRANSFORM_FEATURES = [
    "review_count",
    "reviews_per_week",
    "tenure_days",
    "burst_score",
    "unique_sellers",
]

# Review-level raw feature columns (matches l5_feature_table.csv columns
# minus leakage/label/date/id columns).  Split into review-own and
# reviewer-aggregate for clarity.
L5_REVIEW_OWN_FEATURES = [
    "rating",
    "review_length",
    "word_count",
    "exclamation_count",
    "question_count",
    "capital_ratio",
    "avg_word_length",
    "day_of_week",
    "month",
]

L5_REVIEWER_AGG_FEATURES = [
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

L5_L2_FEATURES = [
    "l2_rule_match_count",
    "l2_rule_max_weight",
    "l2_rule_partial_mean",
    "l2_rule_partial_max",
]

L5_L4_FEATURES = [
    "kmeans_cluster_id",
    "dbscan_is_noise",
]

# Combined raw features list (before one-hot encoding)
L5_RAW_FEATURES = (
    L5_REVIEW_OWN_FEATURES
    + L5_REVIEWER_AGG_FEATURES
    + L5_L2_FEATURES
    + L5_L4_FEATURES
)


def prepare_for_l5_model(df: pd.DataFrame, model) -> pd.DataFrame:
    """One-hot encode cluster IDs and align columns with the saved L5 model.

    Replicates the _prepare_features transformation from L5/02_train_models.py
    then aligns columns with what the model was trained on (adds missing
    one-hot columns as 0, drops extras).
    """
    X = df.copy()

    # One-hot encode kmeans_cluster_id only (no dbscan_cluster column)
    cluster_cols = [c for c in ("kmeans_cluster_id",) if c in X.columns]
    if cluster_cols:
        for c in cluster_cols:
            X[c] = X[c].astype(int).astype(str)
        X = pd.get_dummies(X, columns=cluster_cols, drop_first=False)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # Align with model's expected features
    if hasattr(model, "feature_names_in_"):
        expected = model.feature_names_in_
    else:
        clf = model.named_steps.get("clf", model)
        expected = clf.feature_names_in_

    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[expected]

    return X


# ---------------------------------------------------------------------------
# L2 discretisation (mirrors L2_FPGrowth/01_basket_encoding.py)
# ---------------------------------------------------------------------------

def rc_bin(value: float) -> str:
    if value <= 2:
        return "review_count=Low"
    if value <= 10:
        return "review_count=Medium"
    return "review_count=High"


def tenure_bin(value: float) -> str:
    if value < 30:
        return "tenure=new"
    if value <= 180:
        return "tenure=moderate"
    if value <= 365:
        return "tenure=established"
    return "tenure=veteran"


def seller_bin(value: float) -> str:
    return "seller_conc=High" if value >= 0.5 else "seller_conc=Low"


def burst_bin(value: float) -> str:
    return "burst=Bursty" if value > 2 else "burst=Normal"


def build_item_set(row) -> set[str]:
    """Replicate L5._build_item_set for a single reviewer dict/row."""
    items: set[str] = set()
    get = row.get if hasattr(row, "get") else (lambda k, default=None: row[k] if k in row else default)

    rc = get("review_count")
    td = get("tenure_days")
    sc = get("max_seller_fraction")
    bs = get("burst_score")

    if pd.notna(rc):
        items.add(rc_bin(float(rc)))
    if pd.notna(td):
        items.add(tenure_bin(float(td)))
    if pd.notna(sc):
        items.add(seller_bin(float(sc)))
    if pd.notna(bs):
        items.add(burst_bin(float(bs)))
    return items


def parse_frozenset(raw: str) -> frozenset[str]:
    """Parse the stringified frozenset stored in L2 CSVs."""
    try:
        return eval(raw, {"__builtins__": {}}, {"frozenset": frozenset})
    except Exception:
        return frozenset()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_reviewer_profiles() -> pd.DataFrame:
    """Load the L1 reviewer profile table, trying both canonical locations."""
    for path in (L1_PROFILES, L1_PROFILES_FALLBACK):
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        "reviewer_profiles.csv not found. Run L1_ETL_OLAP/main.py first."
    )


def load_holdout_user_ids() -> set:
    """Return the deduplicated set of user_ids from the L5 supervised holdout.

    L5 now outputs review-level predictions, so multiple rows per user_id.
    Deduplicate to get the unique reviewer set.
    """
    return set(pd.read_csv(L5_SUP_PRED, usecols=["user_id"])["user_id"].unique())


def load_best_supervised_name() -> str:
    if L5_MODEL_NAME.exists():
        return L5_MODEL_NAME.read_text(encoding="utf-8").strip()
    return "MLP"


def load_l2_rules() -> pd.DataFrame:
    """Load spam-correlated rules with antecedents parsed into frozensets."""
    rules = pd.read_csv(L2_RULES)
    rules["antecedents"] = rules["antecedents"].astype(str).apply(parse_frozenset)
    rules = rules[rules["antecedents"].apply(len) > 0].reset_index(drop=True)
    return rules


def log_transform(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Apply log1p to selected columns (matches L4/01_preprocessing)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = np.log1p(out[c].astype(float))
    return out


def fit_clustering_scaler() -> tuple[StandardScaler, pd.DataFrame]:
    """Refit the L4 StandardScaler on clustering_features_raw.csv.

    L4's clustering_features_raw.csv already stores the log1p-transformed
    values; fitting StandardScaler on it reproduces the exact scaler used
    by L4_Clustering/01_preprocessing.py.

    Returns:
        (scaler, scaled_df) where scaled_df is indexed by user_id.
    """
    raw = pd.read_csv(L4_RAW)
    feat = raw[CLUSTERING_FEATURES].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat)
    scaled_df = pd.DataFrame(scaled, columns=CLUSTERING_FEATURES)
    scaled_df.insert(0, "user_id", raw["user_id"].values)
    return scaler, scaled_df


def transform_new_profiles(scaler: StandardScaler, profiles: pd.DataFrame) -> np.ndarray:
    """Log1p + StandardScaler-transform a DataFrame of raw behavioural features."""
    feat = profiles[CLUSTERING_FEATURES].copy()
    for c in LOG_TRANSFORM_FEATURES:
        feat[c] = np.log1p(feat[c].astype(float))
    return scaler.transform(feat)


# ---------------------------------------------------------------------------
# Review-level → reviewer-level aggregation for L5 scores
# ---------------------------------------------------------------------------

def aggregate_review_scores_to_reviewer(
    df: pd.DataFrame,
    score_col: str,
    agg: str = "max",
) -> pd.DataFrame:
    """Aggregate review-level scores to one score per reviewer (user_id).

    Args:
        df: DataFrame with user_id and score_col columns
        score_col: column name containing the score to aggregate
        agg: aggregation function ("max", "mean", etc.)

    Returns:
        DataFrame with columns [user_id, {score_col}] — one row per reviewer.
    """
    return df.groupby("user_id", as_index=False)[score_col].agg(agg)
