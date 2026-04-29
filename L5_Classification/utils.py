"""Shared L5 helpers: deterministic train/test split, feature prep, imputation.

Extracted from 02_train_models.py and 03_anomaly_detection.py to keep the
two scripts using identical splits and feature transformations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def get_group_stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """StratifiedGroupKFold with the fold whose test ratio is closest to test_size."""
    n_splits = max(2, int(round(1 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_split = None
    best_gap = float("inf")
    for tr_idx, te_idx in sgkf.split(X, y, groups):
        gap = abs(len(te_idx) / len(y) - test_size)
        if gap < best_gap:
            best_gap = gap
            best_split = (tr_idx, te_idx)

    if best_split is None:
        return train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    return best_split


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare review-level features. Label = is_spam, groups = user_id.

    Drops leakage/id/date columns, one-hot encodes kmeans_cluster_id, coerces
    everything else to numeric. Boolean columns become int.
    """
    if "is_spam" not in df.columns:
        raise KeyError("Input feature table must contain is_spam")
    if "user_id" not in df.columns:
        raise KeyError("Input feature table must contain user_id")

    leakage_cols = {
        "user_id", "prod_id", "is_spam",
        "spam_label", "spam_rate", "spam_count", "is_spam_reviewer", "label",
    }
    drop_cols = {"first_review_date", "last_review_date", "review_date", "date", "year_month"}
    exclude = leakage_cols | drop_cols

    X = df[[c for c in df.columns if c not in exclude]].copy()
    y = df["is_spam"].astype(int).to_numpy()
    groups = df["user_id"].to_numpy()

    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    cluster_cols = [c for c in ("kmeans_cluster_id",) if c in X.columns]
    if cluster_cols:
        for c in cluster_cols:
            X[c] = X[c].astype(int).astype(str)
        X = pd.get_dummies(X, columns=cluster_cols, drop_first=False)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y, groups


def impute_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaNs using training-set median only (no test-set leakage)."""
    train_median = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_median).fillna(0)
    X_test = X_test.fillna(train_median).fillna(0)
    return X_train, X_test
