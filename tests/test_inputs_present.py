"""Smoke test: every input the dev tasks depend on must exist on disk.

Row identity (set after inspection on 2026-04-28):
  ROW_LEVEL: review  — supervised_holdout_predictions.csv has > unique user_id,
  so each row is one review (joins downstream require user_id alone for
  reviewer-level features and (user_id, prod_id) for review-level merges).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "L5_Classification/outputs/supervised_holdout_predictions.csv",
    "L5_Classification/outputs/anomaly_holdout_scores.csv",
    "L5_Classification/outputs/l5_feature_table.csv",
    "L5_Classification/outputs/supervised_threshold_metadata.json",
    "L5_Classification/outputs/supervised_best_model_name.txt",
    "L6_Validation/outputs/synthetic_profiles.csv",
    "L6_Validation/outputs/synthetic_detection_results.csv",
    "L6_Validation/02_synthetic_injection.py",
]


def test_all_inputs_exist():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    assert not missing, f"Missing inputs: {missing}"


def test_holdout_predictions_schema():
    df = pd.read_csv(ROOT / "L5_Classification/outputs/supervised_holdout_predictions.csv")
    expected = {"user_id", "y_true", "MLP_score", "Random Forest_score", "Decision Tree_score"}
    assert expected.issubset(df.columns), f"Missing cols: {expected - set(df.columns)}"
    assert len(df) > 0


def test_feature_table_schema():
    df = pd.read_csv(ROOT / "L5_Classification/outputs/l5_feature_table.csv", nrows=5)
    must_have = {
        "user_id", "prod_id", "is_spam", "rating",
        "tenure_days", "l2_rule_max_weight", "kmeans_cluster_id", "dbscan_is_noise",
    }
    assert must_have.issubset(df.columns), f"Missing cols: {must_have - set(df.columns)}"


def test_l3_signal_present():
    """Recent integration: deberta_spam_prob now exists in the feature table."""
    df = pd.read_csv(ROOT / "L5_Classification/outputs/l5_feature_table.csv", nrows=5)
    assert "deberta_spam_prob" in df.columns, "L3 signal not wired into L5 feature table"
