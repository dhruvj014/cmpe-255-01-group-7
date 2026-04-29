"""Train Layer-5 anomaly detectors on the review-level feature table.

Algorithms:
    - Isolation Forest
    - Local Outlier Factor (novelty mode)

Trained on legitimate reviews only (is_spam=0) from the training split.
With per-review features (review_length, rating, temporal), spam reviews
should be more distinguishable than reviewer-level aggregates.

Outputs:
    - outputs/anomaly_model_metrics.csv
    - outputs/anomaly_holdout_scores.csv
    - plots/l5_anomaly_roc.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_group_stratified_split, impute_split, prepare_features  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L5_DIR = PROJECT_ROOT / "L5_Classification"
OUTPUT_DIR = L5_DIR / "outputs"
PLOT_DIR = L5_DIR / "plots"
INPUT_PATH = OUTPUT_DIR / "l5_feature_table.csv"

RANDOM_STATE = 42


_prepare_features = prepare_features
_impute_split = impute_split


def _evaluate(y_true: np.ndarray, scores: np.ndarray, pred_binary: np.ndarray, name: str) -> dict:
    return {
        "model": name,
        "auc_roc": roc_auc_score(y_true, scores),
        "f1": f1_score(y_true, pred_binary),
        "avg_precision": average_precision_score(y_true, scores),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH}. Run 01_build_feature_table.py first.")

    df = pd.read_csv(INPUT_PATH)
    X, y, groups = _prepare_features(df)

    train_idx, test_idx = get_group_stratified_split(X, y, groups, test_size=0.20, random_state=RANDOM_STATE)
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    X_train, X_test = _impute_split(X_train, X_test)
    y_train, y_test = y[train_idx], y[test_idx]
    g_test = groups[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train on legitimate reviews only (is_spam=0)
    legit_mask = y_train == 0
    X_train_legit = X_train_scaled[legit_mask]

    contamination = float(np.clip(np.round(y_train.mean(), 3), 0.01, 0.40))
    print(f"Contamination set to {contamination:.3f}")
    print(f"Training rows (legit only): {X_train_legit.shape[0]:,}")
    print(f"Test rows: {len(X_test):,}")

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train_legit)
    iso_raw = iso.predict(X_test_scaled)
    iso_pred = np.where(iso_raw == -1, 1, 0)
    iso_scores = -iso.score_samples(X_test_scaled)

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(X_train_legit)
    lof_raw = lof.predict(X_test_scaled)
    lof_pred = np.where(lof_raw == -1, 1, 0)
    lof_scores = -lof.score_samples(X_test_scaled)

    metrics = pd.DataFrame(
        [
            _evaluate(y_test, iso_scores, iso_pred, "Isolation Forest"),
            _evaluate(y_test, lof_scores, lof_pred, "LOF"),
        ]
    ).sort_values("auc_roc", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "anomaly_model_metrics.csv", index=False)

    scores_df = pd.DataFrame(
        {
            "user_id": g_test,
            "y_true": y_test,
            "isolation_forest_score": iso_scores,
            "isolation_forest_pred": iso_pred,
            "lof_score": lof_scores,
            "lof_pred": lof_pred,
        }
    )
    scores_df.to_csv(OUTPUT_DIR / "anomaly_holdout_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in [
        ("Isolation Forest", iso_scores),
        ("LOF", lof_scores),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("L5 Anomaly Models - ROC (Review-Level)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "l5_anomaly_roc.png", dpi=150)
    plt.close(fig)

    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print("\nSaved anomaly outputs to L5_Classification/outputs and L5_Classification/plots")


if __name__ == "__main__":
    main()
