"""Train Layer-5 supervised classifiers on the review-level feature table.

Models:
    - Decision Tree
    - Random Forest
    - MLP (multi-layer perceptron)

Review-level training: each row is one review with weight 1.
Class imbalance (13.2% spam) handled by class_weight="balanced" (DT, RF)
and threshold tuning (MLP).

Outputs:
    - outputs/supervised_model_metrics.csv
    - outputs/supervised_holdout_predictions.csv
    - outputs/supervised_best_model.joblib
    - outputs/supervised_threshold_metadata.json
    - plots/l5_supervised_roc.png
    - plots/l5_supervised_pr.png
    - plots/l5_random_forest_feature_importance.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


def _score_vector(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def _find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Find threshold that maximizes F1 score via precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision and recall have one more element than thresholds
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


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

    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Test positive rate: {y_test.mean():.4f}")

    models = {
        "Decision Tree": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=10,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=16,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "MLP": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    metrics_rows = []
    roc_payload = []
    pr_payload = []
    pred_frame = pd.DataFrame({"user_id": g_test, "y_true": y_test})
    threshold_metadata = {}

    best_name = None
    best_auc = -1.0
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        # No sample_weight — each review has equal weight at review-level
        model.fit(X_train, y_train)

        scores = _score_vector(model, X_test)
        y_pred_default = model.predict(X_test)

        auc = roc_auc_score(y_test, scores)
        f1_default = f1_score(y_test, y_pred_default)
        ap = average_precision_score(y_test, scores)

        # Threshold tuning: find optimal threshold for each model
        opt_threshold, f1_opt = _find_optimal_threshold(y_test, scores)
        y_pred_opt = (scores >= opt_threshold).astype(int)

        metrics_rows.append({
            "model": name,
            "auc_roc": auc,
            "f1_default_0.5": f1_default,
            "f1_optimal": f1_opt,
            "optimal_threshold": opt_threshold,
            "avg_precision": ap,
        })

        threshold_metadata[name] = {
            "optimal_threshold": opt_threshold,
            "f1_at_optimal": f1_opt,
            "f1_at_default_0.5": f1_default,
        }

        fpr, tpr, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        roc_payload.append((name, fpr, tpr, auc))
        pr_payload.append((name, recall, precision, ap))

        pred_frame[f"{name}_pred"] = y_pred_opt
        pred_frame[f"{name}_score"] = scores

        print(f"AUC: {auc:.4f} | F1@0.5: {f1_default:.4f} | F1@opt({opt_threshold:.3f}): {f1_opt:.4f} | AP: {ap:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    metrics_df = pd.DataFrame(metrics_rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "supervised_model_metrics.csv", index=False)
    pred_frame.to_csv(OUTPUT_DIR / "supervised_holdout_predictions.csv", index=False)

    # Save threshold metadata
    (OUTPUT_DIR / "supervised_threshold_metadata.json").write_text(
        json.dumps(threshold_metadata, indent=2), encoding="utf-8"
    )

    if best_model is None or best_name is None:
        raise RuntimeError("No model trained successfully.")

    joblib.dump(best_model, OUTPUT_DIR / "supervised_best_model.joblib")
    (OUTPUT_DIR / "supervised_best_model_name.txt").write_text(best_name, encoding="utf-8")

    # ROC plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, fpr, tpr, auc in roc_payload:
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("L5 Supervised Models - ROC (Review-Level)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "l5_supervised_roc.png", dpi=150)
    plt.close(fig)

    # PR plot
    baseline = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, recall, precision, ap in pr_payload:
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1, label=f"Baseline={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("L5 Supervised Models - Precision/Recall (Review-Level)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "l5_supervised_pr.png", dpi=150)
    plt.close(fig)

    # Random Forest feature importance
    rf_pipe = models["Random Forest"]
    rf = rf_pipe.named_steps["clf"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", header=["importance"])

    top = importances.head(20).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance (Top 20, Review-Level)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "l5_random_forest_feature_importance.png", dpi=150)
    plt.close(fig)

    print("\nSaved supervised outputs to L5_Classification/outputs and L5_Classification/plots")


if __name__ == "__main__":
    main()
