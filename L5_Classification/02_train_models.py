"""Train Layer-5 supervised classifiers on the fused feature table.

Models:
    - Decision Tree
    - Random Forest
    - MLP (multi-layer perceptron)

Sample weighting: each reviewer is weighted by review_count so that
high-activity accounts (more signal) contribute more to the loss.

Outputs:
    - outputs/supervised_model_metrics.csv
    - outputs/supervised_holdout_predictions.csv
    - outputs/supervised_best_model.joblib
    - plots/l5_supervised_roc.png
    - plots/l5_supervised_pr.png
    - plots/l5_random_forest_feature_importance.png
"""

from __future__ import annotations

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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L5_DIR = PROJECT_ROOT / "L5_Classification"
OUTPUT_DIR = L5_DIR / "outputs"
PLOT_DIR = L5_DIR / "plots"
INPUT_PATH = OUTPUT_DIR / "l5_feature_table.csv"

RANDOM_STATE = 42


def get_group_stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    n_splits = max(2, int(round(1 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_split = None
    best_gap = float("inf")

    for tr_idx, te_idx in sgkf.split(X, y, groups):
        current_ratio = len(te_idx) / len(y)
        gap = abs(current_ratio - test_size)
        if gap < best_gap:
            best_gap = gap
            best_split = (tr_idx, te_idx)

    if best_split is None:
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        return tr_idx, te_idx

    return best_split


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if "spam_label" not in df.columns:
        raise KeyError("Input feature table must contain spam_label")
    if "user_id" not in df.columns:
        raise KeyError("Input feature table must contain user_id")

    leakage_cols = {
        "user_id",
        "spam_label",
        "spam_rate",
        "spam_count",
        "is_spam",
        "is_spam_reviewer",
        "label",
    }
    # Non-numeric columns that should not be used as features.
    drop_cols = {"first_review_date", "last_review_date"}

    exclude = leakage_cols | drop_cols
    X = df[[c for c in df.columns if c not in exclude]].copy()
    y = df["spam_label"].astype(int).to_numpy()
    groups = df["user_id"].to_numpy()

    # Keep model input strictly numeric.
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    # Safety: if a column is entirely NaN (no valid median), fill with 0.
    X = X.fillna(0)

    return X, y, groups


def _score_vector(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH}. Run 01_build_feature_table.py first.")

    df = pd.read_csv(INPUT_PATH)
    X, y, groups = _prepare_features(df)

    # Sample weights: reviewers with more reviews carry more signal.
    if "review_count" in df.columns:
        sample_weights = df["review_count"].to_numpy().astype(float)
    else:
        sample_weights = np.ones(len(df))

    train_idx, test_idx = get_group_stratified_split(X, y, groups, test_size=0.20, random_state=RANDOM_STATE)
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    g_test = groups[test_idx]
    w_train = sample_weights[train_idx]

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

    # Models whose underlying estimator accepts sample_weight.
    # MLP does not support sample_weight natively.
    SUPPORTS_WEIGHT = {"Decision Tree", "Random Forest"}

    metrics_rows = []
    roc_payload = []
    pr_payload = []
    pred_frame = pd.DataFrame({"user_id": g_test, "y_true": y_test})

    best_name = None
    best_auc = -1.0
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name in SUPPORTS_WEIGHT:
            model.fit(X_train, y_train, clf__sample_weight=w_train)
        else:
            model.fit(X_train, y_train)

        scores = _score_vector(model, X_test)
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, scores)
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, scores)

        metrics_rows.append({"model": name, "auc_roc": auc, "f1": f1, "avg_precision": ap})

        fpr, tpr, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        roc_payload.append((name, fpr, tpr, auc))
        pr_payload.append((name, recall, precision, ap))

        pred_frame[f"{name}_pred"] = y_pred
        pred_frame[f"{name}_score"] = scores

        print(f"AUC: {auc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    metrics_df = pd.DataFrame(metrics_rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "supervised_model_metrics.csv", index=False)
    pred_frame.to_csv(OUTPUT_DIR / "supervised_holdout_predictions.csv", index=False)

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
    ax.set_title("L5 Supervised Models - ROC")
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
    ax.set_title("L5 Supervised Models - Precision/Recall")
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
    ax.set_title("Random Forest Feature Importance (Top 20)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "l5_random_forest_feature_importance.png", dpi=150)
    plt.close(fig)

    print("\nSaved supervised outputs to L5_Classification/outputs and L5_Classification/plots")


if __name__ == "__main__":
    main()
