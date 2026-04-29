"""Refresh L3 DeBERTa headline metrics from saved predictions, with a
proper F1-macro threshold sweep on the val split.

Updates:
    L3/outputs/deberta_metrics.json
    L3/outputs/deberta_threshold_metadata.json
    L3/plots/l3_score_distribution.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
L3_DIR = PROJECT_ROOT / "L3"
DATA_CSV = PROJECT_ROOT / "L1_ETL_OLAP" / "output_csv" / "reviews_enriched.csv"
PREDICTIONS_CSV = L3_DIR / "outputs" / "deberta_predictions.csv"
METRICS_JSON = L3_DIR / "outputs" / "deberta_metrics.json"
THRESHOLD_JSON = L3_DIR / "outputs" / "deberta_threshold_metadata.json"
SCORE_DIST_PNG = L3_DIR / "plots" / "l3_score_distribution.png"

RANDOM_SEED = 42


def group_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    df = df.reset_index(drop=True)
    labels = df["is_spam"].values
    groups = df["user_id"].values
    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(sgkf1.split(df, labels, groups))
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_idx_local, test_idx_local = next(
        sgkf2.split(temp_df, temp_df["is_spam"].values, temp_df["user_id"].values)
    )
    return (
        df.iloc[train_idx].reset_index(drop=True),
        temp_df.iloc[val_idx_local].reset_index(drop=True),
        temp_df.iloc[test_idx_local].reset_index(drop=True),
    )


def load_filtered() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_CSV,
        usecols=["user_id", "prod_id", "date", "review_text", "is_spam"],
        engine="python",
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["review_text", "is_spam"])
    df["is_spam"] = pd.to_numeric(df["is_spam"], errors="coerce")
    df = df.dropna(subset=["is_spam"]).copy()
    df["is_spam"] = df["is_spam"].astype(int)
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[
        (df["review_text"] != "")
        & (df["review_text"].str.lower() != "nan")
        & (df["review_text"].str.len() > 10)
    ].reset_index(drop=True)
    return df


def sweep_macro(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        f1 = f1_score(y, (p >= thr).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def main() -> None:
    df = load_filtered()
    _, val_df, test_df = group_split(df, RANDOM_SEED)

    preds = pd.read_csv(
        PREDICTIONS_CSV,
        usecols=["user_id", "prod_id", "date", "is_spam", "deberta_spam_prob"],
    )
    key = ["user_id", "prod_id", "date"]
    val_m = val_df[key + ["is_spam"]].merge(preds[key + ["deberta_spam_prob"]], on=key, how="inner")
    test_m = test_df[key + ["is_spam"]].merge(preds[key + ["deberta_spam_prob"]], on=key, how="inner")

    y_val, p_val = val_m["is_spam"].to_numpy(), val_m["deberta_spam_prob"].to_numpy()
    y_test, p_test = test_m["is_spam"].to_numpy(), test_m["deberta_spam_prob"].to_numpy()

    thr_macro, f1_val = sweep_macro(y_val, p_val)
    pred_def = (p_test >= 0.5).astype(int)
    pred_opt = (p_test >= thr_macro).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, p_test)),
        "avg_precision": float(average_precision_score(y_test, p_test)),
        "f1_macro_default": float(f1_score(y_test, pred_def, average="macro", zero_division=0)),
        "f1_spam_default": float(f1_score(y_test, pred_def, pos_label=1, zero_division=0)),
        "f1_macro_optimal": float(f1_score(y_test, pred_opt, average="macro", zero_division=0)),
        "f1_spam_optimal": float(f1_score(y_test, pred_opt, pos_label=1, zero_division=0)),
        "optimal_threshold": float(thr_macro),
    }
    print(f"val F1-macro at thr={thr_macro:.4f}: {f1_val:.4f}")
    for k, v in metrics.items():
        print(f"  {k:22s} {v:.4f}")

    existing = json.loads(METRICS_JSON.read_text())
    existing.update({k: round(v, 4) for k, v in metrics.items()})
    existing["threshold_objective"] = "f1_macro_on_val"
    METRICS_JSON.write_text(json.dumps(existing, indent=2))

    THRESHOLD_JSON.write_text(json.dumps(
        {
            "model": existing.get("model", "microsoft/deberta-base"),
            "optimal_threshold": round(metrics["optimal_threshold"], 4),
            "f1_macro_at_optimal": round(metrics["f1_macro_optimal"], 4),
            "f1_spam_at_optimal": round(metrics["f1_spam_optimal"], 4),
            "score_column": "deberta_spam_prob",
            "threshold_objective": "f1_macro_on_val",
        },
        indent=2,
    ))

    SCORE_DIST_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_test[y_test == 0], bins=60, alpha=0.6, label="genuine (test)", density=True)
    ax.hist(p_test[y_test == 1], bins=60, alpha=0.6, label="spam (test)", density=True)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="default thr = 0.5")
    ax.axvline(
        metrics["optimal_threshold"], color="red", linestyle="--", linewidth=1.2,
        label=f"F1-macro thr = {metrics['optimal_threshold']:.3f}",
    )
    ax.set_xlabel("DeBERTa spam probability")
    ax.set_ylabel("Density")
    ax.set_title("L3 DeBERTa score distribution (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCORE_DIST_PNG, dpi=300)
    plt.close(fig)
    print(f"\nWrote {METRICS_JSON.name}, {THRESHOLD_JSON.name}, {SCORE_DIST_PNG.name}")


if __name__ == "__main__":
    main()
