"""Stacked L5 ensemble: combines L2 rule scores + L4 cluster signals +
L5-supervised (MLP) + L5-anomaly (Isolation Forest) via logistic regression.

Note: the L5 supervised + anomaly models were retrained after L3 was wired
into the feature table, so the MLP_score and isolation_forest_score columns
already incorporate the deberta_spam_prob signal. This ensemble therefore
represents the *full* multi-layer stack, not a behavioral-only baseline.
The pre-L3 contrast is handled in 04_ablation_study.py.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "L5_Classification/outputs"
PLOTS = ROOT / "L5_Classification/plots"
PLOTS.mkdir(exist_ok=True)

RNG = 42

sup = pd.read_csv(OUT / "supervised_holdout_predictions.csv")
ano = pd.read_csv(OUT / "anomaly_holdout_scores.csv")
feat = pd.read_csv(
    OUT / "l5_feature_table.csv",
    usecols=["user_id", "l2_rule_max_weight", "l2_rule_partial_mean",
             "kmeans_cluster_id", "dbscan_is_noise"],
).drop_duplicates("user_id")

assert (sup["user_id"].values == ano["user_id"].values).all(), \
    "supervised and anomaly holdout files are not row-aligned"
df = sup[["user_id", "y_true", "MLP_score"]].copy()
df["isolation_forest_score"] = ano["isolation_forest_score"].values
df = df.merge(feat, on="user_id", how="left")

if_min, if_max = df["isolation_forest_score"].min(), df["isolation_forest_score"].max()
df["if_norm"] = (df["isolation_forest_score"] - if_min) / (if_max - if_min + 1e-9)
kmeans_oh = pd.get_dummies(df["kmeans_cluster_id"], prefix="km").astype(float)
X = pd.concat([
    df[["MLP_score", "if_norm", "l2_rule_max_weight", "l2_rule_partial_mean", "dbscan_is_noise"]].astype(float),
    kmeans_oh,
], axis=1)
y = df["y_true"].astype(int).values

X_fit, X_eval, y_fit, y_eval, idx_fit, idx_eval = train_test_split(
    X, y, np.arange(len(df)), test_size=0.5, random_state=RNG, stratify=y
)
scaler = StandardScaler().fit(X_fit)
clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG)
clf.fit(scaler.transform(X_fit), y_fit)
scores = clf.predict_proba(scaler.transform(X_eval))[:, 1]

prec, rec, thr = precision_recall_curve(y_eval, scores)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
opt_thr = float(thr[best]) if len(thr) else 0.5
preds = (scores >= opt_thr).astype(int)

out_df = pd.DataFrame({
    "user_id": df.iloc[idx_eval]["user_id"].values,
    "y_true": y_eval,
    "score": scores,
    "pred_at_optimal_threshold": preds,
})
out_df.to_csv(OUT / "ensemble_predictions.csv", index=False)

metrics = {
    "auc_roc": float(roc_auc_score(y_eval, scores)),
    "avg_precision": float(average_precision_score(y_eval, scores)),
    "f1_at_05": float(f1_score(y_eval, (scores >= 0.5).astype(int))),
    "f1_at_optimal": float(f1_score(y_eval, preds)),
    "optimal_threshold": opt_thr,
    "brier": float(brier_score_loss(y_eval, scores)),
    "n_eval": int(len(y_eval)),
    "positive_rate_eval": float(y_eval.mean()),
}
(OUT / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2))

fpr, tpr, _ = roc_curve(y_eval, scores)
plt.figure()
plt.plot(fpr, tpr, label=f"Ensemble (AUC={metrics['auc_roc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("L5 Stacked Ensemble — ROC")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "ensemble_roc.png", dpi=300)
plt.close()

plt.figure()
plt.plot(rec, prec, label=f"Ensemble (AP={metrics['avg_precision']:.3f})")
plt.axhline(y_eval.mean(), color="k", ls="--", alpha=0.4, label=f"Baseline={y_eval.mean():.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("L5 Stacked Ensemble — Precision-Recall")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "ensemble_pr.png", dpi=300)
plt.close()

print(f"Ensemble AUC={metrics['auc_roc']:.3f}  F1@opt={metrics['f1_at_optimal']:.3f}  thr={opt_thr:.3f}")
