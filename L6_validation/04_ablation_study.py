"""
Ablation study across layer subsets on the L5 holdout (no model retraining).

For "behavioral" rows we use saved scores from the pre-L3 L5 retrain
(L5_Classification/outputs/behavioral_baseline_scores.csv).
For "+L3" rows we use the current L5 holdout files, which were produced
by L5 models retrained after deberta_spam_prob was wired into the feature
table (so MLP_score / isolation_forest_score now incorporate the L3 signal).

Pairwise / stacker rows fit logistic regression on a 50/50 holdout split.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
L5_OUT = ROOT / "L5_Classification/outputs"
L6_OUT = ROOT / "L6_Validation/outputs"
L6_PLOTS = ROOT / "L6_Validation/plots"
L6_OUT.mkdir(parents=True, exist_ok=True)
L6_PLOTS.mkdir(parents=True, exist_ok=True)

RNG = 42


def best_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f[:-1])) if len(t) else 0
    return float(f[i]), float(t[i] if len(t) else 0.5)


def recall_at_fpr(y, s, target_fpr=0.1):
    fpr, tpr, _ = roc_curve(y, s)
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def evaluate(y, s):
    f1, _ = best_f1(y, s)
    return {
        "auc_roc": float(roc_auc_score(y, s)),
        "f1_optimal": f1,
        "avg_precision": float(average_precision_score(y, s)),
        "recall_at_fpr_0_1": recall_at_fpr(y, s),
    }


def stacker_eval(X, y):
    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X, y, test_size=0.5, random_state=RNG, stratify=y
    )
    sc = StandardScaler().fit(X_fit)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG)
    clf.fit(sc.transform(X_fit), y_fit)
    s = clf.predict_proba(sc.transform(X_eval))[:, 1]
    return evaluate(y_eval, s)


# ---- Load all rows aligned by index ----
sup = pd.read_csv(L5_OUT / "supervised_holdout_predictions.csv")
ano = pd.read_csv(L5_OUT / "anomaly_holdout_scores.csv")
beh = pd.read_csv(L5_OUT / "behavioral_baseline_scores.csv")

assert (sup["user_id"].values == ano["user_id"].values).all()
assert (sup["user_id"].values == beh["user_id"].values).all()

feat = pd.read_csv(
    L5_OUT / "l5_feature_table.csv",
    usecols=["user_id", "l2_rule_max_weight", "l2_rule_partial_mean",
             "kmeans_cluster_id", "dbscan_is_noise"],
).drop_duplicates("user_id")

df = sup[["user_id", "y_true", "MLP_score"]].copy()
df["isolation_forest_score"] = ano["isolation_forest_score"].values
df["behavioral_mlp_score"] = beh["behavioral_mlp_score"].values
df["behavioral_if_score"] = beh["behavioral_if_score"].values
df = df.merge(feat, on="user_id", how="left")

y = df["y_true"].astype(int).values

# ---- Single-layer rows (behavioral baselines where applicable) ----
results = []
results.append(("L2-only", evaluate(y, df["l2_rule_max_weight"].values)))

cluster_rate = df.groupby("kmeans_cluster_id")["y_true"].transform("mean").values
results.append(("L4-only", evaluate(y, cluster_rate)))

# L5-supervised / L5-anomaly here = behavioral-only versions (no L3)
results.append(("L5-supervised-only", evaluate(y, df["behavioral_mlp_score"].values)))
results.append(("L5-anomaly-only", evaluate(y, df["behavioral_if_score"].values)))

# ---- Stacker rows ----
X_l2_l4 = pd.concat([
    df[["l2_rule_max_weight", "l2_rule_partial_mean", "dbscan_is_noise"]].astype(float),
    pd.get_dummies(df["kmeans_cluster_id"], prefix="km").astype(float),
], axis=1)
results.append(("L2+L4", stacker_eval(X_l2_l4, y)))

# Full behavioral = L2 + L4 + L5(no-L3)
X_full_beh = pd.concat([
    df[["behavioral_mlp_score", "behavioral_if_score",
        "l2_rule_max_weight", "l2_rule_partial_mean", "dbscan_is_noise"]].astype(float),
    pd.get_dummies(df["kmeans_cluster_id"], prefix="km").astype(float),
], axis=1)
results.append(("L2+L4+L5 (full behavioral)", stacker_eval(X_full_beh, y)))

# Full + L3 = L2 + L4 + L5(with L3)
X_full_l3 = pd.concat([
    df[["MLP_score", "isolation_forest_score",
        "l2_rule_max_weight", "l2_rule_partial_mean", "dbscan_is_noise"]].astype(float),
    pd.get_dummies(df["kmeans_cluster_id"], prefix="km").astype(float),
], axis=1)
results.append(("Full + L3", stacker_eval(X_full_l3, y)))

# ---- Build dataframe ----
rows = [{"config": name, **m} for name, m in results]
table = pd.DataFrame(rows)
table.to_csv(L6_OUT / "ablation_table.csv", index=False)

# ---- LaTeX ----
def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


lines = [
    r"\begin{table}[!t]", r"\centering",
    r"\caption{Ablation study across layer configurations on the L5 holdout.}",
    r"\label{tab:ablation}",
    r"\begin{tabular}{lcccc}", r"\toprule",
    r"Configuration & AUC-ROC & F1@optimal & Avg Precision & Recall@FPR=0.1 \\",
    r"\midrule",
]
for r in rows:
    lines.append(
        f"{r['config']} & {fmt(r['auc_roc'])} & {fmt(r['f1_optimal'])} & "
        f"{fmt(r['avg_precision'])} & {fmt(r['recall_at_fpr_0_1'])} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(L6_OUT / "ablation_table.tex").write_text("\n".join(lines))

# ---- Bar plot ----
plt.figure(figsize=(8, 4.2))
plt.barh([r["config"] for r in rows], [r["auc_roc"] for r in rows])
plt.axvline(0.5, color="k", ls="--", alpha=0.4)
plt.xlabel("AUC-ROC")
plt.title("Ablation: AUC by layer configuration")
plt.tight_layout()
plt.savefig(L6_PLOTS / "ablation_bar.png", dpi=300)
plt.close()

print("Ablation rows:")
for r in rows:
    print(f"  {r['config']:<32}  AUC={r['auc_roc']:.3f}  F1@opt={r['f1_optimal']:.3f}  AP={r['avg_precision']:.3f}")
