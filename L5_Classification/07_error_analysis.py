"""L5 error analysis on the best supervised model.

Buckets MLP holdout predictions by tenure x rating x outcome (TP/FP/FN/TN).
Surfaces the features most different in FP-vs-TP and FN-vs-TN populations.
Outputs:
    L5_Classification/outputs/error_buckets.csv
    L5_Classification/outputs/feature_delta_fp.csv
    L5_Classification/outputs/feature_delta_fn.csv
    L5_Classification/plots/error_heatmap.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "L5_Classification/outputs"
PLOTS = ROOT / "L5_Classification/plots"
PLOTS.mkdir(exist_ok=True)

best_model = (OUT / "supervised_best_model_name.txt").read_text().strip()
thr_meta = json.loads((OUT / "supervised_threshold_metadata.json").read_text())
thr_entry = thr_meta.get(best_model, {})
thr = float(thr_entry.get("optimal_threshold", 0.5))

print(f"Best model: {best_model}  threshold: {thr:.4f}")

sup = pd.read_csv(OUT / "supervised_holdout_predictions.csv")
score_col = f"{best_model}_score"
if score_col not in sup.columns:
    raise KeyError(f"{score_col} missing from holdout predictions; have: {list(sup.columns)}")

sup["pred"] = (sup[score_col] >= thr).astype(int)


def _outcome(row):
    if row["y_true"] == 1 and row["pred"] == 1:
        return "TP"
    if row["y_true"] == 0 and row["pred"] == 1:
        return "FP"
    if row["y_true"] == 1 and row["pred"] == 0:
        return "FN"
    return "TN"


sup["outcome"] = sup.apply(_outcome, axis=1)

# Feature table is review-level (user_id + prod_id keys). Holdout is row-aligned
# with a deterministic split on the same row order, so we can pull features by
# row index from a re-built feature table aligned with sup row positions.
# Simpler approach: deduplicate features per user_id and merge — accepting that
# a user with multiple reviews gets a single representative feature row.
feat = pd.read_csv(OUT / "l5_feature_table.csv").drop_duplicates("user_id")
m = sup.merge(feat, on="user_id", how="left", suffixes=("", "_feat"))


def tenure_bucket(d):
    if pd.isna(d):
        return "unknown"
    if d < 30:
        return "new"
    if d < 365:
        return "moderate"
    if d < 1000:
        return "established"
    return "veteran"


m["tenure_bucket"] = m["tenure_days"].apply(tenure_bucket)

# Rating column collision: feature table has 'rating' (review-level) so prefer
# the merged column. If suffix conflict exists, fall back to rating_feat.
rating_col = "rating" if "rating" in m.columns else "rating_feat"

buckets = m.groupby(["tenure_bucket", rating_col, "outcome"]).size().reset_index(name="count")
buckets = buckets.rename(columns={rating_col: "rating"})
buckets.to_csv(OUT / "error_buckets.csv", index=False)

err = m.assign(is_err=m["outcome"].isin(["FP", "FN"]).astype(int))
piv = err.groupby(["tenure_bucket", rating_col])["is_err"].mean().unstack(fill_value=0)
plt.figure(figsize=(6, 4))
sns.heatmap(piv, annot=True, fmt=".2f", cmap="Reds")
plt.title(f"L5 ({best_model}) Error Rate by Tenure x Rating")
plt.tight_layout()
plt.savefig(PLOTS / "error_heatmap.png", dpi=300)
plt.close()

NUMERIC = [
    "review_length", "word_count", "exclamation_count", "capital_ratio",
    "avg_word_length", "review_count", "avg_rating", "rating_std",
    "tenure_days", "reviews_per_week", "max_seller_fraction",
    "burst_score", "rating_entropy",
    "l2_rule_max_weight", "l2_rule_partial_mean",
    "deberta_spam_prob",
]
NUMERIC = [c for c in NUMERIC if c in m.columns]


def deltas(pos_label, neg_label):
    pos = m[m["outcome"] == pos_label][NUMERIC].mean()
    neg = m[m["outcome"] == neg_label][NUMERIC].mean()
    d = (pos - neg).rename("delta").to_frame()
    d["abs_delta"] = d["delta"].abs()
    d.index.name = "feature"
    return d.sort_values("abs_delta", ascending=False).reset_index().head(10)


deltas("FP", "TP").to_csv(OUT / "feature_delta_fp.csv", index=False)
deltas("FN", "TN").to_csv(OUT / "feature_delta_fn.csv", index=False)

# Print summary
print("\nOutcome counts:")
print(m["outcome"].value_counts())
print("\nTop FP-vs-TP deltas:")
print(pd.read_csv(OUT / "feature_delta_fp.csv"))
print("Error analysis complete.")
