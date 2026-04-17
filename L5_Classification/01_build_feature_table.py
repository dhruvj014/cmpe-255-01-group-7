"""Build Layer-5 feature table by fusing L1, L2, and L4 outputs.

Inputs (best-effort auto-detected):
    - reviewer_features.csv or L1 reviewer_profiles.csv
    - L2_FPGrowth/outputs/spam_correlated_rules.csv
    - L4_Clustering/outputs/reviewer_clusters.csv
    - L4_Clustering/outputs/dbscan_results.csv

Outputs:
    - L5_Classification/outputs/l5_feature_table.csv
    - L5_Classification/outputs/l5_feature_columns.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
L5_OUTPUT_DIR = PROJECT_ROOT / "L5_Classification" / "outputs"

BASE_FEATURE_CANDIDATES = [
    PROJECT_ROOT / "L1_ETL_OLAP" / "output_csv" / "reviewer_profiles.csv",
    PROJECT_ROOT / "reviewer_profiles.csv",
    PROJECT_ROOT / "reviewer_features.csv",
]

L2_RULES_PATH = PROJECT_ROOT / "L2_FPGrowth" / "outputs" / "spam_correlated_rules.csv"
L4_KMEANS_PATH = PROJECT_ROOT / "L4_Clustering" / "outputs" / "reviewer_clusters.csv"
L4_DBSCAN_PATH = PROJECT_ROOT / "L4_Clustering" / "outputs" / "dbscan_results.csv"
L4_KMEANS_SUMMARY_PATH = PROJECT_ROOT / "L4_Clustering" / "outputs" / "kmeans_cluster_summary.csv"
L4_DBSCAN_SUMMARY_PATH = PROJECT_ROOT / "L4_Clustering" / "outputs" / "dbscan_cluster_summary.csv"

TOP_L2_RULES = 50


def _pick_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in paths]}")


def _normalize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "total_reviews": "review_count",
        "mean_rating": "avg_rating",
        "mean_review_length": "avg_review_length",
        "mean_caps_ratio": "capital_ratio",
        "mean_exclamations": "avg_exclamations",
        "concentration": "max_seller_fraction",
    }
    # Rename (not copy) so we don't end up with duplicate feature pairs.
    applicable = {old: new for old, new in rename_map.items()
                  if old in df.columns and new not in df.columns}
    df = df.rename(columns=applicable)

    if "user_id" not in df.columns:
        raise KeyError("Base feature table must include user_id")
    if "spam_rate" not in df.columns:
        raise KeyError("Base feature table must include spam_rate")

    return df


def _parse_frozenset(raw: str) -> frozenset[str]:
    try:
        return eval(raw, {"__builtins__": {}}, {"frozenset": frozenset})
    except Exception:
        return frozenset()


def _rc_bin(value: float) -> str:
    if value <= 2:
        return "review_count=Low"
    if value <= 10:
        return "review_count=Medium"
    return "review_count=High"


def _tenure_bin(value: float) -> str:
    if value < 30:
        return "tenure=new"
    if value <= 180:
        return "tenure=moderate"
    if value <= 365:
        return "tenure=established"
    return "tenure=veteran"


def _seller_bin(value: float) -> str:
    return "seller_conc=High" if value >= 0.5 else "seller_conc=Low"


def _burst_bin(value: float) -> str:
    return "burst=Bursty" if value > 2 else "burst=Normal"


def _build_item_set(row: pd.Series) -> set[str]:
    items: set[str] = set()

    if pd.notna(row.get("review_count", np.nan)):
        items.add(_rc_bin(float(row["review_count"])))
    if pd.notna(row.get("tenure_days", np.nan)):
        items.add(_tenure_bin(float(row["tenure_days"])))
    if pd.notna(row.get("max_seller_fraction", np.nan)):
        items.add(_seller_bin(float(row["max_seller_fraction"])))
    if pd.notna(row.get("burst_score", np.nan)):
        items.add(_burst_bin(float(row["burst_score"])))

    return items


def _load_l2_rules() -> pd.DataFrame:
    if not L2_RULES_PATH.exists():
        return pd.DataFrame(columns=["antecedents", "support", "confidence", "lift", "antecedent_spam_rate"])

    rules = pd.read_csv(L2_RULES_PATH)
    required = {"antecedents", "support", "confidence", "lift", "antecedent_spam_rate"}
    missing = sorted(required - set(rules.columns))
    if missing:
        raise KeyError(f"Missing columns in {L2_RULES_PATH}: {missing}")

    rules["antecedents"] = rules["antecedents"].astype(str).apply(_parse_frozenset)
    rules = rules[rules["antecedents"].apply(len) > 0].copy()
    rules["rule_weight"] = (
        rules["support"].astype(float)
        * rules["confidence"].astype(float)
        * rules["lift"].astype(float)
        * rules["antecedent_spam_rate"].astype(float)
    )
    rules = rules.sort_values("rule_weight", ascending=False).head(TOP_L2_RULES).reset_index(drop=True)
    return rules


def _compute_l2_features(base: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame(
            {
                "user_id": base["user_id"],
                "l2_rule_match_count": 0,
                "l2_rule_max_weight": 0.0,
                "l2_rule_partial_mean": 0.0,
                "l2_rule_partial_max": 0.0,
            }
        )

    antecedents = rules["antecedents"].tolist()
    weights = rules["rule_weight"].to_numpy(dtype=float)

    match_count = np.zeros(len(base), dtype=int)
    max_weight = np.zeros(len(base), dtype=float)
    partial_mean = np.zeros(len(base), dtype=float)
    partial_max = np.zeros(len(base), dtype=float)

    for idx, row in base.iterrows():
        items = _build_item_set(row)
        if not items:
            continue

        partial_scores: list[float] = []
        full_hits = 0
        max_hit_weight = 0.0
        max_partial = 0.0

        for ant, wt in zip(antecedents, weights):
            overlap = len(items.intersection(ant))
            if overlap == 0:
                partial_scores.append(0.0)
                continue

            frac = overlap / max(1, len(ant))
            pscore = float(frac * wt)
            partial_scores.append(pscore)
            if pscore > max_partial:
                max_partial = pscore

            if ant.issubset(items):
                full_hits += 1
                if wt > max_hit_weight:
                    max_hit_weight = float(wt)

        if partial_scores:
            partial_mean[idx] = float(np.mean(partial_scores))
            partial_max[idx] = max_partial
        match_count[idx] = full_hits
        max_weight[idx] = max_hit_weight

    return pd.DataFrame(
        {
            "user_id": base["user_id"],
            "l2_rule_match_count": match_count,
            "l2_rule_max_weight": max_weight,
            "l2_rule_partial_mean": partial_mean,
            "l2_rule_partial_max": partial_max,
        }
    )


def _load_l4_features(base_user_ids: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"user_id": base_user_ids})

    if L4_KMEANS_PATH.exists():
        km = pd.read_csv(L4_KMEANS_PATH, usecols=["user_id", "cluster_id"])
        out = out.merge(km, on="user_id", how="left")
        out.rename(columns={"cluster_id": "kmeans_cluster_id"}, inplace=True)

    if L4_DBSCAN_PATH.exists():
        db = pd.read_csv(L4_DBSCAN_PATH, usecols=["user_id", "dbscan_cluster"])
        out = out.merge(db, on="user_id", how="left")
        out["dbscan_is_noise"] = (out["dbscan_cluster"] == -1).astype(int)

    # NOTE: cluster-level spam_rate_mean was previously joined here, but that
    # leaks the target variable (spam_rate) into the feature set.  Cluster IDs
    # and the noise flag already capture the structural signal without leakage.

    return out


def main() -> None:
    L5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_path = _pick_existing(BASE_FEATURE_CANDIDATES)
    base = pd.read_csv(base_path)
    base = _normalize_base_columns(base)

    base = base.copy()
    # Label: any reviewer with at least one spam-flagged review is positive.
    # Previously used > 0.5, but 65% of reviewers have exactly 1 review,
    # making > 0.5 equivalent to "all reviews are spam" — too restrictive.
    base["spam_label"] = (base["spam_rate"].astype(float) > 0).astype(int)

    l2_rules = _load_l2_rules()
    l2_features = _compute_l2_features(base, l2_rules)
    l4_features = _load_l4_features(base["user_id"])

    final_df = base.merge(l2_features, on="user_id", how="left")
    final_df = final_df.merge(l4_features, on="user_id", how="left")

    # Fill obvious nulls from optional merges.
    for col in [
        "kmeans_cluster_id",
        "dbscan_cluster",
        "dbscan_is_noise",
    ]:
        if col in final_df.columns:
            if col == "dbscan_is_noise":
                final_df[col] = final_df[col].fillna(0)
            else:
                final_df[col] = final_df[col].fillna(-1)

    output_path = L5_OUTPUT_DIR / "l5_feature_table.csv"
    final_df.to_csv(output_path, index=False)

    feature_columns = [
        c
        for c in final_df.columns
        if c
        not in {
            "user_id",
            "spam_label",
            "spam_rate",
            "spam_count",
            "is_spam",
            "is_spam_reviewer",
            "label",
        }
    ]
    (L5_OUTPUT_DIR / "l5_feature_columns.txt").write_text("\n".join(feature_columns), encoding="utf-8")

    metadata = {
        "base_feature_source": str(base_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(final_df)),
        "columns": int(len(final_df.columns)),
        "positive_rate": float(final_df["spam_label"].mean()),
        "l2_rules_used": int(len(l2_rules)),
        "has_kmeans": bool(L4_KMEANS_PATH.exists()),
        "has_dbscan": bool(L4_DBSCAN_PATH.exists()),
    }
    (L5_OUTPUT_DIR / "l5_build_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Base source: {base_path}")
    print(f"Rows: {len(final_df):,}")
    print(f"Columns: {len(final_df.columns)}")
    print(f"Label positive rate: {final_df['spam_label'].mean():.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
