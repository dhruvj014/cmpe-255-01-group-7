"""Synthetic attack injection — L6 validation.

Generates 40 synthetic fake-reviewer profiles across three difficulty tiers
and runs each layer's detection logic on them.  The goal is to quantify
*per-layer* detection rate as a function of how well the attacker
camouflages their behavioural fingerprint.

For L5 (review-level model), each synthetic reviewer generates N synthetic
reviews.  Detection is aggregated: reviewer flagged if max(review_scores) >
threshold.

Tiers:
    easy   (15 profiles)  matches the L2 rule fingerprint directly
    medium (15 profiles)  evades L2 but stays unusual enough for L4
    hard   (10 profiles)  veteran-style camouflage; expected to evade
                          every behavioural layer (L3 would catch these)

Outputs:
    outputs/synthetic_profiles.csv
    outputs/synthetic_detection_results.csv
    plots/detection_rate_by_layer.png
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from utils import (
    CLUSTERING_FEATURES,
    L4_DB_SUMMARY,
    L4_DBSCAN,
    L4_KM_SUMMARY,
    L4_KMEANS,
    L5_FEATURE_TABLE,
    L5_MODEL,
    L5_RAW_FEATURES,
    OUTPUT_DIR,
    PLOT_DIR,
    build_item_set,
    fit_clustering_scaler,
    load_l2_rules,
    prepare_for_l5_model,
    transform_new_profiles,
)

RANDOM_STATE = 42
SPAM_RATE_THRESHOLD = 0.20  # matches plan: cluster flagged if spam_rate_mean > 0.20

LAYERS = ["L2", "L4-kmeans", "L4-dbscan", "L5-supervised", "L5-anomaly"]
TIER_COUNTS = {"easy": 15, "medium": 15, "hard": 10}


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

def _sample(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _make_profile(rng: np.random.Generator, tier: str, idx: int) -> dict:
    """Build a single synthetic reviewer profile for the given tier."""
    if tier == "easy":
        review_count = 1
        tenure_days = 0.0
        max_seller_fraction = 1.0
        burst_score = 1.0
        rating_entropy = 0.0
        avg_rating = float(rng.choice([1.0, 5.0]))
        rating_std = 0.0
        unique_sellers = 1
        avg_review_length = float(rng.normal(70, 15))
    elif tier == "medium":
        review_count = int(rng.integers(3, 7))  # 3-6
        tenure_days = _sample(rng, 30, 90)
        max_seller_fraction = _sample(rng, 0.40, 0.60)
        burst_score = _sample(rng, 2.0, 4.0)
        rating_entropy = _sample(rng, 0.3, 0.7)
        avg_rating = _sample(rng, 3.5, 4.5)
        rating_std = _sample(rng, 0.4, 0.9)
        unique_sellers = max(1, int(round(review_count * _sample(rng, 0.4, 0.7))))
        avg_review_length = float(rng.normal(90, 20))
    elif tier == "hard":
        review_count = int(rng.integers(8, 16))  # 8-15
        tenure_days = _sample(rng, 400, 800)
        max_seller_fraction = _sample(rng, 0.15, 0.30)
        burst_score = _sample(rng, 1.0, 2.0)
        rating_entropy = _sample(rng, 1.0, 1.5)
        avg_rating = _sample(rng, 3.5, 4.0)
        rating_std = _sample(rng, 0.9, 1.3)
        unique_sellers = max(2, int(round(review_count * _sample(rng, 0.6, 0.9))))
        avg_review_length = float(rng.normal(130, 25))
    else:
        raise ValueError(f"Unknown tier {tier!r}")

    if tenure_days > 0:
        reviews_per_week = review_count / max(tenure_days / 7.0, 1.0)
    else:
        reviews_per_week = float(review_count)
    avg_days_between_reviews = 0.0 if review_count <= 1 else tenure_days / max(review_count - 1, 1)
    avg_word_count = max(1.0, avg_review_length / 5.5)

    return {
        "user_id": f"SYNTH_{tier}_{idx:03d}",
        "tier": tier,
        "review_count": review_count,
        "tenure_days": tenure_days,
        "max_seller_fraction": max_seller_fraction,
        "burst_score": burst_score,
        "rating_entropy": rating_entropy,
        "avg_rating": avg_rating,
        "rating_std": rating_std,
        "unique_sellers": unique_sellers,
        "avg_review_length": max(20.0, avg_review_length),
        "avg_word_count": avg_word_count,
        "reviews_per_week": reviews_per_week,
        "avg_days_between_reviews": avg_days_between_reviews,
        "is_spam_truth": 1,  # all synthetic profiles are known spammers
    }


def generate_synthetic_profiles() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict] = []
    for tier, count in TIER_COUNTS.items():
        for i in range(count):
            rows.append(_make_profile(rng, tier, i))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generate synthetic REVIEWS from reviewer profiles (for review-level L5)
# ---------------------------------------------------------------------------

def _generate_synthetic_reviews(profiles: pd.DataFrame) -> pd.DataFrame:
    """For each synthetic reviewer profile, generate N synthetic reviews.

    Each review inherits the reviewer's aggregate features and gets its own
    review-level features (rating, review_length, etc.).
    """
    rng = np.random.default_rng(RANDOM_STATE + 1)
    review_rows = []

    for _, prof in profiles.iterrows():
        n_reviews = int(prof["review_count"])
        tier = prof["tier"]

        for r_idx in range(n_reviews):
            # Sample review-level features based on tier
            if tier == "easy":
                rating = float(prof["avg_rating"])  # extreme rating
                review_length = max(20, float(rng.normal(70, 15)))
            elif tier == "medium":
                rating = float(np.clip(rng.normal(prof["avg_rating"], prof["rating_std"]), 1, 5))
                review_length = max(20, float(rng.normal(90, 20)))
            else:  # hard
                rating = float(np.clip(rng.normal(prof["avg_rating"], prof["rating_std"]), 1, 5))
                review_length = max(20, float(rng.normal(130, 25)))

            word_count = max(1, int(review_length / 5.5))
            exclamation_count = int(rng.poisson(1 if tier == "easy" else 0.3))
            question_count = int(rng.poisson(0.2))
            capital_ratio = float(np.clip(rng.normal(0.05 if tier != "easy" else 0.15, 0.03), 0, 1))
            avg_word_length = float(np.clip(rng.normal(4.5, 0.5), 2, 8))
            day_of_week = int(rng.integers(0, 7))
            month = int(rng.integers(1, 13))

            review_row = {
                "user_id": prof["user_id"],
                "tier": tier,
                # Review-own features
                "rating": round(rating, 1),
                "review_length": int(review_length),
                "word_count": word_count,
                "exclamation_count": exclamation_count,
                "question_count": question_count,
                "capital_ratio": round(capital_ratio, 4),
                "avg_word_length": round(avg_word_length, 2),
                "day_of_week": day_of_week,
                "month": month,
                # Reviewer-aggregate features (inherited from profile)
                "review_count": int(prof["review_count"]),
                "avg_rating": float(prof["avg_rating"]),
                "rating_std": float(prof["rating_std"]),
                "avg_review_length": float(prof["avg_review_length"]),
                "avg_word_count": float(prof["avg_word_count"]),
                "unique_sellers": int(prof["unique_sellers"]),
                "tenure_days": float(prof["tenure_days"]),
                "reviews_per_week": float(prof["reviews_per_week"]),
                "max_seller_fraction": float(prof["max_seller_fraction"]),
                "avg_days_between_reviews": float(prof["avg_days_between_reviews"]),
                "burst_score": float(prof["burst_score"]),
                "rating_entropy": float(prof["rating_entropy"]),
            }
            review_rows.append(review_row)

    return pd.DataFrame(review_rows)


# ---------------------------------------------------------------------------
# Layer-specific detection
# ---------------------------------------------------------------------------

def detect_l2(profiles: pd.DataFrame) -> np.ndarray:
    """Flag profiles whose discretised basket contains any spam-correlated
    rule antecedent (full-match). Unchanged — operates on reviewer-level.
    """
    rules = load_l2_rules()
    antecedents = rules["antecedents"].tolist()

    flags = np.zeros(len(profiles), dtype=int)
    for i, row in profiles.iterrows():
        items = build_item_set(row)
        for ant in antecedents:
            if ant.issubset(items):
                flags[i] = 1
                break
    return flags


def assign_l4_clusters(profiles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour cluster assignment for synthetic profiles.
    Unchanged — operates on reviewer-level.
    """
    scaler, scaled_df = fit_clustering_scaler()
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(scaled_df[CLUSTERING_FEATURES].values)

    synth_scaled = transform_new_profiles(scaler, profiles)
    _, idx = nn.kneighbors(synth_scaled)
    neighbour_user_ids = scaled_df.iloc[idx.flatten()]["user_id"].values

    km = pd.read_csv(L4_KMEANS, usecols=["user_id", "cluster_id"])
    db = pd.read_csv(L4_DBSCAN, usecols=["user_id", "dbscan_cluster"])
    km_map = dict(zip(km["user_id"], km["cluster_id"]))
    db_map = dict(zip(db["user_id"], db["dbscan_cluster"]))

    kmeans_ids = np.array([km_map.get(int(uid), -1) for uid in neighbour_user_ids], dtype=int)
    dbscan_ids = np.array([db_map.get(int(uid), -1) for uid in neighbour_user_ids], dtype=int)
    return kmeans_ids, dbscan_ids


def detect_l4(kmeans_ids: np.ndarray, dbscan_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flag clusters whose spam_rate_mean > SPAM_RATE_THRESHOLD.  DBSCAN
    noise (-1) is always flagged. Unchanged — operates on reviewer-level."""
    km_sum = pd.read_csv(L4_KM_SUMMARY, usecols=["cluster_id", "spam_rate_mean"])
    db_sum = pd.read_csv(L4_DB_SUMMARY, usecols=["dbscan_cluster", "spam_rate_mean"])

    km_flagged = set(km_sum[km_sum["spam_rate_mean"] > SPAM_RATE_THRESHOLD]["cluster_id"].values)
    db_flagged = set(db_sum[db_sum["spam_rate_mean"] > SPAM_RATE_THRESHOLD]["dbscan_cluster"].values)

    km_flag = np.array([1 if c in km_flagged else 0 for c in kmeans_ids], dtype=int)
    db_flag = np.array(
        [1 if (c == -1) or (c in db_flagged) else 0 for c in dbscan_ids], dtype=int
    )
    return km_flag, db_flag


def _load_optimal_threshold(model_name: str) -> float:
    """Load the optimal F1 threshold from supervised_threshold_metadata.json."""
    import json
    meta_path = L5_MODEL.parent / "supervised_threshold_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if model_name in meta:
            return float(meta[model_name]["optimal_threshold"])
    return 0.5


def detect_l5_supervised(
    reviews_with_features: pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the persisted best supervised model on synthetic review-level vectors.

    Scores each synthetic review, then aggregates to reviewer-level:
    reviewer flagged if max(review_scores) > optimal threshold.
    """
    from utils import load_best_supervised_name

    model = joblib.load(L5_MODEL)
    model_name = load_best_supervised_name()
    threshold = _load_optimal_threshold(model_name)

    X = reviews_with_features[L5_RAW_FEATURES].copy()
    X = prepare_for_l5_model(X, model)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = model.predict(X).astype(float)

    # Aggregate review-level scores to reviewer-level: max(score) per user
    reviews_with_features = reviews_with_features.copy()
    reviews_with_features["_score"] = scores

    reviewer_scores = reviews_with_features.groupby("user_id", as_index=False)["_score"].max()

    # Align with profiles order
    score_map = dict(zip(reviewer_scores["user_id"], reviewer_scores["_score"]))
    profile_scores = np.array([score_map.get(uid, 0.0) for uid in profiles["user_id"]])
    profile_preds = (profile_scores >= threshold).astype(int)

    return profile_preds, profile_scores


def _onehot_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode kmeans_cluster_id only, matching L5's _prepare_features."""
    X = df.copy()
    cluster_cols = [c for c in ("kmeans_cluster_id",) if c in X.columns]
    if cluster_cols:
        for c in cluster_cols:
            X[c] = X[c].astype(int).astype(str)
        X = pd.get_dummies(X, columns=cluster_cols, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X


def detect_l5_anomaly(
    reviews_with_features: pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Isolation Forest on legitimate L5 review-level feature vectors and
    score the synthetic reviews. Aggregate to reviewer-level."""
    feat_df = pd.read_csv(L5_FEATURE_TABLE)

    X_train = _onehot_clusters(feat_df[L5_RAW_FEATURES].copy())
    y_train = (feat_df["is_spam"].astype(int) == 1).to_numpy()
    legit_mask = ~y_train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_legit = X_train_scaled[legit_mask]

    contamination = float(np.clip(np.round(y_train.mean(), 3), 0.01, 0.40))
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train_legit)

    X_synth = _onehot_clusters(reviews_with_features[L5_RAW_FEATURES].copy())
    # Align columns with training set
    for col in X_train.columns:
        if col not in X_synth.columns:
            X_synth[col] = 0
    X_synth = X_synth.reindex(columns=X_train.columns, fill_value=0)

    X_synth_scaled = scaler.transform(X_synth)
    scores = -iso.score_samples(X_synth_scaled)

    # Aggregate review-level scores to reviewer-level
    reviews_with_features = reviews_with_features.copy()
    reviews_with_features["_anom_score"] = scores

    reviewer_scores = reviews_with_features.groupby("user_id", as_index=False)["_anom_score"].max()
    score_map = dict(zip(reviewer_scores["user_id"], reviewer_scores["_anom_score"]))

    profile_scores = np.array([score_map.get(uid, 0.0) for uid in profiles["user_id"]])

    # Use median score of all reviews as threshold (same as IF's internal logic)
    raw_preds = iso.predict(X_synth_scaled)
    reviews_with_features["_anom_pred"] = np.where(raw_preds == -1, 1, 0)
    reviewer_preds = reviews_with_features.groupby("user_id", as_index=False)["_anom_pred"].max()
    pred_map = dict(zip(reviewer_preds["user_id"], reviewer_preds["_anom_pred"]))
    profile_preds = np.array([pred_map.get(uid, 0) for uid in profiles["user_id"]], dtype=int)

    return profile_preds, profile_scores


# ---------------------------------------------------------------------------
# L2 feature engineering for synthetic profiles (mirrors L5/01 build)
# ---------------------------------------------------------------------------

def compute_l2_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """Compute the four l2_* feature columns used by the L5 model."""
    from utils import L2_RULES, parse_frozenset

    rules = pd.read_csv(L2_RULES)
    rules["antecedents"] = rules["antecedents"].astype(str).apply(parse_frozenset)
    rules = rules[rules["antecedents"].apply(len) > 0].copy()
    rules["rule_weight"] = (
        rules["support"].astype(float)
        * rules["confidence"].astype(float)
        * rules["lift"].astype(float)
        * rules["antecedent_spam_rate"].astype(float)
    )
    rules = rules.sort_values("rule_weight", ascending=False).head(50).reset_index(drop=True)

    antecedents = rules["antecedents"].tolist()
    weights = rules["rule_weight"].to_numpy(dtype=float)

    rows = []
    for _, row in profiles.iterrows():
        items = build_item_set(row)
        partial_scores = []
        full_hits = 0
        max_hit_weight = 0.0
        max_partial = 0.0
        for ant, wt in zip(antecedents, weights):
            overlap = len(items & ant)
            if overlap == 0:
                partial_scores.append(0.0)
                continue
            frac = overlap / max(1, len(ant))
            p = float(frac * wt)
            partial_scores.append(p)
            if p > max_partial:
                max_partial = p
            if ant.issubset(items):
                full_hits += 1
                if wt > max_hit_weight:
                    max_hit_weight = float(wt)
        rows.append({
            "l2_rule_match_count": int(full_hits),
            "l2_rule_max_weight": float(max_hit_weight),
            "l2_rule_partial_mean": float(np.mean(partial_scores)) if partial_scores else 0.0,
            "l2_rule_partial_max": float(max_partial),
        })
    return pd.DataFrame(rows, index=profiles.index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic profiles...")
    profiles = generate_synthetic_profiles()
    print(f"  {len(profiles)} profiles: "
          + ", ".join(f"{t}={c}" for t, c in TIER_COUNTS.items()))

    profiles_path = OUTPUT_DIR / "synthetic_profiles.csv"
    profiles.to_csv(profiles_path, index=False)
    print(f"  Saved {profiles_path}")

    # --- L2 detection + rule features for L5 input ---
    print("\nL2 detection (rule antecedent match)...")
    l2_flag = detect_l2(profiles)

    l2_features = compute_l2_features(profiles)
    profiles_aug = pd.concat([profiles.reset_index(drop=True), l2_features], axis=1)

    # --- L4 cluster assignment + detection ---
    print("L4 nearest-neighbour cluster assignment...")
    kmeans_ids, dbscan_ids = assign_l4_clusters(profiles_aug)
    profiles_aug["kmeans_cluster_id"] = kmeans_ids
    profiles_aug["dbscan_cluster"] = dbscan_ids
    profiles_aug["dbscan_is_noise"] = (dbscan_ids == -1).astype(int)

    km_flag, db_flag = detect_l4(kmeans_ids, dbscan_ids)

    # --- Generate synthetic REVIEWS for review-level L5 ---
    print("Generating synthetic reviews for review-level L5...")
    synth_reviews = _generate_synthetic_reviews(profiles_aug)
    # Attach L2 features and L4 cluster assignments to each review
    # (inherited from the reviewer profile — same for all reviews of a user)
    l2_cols = ["l2_rule_match_count", "l2_rule_max_weight", "l2_rule_partial_mean", "l2_rule_partial_max"]
    l4_cols = ["kmeans_cluster_id", "dbscan_is_noise"]
    reviewer_meta = profiles_aug[["user_id"] + l2_cols + l4_cols].copy()
    # Drop any L2/L4 cols already in synth_reviews to avoid duplication
    synth_reviews = synth_reviews.drop(columns=[c for c in l2_cols + l4_cols if c in synth_reviews.columns], errors="ignore")
    synth_reviews = synth_reviews.merge(reviewer_meta, on="user_id", how="left")
    print(f"  Generated {len(synth_reviews)} synthetic reviews")

    # --- L5 supervised ---
    print("L5 supervised model inference (review-level)...")
    sup_flag, sup_score = detect_l5_supervised(synth_reviews, profiles)

    # --- L5 anomaly ---
    print("L5 anomaly (Isolation Forest refit, review-level)...")
    anom_flag, anom_score = detect_l5_anomaly(synth_reviews, profiles)

    # --- Assemble results ---
    results = profiles[["user_id", "tier"]].copy()
    results["L2_flag"] = l2_flag
    results["L4-kmeans_flag"] = km_flag
    results["L4-kmeans_cluster"] = kmeans_ids
    results["L4-dbscan_flag"] = db_flag
    results["L4-dbscan_cluster"] = dbscan_ids
    results["L5-supervised_flag"] = sup_flag
    results["L5-supervised_score"] = sup_score
    results["L5-anomaly_flag"] = anom_flag
    results["L5-anomaly_score"] = anom_score

    results_path = OUTPUT_DIR / "synthetic_detection_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nSaved {results_path}")

    # --- Detection-rate-by-layer bar chart ---
    flag_cols = {
        "L2": "L2_flag",
        "L4-kmeans": "L4-kmeans_flag",
        "L4-dbscan": "L4-dbscan_flag",
        "L5-supervised": "L5-supervised_flag",
        "L5-anomaly": "L5-anomaly_flag",
    }
    tiers = list(TIER_COUNTS.keys())
    rates = {layer: [results[results["tier"] == t][col].mean()
                     for t in tiers] for layer, col in flag_cols.items()}

    x = np.arange(len(tiers))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (layer, vals) in enumerate(rates.items()):
        offset = (i - len(rates) / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=layer)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} (n={TIER_COUNTS[t]})" for t in tiers])
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Synthetic Attack Detection Rate by Layer and Tier")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=3)
    plt.tight_layout()
    chart_path = PLOT_DIR / "detection_rate_by_layer.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved {chart_path}")

    # --- Console summary ---
    print("\n" + "=" * 70)
    print("DETECTION RATES")
    print("=" * 70)
    header = f"{'Layer':15s} " + " ".join(f"{t:>10s}" for t in tiers)
    print(header)
    print("-" * len(header))
    for layer, vals in rates.items():
        print(f"{layer:15s} " + " ".join(f"{v:>10.2f}" for v in vals))


if __name__ == "__main__":
    main()
