"""Basket Encoding for FP-Growth Analysis.

Reads reviewer_profiles.csv (produced by the L1 ETL pipeline) and discretizes
four numeric columns into categorical basket items suitable for frequent-
itemset mining.

Outputs:
    - encoded_baskets.csv  : per-reviewer basket (pipe-separated items) + spam_rate
    - basket_stats.csv     : count and mean spam_rate for each unique basket item

Usage:
    python L2_FPGrowth/01_basket_encoding.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJECT_ROOT, "L1_ETL_OLAP", "output_csv", "reviewer_profiles.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "L2_FPGrowth", "outputs")


# ---------------------------------------------------------------------------
# Discretization helpers
# ---------------------------------------------------------------------------

def discretize_review_count(value: int) -> str:
    """Bin review_count into Low / Medium / High.

    Thresholds:
        <= 2   -> Low    (casual reviewers)
        3-10   -> Medium (regular contributors)
        > 10   -> High   (power reviewers)
    """
    if value <= 2:
        return "review_count=Low"
    elif value <= 10:
        return "review_count=Medium"
    else:
        return "review_count=High"


def discretize_tenure(value: float) -> str:
    """Bin tenure_days into four activity-window categories.

    Thresholds:
        < 30    -> new          (less than a month)
        30-180  -> moderate     (1-6 months)
        181-365 -> established  (6-12 months)
        > 365   -> veteran      (more than a year)
    """
    if value < 30:
        return "tenure=new"
    elif value <= 180:
        return "tenure=moderate"
    elif value <= 365:
        return "tenure=established"
    else:
        return "tenure=veteran"


def discretize_seller_concentration(value: float) -> str:
    """Bin max_seller_fraction into Low / High.

    A reviewer whose most-reviewed seller accounts for >= 50 % of their
    reviews is considered highly concentrated.
    """
    if value < 0.5:
        return "seller_conc=Low"
    else:
        return "seller_conc=High"


def discretize_burst(value: float) -> str:
    """Bin burst_score into Normal / Bursty.

    burst_score > 2  means the reviewer posted more than 2 reviews
    in their densest 7-day window.
    """
    if value <= 2:
        return "burst=Normal"
    else:
        return "burst=Bursty"


def build_basket(row: pd.Series) -> list[str]:
    """Convert a single reviewer row into a list of basket item strings."""
    return [
        discretize_review_count(row["review_count"]),
        discretize_tenure(row["tenure_days"]),
        discretize_seller_concentration(row["max_seller_fraction"]),
        discretize_burst(row["burst_score"]),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Read reviewer features, encode baskets, write outputs, print summary."""

    # ---- 1. Load data ----
    print(f"Reading {INPUT_CSV} ...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run the L1 ETL pipeline first "
            "(L1_ETL_OLAP/main.py) to generate reviewer_profiles.csv."
        )
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Verify required columns are present
    required_cols = ["user_id", "review_count", "tenure_days",
                     "max_seller_fraction", "burst_score", "spam_rate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in reviewer_profiles.csv: {missing}")

    # ---- 2. Discretize each column individually (for stats later) ----
    df["rc_bin"] = df["review_count"].apply(discretize_review_count)
    df["tenure_bin"] = df["tenure_days"].apply(discretize_tenure)
    df["sc_bin"] = df["max_seller_fraction"].apply(discretize_seller_concentration)
    df["burst_bin"] = df["burst_score"].apply(discretize_burst)

    # ---- 3. Build basket per reviewer ----
    df["basket_list"] = df.apply(build_basket, axis=1)
    # Pipe-separated string for CSV storage
    df["basket"] = df["basket_list"].apply(lambda items: "|".join(items))

    # ---- 4. Write encoded_baskets.csv ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    baskets_path = os.path.join(OUTPUT_DIR, "encoded_baskets.csv")
    df[["user_id", "basket", "spam_rate"]].to_csv(baskets_path, index=False)
    print(f"\n  Wrote {baskets_path}")

    # ---- 5. Build basket_stats.csv ----
    # Explode each basket into individual items so we can aggregate per item
    exploded = df[["user_id", "basket_list", "spam_rate"]].explode("basket_list")
    exploded.rename(columns={"basket_list": "item"}, inplace=True)

    stats = (
        exploded
        .groupby("item")
        .agg(count=("user_id", "size"), spam_rate_mean=("spam_rate", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    stats_path = os.path.join(OUTPUT_DIR, "basket_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"  Wrote {stats_path}")

    # ---- 6. Print summary ----
    print("\n" + "=" * 60)
    print("BASKET ENCODING SUMMARY")
    print("=" * 60)

    print(f"\nTotal reviewers encoded: {len(df):,}")

    # Distribution per category
    bin_cols = {
        "review_count": "rc_bin",
        "tenure":       "tenure_bin",
        "seller_conc":  "sc_bin",
        "burst":        "burst_bin",
    }

    for label, col in bin_cols.items():
        dist = df[col].value_counts().sort_index()
        print(f"\n-- {label} distribution --")
        for val, cnt in dist.items():
            print(f"  {val:30s}  {cnt:>8,}  ({cnt / len(df) * 100:5.1f}%)")

    # Mean spam_rate per tenure bin (sanity check)
    print("\n-- Mean spam_rate by tenure bin --")
    tenure_spam = (
        df.groupby("tenure_bin")["spam_rate"]
        .mean()
        .reindex(["tenure=new", "tenure=moderate",
                  "tenure=established", "tenure=veteran"])
    )
    for bin_name, rate in tenure_spam.items():
        print(f"  {bin_name:30s}  spam_rate = {rate:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
