# pip install mlxtend
"""FP-Growth Frequent Itemset Mining.

Reads the pipe-separated baskets produced by 01_basket_encoding.py, one-hot
encodes them with mlxtend's TransactionEncoder, and mines frequent itemsets
using the FP-Growth algorithm.

Outputs:
    - frequent_itemsets.csv : itemsets sorted by support (descending)

Usage:
    python L2_FPGrowth/02_fpgrowth_mining.py
"""

import os

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJECT_ROOT, "L2_FPGrowth", "outputs", "encoded_baskets.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "L2_FPGrowth", "outputs")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# min_support=0.05 was chosen because with 260,277 reviewers an itemset must
# appear in at least ~13,000 reviewers to be considered frequent.  This
# threshold filters out noisy, rare combinations while preserving meaningful
# behavioural patterns worth investigating.
MIN_SUPPORT = 0.05


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run FP-Growth mining and save / print results."""

    # ---- 1. Load encoded baskets ----
    print(f"Reading {INPUT_CSV} ...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run 01_basket_encoding.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df):,} reviewer baskets")

    # ---- 2. Reconstruct basket lists from pipe-separated strings ----
    transactions = df["basket"].str.split("|").tolist()

    # ---- 3. One-hot encode with TransactionEncoder ----
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    print(f"  Transaction matrix: {te_df.shape[0]:,} rows × {te_df.shape[1]} items")

    # ---- 4. Run FP-Growth ----
    print(f"\nRunning FP-Growth (min_support={MIN_SUPPORT}) ...")
    freq = fpgrowth(te_df, min_support=MIN_SUPPORT, use_colnames=True)
    freq = freq.sort_values("support", ascending=False).reset_index(drop=True)

    # ---- 5. Save results ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "frequent_itemsets.csv")
    freq.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")

    # ---- 6. Console summary ----
    print("\n" + "=" * 60)
    print("FP-GROWTH RESULTS")
    print("=" * 60)

    print(f"\nTotal frequent itemsets found: {len(freq):,}")

    print("\n-- Top 15 itemsets by support --")
    for _, row in freq.head(15).iterrows():
        items = ", ".join(sorted(row["itemsets"]))
        print(f"  {row['support']:.4f}  {{ {items} }}")

    freq["size"] = freq["itemsets"].apply(len)
    size_counts = freq["size"].value_counts().sort_index()
    print("\n-- Itemset count by size --")
    for size, count in size_counts.items():
        print(f"  {size}-item:  {count:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
