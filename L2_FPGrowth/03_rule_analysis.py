"""Association Rule Mining & Spam Correlation Analysis.

Reads frequent itemsets from FP-Growth (02_fpgrowth_mining.py) and extracts
association rules using mlxtend.  Then correlates each rule's antecedent
pattern with reviewer spam rates to surface behavioural profiles that are
disproportionately associated with spam activity.

Outputs:
    - association_rules.csv      : all rules with support, confidence, lift
    - spam_correlated_rules.csv  : rules whose antecedent group has spam_rate > 0.20
    - plots/l2_top_rules_spam_rate.png : bar chart of top rules by spam rate

Usage:
    python L2_FPGrowth/03_rule_analysis.py
"""

import ast
import os

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import association_rules

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMSETS_CSV = os.path.join(PROJECT_ROOT, "L2_FPGrowth", "outputs", "frequent_itemsets.csv")
BASKETS_CSV = os.path.join(PROJECT_ROOT, "L2_FPGrowth", "outputs", "encoded_baskets.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "L2_FPGrowth", "outputs")
PLOT_DIR = os.path.join(PROJECT_ROOT, "L2_FPGrowth", "plots")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MIN_LIFT = 1.2          # only keep rules with lift above this threshold
SPAM_RATE_CUTOFF = 0.20 # >1.5× the 13.2% dataset average
DATASET_AVG_SPAM = 0.132


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frozenset(text: str) -> frozenset:
    """Reconstruct a frozenset from its CSV string representation."""
    return eval(text, {"__builtins__": {}}, {"frozenset": frozenset})


def _frozenset_to_str(fs: frozenset) -> str:
    """Pretty-print a frozenset as a sorted, comma-separated string."""
    return ", ".join(sorted(fs))


def _compute_antecedent_spam_rates(
    rules: pd.DataFrame, baskets: pd.DataFrame
) -> pd.Series:
    """For each rule, find reviewers whose basket contains ALL antecedent items
    and return the mean spam_rate of that group.

    Interpretation: a high value means reviewers exhibiting the antecedent
    pattern are, on average, flagged as spam far more often than the dataset
    baseline (13.2%).  Combined with high lift, this signals that the
    antecedent → consequent pattern is both statistically surprising AND
    concentrated among likely spammers.
    """
    basket_items = baskets["basket"].str.split("|").apply(set)
    spam_rates = baskets["spam_rate"].values
    results = []
    for antecedent in rules["antecedents"]:
        mask = basket_items.apply(lambda b: antecedent.issubset(b))
        group_spam = spam_rates[mask.values]
        results.append(group_spam.mean() if len(group_spam) > 0 else 0.0)
    return pd.Series(results, index=rules.index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- 1. Load frequent itemsets ----
    print(f"Reading {ITEMSETS_CSV} ...")
    if not os.path.exists(ITEMSETS_CSV):
        raise FileNotFoundError(
            f"{ITEMSETS_CSV} not found. Run 02_fpgrowth_mining.py first."
        )
    freq = pd.read_csv(ITEMSETS_CSV)
    freq["itemsets"] = freq["itemsets"].apply(_parse_frozenset)
    print(f"  Loaded {len(freq):,} frequent itemsets")

    # ---- 2. Extract association rules (lift >= 1.2) ----
    print(f"\nExtracting association rules (metric='lift', min_threshold={MIN_LIFT}) ...")
    rules = association_rules(freq, metric="lift", min_threshold=MIN_LIFT)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    print(f"  Found {len(rules):,} rules")

    if rules.empty:
        print("No rules found. Try lowering MIN_LIFT or MIN_SUPPORT.")
        return

    # ---- 3. Save all rules ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    rules_path = os.path.join(OUTPUT_DIR, "association_rules.csv")
    rules[out_cols].to_csv(rules_path, index=False)
    print(f"  Wrote {rules_path}")

    # ---- 4. Spam correlation analysis ----
    print(f"\nReading {BASKETS_CSV} for spam correlation ...")
    baskets = pd.read_csv(BASKETS_CSV)

    print("  Computing antecedent_spam_rate for each rule ...")
    rules["antecedent_spam_rate"] = _compute_antecedent_spam_rates(rules, baskets)

    # High lift means the co-occurrence of antecedent and consequent is much
    # more frequent than expected by chance.  When the same antecedent group
    # also carries a spam_rate well above the dataset average (13.2%), we have
    # a double signal: the behavioural pattern is (a) statistically surprising
    # AND (b) concentrated among spammers.  These rules are prime candidates
    # for features in a downstream spam classifier or for analyst review.
    spam_rules = (
        rules[rules["antecedent_spam_rate"] > SPAM_RATE_CUTOFF]
        .sort_values("antecedent_spam_rate", ascending=False)
        .reset_index(drop=True)
    )
    print(f"  {len(spam_rules):,} rules with antecedent_spam_rate > {SPAM_RATE_CUTOFF}")

    spam_path = os.path.join(OUTPUT_DIR, "spam_correlated_rules.csv")
    spam_cols = out_cols + ["antecedent_spam_rate"]
    spam_rules[spam_cols].to_csv(spam_path, index=False)
    print(f"  Wrote {spam_path}")

    # ---- 5. Console summary ----
    print("\n" + "=" * 70)
    print("TOP 10 SPAM-CORRELATED ASSOCIATION RULES")
    print("=" * 70)
    top10 = spam_rules.head(10)
    header = (
        f"{'antecedents':<40s} {'consequents':<25s} "
        f"{'supp':>6s} {'conf':>6s} {'lift':>6s} {'spam%':>6s}"
    )
    print(header)
    print("-" * len(header))
    for _, r in top10.iterrows():
        print(
            f"{_frozenset_to_str(r['antecedents']):<40s} "
            f"{_frozenset_to_str(r['consequents']):<25s} "
            f"{r['support']:6.3f} {r['confidence']:6.3f} "
            f"{r['lift']:6.2f} {r['antecedent_spam_rate']:6.3f}"
        )

    # ---- 6. Bar chart ----
    os.makedirs(PLOT_DIR, exist_ok=True)
    top_n = spam_rules.head(15)
    labels = [_frozenset_to_str(a) for a in top_n["antecedents"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(top_n)), top_n["antecedent_spam_rate"], color="#e74c3c")
    ax.axhline(y=DATASET_AVG_SPAM, color="red", linestyle="--", linewidth=1.2,
               label=f"Dataset avg spam rate ({DATASET_AVG_SPAM:.1%})")
    ax.set_xticks(range(len(top_n)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Rule Antecedent")
    ax.set_ylabel("Antecedent Spam Rate")
    ax.set_title("Top Association Rules by Spam Rate (FP-Growth)")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(PLOT_DIR, "l2_top_rules_spam_rate.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved plot to {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
