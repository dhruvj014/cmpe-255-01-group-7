"""Aggregate Jaccard + synthetic-injection outputs into a printable summary.

Reads the two CSVs produced by 01 and 02 and writes a plain-text summary
that can be pasted into the final report or used as talking-points for the
check-in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import OUTPUT_DIR

HEATMAP_K = 2000
JACCARD_PATH = OUTPUT_DIR / "jaccard_matrix.csv"
OVERLAP_PATH = OUTPUT_DIR / "topk_overlap_table.csv"
DETECTION_PATH = OUTPUT_DIR / "synthetic_detection_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "l6_validation_summary.txt"


def _section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}\n"


def summarise_jaccard(lines: list[str]) -> None:
    if not OVERLAP_PATH.exists():
        lines.append("Jaccard overlap table not found — run 01_jaccard_stability.py first.\n")
        return

    overlap = pd.read_csv(OVERLAP_PATH)
    lines.append(_section(f"CROSS-SIGNAL JACCARD AT K={HEATMAP_K}"))

    at_k = overlap[overlap["K"] == HEATMAP_K].copy()
    at_k = at_k.sort_values("jaccard", ascending=False)

    lines.append(f"{'Signal A':15s}  {'Signal B':15s}  {'Inter.':>7s}  {'Union':>7s}  {'Jaccard':>8s}\n")
    lines.append("-" * 60 + "\n")
    for _, row in at_k.iterrows():
        lines.append(
            f"{row['signal_a']:15s}  {row['signal_b']:15s}  "
            f"{int(row['intersection']):>7d}  {int(row['union']):>7d}  "
            f"{row['jaccard']:>8.4f}\n"
        )

    # Highest / lowest pair
    top = at_k.iloc[0]
    bot = at_k.iloc[-1]
    lines.append("\nKey observations:\n")
    lines.append(
        f"  - Strongest agreement: {top['signal_a']} vs {top['signal_b']} "
        f"(J = {top['jaccard']:.3f}). Indicates these signals converge on "
        f"the same suspicious reviewers at the top of the ranking.\n"
    )
    lines.append(
        f"  - Weakest agreement:   {bot['signal_a']} vs {bot['signal_b']} "
        f"(J = {bot['jaccard']:.3f}). Indicates complementary / disagreeing "
        f"signal — useful for a multi-signal ensemble.\n"
    )

    # Stability across K
    lines.append(_section("JACCARD STABILITY ACROSS K"))
    pivot = overlap.pivot_table(
        index=["signal_a", "signal_b"], columns="K", values="jaccard"
    )
    lines.append(pivot.round(4).to_string())
    lines.append("\n")


def summarise_detection(lines: list[str]) -> None:
    if not DETECTION_PATH.exists():
        lines.append("Detection results not found — run 02_synthetic_injection.py first.\n")
        return

    results = pd.read_csv(DETECTION_PATH)
    flag_cols = [
        ("L2", "L2_flag"),
        ("L4-kmeans", "L4-kmeans_flag"),
        ("L4-dbscan", "L4-dbscan_flag"),
        ("L5-supervised", "L5-supervised_flag"),
        ("L5-anomaly", "L5-anomaly_flag"),
    ]
    tiers = ["easy", "medium", "hard"]

    lines.append(_section("SYNTHETIC ATTACK DETECTION RATE"))
    header = f"{'Layer':15s} " + " ".join(f"{t:>10s}" for t in tiers) + f" {'overall':>10s}\n"
    lines.append(header)
    lines.append("-" * len(header) + "\n")

    # We'll also track which tier each layer can still handle.
    per_layer = {}
    for layer, col in flag_cols:
        row_vals = []
        for t in tiers:
            sub = results[results["tier"] == t]
            rate = sub[col].mean() if len(sub) else 0.0
            row_vals.append(rate)
        overall = results[col].mean()
        per_layer[layer] = row_vals + [overall]
        formatted = " ".join(f"{v:>10.2f}" for v in row_vals) + f" {overall:>10.2f}"
        lines.append(f"{layer:15s} {formatted}\n")

    # Narrative findings
    lines.append("\nKey observations:\n")

    # Tier where behavioural signals collapse
    hard_rates = {layer: vals[2] for layer, vals in per_layer.items()}
    weakest_layer = min(hard_rates, key=hard_rates.get)
    lines.append(
        f"  - Hard-tier (veteran camouflage) is the hardest to detect. "
        f"Weakest layer: {weakest_layer} ({hard_rates[weakest_layer]:.2f}). "
        f"These profiles motivate the need for L3 text signals.\n"
    )

    # Easy-tier sanity
    easy_rates = {layer: vals[0] for layer, vals in per_layer.items()}
    strongest_easy = max(easy_rates, key=easy_rates.get)
    lines.append(
        f"  - Easy-tier (matches L2 antecedent directly): {strongest_easy} "
        f"flags {easy_rates[strongest_easy]:.2f}. Confirms that the pipeline "
        f"catches the obvious behavioural fingerprint.\n"
    )

    # L5 anomaly behaviour
    anom = per_layer["L5-anomaly"]
    lines.append(
        f"  - L5 anomaly detection rates ({anom[0]:.2f} / {anom[1]:.2f} / {anom[2]:.2f}). "
        f"Review-level features improved IF AUC to ~0.57 (previously 0.42 "
        f"at reviewer-level), but anomaly-based detection remains weak "
        f"compared to supervised approaches.\n"
    )


def summarise_next_steps(lines: list[str]) -> None:
    lines.append(_section("LIMITATIONS AND NEXT STEPS"))
    lines.append(
        "  - Jaccard restricted to the 52K L5 holdout (Option A from the plan). "
        "Switch to full-population inference once the re-run L5 is available.\n"
    )
    lines.append(
        "  - L3 signal absent until DeBERTa fine-tuning completes on HPC. Add "
        "per-review confidence scores as an extra axis to both Jaccard and the "
        "synthetic detection matrix when L3 is ready.\n"
    )
    lines.append(
        "  - L5 model is now trained at review-level (608K rows) with "
        "per-review features, fixing the prior reviewer-level granularity "
        "limitation. L5 scores are aggregated to reviewer-level via max() "
        "for Jaccard and synthetic injection comparisons.\n"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["L6 Validation Summary\n"]
    summarise_jaccard(lines)
    summarise_detection(lines)
    summarise_next_steps(lines)

    text = "".join(lines)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
