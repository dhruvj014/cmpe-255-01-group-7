from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "L6_Validation/outputs"


def test_hard_tier_count_increased():
    df = pd.read_csv(OUT / "synthetic_profiles.csv")
    hard = df[df["tier"] == "hard"]
    assert len(hard) >= 30, f"Expected >=30 hard profiles, got {len(hard)}"


def test_total_profiles_consistent():
    df = pd.read_csv(OUT / "synthetic_profiles.csv")
    res = pd.read_csv(OUT / "synthetic_detection_results.csv")
    assert len(res) >= len(df) // 2, "Detection results must cover all/most profiles"


def test_tenure_widened():
    df = pd.read_csv(OUT / "synthetic_profiles.csv")
    hard = df[df["tier"] == "hard"]
    assert hard["tenure_days"].max() >= 900, "Hard tier must reach near 1000 days"
    assert hard["tenure_days"].min() <= 350, "Hard tier must include ~300-day profiles"
