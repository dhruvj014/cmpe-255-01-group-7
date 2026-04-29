from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "L5_Classification/outputs"
PLOTS = ROOT / "L5_Classification/plots"


def test_buckets_written():
    df = pd.read_csv(OUT / "error_buckets.csv")
    for c in ["tenure_bucket", "rating", "outcome", "count"]:
        assert c in df.columns
    assert set(df["outcome"]) == {"TP", "FP", "FN", "TN"}


def test_feature_deltas_written():
    fp = pd.read_csv(OUT / "feature_delta_fp.csv")
    fn = pd.read_csv(OUT / "feature_delta_fn.csv")
    for df in (fp, fn):
        assert {"feature", "delta", "abs_delta"}.issubset(df.columns)
        assert len(df) >= 10


def test_heatmap_written():
    assert (PLOTS / "error_heatmap.png").exists()
