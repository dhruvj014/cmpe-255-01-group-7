import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "L5_Classification/outputs"
PLOTS = ROOT / "L5_Classification/plots"


def test_calibration_metrics_written():
    m = json.loads((OUT / "calibration_metrics.json").read_text())
    for variant in ["uncalibrated", "platt", "isotonic"]:
        assert variant in m
        assert "brier" in m[variant]
        assert "ece" in m[variant]


def test_reliability_plot_written():
    assert (PLOTS / "calibration_reliability.png").exists()
