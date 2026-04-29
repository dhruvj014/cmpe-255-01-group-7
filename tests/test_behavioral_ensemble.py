"""Ensemble produces predictions, metrics, and plots."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "L5_Classification/outputs"
PLOTS = ROOT / "L5_Classification/plots"


def test_ensemble_predictions_written():
    df = pd.read_csv(OUT / "ensemble_predictions.csv")
    assert {"user_id", "score", "pred_at_optimal_threshold"}.issubset(df.columns)
    assert len(df) > 1000
    assert df["score"].between(0, 1).all()


def test_ensemble_metrics_written():
    m = json.loads((OUT / "ensemble_metrics.json").read_text())
    for k in ["auc_roc", "f1_at_05", "f1_at_optimal", "optimal_threshold",
              "avg_precision", "brier"]:
        assert k in m, f"missing metric: {k}"
    assert 0 <= m["auc_roc"] <= 1


def test_ensemble_plots_written():
    assert (PLOTS / "ensemble_roc.png").exists()
    assert (PLOTS / "ensemble_pr.png").exists()
