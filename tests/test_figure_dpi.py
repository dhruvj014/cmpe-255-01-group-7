"""Verify all paper figures are >=300 DPI."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

PAPER_FIGURES = [
    "L5_Classification/plots/ensemble_roc.png",
    "L5_Classification/plots/ensemble_pr.png",
    "L5_Classification/plots/error_heatmap.png",
    "L5_Classification/plots/calibration_reliability.png",
    "L6_Validation/plots/ablation_bar.png",
    "L6_Validation/plots/detection_rate_by_layer.png",
    "L3/plots/l3_score_distribution.png",
]


def test_paper_figures_300_dpi():
    bad = []
    for rel in PAPER_FIGURES:
        p = ROOT / rel
        if not p.exists():
            bad.append(f"{rel} missing")
            continue
        img = Image.open(p)
        dpi = img.info.get("dpi", (72, 72))
        # Allow tiny float-precision slop from matplotlib (writes ~299.9994)
        if min(dpi) < 299.5:
            bad.append(f"{rel} dpi={dpi}")
    assert not bad, "\n".join(bad)
