from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "L6_Validation/outputs"
PLOTS = ROOT / "L6_Validation/plots"

EXPECTED_CONFIGS = {
    "L2-only",
    "L4-only",
    "L5-supervised-only",
    "L5-anomaly-only",
    "L2+L4",
    "L2+L4+L5 (full behavioral)",
    "Full + L3",
}


def test_ablation_csv_written():
    df = pd.read_csv(OUT / "ablation_table.csv")
    assert set(df["config"]) == EXPECTED_CONFIGS
    for col in ["auc_roc", "f1_optimal", "avg_precision", "recall_at_fpr_0_1"]:
        assert col in df.columns
    # Every row now has real numbers (no <TBD> after L3 integration)
    l3_row = df[df["config"] == "Full + L3"].iloc[0]
    assert isinstance(l3_row["auc_roc"], float)
    assert 0 <= l3_row["auc_roc"] <= 1


def test_ablation_tex_written():
    tex = (OUT / "ablation_table.tex").read_text()
    assert r"\begin{tabular}" in tex
    assert "Full + L3" in tex


def test_ablation_plot_written():
    assert (PLOTS / "ablation_bar.png").exists()
