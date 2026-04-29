"""Post-hoc calibration of L5 MLP scores.

Splits the holdout 50/50, fits Platt and isotonic calibration on one half,
evaluates Brier score + Expected Calibration Error on the other half.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "L5_Classification/outputs"
PLOTS = ROOT / "L5_Classification/plots"
PLOTS.mkdir(exist_ok=True)
RNG = 42

sup = pd.read_csv(OUT / "supervised_holdout_predictions.csv")
y = sup["y_true"].astype(int).values
s = sup["MLP_score"].astype(float).values

s_fit, s_eval, y_fit, y_eval = train_test_split(
    s, y, test_size=0.5, random_state=RNG, stratify=y
)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


lr = LogisticRegression(max_iter=2000, random_state=RNG)
lr.fit(logit(s_fit).reshape(-1, 1), y_fit)
platt = lr.predict_proba(logit(s_eval).reshape(-1, 1))[:, 1]

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(s_fit, y_fit)
iso_p = iso.predict(s_eval)


def ece(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        e += (mask.sum() / len(p)) * abs(conf - acc)
    return float(e)


metrics = {
    "uncalibrated": {"brier": float(brier_score_loss(y_eval, s_eval)), "ece": ece(y_eval, s_eval)},
    "platt":        {"brier": float(brier_score_loss(y_eval, platt)),  "ece": ece(y_eval, platt)},
    "isotonic":     {"brier": float(brier_score_loss(y_eval, iso_p)),  "ece": ece(y_eval, iso_p)},
}
(OUT / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2))


def bin_curve(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        xs.append(p[mask].mean())
        ys.append(y[mask].mean())
    return xs, ys


plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
for name, p, c, key in [
    ("Uncalibrated", s_eval, "C0", "uncalibrated"),
    ("Platt", platt, "C1", "platt"),
    ("Isotonic", iso_p, "C2", "isotonic"),
]:
    xs, ys = bin_curve(p, y_eval)
    plt.plot(xs, ys, "o-", color=c, label=f"{name} (Brier={metrics[key]['brier']:.3f})")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("L5 MLP Reliability Diagram")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "calibration_reliability.png", dpi=300)
plt.close()

print("Calibration metrics:")
print(json.dumps(metrics, indent=2))
