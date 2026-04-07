"""Run the full Layer-5 pipeline.

Order:
    1) Build fused feature table (L1 + L2 + L4 signals)
    2) Train supervised models (Decision Tree / RF / SVM)
    3) Train anomaly models (Isolation Forest / LOF)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

L5_DIR = Path(__file__).resolve().parent


def _run(script_name: str) -> None:
    script_path = L5_DIR / script_name
    print(f"\n{'=' * 70}")
    print(f"Running {script_path.name}")
    print(f"{'=' * 70}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    _run("01_build_feature_table.py")
    _run("02_train_models.py")
    _run("03_anomaly_detection.py")
    print("\nLayer-5 pipeline completed.")


if __name__ == "__main__":
    main()
