"""Run the full Layer-6 validation pipeline.

Order:
    1) Cross-signal Jaccard stability
    2) Synthetic attack injection + per-layer detection
    3) Consolidated summary report
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

L6_DIR = Path(__file__).resolve().parent


def _run(script_name: str) -> None:
    script_path = L6_DIR / script_name
    print(f"\n{'=' * 70}")
    print(f"Running {script_path.name}")
    print(f"{'=' * 70}")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=L6_DIR)


def main() -> None:
    _run("01_jaccard_stability.py")
    _run("02_synthetic_injection.py")
    _run("03_summary_report.py")
    print("\nLayer-6 validation completed.")


if __name__ == "__main__":
    main()
