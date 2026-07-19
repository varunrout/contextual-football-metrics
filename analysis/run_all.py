"""
analysis/run_all.py
===================
Master runner — executes every analysis script in numbered order.

Usage
-----
    python analysis/run_all.py
    python analysis/run_all.py --stop-on-error   # (default behaviour)
    python analysis/run_all.py --continue-on-error

Output
------
  Prints script name, start time, duration, and exit code for each step.
  Exits with a non-zero code if any script fails (unless --continue-on-error).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "analysis/01_data_quality.py",
    "analysis/02_feature_stability.py",
    "analysis/03_univariate.py",
    "analysis/04_bivariate_cxg.py",
    "analysis/05_bivariate_cxa.py",
    "analysis/06_bivariate_cxt.py",
    "analysis/07_correlations.py",
    "analysis/08_eda_shots.py",
    "analysis/09_eda_sequences.py",
    "analysis/10_eda_opponents.py",
    "analysis/11_hypothesis_cxg.py",
    "analysis/12_hypothesis_cxa.py",
    "analysis/13_hypothesis_cxt.py",
    "analysis/14_statsbomb_baseline.py",
    "analysis/15_zone_xt_priors.py",
    "analysis/16_deep_eda.py",
]


def _run_script(script: str) -> tuple[int, float]:
    path = _ROOT / script
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(_ROOT),
        capture_output=False,
    )
    elapsed = time.time() - t0
    return result.returncode, elapsed


def main(continue_on_error: bool = False) -> int:
    total_start = time.time()
    failures: list[str] = []

    print(f"\n{'=' * 60}")
    print("  Analysis run_all.py")
    print(f"  {len(SCRIPTS)} scripts to execute")
    print(f"{'=' * 60}\n")

    for i, script in enumerate(SCRIPTS, 1):
        print(f"[{i:02d}/{len(SCRIPTS)}] {script}")
        t_start = time.strftime("%H:%M:%S")
        print(f"       started  {t_start}")

        code, elapsed = _run_script(script)
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"       {status}  ({elapsed:.1f}s)\n")

        if code != 0:
            failures.append(script)
            if not continue_on_error:
                print(f"Aborting — {script} failed with exit code {code}.")
                print("Re-run with --continue-on-error to skip failures.")
                break

    total = time.time() - total_start
    print(f"{'=' * 60}")
    print(f"  Total time: {total:.1f}s")
    if failures:
        print(f"  FAILED ({len(failures)}): {', '.join(failures)}")
    else:
        print("  All scripts completed successfully.")
    print(f"{'=' * 60}\n")

    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all pre-modelling analysis scripts.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running subsequent scripts even if one fails.",
    )
    args = parser.parse_args()
    sys.exit(main(continue_on_error=args.continue_on_error))
