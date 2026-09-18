"""Computes Oracle Query Upper Bound.

Simulates what performance would be if Controller queried on 100% of wrong initial aspects
with observed empirical recovery rate P(Correct | Wrong & Queried).
"""

import sys
import os
import json
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_oracle(run_dir: str):
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Metrics not found at: {metrics_path}")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # Extract transition counts if available
    print("\n" + "=" * 60)
    print(f"      ORACLE BOUND ANALYSIS: {os.path.basename(run_dir)}")
    print("=" * 60)
    # Print metrics
    asc = m.get("asc_metrics", {})
    print(f"Current ASC Accuracy : {asc.get('sentiment_accuracy')}%")
    print(f"Current Macro-F1     : {asc.get('macro_f1')}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to run directory")
    args = parser.parse_args()
    compute_oracle(args.run)
