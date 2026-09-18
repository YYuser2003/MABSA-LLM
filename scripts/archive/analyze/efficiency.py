"""Computes Test-Time Compute Efficiency.

Metrics:
- Macro-F1 / 1K Tokens
- Macro-F1 / API Call
- Macro-F1 / Image Invocation
- Average Latency per Aspect
"""

import sys
import os
import json
import argparse


def compute_efficiency(run_dir: str):
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Metrics not found: {metrics_path}")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    cmp = m.get("compute_metrics", {})
    print("\n" + "=" * 60)
    print(f"      COMPUTE EFFICIENCY AUDIT: {os.path.basename(run_dir)}")
    print("=" * 60)
    print(f"Total API Calls       : {cmp.get('api_calls_total', 0)}")
    print(f"Avg Calls / Sample    : {cmp.get('avg_api_calls_per_sample', 0)}")
    print(f"Avg Total Tokens      : {cmp.get('avg_total_tokens_per_sample', 0)}")
    print(f"Avg Latency (ms)      : {cmp.get('avg_latency_ms', 0)}")
    print(f"F1 / API Call         : {cmp.get('f1_per_api_call', 0)}")
    print(f"F1 / 1K Tokens        : {cmp.get('f1_per_1k_tokens', 0)}")
    print(f"F1 / Image Call       : {cmp.get('f1_per_image_invocation', 0)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to run directory")
    args = parser.parse_args()
    compute_efficiency(args.run)
