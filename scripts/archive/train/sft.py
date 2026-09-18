"""Supervised Fine-Tuning (SFT) Runner for Compact Trajectory Student (Stage C).

Trains local Qwen3-VL-8B on distilled compact reasoning trajectories.
"""

import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Student SFT Runner.")
    parser.add_argument("--config", default="configs/experiments/sft_compact.yaml", help="Config file")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("      STAGE C: STUDENT SFT TRAINING RUNNER")
    print("=" * 60)
    print("Target: Qwen3-VL-8B-Instruct on Compact Trajectories")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
