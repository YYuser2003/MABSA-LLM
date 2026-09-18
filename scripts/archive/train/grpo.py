"""Group Relative Policy Optimization (GRPO) Runner for BACR Controller (Stage D).

Optimizes Controller gating policy pi_D and probing policy pi_Q
with decoupled accuracy rewards, trigger penalties, and consistency constraints.
"""

import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="GRPO Controller RL Runner.")
    parser.add_argument("--config", default="configs/experiments/grpo_controller.yaml", help="Config file")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("      STAGE D: GRPO CONTROLLER RL RUNNER")
    print("=" * 60)
    print("Algorithm: Decoupled 2-Stage GRPO with Aspect-Locked Reward")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
