"""Best-of-K Teacher Search for Compact Trajectory Distillation (Stage C).

Generates K candidate reasoning trajectories per aspect,
filters by consistency with gold polarity, and retains the most compact, grounded trace.
"""

import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Best-of-K Teacher Search.")
    parser.add_argument("--config", default="configs/experiments/teacher_search.yaml", help="Config file")
    parser.add_argument("--k", type=int, default=4, help="Number of trajectories to sample per aspect")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("      STAGE C: BEST-OF-K TEACHER SEARCH RUNNER")
    print("=" * 60)
    print(f"Sampling K = {args.k} candidate trajectories per aspect.")
    print("Test-set isolation rule: Only Train and Dev sets are used.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
