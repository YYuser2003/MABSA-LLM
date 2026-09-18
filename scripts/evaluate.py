"""Unified Offline Evaluation Suite for BACR / MABSA-LLM.

Usage:
  python scripts/evaluate.py --run outputs/runs/20260918_bacr_twitter2015_test
  python scripts/evaluate.py --pred outputs/runs/xxx/trajectories.jsonl --gold data/processed/twitter2015_test.jsonl
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bacr.evaluator import BACREvaluator, print_evaluation_report


def evaluate_run(run_dir: str):
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = os.path.join(run_dir, "config.yaml")
    traj_path = os.path.join(run_dir, "trajectories.jsonl")

    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectories file not found at: {traj_path}")

    # Determine gold path from config or default
    gold_path = None
    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            ds = cfg.get("dataset", {}).get("dataset", {})
            split = cfg.get("cli_args", {}).get("split") or "test"
            gold_rel = ds.get("splits", {}).get(split)
            if gold_rel:
                gold_path = os.path.join(PROJECT_ROOT, gold_rel)

    if not gold_path or not os.path.exists(gold_path):
        # Infer from run directory name
        if "twitter2017" in run_dir or "tw17" in run_dir:
            gold_path = os.path.join(PROJECT_ROOT, "data", "processed", "twitter2017_test.jsonl")
        else:
            gold_path = os.path.join(PROJECT_ROOT, "data", "processed", "twitter2015_test.jsonl")

    print(f"\n[EVALUATION] Evaluating run: {run_dir}")
    print(f"[EVALUATION] Using gold file: {gold_path}")

    evaluator = BACREvaluator(gold_file=gold_path)
    metrics = evaluator.evaluate_predictions(pred_file=traj_path)
    print_evaluation_report(metrics)

    out_metrics = os.path.join(run_dir, "metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved updated metrics to {out_metrics}\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Offline Evaluator.")
    parser.add_argument("--run", type=str, help="Path to run directory.")
    parser.add_argument("--pred", type=str, help="Path to predictions/trajectories JSONL.")
    parser.add_argument("--gold", type=str, help="Path to gold dataset JSONL.")
    args = parser.parse_args()

    if args.run:
        evaluate_run(args.run)
    elif args.pred and args.gold:
        evaluator = BACREvaluator(gold_file=args.gold)
        metrics = evaluator.evaluate_predictions(pred_file=args.pred)
        print_evaluation_report(metrics)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
