"""BACR-v3 Counterfactual Route Builder.

Performs offline counterfactual outcome analysis to derive optimal routing labels:
Given an episode (T, I, a, s*), determines:
- Outcome under FINALIZE:   c_fin (reward relative to H_A)
- Outcome under TEXT_RETHINK: c_text
- Outcome under VISION_PROBE: c_vis

Optimal Route Label Policy (Compute-Aware Pareto Optimal):
1. If H_A is already correct (c_fin == 1) and neither action is needed:
   Optimal Action = FINALIZE (conserve budget).
2. If H_A is incorrect (c_fin == 0) and TEXT_RETHINK rescues (c_text == 1):
   Optimal Action = TEXT_RETHINK (cost-effective text rescue).
3. If H_A is incorrect and VISION_PROBE rescues (c_vis == 1):
   Optimal Action = VISION_PROBE (grounded cross-modal rescue).
4. If neither action rescues or both cause harm:
   Optimal Action = FINALIZE (prevent futile / harmful computation).

Usage:
    python scripts/prepare/build_counterfactual_routes.py --trajectories results/runs/<run_id>/trajectories.jsonl --output data/sft/v3/counterfactual_routes.jsonl
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bacr.schemas_v3 import ControllerAction


def build_counterfactual_routes(traj_file: str, output_file: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    count = 0
    action_counts = {"FINALIZE": 0, "TEXT_RETHINK": 0, "VISION_PROBE": 0}

    with open(traj_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except Exception:
                continue

            gold_pairs = {p[0].lower(): p[1] for p in traj.get("gold_pairs", [])}
            target_asp = traj.get("target_aspect", {})
            target_text = target_asp.get("text", "") if isinstance(target_asp, dict) else str(target_asp)
            if not target_text and gold_pairs:
                target_text = list(gold_pairs.keys())[0]

            gold_sent = gold_pairs.get(target_text.lower())
            if not gold_sent:
                continue

            anchor_pairs = {p[0].lower(): p[1] for p in traj.get("text_anchor", {}).get("pairs", [])}
            s_anchor = anchor_pairs.get(target_text.lower(), "NEU")
            anchor_correct = (s_anchor == gold_sent)

            # Analyze transitions
            transitions = traj.get("transitions", [])
            text_rescue = False
            vision_rescue = False

            for t in transitions:
                act = t.get("action")
                cand_sent = t.get("candidate_sentiment") or t.get("post_sentiment")
                if act == "TEXT_RETHINK" and cand_sent == gold_sent and not anchor_correct:
                    text_rescue = True
                elif act == "VISION_PROBE" and cand_sent == gold_sent and not anchor_correct:
                    vision_rescue = True

            # Optimal route selection
            if anchor_correct:
                optimal_action = ControllerAction.FINALIZE.value
                rationale = "Anchor hypothesis is already correct; finalize to conserve compute."
            elif text_rescue:
                optimal_action = ControllerAction.TEXT_RETHINK.value
                rationale = "Text error is recoverable via linguistic re-deliberation without visual costs."
            elif vision_rescue:
                optimal_action = ControllerAction.VISION_PROBE.value
                rationale = "Missing affect or physical incongruity is verified and rescued by visual probe."
            else:
                optimal_action = ControllerAction.FINALIZE.value
                rationale = "Neither intervention rescues; finalize to prevent fruitless compute expenditure."

            action_counts[optimal_action] += 1

            record = {
                "sample_id": traj.get("sample_id", "unknown"),
                "target_aspect": target_text,
                "gold_sentiment": gold_sent,
                "anchor_sentiment": s_anchor,
                "optimal_action": optimal_action,
                "rationale": rationale,
                "text_risk": (traj.get("initial_contrast") or {}).get("text_risk", {}),
                "visual_opportunity": (traj.get("initial_contrast") or {}).get("visual_opportunity", {})
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Successfully generated {count} counterfactual route records to {output_file}:")
    for act, c in action_counts.items():
        pct = round(100.0 * c / max(1, count), 1)
        print(f"  - {act}: {c} ({pct}%)")


def main():
    parser = argparse.ArgumentParser(description="BACR-v3 Counterfactual Route Builder")
    parser.add_argument("--trajectories", "-t", required=True, help="Path to trajectories.jsonl")
    parser.add_argument("--output", "-o", default="data/sft/v3/counterfactual_routes.jsonl", help="Output path")
    args = parser.parse_args()

    build_counterfactual_routes(args.trajectories, args.output)


if __name__ == "__main__":
    main()
