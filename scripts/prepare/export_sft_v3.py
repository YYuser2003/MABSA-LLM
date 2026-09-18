"""BACR-v3 SFT Data Exporter.

Extracts isolated, modular training datasets from BACR-v3 trajectories for student fine-tuning:
1. TEXT_ANCHOR:     (T, a) -> H_A
2. TEXT_RISK:       (H_A, H_B, a) -> R_T (zero visual cues)
3. CONTROLLER_ROUTE:(R_T, O_V, History, Budget) -> RouteDecision
4. EVIDENCE_FIREWALL:(q_V, a, E_raw) -> E_tilde (strictly fail-closed)
5. EVIDENCE_FUSION: (T, a, H_B, E_tilde) -> H_F
6. REVISION_VERIFY: (H_A, H_B, H_F, E_tilde) -> CVDecision

Usage:
    python scripts/prepare/export_sft_v3.py --input results/runs/<run_id>/trajectories.jsonl --output_dir data/sft/v3
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bacr.meta_controller import load_prompt


def export_sft_datasets(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load canonical system prompts
    sys_ta = load_prompt("text_anchor.md")
    sys_text_risk = load_prompt("controller_text_risk.md")
    sys_route = load_prompt("controller_router.md")
    sys_firewall = load_prompt("controller_evidence_firewall.md")
    sys_fusion = load_prompt("text_evidence_fusion.md")
    sys_verifier = load_prompt("controller_revision_verifier.md")

    files = {
        "text_anchor": open(os.path.join(output_dir, "sft_text_anchor.jsonl"), "w", encoding="utf-8"),
        "text_risk": open(os.path.join(output_dir, "sft_text_risk.jsonl"), "w", encoding="utf-8"),
        "controller_route": open(os.path.join(output_dir, "sft_controller_route.jsonl"), "w", encoding="utf-8"),
        "evidence_firewall": open(os.path.join(output_dir, "sft_evidence_firewall.jsonl"), "w", encoding="utf-8"),
        "evidence_fusion": open(os.path.join(output_dir, "sft_evidence_fusion.jsonl"), "w", encoding="utf-8"),
        "revision_verify": open(os.path.join(output_dir, "sft_revision_verify.jsonl"), "w", encoding="utf-8")
    }

    counts = {k: 0 for k in files}

    def write_sample(writer_key: str, sys_p: str, usr_p: str, assistant_response: Any):
        if not assistant_response:
            return
        ans_str = json.dumps(assistant_response, ensure_ascii=False, indent=2) if isinstance(assistant_response, (dict, list)) else str(assistant_response)
        entry = {
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
                {"role": "assistant", "content": ans_str}
            ]
        }
        files[writer_key].write(json.dumps(entry, ensure_ascii=False) + "\n")
        counts[writer_key] += 1

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except Exception:
                continue

            text = traj.get("text", "")
            target_asp = traj.get("target_aspect", {})
            target_text = target_asp.get("text", "") if isinstance(target_asp, dict) else str(target_asp)
            if not target_text and traj.get("gold_pairs"):
                target_text = traj["gold_pairs"][0][0]

            # 1. TEXT_ANCHOR SFT
            h_a = traj.get("text_anchor", {}).get("ledger", {})
            if h_a and text:
                usr_ta = f'Tweet Text:\n"{text}"\n\nTarget Aspects to evaluate: ["{target_text}"]'
                write_sample("text_anchor", sys_ta, usr_ta, h_a)

            # 2. EVIDENCE_FIREWALL, FUSION, and REVISION_VERIFY from rounds
            rounds = traj.get("rounds", [])
            for r in rounds:
                action = r.get("action")
                if action == "VISION_PROBE":
                    q = r.get("question", "")
                    raw_ev = r.get("raw_evidence")
                    filtered_ev = r.get("filtered_evidence")
                    if q and raw_ev and filtered_ev:
                        usr_fw = (
                            f'Dispatched Question: "{q}"\n'
                            f'Target Aspect: "{target_text}"\n\n'
                            f'Raw Visual Sensor Response (Et):\n{json.dumps(raw_ev, ensure_ascii=False, indent=2)}\n\n'
                            f'Filter speculative inferences, verify physical facts, assess revision support, and output in valid JSON.'
                        )
                        write_sample("evidence_firewall", sys_firewall, usr_fw, filtered_ev)

                    cand_fusion = r.get("candidate_fusion")
                    ev_bundle = r.get("evidence_bundle")
                    if cand_fusion and ev_bundle:
                        usr_fus = (
                            f'Tweet Text:\n"{text}"\n\n'
                            f'Verified Visual Proof from Firewall (E~):\n{json.dumps(ev_bundle, ensure_ascii=False, indent=2)}\n\n'
                            f'Synthesize verifiable facts into updated sentiment hypothesis in valid JSON.'
                        )
                        write_sample("evidence_fusion", sys_fusion, usr_fus, cand_fusion)

                    v_decision = r.get("verifier_decision")
                    if v_decision and cand_fusion:
                        usr_ver = (
                            f'Target Aspect: "{target_text}"\n\n'
                            f'Current Candidate Ledger:\n{json.dumps(cand_fusion, ensure_ascii=False, indent=2)}\n\n'
                            f'Verified Visual Proof:\n{json.dumps(ev_bundle or {}, ensure_ascii=False, indent=2)}\n\n'
                            f'Verify revision and output decision in valid JSON.'
                        )
                        write_sample("revision_verify", sys_verifier, usr_ver, v_decision)

            # 3. CONTROLLER_ROUTE from initial_contrast / risk_diagnosis
            init_c_d = traj.get("initial_contrast") or traj.get("risk_diagnosis", {})
            if init_c_d:
                t_risk = init_c_d.get("text_risk", {})
                v_opp = init_c_d.get("visual_opportunity", {})
                act = init_c_d.get("action", "FINALIZE")
                if t_risk and v_opp:
                    usr_route = (
                        f'Textual Risk Assessment (R_T):\n{json.dumps(t_risk, ensure_ascii=False, indent=2)}\n\n'
                        f'Visual Opportunity Assessment (O_V):\n{json.dumps(v_opp, ensure_ascii=False, indent=2)}\n\n'
                        f'Interaction History: None (Round 0)\n'
                        f'Remaining Budget: 2\n\n'
                        f'Select discrete action and output valid JSON.'
                    )
                    route_ans = {
                        "action": act,
                        "rationale": init_c_d.get("decision_reason", ""),
                        "query_type": init_c_d.get("query_type")
                    }
                    write_sample("controller_route", sys_route, usr_route, route_ans)

    for f in files.values():
        f.close()

    print(f"Successfully exported SFT datasets to {output_dir}:")
    for k, cnt in counts.items():
        print(f"  - {k}: {cnt} samples")


def main():
    parser = argparse.ArgumentParser(description="BACR-v3 SFT Data Exporter")
    parser.add_argument("--input", "-i", required=True, help="Input trajectories.jsonl path")
    parser.add_argument("--output_dir", "-o", default="data/sft/v3", help="Output directory for SFT jsonl files")
    args = parser.parse_args()

    export_sft_datasets(args.input, args.output_dir)


if __name__ == "__main__":
    main()
