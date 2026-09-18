"""BACR-v3 Master Orchestration Pipeline.

Implements the verified architecture:
HA, HB, V0 -> CD -> [TR -> CT | IP -> CE -> TF -> CV] <= B_deep -> Y_final

Where:
- HA: Immutable Historical Text Anchor (retained forever for attribution & RL reward baseline)
- HB: Current Verified Text Baseline (mutable; updated when TR passes CT audit)
- V0: Global Visual Sensor Opportunity Map (strictly Control Plane only, IG not-> T)
- CD: Risk-Aware Meta-Controller diagnosing failure modes and allocating discrete action A_C in {FINALIZE, TEXT_RETHINK, VISION_PROBE}
- TR: Text Re-Deliberator resolving linguistic/pragmatic ambiguity without image exposure
- CT: Text Revision Verifier auditing TR to prevent ungrounded linguistic drift (gates HB <- H_RT)
- IP: Targeted Deep Visual Probe answering atomic physical inquiries
- CE: Evidence Firewall sanitizing physical facts and assessing revision support
- TF: Evidence Fusion Reasoner updating sentiment hypothesis based on verified visual proof
- CV: Revision Verifier enforcing Text Baseline protection with REVERT_TEXT_BASELINE safeguard rule
- S_t: Aspect-Level State Space tracking each aspect independently to prevent cross-aspect contamination
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from bacr.client import BaseClient, GeminiClient
from bacr.text_reasoner import TextReasoner
from bacr.meta_controller import MetaController
from bacr.vision_sensor import VisionSensor
from bacr.schemas_v3 import (
    validate_structured_ledger,
    validate_risk_diagnosis,
    validate_evidence_firewall,
    validate_text_revision_audit,
    validate_revision_verifier
)


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class BACRPipelineV3:
    def __init__(
        self,
        client: Optional[BaseClient] = None,
        run_id: str = "bacr_v3_experiment",
        max_visual_probes: int = 2
    ):
        self.client = client or GeminiClient(allow_fallback=False)
        self.run_id = run_id
        self.max_visual_probes = max_visual_probes

        self.text_reasoner = TextReasoner(self.client)
        self.meta_controller = MetaController(self.client)
        self.vision_sensor = VisionSensor(self.client)

    def run_sample(
        self,
        sample: Dict[str, Any],
        image_base_dir: str,
        max_visual_probes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Runs the complete BACR-v3 protocol for a single sample."""
        sample_id = sample.get("sample_id", "unknown")
        text = sample.get("text", "")
        rel_img_path = sample.get("image", "")
        full_img_path = os.path.join(image_base_dir, rel_img_path)
        gold_pairs = sample.get("pairs", [])
        image_hash = compute_file_sha256(full_img_path)

        k_max = self.max_visual_probes if max_visual_probes is None else max_visual_probes

        # Compute tracking counters
        compute = {
            "api_calls_total": 0,
            "text_calls": 0,
            "vision_calls": 0,
            "controller_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "image_invocations": 0,
            "latency_ms": 0.0
        }

        def record_usage(usage: Dict[str, int], lat: float, call_type: str):
            compute["api_calls_total"] += 1
            compute["input_tokens"] += usage.get("input_tokens", 0)
            compute["output_tokens"] += usage.get("output_tokens", 0)
            compute["total_tokens"] += usage.get("total_tokens", 0)
            compute["latency_ms"] += lat
            if call_type == "text":
                compute["text_calls"] += 1
            elif call_type == "vision":
                compute["vision_calls"] += 1
                compute["image_invocations"] += 1
            elif call_type == "controller":
                compute["controller_calls"] += 1

        # ====================================================================
        # Stage 0: TA Text Anchor Reasoner (H_A)
        # H_A is immutable and retained permanently for attribution and RL baseline
        # ====================================================================
        target_aspects = sample.get("aspects")
        cached_t0 = sample.get("text_initial_cached") or sample.get("canonical_t0")

        h_a, u_ta, l_ta = self.text_reasoner.generate_anchor(
            text=text,
            target_aspects=target_aspects,
            cached_t0=cached_t0
        )
        if u_ta.get("total_tokens", 0) > 0:
            record_usage(u_ta, l_ta, "text")

        y_a = h_a.get("pairs", [])

        # Initialize H_B (Current Verified Text Baseline)
        h_b = json.loads(json.dumps(h_a))
        y_b = list(y_a)

        # Initialize Aspect-Level State Space S_t = {S_t^{a_1}, ..., S_t^{a_M}}
        aspect_states: Dict[str, Dict[str, Any]] = {}
        for a in h_a.get("aspects", []):
            aid = a.get("aspect_id", "a_01")
            aspect_states[aid] = {
                "aspect_id": aid,
                "text": a.get("text", ""),
                "span": a.get("span", [0, 0]),
                "anchor_sentiment": a.get("sentiment", "NEU"),
                "baseline_sentiment": a.get("sentiment", "NEU"),
                "current_sentiment": a.get("sentiment", "NEU"),
                "risks": a.get("risks", []),
                "status": "ACTIVE"
            }

        # ====================================================================
        # Stage 1: IG Global Visual Sensor (V_0: Opportunity Map ONLY)
        # CRITICAL INVARIANT: V_0 is sent ONLY to C_D, NEVER into T!
        # ====================================================================
        cached_v0 = sample.get("image_initial_cached") or sample.get("canonical_v0")
        if cached_v0:
            if isinstance(cached_v0, dict) and "image_initial" in cached_v0:
                v_0 = cached_v0["image_initial"]
            else:
                v_0 = cached_v0
        else:
            v_0, u_ig, l_ig = self.vision_sensor.perceive_global(full_img_path)
            record_usage(u_ig, l_ig, "vision")

        # Working variables & deep budget tracking
        budget_deep = k_max
        rounds_record: List[Dict[str, Any]] = []
        stage_predictions: Dict[str, Any] = {
            "Y_T": y_a,      # Immutable Anchor
            "Y_HB": y_b,     # Verified Text Baseline
            "Y_TV": y_b,     # Pre-audit candidate
            "Y_probes": [],
            "Y_final": []
        }
        initial_c_d: Optional[Dict[str, Any]] = None

        # ====================================================================
        # Deep Controller Loop (Budget B_deep <= 2)
        # ====================================================================
        step_counter = 0
        while budget_deep > 0:
            step_counter += 1
            # CD: Risk-Aware Diagnosis & Compute Allocation
            c_d, u_cd, l_cd = self.meta_controller.diagnose_risk(
                h_a=h_a,
                v0=v_0,
                h_b=h_b,
                aspect_states=aspect_states,
                history=rounds_record,
                budget=budget_deep
            )
            record_usage(u_cd, l_cd, "controller")
            if initial_c_d is None:
                initial_c_d = c_d

            cd_action = c_d.get("action", "FINALIZE")
            target_aid = c_d.get("target_aspect_id")

            # Routing
            if cd_action == "FINALIZE" or budget_deep <= 0:
                break

            elif cd_action == "TEXT_RETHINK":
                budget_deep -= 1
                critique = c_d.get("critique_for_text") or c_d.get("decision_reason", "Re-examine linguistic context.")

                # TR: Text Re-deliberation on current text baseline H_B
                h_rethink, u_tr, l_tr = self.text_reasoner.rethink_text(
                    text=text,
                    h_baseline=h_b,
                    critique=critique,
                    target_aspect_id=target_aid
                )
                record_usage(u_tr, l_tr, "text")
                y_rethink = h_rethink.get("pairs", [])
                stage_predictions["Y_probes"].append(y_rethink)
                stage_predictions["Y_TV"] = y_rethink

                # CT: Text Revision Verifier (Prevent overthinking drift)
                c_t, u_ct, l_ct = self.meta_controller.audit_text_revision(
                    h_b=h_b,
                    h_rethink=h_rethink,
                    critique=critique,
                    target_aspect_id=target_aid
                )
                record_usage(u_ct, l_ct, "controller")
                ct_decision = c_t.get("decision", "ACCEPT_TEXT_REVISION")

                round_entry = {
                    "round": step_counter,
                    "type": "TEXT_RETHINK",
                    "step_ref": f"text_rethink_{step_counter}",
                    "target_aspect_id": target_aid,
                    "critique": critique,
                    "revised_ledger": h_rethink,
                    "text_verifier_decision": c_t,
                    "verifier_decision": c_t,  # For backward-compatible evaluator
                    "pairs": y_rethink
                }
                rounds_record.append(round_entry)

                if ct_decision == "ACCEPT_TEXT_REVISION":
                    # Update Verified Text Baseline H_B
                    h_b = json.loads(json.dumps(h_rethink))
                    stage_predictions["Y_HB"] = h_b.get("pairs", [])
                    if target_aid and target_aid in aspect_states:
                        new_sent = next((a["sentiment"] for a in h_b["aspects"] if a.get("aspect_id") == target_aid), "NEU")
                        aspect_states[target_aid]["baseline_sentiment"] = new_sent
                        aspect_states[target_aid]["current_sentiment"] = new_sent
                        aspect_states[target_aid]["status"] = "VERIFIED"
                elif ct_decision == "REVERT_TEXT_BASELINE":
                    # Keep H_B unchanged
                    pass
                elif ct_decision == "VISION_PROBE":
                    # Text audit suggests visual inspection is required
                    pass

                # After text rethink, continue loop to let CD decide next action or finalize
                continue

            elif cd_action == "VISION_PROBE":
                budget_deep -= 1
                question = c_d.get("question_for_vision") or c_d.get("question")
                if not question:
                    break

                step_ref = f"vision_probe_{step_counter}"
                target_asp_obj = next((a for a in h_b.get("aspects", []) if a.get("aspect_id") == target_aid), None)
                target_asp_text = target_asp_obj["text"] if target_asp_obj else ""

                # 1. IP: Targeted Deep Visual Probe
                raw_evidence, u_ip, l_ip = self.vision_sensor.probe_deep(
                    image_path=full_img_path,
                    question=question
                )
                record_usage(u_ip, l_ip, "vision")

                # 2. CE: Evidence Firewall (Sanitizes & assesses revision_support)
                filtered_evidence, u_ce, l_ce = self.meta_controller.filter_evidence_firewall(
                    question=question,
                    raw_evidence=raw_evidence,
                    target_aspect_id=target_aid,
                    target_aspect_text=target_asp_text
                )
                record_usage(u_ce, l_ce, "controller")

                # 3. TF: Evidence Fusion Reasoner on current text baseline H_B
                h_fusion, u_tf, l_tf = self.text_reasoner.fuse_evidence(
                    text=text,
                    h_baseline=h_b,
                    verified_evidence=filtered_evidence,
                    step_ref=step_ref,
                    target_aspect_id=target_aid
                )
                record_usage(u_tf, l_tf, "text")
                y_fusion = h_fusion.get("pairs", [])
                stage_predictions["Y_probes"].append(y_fusion)
                stage_predictions["Y_TV"] = y_fusion

                # 4. CV: Revision Verifier (Enforcing REVERT_TEXT_BASELINE Safeguard)
                c_v, u_cv, l_cv = self.meta_controller.verify_revision(
                    h_a=h_a,
                    h_current=h_fusion,
                    verified_evidence=filtered_evidence,
                    h_b=h_b,
                    budget=budget_deep
                )
                record_usage(u_cv, l_cv, "controller")
                cv_decision = c_v.get("decision", "REVERT_TEXT_BASELINE")

                round_entry = {
                    "round": step_counter,
                    "type": "VISION_PROBE",
                    "step_ref": step_ref,
                    "target_aspect_id": target_aid,
                    "target_aspect_text": target_asp_text,
                    "question": question,
                    "raw_visual_evidence": raw_evidence,
                    "filtered_evidence": filtered_evidence,
                    "text_fusion": h_fusion,
                    "verifier_decision": c_v,
                    "pairs": y_fusion
                }
                rounds_record.append(round_entry)

                if cv_decision == "ACCEPT_REVISION":
                    # Visual revision conclusively justified: update active baseline
                    h_b = json.loads(json.dumps(h_fusion))
                    stage_predictions["Y_HB"] = h_b.get("pairs", [])
                    if target_aid and target_aid in aspect_states:
                        new_sent = next((a["sentiment"] for a in h_b["aspects"] if a.get("aspect_id") == target_aid), "NEU")
                        aspect_states[target_aid]["baseline_sentiment"] = new_sent
                        aspect_states[target_aid]["current_sentiment"] = new_sent
                        aspect_states[target_aid]["status"] = "VERIFIED"
                    break
                elif cv_decision in ["REVERT_TEXT_BASELINE", "REVERT_ANCHOR"]:
                    # Safeguard: retain verified text baseline H_B
                    break
                elif cv_decision == "QUERY_AGAIN" and budget_deep > 0:
                    h_b = json.loads(json.dumps(h_fusion))
                    continue
                else:
                    # Default safeguard fallback
                    break

        y_final = h_b.get("pairs", [])
        stage_predictions["Y_final"] = y_final
        compute["latency_ms"] = round(compute["latency_ms"], 1)

        num_probes = len([r for r in rounds_record if r.get("type") == "VISION_PROBE"])
        risk_type = (initial_c_d or {}).get("risk_diagnosis", {}).get("risk_type", "NO_RISK")
        main_action = (initial_c_d or {}).get("action", "FINALIZE")

        return {
            "sample_id": sample_id,
            "text": text,
            "image": rel_img_path,
            "gold_pairs": gold_pairs,
            "contrast_type": risk_type,
            "action": main_action,
            "text_anchor": {
                "ledger": h_a,
                "pairs": y_a
            },
            "text_baseline": {
                "ledger": h_b,
                "pairs": y_final
            },
            "text_only": {
                "ledger": h_a,
                "pairs": y_a
            },
            "vision_opportunity_map": v_0,
            "vision_initial": v_0,
            "text_visual_global": {
                "ledger": h_b,
                "pairs": stage_predictions["Y_TV"]
            },
            "initial_contrast": initial_c_d or {},
            "risk_diagnosis": initial_c_d or {},
            "aspect_states": aspect_states,
            "rounds": rounds_record,
            "final_ledger": h_b,
            "final_pairs": y_final,
            "stage_predictions": stage_predictions,
            "num_visual_probes": num_probes,
            "compute": compute
        }
