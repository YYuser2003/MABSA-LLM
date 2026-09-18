"""BACR-v3 Pipeline Orchestrator.

Architecture:
  H_A (Text Anchor) || I_G (Visual Sensor)
         --> C_D (Risk Diagnosis, receives policy_history ONLY)
         --> [T_R -> C_T]  (Text Re-deliberation, C_Q^T has NO visual access)
             or [I_P -> C_E -> T_F -> C_V] (Strictly Fail-Closed Evidence Firewall)
         --> Y_final

Core Invariant:
  NO VERIFIER APPROVAL, NO BASELINE MUTATION.
  H_B is mutated ONLY when C_T audits ACCEPT_TEXT_REVISION or C_V audits ACCEPT_REVISION.
  QUERY_AGAIN NEVER mutates H_B; instead it accumulates verified evidence and retries.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

from bacr.client import GeminiClient
from bacr.text_reasoner import TextReasoner
from bacr.vision_sensor import VisionSensor
from bacr.meta_controller import MetaController
from bacr.schemas_v3 import (
    ControllerAction,
    RiskDecision,
    EvidenceFirewallOutput,
    TextRevisionAuditOutput,
    CTDecision,
    RevisionVerifierOutput,
    CVDecision,
    TransitionRecord
)


class BACRPipelineV3:
    def __init__(
        self,
        client: GeminiClient,
        run_id: Optional[str] = None,
        max_visual_probes: int = 2
    ):
        self.client = client
        self.run_id = run_id or f"bacr_v3_{int(time.time())}"
        self.max_deep_actions = max_visual_probes

        # Modular Agentic Roles
        self.text_reasoner = TextReasoner(client)
        self.vision_sensor = VisionSensor(client)
        self.meta_controller = MetaController(client)

    def run_sample(
        self,
        sample: Dict[str, Any],
        image_base_dir: str = "",
        max_visual_probes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Executes full BACR-v3 inference loop for a single multimodal tweet sample."""
        sample_id = sample.get("sample_id", "unknown")
        text = sample.get("text", "")
        rel_img_path = sample.get("image", "")
        full_img_path = os.path.join(image_base_dir, rel_img_path) if image_base_dir else rel_img_path

        gold_pairs = sample.get("pairs", [])
        gold_aspects = [p[0] for p in gold_pairs]

        k_max = max_visual_probes if max_visual_probes is not None else self.max_deep_actions

        # Compute & latency tracking
        compute = {
            "api_calls_text": 0,
            "api_calls_vision": 0,
            "api_calls_controller": 0,
            "api_calls_total": 0,
            "total_tokens": 0,
            "image_invocations": 0,
            "latency_ms": 0.0
        }

        def record_usage(usage: Dict[str, int], lat: float, role: str):
            compute["api_calls_total"] += 1
            compute["latency_ms"] += lat
            compute["total_tokens"] += usage.get("total_tokens", 0)
            if role == "text":
                compute["api_calls_text"] += 1
            elif role == "vision":
                compute["api_calls_vision"] += 1
                compute["image_invocations"] += 1
            elif role == "controller":
                compute["api_calls_controller"] += 1

        # ====================================================================
        # Phase 1: Text Anchor Reasoner (TA) -> Immutable H_A
        # ====================================================================
        cached_t0 = sample.get("text_initial_cached")
        h_a, u_ta, l_ta = self.text_reasoner.generate_anchor(
            text=text,
            target_aspects=gold_aspects,
            cached_t0=cached_t0
        )
        record_usage(u_ta, l_ta, "text")
        y_a = h_a.get("pairs", [])

        # Initialize Verified Text Baseline H_B identical to H_A at step 0
        h_b = json.loads(json.dumps(h_a))
        y_b = list(y_a)

        # Initialize Aspect-Level State Space S_t^a
        aspect_states: Dict[str, Any] = {}
        for a in h_a.get("aspects", []):
            aid = a.get("aspect_id", f"a_{len(aspect_states)+1:02d}")
            sent = a.get("sentiment", "NEU")
            aspect_states[aid] = {
                "aspect_id": aid,
                "text": a.get("text", ""),
                "span": a.get("span", [0, 0]),
                "anchor_sentiment": sent,
                "baseline_sentiment": sent,
                "current_candidate": None,
                "accepted_evidence": [],
                "risks": a.get("risks", []),
                "action_history": [],
                "status": "ACTIVE"
            }

        # ====================================================================
        # Phase 2: Global Visual Sensor (IG) -> V_0 (Control Plane Only)
        # ====================================================================
        cached_v0 = sample.get("image_initial_cached")
        if cached_v0:
            if isinstance(cached_v0, dict) and "image_initial" in cached_v0:
                v_0 = cached_v0["image_initial"]
            else:
                v_0 = cached_v0
        else:
            v_0, u_ig, l_ig = self.vision_sensor.perceive_global(full_img_path)
            record_usage(u_ig, l_ig, "vision")

        # Working state & histories
        budget_deep = k_max
        audit_history: List[Dict[str, Any]] = []    # Full logs (raw evidence included)
        policy_history: List[Dict[str, Any]] = []   # Sanitized history passed to C_D
        transitions: List[Dict[str, Any]] = []      # Standard RL transition records

        initial_c_d: Optional[Dict[str, Any]] = None
        step_counter = 0

        # Helper to execute a visual probe routine
        def execute_visual_probe(
            target_aid: Optional[str],
            probe_question: str,
            current_hb: Dict[str, Any],
            budget_left: int
        ) -> Tuple[Dict[str, Any], str, List[str], int]:
            """Performs IP -> CE -> TF -> CV loop with multi-round QUERY_AGAIN evidence accumulation.
            Strict Invariant: Does NOT mutate current_hb until CV accepts.
            Each probe iteration consumes 1 budget and logs a separate transition.
            """
            nonlocal step_counter
            target_asp_obj = next((a for a in current_hb.get("aspects", []) if a.get("aspect_id") == target_aid), None)
            target_asp_text = target_asp_obj["text"] if target_asp_obj else ""
            pre_sent = target_asp_obj.get("sentiment", "NEU") if target_asp_obj else "NEU"

            accumulated_evidence: List[str] = []
            final_cv_decision = "REVERT_TEXT_BASELINE"
            final_h_fusion = json.loads(json.dumps(current_hb))

            current_q = probe_question
            while budget_left > 0 and current_q:
                budget_left -= 1
                step_counter += 1
                step_ref = f"vision_probe_{step_counter}"

                # 1. IP: Targeted Deep Visual Probe
                raw_ev, u_ip, l_ip = self.vision_sensor.probe_deep(
                    image_path=full_img_path,
                    question=current_q
                )
                record_usage(u_ip, l_ip, "vision")

                # 2. CE: Strictly Fail-Closed Evidence Firewall
                filtered_ev, u_ce, l_ce = self.meta_controller.filter_evidence_firewall(
                    question=current_q,
                    raw_evidence=raw_ev,
                    target_aspect_id=target_aid,
                    target_aspect_text=target_asp_text
                )
                record_usage(u_ce, l_ce, "controller")

                for fact in filtered_ev.get("usable_evidence", []):
                    if fact and fact not in accumulated_evidence:
                        accumulated_evidence.append(fact)

                combined_evidence_obj = dict(filtered_ev)
                combined_evidence_obj["usable_evidence"] = accumulated_evidence

                # 3. TF: Evidence Fusion Reasoner on immutable current_hb
                h_fusion, u_tf, l_tf = self.text_reasoner.fuse_evidence(
                    text=text,
                    h_baseline=current_hb,
                    verified_evidence=combined_evidence_obj,
                    step_ref=step_ref,
                    target_aspect_id=target_aid
                )
                record_usage(u_tf, l_tf, "text")
                final_h_fusion = h_fusion

                # 4. CV: Revision Verifier (Enforces REVERT_TEXT_BASELINE Safeguard)
                c_v, u_cv, l_cv = self.meta_controller.verify_revision(
                    h_a=h_a,
                    h_current=h_fusion,
                    verified_evidence=combined_evidence_obj,
                    h_b=current_hb,
                    budget=budget_left
                )
                record_usage(u_cv, l_cv, "controller")
                cv_dec = c_v.get("decision", "REVERT_TEXT_BASELINE")
                final_cv_decision = cv_dec

                cand_sent = next((a["sentiment"] for a in h_fusion.get("aspects", []) if a.get("aspect_id") == target_aid), pre_sent)
                post_sent = cand_sent if cv_dec == "ACCEPT_REVISION" else pre_sent

                # Log each turn as an explicit transition
                transitions.append(TransitionRecord(
                    sample_id=sample_id,
                    aspect_id=target_aid,
                    step=step_counter,
                    pre_sentiment=pre_sent,
                    action="VISION_PROBE",
                    critique_or_question=current_q,
                    observation={"sanitized_evidence": filtered_ev.get("usable_evidence", [])},
                    candidate_sentiment=cand_sent,
                    verifier_decision=cv_dec,
                    post_sentiment=post_sent,
                    done=(budget_left <= 0)
                ).model_dump())

                audit_history.append({
                    "step": step_counter,
                    "action": "VISION_PROBE",
                    "target_aspect_id": target_aid,
                    "question": current_q,
                    "raw_evidence": raw_ev,
                    "filtered_evidence": filtered_ev,
                    "candidate_fusion": h_fusion,
                    "verifier_decision": c_v,
                    "baseline_mutated": (cv_dec == "ACCEPT_REVISION"),
                    "pairs": final_h_fusion.get("pairs", []) if cv_dec == "ACCEPT_REVISION" else current_hb.get("pairs", [])
                })

                policy_history.append({
                    "step": step_counter,
                    "action": "VISION_PROBE",
                    "target_aspect_id": target_aid,
                    "sanitized_evidence": filtered_ev.get("usable_evidence", []),
                    "verifier_decision": cv_dec,
                    "baseline_mutated": (cv_dec == "ACCEPT_REVISION")
                })

                if cv_dec == "QUERY_AGAIN" and budget_left > 0:
                    current_q = c_v.get("next_question") or ""
                else:
                    break

            return final_h_fusion, final_cv_decision, accumulated_evidence, budget_left

        # ====================================================================
        # Deep Controller Loop (Budget B_deep <= 2)
        # ====================================================================
        while budget_deep > 0:
            # CD: Risk-Aware Diagnosis & Compute Allocation (Observes policy_history ONLY)
            c_d, u_cd, l_cd = self.meta_controller.diagnose_risk(
                h_a=h_a,
                v0=v_0,
                h_b=h_b,
                aspect_states=aspect_states,
                history=policy_history,
                budget=budget_deep
            )
            record_usage(u_cd, l_cd, "controller")
            if initial_c_d is None:
                initial_c_d = c_d

            cd_action = c_d.get("action", "FINALIZE")
            target_aid = c_d.get("target_aspect_id")
            target_asp_obj = next((a for a in h_b.get("aspects", []) if a.get("aspect_id") == target_aid), None)
            target_asp_text = target_asp_obj["text"] if target_asp_obj else ""
            pre_sentiment = target_asp_obj.get("sentiment", "NEU") if target_asp_obj else "NEU"

            if cd_action == "FINALIZE" or budget_deep <= 0:
                break

            elif cd_action == "TEXT_RETHINK":
                budget_deep -= 1
                step_counter += 1

                # Physical Decoupling of Critique: C_Q^T receives ONLY H_B and diagnosed risk; V0 is physically excluded!
                risk_info = c_d.get("risk_diagnosis", {})
                critique, u_cq, l_cq = self.meta_controller.generate_text_critique(
                    h_b=h_b,
                    risk_diagnosis=risk_info,
                    target_aspect_id=target_aid,
                    target_aspect_text=target_asp_text
                )
                record_usage(u_cq, l_cq, "controller")

                # TR: Text Re-deliberation on current verified text baseline H_B
                h_rethink, u_tr, l_tr = self.text_reasoner.rethink_text(
                    text=text,
                    h_baseline=h_b,
                    critique=critique,
                    target_aspect_id=target_aid
                )
                record_usage(u_tr, l_tr, "text")
                candidate_sent = next((a["sentiment"] for a in h_rethink.get("aspects", []) if a.get("aspect_id") == target_aid), pre_sentiment)

                # CT: Text Revision Verifier (Prevents overthinking drift)
                c_t, u_ct, l_ct = self.meta_controller.audit_text_revision(
                    h_b=h_b,
                    h_rethink=h_rethink,
                    critique=critique,
                    target_aspect_id=target_aid
                )
                record_usage(u_ct, l_ct, "controller")
                ct_decision = c_t.get("decision", "REVERT_TEXT_BASELINE")

                post_sentiment = pre_sentiment
                baseline_mutated = False

                if ct_decision == "ACCEPT_TEXT_REVISION":
                    h_b = json.loads(json.dumps(h_rethink))
                    post_sentiment = candidate_sent
                    baseline_mutated = True
                    if target_aid and target_aid in aspect_states:
                        aspect_states[target_aid]["baseline_sentiment"] = post_sentiment
                        aspect_states[target_aid]["current_candidate"] = None
                        aspect_states[target_aid]["status"] = "VERIFIED"
                        aspect_states[target_aid]["action_history"].append("TEXT_RETHINK:ACCEPT")

                elif ct_decision == "ESCALATE_TO_VISION" and budget_deep > 0:
                    v_gap = c_t.get("visual_gap") or f"Check physical visual evidence for {target_asp_text}"
                    h_fusion, cv_dec, acc_ev, budget_deep = execute_visual_probe(
                        target_aid=target_aid,
                        probe_question=v_gap,
                        current_hb=h_b,
                        budget_left=budget_deep
                    )
                    if cv_dec == "ACCEPT_REVISION":
                        h_b = json.loads(json.dumps(h_fusion))
                        cand_v_sent = next((a["sentiment"] for a in h_fusion.get("aspects", []) if a.get("aspect_id") == target_aid), pre_sentiment)
                        if target_aid and target_aid in aspect_states:
                            aspect_states[target_aid]["baseline_sentiment"] = cand_v_sent
                            aspect_states[target_aid]["status"] = "VERIFIED"
                            aspect_states[target_aid]["action_history"].append("ESCALATE_TO_VISION:ACCEPT")

                else:
                    if target_aid and target_aid in aspect_states:
                        aspect_states[target_aid]["action_history"].append("TEXT_RETHINK:REVERT")

                # Log to histories
                audit_record = {
                    "step": step_counter,
                    "action": "TEXT_RETHINK",
                    "target_aspect_id": target_aid,
                    "critique": critique,
                    "revised_candidate": h_rethink,
                    "ct_decision": c_t,
                    "baseline_mutated": baseline_mutated,
                    "pairs": h_b.get("pairs", [])
                }
                audit_history.append(audit_record)

                policy_record = {
                    "step": step_counter,
                    "action": "TEXT_RETHINK",
                    "target_aspect_id": target_aid,
                    "decision": ct_decision,
                    "baseline_mutated": baseline_mutated
                }
                policy_history.append(policy_record)

                transitions.append(TransitionRecord(
                    sample_id=sample_id,
                    aspect_id=target_aid,
                    step=step_counter,
                    pre_sentiment=pre_sentiment,
                    action="TEXT_RETHINK",
                    critique_or_question=critique,
                    observation={"critique": critique},
                    candidate_sentiment=candidate_sent,
                    verifier_decision=ct_decision,
                    post_sentiment=post_sentiment,
                    done=(budget_deep <= 0)
                ).model_dump())

            elif cd_action == "VISION_PROBE":
                question = c_d.get("question_for_vision") or c_d.get("question")
                if not question:
                    break

                h_fusion, cv_dec, acc_ev, budget_deep = execute_visual_probe(
                    target_aid=target_aid,
                    probe_question=question,
                    current_hb=h_b,
                    budget_left=budget_deep
                )

                if cv_dec == "ACCEPT_REVISION":
                    h_b = json.loads(json.dumps(h_fusion))
                    cand_v_sent = next((a["sentiment"] for a in h_fusion.get("aspects", []) if a.get("aspect_id") == target_aid), pre_sentiment)
                    if target_aid and target_aid in aspect_states:
                        aspect_states[target_aid]["baseline_sentiment"] = cand_v_sent
                        aspect_states[target_aid]["accepted_evidence"].extend(acc_ev)
                        aspect_states[target_aid]["status"] = "VERIFIED"
                        aspect_states[target_aid]["action_history"].append("VISION_PROBE:ACCEPT")
                else:
                    if target_aid and target_aid in aspect_states:
                        aspect_states[target_aid]["action_history"].append("VISION_PROBE:REVERT")

        y_final = h_b.get("pairs", [])
        compute["latency_ms"] = round(compute["latency_ms"], 1)

        num_v_probes = len([t for t in transitions if t["action"] == "VISION_PROBE"])
        num_t_rethinks = len([t for t in transitions if t["action"] == "TEXT_RETHINK"])

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
            "initial_contrast": initial_c_d or {},
            "risk_diagnosis": initial_c_d or {},
            "aspect_states": aspect_states,
            "transitions": transitions,
            "rounds": audit_history,
            "policy_history": policy_history,
            "final_ledger": h_b,
            "final_pairs": y_final,
            "num_visual_probes": num_v_probes,
            "num_text_queries": num_t_rethinks,
            "num_queries_total": num_v_probes + num_t_rethinks,
            "compute": compute
        }
