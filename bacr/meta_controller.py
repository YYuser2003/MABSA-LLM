"""Meta-Cognitive Controller (C) for BACR-v3.

Executes Risk-Aware Diagnosis and Evidence Governance without raw modality exposure:
- CD: Risk-Aware Diagnosis & Compute Allocation (FINALIZE, TEXT_RETHINK, VISION_PROBE)
- CE: Evidence Firewall (intercepts, sanitizes, and verifies raw visual probe output)
- CV: Revision Verifier (strictly anchored to immutable H_A with Revert-to-Anchor safety rule)
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

from bacr.client import BaseClient
from bacr.schemas_v3 import (
    RiskDiagnosisDecision,
    EvidenceFirewallResult,
    RevisionAuditDecision
)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts", "v3")


def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class MetaController:
    def __init__(self, client: BaseClient):
        self.client = client
        self.prompt_diagnosis = load_prompt("controller_diagnosis.md")
        self.prompt_text_verifier = load_prompt("controller_text_verifier.md")
        self.prompt_firewall = load_prompt("controller_evidence_firewall.md")
        self.prompt_verifier = load_prompt("controller_revision_verifier.md")

    def diagnose_risk(
        self,
        h_a: Dict[str, Any],
        v0: Dict[str, Any],
        h_b: Optional[Dict[str, Any]] = None,
        aspect_states: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        budget: int = 2
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """CD: Evaluates HA, HB, and V0 to diagnose error risks and allocate verification actions."""
        current_baseline = h_b or h_a
        hist_str = json.dumps(history, ensure_ascii=False, indent=2) if history else "None (Round 0 - initial check)"
        states_str = json.dumps(aspect_states, ensure_ascii=False, indent=2) if aspect_states else "None"
        
        user_prompt = (
            f'Immutable Historical Text Anchor (H_A):\n{json.dumps(h_a, ensure_ascii=False, indent=2)}\n\n'
            f'Current Verified Text Baseline (H_B):\n{json.dumps(current_baseline, ensure_ascii=False, indent=2)}\n\n'
            f'Per-Aspect Internal States (S_t):\n{states_str}\n\n'
            f'Global Visual Sketch (V0 - Opportunity Map ONLY):\n{json.dumps(v0, ensure_ascii=False, indent=2)}\n\n'
            f'Interaction History:\n{hist_str}\n\n'
            f'Remaining Deep Budget: {budget}\n\n'
            f'Diagnose error risk and output your discrete compute allocation in valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_diagnosis,
            user_prompt=user_prompt
        )

        # Programmatic sanity check
        action = res.get("action", "FINALIZE")
        if budget <= 0 and action in ["TEXT_RETHINK", "VISION_PROBE"]:
            res["action"] = "FINALIZE"
            res["decision_reason"] = (res.get("decision_reason", "") + " [Budget exhausted, finalized.]").strip()

        # Semantic Firewall: Sanitize critique_for_text to prevent implicit visual leakage
        raw_critique = res.get("critique_for_text")
        if raw_critique:
            visual_leak_words = [
                "image", "photo", "picture", "visual", "smiling", "smile", 
                "facial", "expression", "wearing", "background", "depicts", "shown"
            ]
            critique_lower = raw_critique.lower()
            if any(w in critique_lower for w in visual_leak_words):
                res["critique_for_text"] = "Re-examine the tweet's linguistic syntax, modifier scope, and reporting frame neutrality without assuming external visual cues."
                res["critique_sanitized"] = True

        # Legacy compatibility keys
        res["contrast_type"] = res.get("risk_diagnosis", {}).get("risk_type", "NO_RISK")
        if action == "VISION_PROBE" and "question" not in res:
            res["question"] = res.get("question_for_vision")

        return res, usage, lat

    def contrast_audit(
        self,
        h_t: Dict[str, Any],
        h_tv: Dict[str, Any],
        v0: Dict[str, Any],
        budget: int = 2
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Legacy alias for backward compatibility."""
        return self.diagnose_risk(h_a=h_t, v0=v0, budget=budget)

    def audit_text_revision(
        self,
        h_b: Dict[str, Any],
        h_rethink: Dict[str, Any],
        critique: str,
        target_aspect_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """CT: Text Revision Verifier auditing TR to prevent ungrounded linguistic drift."""
        user_prompt = (
            f'Current Verified Text Baseline (H_B):\n{json.dumps(h_b, ensure_ascii=False, indent=2)}\n\n'
            f'Proposed Candidate Text Revision (H_RT):\n{json.dumps(h_rethink, ensure_ascii=False, indent=2)}\n\n'
            f'Dispatched Linguistic Critique:\n"{critique}"\n\n'
            f'Target Aspect Focus: {target_aspect_id or "All Aspects"}\n\n'
            f'Audit whether the linguistic revision is grounded in explicit tweet syntax and output your decision in valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_text_verifier,
            user_prompt=user_prompt
        )

        decision = res.get("decision", "ACCEPT_TEXT_REVISION")
        valid_decisions = ["ACCEPT_TEXT_REVISION", "REVERT_TEXT_BASELINE", "VISION_PROBE"]
        if decision not in valid_decisions:
            res["decision"] = "ACCEPT_TEXT_REVISION"

        return res, usage, lat

    def filter_evidence_firewall(
        self,
        question: str,
        raw_evidence: Dict[str, Any],
        target_aspect_id: Optional[str] = None,
        target_aspect_text: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """CE: Evidence Firewall: sanitizes raw probe output and assesses revision support."""
        user_prompt = (
            f'Dispatched Question: "{question}"\n'
            f'Target Aspect: "{target_aspect_text or "General"}" (ID: {target_aspect_id or "a_01"})\n\n'
            f'Raw Visual Sensor Response (Et):\n{json.dumps(raw_evidence, ensure_ascii=False, indent=2)}\n\n'
            f'Filter speculative inferences, verify physical facts, assess revision support, and output in valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_firewall,
            user_prompt=user_prompt
        )

        # Fallback safeguard
        if "usable_evidence" not in res or not isinstance(res["usable_evidence"], list):
            res["usable_evidence"] = raw_evidence.get("observable_evidence", [])
            if not res["usable_evidence"] and raw_evidence.get("answer"):
                res["usable_evidence"] = [raw_evidence["answer"]]

        if "status" not in res:
            res["status"] = "VALID" if not raw_evidence.get("insufficient_visual_evidence") else "INSUFFICIENT"

        if "revision_support" not in res:
            res["revision_support"] = "SUPPORTS_REVISION" if res["status"] == "VALID" else "NON_DECISIVE"

        # Legacy compatibility keys
        res["evidence_status"] = res.get("status", "VALID")
        res["rejected_content"] = res.get("rejected_inferences", [])

        return res, usage, lat

    def verify_evidence(
        self,
        question: str,
        raw_evidence: Dict[str, Any],
        target_aspect_id: Optional[str] = None,
        target_aspect_text: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Legacy alias for filter_evidence_firewall."""
        return self.filter_evidence_firewall(
            question=question,
            raw_evidence=raw_evidence,
            target_aspect_id=target_aspect_id,
            target_aspect_text=target_aspect_text
        )

    def verify_revision(
        self,
        h_a: Dict[str, Any],
        h_current: Dict[str, Any],
        verified_evidence: Dict[str, Any],
        h_b: Optional[Dict[str, Any]] = None,
        budget: int = 1
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """CV: Revision Verifier anchored strictly to HB with REVERT_TEXT_BASELINE safety rule."""
        current_baseline = h_b or h_a
        user_prompt = (
            f'Immutable Historical Text Anchor (H_A):\n{json.dumps(h_a, ensure_ascii=False, indent=2)}\n\n'
            f'Current Verified Text Baseline (H_B):\n{json.dumps(current_baseline, ensure_ascii=False, indent=2)}\n\n'
            f'Current Candidate Ledger (H_current):\n{json.dumps(h_current, ensure_ascii=False, indent=2)}\n\n'
            f'Verified Visual Proof from Firewall (E~):\n{json.dumps(verified_evidence, ensure_ascii=False, indent=2)}\n\n'
            f'Remaining Deep Budget: {budget}\n\n'
            f'Evaluate whether the candidate revision away from H_B is genuinely proven by verified facts and output your decision in valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_verifier,
            user_prompt=user_prompt
        )

        decision = res.get("decision", "ACCEPT_REVISION")
        
        # Hard Rule Safeguard: If verified evidence is INSUFFICIENT, INVALID, UNBOUND, or CONTRADICTS_REVISION, force REVERT_TEXT_BASELINE
        ev_status = verified_evidence.get("status") or verified_evidence.get("evidence_status")
        ev_binding = verified_evidence.get("target_binding")
        usable_ev = verified_evidence.get("usable_evidence", [])
        rev_sup = verified_evidence.get("revision_support", "SUPPORTS_REVISION")
        
        if ev_status in ["INSUFFICIENT", "INVALID"] or ev_binding == "UNBOUND" or not usable_ev or rev_sup == "CONTRADICTS_REVISION":
            if decision not in ["REVERT_TEXT_BASELINE", "REVERT_ANCHOR", "REJECT_REVISION"]:
                res["decision"] = "REVERT_TEXT_BASELINE"
                res["audit_rationale"] = (
                    res.get("audit_rationale", "") + 
                    " [Safeguard Activated: Evidence was insufficient/unbound/contradictory; automatically reverted to Text Baseline H_B.]"
                ).strip()
                decision = "REVERT_TEXT_BASELINE"

        if decision == "QUERY_AGAIN" and budget <= 0:
            res["decision"] = "REVERT_TEXT_BASELINE"

        return res, usage, lat

    def revision_audit(
        self,
        h_previous: Dict[str, Any],
        h_revised: Dict[str, Any],
        verified_evidence: Dict[str, Any],
        budget: int = 1
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Legacy alias for verify_revision."""
        return self.verify_revision(
            h_a=h_previous,
            h_current=h_revised,
            verified_evidence=verified_evidence,
            budget=budget
        )

