"""Meta-Cognitive Controller (C) for BACR-v3.

Executes Risk-Aware Diagnosis and Evidence Governance without raw modality exposure:
- CD: Risk-Aware Diagnosis & Compute Allocation (FINALIZE, TEXT_RETHINK, VISION_PROBE)
- C_Q^T: Physically decoupled linguistic critique generator (never sees V0)
- CT: Text Revision Verifier (audits text re-deliberation to prevent drift)
- CE: Evidence Firewall (strictly fail-closed filter for verifiable physical facts)
- CV: Revision Verifier (strictly anchored to HB with Revert-to-Baseline safety rule)
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

from bacr.client import BaseClient
from bacr.schemas_v3 import (
    ControllerAction,
    RiskDecision,
    EvidenceFirewallOutput,
    EvidenceStatus,
    TargetBinding,
    RevisionSupport,
    TextRevisionAuditOutput,
    CTDecision,
    RevisionVerifierOutput,
    CVDecision
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
            f'Policy History (Sanitized Verifier Decisions Only):\n{hist_str}\n\n'
            f'Remaining Deep Budget: {budget}\n\n'
            f'Diagnose error risk and output your discrete compute allocation in valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_diagnosis,
            user_prompt=user_prompt
        )

        # Validate with strict Pydantic model
        validated = RiskDecision.validate_or_fallback(res)
        res_dict = validated.model_dump()
        res_dict["action"] = validated.action.value

        # Programmatic budget exhaustion safeguard
        if budget <= 0 and validated.action in [ControllerAction.TEXT_RETHINK, ControllerAction.VISION_PROBE]:
            res_dict["action"] = ControllerAction.FINALIZE.value
            res_dict["decision_reason"] = (res_dict.get("decision_reason", "") + " [Budget exhausted, finalized.]").strip()

        # Semantic Firewall: Sanitize critique_for_text to prevent implicit visual leakage
        raw_critique = res_dict.get("critique_for_text")
        if raw_critique:
            visual_leak_words = [
                "image", "photo", "picture", "visual", "smiling", "smile", 
                "facial", "expression", "wearing", "background", "depicts", "shown"
            ]
            critique_lower = raw_critique.lower()
            if any(w in critique_lower for w in visual_leak_words):
                res_dict["critique_for_text"] = "Re-examine the tweet's linguistic syntax, modifier scope, and reporting frame neutrality without assuming external visual cues."
                res_dict["critique_sanitized"] = True

        # Compatibility keys
        res_dict["contrast_type"] = res_dict.get("risk_diagnosis", {}).get("risk_type", "NO_RISK")
        if res_dict["action"] == "VISION_PROBE" and "question" not in res_dict:
            res_dict["question"] = res_dict.get("question_for_vision")

        return res_dict, usage, lat

    def generate_text_critique(
        self,
        h_b: Dict[str, Any],
        risk_diagnosis: Dict[str, Any],
        target_aspect_id: Optional[str] = None,
        target_aspect_text: Optional[str] = None
    ) -> Tuple[str, Dict[str, int], float]:
        """C_Q^T: Generates a purely linguistic critique for text re-deliberation (T_R).
        STRICT PHYSICAL FIREWALL: Receives ONLY H_B and diagnosed risk; V0 is physically excluded.
        """
        user_prompt = (
            f'Current Verified Text Baseline (H_B):\n{json.dumps(h_b, ensure_ascii=False, indent=2)}\n\n'
            f'Target Aspect: "{target_aspect_text or "Target"}" (ID: {target_aspect_id or "a_01"})\n\n'
            f'Diagnosed Textual Risk: {json.dumps(risk_diagnosis, ensure_ascii=False, indent=2)}\n\n'
            f'Generate a concise, purely linguistic critique instructing the Text Reasoner how to re-evaluate '
            f'syntactic dependency, modifier attachment, or reporting frame neutrality. '
            f'Do NOT mention any images or visual cues. Output valid JSON: {{"critique": "..."}}'
        )
        system_prompt = (
            "You are a Meta-Cognitive Linguistic Auditor. Generate a targeted linguistic critique "
            "based solely on tweet text syntax and current baseline. You have zero access to visual cues."
        )
        res, usage, lat = self.client.call_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        critique = ""
        if isinstance(res, dict):
            critique = res.get("critique") or res.get("critique_for_text") or ""
            if not critique:
                critique = next((str(v) for k, v in res.items() if isinstance(v, str) and len(v) > 10), "")
        if not critique:
            critique = f"Re-evaluate whether {target_aspect_text or 'the target aspect'} is modified by evaluative sentiment words or is merely a neutral reporting entity."
        return critique, usage, lat

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

        # Validate with strict Pydantic model
        validated = TextRevisionAuditOutput.validate_or_fallback(res)
        res_dict = validated.model_dump()
        res_dict["decision"] = validated.decision.value

        return res_dict, usage, lat

    def filter_evidence_firewall(
        self,
        question: str,
        raw_evidence: Dict[str, Any],
        target_aspect_id: Optional[str] = None,
        target_aspect_text: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """CE: Evidence Firewall: strictly fail-closed filter and revision support assessor."""
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

        # Strict Fail-Closed Validation
        validated = EvidenceFirewallOutput.validate_fail_closed(res)
        res_dict = validated.model_dump()
        res_dict["status"] = validated.status.value
        res_dict["target_binding"] = validated.target_binding.value
        res_dict["relevance"] = validated.relevance.value
        res_dict["revision_support"] = validated.revision_support.value

        # Legacy compatibility keys
        res_dict["evidence_status"] = res_dict["status"]
        res_dict["rejected_content"] = res_dict["rejected_inferences"]

        return res_dict, usage, lat

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

        # Validate with strict Pydantic model
        validated = RevisionVerifierOutput.validate_or_fallback(res)
        res_dict = validated.model_dump()
        decision = validated.decision.value
        
        # Hard Rule Safeguard: If verified evidence is INSUFFICIENT, INVALID, UNBOUND, or NON_DECISIVE/CONTRADICTS, force REVERT_TEXT_BASELINE
        ev_status = verified_evidence.get("status") or verified_evidence.get("evidence_status")
        ev_binding = verified_evidence.get("target_binding")
        usable_ev = verified_evidence.get("usable_evidence", [])
        rev_sup = verified_evidence.get("revision_support", "NON_DECISIVE")
        
        is_evidence_invalid = (
            ev_status in ["INSUFFICIENT", "INVALID"]
            or ev_binding == "UNBOUND"
            or not usable_ev
            or rev_sup in ["CONTRADICTS_REVISION", "NON_DECISIVE"]
        )
        if is_evidence_invalid:
            if decision not in ["REVERT_TEXT_BASELINE", "REVERT_ANCHOR", "REJECT_REVISION"]:
                res_dict["decision"] = "REVERT_TEXT_BASELINE"
                res_dict["audit_rationale"] = (
                    res_dict.get("audit_rationale", "") + 
                    " [Safeguard Activated: Evidence was insufficient/unbound/contradictory/non-decisive; automatically reverted to Text Baseline H_B.]"
                ).strip()
                decision = "REVERT_TEXT_BASELINE"

        if decision == "QUERY_AGAIN" and budget <= 0:
            res_dict["decision"] = "REVERT_TEXT_BASELINE"

        return res_dict, usage, lat

    # Compatibility legacy aliases
    def contrast_audit(self, h_t: Dict[str, Any], h_tv: Dict[str, Any], v0: Dict[str, Any], budget: int = 2):
        return self.diagnose_risk(h_a=h_t, v0=v0, budget=budget)

    def verify_evidence(self, question: str, raw_evidence: Dict[str, Any], target_aspect_id: Optional[str] = None, target_aspect_text: Optional[str] = None):
        return self.filter_evidence_firewall(question=question, raw_evidence=raw_evidence, target_aspect_id=target_aspect_id, target_aspect_text=target_aspect_text)

    def revision_audit(self, h_previous: Dict[str, Any], h_revised: Dict[str, Any], verified_evidence: Dict[str, Any], budget: int = 1):
        return self.verify_revision(h_a=h_previous, h_current=h_revised, verified_evidence=verified_evidence, budget=budget)
