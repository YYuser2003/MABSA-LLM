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
from typing import Dict, Any, List, Optional, Tuple, Union

from bacr.client import BaseClient
from bacr.schemas_v3 import (
    ControllerAction,
    RiskDecision,
    TextRiskItem,
    VisualOpportunityItem,
    EvidenceFirewallOutput,
    EvidenceProbeItem,
    EvidenceBundle,
    EvidenceStatus,
    TargetBinding,
    RelevanceLevel,
    RevisionSupport,
    TextRevisionAuditOutput,
    CTDecision,
    RevisionVerifierOutput,
    CVDecision,
    TextRiskAssessment,
    VisualOpportunityAssessment,
    RouteDecision,
    RiskType,
    RiskLevel
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
        self.prompt_text_risk = load_prompt("controller_text_risk.md")
        self.prompt_visual_opportunity = load_prompt("controller_visual_opportunity.md")
        self.prompt_router = load_prompt("controller_router.md")
        self.prompt_text_verifier = load_prompt("controller_text_verifier.md")
        self.prompt_firewall = load_prompt("controller_evidence_firewall.md")
        self.prompt_verifier = load_prompt("controller_revision_verifier.md")

    def diagnose_text_risk(
        self,
        h_a: Dict[str, Any],
        h_b: Optional[Dict[str, Any]] = None,
        target_aspect_text: Optional[str] = None,
        target_aspect_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """C_T^risk: Purely textual risk diagnosis without any visual cues (strictly physically decoupled)."""
        current_baseline = h_b or h_a
        target_label = target_aspect_text or target_aspect_id or "Target Aspect"
        user_prompt = (
            f'Target Aspect: "{target_label}"\n\n'
            f'Text Anchor Hypothesis (H_A):\n{json.dumps(h_a, ensure_ascii=False, indent=2)}\n\n'
            f'Current Text Baseline (H_B):\n{json.dumps(current_baseline, ensure_ascii=False, indent=2)}\n\n'
            f'Diagnose linguistic and syntactic error risks for "{target_label}" and output valid JSON.'
        )
        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_text_risk,
            user_prompt=user_prompt
        )
        if not isinstance(res, dict):
            res = {}
        risk_type_str = str(res.get("risk_type", "NO_RISK")).upper().strip()
        risk_level_str = str(res.get("risk_level", "LOW")).upper().strip()
        basis_str = str(res.get("basis", ""))

        valid_types = [r.value for r in RiskType]
        if risk_type_str not in valid_types:
            risk_type_str = "UNKNOWN"
        valid_levels = [r.value for r in RiskLevel]
        if risk_level_str not in valid_levels:
            risk_level_str = "LOW"

        assessment = TextRiskAssessment(
            risk_type=RiskType(risk_type_str),
            risk_level=RiskLevel(risk_level_str),
            basis=basis_str
        )
        return assessment.model_dump(), usage, lat

    def assess_visual_opportunity(
        self,
        target_aspect_text: str,
        v_0: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """C_V^opp: Physical visual opportunity assessment without sentiment hypotheses."""
        user_prompt = (
            f'Target Aspect: "{target_aspect_text}"\n\n'
            f'Global Visual Sketch (V_0):\n{json.dumps(v_0, ensure_ascii=False, indent=2)}\n\n'
            f'Assess physical visual opportunity for "{target_aspect_text}" and output valid JSON.'
        )
        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_visual_opportunity,
            user_prompt=user_prompt
        )
        if not isinstance(res, dict):
            res = {}
        opp_type = str(res.get("opportunity_type", "NO_OPPORTUNITY")).upper().strip()
        tgt_vis = bool(res.get("target_visible", False))
        basis_code = str(res.get("basis_code", "GENERIC_DECORATIVE")).upper().strip()

        assessment = VisualOpportunityAssessment(
            opportunity_type=opp_type,
            target_visible=tgt_vis,
            basis_code=basis_code
        )
        return assessment.model_dump(), usage, lat

    def route_action(
        self,
        text_risk: Dict[str, Any],
        visual_opportunity: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        budget: int = 2,
        action_mask: Optional[List[str]] = None,
        pending_visual_gap: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """C_R: Discrete routing given decoupled risk, opportunity, history, and budget."""
        valid_actions = action_mask or [a.value for a in ControllerAction]
        hist_str = json.dumps(history, ensure_ascii=False, indent=2) if history else "None (Round 0)"
        gap_str = f'Pending Visual Gap: "{pending_visual_gap}"' if pending_visual_gap else "Pending Visual Gap: None"

        user_prompt = (
            f'Textual Risk Assessment (R_T):\n{json.dumps(text_risk, ensure_ascii=False, indent=2)}\n\n'
            f'Visual Opportunity Assessment (O_V):\n{json.dumps(visual_opportunity, ensure_ascii=False, indent=2)}\n\n'
            f'{gap_str}\n\n'
            f'Interaction History:\n{hist_str}\n\n'
            f'Remaining Budget: {budget}\n'
            f'Action Mask (Permitted Actions): {valid_actions}\n\n'
            f'Select discrete action and output valid JSON.'
        )
        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_router,
            user_prompt=user_prompt
        )
        if not isinstance(res, dict):
            res = {}
        act_str = str(res.get("action", "FINALIZE")).upper().strip()
        if act_str not in valid_actions or budget <= 0:
            act_str = ControllerAction.FINALIZE.value

        decision = RouteDecision(
            action=ControllerAction(act_str),
            rationale=str(res.get("rationale", ""))
        )
        res_dict = decision.model_dump()
        res_dict["action"] = decision.action.value
        res_dict["query_type"] = res.get("query_type")
        return res_dict, usage, lat

    def generate_visual_question(
        self,
        target_aspect_text: str,
        v_0: Optional[Dict[str, Any]] = None,
        pending_gap: Optional[str] = None,
        query_type: Optional[str] = None
    ) -> str:
        """Generates neutral, non-leading visual question grounded in the query_type taxonomy."""
        if pending_gap:
            return pending_gap
        q_type = str(query_type or "FACIAL_EXPRESSION").upper().strip()
        if "FACE" in q_type or "EXPRESSION" in q_type:
            return f"What specific facial expression or emotion is displayed by {target_aspect_text}?"
        elif "ACTION" in q_type or "GESTURE" in q_type:
            return f"What physical action, pose, or interaction is {target_aspect_text} performing?"
        elif "TEXT" in q_type or "SIGN" in q_type:
            return f"What exact text, slogan, or logo associated with {target_aspect_text} is visible?"
        elif "OBJECT" in q_type or "PRESENCE" in q_type:
            return f"What is the physical condition and state of {target_aspect_text} in the scene?"
        else:
            return f"Describe the verifiable physical visual details and setting of {target_aspect_text}."

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
        risk_diagnosis: Union[Dict[str, Any], TextRiskItem],
        target_aspect_id: Optional[str] = None,
        target_aspect_text: Optional[str] = None
    ) -> Tuple[str, Dict[str, int], float]:
        """C_Q^T: Generates a purely linguistic critique for text re-deliberation (T_R).
        STRICT PHYSICAL FIREWALL: Receives ONLY H_B and diagnosed textual risk; V0 is physically excluded.
        """
        # Ensure purely textual risk payload
        if isinstance(risk_diagnosis, TextRiskItem):
            clean_risk = {"type": risk_diagnosis.type, "basis": risk_diagnosis.basis}
        elif isinstance(risk_diagnosis, dict):
            if "text_risk" in risk_diagnosis and isinstance(risk_diagnosis["text_risk"], dict):
                clean_risk = {
                    "type": risk_diagnosis["text_risk"].get("type", "NO_RISK"),
                    "basis": risk_diagnosis["text_risk"].get("basis", "")
                }
            else:
                clean_risk = {
                    "type": risk_diagnosis.get("type", risk_diagnosis.get("risk_type", "NO_RISK")),
                    "basis": risk_diagnosis.get("basis", risk_diagnosis.get("risk_description", ""))
                }
        else:
            clean_risk = {"type": "NO_RISK", "basis": ""}

        # Sanitize any accidental visual terms in basis
        visual_leak_words = [
            "image", "photo", "picture", "visual", "smiling", "smile", 
            "facial", "expression", "wearing", "background", "depicts", "shown"
        ]
        b_str = clean_risk.get("basis", "").lower()
        if any(w in b_str for w in visual_leak_words):
            clean_risk["basis"] = "Re-evaluate linguistic modifier attachment and syntactic scope neutrality."

        user_prompt = (
            f'Current Verified Text Baseline (H_B):\n{json.dumps(h_b, ensure_ascii=False, indent=2)}\n\n'
            f'Target Aspect: "{target_aspect_text or "Target"}" (ID: {target_aspect_id or "a_01"})\n\n'
            f'Diagnosed Textual Risk: {json.dumps(clean_risk, ensure_ascii=False, indent=2)}\n\n'
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

    def aggregate_evidence_bundle(
        self,
        probes: List[Dict[str, Any]],
        target_aspect_id: Optional[str] = None
    ) -> EvidenceBundle:
        """C_A: Deterministic Evidence Aggregator compiling multi-turn probe results into an EvidenceBundle with provenance citations."""
        probe_items: List[EvidenceProbeItem] = []
        accepted_facts: List[Dict[str, str]] = []
        has_support = False
        has_contradict = False

        for idx, p in enumerate(probes):
            p_id = p.get("probe_id") or f"probe_{idx + 1}"
            facts = p.get("usable_evidence", p.get("facts", []))
            st = str(p.get("status", p.get("evidence_status", "INVALID"))).upper().strip()
            tb = str(p.get("target_binding", "UNBOUND")).upper().strip()
            rel = str(p.get("relevance", "LOW")).upper().strip()
            rs = str(p.get("revision_support", "NON_DECISIVE")).upper().strip()

            item = EvidenceProbeItem(
                probe_id=p_id,
                question=p.get("question", ""),
                status=EvidenceStatus(st) if st in [s.value for s in EvidenceStatus] else EvidenceStatus.INVALID,
                target_binding=TargetBinding(tb) if tb in [b.value for b in TargetBinding] else TargetBinding.UNBOUND,
                relevance=RelevanceLevel(rel) if rel in [r.value for r in RelevanceLevel] else RelevanceLevel.LOW,
                facts=facts,
                revision_support=RevisionSupport(rs) if rs in [s.value for s in RevisionSupport] else RevisionSupport.NON_DECISIVE,
                verification_notes=p.get("verification_notes", "")
            )
            probe_items.append(item)

            if item.status == EvidenceStatus.VALID and item.target_binding == TargetBinding.DIRECT:
                for f in facts:
                    if f and not any(af["content"] == f for af in accepted_facts):
                        accepted_facts.append({"ref": p_id, "content": f})

            if item.revision_support == RevisionSupport.SUPPORTS_REVISION:
                has_support = True
            elif item.revision_support == RevisionSupport.CONTRADICTS_REVISION:
                has_contradict = True

        combined_support = RevisionSupport.NON_DECISIVE
        if has_support and not has_contradict:
            combined_support = RevisionSupport.SUPPORTS_REVISION
        elif has_contradict:
            combined_support = RevisionSupport.CONTRADICTS_REVISION

        usable_list = [f["content"] for f in accepted_facts]
        overall_status = EvidenceStatus.VALID if accepted_facts else EvidenceStatus.INVALID
        overall_binding = TargetBinding.DIRECT if accepted_facts else TargetBinding.UNBOUND
        return EvidenceBundle(
            probes=probe_items,
            accepted_facts=accepted_facts,
            usable_evidence=usable_list,
            combined_support=combined_support,
            revision_support=combined_support,
            status=overall_status,
            target_binding=overall_binding,
            target_aspect_id=target_aspect_id
        )

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
        usable_ev = verified_evidence.get("usable_evidence")
        if usable_ev is None:
            usable_ev = [f["content"] for f in verified_evidence.get("accepted_facts", [])]
        rev_sup = verified_evidence.get("revision_support") or verified_evidence.get("combined_support", "NON_DECISIVE")

        is_evidence_invalid = (
            ev_status in ["INSUFFICIENT", "INVALID"]
            or ev_binding == "UNBOUND"
            or not usable_ev
            or rev_sup in ["CONTRADICTS_REVISION", "NON_DECISIVE"]
        )
        if decision == "ACCEPT_REVISION" and is_evidence_invalid:
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
