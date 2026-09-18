"""Meta-Cognitive Controller (C) for BACR-v3 Minimal Teacher.

Executes Risk-Aware Routing, Evidence Firewall, and Revision Audit:
- C_R: Route Decision (KEEP, TEXT, VISION) given (aspect, T0, V0).
- C_E: Evidence Firewall (strictly fail-closed filter for verifiable physical facts).
- C_F: Final Audit Verifier (strictly anchored to T0 with Revert-to-Baseline safety rule).
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from bacr.client import BaseClient
from bacr.schemas_v3 import (
    RouteDecision,
    RouteAction,
    EvidenceResult,
    EvidenceStatus,
    TargetBinding,
    FinalAudit,
    AuditDecision,
    CandidatePrediction
)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts", "v3")


def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


SAFE_TEXT_CRITIQUES = {
    "AFFECT_SPILLOVER": (
        "Check whether sentiment from adjacent clauses, background entities, or trailing hashtags "
        "improperly spilled over to target aspect '{aspect}'. Deliberate strictly on "
        "evaluative modifiers syntactically bound to '{aspect}'."
    ),
    "REPORTING_FRAME": (
        "Check whether the text merely reports or quotes someone else's emotional statement rather "
        "than expressing authorial sentiment towards '{aspect}'. Verify if the author's stance towards '{aspect}' "
        "is neutral reporting."
    ),
    "PRAGMATIC_AFFECT": (
        "Check for subtle irony, rhetorical framing, or contrastive discourse markers (e.g. 'but', 'although') "
        "modifying '{aspect}'. Determine if surface words mask the underlying contextual stance."
    ),
    "MISSING_AFFECT": (
        "Re-evaluate whether the text contains implicit or subtle affective cues directly attached to '{aspect}', "
        "or if the authorial stance is strictly neutral."
    ),
    "NO_RISK": (
        "Re-evaluate syntactic dependencies directly attached to target aspect '{aspect}' "
        "and verify if initial sentiment hypothesis ({sentiment}) is fully justified by the text alone."
    ),
}

# Aliases for v3 standardized risk taxonomy
SAFE_TEXT_CRITIQUES["TEXT_OVER_REASONING"] = SAFE_TEXT_CRITIQUES["AFFECT_SPILLOVER"]
SAFE_TEXT_CRITIQUES["ENTITY_GROUNDING_FAILURE"] = SAFE_TEXT_CRITIQUES["AFFECT_SPILLOVER"]
SAFE_TEXT_CRITIQUES["IRONY_POSSIBLE"] = SAFE_TEXT_CRITIQUES["PRAGMATIC_AFFECT"]
SAFE_TEXT_CRITIQUES["MISSING_VISUAL_AFFECT"] = SAFE_TEXT_CRITIQUES["MISSING_AFFECT"]
SAFE_TEXT_CRITIQUES["INSUFFICIENT_CONTEXT"] = SAFE_TEXT_CRITIQUES["MISSING_AFFECT"]


class MetaController:
    def __init__(self, client: BaseClient):
        self.client = client
        try:
            self.prompt_route = load_prompt("controller_diagnosis.md")
        except Exception:
            self.prompt_route = load_prompt("controller_route.md")

        try:
            self.prompt_firewall = load_prompt("evidence_firewall.md")
        except Exception:
            self.prompt_firewall = load_prompt("controller_evidence_firewall.md")

        try:
            self.prompt_audit = load_prompt("revision_verifier.md")
        except Exception:
            self.prompt_audit = load_prompt("controller_final_audit.md")

    def generate_text_critique(
        self,
        aspect: str,
        anchor: Dict[str, Any],
        risk_type: str = "NO_RISK"
    ) -> str:
        """Isolated Text Critique Generator (C_Q^T).
        Physically isolates text critique from V0 - receives strictly (aspect, anchor, risk_type).
        Zero visual sketch or image information can physically enter this generator.
        """
        sentiment = anchor.get("sentiment", "NEU")
        rt_norm = str(risk_type).upper().strip()
        template = SAFE_TEXT_CRITIQUES.get(rt_norm, SAFE_TEXT_CRITIQUES.get(risk_type, SAFE_TEXT_CRITIQUES["NO_RISK"]))
        return template.format(aspect=aspect, sentiment=sentiment)

    def decide_route(
        self,
        aspect: str,
        anchor: Dict[str, Any],
        visual_sketch: Dict[str, Any]
    ) -> Tuple[RouteDecision, Dict[str, int], float]:
        """C_R: Decides discrete route action in {KEEP, TEXT, VISION} given (a_gold, T0, V0).
        Physical isolation: text critique is generated strictly by isolated C_Q^T without visual cues.
        """
        user_prompt = (
            f'Target Aspect: "{aspect}"\n\n'
            f'Initial Text Baseline (T0):\n{json.dumps(anchor, ensure_ascii=False, indent=2)}\n\n'
            f'Global Visual Opportunity Map (V0):\n{json.dumps(visual_sketch, ensure_ascii=False, indent=2)}\n\n'
            f'Audit the text baseline hypothesis, classify the risk, and decide whether to FINALIZE baseline, '
            f'route to TEXT_REVIEW, or route to VISION_QUERY. Output valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_route,
            user_prompt=user_prompt
        )

        decision = RouteDecision.validate_or_fallback(res)

        # Physical isolation: generate text critique strictly through isolated C_Q^T
        if decision.action == RouteAction.TEXT:
            decision.critique = self.generate_text_critique(
                aspect=aspect,
                anchor=anchor,
                risk_type=decision.risk_type
            )

        return decision, usage, lat

    def filter_evidence(
        self,
        aspect: str,
        question: str,
        raw_evidence: Dict[str, Any]
    ) -> Tuple[EvidenceResult, Dict[str, int], float]:
        """C_E: Evidence Firewall enforcing fail-closed physical observability and aspect binding."""
        user_prompt = (
            f'Dispatched Question: "{question}"\n'
            f'Target Aspect: "{aspect}"\n\n'
            f'Raw Visual Sensor Response:\n{json.dumps(raw_evidence, ensure_ascii=False, indent=2)}\n\n'
            f'Filter speculative inferences, verify physical facts, assess target binding, and output in valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_firewall,
            user_prompt=user_prompt
        )

        result = EvidenceResult.validate_fail_closed(res)
        return result, usage, lat

    def audit_revision(
        self,
        route: str,
        text: str,
        aspect: str,
        anchor: Dict[str, Any],
        candidate: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None
    ) -> Tuple[FinalAudit, Dict[str, int], float]:
        """C_F: Final Audit Verifier auditing candidate revision against T0 baseline with counterfactual checks."""
        cand_sent = candidate.get("sentiment", "NEU")
        anchor_sent = anchor.get("sentiment", "NEU")

        # Shortcut: if candidate did not change sentiment, accept trivially
        if cand_sent == anchor_sent:
            return FinalAudit(decision=AuditDecision.ACCEPT, reason="Candidate matches initial baseline sentiment.", visual_necessary=False, evidence_sufficient=True), {"total_tokens": 0}, 0.0

        ev_str = f'\nVerified Visual Evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}\n' if evidence else ""
        user_prompt = (
            f'Route: {route}\n'
            f'Tweet Text: "{text}"\n'
            f'Target Aspect: "{aspect}"\n\n'
            f'Initial Baseline (T0):\n{json.dumps(anchor, ensure_ascii=False, indent=2)}\n\n'
            f'Candidate Revision:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}\n'
            f'{ev_str}\n'
            f'Perform counterfactual audit (evidence sufficiency, visual necessity, and anchor comparison) '
            f'and output in valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_audit,
            user_prompt=user_prompt
        )

        audit = FinalAudit.validate_or_fallback(res)

        # Hard-rule safeguard: if VISION route, verify that evidence is genuinely valid, direct, and has usable/clean evidence
        if route == "VISION":
            ev_status = evidence.get("status") if isinstance(evidence, dict) else None
            ev_binding = evidence.get("target_binding") if isinstance(evidence, dict) else None
            ev_relevance = evidence.get("relevance") if isinstance(evidence, dict) else None
            ev_support = evidence.get("revision_support") if isinstance(evidence, dict) else None
            usable = evidence.get("clean_evidence") or evidence.get("usable_evidence") if isinstance(evidence, dict) else None

            # Fail-closed gate: must be VALID, DIRECT, with non-empty clean/usable evidence
            is_unqualified = (
                ev_status != "VALID"
                or ev_binding != "DIRECT"
                or not usable
                or (ev_relevance is not None and ev_relevance in ["LOW", "MEDIUM"])
                or (ev_support is not None and ev_support in ["CONTRADICTS_REVISION", "NON_DECISIVE"])
            )

            if is_unqualified:
                audit.decision = AuditDecision.REVERT
                audit.reason = (audit.reason + " [Safeguard: Evidence did not meet strict epistemic gate (VALID+DIRECT+USABLE); forced REVERT to T0.]").strip()

        return audit, usage, lat

    # Compatibility Aliases
    filter_evidence_firewall = filter_evidence
    verify_revision = audit_revision
