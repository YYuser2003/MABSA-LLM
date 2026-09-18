"""BACR-v3 Schemas and Validation Module.

Defines schemas for:
1. Text Anchor Reasoner (TA / HA) with standardized Risk Profile.
2. Global Visual Sensor (IG / V0) as Visual Opportunity Map.
3. Risk-aware Diagnosis Controller (CD) allocating discrete actions.
4. Text Rethink Reasoner (TR / HR^T) without image access.
5. Text Revision Verifier (CT) auditing text revisions to update HB.
6. Targeted Visual Probe (IP) for verifiable physical facts.
7. Evidence Firewall (CE / E~) filtering ungrounded inferences and assessing revision support.
8. Evidence Fusion Reasoner (TF / HF) fusing verified evidence with text baseline.
9. Revision Verifier Controller (CV) anchored to HB (revert-to-text-baseline).
10. AspectState for native multi-aspect MDP and isolation.
"""

from typing import Dict, Any, List, Optional, Tuple, TypedDict


class RiskItem(TypedDict, total=False):
    type: str   # MISSING_AFFECT, AFFECT_SPILLOVER, REPORTING_FRAME, PRAGMATIC_AFFECT, IRONY_CONFLICT
    level: str  # LOW, MEDIUM, HIGH
    basis: str  # short textual justification


class RiskProfile(TypedDict, total=False):
    affect_spillover: str     # low, medium, high
    missing_affect: str       # low, medium, high
    reporting_frame: str      # low, medium, high
    pragmatic_blindness: str  # low, medium, high
    irony_conflict: str       # low, medium, high


class VisualEvidenceItem(TypedDict, total=False):
    ref: str
    content: str


class AspectLedgerItem(TypedDict, total=False):
    aspect_id: str
    text: str
    span: List[int]
    sentiment: str  # POS, NEG, NEU
    text_evidence: List[str]
    visual_evidence: Optional[List[VisualEvidenceItem]]
    rationale: str
    assumptions: List[str]
    uncertainties: List[str]
    risks: Optional[List[RiskItem]]
    risk_profile: Optional[RiskProfile]
    evidence_refs: Optional[List[str]]
    resolved_assumptions: Optional[List[str]]
    remaining_uncertainties: Optional[List[str]]
    revision_reason: Optional[str]


class TextAnchorLedger(TypedDict, total=False):
    aspects: List[AspectLedgerItem]
    pairs: List[List[str]]


# Legacy alias for backward compatibility in imports
StructuredReasoningLedger = TextAnchorLedger


class AspectState(TypedDict, total=False):
    aspect_id: str
    text: str
    span: List[int]
    anchor: Dict[str, Any]         # Immutable H_A^a
    text_baseline: Dict[str, Any]  # Current verified H_B^a
    current_candidate: Dict[str, Any]
    risks: List[RiskItem]
    status: str                    # ACTIVE, VERIFIED, FINALIZED


class GlobalVisualSketch(TypedDict, total=False):
    scene: str
    description: str
    possible_entities: List[Dict[str, Any]]
    ocr: List[str]
    salient_visual_cues: List[str]
    target_presence_hints: List[Dict[str, Any]]
    visual_affect_cues: List[str]


class RiskDiagnosisDetail(TypedDict, total=False):
    risk_type: str  # NO_RISK, OVER_POLARIZATION, UNDER_POLARIZATION, PRAGMATIC_UNCERTAINTY, CROSS_MODAL_CONFLICT
    risk_description: str


class RiskDiagnosisDecision(TypedDict, total=False):
    risk_diagnosis: RiskDiagnosisDetail
    action: str  # FINALIZE, TEXT_RETHINK, VISION_PROBE
    target_aspect_id: Optional[str]
    critique_for_text: Optional[str]
    question_for_vision: Optional[str]
    decision_reason: str


# Legacy alias
ContrastAuditResult = RiskDiagnosisDecision


class DeepVisionResult(TypedDict, total=False):
    answer: str
    observable_evidence: List[str]
    certainty: str  # low, medium, high
    insufficient_visual_evidence: bool


class EvidenceFirewallResult(TypedDict, total=False):
    status: str            # VALID, INVALID, INSUFFICIENT
    target_binding: str    # DIRECT, INDIRECT, UNBOUND
    relevance: str         # HIGH, MEDIUM, LOW
    usable_evidence: List[str]
    rejected_inferences: List[str]
    revision_support: str  # SUPPORTS_REVISION, CONTRADICTS_REVISION, NON_DECISIVE
    verification_notes: str


# Legacy alias
EvidenceVerificationResult = EvidenceFirewallResult


class TextRevisionAuditDecision(TypedDict, total=False):
    decision: str  # ACCEPT_TEXT_REVISION, REVERT_TEXT_BASELINE, VISION_PROBE
    text_evidence_verified: bool
    modifier_scope_changed: bool
    reporting_frame_decoupled: bool
    audit_rationale: str
    target_aspect_id: Optional[str]


class RevisionAuditDecision(TypedDict, total=False):
    decision: str  # ACCEPT_REVISION, REVERT_TEXT_BASELINE, REVERT_ANCHOR, QUERY_AGAIN
    audit_rationale: str
    target_aspect_id: Optional[str]
    next_question: Optional[str]


# Legacy alias
RevisionAuditResult = RevisionAuditDecision


# ============================================================================
# Validation Functions
# ============================================================================

def validate_structured_ledger(data: Any) -> Tuple[bool, str]:
    """Validates that data conforms to TextAnchorLedger / StructuredReasoningLedger."""
    if not isinstance(data, dict):
        return False, "Ledger must be a JSON dictionary."
    aspects = data.get("aspects")
    if not isinstance(aspects, list):
        return False, "Ledger must contain an 'aspects' list."
    for idx, a in enumerate(aspects):
        if not isinstance(a, dict):
            return False, f"Aspect item at index {idx} must be a dictionary."
        if not a.get("aspect_id"):
            return False, f"Aspect item at index {idx} missing 'aspect_id'."
        if not a.get("text"):
            return False, f"Aspect item at index {idx} missing 'text'."
        if a.get("sentiment") not in ["POS", "NEG", "NEU"]:
            return False, f"Aspect {a.get('text')} has invalid sentiment: '{a.get('sentiment')}'."
        if "assumptions" not in a or not isinstance(a.get("assumptions"), list):
            return False, f"Aspect {a.get('text')} must include 'assumptions' list."
        if "uncertainties" not in a or not isinstance(a.get("uncertainties"), list):
            return False, f"Aspect {a.get('text')} must include 'uncertainties' list."
    return True, "Valid structured ledger."


def validate_risk_diagnosis(data: Any) -> Tuple[bool, str]:
    """Validates CD Risk Diagnosis output."""
    if not isinstance(data, dict):
        return False, "Risk diagnosis must be a JSON dictionary."
    action = data.get("action")
    valid_actions = ["FINALIZE", "TEXT_RETHINK", "VISION_PROBE"]
    if action not in valid_actions:
        return False, f"Invalid action: '{action}'. Expected one of {valid_actions}."
    
    if action == "TEXT_RETHINK":
        if not data.get("target_aspect_id"):
            return False, "action 'TEXT_RETHINK' requires non-null 'target_aspect_id'."
        critique = data.get("critique_for_text")
        if not critique or len(str(critique).strip()) == 0:
            return False, "action 'TEXT_RETHINK' requires non-empty 'critique_for_text'."
            
    elif action == "VISION_PROBE":
        if not data.get("target_aspect_id"):
            return False, "action 'VISION_PROBE' requires non-null 'target_aspect_id'."
        q = data.get("question_for_vision") or data.get("question")
        if not q or len(str(q).strip()) == 0:
            return False, "action 'VISION_PROBE' requires a non-empty 'question_for_vision'."
            
    return True, "Valid risk diagnosis."


# Legacy validator alias
validate_contrast_audit = validate_risk_diagnosis


def validate_evidence_firewall(data: Any) -> Tuple[bool, str]:
    """Validates CE Evidence Firewall output."""
    if not isinstance(data, dict):
        return False, "Evidence firewall output must be a JSON dictionary."
    status = data.get("status") or data.get("evidence_status")
    valid_statuses = ["VALID", "INVALID", "INSUFFICIENT", "PARTIAL", "REJECTED"]
    if status not in valid_statuses:
        return False, f"Invalid status: '{status}'."
    binding = data.get("target_binding")
    if binding not in ["DIRECT", "INDIRECT", "UNBOUND"]:
        return False, f"Invalid target_binding: '{binding}'."
    if not isinstance(data.get("usable_evidence"), list):
        return False, "'usable_evidence' must be a list of string facts."
    return True, "Valid evidence firewall output."


# Legacy validator alias
validate_evidence_verification = validate_evidence_firewall


def validate_text_revision_audit(data: Any) -> Tuple[bool, str]:
    """Validates CT Text Revision Verifier output."""
    if not isinstance(data, dict):
        return False, "Text revision audit must be a JSON dictionary."
    decision = data.get("decision")
    valid_decisions = ["ACCEPT_TEXT_REVISION", "REVERT_TEXT_BASELINE", "VISION_PROBE", "CERTIFY", "REJECT_REVISION"]
    if decision not in valid_decisions:
        return False, f"Invalid decision: '{decision}'."
    return True, "Valid text revision audit."


def validate_revision_verifier(data: Any) -> Tuple[bool, str]:
    """Validates CV Revision Verifier output."""
    if not isinstance(data, dict):
        return False, "Revision verifier must be a JSON dictionary."
    decision = data.get("decision")
    valid_decisions = ["ACCEPT_REVISION", "REVERT_TEXT_BASELINE", "REVERT_ANCHOR", "QUERY_AGAIN", "CERTIFY", "REJECT_REVISION"]
    if decision not in valid_decisions:
        return False, f"Invalid decision: '{decision}'."
    if decision == "QUERY_AGAIN":
        if not data.get("next_question"):
            return False, "decision 'QUERY_AGAIN' requires 'next_question'."
    return True, "Valid revision verifier."


# Legacy validator alias
validate_revision_audit = validate_revision_verifier

