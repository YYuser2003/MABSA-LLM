"""BACR-v3 Schemas and Validation Module using Pydantic v2.

Defines strict schemas for:
1. ControllerAction, RiskType, Decision, Sentiment, TaskMode Enums.
2. TextRiskItem, VisualOpportunityItem for Decoupled Control.
3. RiskDecision (CD) with strict validation & action prerequisite checks.
4. EvidenceProbeItem & EvidenceBundle for multi-turn structured evidence governance.
5. EvidenceFirewallOutput (CE) with fail-closed guarantee.
6. TextRevisionAuditOutput (CT) supporting ESCALATE_TO_VISION.
7. RevisionVerifierOutput (CV) anchored to HB.
8. BudgetState for action masking and compute allocation accounting.
9. TransitionRecord for RL / SFT serialization with aspect_text & aspect_span.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================

class ControllerAction(str, Enum):
    FINALIZE = "FINALIZE"
    TEXT_RETHINK = "TEXT_RETHINK"
    VISION_PROBE = "VISION_PROBE"


class Sentiment(str, Enum):
    POS = "POS"
    NEG = "NEG"
    NEU = "NEU"


class TaskMode(str, Enum):
    TARGET_GUIDED = "target_guided"
    OPEN_JOINT = "open_joint"


class RiskType(str, Enum):
    NO_RISK = "NO_RISK"
    AFFECT_SPILLOVER = "AFFECT_SPILLOVER"
    MISSING_AFFECT = "MISSING_AFFECT"
    REPORTING_FRAME = "REPORTING_FRAME"
    PRAGMATIC_AFFECT = "PRAGMATIC_AFFECT"
    IRONIC_CONFLICT = "IRONIC_CONFLICT"
    OVER_POLARIZATION = "OVER_POLARIZATION"
    UNDER_POLARIZATION = "UNDER_POLARIZATION"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CTDecision(str, Enum):
    ACCEPT_TEXT_REVISION = "ACCEPT_TEXT_REVISION"
    REVERT_TEXT_BASELINE = "REVERT_TEXT_BASELINE"
    ESCALATE_TO_VISION = "ESCALATE_TO_VISION"
    # Aliases
    VISION_PROBE = "ESCALATE_TO_VISION"
    ACCEPT = "ACCEPT_TEXT_REVISION"
    REVERT = "REVERT_TEXT_BASELINE"


class CVDecision(str, Enum):
    ACCEPT_REVISION = "ACCEPT_REVISION"
    REVERT_TEXT_BASELINE = "REVERT_TEXT_BASELINE"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    QUERY_AGAIN = "NEED_MORE_EVIDENCE"
    # Legacy aliases
    REVERT_ANCHOR = "REVERT_TEXT_BASELINE"
    REJECT_REVISION = "REVERT_TEXT_BASELINE"
    CERTIFY = "ACCEPT_REVISION"


class EvidenceStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    INSUFFICIENT = "INSUFFICIENT"
    PARTIAL = "PARTIAL"


class TargetBinding(str, Enum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    UNBOUND = "UNBOUND"


class RelevanceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RevisionSupport(str, Enum):
    SUPPORTS_REVISION = "SUPPORTS_REVISION"
    CONTRADICTS_REVISION = "CONTRADICTS_REVISION"
    NON_DECISIVE = "NON_DECISIVE"


# ============================================================================
# Core Models
# ============================================================================

class RiskItem(BaseModel):
    type: str
    level: str = "MEDIUM"
    basis: str = ""


class AspectLedgerItem(BaseModel):
    aspect_id: str
    text: str
    span: List[int] = Field(default_factory=lambda: [0, 0])
    sentiment: str = "NEU"
    evidence_text: List[str] = Field(default_factory=list)
    rationale: str = ""
    assumptions: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    risks: Optional[List[RiskItem]] = None

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: Any) -> str:
        s = str(v).upper().strip()
        if s in ["POS", "POSITIVE"]:
            return "POS"
        elif s in ["NEG", "NEGATIVE"]:
            return "NEG"
        return "NEU"


class TextAnchorLedger(BaseModel):
    aspects: List[AspectLedgerItem]
    pairs: List[List[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_pairs(self) -> "TextAnchorLedger":
        expected_pairs = [[a.text, a.sentiment] for a in self.aspects]
        if not self.pairs:
            self.pairs = expected_pairs
        return self


# Alias
StructuredReasoningLedger = TextAnchorLedger


class TargetAspect(BaseModel):
    aspect_uid: str
    text: str
    span: List[int] = Field(default_factory=lambda: [0, 0])


class TextRiskAssessment(BaseModel):
    risk_type: RiskType = RiskType.NO_RISK
    risk_level: RiskLevel = RiskLevel.LOW
    basis: str = ""  # Purely textual/syntactic error mechanism (zero visual cues)


class VisualOpportunityAssessment(BaseModel):
    opportunity_type: str = "NO_OPPORTUNITY"
    target_visible: bool = False
    basis_code: str = ""  # TARGET_PRESENT, FACE_SMILING, OBJECT_CLEAR, GENERIC_DECORATIVE


class RouteDecision(BaseModel):
    action: ControllerAction = ControllerAction.FINALIZE
    rationale: str = ""
    target_aspect_id: Optional[str] = None


class AspectEpisodeState(BaseModel):
    sample_id: str
    target_aspect: TargetAspect
    anchor_sentiment: str = "NEU"
    baseline_sentiment: str = "NEU"
    current_candidate: Optional[str] = None
    accepted_evidence: List[str] = Field(default_factory=list)
    pending_visual_gap: Optional[str] = None
    recommended_next_action: Optional[str] = None
    action_history: List[str] = Field(default_factory=list)
    status: str = "ACTIVE"  # ACTIVE, VERIFIED, FINALIZED


class AspectState(BaseModel):
    aspect_id: str
    text: str
    span: List[int] = Field(default_factory=lambda: [0, 0])
    aspect_uid: Optional[str] = None
    anchor_sentiment: str = "NEU"
    baseline_sentiment: str = "NEU"
    current_candidate: Optional[str] = None
    accepted_evidence: List[str] = Field(default_factory=list)
    risks: List[RiskItem] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    status: str = "ACTIVE"  # ACTIVE, VERIFIED, FINALIZED


class TextRiskItem(BaseModel):
    type: str = "NO_RISK"
    basis: str = ""  # Purely textual / syntactic ground (no visual cues)


class VisualOpportunityItem(BaseModel):
    type: str = "NO_OPPORTUNITY"
    basis_code: str = ""  # e.g. TARGET_VISIBLE, FACE_PRESENT, SCENE_CONTEXT


class RiskDiagnosisDetail(BaseModel):
    risk_type: str = "NO_RISK"
    risk_description: str = ""


class RiskDecision(BaseModel):
    text_risk: TextRiskItem = Field(default_factory=TextRiskItem)
    visual_opportunity: VisualOpportunityItem = Field(default_factory=VisualOpportunityItem)
    risk_diagnosis: RiskDiagnosisDetail = Field(default_factory=RiskDiagnosisDetail)
    action: ControllerAction = ControllerAction.FINALIZE
    target_aspect_id: Optional[str] = None
    critique_for_text: Optional[str] = None
    question_for_vision: Optional[str] = None
    decision_reason: str = ""

    @classmethod
    def validate_or_fallback(cls, data: Any) -> "RiskDecision":
        """Strict validation with safe FINALIZE fallback and action prerequisite checking."""
        if not isinstance(data, dict):
            return cls(action=ControllerAction.FINALIZE, decision_reason="Invalid payload type; finalized.")
        try:
            act = str(data.get("action", "FINALIZE")).upper().strip()
            if act not in [a.value for a in ControllerAction]:
                act = ControllerAction.FINALIZE.value

            # Parse decoupled text_risk and visual_opportunity
            t_risk_dict = data.get("text_risk", {})
            if not isinstance(t_risk_dict, dict):
                t_risk_dict = {"type": str(t_risk_dict), "basis": ""}
            text_risk = TextRiskItem(
                type=t_risk_dict.get("type", "NO_RISK"),
                basis=t_risk_dict.get("basis", "")
            )

            v_opp_dict = data.get("visual_opportunity", {})
            if not isinstance(v_opp_dict, dict):
                v_opp_dict = {"type": str(v_opp_dict), "basis_code": ""}
            visual_opp = VisualOpportunityItem(
                type=v_opp_dict.get("type", "NO_OPPORTUNITY"),
                basis_code=v_opp_dict.get("basis_code", "")
            )

            # Backward compatibility for risk_diagnosis
            diag_dict = data.get("risk_diagnosis", {})
            if not isinstance(diag_dict, dict):
                diag_dict = {"risk_type": str(diag_dict), "risk_description": ""}
            risk_diag = RiskDiagnosisDetail(
                risk_type=diag_dict.get("risk_type", text_risk.type),
                risk_description=diag_dict.get("risk_description", text_risk.basis)
            )

            target_aid = data.get("target_aspect_id")
            critique = data.get("critique_for_text")
            q_vision = data.get("question_for_vision") or data.get("question")

            # Enforce action prerequisites
            if act == ControllerAction.TEXT_RETHINK.value and not target_aid:
                act = ControllerAction.FINALIZE.value
                data["decision_reason"] = "TEXT_RETHINK requested without target_aspect_id; falling back to FINALIZE."
            elif act == ControllerAction.VISION_PROBE.value and not target_aid:
                act = ControllerAction.FINALIZE.value
                data["decision_reason"] = "VISION_PROBE requested without target_aspect_id; falling back to FINALIZE."

            return cls(
                text_risk=text_risk,
                visual_opportunity=visual_opp,
                risk_diagnosis=risk_diag,
                action=ControllerAction(act),
                target_aspect_id=target_aid,
                critique_for_text=critique,
                question_for_vision=q_vision,
                decision_reason=data.get("decision_reason", "")
            )
        except Exception:
            return cls(action=ControllerAction.FINALIZE, decision_reason="Schema parse error; fallback to FINALIZE.")


class EvidenceFirewallOutput(BaseModel):
    status: EvidenceStatus = EvidenceStatus.INVALID
    target_binding: TargetBinding = TargetBinding.UNBOUND
    relevance: RelevanceLevel = RelevanceLevel.LOW
    usable_evidence: List[str] = Field(default_factory=list)
    rejected_inferences: List[str] = Field(default_factory=list)
    revision_support: RevisionSupport = RevisionSupport.NON_DECISIVE
    verification_notes: str = ""

    @classmethod
    def validate_fail_closed(cls, data: Any) -> "EvidenceFirewallOutput":
        """Strict fail-closed validation: any error or missing key falls back to INVALID/NON_DECISIVE."""
        fail_closed_default = cls(
            status=EvidenceStatus.INVALID,
            target_binding=TargetBinding.UNBOUND,
            relevance=RelevanceLevel.LOW,
            usable_evidence=[],
            rejected_inferences=[],
            revision_support=RevisionSupport.NON_DECISIVE,
            verification_notes="Firewall fail-closed triggered: invalid or incomplete evidence output."
        )
        if not isinstance(data, dict):
            return fail_closed_default

        try:
            st = str(data.get("status", data.get("evidence_status", ""))).upper().strip()
            if st not in [s.value for s in EvidenceStatus]:
                return fail_closed_default

            ev = data.get("usable_evidence")
            if not isinstance(ev, list):
                return fail_closed_default

            tb = str(data.get("target_binding", "UNBOUND")).upper().strip()
            if tb not in [b.value for b in TargetBinding]:
                tb = TargetBinding.UNBOUND.value

            rel = str(data.get("relevance", "LOW")).upper().strip()
            if rel not in [r.value for r in RelevanceLevel]:
                rel = RelevanceLevel.LOW.value

            rs = str(data.get("revision_support", "NON_DECISIVE")).upper().strip()
            if rs not in [s.value for s in RevisionSupport]:
                rs = RevisionSupport.NON_DECISIVE.value

            status_enum = EvidenceStatus(st)
            if status_enum in [EvidenceStatus.INVALID, EvidenceStatus.INSUFFICIENT]:
                usable = []
                support = RevisionSupport.NON_DECISIVE
            else:
                usable = [str(x) for x in ev if str(x).strip()]
                support = RevisionSupport(rs)

            return cls(
                status=status_enum,
                target_binding=TargetBinding(tb),
                relevance=RelevanceLevel(rel),
                usable_evidence=usable,
                rejected_inferences=data.get("rejected_inferences", []),
                revision_support=support,
                verification_notes=data.get("verification_notes", "")
            )
        except Exception:
            return fail_closed_default


class EvidenceProbeItem(BaseModel):
    probe_id: str
    question: str
    query_type: str = "GENERAL"  # FACIAL_EXPRESSION, ACTION_STATE, OBJECT_PRESENCE, SCENE_CONTEXT
    status: EvidenceStatus = EvidenceStatus.INVALID
    target_binding: TargetBinding = TargetBinding.UNBOUND
    relevance: RelevanceLevel = RelevanceLevel.LOW
    facts: List[str] = Field(default_factory=list)
    revision_support: RevisionSupport = RevisionSupport.NON_DECISIVE
    verification_notes: str = ""


class EvidenceBundle(BaseModel):
    probes: List[EvidenceProbeItem] = Field(default_factory=list)
    accepted_facts: List[Dict[str, str]] = Field(default_factory=list)  # [{"ref": "probe_1", "content": "..."}]
    usable_evidence: List[str] = Field(default_factory=list)
    combined_support: RevisionSupport = RevisionSupport.NON_DECISIVE
    revision_support: RevisionSupport = RevisionSupport.NON_DECISIVE
    status: EvidenceStatus = EvidenceStatus.VALID
    target_binding: TargetBinding = TargetBinding.DIRECT
    target_aspect_id: Optional[str] = None


class TextRevisionAuditOutput(BaseModel):
    decision: CTDecision = CTDecision.REVERT_TEXT_BASELINE
    text_evidence_verified: bool = False
    modifier_scope_changed: bool = False
    reporting_frame_decoupled: bool = False
    audit_rationale: str = ""
    target_aspect_id: Optional[str] = None
    visual_gap: Optional[str] = None
    question_for_vision: Optional[str] = None

    @classmethod
    def validate_or_fallback(cls, data: Any) -> "TextRevisionAuditOutput":
        """Validates CT output; on failure falls back to REVERT_TEXT_BASELINE."""
        fail_safe = cls(
            decision=CTDecision.REVERT_TEXT_BASELINE,
            audit_rationale="CT validation failure; default to REVERT_TEXT_BASELINE."
        )
        if not isinstance(data, dict):
            return fail_safe
        try:
            raw_dec = str(data.get("decision", "")).upper().strip()
            if raw_dec in ["ACCEPT", "ACCEPT_TEXT_REVISION"]:
                dec = CTDecision.ACCEPT_TEXT_REVISION
            elif raw_dec in ["VISION_PROBE", "ESCALATE_TO_VISION"]:
                dec = CTDecision.ESCALATE_TO_VISION
            else:
                dec = CTDecision.REVERT_TEXT_BASELINE

            return cls(
                decision=dec,
                text_evidence_verified=bool(data.get("text_evidence_verified", False)),
                modifier_scope_changed=bool(data.get("modifier_scope_changed", False)),
                reporting_frame_decoupled=bool(data.get("reporting_frame_decoupled", False)),
                audit_rationale=str(data.get("audit_rationale", "")),
                target_aspect_id=data.get("target_aspect_id"),
                visual_gap=data.get("visual_gap"),
                question_for_vision=data.get("question_for_vision")
            )
        except Exception:
            return fail_safe


class RevisionVerifierOutput(BaseModel):
    decision: CVDecision = CVDecision.REVERT_TEXT_BASELINE
    audit_rationale: str = ""
    target_aspect_id: Optional[str] = None
    next_question: Optional[str] = None

    @classmethod
    def validate_or_fallback(cls, data: Any) -> "RevisionVerifierOutput":
        """Validates CV output; on failure falls back to REVERT_TEXT_BASELINE."""
        fail_safe = cls(
            decision=CVDecision.REVERT_TEXT_BASELINE,
            audit_rationale="CV validation failure; default to REVERT_TEXT_BASELINE."
        )
        if not isinstance(data, dict):
            return fail_safe
        try:
            raw_dec = str(data.get("decision", "")).upper().strip()
            if raw_dec in ["ACCEPT_REVISION", "CERTIFY", "ACCEPT"]:
                dec = CVDecision.ACCEPT_REVISION
            elif raw_dec in ["NEED_MORE_EVIDENCE", "QUERY_AGAIN"]:
                dec = CVDecision.NEED_MORE_EVIDENCE
            else:
                dec = CVDecision.REVERT_TEXT_BASELINE

            return cls(
                decision=dec,
                audit_rationale=str(data.get("audit_rationale", "")),
                target_aspect_id=data.get("target_aspect_id"),
                next_question=data.get("next_question")
            )
        except Exception:
            return fail_safe


class BudgetState(BaseModel):
    max_deep_actions: int = 2
    max_text_rethinks: int = 1
    max_visual_probes: int = 2
    deep_remaining: int = 2
    text_remaining: int = 1
    vision_remaining: int = 2

    def consume_text(self):
        self.deep_remaining = max(0, self.deep_remaining - 1)
        self.text_remaining = max(0, self.text_remaining - 1)

    def consume_vision(self):
        self.deep_remaining = max(0, self.deep_remaining - 1)
        self.vision_remaining = max(0, self.vision_remaining - 1)

    def get_valid_actions(self) -> List[str]:
        if self.deep_remaining <= 0:
            return [ControllerAction.FINALIZE.value]
        valid = [ControllerAction.FINALIZE.value]
        if self.text_remaining > 0:
            valid.append(ControllerAction.TEXT_RETHINK.value)
        if self.vision_remaining > 0:
            valid.append(ControllerAction.VISION_PROBE.value)
        return valid


class TransitionRecord(BaseModel):
    sample_id: str
    aspect_id: Optional[str] = None
    aspect_text: Optional[str] = None
    aspect_span: Optional[List[int]] = None
    aspect_uid: Optional[str] = None
    step: int
    pre_sentiment: str
    action: str
    critique_or_question: Optional[str] = None
    observation: Dict[str, Any] = Field(default_factory=dict)
    candidate_sentiment: Optional[str] = None
    verifier_decision: str
    post_sentiment: str
    done: bool = False


# ============================================================================
# Compatibility Validation Functions for existing callers
# ============================================================================

def validate_structured_ledger(data: Any) -> Tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Ledger must be a JSON dictionary."
    aspects = data.get("aspects")
    if not isinstance(aspects, list):
        return False, "Ledger must contain an 'aspects' list."
    return True, "Valid structured ledger."


def validate_risk_diagnosis(data: Any) -> Tuple[bool, str]:
    try:
        obj = RiskDecision.validate_or_fallback(data)
        if obj.action == ControllerAction.TEXT_RETHINK and not obj.target_aspect_id:
            return False, "TEXT_RETHINK requires target_aspect_id"
        return True, "Valid risk diagnosis"
    except Exception as e:
        return False, str(e)


def validate_evidence_firewall(data: Any) -> Tuple[bool, str]:
    try:
        obj = EvidenceFirewallOutput.validate_fail_closed(data)
        if obj.status == EvidenceStatus.INVALID:
            return False, "Evidence marked INVALID under fail-closed contract"
        return True, "Valid evidence firewall output"
    except Exception as e:
        return False, str(e)


def validate_text_revision_audit(data: Any) -> Tuple[bool, str]:
    try:
        obj = TextRevisionAuditOutput.validate_or_fallback(data)
        return True, f"Valid decision: {obj.decision.value}"
    except Exception as e:
        return False, str(e)


def validate_revision_verifier(data: Any) -> Tuple[bool, str]:
    try:
        obj = RevisionVerifierOutput.validate_or_fallback(data)
        return True, f"Valid decision: {obj.decision.value}"
    except Exception as e:
        return False, str(e)


# Legacy aliases
validate_contrast_audit = validate_risk_diagnosis
validate_evidence_verification = validate_evidence_firewall
validate_revision_audit = validate_revision_verifier
ContrastAuditResult = RiskDecision
EvidenceFirewallResult = EvidenceFirewallOutput
EvidenceVerificationResult = EvidenceFirewallOutput
TextRevisionAuditDecision = TextRevisionAuditOutput
RevisionAuditDecision = RevisionVerifierOutput
