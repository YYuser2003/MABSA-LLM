"""BACR-v3 Minimal Teacher Schemas using Pydantic v2.

Defines the 4 essential data contracts for the single-pass Teacher pipeline:
1. RouteDecision:       Action in {KEEP, TEXT, VISION} with risk diagnosis, critique, or question
2. EvidenceResult:      Fail-closed Evidence Firewall output in {VALID, INVALID, INSUFFICIENT}
3. CandidatePrediction: Deliberated sentiment candidate {aspect, sentiment, reason, evidence}
4. FinalAudit:          Decision in {ACCEPT, REVERT} to safeguard baseline hypothesis
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enums
# ============================================================================

class RouteAction(str, Enum):
    KEEP = "KEEP"
    TEXT = "TEXT"
    VISION = "VISION"


class Sentiment(str, Enum):
    POS = "POS"
    NEG = "NEG"
    NEU = "NEU"


class EvidenceStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    INSUFFICIENT = "INSUFFICIENT"


class TargetBinding(str, Enum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    UNBOUND = "UNBOUND"


class EvidenceRelevance(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RevisionSupport(str, Enum):
    SUPPORTS_REVISION = "SUPPORTS_REVISION"
    CONTRADICTS_REVISION = "CONTRADICTS_REVISION"
    NON_DECISIVE = "NON_DECISIVE"


class AuditDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REVERT = "REVERT"


# ============================================================================
# Core Teacher Models
# ============================================================================

VALID_SENTIMENT_MAP = {
    "POS": "POS",
    "POSITIVE": "POS",
    "NEG": "NEG",
    "NEGATIVE": "NEG",
    "NEU": "NEU",
    "NEUTRAL": "NEU"
}


class RouteDecision(BaseModel):
    action: RouteAction = RouteAction.KEEP
    risk_type: str = "NO_RISK"
    reason: str = ""
    critique: Optional[str] = None
    question: Optional[str] = None

    @classmethod
    def validate_or_fallback(cls, data: Any) -> "RouteDecision":
        """Strict validation with safe KEEP fallback.
        Fail-closed invariant: VISION route without an explicit, non-empty question falls back to KEEP.
        """
        if not isinstance(data, dict):
            return cls(action=RouteAction.KEEP, reason="Invalid response payload; fallback to KEEP.")
        try:
            raw_act = str(data.get("action", "KEEP")).upper().strip()
            if raw_act not in [a.value for a in RouteAction]:
                raw_act = RouteAction.KEEP.value

            critique = data.get("critique") or data.get("critique_for_text")
            raw_q = data.get("question") or data.get("question_for_vision")
            question = str(raw_q).strip() if raw_q and str(raw_q).strip() else None

            action_enum = RouteAction(raw_act)
            reason = str(data.get("reason", data.get("decision_reason", "")))

            # Fail-closed: cannot enter VISION without an explicit factual question
            if action_enum == RouteAction.VISION and not question:
                action_enum = RouteAction.KEEP
                reason = f"{reason} [Fail-closed: VISION requested without valid question; safely downgraded to KEEP.]".strip()

            return cls(
                action=action_enum,
                risk_type=str(data.get("risk_type", "NO_RISK")),
                reason=reason,
                critique=critique,
                question=question
            )
        except Exception:
            return cls(action=RouteAction.KEEP, reason="Schema parsing failed; fallback to KEEP.")


class EvidenceResult(BaseModel):
    status: EvidenceStatus = EvidenceStatus.INVALID
    target_binding: TargetBinding = TargetBinding.UNBOUND
    relevance: EvidenceRelevance = EvidenceRelevance.LOW
    revision_support: RevisionSupport = RevisionSupport.NON_DECISIVE
    usable_evidence: List[str] = Field(default_factory=list)
    rejected_inferences: List[str] = Field(default_factory=list)
    verification_notes: str = ""

    @classmethod
    def validate_fail_closed(cls, data: Any) -> "EvidenceResult":
        """Strict fail-closed validation: any anomaly forces INVALID status."""
        fail_default = cls(
            status=EvidenceStatus.INVALID,
            target_binding=TargetBinding.UNBOUND,
            relevance=EvidenceRelevance.LOW,
            revision_support=RevisionSupport.NON_DECISIVE,
            usable_evidence=[],
            rejected_inferences=[],
            verification_notes="Fail-closed default due to invalid data structure."
        )
        if not isinstance(data, dict):
            return fail_default

        try:
            st = str(data.get("status", data.get("evidence_status", ""))).upper().strip()
            if st not in [s.value for s in EvidenceStatus]:
                return fail_default

            tb = str(data.get("target_binding", "UNBOUND")).upper().strip()
            if tb not in [b.value for b in TargetBinding]:
                tb = TargetBinding.UNBOUND.value

            rel_str = str(data.get("relevance", "LOW")).upper().strip()
            if rel_str not in [r.value for r in EvidenceRelevance]:
                rel_str = EvidenceRelevance.LOW.value

            sup_str = str(data.get("revision_support", "NON_DECISIVE")).upper().strip()
            if sup_str not in [s.value for s in RevisionSupport]:
                sup_str = RevisionSupport.NON_DECISIVE.value

            raw_usable = data.get("usable_evidence", [])
            if not isinstance(raw_usable, list):
                return fail_default

            status_enum = EvidenceStatus(st)
            binding_enum = TargetBinding(tb)
            rel_enum = EvidenceRelevance(rel_str)
            sup_enum = RevisionSupport(sup_str)

            # Strict fail-closed: usable_evidence must be cleared unless all 4 conditions hold:
            # 1. status == VALID
            # 2. target_binding == DIRECT
            # 3. relevance == HIGH
            # 4. revision_support == SUPPORTS_REVISION
            if (
                status_enum != EvidenceStatus.VALID
                or binding_enum != TargetBinding.DIRECT
                or rel_enum != EvidenceRelevance.HIGH
                or sup_enum != RevisionSupport.SUPPORTS_REVISION
            ):
                usable = []
            else:
                usable = [str(x).strip() for x in raw_usable if str(x).strip()]

            notes = str(data.get("verification_notes", data.get("reason", data.get("notes", ""))))

            return cls(
                status=status_enum,
                target_binding=binding_enum,
                relevance=rel_enum,
                revision_support=sup_enum,
                usable_evidence=usable,
                rejected_inferences=[str(x) for x in data.get("rejected_inferences", []) if str(x).strip()],
                verification_notes=notes
            )
        except Exception:
            return fail_default


class CandidatePrediction(BaseModel):
    aspect: str
    sentiment: str = "NEU"
    reason: str = ""
    evidence: List[str] = Field(default_factory=list)

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: Any) -> str:
        s = str(v).upper().strip()
        if s in VALID_SENTIMENT_MAP:
            return VALID_SENTIMENT_MAP[s]
        raise ValueError(f"Invalid sentiment value '{v}'. Must be one of {list(VALID_SENTIMENT_MAP.keys())}.")

    @classmethod
    def validate_or_fallback(cls, data: Any, default_aspect: str = "", default_sentiment: str = "NEU") -> "CandidatePrediction":
        """Validates candidate response; strictly reverts to default_sentiment anchor on invalid sentiment."""
        safe_sentiment = VALID_SENTIMENT_MAP.get(str(default_sentiment).upper().strip(), "NEU")
        fallback = cls(
            aspect=default_aspect,
            sentiment=safe_sentiment,
            reason="Invalid candidate payload; reverted to baseline.",
            evidence=[]
        )
        if not isinstance(data, dict):
            return fallback
        try:
            # Handle potential nested 'aspects' list
            if "aspects" in data and isinstance(data["aspects"], list) and data["aspects"]:
                target_dict = data["aspects"][0]
                aspect_name = str(target_dict.get("text", default_aspect))
                reason = str(target_dict.get("rationale", target_dict.get("reason", "")))
                evidence = list(target_dict.get("text_evidence", target_dict.get("evidence", [])))
            else:
                target_dict = data
                aspect_name = str(target_dict.get("aspect", default_aspect))
                reason = str(target_dict.get("reason", target_dict.get("rationale", "")))
                evidence = list(target_dict.get("evidence", target_dict.get("text_evidence", [])))

            raw_sent = str(target_dict.get("sentiment", "")).upper().strip()
            if raw_sent not in VALID_SENTIMENT_MAP:
                return cls(
                    aspect=default_aspect,
                    sentiment=safe_sentiment,
                    reason=f"Invalid sentiment output '{raw_sent}'; strictly reverted to baseline.",
                    evidence=[]
                )

            return cls(
                aspect=aspect_name,
                sentiment=VALID_SENTIMENT_MAP[raw_sent],
                reason=reason,
                evidence=evidence
            )
        except Exception:
            return fallback


class FinalAudit(BaseModel):
    decision: AuditDecision = AuditDecision.REVERT
    reason: str = ""

    @classmethod
    def validate_or_fallback(cls, data: Any) -> "FinalAudit":
        """Validates audit response; defaults to REVERT on any failure."""
        fail_safe = cls(decision=AuditDecision.REVERT, reason="Audit parse failure; default to REVERT.")
        if not isinstance(data, dict):
            return fail_safe
        try:
            raw_dec = str(data.get("decision", "")).upper().strip()
            if raw_dec in ["ACCEPT", "ACCEPT_REVISION", "ACCEPT_TEXT_REVISION"]:
                dec = AuditDecision.ACCEPT
            else:
                dec = AuditDecision.REVERT

            return cls(
                decision=dec,
                reason=str(data.get("reason", data.get("audit_rationale", "")))
            )
        except Exception:
            return fail_safe


class TrainingTransitionRecord(BaseModel):
    """Normalized transition record capturing full MDP state for SFT and RL training."""
    aspect_index: int = 0
    decision_step: int = 0
    step: int = 1  # compatibility: decision_step + 1
    aspect_id: str = ""
    span: Optional[Tuple[int, int]] = None
    sample_id: str
    aspect: str
    aspect_text: str
    t0_sentiment: str
    pre_sentiment: str
    state_before: Dict[str, Any] = Field(default_factory=dict)
    route: str  # KEEP, TEXT, VISION
    risk_type: str = "NO_RISK"
    route_reason: str = ""
    action: str  # KEEP, TEXT_RETHINK, VISION_PROBE
    action_mask: List[int] = Field(default_factory=lambda: [1, 1, 1])  # [KEEP, TEXT, VISION]
    critique: Optional[str] = None
    question: Optional[str] = None
    raw_evidence: Optional[Dict[str, Any]] = None
    verified_evidence: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    candidate: Optional[Dict[str, Any]] = None
    candidate_sentiment: Optional[str] = None
    audit: Optional[Dict[str, Any]] = None
    audit_decision: str = "N/A"
    verifier_decision: str = "NO_OP"
    post_sentiment: str
    final_sentiment: str
    compute: Optional[Dict[str, Any]] = None


# ============================================================================
# Compatibility Aliases
# ============================================================================
ControllerAction = RouteAction
EvidenceFirewallOutput = EvidenceResult
RevisionVerifierOutput = FinalAudit
RiskDecision = RouteDecision

