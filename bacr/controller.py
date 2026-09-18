"""BACR Controller Policy Module.

Implements the decoupled, hierarchical policy chain for BACR-v2:
1. Gap Diagnosis Policy:        \pi_G(g | S_t) \in \mathcal{G}
2. Modality Routing Policy:     \pi_D(d | S_t, g_t) \in {VISUAL, TEXT, STOP}
3. Question Generation Policy:  \pi_Q(q | S_t, g_t, d_t)
4. Belief Revision Policy:      \pi_R(Y | S_t, e_t)
"""

from typing import Dict, Any, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
import json


ProbeDirection = Literal["VISUAL_PROBE", "TEXT_PROBE", "NO_PROBE"]
GapType = Literal[
    "NO_GAP",
    "VISUAL_AFFECT_GAP",
    "VISUAL_GROUNDING_GAP",
    "PRAGMATIC_IRONY_GAP",
    "TEXT_SEMANTIC_GAP",
    "CROSS_MODAL_CONFLICT"
]


@dataclass
class ControllerDecision:
    """Represents a discrete turn decision by the BACR Controller."""
    direction: ProbeDirection
    gap_type: str = "NO_GAP"
    gap_description: str = ""
    target_aspect: Optional[str] = None
    question: Optional[str] = None
    target_modality: Optional[str] = None
    confidence_before: float = 0.0
    reasoning: str = ""
    stop: bool = False


@dataclass
class AspectBeliefState:
    """State tracking per aspect throughout the multi-turn probing interaction."""
    aspect_id: str
    aspect_text: str
    span: Tuple[int, int]
    current_sentiment: str
    confidence: float
    reason: str = ""
    linguistic_flags: List[str] = field(default_factory=list)
    visual_evidence: List[str] = field(default_factory=list)
    text_evidence: List[str] = field(default_factory=list)
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    is_locked: bool = False


class BACRController:
    """Unified Controller Policy orchestrating bidirectional cross-modal probes."""

    def __init__(
        self,
        max_turns: int = 2,
        confidence_threshold: float = 0.85
    ):
        self.max_turns = max_turns
        self.confidence_threshold = confidence_threshold

    def evaluate_termination(
        self,
        turn: int,
        beliefs: List[AspectBeliefState]
    ) -> bool:
        """Determines if the probing interaction should terminate."""
        if turn >= self.max_turns:
            return True
        # If all aspects have high confidence and no pending ambiguity
        if all(b.confidence >= self.confidence_threshold for b in beliefs):
            return True
        return False

    def select_probing_action(
        self,
        turn: int,
        beliefs: List[AspectBeliefState],
        raw_controller_response: Optional[Dict[str, Any]] = None
    ) -> ControllerDecision:
        """Parses or determines next probing action from controller output."""
        if self.evaluate_termination(turn, beliefs):
            return ControllerDecision(
                direction="NO_PROBE",
                gap_type="NO_GAP",
                gap_description="Budget exhausted or confidence threshold met",
                stop=True,
                reasoning="Max turns reached or high confidence"
            )

        if raw_controller_response:
            # Parse gap diagnosis
            gap_diag = raw_controller_response.get("gap_diagnosis", {})
            gap_type = gap_diag.get("gap_type", "NO_GAP") if isinstance(gap_diag, dict) else "NO_GAP"
            gap_desc = gap_diag.get("gap_description", "") if isinstance(gap_diag, dict) else ""

            # Parse direction
            direction_str = str(raw_controller_response.get("direction", "NONE")).upper()
            next_action = str(raw_controller_response.get("next_action", "STOP")).upper()

            if next_action in ["STOP"] or gap_type == "NO_GAP" or direction_str in ["NONE", "STOP"]:
                direction = "NO_PROBE"
                is_stop = True
            elif "VISUAL" in direction_str:
                direction = "VISUAL_PROBE"
                is_stop = False
            elif "TEXT" in direction_str:
                direction = "TEXT_PROBE"
                is_stop = False
            else:
                direction = "NO_PROBE"
                is_stop = True

            target_aspect_id = raw_controller_response.get("target_aspect_id") or raw_controller_response.get("target_aspect")

            return ControllerDecision(
                direction=direction,
                gap_type=gap_type,
                gap_description=gap_desc,
                target_aspect=target_aspect_id,
                question=raw_controller_response.get("question"),
                target_modality="image" if direction == "VISUAL_PROBE" else ("text" if direction == "TEXT_PROBE" else None),
                confidence_before=float(raw_controller_response.get("confidence", 0.5)),
                reasoning=raw_controller_response.get("decision_reason", raw_controller_response.get("reasoning", "")),
                stop=is_stop
            )

        return ControllerDecision(direction="NO_PROBE", gap_type="NO_GAP", stop=True)
