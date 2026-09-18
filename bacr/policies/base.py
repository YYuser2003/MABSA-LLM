"""Controller Policy and Query Generator Abstractions for BACR-v3.

Decouples routing from query generation for tractable RL credit assignment:
- RouterPolicy: pi_R(S_t) -> (action: FINALIZE | TEXT_RETHINK | VISION_PROBE, target_aspect_id)
- TextCritiqueGenerator: q_t^T = C_Q^T(H_B, text_risk) (Frozen / Decoupled)
- VisualQuestionGenerator: q_t^V = C_Q^V(H_B, V_0, aspect) (Frozen / Decoupled)

Policies:
- GeminiTeacherPolicy (Full teacher policy via MetaController)
- RouterControllerPolicy (Modular policy combining Router + Generators)
- RulePolicy (Deterministic heuristic policy)
- RandomPolicy (Uniform exploratory policy respecting action_mask)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import random
import json

from bacr.schemas_v3 import ControllerAction, RiskDecision
from bacr.client import BaseClient


class BaseControllerPolicy(ABC):
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Selects action given the current environment state.
        Returns dict with keys: 'action' (FINALIZE, TEXT_RETHINK, VISION_PROBE),
        'target_aspect_id', and optional queries.
        """
        pass


class BaseRouterPolicy(ABC):
    @abstractmethod
    def select_route(self, state: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Outputs (action, target_aspect_id) respecting state['action_mask']."""
        pass


class TextCritiqueGenerator:
    """Generates targeted linguistic critique without visual access."""
    def __init__(self, meta_controller):
        self.meta_controller = meta_controller

    def generate(self, h_b: Dict[str, Any], text_risk: Any, target_aid: str, target_text: str) -> str:
        critique, _, _ = self.meta_controller.generate_text_critique(
            h_b=h_b,
            risk_diagnosis=text_risk,
            target_aspect_id=target_aid,
            target_aspect_text=target_text
        )
        return critique


class VisualQuestionGenerator:
    """Generates focused visual inquiry given target aspect and V0 opportunities."""
    def __init__(self, client: Optional[BaseClient] = None):
        self.client = client

    def generate(self, h_b: Dict[str, Any], v_0: Dict[str, Any], target_aid: str, target_text: str) -> str:
        # Default targeted heuristic query
        return f"Examine physical facial expressions, gestures, and clothing text for {target_text}."


class GeminiTeacherPolicy(BaseControllerPolicy):
    """Full LLM Teacher Policy executing Risk Diagnosis via MetaController."""

    def __init__(self, meta_controller=None, client: Optional[BaseClient] = None):
        if meta_controller is not None:
            self.meta_controller = meta_controller
        elif client is not None:
            from bacr.meta_controller import MetaController
            self.meta_controller = MetaController(client)
        else:
            raise ValueError("GeminiTeacherPolicy requires either meta_controller or client.")

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        action_mask = state.get("action_mask", [a.value for a in ControllerAction])
        target_aid = state.get("target_aspect_id")

        # 1. Check for pending escalation or query-again recommendations from previous verifier turn
        rec_action = state.get("recommended_next_action")
        if rec_action and rec_action in action_mask:
            q_gap = state.get("pending_visual_gap")
            return {
                "action": rec_action,
                "target_aspect_id": target_aid,
                "question_for_vision": q_gap,
                "decision_reason": f"Following verifier recommendation: {rec_action}"
            }

        # 2. Standard risk diagnosis
        h_a = state.get("h_a", {})
        h_b = state.get("h_b", {})
        v0 = state.get("v_0", {})
        aspect_states = state.get("aspect_states", {})
        policy_history = state.get("policy_history", [])
        budget = state.get("budget", 2)

        c_d, usage, lat = self.meta_controller.diagnose_risk(
            h_a=h_a,
            v0=v0,
            h_b=h_b,
            aspect_states=aspect_states,
            history=policy_history,
            budget=budget
        )

        # Enforce action mask
        if c_d.get("action") not in action_mask:
            c_d["action"] = ControllerAction.FINALIZE.value

        return c_d


class DecoupledTeacherPolicy(BaseControllerPolicy):
    """Teacher Policy executing Decoupled 3-stage control:
    C_T^risk(H_A, H_B) -> C_V^opp(a, V_0) -> C_R(R_T, O_V, History, Budget)
    """

    def __init__(self, meta_controller=None, client: Optional[BaseClient] = None):
        if meta_controller is not None:
            self.meta_controller = meta_controller
        elif client is not None:
            from bacr.meta_controller import MetaController
            self.meta_controller = MetaController(client)
        else:
            raise ValueError("DecoupledTeacherPolicy requires either meta_controller or client.")

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        action_mask = state.get("action_mask", [a.value for a in ControllerAction])
        target_aid = state.get("target_aspect_id")
        target_asp = state.get("target_aspect", {})
        target_text = target_asp.get("text", "") if isinstance(target_asp, dict) else str(target_asp)

        # 1. Check for pending escalation recommendation
        rec_action = state.get("recommended_next_action")
        if rec_action and rec_action in action_mask:
            q_gap = state.get("pending_visual_gap")
            return {
                "action": rec_action,
                "target_aspect_id": target_aid,
                "question_for_vision": q_gap,
                "decision_reason": f"Following verifier recommendation: {rec_action}"
            }

        # 2. Decoupled Stage A: Purely Textual Risk Diagnosis (zero visual access)
        h_a = state.get("h_a", {})
        h_b = state.get("h_b", {})
        text_risk, _, _ = self.meta_controller.diagnose_text_risk(
            h_a=h_a,
            h_b=h_b,
            target_aspect_text=target_text,
            target_aspect_id=target_aid
        )

        # 3. Decoupled Stage B: Physical Visual Opportunity (zero text hypotheses)
        v0 = state.get("v_0", {})
        visual_opp, _, _ = self.meta_controller.assess_visual_opportunity(
            target_aspect_text=target_text,
            v_0=v0
        )

        # 4. Decoupled Stage C: Discrete Routing
        policy_history = state.get("policy_history", [])
        budget = state.get("budget", 2)
        pending_gap = state.get("pending_visual_gap")
        route_dec, _, _ = self.meta_controller.route_action(
            text_risk=text_risk,
            visual_opportunity=visual_opp,
            history=policy_history,
            budget=budget,
            action_mask=action_mask,
            pending_visual_gap=pending_gap
        )

        chosen_action = route_dec.get("action", ControllerAction.FINALIZE.value)
        out_dict = {
            "action": chosen_action,
            "target_aspect_id": target_aid,
            "text_risk": text_risk,
            "visual_opportunity": visual_opp,
            "decision_reason": route_dec.get("rationale", "")
        }

        if chosen_action == ControllerAction.TEXT_RETHINK.value:
            out_dict["critique_for_text"] = None  # Will be generated or populated
        elif chosen_action == ControllerAction.VISION_PROBE.value:
            out_dict["query_type"] = route_dec.get("query_type")
            out_dict["question_for_vision"] = self.meta_controller.generate_visual_question(
                target_aspect_text=target_text,
                v_0=v0,
                pending_gap=pending_gap,
                query_type=route_dec.get("query_type")
            )

        return out_dict


class RulePolicy(BaseControllerPolicy):
    """Rule-based baseline: probes vision for active neutral aspects, otherwise finalizes."""

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        action_mask = state.get("action_mask", [ControllerAction.FINALIZE.value])
        aspect_states = state.get("aspect_states", {})
        target_aid = state.get("target_aspect_id") or (list(aspect_states.keys())[0] if aspect_states else "a_01")

        rec_action = state.get("recommended_next_action")
        if rec_action and rec_action in action_mask:
            return {
                "action": rec_action,
                "target_aspect_id": target_aid,
                "question_for_vision": state.get("pending_visual_gap"),
                "decision_reason": f"Following recommendation: {rec_action}"
            }

        if ControllerAction.VISION_PROBE.value in action_mask and aspect_states:
            # Probe first active neutral aspect
            for aid, asp in aspect_states.items():
                if asp.get("baseline_sentiment") == "NEU" and asp.get("status") != "VERIFIED":
                    return {
                        "action": ControllerAction.VISION_PROBE.value,
                        "target_aspect_id": aid,
                        "question_for_vision": f"What physical cues or facial expression are visible for {asp.get('text', 'the aspect')}?"
                    }

        if ControllerAction.TEXT_RETHINK.value in action_mask and aspect_states:
            # Rethink first non-neutral aspect that has not been rethought
            for aid, asp in aspect_states.items():
                if asp.get("baseline_sentiment") in ["POS", "NEG"] and asp.get("status") != "VERIFIED":
                    return {
                        "action": ControllerAction.TEXT_RETHINK.value,
                        "target_aspect_id": aid,
                        "critique_for_text": f"Re-evaluate whether {asp.get('text')} is directly modified by sentiment or is reporting frame."
                    }

        return {"action": ControllerAction.FINALIZE.value}


class RandomPolicy(BaseControllerPolicy):
    """Uniform random baseline respecting action mask."""

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        action_mask = state.get("action_mask", [ControllerAction.FINALIZE.value])
        aspect_states = state.get("aspect_states", {})

        chosen_act = random.choice(action_mask)
        if chosen_act == ControllerAction.FINALIZE.value or not aspect_states:
            return {"action": ControllerAction.FINALIZE.value}

        target_aid = random.choice(list(aspect_states.keys()))
        target_text = aspect_states[target_aid].get("text", "the aspect")

        return {
            "action": chosen_act,
            "target_aspect_id": target_aid,
            "critique_for_text": f"Re-examine linguistic context for {target_text}." if chosen_act == "TEXT_RETHINK" else None,
            "question_for_vision": f"Examine visual scene for {target_text}." if chosen_act == "VISION_PROBE" else None
        }
