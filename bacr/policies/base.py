"""Controller Policy Abstractions for BACR-v3.

Decouples policy decision-making from environment execution:
- BaseControllerPolicy (ABC)
- GeminiTeacherPolicy (LLM Teacher Policy via MetaController)
- RulePolicy (Deterministic heuristic policy)
- RandomPolicy (Uniform random exploratory baseline)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import random

from bacr.schemas_v3 import ControllerAction, RiskDecision


class BaseControllerPolicy(ABC):
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Selects action given the current environment state.
        Returns dict with keys: 'action' (FINALIZE, TEXT_RETHINK, VISION_PROBE),
        'target_aspect_id', and optional metadata.
        """
        pass


class GeminiTeacherPolicy(BaseControllerPolicy):
    def __init__(self, meta_controller):
        self.meta_controller = meta_controller

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
        return c_d


class RulePolicy(BaseControllerPolicy):
    """Rule-based baseline: probes vision if missing affect, otherwise finalizes."""
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        aspect_states = state.get("aspect_states", {})
        budget = state.get("budget", 0)
        if budget <= 0 or not aspect_states:
            return {"action": ControllerAction.FINALIZE.value}

        # Find first active neutral aspect
        for aid, asp in aspect_states.items():
            if asp.get("baseline_sentiment") == "NEU" and asp.get("status") != "VERIFIED":
                return {
                    "action": ControllerAction.VISION_PROBE.value,
                    "target_aspect_id": aid,
                    "question_for_vision": f"What physical cues or facial expression are visible for {asp.get('text', 'the aspect')}?"
                }

        return {"action": ControllerAction.FINALIZE.value}


class RandomPolicy(BaseControllerPolicy):
    """Uniform random baseline for exploration / sanity checks."""
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        budget = state.get("budget", 0)
        aspect_states = state.get("aspect_states", {})
        if budget <= 0 or not aspect_states:
            return {"action": ControllerAction.FINALIZE.value}

        actions = [ControllerAction.FINALIZE.value, ControllerAction.TEXT_RETHINK.value, ControllerAction.VISION_PROBE.value]
        chosen_act = random.choice(actions)
        if chosen_act == ControllerAction.FINALIZE.value:
            return {"action": chosen_act}

        target_aid = random.choice(list(aspect_states.keys()))
        return {
            "action": chosen_act,
            "target_aspect_id": target_aid,
            "critique_for_text": "Re-examine linguistic context." if chosen_act == "TEXT_RETHINK" else None,
            "question_for_vision": "Examine visual scene." if chosen_act == "VISION_PROBE" else None
        }
