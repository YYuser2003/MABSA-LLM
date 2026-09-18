"""BACR Environment (BACREnv) for Aspect-Level MDP and Controller RL / GRPO.

Standard Gym-like RL interface:
    state = env.reset(sample)
    next_state, reward, done, info = env.step(action)

Action space:
    A_t = (action: FINALIZE | TEXT_RETHINK | VISION_PROBE, target_aspect_id: str)

State space:
    S_t = {
        "h_a": Immutable Text Anchor,
        "h_b": Current Verified Text Baseline,
        "aspect_states": Dict of AspectState,
        "v_0": Global Visual Sketch (Opportunity Map),
        "policy_history": Sanitized interaction history,
        "budget": Remaining deep budget,
        "done": bool
    }

Gold labels are strictly kept internal to calculate reward, never exposed to state!
"""

import os
import json
from typing import Dict, Any, Tuple, Optional, List

from bacr.client import BaseClient
from bacr.text_reasoner import TextReasoner
from bacr.vision_sensor import VisionSensor
from bacr.meta_controller import MetaController
from bacr.schemas_v3 import (
    ControllerAction,
    EvidenceFirewallOutput,
    TextRevisionAuditOutput,
    RevisionVerifierOutput,
    TransitionRecord
)


class BACREnv:
    def __init__(
        self,
        client: BaseClient,
        max_budget: int = 2
    ):
        self.client = client
        self.max_budget = max_budget
        self.text_reasoner = TextReasoner(client)
        self.vision_sensor = VisionSensor(client)
        self.meta_controller = MetaController(client)

        # Runtime state
        self.current_sample: Dict[str, Any] = {}
        self.image_base_dir: str = ""
        self.h_a: Dict[str, Any] = {}
        self.h_b: Dict[str, Any] = {}
        self.v_0: Dict[str, Any] = {}
        self.aspect_states: Dict[str, Any] = {}
        self.policy_history: List[Dict[str, Any]] = []
        self.audit_history: List[Dict[str, Any]] = []
        self.transitions: List[Dict[str, Any]] = []
        self.budget: int = max_budget
        self.step_count: int = 0
        self.done: bool = False

    def reset(self, sample: Dict[str, Any], image_base_dir: str = "") -> Dict[str, Any]:
        """Resets the environment for a new tweet sample and returns initial state S_0."""
        self.current_sample = sample
        self.image_base_dir = image_base_dir
        self.budget = self.max_budget
        self.step_count = 0
        self.done = False
        self.policy_history = []
        self.audit_history = []
        self.transitions = []

        text = sample.get("text", "")
        gold_pairs = sample.get("pairs", [])
        gold_aspects = [p[0] for p in gold_pairs]

        # Phase 1: TA -> H_A
        cached_t0 = sample.get("text_initial_cached")
        self.h_a, _, _ = self.text_reasoner.generate_anchor(
            text=text,
            target_aspects=gold_aspects,
            cached_t0=cached_t0
        )
        self.h_b = json.loads(json.dumps(self.h_a))

        # Build aspect states
        self.aspect_states = {}
        for a in self.h_a.get("aspects", []):
            aid = a.get("aspect_id", f"a_{len(self.aspect_states)+1:02d}")
            sent = a.get("sentiment", "NEU")
            self.aspect_states[aid] = {
                "aspect_id": aid,
                "text": a.get("text", ""),
                "span": a.get("span", [0, 0]),
                "anchor_sentiment": sent,
                "baseline_sentiment": sent,
                "current_candidate": None,
                "accepted_evidence": [],
                "risks": a.get("risks", []),
                "action_history": [],
                "status": "ACTIVE"
            }

        # Phase 2: IG -> V_0 (Control plane only)
        rel_img = sample.get("image", "")
        full_img = os.path.join(image_base_dir, rel_img) if image_base_dir else rel_img
        cached_v0 = sample.get("image_initial_cached")
        if cached_v0:
            if isinstance(cached_v0, dict) and "image_initial" in cached_v0:
                self.v_0 = cached_v0["image_initial"]
            else:
                self.v_0 = cached_v0
        else:
            self.v_0, _, _ = self.vision_sensor.perceive_global(full_img)

        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        """Returns observable state S_t for the Controller policy."""
        return {
            "h_a": self.h_a,
            "h_b": self.h_b,
            "aspect_states": self.aspect_states,
            "v_0": self.v_0,
            "policy_history": self.policy_history,
            "budget": self.budget,
            "done": self.done
        }

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Applies controller action and transitions to S_{t+1}.
        Returns (next_state, reward, done, info).
        """
        if self.done:
            return self._get_state(), 0.0, True, {"msg": "Already done"}

        self.step_count += 1
        action = action_dict.get("action", ControllerAction.FINALIZE.value)
        target_aid = action_dict.get("target_aspect_id")
        text = self.current_sample.get("text", "")
        rel_img = self.current_sample.get("image", "")
        full_img = os.path.join(self.image_base_dir, rel_img) if self.image_base_dir else rel_img

        target_asp_obj = next((a for a in self.h_b.get("aspects", []) if a.get("aspect_id") == target_aid), None)
        target_asp_text = target_asp_obj["text"] if target_asp_obj else ""
        pre_sentiment = target_asp_obj.get("sentiment", "NEU") if target_asp_obj else "NEU"

        if action == ControllerAction.FINALIZE.value or self.budget <= 0:
            self.done = True
            reward = self._calculate_final_reward()
            return self._get_state(), reward, True, {"action": "FINALIZE"}

        reward = 0.0

        if action == ControllerAction.TEXT_RETHINK.value:
            self.budget -= 1
            # Physically decoupled critique
            critique = action_dict.get("critique_for_text")
            if not critique:
                critique, _, _ = self.meta_controller.generate_text_critique(
                    h_b=self.h_b,
                    risk_diagnosis=action_dict.get("risk_diagnosis", {}),
                    target_aspect_id=target_aid,
                    target_aspect_text=target_asp_text
                )

            # TR: Text Rethink
            h_rethink, _, _ = self.text_reasoner.rethink_text(
                text=text,
                h_baseline=self.h_b,
                critique=critique,
                target_aspect_id=target_aid
            )
            cand_sent = next((a["sentiment"] for a in h_rethink.get("aspects", []) if a.get("aspect_id") == target_aid), pre_sentiment)

            # CT: Text Verifier Audit
            c_t, _, _ = self.meta_controller.audit_text_revision(
                h_b=self.h_b,
                h_rethink=h_rethink,
                critique=critique,
                target_aspect_id=target_aid
            )
            ct_decision = c_t.get("decision", "REVERT_TEXT_BASELINE")

            post_sentiment = pre_sentiment
            if ct_decision == "ACCEPT_TEXT_REVISION":
                self.h_b = json.loads(json.dumps(h_rethink))
                post_sentiment = cand_sent
                if target_aid and target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["baseline_sentiment"] = post_sentiment
                    self.aspect_states[target_aid]["status"] = "VERIFIED"

            self.policy_history.append({
                "step": self.step_count,
                "action": "TEXT_RETHINK",
                "target_aspect_id": target_aid,
                "decision": ct_decision
            })

            self.transitions.append(TransitionRecord(
                sample_id=self.current_sample.get("sample_id", "unknown"),
                aspect_id=target_aid,
                step=self.step_count,
                pre_sentiment=pre_sentiment,
                action="TEXT_RETHINK",
                critique_or_question=critique,
                candidate_sentiment=cand_sent,
                verifier_decision=ct_decision,
                post_sentiment=post_sentiment,
                done=(self.budget <= 0)
            ).model_dump())

            reward = self._step_reward(pre_sentiment, post_sentiment, target_asp_text)

        elif action == ControllerAction.VISION_PROBE.value:
            self.budget -= 1
            question = action_dict.get("question_for_vision") or f"Check visual facts for {target_asp_text}"
            
            # IP: Probe Deep
            raw_ev, _, _ = self.vision_sensor.probe_deep(image_path=full_img, question=question)

            # CE: Fail-Closed Firewall
            filtered_ev, _, _ = self.meta_controller.filter_evidence_firewall(
                question=question,
                raw_evidence=raw_ev,
                target_aspect_id=target_aid,
                target_aspect_text=target_asp_text
            )

            # TF: Evidence Fusion
            h_fusion, _, _ = self.text_reasoner.fuse_evidence(
                text=text,
                h_baseline=self.h_b,
                verified_evidence=filtered_ev,
                step_ref=f"probe_{self.step_count}",
                target_aspect_id=target_aid
            )
            cand_sent = next((a["sentiment"] for a in h_fusion.get("aspects", []) if a.get("aspect_id") == target_aid), pre_sentiment)

            # CV: Revision Verifier
            c_v, _, _ = self.meta_controller.verify_revision(
                h_a=self.h_a,
                h_current=h_fusion,
                verified_evidence=filtered_ev,
                h_b=self.h_b,
                budget=self.budget
            )
            cv_decision = c_v.get("decision", "REVERT_TEXT_BASELINE")

            post_sentiment = pre_sentiment
            if cv_decision == "ACCEPT_REVISION":
                self.h_b = json.loads(json.dumps(h_fusion))
                post_sentiment = cand_sent
                if target_aid and target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["baseline_sentiment"] = post_sentiment
                    self.aspect_states[target_aid]["status"] = "VERIFIED"

            self.policy_history.append({
                "step": self.step_count,
                "action": "VISION_PROBE",
                "target_aspect_id": target_aid,
                "decision": cv_decision
            })

            self.transitions.append(TransitionRecord(
                sample_id=self.current_sample.get("sample_id", "unknown"),
                aspect_id=target_aid,
                step=self.step_count,
                pre_sentiment=pre_sentiment,
                action="VISION_PROBE",
                critique_or_question=question,
                candidate_sentiment=cand_sent,
                verifier_decision=cv_decision,
                post_sentiment=post_sentiment,
                done=(self.budget <= 0)
            ).model_dump())

            reward = self._step_reward(pre_sentiment, post_sentiment, target_asp_text)

        if self.budget <= 0:
            self.done = True

        info = {
            "step": self.step_count,
            "action": action,
            "target_aspect": target_asp_text,
            "final_pairs": self.h_b.get("pairs", [])
        }
        return self._get_state(), reward, self.done, info

    def _step_reward(self, pre: str, post: str, aspect_text: str) -> float:
        """Step reward based on marginal utility against gold."""
        gold_pairs = {p[0]: p[1] for p in self.current_sample.get("pairs", [])}
        gold = gold_pairs.get(aspect_text)
        if not gold:
            return 0.0
        if pre != gold and post == gold:
            return 1.0   # Recovery reward
        elif pre == gold and post != gold:
            return -1.0  # Harm penalty
        return 0.0

    def _calculate_final_reward(self) -> float:
        """Calculates final end-to-end task accuracy reward."""
        gold_set = {tuple(p) for p in self.current_sample.get("pairs", [])}
        pred_set = {tuple(p) for p in self.h_b.get("pairs", [])}
        return 1.0 if gold_set == pred_set else -0.5
