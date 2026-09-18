"""BACR Environment (BACREnv) for Aspect-Level MDP and Controller RL / GRPO.

Standard Gym-like RL interface:
    state = env.reset(sample, image_base_dir, task_mode)
    next_state, reward, done, info = env.step(action)

Action space:
    A_t = (action: FINALIZE | TEXT_RETHINK | VISION_PROBE, target_aspect_id: str)

State space S_t (Observable to Controller):
    - h_a: Immutable Text Anchor Ledger
    - h_b: Current Verified Text Baseline Ledger
    - aspect_states: Dict of AspectState
    - v_0: Global Visual Sketch (Opportunity Map)
    - policy_history: Sanitized interaction history (verifier decisions only)
    - budget_state: BudgetState with action mask
    - action_mask: List of valid actions
    - done: bool

GOLD LABELS ARE STRICTLY KEPT INTERNAL TO CALCULATE REWARD.
THEY NEVER APPEAR IN THE OBSERVABLE STATE S_t OR POLICY INPUTS!
"""

import os
import json
import time
from typing import Dict, Any, Tuple, Optional, List, Union

from bacr.client import BaseClient
from bacr.text_reasoner import TextReasoner
from bacr.vision_sensor import VisionSensor
from bacr.meta_controller import MetaController
from bacr.schemas_v3 import (
    ControllerAction,
    TaskMode,
    TextRiskItem,
    VisualOpportunityItem,
    RiskDecision,
    EvidenceFirewallOutput,
    EvidenceProbeItem,
    EvidenceBundle,
    TextRevisionAuditOutput,
    RevisionVerifierOutput,
    BudgetState,
    TransitionRecord
)


class BACREnv:
    def __init__(
        self,
        client: BaseClient,
        budget_config: Optional[Dict[str, int]] = None,
        task_mode: Union[TaskMode, str] = TaskMode.TARGET_GUIDED
    ):
        self.client = client
        b_cfg = budget_config or {}
        self.max_deep_actions = b_cfg.get("max_deep_actions", 2)
        self.max_text_rethinks = b_cfg.get("max_text_rethinks", 1)
        self.max_visual_probes = b_cfg.get("max_visual_probes", 2)

        if isinstance(task_mode, str):
            task_mode = TaskMode(task_mode)
        self.default_task_mode = task_mode

        # Modular Agentic Roles
        self.text_reasoner = TextReasoner(client)
        self.vision_sensor = VisionSensor(client)
        self.meta_controller = MetaController(client)

        # Runtime state
        self.current_sample: Dict[str, Any] = {}
        self.image_base_dir: str = ""
        self.task_mode: TaskMode = self.default_task_mode
        self.h_a: Dict[str, Any] = {}
        self.h_b: Dict[str, Any] = {}
        self.v_0: Dict[str, Any] = {}
        self.aspect_states: Dict[str, Any] = {}
        self.policy_history: List[Dict[str, Any]] = []
        self.audit_history: List[Dict[str, Any]] = []
        self.transitions: List[Dict[str, Any]] = []
        self.evidence_bundles: List[Dict[str, Any]] = []
        self.initial_c_d: Optional[Dict[str, Any]] = None
        self.budget_state: BudgetState = BudgetState()
        self.step_count: int = 0
        self.done: bool = False

        # Compute accounting
        self.compute = {
            "api_calls_text": 0,
            "api_calls_vision": 0,
            "api_calls_controller": 0,
            "api_calls_total": 0,
            "total_tokens": 0,
            "image_invocations": 0,
            "latency_ms": 0.0
        }

    def _record_usage(self, usage: Dict[str, int], lat: float, role: str):
        self.compute["api_calls_total"] += 1
        self.compute["latency_ms"] += lat
        self.compute["total_tokens"] += usage.get("total_tokens", 0)
        if role == "text":
            self.compute["api_calls_text"] += 1
        elif role == "vision":
            self.compute["api_calls_vision"] += 1
            self.compute["image_invocations"] += 1
        elif role == "controller":
            self.compute["api_calls_controller"] += 1

    def reset(
        self,
        sample: Dict[str, Any],
        image_base_dir: str = "",
        task_mode: Optional[Union[TaskMode, str]] = None
    ) -> Dict[str, Any]:
        """Resets the environment for a new sample and returns observable state S_0."""
        self.current_sample = sample
        self.image_base_dir = image_base_dir
        if task_mode is not None:
            self.task_mode = TaskMode(task_mode) if isinstance(task_mode, str) else task_mode
        else:
            self.task_mode = self.default_task_mode

        self.step_count = 0
        self.done = False
        self.policy_history = []
        self.audit_history = []
        self.transitions = []
        self.evidence_bundles = []
        self.initial_c_d = None

        self.budget_state = BudgetState(
            max_deep_actions=self.max_deep_actions,
            max_text_rethinks=self.max_text_rethinks,
            max_visual_probes=self.max_visual_probes,
            deep_remaining=self.max_deep_actions,
            text_remaining=self.max_text_rethinks,
            vision_remaining=self.max_visual_probes
        )

        self.compute = {
            "api_calls_text": 0,
            "api_calls_vision": 0,
            "api_calls_controller": 0,
            "api_calls_total": 0,
            "total_tokens": 0,
            "image_invocations": 0,
            "latency_ms": 0.0
        }

        sample_id = sample.get("sample_id", "unknown")
        text = sample.get("text", "")

        # Target-Guided vs Open Joint Mode
        if self.task_mode == TaskMode.TARGET_GUIDED:
            gold_pairs = sample.get("pairs", [])
            target_aspects = [p[0] for p in gold_pairs]
        else:
            # Open Joint: Zero access to gold pairs during inference!
            target_aspects = None

        # Phase 1: TA -> Immutable H_A
        cached_t0 = sample.get("text_initial_cached")
        self.h_a, u_ta, l_ta = self.text_reasoner.generate_anchor(
            text=text,
            target_aspects=target_aspects,
            cached_t0=cached_t0
        )
        self._record_usage(u_ta, l_ta, "text")
        self.h_b = json.loads(json.dumps(self.h_a))

        # Build aspect states with persistent aspect_uid
        self.aspect_states = {}
        for idx, a in enumerate(self.h_a.get("aspects", [])):
            aid = a.get("aspect_id", f"a_{idx+1:02d}")
            asp_text = a.get("text", "")
            span = a.get("span", [0, 0])
            sent = a.get("sentiment", "NEU")
            uid = f"{sample_id}_{asp_text}_{span[0]}_{span[1]}"
            self.aspect_states[aid] = {
                "aspect_id": aid,
                "text": asp_text,
                "span": span,
                "aspect_uid": uid,
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
            self.v_0, u_ig, l_ig = self.vision_sensor.perceive_global(full_img)
            self._record_usage(u_ig, l_ig, "vision")

        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        """Returns observable state S_t for Controller Policy (Gold labels excluded!)."""
        valid_actions = self.budget_state.get_valid_actions()
        return {
            "h_a": self.h_a,
            "h_b": self.h_b,
            "aspect_states": self.aspect_states,
            "v_0": self.v_0,
            "policy_history": self.policy_history,
            "budget": self.budget_state.deep_remaining,
            "budget_state": self.budget_state.model_dump(),
            "action_mask": valid_actions,
            "done": self.done
        }

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Executes one intervention action and transitions to S_{t+1}.
        Returns (next_state, reward, done, info).
        """
        if self.done:
            return self._get_state(), 0.0, True, {"msg": "Already done"}

        if self.initial_c_d is None:
            self.initial_c_d = action_dict

        action = action_dict.get("action", ControllerAction.FINALIZE.value)
        target_aid = action_dict.get("target_aspect_id")

        # Action masking check
        valid_actions = self.budget_state.get_valid_actions()
        if action not in valid_actions:
            action = ControllerAction.FINALIZE.value

        sample_id = self.current_sample.get("sample_id", "unknown")
        text = self.current_sample.get("text", "")
        rel_img = self.current_sample.get("image", "")
        full_img = os.path.join(self.image_base_dir, rel_img) if self.image_base_dir else rel_img

        target_asp_obj = next((a for a in self.h_b.get("aspects", []) if a.get("aspect_id") == target_aid), None)
        target_asp_text = target_asp_obj["text"] if target_asp_obj else ""
        target_span = target_asp_obj.get("span", [0, 0]) if target_asp_obj else [0, 0]
        target_uid = self.aspect_states.get(target_aid, {}).get("aspect_uid") if target_aid else None
        pre_sentiment = target_asp_obj.get("sentiment", "NEU") if target_asp_obj else "NEU"

        if action == ControllerAction.FINALIZE.value or self.budget_state.deep_remaining <= 0:
            self.done = True
            reward = self._calculate_final_reward()
            return self._get_state(), reward, True, {"action": "FINALIZE"}

        step_reward = 0.0

        # ====================================================================
        # Action: TEXT_RETHINK
        # ====================================================================
        if action == ControllerAction.TEXT_RETHINK.value:
            self.budget_state.consume_text()
            self.step_count += 1
            text_step = self.step_count

            # Physical critique decoupling: receives ONLY text risk, zero visual access
            critique = action_dict.get("critique_for_text")
            if not critique:
                text_risk = action_dict.get("text_risk") or action_dict.get("risk_diagnosis", {})
                critique, u_cq, l_cq = self.meta_controller.generate_text_critique(
                    h_b=self.h_b,
                    risk_diagnosis=text_risk,
                    target_aspect_id=target_aid,
                    target_aspect_text=target_asp_text
                )
                self._record_usage(u_cq, l_cq, "controller")

            # TR: Text Rethink
            h_rethink, u_tr, l_tr = self.text_reasoner.rethink_text(
                text=text,
                h_baseline=self.h_b,
                critique=critique,
                target_aspect_id=target_aid
            )
            self._record_usage(u_tr, l_tr, "text")
            candidate_sent = next((a["sentiment"] for a in h_rethink.get("aspects", []) if a.get("aspect_id") == target_aid), pre_sentiment)

            # CT: Text Verifier Audit
            c_t, u_ct, l_ct = self.meta_controller.audit_text_revision(
                h_b=self.h_b,
                h_rethink=h_rethink,
                critique=critique,
                target_aspect_id=target_aid
            )
            self._record_usage(u_ct, l_ct, "controller")
            ct_decision = c_t.get("decision", "REVERT_TEXT_BASELINE")

            post_sentiment = pre_sentiment
            baseline_mutated = False

            if ct_decision == "ACCEPT_TEXT_REVISION":
                self.h_b = json.loads(json.dumps(h_rethink))
                post_sentiment = candidate_sent
                baseline_mutated = True
                if target_aid and target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["baseline_sentiment"] = post_sentiment
                    self.aspect_states[target_aid]["status"] = "VERIFIED"
                    self.aspect_states[target_aid]["action_history"].append("TEXT_RETHINK:ACCEPT")
            else:
                if target_aid and target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["action_history"].append("TEXT_RETHINK:REVERT")

            # Invariant: LOG TEXT TRANSITION IMMEDIATELY BEFORE ANY ESCALATION!
            self.transitions.append(TransitionRecord(
                sample_id=sample_id,
                aspect_id=target_aid,
                aspect_text=target_asp_text,
                aspect_span=target_span,
                aspect_uid=target_uid,
                step=text_step,
                pre_sentiment=pre_sentiment,
                action="TEXT_RETHINK",
                critique_or_question=critique,
                observation={"critique": critique},
                candidate_sentiment=candidate_sent,
                verifier_decision=ct_decision,
                post_sentiment=post_sentiment,
                done=(self.budget_state.deep_remaining <= 0)
            ).model_dump())

            self.audit_history.append({
                "step": text_step,
                "action": "TEXT_RETHINK",
                "target_aspect_id": target_aid,
                "critique": critique,
                "revised_candidate": h_rethink,
                "ct_decision": c_t,
                "baseline_mutated": baseline_mutated,
                "pairs": self.h_b.get("pairs", [])
            })

            self.policy_history.append({
                "step": text_step,
                "action": "TEXT_RETHINK",
                "target_aspect_id": target_aid,
                "decision": ct_decision,
                "baseline_mutated": baseline_mutated
            })

            step_reward += self._step_reward(pre_sentiment, post_sentiment, target_asp_text, action_type="TEXT", decision=ct_decision)

            # Escalation to Vision if CT triggers ESCALATE_TO_VISION
            if ct_decision == "ESCALATE_TO_VISION" and self.budget_state.vision_remaining > 0 and self.budget_state.deep_remaining > 0:
                v_gap = c_t.get("visual_gap") or f"Check physical visual facts for {target_asp_text}"
                esc_reward = self._execute_visual_probe_loop(
                    target_aid=target_aid,
                    target_asp_text=target_asp_text,
                    target_span=target_span,
                    target_uid=target_uid,
                    initial_question=v_gap,
                    text=text,
                    full_img=full_img,
                    sample_id=sample_id,
                    pre_sentiment=post_sentiment
                )
                step_reward += esc_reward

        # ====================================================================
        # Action: VISION_PROBE
        # ====================================================================
        elif action == ControllerAction.VISION_PROBE.value:
            question = action_dict.get("question_for_vision") or action_dict.get("question") or f"Check visual facts for {target_asp_text}"
            step_reward += self._execute_visual_probe_loop(
                target_aid=target_aid,
                target_asp_text=target_asp_text,
                target_span=target_span,
                target_uid=target_uid,
                initial_question=question,
                text=text,
                full_img=full_img,
                sample_id=sample_id,
                pre_sentiment=pre_sentiment
            )

        if self.budget_state.deep_remaining <= 0:
            self.done = True

        info = {
            "step": self.step_count,
            "action": action,
            "target_aspect": target_asp_text,
            "final_pairs": self.h_b.get("pairs", [])
        }
        return self._get_state(), step_reward, self.done, info

    def _execute_visual_probe_loop(
        self,
        target_aid: Optional[str],
        target_asp_text: str,
        target_span: List[int],
        target_uid: Optional[str],
        initial_question: str,
        text: str,
        full_img: str,
        sample_id: str,
        pre_sentiment: str
    ) -> float:
        """Executes multi-turn visual probe loop with structured EvidenceBundle aggregation.
        Strict Invariant: Never mutates H_B until CV explicitly issues ACCEPT_REVISION.
        """
        current_q = initial_question
        probe_history_raw: List[Dict[str, Any]] = []
        accumulated_facts: List[str] = []
        final_cv_dec = "REVERT_TEXT_BASELINE"
        total_loop_reward = 0.0

        while self.budget_state.deep_remaining > 0 and self.budget_state.vision_remaining > 0 and current_q:
            self.budget_state.consume_vision()
            self.step_count += 1
            probe_step = self.step_count
            step_ref = f"vision_probe_{probe_step}"

            # 1. IP: Targeted Deep Visual Probe
            raw_ev, u_ip, l_ip = self.vision_sensor.probe_deep(
                image_path=full_img,
                question=current_q
            )
            self._record_usage(u_ip, l_ip, "vision")

            # 2. CE: Strictly Fail-Closed Evidence Firewall
            filtered_ev, u_ce, l_ce = self.meta_controller.filter_evidence_firewall(
                question=current_q,
                raw_evidence=raw_ev,
                target_aspect_id=target_aid,
                target_aspect_text=target_asp_text
            )
            self._record_usage(u_ce, l_ce, "controller")
            filtered_ev["probe_id"] = step_ref
            filtered_ev["question"] = current_q
            probe_history_raw.append(filtered_ev)

            # 3. CA: Aggregate into structured EvidenceBundle
            evidence_bundle = self.meta_controller.aggregate_evidence_bundle(
                probes=probe_history_raw,
                target_aspect_id=target_aid
            )
            self.evidence_bundles.append(evidence_bundle.model_dump())

            # 4. TF: Evidence Fusion Reasoner on immutable current H_B
            h_fusion, u_tf, l_tf = self.text_reasoner.fuse_evidence(
                text=text,
                h_baseline=self.h_b,
                verified_evidence=evidence_bundle,
                step_ref=step_ref,
                target_aspect_id=target_aid
            )
            self._record_usage(u_tf, l_tf, "text")

            cand_sent = next((a["sentiment"] for a in h_fusion.get("aspects", []) if a.get("aspect_id") == target_aid), pre_sentiment)

            # 5. CV: Revision Verifier anchored to H_B
            c_v, u_cv, l_cv = self.meta_controller.verify_revision(
                h_a=self.h_a,
                h_current=h_fusion,
                verified_evidence=evidence_bundle.model_dump(),
                h_b=self.h_b,
                budget=self.budget_state.deep_remaining
            )
            self._record_usage(u_cv, l_cv, "controller")
            cv_dec = c_v.get("decision", "REVERT_TEXT_BASELINE")
            final_cv_dec = cv_dec

            post_sentiment = cand_sent if cv_dec == "ACCEPT_REVISION" else pre_sentiment
            baseline_mutated = (cv_dec == "ACCEPT_REVISION")

            if baseline_mutated:
                self.h_b = json.loads(json.dumps(h_fusion))
                if target_aid and target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["baseline_sentiment"] = post_sentiment
                    self.aspect_states[target_aid]["accepted_evidence"].extend(
                        [f["content"] for f in evidence_bundle.accepted_facts]
                    )
                    self.aspect_states[target_aid]["status"] = "VERIFIED"
                    self.aspect_states[target_aid]["action_history"].append("VISION_PROBE:ACCEPT")

            # Log each turn as an explicit transition with strictly monotonic step
            self.transitions.append(TransitionRecord(
                sample_id=sample_id,
                aspect_id=target_aid,
                aspect_text=target_asp_text,
                aspect_span=target_span,
                aspect_uid=target_uid,
                step=probe_step,
                pre_sentiment=pre_sentiment,
                action="VISION_PROBE",
                critique_or_question=current_q,
                observation={"sanitized_evidence": [f["content"] for f in evidence_bundle.accepted_facts]},
                candidate_sentiment=cand_sent,
                verifier_decision=cv_dec,
                post_sentiment=post_sentiment,
                done=(self.budget_state.deep_remaining <= 0)
            ).model_dump())

            self.audit_history.append({
                "step": probe_step,
                "action": "VISION_PROBE",
                "target_aspect_id": target_aid,
                "question": current_q,
                "raw_evidence": raw_ev,
                "filtered_evidence": filtered_ev,
                "evidence_bundle": evidence_bundle.model_dump(),
                "candidate_fusion": h_fusion,
                "verifier_decision": c_v,
                "baseline_mutated": baseline_mutated,
                "pairs": self.h_b.get("pairs", [])
            })

            self.policy_history.append({
                "step": probe_step,
                "action": "VISION_PROBE",
                "target_aspect_id": target_aid,
                "sanitized_evidence": [f["content"] for f in evidence_bundle.accepted_facts],
                "verifier_decision": cv_dec,
                "baseline_mutated": baseline_mutated
            })

            turn_reward = self._step_reward(pre_sentiment, post_sentiment, target_asp_text, action_type="VISION", decision=cv_dec)
            total_loop_reward += turn_reward

            if cv_dec == "QUERY_AGAIN" and self.budget_state.deep_remaining > 0:
                current_q = c_v.get("next_question") or ""
            else:
                break

        return total_loop_reward

    def _step_reward(
        self,
        pre: str,
        post: str,
        aspect_text: str,
        action_type: str = "TEXT",
        decision: str = "ACCEPT"
    ) -> float:
        """Asymmetric aspect-level step reward.
        R_t = alpha * I(W -> C) - beta * I(C -> W) - lambda_T * I(TEXT) - lambda_V * I(VISION) - lambda_U * I(unsupported)
        where beta (1.5) > alpha (1.0), heavily penalizing harmful modifications.
        """
        gold_pairs = {p[0]: p[1] for p in self.current_sample.get("pairs", [])}
        gold = gold_pairs.get(aspect_text)
        reward = 0.0

        # Cost penalties
        lambda_t = 0.05
        lambda_v = 0.10
        lambda_u = 0.20

        if action_type == "TEXT":
            reward -= lambda_t
        elif action_type == "VISION":
            reward -= lambda_v

        if decision in ["REVERT_TEXT_BASELINE", "REVERT_ANCHOR", "REJECT_REVISION"]:
            reward -= lambda_u

        if not gold:
            return round(reward, 4)

        alpha = 1.0
        beta = 1.5

        if pre != gold and post == gold:
            reward += alpha  # Rescued
        elif pre == gold and post != gold:
            reward -= beta   # Harmed

        return round(reward, 4)

    def _calculate_final_reward(self) -> float:
        """Calculates final end-to-end task accuracy reward."""
        gold_set = {tuple(p) for p in self.current_sample.get("pairs", [])}
        pred_set = {tuple(p) for p in self.h_b.get("pairs", [])}
        return 1.0 if gold_set == pred_set else -0.5

    def export_trajectory(self) -> Dict[str, Any]:
        """Exports the full trajectory conforming to the standardized evaluation & SFT format."""
        sample_id = self.current_sample.get("sample_id", "unknown")
        text = self.current_sample.get("text", "")
        rel_img = self.current_sample.get("image", "")
        gold_pairs = self.current_sample.get("pairs", [])

        y_a = self.h_a.get("pairs", [])
        y_final = self.h_b.get("pairs", [])

        num_v_probes = len([t for t in self.transitions if t["action"] == "VISION_PROBE"])
        num_t_rethinks = len([t for t in self.transitions if t["action"] == "TEXT_RETHINK"])

        risk_type = (self.initial_c_d or {}).get("risk_diagnosis", {}).get("risk_type", "NO_RISK")
        main_action = (self.initial_c_d or {}).get("action", "FINALIZE")

        self.compute["latency_ms"] = round(self.compute["latency_ms"], 1)

        return {
            "sample_id": sample_id,
            "text": text,
            "image": rel_img,
            "gold_pairs": gold_pairs,
            "task_mode": self.task_mode.value,
            "contrast_type": risk_type,
            "action": main_action,
            "text_anchor": {
                "ledger": self.h_a,
                "pairs": y_a
            },
            "text_baseline": {
                "ledger": self.h_b,
                "pairs": y_final
            },
            "text_only": {
                "ledger": self.h_a,
                "pairs": y_a
            },
            "vision_opportunity_map": self.v_0,
            "vision_initial": self.v_0,
            "initial_contrast": self.initial_c_d or {},
            "risk_diagnosis": self.initial_c_d or {},
            "aspect_states": self.aspect_states,
            "transitions": self.transitions,
            "rounds": self.audit_history,
            "policy_history": self.policy_history,
            "evidence_bundles": self.evidence_bundles,
            "final_ledger": self.h_b,
            "final_pairs": y_final,
            "num_visual_probes": num_v_probes,
            "num_text_queries": num_t_rethinks,
            "num_queries_total": num_v_probes + num_t_rethinks,
            "compute": self.compute
        }
