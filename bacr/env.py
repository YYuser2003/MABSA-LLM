"""BACR Environment (BACREnv) for Aspect-Level MDP and Controller RL / GRPO.

Standard Gym-like RL interface:
    state = env.reset(sample, target_aspect=None, image_base_dir="", task_mode=None)
    next_state, reward, done, info = env.step(action)

Action space:
    A_t = FINALIZE | TEXT_RETHINK | VISION_PROBE

Guarantees:
1. Strict 1 Policy Action = 1 Env Transition:
   - TEXT_RETHINK executes T_R -> C_T. If escalated, returns S_{t+1} with pending_visual_gap.
     NO internal auto-cascading into vision!
   - VISION_PROBE executes 1 probe turn I_P -> C_E -> T_F -> C_V. If more evidence needed,
     returns S_{t+1} with pending_visual_gap. NO internal while-looping!
2. Single-Aspect Episode factorization:
   - Evaluates exactly one aspect decision at a time.
3. Strict Information Boundaries:
   - a_gold in S_t, but gold sentiment s* is strictly private in self._gold_sentiment.
   - Modality isolation: C_T^risk sees zero visual cues, C_V^opp sees zero sentiment hypotheses.
4. Anchor-Relative Asymmetric Reward:
   - +1.0 for rescue, -1.5 for harm, minus compute penalties lambda_T=0.03, lambda_V=0.08.
"""

import os
import json
import copy
from typing import Dict, Any, Tuple, Optional, List, Union

from bacr.client import BaseClient
from bacr.text_reasoner import TextReasoner
from bacr.vision_sensor import VisionSensor
from bacr.meta_controller import MetaController
from bacr.schemas_v3 import (
    ControllerAction,
    TaskMode,
    TargetAspect,
    AspectEpisodeState,
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
        client: Optional[BaseClient] = None,
        budget_config: Optional[Dict[str, int]] = None,
        task_mode: Union[TaskMode, str] = TaskMode.TARGET_GUIDED,
        text_client: Optional[BaseClient] = None,
        controller_client: Optional[BaseClient] = None,
        vision_client: Optional[BaseClient] = None
    ):
        self.client = client
        self.text_client = text_client or client
        self.controller_client = controller_client or client
        self.vision_client = vision_client or client

        if self.text_client is None:
            raise ValueError("BACREnv requires at least one client (client or text_client).")

        b_cfg = budget_config or {}
        self.max_deep_actions = b_cfg.get("max_deep_actions", 2)
        self.max_text_rethinks = b_cfg.get("max_text_rethinks", 1)
        self.max_visual_probes = b_cfg.get("max_visual_probes", 2)

        if isinstance(task_mode, str):
            task_mode = TaskMode(task_mode)
        self.default_task_mode = task_mode

        # Modular Agentic Roles with role-specific clients
        self.text_reasoner = TextReasoner(self.text_client)
        self.vision_sensor = VisionSensor(self.vision_client)
        self.meta_controller = MetaController(self.controller_client)

        # Public Context & Private Gold Store
        self.public_context: Dict[str, Any] = {}
        self._gold_sentiment: Optional[str] = None
        self._gold_pairs_map: Dict[str, str] = {}
        self.target_aspect: Optional[TargetAspect] = None
        self.image_base_dir: str = ""
        self.task_mode: TaskMode = self.default_task_mode

        # Runtime states
        self.h_a: Dict[str, Any] = {}
        self.h_b: Dict[str, Any] = {}
        self.v_0: Dict[str, Any] = {}
        self.aspect_states: Dict[str, Any] = {}
        self.aspect_episode_state: Optional[AspectEpisodeState] = None
        self.pending_visual_gap: Optional[str] = None
        self.recommended_next_action: Optional[str] = None
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
        target_aspect: Optional[Union[TargetAspect, Dict[str, Any], str]] = None,
        image_base_dir: str = "",
        task_mode: Optional[Union[TaskMode, str]] = None
    ) -> Dict[str, Any]:
        """Resets the environment for a single-aspect episode and returns observable state S_0.
        Gold sentiment is strictly kept private in self._gold_sentiment (NOT in observable state)!
        """
        sample_id = sample.get("sample_id", "unknown")
        text = sample.get("text", "")
        rel_img = sample.get("image", "")
        self.image_base_dir = image_base_dir

        if task_mode is not None:
            self.task_mode = TaskMode(task_mode) if isinstance(task_mode, str) else task_mode
        else:
            self.task_mode = self.default_task_mode

        # Strictly separate public context from private gold label
        self.public_context = {
            "sample_id": sample_id,
            "text": text,
            "image": rel_img
        }

        # Store gold pairs privately for reward computation only
        raw_pairs = sample.get("pairs", [])
        self._gold_pairs_map = {p[0].strip().lower(): p[1] for p in raw_pairs if len(p) >= 2}

        # Resolve single target aspect
        if target_aspect is not None:
            if isinstance(target_aspect, TargetAspect):
                self.target_aspect = target_aspect
            elif isinstance(target_aspect, dict):
                self.target_aspect = TargetAspect(
                    aspect_uid=target_aspect.get("aspect_uid", f"{sample_id}_{target_aspect.get('text', 'a')}_0_0"),
                    text=target_aspect.get("text", ""),
                    span=target_aspect.get("span", [0, 0])
                )
            else:
                asp_str = str(target_aspect)
                self.target_aspect = TargetAspect(
                    aspect_uid=f"{sample_id}_{asp_str}_0_0",
                    text=asp_str,
                    span=[0, len(asp_str)]
                )
        elif self.task_mode == TaskMode.TARGET_GUIDED and raw_pairs:
            # Default to first aspect in sample
            first_text = raw_pairs[0][0]
            span = [0, len(first_text)]
            self.target_aspect = TargetAspect(
                aspect_uid=f"{sample_id}_{first_text}_{span[0]}_{span[1]}",
                text=first_text,
                span=span
            )
        else:
            self.target_aspect = TargetAspect(
                aspect_uid=f"{sample_id}_target_0_0",
                text="target",
                span=[0, 0]
            )

        # Private gold sentiment lookup for target aspect
        self._gold_sentiment = self._gold_pairs_map.get(self.target_aspect.text.strip().lower())

        self.step_count = 0
        self.done = False
        self.pending_visual_gap = None
        self.recommended_next_action = None
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

        # Target-Guided vs Open Joint Mode
        if self.task_mode == TaskMode.TARGET_GUIDED:
            # In single aspect episode, target aspect is explicitly guided
            target_aspects = [self.target_aspect.text] if self.target_aspect.text != "target" else [p[0] for p in raw_pairs]
        else:
            # Open Joint: Zero access to gold pairs or target aspects during inference
            target_aspects = None

        # Phase 1: TA -> Immutable H_A
        cached_t0 = sample.get("text_initial_cached")
        self.h_a, u_ta, l_ta = self.text_reasoner.generate_anchor(
            text=text,
            target_aspects=target_aspects,
            cached_t0=cached_t0
        )
        self._record_usage(u_ta, l_ta, "text")
        self.h_b = copy.deepcopy(self.h_a)

        # Build aspect states and episode state
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

        # Match anchor sentiment for target aspect
        anchor_sent = "NEU"
        for a in self.h_a.get("aspects", []):
            if a.get("text", "").strip().lower() == self.target_aspect.text.strip().lower():
                anchor_sent = a.get("sentiment", "NEU")
                break
        if anchor_sent == "NEU" and self.aspect_states:
            # Fallback to first aspect state
            first_st = list(self.aspect_states.values())[0]
            anchor_sent = first_st.get("anchor_sentiment", "NEU")

        self.aspect_episode_state = AspectEpisodeState(
            sample_id=sample_id,
            target_aspect=self.target_aspect,
            anchor_sentiment=anchor_sent,
            baseline_sentiment=anchor_sent,
            current_candidate=None,
            accepted_evidence=[],
            pending_visual_gap=None,
            recommended_next_action=None,
            action_history=[],
            status="ACTIVE"
        )

        # Phase 2: IG -> V_0 (Control plane only)
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
        """Returns observable state S_t for Controller Policy (Gold sentiment strictly excluded!)."""
        valid_actions = self.budget_state.get_valid_actions()
        target_aid = list(self.aspect_states.keys())[0] if self.aspect_states else "a_01"
        return {
            "sample_id": self.public_context.get("sample_id", "unknown"),
            "text": self.public_context.get("text", ""),
            "image": self.public_context.get("image", ""),
            "target_aspect": self.target_aspect.model_dump() if self.target_aspect else {},
            "target_aspect_id": target_aid,
            "target_aspect_uid": self.target_aspect.aspect_uid if self.target_aspect else None,
            "h_a": self.h_a,
            "h_b": self.h_b,
            "v_0": self.v_0,
            "aspect_episode_state": self.aspect_episode_state.model_dump() if self.aspect_episode_state else {},
            "aspect_states": self.aspect_states,
            "policy_history": self.policy_history,
            "budget": self.budget_state.deep_remaining,
            "budget_state": self.budget_state.model_dump(),
            "action_mask": valid_actions,
            "pending_visual_gap": self.pending_visual_gap,
            "recommended_next_action": self.recommended_next_action,
            "done": self.done
        }

    def step(self, action_input: Union[Dict[str, Any], str]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Executes strictly ONE macro step and transitions to S_{t+1}.
        Invariant: Zero auto-cascading! If CT escalates or CV needs more evidence,
        returns S_{t+1} with pending_visual_gap for the policy to route next.
        """
        if self.done:
            return self._get_state(), 0.0, True, {"msg": "Already done"}

        # Parse action input
        if isinstance(action_input, str):
            action = action_input.upper().strip()
            action_dict: Dict[str, Any] = {"action": action}
        elif isinstance(action_input, dict):
            action_dict = action_input
            action = str(action_dict.get("action", ControllerAction.FINALIZE.value)).upper().strip()
        else:
            action_dict = {"action": ControllerAction.FINALIZE.value}
            action = ControllerAction.FINALIZE.value

        if self.initial_c_d is None:
            self.initial_c_d = action_dict

        # Action masking check
        valid_actions = self.budget_state.get_valid_actions()
        if action not in valid_actions:
            action = ControllerAction.FINALIZE.value

        sample_id = self.public_context.get("sample_id", "unknown")
        text = self.public_context.get("text", "")
        rel_img = self.public_context.get("image", "")
        full_img = os.path.join(self.image_base_dir, rel_img) if self.image_base_dir else rel_img

        # Resolve target aspect text and id
        target_aid = action_dict.get("target_aspect_id")
        if not target_aid:
            target_aid = list(self.aspect_states.keys())[0] if self.aspect_states else "a_01"

        target_asp_text = self.target_aspect.text if self.target_aspect else ""
        target_span = self.target_aspect.span if self.target_aspect else [0, 0]
        target_uid = self.target_aspect.aspect_uid if self.target_aspect else f"{sample_id}_{target_asp_text}_0_0"

        # Pre-sentiment
        pre_sentiment = self.aspect_episode_state.baseline_sentiment if self.aspect_episode_state else "NEU"

        # ====================================================================
        # Action: FINALIZE
        # ====================================================================
        if action == ControllerAction.FINALIZE.value or self.budget_state.deep_remaining <= 0:
            self.done = True
            reward = self._calculate_final_reward()
            return self._get_state(), reward, True, {"action": "FINALIZE"}

        step_reward = 0.0

        # ====================================================================
        # Action: TEXT_RETHINK (Strictly 1 macro turn, NO auto-cascading)
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

            # Candidate sentiment
            cand_sent = pre_sentiment
            for a in h_rethink.get("aspects", []):
                if a.get("text", "").strip().lower() == target_asp_text.strip().lower() or a.get("aspect_id") == target_aid:
                    cand_sent = a.get("sentiment", pre_sentiment)
                    break

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
                self.h_b = copy.deepcopy(h_rethink)
                post_sentiment = cand_sent
                baseline_mutated = True
                if self.aspect_episode_state:
                    self.aspect_episode_state.baseline_sentiment = post_sentiment
                    self.aspect_episode_state.status = "VERIFIED"
                    self.aspect_episode_state.action_history.append("TEXT_RETHINK:ACCEPT")
                if target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["baseline_sentiment"] = post_sentiment
                    self.aspect_states[target_aid]["status"] = "VERIFIED"
                    self.aspect_states[target_aid]["action_history"].append("TEXT_RETHINK:ACCEPT")
            else:
                if self.aspect_episode_state:
                    self.aspect_episode_state.action_history.append("TEXT_RETHINK:REVERT")
                if target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["action_history"].append("TEXT_RETHINK:REVERT")

            # Invariant: If CT escalates to vision, record gap on state without auto-cascading!
            if ct_decision == "ESCALATE_TO_VISION":
                v_gap = c_t.get("visual_gap") or f"Check physical visual facts for {target_asp_text}"
                self.pending_visual_gap = v_gap
                self.recommended_next_action = ControllerAction.VISION_PROBE.value
                if self.aspect_episode_state:
                    self.aspect_episode_state.pending_visual_gap = v_gap
                    self.aspect_episode_state.recommended_next_action = ControllerAction.VISION_PROBE.value

            # Log transition record
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
                candidate_sentiment=cand_sent,
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

            step_reward += self._step_reward(pre_sentiment, post_sentiment, action_type="TEXT", decision=ct_decision)

        # ====================================================================
        # Action: VISION_PROBE (Strictly 1 probe turn, NO auto-looping)
        # ====================================================================
        elif action == ControllerAction.VISION_PROBE.value:
            self.budget_state.consume_vision()
            self.step_count += 1
            probe_step = self.step_count
            step_ref = f"vision_probe_{probe_step}"

            # Determine inquiry question
            question = (
                self.pending_visual_gap
                or action_dict.get("question_for_vision")
                or action_dict.get("question")
            )
            if not question:
                question = self.meta_controller.generate_visual_question(
                    target_aspect_text=target_asp_text,
                    v_0=self.v_0,
                    query_type=action_dict.get("query_type")
                )

            # Consume pending gap
            self.pending_visual_gap = None
            self.recommended_next_action = None
            if self.aspect_episode_state:
                self.aspect_episode_state.pending_visual_gap = None
                self.aspect_episode_state.recommended_next_action = None

            # 1. IP: Targeted Visual Sensor Probe
            raw_ev, u_ip, l_ip = self.vision_sensor.probe_deep(
                image_path=full_img,
                question=question
            )
            self._record_usage(u_ip, l_ip, "vision")

            # 2. CE: Strictly Fail-Closed Evidence Firewall
            filtered_ev, u_ce, l_ce = self.meta_controller.filter_evidence_firewall(
                question=question,
                raw_evidence=raw_ev,
                target_aspect_id=target_aid,
                target_aspect_text=target_asp_text
            )
            self._record_usage(u_ce, l_ce, "controller")
            filtered_ev["probe_id"] = step_ref
            filtered_ev["question"] = question

            # 3. CA: Compile into EvidenceBundle with provenance
            past_probes = [b for b in self.evidence_bundles if b.get("target_aspect_id") == target_aid]
            probe_history_raw = [p for b in past_probes for p in b.get("probes", [])] + [filtered_ev]
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

            # Candidate sentiment
            cand_sent = pre_sentiment
            for a in h_fusion.get("aspects", []):
                if a.get("text", "").strip().lower() == target_asp_text.strip().lower() or a.get("aspect_id") == target_aid:
                    cand_sent = a.get("sentiment", pre_sentiment)
                    break

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

            post_sentiment = pre_sentiment
            baseline_mutated = False

            if cv_dec == "ACCEPT_REVISION":
                self.h_b = copy.deepcopy(h_fusion)
                post_sentiment = cand_sent
                baseline_mutated = True
                if self.aspect_episode_state:
                    self.aspect_episode_state.baseline_sentiment = post_sentiment
                    self.aspect_episode_state.accepted_evidence.extend(
                        [f["content"] for f in evidence_bundle.accepted_facts]
                    )
                    self.aspect_episode_state.status = "VERIFIED"
                    self.aspect_episode_state.action_history.append("VISION_PROBE:ACCEPT")
                if target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["baseline_sentiment"] = post_sentiment
                    self.aspect_states[target_aid]["accepted_evidence"].extend(
                        [f["content"] for f in evidence_bundle.accepted_facts]
                    )
                    self.aspect_states[target_aid]["status"] = "VERIFIED"
                    self.aspect_states[target_aid]["action_history"].append("VISION_PROBE:ACCEPT")
            else:
                if self.aspect_episode_state:
                    self.aspect_episode_state.action_history.append("VISION_PROBE:REVERT")
                if target_aid in self.aspect_states:
                    self.aspect_states[target_aid]["action_history"].append("VISION_PROBE:REVERT")

            # Invariant: If CV requests more evidence, record gap on state without auto-looping!
            if cv_dec in ["NEED_MORE_EVIDENCE", "QUERY_AGAIN"] and self.budget_state.vision_remaining > 0:
                next_q = c_v.get("next_question") or f"Check additional visual details for {target_asp_text}"
                self.pending_visual_gap = next_q
                self.recommended_next_action = ControllerAction.VISION_PROBE.value
                if self.aspect_episode_state:
                    self.aspect_episode_state.pending_visual_gap = next_q
                    self.aspect_episode_state.recommended_next_action = ControllerAction.VISION_PROBE.value

            # Log transition record
            self.transitions.append(TransitionRecord(
                sample_id=sample_id,
                aspect_id=target_aid,
                aspect_text=target_asp_text,
                aspect_span=target_span,
                aspect_uid=target_uid,
                step=probe_step,
                pre_sentiment=pre_sentiment,
                action="VISION_PROBE",
                critique_or_question=question,
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
                "question": question,
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

            step_reward += self._step_reward(pre_sentiment, post_sentiment, action_type="VISION", decision=cv_dec)

        if self.budget_state.deep_remaining <= 0:
            self.done = True

        info = {
            "step": self.step_count,
            "action": action,
            "target_aspect": target_asp_text,
            "final_pairs": self.h_b.get("pairs", [])
        }
        return self._get_state(), step_reward, self.done, info

    def _step_reward(
        self,
        pre: str,
        post: str,
        action_type: str = "TEXT",
        decision: str = "ACCEPT"
    ) -> float:
        """Asymmetric aspect-level step reward.
        R_t = alpha * I(W -> C) - beta * I(C -> W) - lambda_T * I(TEXT) - lambda_V * I(VISION) - lambda_U * I(unsupported)
        where beta (1.5) > alpha (1.0), heavily penalizing harmful modifications.
        """
        reward = 0.0
        lambda_t = 0.05
        lambda_v = 0.10
        lambda_u = 0.10

        if action_type == "TEXT":
            reward -= lambda_t
        elif action_type == "VISION":
            reward -= lambda_v

        if decision in ["REVERT_TEXT_BASELINE", "REVERT_ANCHOR", "REJECT_REVISION"]:
            reward -= lambda_u

        if not self._gold_sentiment:
            return round(reward, 4)

        gold = self._gold_sentiment
        alpha = 1.0
        beta = 1.5

        if pre != gold and post == gold:
            reward += alpha  # Rescued
        elif pre == gold and post != gold:
            reward -= beta   # Harmed

        return round(reward, 4)

    def _calculate_final_reward(self) -> float:
        """Calculates Anchor-relative asymmetric final episode reward.
        +1.0 on rescue from wrong anchor hypothesis, -1.5 on harm away from correct anchor.
        """
        gold = self._gold_sentiment
        s_anchor = self.aspect_episode_state.anchor_sentiment if self.aspect_episode_state else "NEU"
        s_final = self.aspect_episode_state.baseline_sentiment if self.aspect_episode_state else "NEU"

        if not gold:
            gold_set = {tuple(p) for p in self._gold_pairs_map.items()}
            pred_set = {tuple(p) for p in self.h_b.get("pairs", [])}
            return 1.0 if (gold_set and gold_set == pred_set) else 0.0

        c_a = 1 if s_anchor == gold else 0
        c_f = 1 if s_final == gold else 0

        # Asymmetric Anchor-relative task reward
        if c_a == 0 and c_f == 1:
            task_reward = 1.0   # Rescued!
        elif c_a == 1 and c_f == 0:
            task_reward = -1.5  # Harmed!
        elif c_a == 1 and c_f == 1:
            task_reward = 1.0   # Maintained correct
        else:
            task_reward = -0.5  # Remained incorrect

        return round(task_reward, 4)

    def export_trajectory(self) -> Dict[str, Any]:
        """Exports the full trajectory conforming to the standardized evaluation & SFT format."""
        sample_id = self.public_context.get("sample_id", "unknown")
        text = self.public_context.get("text", "")
        rel_img = self.public_context.get("image", "")

        y_a = self.h_a.get("pairs", [])
        y_final = self.h_b.get("pairs", [])

        num_v_probes = len([t for t in self.transitions if t["action"] == "VISION_PROBE"])
        num_t_rethinks = len([t for t in self.transitions if t["action"] == "TEXT_RETHINK"])

        risk_type = (self.initial_c_d or {}).get("risk_diagnosis", {}).get("risk_type", "NO_RISK")
        main_action = (self.initial_c_d or {}).get("action", "FINALIZE")

        self.compute["latency_ms"] = round(self.compute["latency_ms"], 1)

        # Reconstruct gold pairs for evaluator compatibility
        gold_pairs = [[k, v] for k, v in self._gold_pairs_map.items()]

        return {
            "sample_id": sample_id,
            "text": text,
            "image": rel_img,
            "gold_pairs": gold_pairs,
            "target_aspect": self.target_aspect.model_dump() if self.target_aspect else {},
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
            "aspect_episode_state": self.aspect_episode_state.model_dump() if self.aspect_episode_state else {},
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
