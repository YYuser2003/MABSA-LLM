"""BACR-v3 Pipeline Orchestrator (Thin Runner over BACREnv + ControllerPolicy).

Architecture:
    Pipeline = BACREnv + ControllerPolicy + Logger

Guarantees complete isomorphism:
    - Teacher Inference: BACREnv + GeminiTeacherPolicy
    - Student Inference: BACREnv + QwenPolicy
    - RL Training:       BACREnv + TrainableControllerPolicy
    - Heuristic Baseline: BACREnv + RulePolicy

All state transitions, fail-closed firewalls, aspect isolation, and verifier reversions
are centralized within BACREnv.
"""

import os
import time
import copy
from typing import Dict, Any, Optional, Union, List

from bacr.client import BaseClient
from bacr.env import BACREnv
from bacr.policies.base import BaseControllerPolicy, GeminiTeacherPolicy
from bacr.schemas_v3 import TaskMode, TargetAspect


class BACRPipelineV3:
    """Production-grade thin pipeline runner over BACREnv and ControllerPolicy."""

    def __init__(
        self,
        client: Optional[BaseClient] = None,
        run_id: Optional[str] = None,
        max_visual_probes: int = 2,
        policy: Optional[BaseControllerPolicy] = None,
        budget_config: Optional[Dict[str, int]] = None,
        task_mode: Union[TaskMode, str] = TaskMode.TARGET_GUIDED,
        text_client: Optional[BaseClient] = None,
        controller_client: Optional[BaseClient] = None,
        vision_client: Optional[BaseClient] = None
    ):
        self.client = client
        self.run_id = run_id or f"bacr_v3_{int(time.time())}"

        b_cfg = budget_config or {
            "max_deep_actions": max_visual_probes,
            "max_text_rethinks": 1,
            "max_visual_probes": max_visual_probes
        }

        # Single source of truth environment
        self.env = BACREnv(
            client=client,
            budget_config=b_cfg,
            task_mode=task_mode,
            text_client=text_client,
            controller_client=controller_client,
            vision_client=vision_client
        )

        # Controller policy
        self.policy = policy or GeminiTeacherPolicy(client=client or controller_client)

        # Expose sub-components for compatibility
        self.text_reasoner = self.env.text_reasoner
        self.vision_sensor = self.env.vision_sensor
        self.meta_controller = self.env.meta_controller
        self.max_deep_actions = self.env.max_deep_actions

    def run_sample(
        self,
        sample: Dict[str, Any],
        image_base_dir: str = "",
        max_visual_probes: Optional[int] = None,
        task_mode: Optional[Union[TaskMode, str]] = None
    ) -> Dict[str, Any]:
        """Executes full BACR-v3 inference loop by stepping through BACREnv.
        In target_guided mode with multiple aspects, factors into independent
        single-aspect episodes and aggregates results.
        """
        if max_visual_probes is not None:
            self.env.max_deep_actions = max_visual_probes
            self.env.max_visual_probes = max_visual_probes

        resolved_mode = task_mode or self.env.task_mode
        if isinstance(resolved_mode, str):
            resolved_mode = TaskMode(resolved_mode)

        gold_pairs = sample.get("pairs", [])
        # If target_guided and multiple aspects exist, factor into independent episodes
        if resolved_mode == TaskMode.TARGET_GUIDED and len(gold_pairs) > 1:
            aspect_trajs: List[Dict[str, Any]] = []
            aggregated_pairs: List[List[str]] = []
            combined_transitions: List[Dict[str, Any]] = []
            combined_rounds: List[Dict[str, Any]] = []
            combined_policy_history: List[Dict[str, Any]] = []
            combined_evidence_bundles: List[Dict[str, Any]] = []
            combined_aspect_states: Dict[str, Any] = {}
            total_compute = {
                "api_calls_text": 0,
                "api_calls_vision": 0,
                "api_calls_controller": 0,
                "api_calls_total": 0,
                "total_tokens": 0,
                "image_invocations": 0,
                "latency_ms": 0.0
            }

            for idx, p in enumerate(gold_pairs):
                asp_text = p[0]
                spans = sample.get("spans", [])
                span = spans[idx] if idx < len(spans) else [0, len(asp_text)]
                target_asp = TargetAspect(
                    aspect_uid=f"{sample.get('sample_id', 's')}_{asp_text}_{span[0]}_{span[1]}",
                    text=asp_text,
                    span=span
                )
                state = self.env.reset(
                    sample=sample,
                    target_aspect=target_asp,
                    image_base_dir=image_base_dir,
                    task_mode=resolved_mode
                )
                while not self.env.done:
                    action = self.policy.act(state)
                    state, reward, done, info = self.env.step(action)

                asp_traj = self.env.export_trajectory()
                aspect_trajs.append(asp_traj)

                # Collect predicted sentiment for this aspect
                asp_sent = "NEU"
                for pair in asp_traj.get("final_pairs", []):
                    if pair[0].strip().lower() == asp_text.strip().lower():
                        asp_sent = pair[1]
                        break
                aggregated_pairs.append([asp_text, asp_sent])

                combined_transitions.extend(asp_traj.get("transitions", []))
                combined_rounds.extend(asp_traj.get("rounds", []))
                combined_policy_history.extend(asp_traj.get("policy_history", []))
                combined_evidence_bundles.extend(asp_traj.get("evidence_bundles", []))
                combined_aspect_states.update(asp_traj.get("aspect_states", {}))

                # Accumulate compute
                comp = asp_traj.get("compute", {})
                for k, v in comp.items():
                    if k in total_compute:
                        total_compute[k] += v

            num_v = len([t for t in combined_transitions if t.get("action") == "VISION_PROBE"])
            num_t = len([t for t in combined_transitions if t.get("action") == "TEXT_RETHINK"])

            base_traj = copy.deepcopy(aspect_trajs[0])
            base_traj["final_pairs"] = aggregated_pairs
            base_traj["gold_pairs"] = gold_pairs
            base_traj["aspect_states"] = combined_aspect_states
            base_traj["transitions"] = combined_transitions
            base_traj["rounds"] = combined_rounds
            base_traj["policy_history"] = combined_policy_history
            base_traj["evidence_bundles"] = combined_evidence_bundles
            base_traj["num_visual_probes"] = num_v
            base_traj["num_text_queries"] = num_t
            base_traj["num_queries_total"] = num_v + num_t
            base_traj["compute"] = total_compute
            base_traj["aspect_trajectories"] = aspect_trajs
            return base_traj

        else:
            # Single aspect or Open Joint mode
            state = self.env.reset(
                sample=sample,
                image_base_dir=image_base_dir,
                task_mode=resolved_mode
            )
            while not self.env.done:
                action = self.policy.act(state)
                state, reward, done, info = self.env.step(action)
            return self.env.export_trajectory()
