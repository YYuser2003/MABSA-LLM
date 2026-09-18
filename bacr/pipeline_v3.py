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
from typing import Dict, Any, Optional, Union

from bacr.client import BaseClient
from bacr.env import BACREnv
from bacr.policies.base import BaseControllerPolicy, GeminiTeacherPolicy
from bacr.schemas_v3 import TaskMode


class BACRPipelineV3:
    """Production-grade thin pipeline runner over BACREnv and ControllerPolicy."""

    def __init__(
        self,
        client: BaseClient,
        run_id: Optional[str] = None,
        max_visual_probes: int = 2,
        policy: Optional[BaseControllerPolicy] = None,
        budget_config: Optional[Dict[str, int]] = None,
        task_mode: Union[TaskMode, str] = TaskMode.TARGET_GUIDED
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
            task_mode=task_mode
        )
        
        # Controller policy
        self.policy = policy or GeminiTeacherPolicy(client=client)

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
        """Executes full BACR-v3 inference loop by stepping through BACREnv."""
        if max_visual_probes is not None:
            self.env.max_deep_actions = max_visual_probes
            self.env.max_visual_probes = max_visual_probes

        # 1. Reset environment to S_0
        state = self.env.reset(
            sample=sample,
            image_base_dir=image_base_dir,
            task_mode=task_mode
        )

        # 2. Step environment until termination
        while not self.env.done:
            action = self.policy.act(state)
            state, reward, done, info = self.env.step(action)

        # 3. Export standardized trajectory
        return self.env.export_trajectory()
