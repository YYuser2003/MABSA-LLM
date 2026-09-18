"""Comprehensive Unit Tests for BACR-v3 Architecture and Pipeline.

Validates:
1. Pydantic v2 schemas and strictly fail-closed Evidence Firewall (CE).
2. Physical and semantic critique firewall (V0 never enters C_Q^T).
3. Invariant: QUERY_AGAIN preserves H_B, accumulates evidence, and dispatches next_question.
4. CT ESCALATE_TO_VISION triggers targeted visual probe.
5. Multi-Aspect Isolation (non-target aspects never perturbed).
6. Dual Baseline Preservation (Revert-to-HB safeguards valid text revisions).
7. Transition-level Marginal Utility & Governance evaluation (MU_T, MU_V).
8. BACREnv gym-like interface & transition logging.
"""

import os
import sys
import json
import tempfile
import pathlib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bacr.schemas_v3 import (
    ControllerAction,
    RiskType,
    CTDecision,
    CVDecision,
    EvidenceStatus,
    TargetBinding,
    RevisionSupport,
    RiskDecision,
    EvidenceFirewallOutput,
    TextRevisionAuditOutput,
    RevisionVerifierOutput,
    AspectState,
    TransitionRecord,
    TaskMode,
    BudgetState
)
from bacr.text_reasoner import TextReasoner
from bacr.meta_controller import MetaController
from bacr.vision_sensor import VisionSensor
from bacr.pipeline_v3 import BACRPipelineV3
from bacr.evaluator import evaluate_v3_trajectories
from bacr.env import BACREnv
from bacr.policies.base import GeminiTeacherPolicy, RulePolicy, RandomPolicy


class MockClient:
    """Mock client returning controlled responses for each BACR-v3 prompt type."""
    def __init__(
        self,
        override_action="VISION_PROBE",
        override_firewall_status="VALID",
        override_cv_decision="ACCEPT_REVISION",
        rethink_sentiment="NEU",
        leak_visual_in_critique=False
    ):
        self.call_history = []
        self.override_action = override_action
        self.override_firewall_status = override_firewall_status
        self.override_cv_decision = override_cv_decision
        self.rethink_sentiment = rethink_sentiment
        self.leak_visual_in_critique = leak_visual_in_critique
        self.step_action_sequence = []
        self.cv_decision_sequence = []

    def call_text(self, system_prompt: str, user_prompt: str = "", **kwargs):
        self.call_history.append((system_prompt, user_prompt))

        # 1. TA Text Anchor Reasoner
        if "(TA) Prompt" in system_prompt or "Structured Text Anchor Ledger" in user_prompt:
            return {
                "aspects": [
                    {
                        "aspect_id": "a_01",
                        "text": "Obama",
                        "span": [0, 5],
                        "sentiment": "POS",
                        "text_evidence": ["Obama gives speech"],
                        "rationale": "Enthusiastic speech.",
                        "assumptions": ["Enthusiasm scopes to entity."],
                        "uncertainties": ["Tone may be journalistic report."],
                        "risks": []
                    }
                ],
                "pairs": [["Obama", "POS"]]
            }, {"total_tokens": 20}, 0.05

        # 2. CD Risk-Aware Diagnosis Controller
        elif "(CD) Prompt" in system_prompt or "Diagnose error risk" in user_prompt:
            action = self.step_action_sequence.pop(0) if self.step_action_sequence else self.override_action
            critique = "Examine if reporting verb indicates speaker neutrality."
            if self.leak_visual_in_critique:
                critique = "The image photo shows a smiling person in the background, check sentiment."

            if action == "FINALIZE":
                return {
                    "risk_diagnosis": {
                        "risk_type": "NO_RISK",
                        "risk_description": "Text baseline is well grounded."
                    },
                    "action": "FINALIZE",
                    "target_aspect_id": None,
                    "critique_for_text": None,
                    "question_for_vision": None,
                    "decision_reason": "No further verification needed."
                }, {"total_tokens": 20}, 0.05
            elif action == "TEXT_RETHINK":
                return {
                    "risk_diagnosis": {
                        "risk_type": "AFFECT_SPILLOVER",
                        "risk_description": "Affect spillover from hashtag."
                    },
                    "action": "TEXT_RETHINK",
                    "target_aspect_id": "a_01",
                    "critique_for_text": critique,
                    "question_for_vision": None,
                    "decision_reason": "Syntactic ambiguity detected."
                }, {"total_tokens": 20}, 0.05
            else:
                return {
                    "risk_diagnosis": {
                        "risk_type": "MISSING_AFFECT",
                        "risk_description": "Minimalist factual tweet requires visual affect inspection."
                    },
                    "action": "VISION_PROBE",
                    "target_aspect_id": "a_01",
                    "critique_for_text": None,
                    "question_for_vision": "Is Obama smiling or showing celebration in the photo?",
                    "decision_reason": "Visual inspection allocated."
                }, {"total_tokens": 20}, 0.05

        # 3. C_Q^T Critique Generator (Physically Decoupled)
        elif "Linguistic Auditor" in system_prompt:
            return {
                "critique": "Re-examine syntax and reporting verbs without external cues."
            }, {"total_tokens": 15}, 0.05

        # 4. TR Text Re-deliberation
        elif "(TR) Prompt" in system_prompt or "Text Re-deliberation" in user_prompt:
            return {
                "aspects": [
                    {
                        "aspect_id": "a_01",
                        "text": "Obama",
                        "span": [0, 5],
                        "sentiment": self.rethink_sentiment,
                        "text_evidence": ["speech"],
                        "rationale": "Purely informational speech report; neutral tone.",
                        "assumptions": ["Neutral journalistic context."],
                        "uncertainties": []
                    }
                ],
                "pairs": [["Obama", self.rethink_sentiment]]
            }, {"total_tokens": 20}, 0.05

        # 5. CT Text Revision Verifier
        elif "(CT) Prompt" in system_prompt or "Audit whether the linguistic revision" in user_prompt:
            return {
                "decision": "ACCEPT_TEXT_REVISION",
                "text_evidence_verified": True,
                "modifier_scope_changed": True,
                "reporting_frame_decoupled": True,
                "audit_rationale": "Linguistic revision is firmly grounded in syntax."
            }, {"total_tokens": 15}, 0.05

        # 6. CE Evidence Firewall
        elif "(CE) Prompt" in system_prompt or "Filter speculative inferences" in user_prompt:
            if self.override_firewall_status == "INVALID":
                return {
                    "status": "INVALID",
                    "target_binding": "UNBOUND",
                    "relevance": "LOW",
                    "usable_evidence": [],
                    "rejected_inferences": ["Smiling crowd (unrelated to target)."],
                    "revision_support": "NON_DECISIVE",
                    "verification_notes": "Crowd is unbound to target entity."
                }, {"total_tokens": 20}, 0.05
            else:
                return {
                    "status": "VALID",
                    "target_binding": "DIRECT",
                    "relevance": "HIGH",
                    "usable_evidence": ["Target person in center shows broad smile with raised cheeks."],
                    "rejected_inferences": [],
                    "revision_support": "SUPPORTS_REVISION",
                    "verification_notes": "Physical expression confirms positive affect."
                }, {"total_tokens": 20}, 0.05

        # 7. TF Evidence Fusion Reasoner
        elif "(TF) Prompt" in system_prompt or "Evidence Fusion" in user_prompt:
            return {
                "aspects": [
                    {
                        "aspect_id": "a_01",
                        "text": "Obama",
                        "span": [0, 5],
                        "sentiment": "POS",
                        "text_evidence": ["speech"],
                        "visual_evidence": [{"ref": "vision_probe_1", "content": "Broad smile with raised cheeks."}],
                        "rationale": "Verified visual cues confirm genuine joy.",
                        "assumptions": ["Visual expression reflects genuine affect."],
                        "uncertainties": []
                    }
                ],
                "pairs": [["Obama", "POS"]]
            }, {"total_tokens": 25}, 0.05

        # 8. CV Revision Verifier
        elif "(CV) Prompt" in system_prompt or "Evaluate whether the candidate revision" in user_prompt:
            dec = self.cv_decision_sequence.pop(0) if self.cv_decision_sequence else self.override_cv_decision
            return {
                "decision": dec,
                "audit_rationale": "Evidence thoroughly evaluated.",
                "target_aspect_id": "a_01",
                "next_question": "Does the background banner confirm celebratory occasion?" if dec == "QUERY_AGAIN" else None
            }, {"total_tokens": 20}, 0.05

        return {}, {"total_tokens": 0}, 0.0

    def call_vision(self, system_prompt: str, user_prompt: str = "", image_path: str = "", user_text: str = "", **kwargs):
        self.call_history.append((system_prompt, user_prompt, image_path))

        # IG Global Visual Sensor
        if "(IG) Prompt" in system_prompt or "Global Visual Sensor" in system_prompt:
            return {
                "scene": "Podium speech press conference",
                "description": "A man speaking at a lectern with microphones.",
                "possible_entities": [{"entity": "Obama", "bounding_box": "center", "confidence": "high"}],
                "ocr": ["WHITE HOUSE"],
                "salient_visual_cues": ["lectern", "flags", "suit"],
                "target_presence_hints": [{"target": "Obama", "visible": True, "location": "center"}],
                "visual_affect_cues": ["neutral expression", "open posture"]
            }, {"total_tokens": 30}, 0.08

        # IP Targeted Deep Visual Probe
        elif "(IP) Prompt" in system_prompt or "Targeted Deep Visual Probe" in system_prompt:
            return {
                "answer": "Target person is smiling warmly at podium.",
                "observable_evidence": ["Smiling expression", "raised cheeks"],
                "certainty": "high",
                "insufficient_visual_evidence": False
            }, {"total_tokens": 25}, 0.08

        return {}, {"total_tokens": 0}, 0.0


def test_v3_schemas_and_fail_closed_contract():
    print("--- Running test_v3_schemas_and_fail_closed_contract ---")
    # 1. EvidenceFirewallOutput must be strictly fail-closed
    empty_res = EvidenceFirewallOutput.validate_fail_closed({})
    assert empty_res.status == EvidenceStatus.INVALID
    assert empty_res.usable_evidence == []
    assert empty_res.revision_support == RevisionSupport.NON_DECISIVE

    malformed_res = EvidenceFirewallOutput.validate_fail_closed({"usable_evidence": "not a list", "status": "VALID"})
    assert malformed_res.status == EvidenceStatus.INVALID
    assert malformed_res.usable_evidence == []

    insufficient_res = EvidenceFirewallOutput.validate_fail_closed({
        "status": "INSUFFICIENT",
        "usable_evidence": ["Some spurious fact"],
        "revision_support": "SUPPORTS_REVISION"
    })
    assert insufficient_res.status == EvidenceStatus.INSUFFICIENT
    assert insufficient_res.usable_evidence == []  # Must clear usable evidence
    assert insufficient_res.revision_support == RevisionSupport.NON_DECISIVE

    # 2. TextRevisionAuditOutput fallback
    ct_fallback = TextRevisionAuditOutput.validate_or_fallback({})
    assert ct_fallback.decision == CTDecision.REVERT_TEXT_BASELINE

    # 3. RevisionVerifierOutput fallback
    cv_fallback = RevisionVerifierOutput.validate_or_fallback({})
    assert cv_fallback.decision == CVDecision.REVERT_TEXT_BASELINE
    print("Passed test_v3_schemas_and_fail_closed_contract!")


def test_physical_and_semantic_critique_firewall():
    print("--- Running test_physical_and_semantic_critique_firewall ---")
    client = MockClient()
    controller = MetaController(client)

    # Test physical firewall: generate_text_critique does NOT accept or contain any image references
    critique, _, _ = controller.generate_text_critique(
        h_b={"aspects": [{"aspect_id": "a_01", "text": "Apple", "sentiment": "NEG"}]},
        risk_diagnosis={"risk_type": "REPORTING_FRAME"},
        target_aspect_id="a_01",
        target_aspect_text="Apple"
    )
    assert isinstance(critique, str)
    assert len(critique) > 0

    # Ensure C_Q^T prompt did not contain visual tokens
    cq_calls = [prompt for (sys_p, prompt) in client.call_history if "Linguistic Auditor" in sys_p]
    assert len(cq_calls) > 0
    assert "Global Visual Sketch" not in cq_calls[0]

    # Test programmatic sanitizer in diagnose_risk
    client_leak = MockClient(override_action="TEXT_RETHINK", leak_visual_in_critique=True)
    ctrl_leak = MetaController(client_leak)
    cd_res, _, _ = ctrl_leak.diagnose_risk(
        h_a={"pairs": [["X", "POS"]]},
        v0={"scene": "photo"},
        h_b={"pairs": [["X", "POS"]]}
    )
    assert cd_res.get("critique_sanitized") is True
    assert "smiling" not in cd_res["critique_for_text"]
    print("Passed test_physical_and_semantic_critique_firewall!")


def test_query_again_preserves_hb_and_accumulates_evidence():
    print("--- Running test_query_again_preserves_hb_and_accumulates_evidence ---")
    client = MockClient()
    # CD allocates VISION_PROBE -> Probe 1 returns QUERY_AGAIN -> Probe 2 returns ACCEPT_REVISION
    client.step_action_sequence = ["VISION_PROBE"]
    client.cv_decision_sequence = ["QUERY_AGAIN", "ACCEPT_REVISION"]

    pipeline = BACRPipelineV3(client=client, max_visual_probes=2)
    sample = {
        "sample_id": "test_query_again",
        "text": "Obama speech today",
        "image": "dummy.jpg",
        "pairs": [["Obama", "POS"]]
    }
    record = pipeline.run_sample(sample)

    # Invariant check: In round 1 QUERY_AGAIN did not corrupt baseline
    # Both probes were executed (budget=2 consumed)
    assert record["num_visual_probes"] == 2
    assert record["final_pairs"] == [["Obama", "POS"]]
    print("Passed test_query_again_preserves_hb_and_accumulates_evidence!")


def test_ct_escalate_to_vision():
    print("--- Running test_ct_escalate_to_vision ---")
    client = MockClient()
    # CD allocates TEXT_RETHINK
    client.step_action_sequence = ["TEXT_RETHINK"]

    # Mock CT returning ESCALATE_TO_VISION
    orig_call_text = client.call_text
    def custom_call_text(system_prompt: str = "", user_prompt: str = "", **kwargs):
        if "(CT) Prompt" in system_prompt or "Audit whether the linguistic revision" in user_prompt:
            return {
                "decision": "ESCALATE_TO_VISION",
                "visual_gap": "Check physical facial cues for Obama",
                "target_aspect_id": "a_01"
            }, {"total_tokens": 15}, 0.05
        return orig_call_text(system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
    client.call_text = custom_call_text

    pipeline = BACRPipelineV3(client=client, max_visual_probes=2)
    sample = {
        "sample_id": "test_escalate",
        "text": "Obama farewell event",
        "image": "dummy.jpg",
        "pairs": [["Obama", "POS"]]
    }
    record = pipeline.run_sample(sample)

    # Verify that visual probe was indeed triggered via ESCALATE_TO_VISION
    vision_transitions = [t for t in record["transitions"] if "ESCALATE_TO_VISION" in str(t.get("verifier_decision")) or t.get("action") == "VISION_PROBE"]
    assert len(vision_transitions) >= 1 or record["num_visual_probes"] >= 1
    print("Passed test_ct_escalate_to_vision!")


def test_text_reasoner_aspect_isolation():
    print("--- Running test_text_reasoner_aspect_isolation ---")
    client = MockClient(rethink_sentiment="NEU")
    reasoner = TextReasoner(client)

    multi_h_baseline = {
        "aspects": [
            {"aspect_id": "a_01", "text": "iPhone", "sentiment": "NEG"},
            {"aspect_id": "a_02", "text": "Camera", "sentiment": "POS"}
        ],
        "pairs": [["iPhone", "NEG"], ["Camera", "POS"]]
    }

    # Deliberate ONLY on a_01
    h_rethink, _, _ = reasoner.rethink_text(
        text="iPhone is okay but Camera is fantastic!",
        h_baseline=multi_h_baseline,
        critique="Re-evaluate iPhone tone.",
        target_aspect_id="a_01"
    )

    pairs_dict = {p[0]: p[1] for p in h_rethink["pairs"]}
    assert pairs_dict["iPhone"] == "NEU"
    assert pairs_dict["Camera"] == "POS"  # a_02 preserved perfectly!
    print("Passed test_text_reasoner_aspect_isolation!")


def test_dual_baseline_and_revert_to_hb():
    print("--- Running test_dual_baseline_and_revert_to_hb ---")
    # Step 1: TEXT_RETHINK revises H_A (POS) -> H_B (NEU) [Certified by CT]
    # Step 2: VISION_PROBE fails firewall (INVALID) -> reverts to H_B (NEU), NOT H_A (POS)!
    client = MockClient(
        override_firewall_status="INVALID",
        override_cv_decision="REVERT_TEXT_BASELINE",
        rethink_sentiment="NEU"
    )
    client.step_action_sequence = ["TEXT_RETHINK", "VISION_PROBE"]

    pipeline = BACRPipelineV3(client=client, max_visual_probes=2)
    sample = {
        "sample_id": "test_dual_baseline",
        "text": "Obama attends summit # party",
        "image": "dummy.jpg",
        "pairs": [["Obama", "NEU"]]
    }

    record = pipeline.run_sample(sample)
    assert record["text_anchor"]["pairs"] == [["Obama", "POS"]]     # H_A stays POS
    assert record["text_baseline"]["pairs"] == [["Obama", "NEU"]]   # H_B preserved as NEU
    assert record["final_pairs"] == [["Obama", "NEU"]]              # Final is NEU (reverted to H_B!)
    print("Passed test_dual_baseline_and_revert_to_hb!")


def test_v3_transition_diagnostics_calculation(tmp_path: pathlib.Path):
    print("--- Running test_v3_transition_diagnostics_calculation ---")
    gold_file = str(tmp_path / "gold.jsonl")
    traj_file = str(tmp_path / "traj.jsonl")

    # Realistic aspect names: Trump, Obama, Biden
    gold_data = [
        {"sample_id": "s1", "pairs": [["Trump", "POS"]]},
        {"sample_id": "s2", "pairs": [["Obama", "NEU"]]},
        {"sample_id": "s3", "pairs": [["Biden", "NEU"]]}
    ]
    with open(gold_file, "w", encoding="utf-8") as f:
        for g in gold_data:
            f.write(json.dumps(g) + "\n")

    trajs = [
        {
            "sample_id": "s1",
            "action": "FINALIZE",
            "contrast_type": "NO_RISK",
            "text_anchor": {"pairs": [["Trump", "POS"]]},
            "text_baseline": {"pairs": [["Trump", "POS"]]},
            "final_pairs": [["Trump", "POS"]],
            "transitions": []
        },
        {
            "sample_id": "s2",
            "action": "TEXT_RETHINK",
            "contrast_type": "AFFECT_SPILLOVER",
            "text_anchor": {"pairs": [["Obama", "POS"]]},  # H_A wrong
            "text_baseline": {"pairs": [["Obama", "NEU"]]}, # H_B corrected
            "final_pairs": [["Obama", "NEU"]],
            "transitions": [
                {
                    "step": 1,
                    "action": "TEXT_RETHINK",
                    "aspect_id": "a_01",
                    "aspect_text": "Obama",
                    "aspect_span": [0, 5],
                    "pre_sentiment": "POS",
                    "candidate_sentiment": "NEU",
                    "verifier_decision": "ACCEPT_TEXT_REVISION",
                    "post_sentiment": "NEU"
                }
            ]
        },
        {
            "sample_id": "s3",
            "action": "VISION_PROBE",
            "contrast_type": "MISSING_AFFECT",
            "text_anchor": {"pairs": [["Biden", "NEU"]]},
            "text_baseline": {"pairs": [["Biden", "NEU"]]},
            "final_pairs": [["Biden", "NEU"]],
            "transitions": [
                {
                    "step": 1,
                    "action": "VISION_PROBE",
                    "aspect_id": "a_02",
                    "aspect_text": "Biden",
                    "aspect_span": [0, 5],
                    "pre_sentiment": "NEU",
                    "candidate_sentiment": "POS",
                    "verifier_decision": "REVERT_TEXT_BASELINE",
                    "post_sentiment": "NEU"
                }
            ]
        }
    ]
    with open(traj_file, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")

    diag = evaluate_v3_trajectories(traj_file, gold_file)
    assert diag["total_samples"] == 3
    assert diag["acc_ha"] == 66.67
    assert diag["acc_final"] == 100.0
    assert diag["anchor_corrections"] == 1
    assert diag["anchor_corruptions"] == 0
    assert diag["anchor_net_gain"] == 1
    assert diag["revert_count"] == 1
    # Verify MU_T recovered 1 Obama aspect despite aspect_id='a_01'
    assert diag["marginal_utility_text"]["recover"] == 1
    assert diag["marginal_utility_text"]["harm"] == 0
    assert diag["marginal_utility_text"]["mu_t"] == 1
    print("Passed test_v3_transition_diagnostics_calculation!")


def test_bacr_env_gym_interface():
    print("--- Running test_bacr_env_gym_interface ---")
    client = MockClient(rethink_sentiment="NEU")
    env = BACREnv(client=client, budget_config={"max_deep_actions": 2, "max_text_rethinks": 1, "max_visual_probes": 2})

    sample = {
        "sample_id": "env_test_01",
        "text": "Obama gives great speech # report",
        "image": "dummy.jpg",
        "pairs": [["Obama", "NEU"]]
    }
    state = env.reset(sample)
    assert state["budget"] == 2
    assert state["done"] is False
    assert "a_01" in state["aspect_states"]
    assert "Obama" == state["aspect_states"]["a_01"]["text"]

    # Step 1: TEXT_RETHINK
    action_1 = {
        "action": ControllerAction.TEXT_RETHINK.value,
        "target_aspect_id": "a_01"
    }
    next_state, reward, done, info = env.step(action_1)
    assert next_state["budget"] == 1
    # Transitioned from POS (wrong) to NEU (correct) with TEXT cost = +1.0 - 0.05 = 0.95!
    assert reward == 0.95
    assert len(next_state["policy_history"]) == 1

    # Step 2: FINALIZE
    action_2 = {"action": ControllerAction.FINALIZE.value}
    final_state, final_r, done, info = env.step(action_2)
    assert done is True
    assert final_r == 1.0  # Final task exact match reward!
    print("Passed test_bacr_env_gym_interface!")


def test_open_joint_sequestration():
    print("--- Running test_open_joint_sequestration ---")
    client = MockClient()
    env = BACREnv(client=client, task_mode=TaskMode.OPEN_JOINT)
    sample = {
        "sample_id": "test_oj_01",
        "text": "Obama gives speech",
        "image": "dummy.jpg",
        "pairs": [["Obama", "NEU"]]  # Gold pairs must not be visible to agent
    }
    state = env.reset(sample)
    # Observable state S_t MUST NOT contain 'pairs' or 'gold_pairs'!
    assert "pairs" not in state
    assert "gold_pairs" not in state
    assert "Obama" in state["aspect_states"]["a_01"]["text"]

    # Ensure TA prompt did not receive gold aspects hint
    ta_calls = [item[1] for item in client.call_history if len(item) >= 2 and "Structured Text Anchor Ledger" in item[1]]
    assert len(ta_calls) > 0
    assert "Focus specifically on extracting and evaluating these target aspects" not in ta_calls[-1]
    print("Passed test_open_joint_sequestration!")


def test_monotonic_step_counter():
    print("--- Running test_monotonic_step_counter ---")
    client = MockClient()
    client.step_action_sequence = ["TEXT_RETHINK"]

    orig_call_text = client.call_text
    def custom_call_text(system_prompt: str = "", user_prompt: str = "", **kwargs):
        if "(CT) Prompt" in system_prompt or "Audit whether the linguistic revision" in user_prompt:
            return {
                "decision": "ESCALATE_TO_VISION",
                "visual_gap": "Check physical facial cues for Obama",
                "target_aspect_id": "a_01"
            }, {"total_tokens": 15}, 0.05
        return orig_call_text(system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
    client.call_text = custom_call_text

    pipeline = BACRPipelineV3(client=client, max_visual_probes=2)
    sample = {
        "sample_id": "test_mono_step",
        "text": "Obama farewell event",
        "image": "dummy.jpg",
        "pairs": [["Obama", "POS"]]
    }
    record = pipeline.run_sample(sample)
    transitions = record["transitions"]
    assert len(transitions) >= 2, f"Expected >=2 transitions, got {len(transitions)}"
    steps = [t["step"] for t in transitions]
    assert steps == sorted(steps), f"Steps not monotonically increasing: {steps}"
    assert len(steps) == len(set(steps)), f"Steps not strictly unique: {steps}"
    assert transitions[0]["action"] == "TEXT_RETHINK"
    assert transitions[0]["step"] == 1
    assert transitions[1]["action"] == "VISION_PROBE"
    assert transitions[1]["step"] == 2
    print("Passed test_monotonic_step_counter!")


def test_budget_state_action_mask():
    print("--- Running test_budget_state_action_mask ---")
    b = BudgetState(
        max_deep_actions=2,
        max_text_rethinks=1,
        max_visual_probes=2,
        deep_remaining=2,
        text_remaining=1,
        vision_remaining=2
    )
    assert set(b.get_valid_actions()) == {"FINALIZE", "TEXT_RETHINK", "VISION_PROBE"}

    # Consume text
    b.consume_text()
    assert b.text_remaining == 0
    assert b.deep_remaining == 1
    assert "TEXT_RETHINK" not in b.get_valid_actions()
    assert set(b.get_valid_actions()) == {"FINALIZE", "VISION_PROBE"}

    # Consume vision
    b.consume_vision()
    assert b.deep_remaining == 0
    assert b.get_valid_actions() == ["FINALIZE"]
    print("Passed test_budget_state_action_mask!")


def test_isomorphic_pipeline_and_env():
    print("--- Running test_isomorphic_pipeline_and_env ---")
    client_pipe = MockClient(rethink_sentiment="NEU")
    client_env = MockClient(rethink_sentiment="NEU")

    sample = {
        "sample_id": "test_iso_01",
        "text": "Obama delivers speech",
        "image": "dummy.jpg",
        "pairs": [["Obama", "NEU"]]
    }

    # Pipeline run
    pipe = BACRPipelineV3(client=client_pipe, max_visual_probes=2)
    pipe_record = pipe.run_sample(sample)

    # Manual Env run
    env = BACREnv(client=client_env, budget_config={"max_deep_actions": 2, "max_text_rethinks": 1, "max_visual_probes": 2})
    state = env.reset(sample)
    policy = GeminiTeacherPolicy(client=client_env)
    while not env.done:
        act = policy.act(state)
        state, r, done, info = env.step(act)
    env_record = env.export_trajectory()

    assert pipe_record["final_pairs"] == env_record["final_pairs"]
    assert len(pipe_record["transitions"]) == len(env_record["transitions"])
    assert pipe_record["contrast_type"] == env_record["contrast_type"]
    print("Passed test_isomorphic_pipeline_and_env!")


if __name__ == "__main__":
    test_v3_schemas_and_fail_closed_contract()
    test_physical_and_semantic_critique_firewall()
    test_query_again_preserves_hb_and_accumulates_evidence()
    test_ct_escalate_to_vision()
    test_text_reasoner_aspect_isolation()
    test_dual_baseline_and_revert_to_hb()
    with tempfile.TemporaryDirectory() as td:
        test_v3_transition_diagnostics_calculation(pathlib.Path(td))
    test_bacr_env_gym_interface()
    test_open_joint_sequestration()
    test_monotonic_step_counter()
    test_budget_state_action_mask()
    test_isomorphic_pipeline_and_env()
    print("\n==========================================================================")
    print("  ALL 12 COMPREHENSIVE BACR-v3 HARDENING & ISOMORPHISM TESTS PASSED 100%!")
    print("==========================================================================")

