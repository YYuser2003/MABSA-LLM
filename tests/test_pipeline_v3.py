"""Comprehensive Unit Tests for BACR-v3 Architecture and Pipeline."""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bacr.schemas_v3 import (
    validate_structured_ledger,
    validate_risk_diagnosis,
    validate_evidence_firewall,
    validate_text_revision_audit,
    validate_revision_verifier
)
from bacr.text_reasoner import TextReasoner
from bacr.meta_controller import MetaController
from bacr.vision_sensor import VisionSensor
from bacr.pipeline_v3 import BACRPipelineV3
from bacr.evaluator import evaluate_v3_trajectories


class MockClient:
    """Mock client returning controlled responses for each BACR-v3 prompt type."""
    def __init__(
        self,
        override_action="VISION_PROBE",
        override_firewall_status="VALID",
        rethink_sentiment="NEU",
        leak_visual_in_critique=False
    ):
        self.call_history = []
        self.override_action = override_action
        self.override_firewall_status = override_firewall_status
        self.rethink_sentiment = rethink_sentiment
        self.leak_visual_in_critique = leak_visual_in_critique
        self.step_action_sequence = []

    def call_text(self, system_prompt: str, user_prompt: str):
        self.call_history.append((system_prompt, user_prompt))

        # 1. TA Text Anchor Reasoner
        if "(TA) Prompt" in system_prompt:
            return {
                "aspects": [
                    {
                        "aspect_id": "a_01",
                        "text": "Obama",
                        "span": [0, 5],
                        "sentiment": "POS",  # Start with POS (e.g. affect spillover)
                        "text_evidence": ["Obama gives speech"],
                        "rationale": "Speech with enthusiastic hashtag.",
                        "assumptions": ["Enthusiasm scopes to speaker."],
                        "uncertainties": ["Tone may be journalistic report."],
                        "risks": [
                            {"type": "AFFECT_SPILLOVER", "level": "HIGH", "basis": "Hashtag excitement may not attach to target entity."}
                        ],
                        "risk_profile": {
                            "affect_spillover": "high",
                            "missing_affect": "low",
                            "reporting_frame": "low",
                            "pragmatic_blindness": "low",
                            "irony_conflict": "low"
                        }
                    }
                ],
                "pairs": [["Obama", "POS"]]
            }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

        # 2. CD Risk-Aware Diagnosis Controller
        elif "(CD) Prompt" in system_prompt:
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
                }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1
            elif action == "TEXT_RETHINK":
                return {
                    "risk_diagnosis": {
                        "risk_type": "OVER_POLARIZATION",
                        "risk_description": "Affect spillover from hashtag."
                    },
                    "action": "TEXT_RETHINK",
                    "target_aspect_id": "a_01",
                    "critique_for_text": critique,
                    "question_for_vision": None,
                    "decision_reason": "Scope check required."
                }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1
            else:
                return {
                    "risk_diagnosis": {
                        "risk_type": "UNDER_POLARIZATION",
                        "risk_description": "Minimalist text needing physical inspection."
                    },
                    "action": "VISION_PROBE",
                    "target_aspect_id": "a_01",
                    "question_for_vision": "Is the person smiling at the podium Barack Obama?",
                    "decision_reason": "Visual inspection of affect needed."
                }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

        # 3. TR Text Re-deliberation
        elif "(TR) Prompt" in system_prompt:
            return {
                "aspects": [
                    {
                        "aspect_id": "a_01",
                        "text": "Obama",
                        "span": [0, 5],
                        "sentiment": self.rethink_sentiment,  # Corrected to NEU
                        "text_evidence": ["Obama gives speech"],
                        "rationale": "Confirmed reporting neutrality, hashtag detached from target.",
                        "risks": []
                    }
                ],
                "pairs": [["Obama", self.rethink_sentiment]]
            }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

        # 4. CT Text Revision Verifier
        elif "(CT) Prompt" in system_prompt:
            return {
                "decision": "ACCEPT_TEXT_REVISION",
                "text_evidence_verified": True,
                "modifier_scope_changed": True,
                "reporting_frame_decoupled": True,
                "audit_rationale": "Modifier scope confirmed detached from target aspect; revision accepted.",
                "target_aspect_id": "a_01"
            }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

        # 5. CE Evidence Firewall
        elif "(CE) Prompt" in system_prompt:
            if self.override_firewall_status == "INSUFFICIENT":
                return {
                    "status": "INSUFFICIENT",
                    "target_binding": "UNBOUND",
                    "relevance": "LOW",
                    "revision_support": "NON_DECISIVE",
                    "usable_evidence": [],
                    "rejected_inferences": ["Subject not clearly visible"],
                    "verification_notes": "Occlusion prevents conclusive observation."
                }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1
            else:
                return {
                    "status": "VALID",
                    "target_binding": "DIRECT",
                    "relevance": "HIGH",
                    "revision_support": "SUPPORTS_REVISION",
                    "usable_evidence": ["The smiling person at the podium has facial structure matching Barack Obama."],
                    "rejected_inferences": ["presidential aura"],
                    "verification_notes": "Identity confirmed directly."
                }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

        # 6. TF Evidence Fusion
        elif "(TF) Prompt" in system_prompt:
            return {
                "aspects": [
                    {
                        "aspect_id": "a_01",
                        "text": "Obama",
                        "span": [0, 5],
                        "sentiment": "POS",
                        "text_evidence": ["Obama gives speech"],
                        "visual_evidence": [{"ref": "vision_probe_1", "content": "Facial structure matches Obama"}],
                        "rationale": "Verified Obama is smiling warmly at podium."
                    }
                ],
                "pairs": [["Obama", "POS"]]
            }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

        # 7. CV Revision Verifier
        elif "(CV) Prompt" in system_prompt:
            if self.override_firewall_status == "INSUFFICIENT":
                return {
                    "decision": "REVERT_TEXT_BASELINE",
                    "audit_rationale": "Visual evidence insufficient; reverting to verified text baseline HB.",
                    "target_aspect_id": "a_01",
                    "next_question": None
                }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1
            else:
                return {
                    "decision": "ACCEPT_REVISION",
                    "audit_rationale": "Identity verified and attached directly to target; positive sentiment justified.",
                    "target_aspect_id": "a_01",
                    "next_question": None
                }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

        return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 0.0

    def call_vision(self, system_prompt: str, user_text: str, image_path: str):
        if "(IG / V0) Prompt" in system_prompt:
            return {
                "scene": "Podium speech",
                "description": "A man speaking at a podium.",
                "possible_entities": [{"identity": "Barack Obama", "status": "supported"}],
                "ocr": ["WHITE HOUSE"],
                "salient_visual_cues": ["microphone", "podium"],
                "visual_affect_cues": ["neutral expression"]
            }, {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40}, 0.2
        elif "(IP) Prompt" in system_prompt:
            return {
                "answer": "The person has facial features matching Barack Obama, displaying a clear smile.",
                "observable_evidence": ["facial bone structure", "parted lips smile"],
                "certainty": "high",
                "insufficient_visual_evidence": False
            }, {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40}, 0.2
        return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 0.0


def test_v3_schemas():
    # 1. Structured ledger validation
    valid_ledger = {
        "aspects": [
            {
                "aspect_id": "a_01",
                "text": "Tesla",
                "span": [0, 5],
                "sentiment": "POS",
                "assumptions": ["good quality"],
                "uncertainties": [],
                "risks": [{"type": "MISSING_AFFECT", "level": "LOW", "basis": "Literal text."}]
            }
        ]
    }
    is_v, msg = validate_structured_ledger(valid_ledger)
    assert is_v, msg

    # 2. Risk diagnosis validation
    valid_cd = {
        "risk_diagnosis": {
            "risk_type": "UNDER_POLARIZATION",
            "risk_description": "Minimalist text"
        },
        "action": "VISION_PROBE",
        "target_aspect_id": "a_01",
        "question_for_vision": "Is the person smiling?"
    }
    is_v, msg = validate_risk_diagnosis(valid_cd)
    assert is_v, msg

    # 3. Text revision audit validation (CT)
    valid_ct = {
        "decision": "ACCEPT_TEXT_REVISION",
        "text_evidence_verified": True,
        "modifier_scope_changed": True,
        "reporting_frame_decoupled": False,
        "audit_rationale": "Clear modifier scope change."
    }
    is_v, msg = validate_text_revision_audit(valid_ct)
    assert is_v, msg

    # 4. Evidence firewall validation
    valid_ce = {
        "status": "VALID",
        "target_binding": "DIRECT",
        "usable_evidence": ["The car is red."],
        "revision_support": "SUPPORTS_REVISION"
    }
    is_v, msg = validate_evidence_firewall(valid_ce)
    assert is_v, msg

    # 5. Revision verifier validation (CV)
    valid_cv = {"decision": "REVERT_TEXT_BASELINE"}
    is_v, msg = validate_revision_verifier(valid_cv)
    assert is_v, msg


def test_semantic_firewall_critique_sanitizer():
    client = MockClient(override_action="TEXT_RETHINK", leak_visual_in_critique=True)
    controller = MetaController(client)

    h_a = {"aspects": [{"aspect_id": "a_01", "text": "Obama", "sentiment": "POS"}]}
    v0 = {"scene": "Press briefing with smiling faces"}

    res, _, _ = controller.diagnose_risk(h_a=h_a, v0=v0, budget=2)
    assert res["action"] == "TEXT_RETHINK"
    # Ensure critique was automatically sanitized by semantic firewall
    critique = res["critique_for_text"]
    assert "image" not in critique.lower()
    assert "photo" not in critique.lower()
    assert "smiling" not in critique.lower()
    assert res.get("critique_sanitized") is True


def test_text_reasoner_aspect_isolation():
    client = MockClient()
    reasoner = TextReasoner(client)

    h_baseline = {
        "aspects": [
            {"aspect_id": "a_01", "text": "Apple", "span": [0, 5], "sentiment": "POS"},
            {"aspect_id": "a_02", "text": "Tim Cook", "span": [10, 18], "sentiment": "NEU"}
        ],
        "pairs": [["Apple", "POS"], ["Tim Cook", "NEU"]]
    }

    # Execute rethink_text targeting ONLY a_01
    ledger, _, _ = reasoner.rethink_text(
        text="Apple CEO Tim Cook announces new device.",
        h_baseline=h_baseline,
        critique="Ensure modifier neutrality for Apple.",
        target_aspect_id="a_01"
    )

    # a_01 updated, a_02 remains strictly locked and untouched
    assert ledger["aspects"][0]["aspect_id"] == "a_01"
    assert ledger["aspects"][1]["aspect_id"] == "a_02"
    assert ledger["aspects"][1]["sentiment"] == "NEU"  # Untouched


def test_meta_controller_safeguard_revert_to_hb():
    client = MockClient()
    controller = MetaController(client)

    h_a = {
        "aspects": [{"aspect_id": "a_01", "text": "Obama", "sentiment": "POS"}],
        "pairs": [["Obama", "POS"]]
    }
    h_b = {
        "aspects": [{"aspect_id": "a_01", "text": "Obama", "sentiment": "NEU"}],
        "pairs": [["Obama", "NEU"]]
    }
    h_candidate = {
        "aspects": [{"aspect_id": "a_01", "text": "Obama", "sentiment": "POS"}],
        "pairs": [["Obama", "POS"]]
    }
    insufficient_ev = {
        "status": "INSUFFICIENT",
        "target_binding": "UNBOUND",
        "usable_evidence": [],
        "revision_support": "NON_DECISIVE"
    }

    # CV must trigger automatic programmatic safeguard to REVERT_TEXT_BASELINE
    res, _, _ = controller.verify_revision(
        h_a=h_a,
        h_b=h_b,
        h_current=h_candidate,
        verified_evidence=insufficient_ev,
        budget=0
    )
    assert res["decision"] == "REVERT_TEXT_BASELINE"
    assert "Safeguard Activated" in res["audit_rationale"]


def test_pipeline_v3_dual_baseline_and_revert_to_hb():
    """Validates the core user request:
    HA starts as POS.
    Step 1: CD dispatches TEXT_RETHINK, TR modifies to NEU, CT certifies it -> HB becomes NEU.
    Step 2: CD dispatches VISION_PROBE, IP fails (INSUFFICIENT evidence).
    Result: System executes REVERT_TEXT_BASELINE, ending at HB (NEU), NOT falling back to old HA (POS)!
    """
    client = MockClient(rethink_sentiment="NEU", override_firewall_status="INSUFFICIENT")
    client.step_action_sequence = ["TEXT_RETHINK", "VISION_PROBE", "FINALIZE"]
    pipeline = BACRPipelineV3(client=client, max_visual_probes=2)

    sample = {
        "sample_id": "test_dual_001",
        "text": "Obama speaks at news press conference #AwesomeEvent.",
        "image": "test.jpg",
        "pairs": [["Obama", "NEU"]]
    }

    record = pipeline.run_sample(sample=sample, image_base_dir=".")

    # 1. HA is permanently preserved as historical POS anchor
    assert record["text_anchor"]["pairs"] == [["Obama", "POS"]]
    # 2. HB was correctly updated to NEU after CT certified TEXT_RETHINK
    assert record["text_baseline"]["pairs"] == [["Obama", "NEU"]]
    # 3. Final pairs retain HB (NEU) instead of corrupting back to HA (POS)
    assert record["final_pairs"] == [["Obama", "NEU"]]
    # 4. Aspect state verified
    assert record["aspect_states"]["a_01"]["baseline_sentiment"] == "NEU"
    assert record["aspect_states"]["a_01"]["anchor_sentiment"] == "POS"


def test_pipeline_v3_vision_probe_flow():
    client = MockClient(override_action="VISION_PROBE", override_firewall_status="VALID")
    pipeline = BACRPipelineV3(client=client, max_visual_probes=2)

    sample = {
        "sample_id": "test_001",
        "text": "Obama gives speech at event.",
        "image": "test.jpg",
        "pairs": [["Obama", "POS"]]
    }

    record = pipeline.run_sample(sample=sample, image_base_dir=".")

    assert record["sample_id"] == "test_001"
    assert len(record["rounds"]) == 1
    assert record["final_pairs"] == [["Obama", "POS"]]
    assert record["rounds"][0]["verifier_decision"]["decision"] == "ACCEPT_REVISION"


def test_pipeline_v3_finalize_flow():
    client = MockClient(override_action="FINALIZE")
    pipeline = BACRPipelineV3(client=client, max_visual_probes=2)

    sample = {
        "sample_id": "test_002",
        "text": "Neutral news report on meeting.",
        "image": "test.jpg",
        "pairs": [["Obama", "POS"]]
    }

    record = pipeline.run_sample(sample=sample, image_base_dir=".")
    assert record["action"] == "FINALIZE"
    assert record["num_visual_probes"] == 0
    assert record["final_pairs"] == [["Obama", "POS"]]


def test_v3_diagnostics_calculation(tmp_path):
    # Prepare dummy trajectory file
    traj_file = str(tmp_path / "trajs.jsonl")
    gold_file = str(tmp_path / "gold.jsonl")

    gold_data = [
        {"sample_id": "s1", "pairs": [["A", "POS"]]},
        {"sample_id": "s2", "pairs": [["B", "POS"]]},
        {"sample_id": "s3", "pairs": [["C", "NEU"]]},
    ]
    with open(gold_file, "w", encoding="utf-8") as f:
        for d in gold_data:
            f.write(json.dumps(d) + "\n")

    trajs = [
        {
            "sample_id": "s1",
            "contrast_type": "NO_RISK",
            "action": "FINALIZE",
            "num_visual_probes": 0,
            "stage_predictions": {
                "Y_T": [["A", "POS"]],
                "Y_TV": [["A", "POS"]],
                "Y_final": [["A", "POS"]]
            }
        },
        {
            "sample_id": "s2",
            "contrast_type": "UNDER_POLARIZATION",
            "action": "VISION_PROBE",
            "num_visual_probes": 1,
            "rounds": [{"verifier_decision": {"decision": "ACCEPT_REVISION"}}],
            "stage_predictions": {
                "Y_T": [["B", "NEU"]],
                "Y_TV": [["B", "POS"]],
                "Y_final": [["B", "POS"]]
            }
        },
        {
            "sample_id": "s3",
            "contrast_type": "UNDER_POLARIZATION",
            "action": "VISION_PROBE",
            "num_visual_probes": 1,
            "rounds": [{"verifier_decision": {"decision": "REVERT_TEXT_BASELINE"}}],
            "stage_predictions": {
                "Y_T": [["C", "NEU"]],
                "Y_TV": [["C", "NEU"]],
                "Y_final": [["C", "NEU"]]
            }
        }
    ]
    with open(traj_file, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")

    diag = evaluate_v3_trajectories(traj_file, gold_file)
    assert diag["total_samples"] == 3
    assert diag["acc_t0"] == 66.67
    assert diag["acc_final"] == 100.0
    assert diag["anchor_corrections"] == 1
    assert diag["anchor_corruptions"] == 0
    assert diag["anchor_net_gain"] == 1
    assert diag["revert_count"] == 1
    assert diag["action_distribution"]["FINALIZE"] == 1
    assert diag["action_distribution"]["VISION_PROBE"] == 2


if __name__ == "__main__":
    test_v3_schemas()
    test_semantic_firewall_critique_sanitizer()
    test_text_reasoner_aspect_isolation()
    test_meta_controller_safeguard_revert_to_hb()
    test_pipeline_v3_dual_baseline_and_revert_to_hb()
    test_pipeline_v3_vision_probe_flow()
    test_pipeline_v3_finalize_flow()
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as td:
        test_v3_diagnostics_calculation(pathlib.Path(td))
    print("All BACR-v3 unit tests (dual baseline, CT audit, aspect state, semantic firewall) passed successfully!")

