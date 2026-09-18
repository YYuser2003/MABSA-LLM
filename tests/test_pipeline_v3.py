"""Comprehensive Unit Tests for BACR-v3 Minimal Teacher Architecture.

Validates the 6 Core Execution Paths and Invariants:
1. Pydantic v2 schemas and fail-closed contracts (EvidenceResult, RouteDecision, FinalAudit).
2. Route KEEP -> Final = T0 (no intervention).
3. Route TEXT + Audit ACCEPT -> Final = Candidate.
4. Route TEXT + Audit REVERT -> Safeguard fallback to T0.
5. Route VISION + Firewall INVALID -> Fail-closed fallback to T0 (TF bypassed).
6. Route VISION + Firewall VALID + Audit ACCEPT -> Final = Candidate.
7. Route VISION + Firewall VALID + Audit REVERT -> Safeguard fallback to T0.
8. Semantic firewall sanitization (V0 never leaks into text critique).
9. Multi-aspect factoring and independent aggregation (K=1 invariant).
10. Teacher verification diagnostic evaluator (Recover - Harm > 0).
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
    RouteAction,
    EvidenceStatus,
    TargetBinding,
    AuditDecision,
    RouteDecision,
    EvidenceResult,
    CandidatePrediction,
    FinalAudit
)
from bacr.text_reasoner import TextReasoner
from bacr.meta_controller import MetaController
from bacr.vision_sensor import VisionSensor
from bacr.pipeline_v3 import BACRPipelineV3
from bacr.evaluator import evaluate_v3_teacher, print_v3_teacher_report


class MockTeacherClient:
    """Mock client returning deterministic responses for BACR-v3 Minimal Teacher."""
    def __init__(
        self,
        route_action="KEEP",
        route_critique=None,
        route_question=None,
        rethink_sentiment="NEU",
        firewall_status="VALID",
        firewall_binding="DIRECT",
        firewall_revision_support="SUPPORTS_REVISION",
        firewall_relevance="HIGH",
        fusion_sentiment="POS",
        audit_decision="ACCEPT",
        leak_visual_in_critique=False
    ):
        self.route_action = route_action
        self.route_critique = route_critique
        self.route_question = route_question
        self.rethink_sentiment = rethink_sentiment
        self.firewall_status = firewall_status
        self.firewall_binding = firewall_binding
        self.firewall_revision_support = firewall_revision_support
        self.firewall_relevance = firewall_relevance
        self.fusion_sentiment = fusion_sentiment
        self.audit_decision = audit_decision
        self.leak_visual_in_critique = leak_visual_in_critique
        self.call_history = []

    def call_text(self, system_prompt: str, user_prompt: str = "", **kwargs):
        self.call_history.append(("call_text", system_prompt, user_prompt))

        # C_R: Route Decision
        if "controller_route.md" in system_prompt or "Meta-Cognitive Routing Controller" in system_prompt:
            crit = self.route_critique
            if self.leak_visual_in_critique:
                crit = "The photo image shows a smiling face, please re-evaluate tone."
            return {
                "risk_type": "AFFECT_SPILLOVER" if self.route_action == "TEXT" else "MISSING_AFFECT",
                "action": self.route_action,
                "reason": f"Selected {self.route_action} route based on audit.",
                "critique": crit,
                "question": self.route_question or "What expression is visible?"
            }, {"total_tokens": 20}, 0.05

        # TR: Text Rethink
        elif "text_rethink.md" in system_prompt or "Text Deliberative Reasoner" in system_prompt:
            return {
                "aspect": "target",
                "sentiment": self.rethink_sentiment,
                "reason": "Re-evaluated text syntax addressing critique.",
                "evidence": ["purely linguistic clause"]
            }, {"total_tokens": 25}, 0.05

        # C_E: Evidence Firewall
        elif "controller_evidence_firewall.md" in system_prompt or "Evidence Firewall Controller" in system_prompt:
            if self.firewall_status == "VALID":
                return {
                    "status": "VALID",
                    "target_binding": self.firewall_binding,
                    "relevance": self.firewall_relevance,
                    "revision_support": self.firewall_revision_support,
                    "usable_evidence": ["Target entity displays broad smile and raised arms."],
                    "rejected_inferences": [],
                    "verification_notes": "Physical verifiable facts bound to target."
                }, {"total_tokens": 20}, 0.05
            else:
                return {
                    "status": self.firewall_status,
                    "target_binding": self.firewall_binding,
                    "relevance": self.firewall_relevance,
                    "revision_support": self.firewall_revision_support,
                    "usable_evidence": [],
                    "rejected_inferences": ["Subjective claim about mood."],
                    "verification_notes": "Evidence is unobservable or unbound."
                }, {"total_tokens": 20}, 0.05

        # TF: Evidence Fusion Reasoner
        elif "text_evidence_fusion.md" in system_prompt or "Deliberative Cross-Modal Reasoner" in system_prompt:
            return {
                "aspect": "target",
                "sentiment": self.fusion_sentiment,
                "reason": "Synthesized text with verified physical visual proof.",
                "evidence": ["Target entity displays broad smile and raised arms."]
            }, {"total_tokens": 30}, 0.05

        # C_F: Final Audit Verifier
        elif "controller_final_audit.md" in system_prompt or "Final Audit Verifier" in system_prompt:
            return {
                "decision": self.audit_decision,
                "reason": f"Revision was evaluated as {self.audit_decision}."
            }, {"total_tokens": 15}, 0.05

        return {}, {"total_tokens": 0}, 0.0

    def call_vision(self, system_prompt: str, user_prompt: str = "", image_path: str = "", user_text: str = "", **kwargs):
        self.call_history.append(("call_vision", system_prompt, user_text or user_prompt))

        # IG: Global Visual Sketch
        if "vision_global.md" in system_prompt or "Global Visual Perception" in system_prompt:
            return {
                "scene": "Press conference hall",
                "description": "Speaker giving presentation at podium.",
                "possible_entities": ["Obama", "Biden"],
                "ocr": ["WHITE HOUSE"],
                "salient_visual_cues": ["podium", "microphones"]
            }, {"total_tokens": 30}, 0.08

        # IP: Deep Probe
        elif "vision_probe.md" in system_prompt or "Targeted Deep Visual Sensor" in system_prompt:
            return {
                "answer": "Target person is smiling warmly at podium.",
                "observable_evidence": ["Raised mouth corners", "smiling cheeks"],
                "certainty": "high",
                "insufficient_visual_evidence": False
            }, {"total_tokens": 25}, 0.08

        return {}, {"total_tokens": 0}, 0.0


def test_v3_schemas_and_fail_closed_contract():
    print("--- Running test_v3_schemas_and_fail_closed_contract ---")
    # 1. RouteDecision validation & fallback
    r_empty = RouteDecision.validate_or_fallback({})
    assert r_empty.action == RouteAction.KEEP

    r_malformed = RouteDecision.validate_or_fallback("not a dict")
    assert r_malformed.action == RouteAction.KEEP

    r_valid = RouteDecision.validate_or_fallback({"action": "TEXT", "critique": "check scope"})
    assert r_valid.action == RouteAction.TEXT
    assert r_valid.critique == "check scope"

    # 2. EvidenceResult fail-closed contracts
    ev_empty = EvidenceResult.validate_fail_closed({})
    assert ev_empty.status == EvidenceStatus.INVALID
    assert ev_empty.usable_evidence == []

    # Invariant: INSUFFICIENT must clear usable_evidence
    ev_insufficient = EvidenceResult.validate_fail_closed({
        "status": "INSUFFICIENT",
        "usable_evidence": ["Spurious visual note"]
    })
    assert ev_insufficient.status == EvidenceStatus.INSUFFICIENT
    assert ev_insufficient.usable_evidence == []

    # Invariant: UNBOUND must clear usable_evidence
    ev_unbound = EvidenceResult.validate_fail_closed({
        "status": "VALID",
        "target_binding": "UNBOUND",
        "usable_evidence": ["Smiling crowd"]
    })
    assert ev_unbound.usable_evidence == []

    # 3. FinalAudit fallback
    fa_fallback = FinalAudit.validate_or_fallback({})
    assert fa_fallback.decision == AuditDecision.REVERT

    fa_accept = FinalAudit.validate_or_fallback({"decision": "ACCEPT", "reason": "Well grounded"})
    assert fa_accept.decision == AuditDecision.ACCEPT
    print("Passed test_v3_schemas_and_fail_closed_contract!")


def test_route_keep():
    print("--- Running test_route_keep ---")
    client = MockTeacherClient(route_action="KEEP")
    pipeline = BACRPipelineV3(client=client)

    sample = {
        "sample_id": "test_keep_01",
        "text": "Regular news announcement regarding Apple",
        "image": "dummy.jpg",
        "pairs": [["Apple", "NEU"]],
        "text_initial_cached": {"pairs": [["Apple", "NEU"]]}
    }
    record = pipeline.run_sample(sample)

    assert record["final_pairs"] == [["Apple", "NEU"]]
    assert len(record["aspect_trajectories"]) == 1
    assert record["aspect_trajectories"][0]["route"] == "KEEP"
    assert record["aspect_trajectories"][0]["final_sentiment"] == "NEU"
    # Neither TR nor TF was called
    tr_calls = [c for c in client.call_history if "text_rethink.md" in c[1]]
    tf_calls = [c for c in client.call_history if "text_evidence_fusion.md" in c[1]]
    assert len(tr_calls) == 0
    assert len(tf_calls) == 0
    print("Passed test_route_keep!")


def test_route_text_accept():
    print("--- Running test_route_text_accept ---")
    client = MockTeacherClient(
        route_action="TEXT",
        rethink_sentiment="NEG",
        audit_decision="ACCEPT"
    )
    pipeline = BACRPipelineV3(client=client)

    sample = {
        "sample_id": "test_text_accept_01",
        "text": "Terrible tragedy struck Microsoft campus # sad",
        "image": "dummy.jpg",
        "pairs": [["Microsoft", "NEG"]],
        "text_initial_cached": {"pairs": [["Microsoft", "NEU"]]}  # T0 was NEU
    }
    record = pipeline.run_sample(sample)

    assert record["final_pairs"] == [["Microsoft", "NEG"]]
    asp_tr = record["aspect_trajectories"][0]
    assert asp_tr["route"] == "TEXT"
    assert asp_tr["candidate_sentiment"] == "NEG"
    assert asp_tr["audit_decision"] == "ACCEPT"
    assert asp_tr["final_sentiment"] == "NEG"
    print("Passed test_route_text_accept!")


def test_route_text_revert():
    print("--- Running test_route_text_revert ---")
    # Controller routes TEXT, candidate is proposed as POS, but CF REVERTS
    client = MockTeacherClient(
        route_action="TEXT",
        rethink_sentiment="POS",
        audit_decision="REVERT"
    )
    pipeline = BACRPipelineV3(client=client)

    sample = {
        "sample_id": "test_text_revert_01",
        "text": "Neutral financial earnings for Google",
        "image": "dummy.jpg",
        "pairs": [["Google", "NEU"]],
        "text_initial_cached": {"pairs": [["Google", "NEU"]]}  # T0 was NEU
    }
    record = pipeline.run_sample(sample)

    # Invariant: CF REVERT forces fallback to T0 (NEU)
    assert record["final_pairs"] == [["Google", "NEU"]]
    asp_tr = record["aspect_trajectories"][0]
    assert asp_tr["route"] == "TEXT"
    assert asp_tr["candidate_sentiment"] == "POS"
    assert asp_tr["audit_decision"] == "REVERT"
    assert asp_tr["final_sentiment"] == "NEU"
    print("Passed test_route_text_revert!")


def test_route_vision_firewall_fail_closed():
    print("--- Running test_route_vision_firewall_fail_closed ---")
    # Controller routes VISION, but Firewall declares INVALID
    client = MockTeacherClient(
        route_action="VISION",
        firewall_status="INVALID",
        firewall_binding="UNBOUND"
    )
    pipeline = BACRPipelineV3(client=client)

    sample = {
        "sample_id": "test_vision_fail_closed_01",
        "text": "Tesla announces event",
        "image": "dummy.jpg",
        "pairs": [["Tesla", "NEU"]],
        "text_initial_cached": {"pairs": [["Tesla", "NEU"]]}
    }
    record = pipeline.run_sample(sample)

    # Invariant: EvidenceInvalid => T0 directly, and TF is NEVER called!
    assert record["final_pairs"] == [["Tesla", "NEU"]]
    asp_tr = record["aspect_trajectories"][0]
    assert asp_tr["route"] == "VISION"
    assert asp_tr["evidence"]["status"] == "INVALID"
    assert asp_tr["audit_decision"] == "REVERT"
    assert asp_tr["final_sentiment"] == "NEU"

    tf_calls = [c for c in client.call_history if "text_evidence_fusion.md" in c[1]]
    assert len(tf_calls) == 0, "TF should NOT have been invoked when firewall was INVALID!"
    print("Passed test_route_vision_firewall_fail_closed!")


def test_route_vision_valid_accept():
    print("--- Running test_route_vision_valid_accept ---")
    client = MockTeacherClient(
        route_action="VISION",
        firewall_status="VALID",
        firewall_binding="DIRECT",
        fusion_sentiment="POS",
        audit_decision="ACCEPT"
    )
    pipeline = BACRPipelineV3(client=client)

    sample = {
        "sample_id": "test_vision_valid_accept_01",
        "text": "Ambiguous caption on photo with Elon Musk",
        "image": "dummy.jpg",
        "pairs": [["Elon Musk", "POS"]],
        "text_initial_cached": {"pairs": [["Elon Musk", "NEU"]]}
    }
    record = pipeline.run_sample(sample)

    assert record["final_pairs"] == [["Elon Musk", "POS"]]
    asp_tr = record["aspect_trajectories"][0]
    assert asp_tr["route"] == "VISION"
    assert asp_tr["evidence"]["status"] == "VALID"
    assert asp_tr["candidate_sentiment"] == "POS"
    assert asp_tr["audit_decision"] == "ACCEPT"
    assert asp_tr["final_sentiment"] == "POS"
    print("Passed test_route_vision_valid_accept!")


def test_route_vision_valid_revert():
    print("--- Running test_route_vision_valid_revert ---")
    # Valid physical evidence, TF suggests NEG, but CF audits as speculative/REVERT
    client = MockTeacherClient(
        route_action="VISION",
        firewall_status="VALID",
        firewall_binding="DIRECT",
        fusion_sentiment="NEG",
        audit_decision="REVERT"
    )
    pipeline = BACRPipelineV3(client=client)

    sample = {
        "sample_id": "test_vision_valid_revert_01",
        "text": "Photo of team celebration",
        "image": "dummy.jpg",
        "pairs": [["Team", "POS"]],
        "text_initial_cached": {"pairs": [["Team", "POS"]]}
    }
    record = pipeline.run_sample(sample)

    # Invariant: Audit REVERT safeguards T0 (POS)
    assert record["final_pairs"] == [["Team", "POS"]]
    asp_tr = record["aspect_trajectories"][0]
    assert asp_tr["route"] == "VISION"
    assert asp_tr["candidate_sentiment"] == "NEG"
    assert asp_tr["audit_decision"] == "REVERT"
    assert asp_tr["final_sentiment"] == "POS"
    print("Passed test_route_vision_valid_revert!")


def test_semantic_firewall_leak_sanitization():
    print("--- Running test_semantic_firewall_leak_sanitization ---")
    client = MockTeacherClient(
        route_action="TEXT",
        leak_visual_in_critique=True
    )
    controller = MetaController(client=client)

    route_dec, _, _ = controller.decide_route(
        aspect="Obama",
        anchor={"sentiment": "POS"},
        visual_sketch={"scene": "photo with smiling face"}
    )
    # Physical isolation: critique is generated from isolated C_Q^T without visual leakage
    assert "smiling" not in route_dec.critique
    assert "photo" not in route_dec.critique
    assert "Obama" in route_dec.critique
    assert "spilled over" in route_dec.critique or "syntax" in route_dec.critique
    print("Passed test_semantic_firewall_leak_sanitization!")


def test_multi_aspect_sample_factoring():
    print("--- Running test_multi_aspect_sample_factoring ---")
    client = MockTeacherClient(route_action="KEEP")
    pipeline = BACRPipelineV3(client=client)

    sample = {
        "sample_id": "test_multi_01",
        "text": "Obama and Biden both spoke at the ceremony",
        "image": "dummy.jpg",
        "pairs": [["Obama", "POS"], ["Biden", "NEU"]],
        "text_initial_cached": {
            "pairs": [["Obama", "POS"], ["Biden", "NEU"]]
        }
    }
    record = pipeline.run_sample(sample)

    # Invariant K=1: Multi-aspect samples must factor cleanly into 2 aspect trajectories
    assert len(record["aspect_trajectories"]) == 2
    assert record["final_pairs"] == [["Obama", "POS"], ["Biden", "NEU"]]
    asps_evaluated = [at["aspect"] for at in record["aspect_trajectories"]]
    assert "Obama" in asps_evaluated
    assert "Biden" in asps_evaluated
    print("Passed test_multi_aspect_sample_factoring!")


def test_evaluator_v3_teacher(tmp_path: pathlib.Path):
    print("--- Running test_evaluator_v3_teacher ---")
    gold_file = str(tmp_path / "gold.jsonl")
    traj_file = str(tmp_path / "traj.jsonl")

    gold_data = [
        {"sample_id": "s1", "pairs": [["Obama", "POS"]]},
        {"sample_id": "s2", "pairs": [["Trump", "NEG"]]},
        {"sample_id": "s3", "pairs": [["Biden", "NEU"]]}
    ]
    with open(gold_file, "w", encoding="utf-8") as f:
        for g in gold_data:
            f.write(json.dumps(g) + "\n")

    # s1: T0=NEU (wrong), Final=POS (correct) -> Recover!
    # s2: T0=NEG (correct), Final=NEG (correct) -> Maintained!
    # s3: T0=NEU (correct), Final=NEU (correct) -> Maintained!
    # Recover = 1, Harm = 0 => Net Gain = +1 (Hypothesis Satisfied!)
    trajs = [
        {
            "sample_id": "s1",
            "t0_pairs": [["Obama", "NEU"]],
            "final_pairs": [["Obama", "POS"]],
            "aspect_trajectories": [
                {
                    "aspect": "Obama",
                    "t0_sentiment": "NEU",
                    "route": "TEXT",
                    "audit_decision": "ACCEPT",
                    "final_sentiment": "POS"
                }
            ]
        },
        {
            "sample_id": "s2",
            "t0_pairs": [["Trump", "NEG"]],
            "final_pairs": [["Trump", "NEG"]],
            "aspect_trajectories": [
                {
                    "aspect": "Trump",
                    "t0_sentiment": "NEG",
                    "route": "KEEP",
                    "audit_decision": "N/A",
                    "final_sentiment": "NEG"
                }
            ]
        },
        {
            "sample_id": "s3",
            "t0_pairs": [["Biden", "NEU"]],
            "final_pairs": [["Biden", "NEU"]],
            "aspect_trajectories": [
                {
                    "aspect": "Biden",
                    "t0_sentiment": "NEU",
                    "route": "VISION",
                    "audit_decision": "REVERT",
                    "final_sentiment": "NEU"
                }
            ]
        }
    ]
    with open(traj_file, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")

    res = evaluate_v3_teacher(traj_file=traj_file, gold_file=gold_file)
    assert res["total_samples"] == 3
    assert res["total_aspects"] == 3
    assert res["total_recover"] == 1
    assert res["total_harm"] == 0
    assert res["net_gain"] == 1
    assert res["hypothesis_satisfied"] is True
    assert res["aspect_t0_acc"] == 66.67
    assert res["aspect_final_acc"] == 100.0

    print_v3_teacher_report(res)
    print("Passed test_evaluator_v3_teacher!")


def test_route_vision_firewall_non_decisive_reverts():
    print("--- Running test_route_vision_firewall_non_decisive_reverts ---")
    # Sensor returns VALID + DIRECT but NON_DECISIVE support -> Firewall must fail-closed & revert to T0
    client = MockTeacherClient(
        route_action="VISION",
        firewall_status="VALID",
        firewall_binding="DIRECT",
        firewall_revision_support="NON_DECISIVE",
        fusion_sentiment="POS",
        audit_decision="ACCEPT"
    )
    pipeline = BACRPipelineV3(client=client)

    sample = {
        "sample_id": "test_non_decisive_01",
        "text": "Check out the new features",
        "image": "dummy.jpg",
        "pairs": [["Features", "NEU"]],
        "text_initial_cached": {"pairs": [["Features", "NEU"]]}
    }
    record = pipeline.run_sample(sample)

    assert record["final_pairs"] == [["Features", "NEU"]]
    asp_tr = record["aspect_trajectories"][0]
    assert asp_tr["route"] == "VISION"
    assert asp_tr["final_sentiment"] == "NEU"
    assert asp_tr["audit_decision"] == "REVERT"

    # TF Deliberation MUST be bypassed when evidence is NON_DECISIVE
    tf_calls = [c for c in client.call_history if "text_evidence_fusion.md" in c[1]]
    assert len(tf_calls) == 0, "TF Deliberative reasoner should be bypassed when revision_support is NON_DECISIVE"
    print("Passed test_route_vision_firewall_non_decisive_reverts!")


def test_evaluator_cw_transition_count(tmp_path: pathlib.Path):
    print("--- Running test_evaluator_cw_transition_count ---")
    from bacr.evaluator import G3Evaluator
    gold_file = str(tmp_path / "cw_gold.jsonl")
    pred_file = str(tmp_path / "cw_pred.jsonl")

    # Sample 1: T0 was correct (POS), final was wrong (NEG) -> This is CW harm!
    # Sample 2: T0 was wrong (NEU), final was correct (POS) -> This is WC recover!
    with open(gold_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": "s1", "pairs": [["ItemA", "POS"]]}) + "\n")
        f.write(json.dumps({"sample_id": "s2", "pairs": [["ItemB", "POS"]]}) + "\n")

    with open(pred_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "sample_id": "s1",
            "text_initial": {"pairs": [["ItemA", "POS"]]},
            "predictions": [["ItemA", "NEG"]]
        }) + "\n")
        f.write(json.dumps({
            "sample_id": "s2",
            "text_initial": {"pairs": [["ItemB", "NEU"]]},
            "predictions": [["ItemB", "POS"]]
        }) + "\n")

    evaluator = G3Evaluator(gold_file=gold_file)
    res = evaluator.evaluate_predictions(pred_file=pred_file)
    sample_diag = res.get("sample_level_diagnostics", {})
    assert sample_diag.get("sample_cw_harmful") == 1, f"Expected sample_cw_harmful == 1, got {sample_diag.get('sample_cw_harmful')}"
    assert sample_diag.get("sample_wc_recovery") == 1, f"Expected sample_wc_recovery == 1, got {sample_diag.get('sample_wc_recovery')}"
    assert sample_diag.get("sample_cc_maintained") == 0
    assert sample_diag.get("sample_ww_unresolved") == 0
    print("Passed test_evaluator_cw_transition_count!")


def test_exact_aspect_matching_no_substring_confusion():
    print("--- Running test_exact_aspect_matching_no_substring_confusion ---")
    client = MockTeacherClient(route_action="KEEP")
    pipeline = BACRPipelineV3(client=client)

    # Sample has aspect "Apple". Cached T0 only has "Apple Watch" with POS.
    # Because substring matching is forbidden, "Apple" must NOT match "Apple Watch",
    # and must fallback to default NEU baseline rather than stealing POS.
    sample = {
        "sample_id": "test_exact_match_01",
        "text": "Apple launched something, Apple Watch is great",
        "image": "dummy.jpg",
        "pairs": [["Apple", "NEU"]],
        "text_initial_cached": {"pairs": [["Apple Watch", "POS"]]}
    }
    record = pipeline.run_sample(sample)
    assert record["t0_pairs"] == [["Apple", "NEU"]], f"Got {record['t0_pairs']}, expected default NEU without substring match"
    print("Passed test_exact_aspect_matching_no_substring_confusion!")


def test_physical_isolation_of_text_critique():
    print("--- Running test_physical_isolation_of_text_critique ---")
    client = MockTeacherClient(route_action="KEEP")
    controller = MetaController(client=client)

    # Isolated generator strictly receives (aspect, anchor, risk_type) - zero visual exposure
    crit_spill = controller.generate_text_critique(
        aspect="MacBook",
        anchor={"sentiment": "NEG"},
        risk_type="AFFECT_SPILLOVER"
    )
    assert "MacBook" in crit_spill
    assert "adjacent clauses" in crit_spill
    assert "photo" not in crit_spill
    assert "image" not in crit_spill

    crit_frame = controller.generate_text_critique(
        aspect="CEO",
        anchor={"sentiment": "POS"},
        risk_type="REPORTING_FRAME"
    )
    assert "CEO" in crit_frame
    assert "reporting" in crit_frame
    assert "visual" not in crit_frame
    print("Passed test_physical_isolation_of_text_critique!")


if __name__ == "__main__":
    test_v3_schemas_and_fail_closed_contract()
    test_route_keep()
    test_route_text_accept()
    test_route_text_revert()
    test_route_vision_firewall_fail_closed()
    test_route_vision_valid_accept()
    test_route_vision_valid_revert()
    test_route_vision_firewall_non_decisive_reverts()
    test_semantic_firewall_leak_sanitization()
    test_physical_isolation_of_text_critique()
    test_exact_aspect_matching_no_substring_confusion()
    test_multi_aspect_sample_factoring()
    with tempfile.TemporaryDirectory() as td:
        test_evaluator_v3_teacher(pathlib.Path(td))
        test_evaluator_cw_transition_count(pathlib.Path(td))

    print("\n" + "=" * 78)
    print("  ALL 14 BACR-v3 TEACHER INVARIANT & ARCHITECTURAL TESTS PASSED 100%!")
    print("=" * 78 + "\n")


