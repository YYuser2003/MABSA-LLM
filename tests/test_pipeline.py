"""Unit tests for BACR Pipeline, Controller & Taxonomy."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bacr.taxonomy import (
    classify_vision_query,
    classify_text_query,
    VISION_QUERY_TYPES,
    TEXT_QUERY_TYPES
)
from bacr.controller import BACRController, AspectBeliefState
from bacr.pipeline import BACRPipeline, G3Pipeline


def test_taxonomy_classification():
    # Vision queries
    assert classify_vision_query("Is this player smiling?") == "AFFECT"
    assert classify_vision_query("What is the text written on the shirt?") == "OCR_LOGO"
    assert classify_vision_query("Is this person recognizable as Donald Trump?") == "IDENTITY"

    # Text queries
    assert classify_vision_query("Is this statement sarcastic or literal?") in ["CONFLICT_CHECK", "GENERAL"]
    assert classify_text_query("Is the tweet sarcastic?") == "SARCASM"
    assert classify_text_query("What does the pronoun 'it' refer to?") == "REFERENCE"


def test_controller_decision():
    controller = BACRController(max_turns=2, confidence_threshold=0.85)
    beliefs = [
        AspectBeliefState(
            aspect_id="a_01",
            aspect_text="Apple",
            span=(0, 5),
            current_sentiment="POS",
            confidence=0.95,
            linguistic_flags=["literal"]
        )
    ]
    assert controller.evaluate_termination(turn=0, beliefs=beliefs) is True

    # Test gap_diagnosis parsing in select_probing_action on uncertain beliefs
    uncertain_beliefs = [
        AspectBeliefState(
            aspect_id="a_01",
            aspect_text="Apple",
            span=(0, 5),
            current_sentiment="NEU",
            confidence=0.5,
            linguistic_flags=["literal"]
        )
    ]
    raw_response = {
        "gap_diagnosis": {
            "gap_type": "VISUAL_AFFECT_GAP",
            "gap_description": "Facial expression unconfirmed."
        },
        "next_action": "QUERY",
        "direction": "VISUAL",
        "target_aspect_id": "a_01",
        "question": "Is the person smiling?"
    }
    decision = controller.select_probing_action(turn=0, beliefs=uncertain_beliefs, raw_controller_response=raw_response)
    assert decision.direction == "VISUAL_PROBE"
    assert decision.gap_type == "VISUAL_AFFECT_GAP"
    assert decision.stop is False
    assert decision.target_aspect == "a_01"


def test_pipeline_raw_modality_masking():
    # Mock client to inspect prompt payload
    class MockClient:
        def __init__(self):
            self.last_user_prompt = ""
            self.model = "mock_model"
            self.thinking_level = "high"
            self.temperature = 0.1
            self.allow_fallback = False

        def call_text(self, system_prompt, user_prompt):
            self.last_user_prompt = user_prompt
            return {
                "gap_diagnosis": {"gap_type": "NO_GAP", "gap_description": "None"},
                "current_aspects": [
                    {"aspect_id": "a_01", "text": "Apple", "span": [0, 5], "sentiment": "POS", "reason": "positive word"}
                ],
                "updates": [],
                "next_action": "STOP",
                "direction": "NONE",
                "stop_type": "natural_stop",
                "decision_reason": "Clear sentiment."
            }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

    mock_client = MockClient()
    pipeline = BACRPipeline(client=mock_client, controller_sees_raw_modalities=False)
    assert pipeline.controller_sees_raw_modalities is False

    # Execute step_controller
    aspects = [{"aspect_id": "a_01", "text": "Apple", "span": [0, 5], "sentiment": "POS", "reason": "great", "linguistic_flags": ["literal"]}]
    v0 = {"scene": "store", "description": "apple store", "possible_entities": [], "ocr": [], "salient_visual_cues": []}

    pipeline.step_controller(
        text="I love Apple products so much!",
        initial_aspects=aspects,
        image_initial=v0,
        history=[],
        current_aspects=aspects
    )

    # CRITICAL CHECK: Raw tweet text must NOT appear in user_prompt when controller_sees_raw_modalities is False!
    assert 'Tweet Text: "I love Apple products so much!"' not in mock_client.last_user_prompt
    assert "Text Cognitive Sketch (R_T^sketch):" in mock_client.last_user_prompt
    assert "Vision Cognitive Sketch (R_V^sketch):" in mock_client.last_user_prompt


def test_aspect_locked_updates():
    from bacr.pipeline import apply_aspect_locked_updates

    aspects = [
        {"aspect_id": "a_01", "text": "Barack Obama", "span": [0, 12], "sentiment": "NEU", "reason": "neutral text"},
        {"aspect_id": "a_02", "text": "yacht", "span": [20, 25], "sentiment": "NEU", "reason": "neutral text"}
    ]

    # Test 1: Valid direct update to targeted aspect
    updates_target = [
        {
            "aspect_id": "a_01",
            "new_sentiment": "POS",
            "evidence_refs": ["vision_step_1"],
            "evidence_quote": "Barack Obama is smiling warmly at the audience",
            "evidence_relation": "target_direct"
        }
    ]
    updated, accepted = apply_aspect_locked_updates(
        aspects, updates_target, valid_step_names=["vision_step_1"], target_aspect_id="a_01"
    )
    assert len(accepted) == 1
    assert updated[0]["sentiment"] == "POS"
    assert updated[1]["sentiment"] == "NEU"

    # Test 2: Target-Scoped Guard blocks affect leakage to a_02 when inquiry was targeting a_01
    leaking_updates = [
        {
            "aspect_id": "a_02",
            "new_sentiment": "POS",
            "evidence_refs": ["vision_step_1"],
            "evidence_quote": "Barack Obama is smiling warmly at the audience",  # No mention of yacht!
            "evidence_relation": "target_direct"
        }
    ]
    updated, accepted = apply_aspect_locked_updates(
        aspects, leaking_updates, valid_step_names=["vision_step_1"], target_aspect_id="a_01"
    )
    assert len(accepted) == 0
    assert updated[1]["sentiment"] == "NEU"

    # Test 3: Rejection of unregistered evidence sources
    unregistered_updates = [
        {
            "aspect_id": "a_01",
            "new_sentiment": "NEG",
            "evidence_refs": ["external_web_source"],
            "evidence_quote": "Negative gossip",
            "evidence_relation": "target_direct"
        }
    ]
    updated, accepted = apply_aspect_locked_updates(
        aspects, unregistered_updates, valid_step_names=["vision_step_1"], target_aspect_id="a_01"
    )
    assert len(accepted) == 0


def test_fact_sensor_guard():
    from bacr.pipeline import validate_question

    valid, _ = validate_question("Is the person wearing a red tie?")
    assert valid is True

    valid, reason = validate_question("Is the sentiment positive or negative?")
    assert valid is False
    assert "Fact-Sensor Guard" in reason

    valid, reason = validate_question("Classify the sentiment of Barack Obama")
    assert valid is False


def test_step_text_deep():
    class MockClient:
        def call_text(self, system_prompt, user_prompt):
            return {
                "aspect_id": "a_01",
                "aspect_text": "camera",
                "answer": "The word 'terrible' modifies 'camera' as a direct predicate adjective.",
                "evidence_spans": ["terrible camera"],
                "linguistic_relation": "modifier_scope",
                "certainty": "high",
                "insufficient_textual_evidence": False
            }, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 0.1

    pipeline = BACRPipeline(client=MockClient())
    data, usage, lat = pipeline.step_text_deep(
        text="The terrible camera ruined my trip.",
        question="Which word modifies camera?",
        aspect_text="camera",
        aspect_id="a_01"
    )

    assert "suggested_polarity" not in data  # Pure evidence sensor!
    assert data["answer"] == "The word 'terrible' modifies 'camera' as a direct predicate adjective."
    assert data["evidence_spans"] == ["terrible camera"]
    assert data["linguistic_relation"] == "modifier_scope"
    assert data["certainty"] == "high"


if __name__ == "__main__":
    test_taxonomy_classification()
    test_controller_decision()
    test_pipeline_raw_modality_masking()
    test_aspect_locked_updates()
    test_fact_sensor_guard()
    test_step_text_deep()
    print("All BACR Pipeline, Controller & Taxonomy tests passed successfully!")
