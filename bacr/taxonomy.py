"""Unified Taxonomy for BACR Active Probing Queries.

Defines:
1. VISION_QUERY_TYPES: 7 analytical categories for visual sensor probing
2. TEXT_QUERY_TYPES: 8 analytical categories for linguistic/pragmatic text deep probing
3. Classification heuristics for runtime & offline diagnostic analysis.
"""

from typing import Literal

VISION_QUERY_TYPES = [
    "IDENTITY",
    "OWNERSHIP",
    "AFFECT",
    "ACTION_STATE",
    "OCR_LOGO",
    "SCENE_CONTEXT",
    "CONFLICT_CHECK",
    "GENERAL"
]

TEXT_QUERY_TYPES = [
    "REFERENCE",
    "OWNERSHIP",
    "SCOPE",
    "SLANG",
    "SARCASM",
    "EVENT_RELATION",
    "ENTITY_RELATION",
    "POLARITY_CUE",
    "GENERAL"
]

VisionQueryCategory = Literal[
    "IDENTITY", "OWNERSHIP", "AFFECT", "ACTION_STATE",
    "OCR_LOGO", "SCENE_CONTEXT", "CONFLICT_CHECK", "GENERAL"
]

TextQueryCategory = Literal[
    "REFERENCE", "OWNERSHIP", "SCOPE", "SLANG",
    "SARCASM", "EVENT_RELATION", "ENTITY_RELATION", "POLARITY_CUE", "GENERAL"
]


def classify_vision_query(question: str) -> VisionQueryCategory:
    """Classifies a Controller visual query into one of the 7 analytical categories."""
    if not question:
        return "GENERAL"

    q_lower = question.lower()

    if any(k in q_lower for k in [
        "promotional", "merely", "stock photo", "staged", "actual outcome",
        "contradict", "conflict", "real performance", "unrelated"
    ]):
        return "CONFLICT_CHECK"

    if any(k in q_lower for k in [
        "ocr", "text", "logo", "banner", "written", "letters", "jersey",
        "number", "signage", "words", "spelling", "label"
    ]):
        return "OCR_LOGO"

    if any(k in q_lower for k in [
        "belong", "which team", "whose", "which entity", "belong to",
        "opposing", "opponent", "credited to"
    ]):
        return "OWNERSHIP"

    if any(k in q_lower for k in [
        "identifiable", "recognizable", "consistent with", "who is", "is the person", "is this person",
        "identity", "recognize", "recognized as", "specific player", "celebrity"
    ]):
        return "IDENTITY"

    if any(k in q_lower for k in [
        "smile", "smiling", "facial", "expression", "affect", "affective",
        "happy", "cheerful", "sad", "angry", "crying", "emotional state", "mood"
    ]):
        return "AFFECT"

    if any(k in q_lower for k in [
        "action", "celebrating", "celebration", "injured", "playing",
        "defeated", "applauding", "clapping", "holding", "standing", "seated"
    ]):
        return "ACTION_STATE"

    if any(k in q_lower for k in [
        "scene", "setting", "context", "event", "ceremony", "stadium",
        "conference", "awards", "atmosphere"
    ]):
        return "SCENE_CONTEXT"

    return "GENERAL"


def classify_text_query(question: str) -> TextQueryCategory:
    """Classifies a Controller text query into one of the 8 pragmatic categories."""
    if not question:
        return "GENERAL"

    q_lower = question.lower()

    if any(k in q_lower for k in ["sarcasm", "sarcastic", "irony", "ironic", "mocking", "literal"]):
        return "SARCASM"

    if any(k in q_lower for k in ["slang", "idiom", "colloquial", "metaphor", "hashtag", "jargon"]):
        return "SLANG"

    if any(k in q_lower for k in ["scope", "negation", "negate", "modifier", "adverb", "clause"]):
        return "SCOPE"

    if any(k in q_lower for k in ["refer", "pronoun", "antecedent", "it", "they", "this", "reference"]):
        return "REFERENCE"

    if any(k in q_lower for k in ["whose", "owner", "target of", "directed to", "towards"]):
        return "OWNERSHIP"

    if any(k in q_lower for k in ["event", "match", "game", "outcome", "happened", "incident"]):
        return "EVENT_RELATION"

    if any(k in q_lower for k in ["between", "entity", "relationship", "vs", "versus"]):
        return "ENTITY_RELATION"

    if any(k in q_lower for k in ["sentiment", "polarity", "positive", "negative", "neutral", "tone"]):
        return "POLARITY_CUE"

    return "GENERAL"


# Backward compatibility alias
classify_query = classify_vision_query
