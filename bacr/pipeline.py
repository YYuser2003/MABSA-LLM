"""Active Visual Reasoning Protocol (G3 v1.1) Core Pipeline.

Key Improvements in v1.1:
1. Aspect-Lock Enforcement: A_final == A_text_init (No aspect addition/deletion).
2. Evidence-Refs Binding & Evidence-Relation: Only target_direct or target_indirect can flip sentiment.
3. Dedicated Budget-Forced Stop Prompt and stop_type logging ('natural_stop' vs 'budget_forced_stop').
4. Fact-Sensor Guard: Intercepts questions asking for direct sentiment classification.
5. Experiment Provenance Fingerprint: SHA256 of image and prompt versions recorded in every trajectory.
6. Granular Compute & Resource Accounting: Tracks calls, tokens, image invocations, and latency per sample.
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from bacr.client import GeminiClient
from bacr.taxonomy import classify_query

SCHEMA_VERSION = "v1.2"

# =========================================================================
# SYSTEM PROMPTS (ALIGNED WITH G3 PROTOCOL v1.2: Option A + Option B)
# =========================================================================

TEXT_INITIAL_SYSTEM_PROMPT = """You are an expert Text-only Aspect-Based Sentiment Analysis reasoner.
Given a raw social media tweet, your mission is to extract the strongest possible text-only prediction alongside a structured cognitive sketch:
1. Extract all opinion target aspect terms mentioned in the text with exact character start and end offsets [start, end].
2. Assign each aspect a stable identifier: a_01, a_02, ...
3. Determine the sentiment polarity (POS, NEG, or NEU) based strictly on textual syntax and vocabulary. Do not guess about any accompanying image.
4. Provide a concise 1-sentence linguistic reason for each sentiment judgment.
5. Extract exact short phrases from the tweet as evidence_spans.
6. If the aspect has residual linguistic ambiguity, note it in unresolved_issue (or null if unambiguous).
7. Populate linguistic_flags ONLY if a specific linguistic challenge is present (e.g. sarcasm_irony, ambiguous_slang, coreference, negation_scope, modifier_scope, polysemy, comparative_scope, aspect_conflict). If standard and clear, output [].

Output strictly as a valid JSON object matching this schema:
{
  "aspects": [
    {
      "aspect_id": "a_01",
      "text": "exact aspect term",
      "span": [0, 5],
      "sentiment": "POS|NEG|NEU",
      "reason": "1-sentence textual justification",
      "evidence_spans": ["exact short phrase from tweet"],
      "linguistic_flags": [],
      "unresolved_issue": null
    }
  ],
  "pairs": [
    ["exact aspect term", "POS|NEG|NEU"]
  ],
  "linguistic_flags": []
}"""

TEXT_INITIAL_TARGET_GUIDED_PROMPT = """You are an expert Text-only Aspect-Based Sentiment Analysis reasoner.
Given a raw social media tweet and a specific list of target aspect terms:
Your task is to produce the best possible initial text prediction alongside a structured cognitive sketch.

For each target aspect:
1. Assign it a stable identifier: a_01, a_02, ... matching the order of the target aspects.
2. Locate its exact character start and end offset [start, end] in the tweet text (or [-1, -1] if implicit/hashtag variation).
3. Determine its sentiment polarity strictly as POS, NEG, or NEU based solely on textual context.
4. Provide a concise 1-sentence textual justification.
5. Extract exact short phrases from the tweet as evidence_spans.
6. Note any residual linguistic ambiguity in unresolved_issue (or null if unambiguous).
7. Populate linguistic_flags ONLY if a specific linguistic challenge is present (sarcasm_irony, ambiguous_slang, coreference, negation_scope, modifier_scope, polysemy, comparative_scope, aspect_conflict). If standard and clear, output [].

Rules:
- You MUST include ALL target aspects from the provided list. Do not omit any.
- Do not introduce aspects not in the provided target list.
- Do not guess or speculate about any accompanying image.

Output strictly as a valid JSON object matching this schema:
{
  "aspects": [
    {
      "aspect_id": "a_01",
      "text": "exact target aspect name",
      "span": [0, 5],
      "sentiment": "POS|NEG|NEU",
      "reason": "1-sentence textual justification based on tweet words",
      "evidence_spans": ["exact short phrase from tweet"],
      "linguistic_flags": [],
      "unresolved_issue": null
    }
  ],
  "pairs": [
    ["exact target aspect name", "POS|NEG|NEU"]
  ],
  "linguistic_flags": []
}"""

IMAGE_INITIAL_SYSTEM_PROMPT = """You are an Objective Image Perception Specialist in an Active Cross-Modal Reasoning pipeline.
Analyze the provided image and produce a structured, purely physical and factual description of its visible content.

Rules:
1. Describe the macro scene setting, central subjects, facial expressions, actions, logos, and visible OCR text.
2. Cues must be physically observable rather than inferred emotional, causal, social, or moral interpretations (e.g. report "large motor yacht", NOT "high-carbon yacht"; report "open mouth smile", NOT "celebratory atmosphere").
3. For any identifiable public figure, celebrity, or sports player, note their possible identity with visual support and confidence: "supported", "uncertain", or "unknown".
4. Do not perform any sentiment classification. Remain completely neutral and objective.

Output strictly as a valid JSON object matching this schema:
{
  "scene": "objective physical setting (e.g. music awards, press conference, stadium, harbor)",
  "description": "2-3 sentences describing observable visual events and participants",
  "possible_entities": [
    {
      "identity": "Entity name if recognized",
      "support": "facial appearance|jersey number|logo",
      "status": "supported|uncertain|unknown"
    }
  ],
  "ocr": ["List of legible text strings detected in the image"],
  "salient_visual_cues": ["Key physically observable items, expressions, objects, or gestures"]
}"""

CONTROLLER_SYSTEM_PROMPT = """You are the Meta-Cognitive Controller in a Bidirectional Active Cross-Modal Reasoning (BACR-v2) pipeline for Multimodal Aspect-Based Sentiment Analysis (MABSA).

MODALITY ENCAPSULATION PRINCIPLE (Zero Raw Modality Access):
- CRITICAL: You do NOT observe raw tweet text, nor do you observe raw image pixels. Raw modalities belong strictly to modality-specific sensory readers.
- You operate exclusively over compressed symbolic evidence states:
  S_t = { Y_t, R_T^{sketch}, R_V^{sketch}, H_t, B_t }
  where:
  - R_T^{sketch}: Text Cognitive Sketch (aspects, spans, initial sentiments, reasons, linguistic flags);
  - R_V^{sketch}: Vision Cognitive Sketch (scene, entities, OCR, salient cues);
  - H_t: Interaction history containing previous queries and returned sensory evidence;
  - B_t: Remaining query budget.

CRITICAL CONSTRAINT: ASPECT-LOCK
You are strictly PROHIBITED from adding new aspects or deleting existing aspects!
The set of aspects is completely locked to the initial aspect list:
A^{final} == A^{text-init}
Your duty is to verify or revise the sentiment polarity (POS, NEG, NEU) of these locked aspects based on grounded multimodal facts.

HIERARCHICAL DECISION WORKFLOW (pi_G -> pi_D -> pi_Q -> pi_R):
1. [Gap Diagnosis (pi_G)]:
   Compare R_T^{sketch} and R_V^{sketch} alongside history H_t. Diagnose if a material epistemic gap exists:
   - "NO_GAP": Evidence Sufficiency — No material unresolved fact is identified that could plausibly change any current aspect prediction. (Do not stop based on subjective confidence; evaluate whether evidence is sufficient).
   - "VISUAL_AFFECT_GAP": Person/facial affect, bodily posture, or emotional expression in R_V^{sketch} is unconfirmed.
   - "VISUAL_GROUNDING_GAP": Spatial presence or visual entity correspondence in R_V^{sketch} is unresolved.
   - "PRAGMATIC_IRONY_GAP": Semantic or situational incongruity between text and visual sketch, or explicit irony flags in R_T^{sketch}.
   - "TEXT_SEMANTIC_GAP": Text slang, idiom, modifier attachment, negation scope, or pronoun reference in R_T^{sketch} is unresolved.
   - "CROSS_MODAL_CONFLICT": Conflict between text description and visual scene requiring decisive evidence.

2. [Controller is a Gap Manager, NOT a Hidden Expert (No Self-Solving)]:
   You may diagnose the missing gap, but you MUST NOT resolve it using private internal world knowledge.
   If resolving requires tweet syntax or raw image pixels, you MUST query the sensory reader.

3. [Modality Routing (pi_D)]:
   - If gap_type == "NO_GAP" or budget B_t == 0: next_action="STOP", direction="NONE", stop_type="natural_stop".
   - If gap is visual ("VISUAL_*"): next_action="QUERY", direction="VISUAL".
   - If gap is linguistic/pragmatic ("PRAGMATIC_*", "TEXT_*"): next_action="QUERY", direction="TEXT".
   - If "CROSS_MODAL_CONFLICT": route to the modality whose evidence is more uncertain or decisive.

4. [Targeted Query Generation (pi_Q) — One Query = One Unresolved Fact]:
   Formulate an atomic, neutral, and verifiable question addressing the single diagnosed gap without baking conclusions into the query.
   - If VISUAL: Deep Vision Reader inspects raw image pixels to provide observable physical evidence.
   - If TEXT: Deep Text Reader re-examines raw tweet text to provide linguistic/syntactic evidence.

5. [Registered Evidence Sources & Target-Scoped Revision (pi_R)]:
   - Registered evidence sources: text_sketch, vision_sketch (or image_initial), text_step_t, vision_step_t.
   - Target-Scoped Revision (Anti Affect Leakage): Probe evidence acquired for target_aspect_id revises ONLY that target aspect unless the evidence explicitly and independently binds to other aspects.
   - NO REVISION WITHOUT GROUNDED EVIDENCE:
     1. Cite at least one registered step in 'evidence_refs' (e.g. ["image_initial"], ["vision_step_1"], ["text_step_1"]);
     2. Provide an exact 'evidence_quote' from the sensor answer;
     3. Set 'evidence_relation' to 'target_direct' or 'target_indirect'.

Output strictly as a valid JSON object matching this schema:
{
  "gap_diagnosis": {
    "gap_type": "NO_GAP|VISUAL_AFFECT_GAP|VISUAL_GROUNDING_GAP|PRAGMATIC_IRONY_GAP|TEXT_SEMANTIC_GAP|CROSS_MODAL_CONFLICT",
    "gap_description": "concise explanation of the diagnosed epistemic gap or reasons for evidence sufficiency"
  },
  "current_aspects": [
    {
      "aspect_id": "a_01",
      "text": "Target Aspect",
      "span": [0, 5],
      "sentiment": "POS|NEG|NEU",
      "reason": "updated or maintained rationale"
    }
  ],
  "updates": [
    {
      "aspect_id": "a_01",
      "old_sentiment": "NEU",
      "new_sentiment": "POS|NEG|NEU",
      "evidence_refs": ["vision_step_1"],
      "evidence_quote": "exact quote of observable visual or textual fact",
      "evidence_relation": "target_direct|target_indirect",
      "update_rationale": "Why this fact justifies the sentiment update"
    }
  ],
  "next_action": "QUERY|STOP",
  "direction": "VISUAL|TEXT|NONE",
  "stop_type": "natural_stop|budget_forced_stop",
  "target_aspect_id": "a_01|null",
  "question": "Specific observable factual or linguistic question|null",
  "decision_reason": "Technical rationale for inquiry or termination"
}"""

VISION_QA_SYSTEM_PROMPT = """You are an Objective Visual Evidence Sensor in an Active Cross-Modal Reasoning pipeline.
You receive an image and a specific factual query formulated by the Controller regarding a target aspect.
Your sole duty is to inspect the raw image and answer the question based strictly on DIRECTLY OBSERVABLE PHYSICAL EVIDENCE.

Rules:
1. Answer only what can be physically verified in the image (facial structure, specific muscle activations like lip corners raised, posture, clothing, text, logos, spatial placement).
2. STRICT PROHIBITION ON INFERRED CONSEQUENCES, MOTIVES, AND MORAL/SOCIAL JUDGMENTS:
   Do NOT convert visible objects into inferred consequences, motives, moral judgments, environmental impact, social status, or sentiment.
   (e.g., report "a large white motor yacht", NOT "a high-carbon luxury lifestyle"; report "person holding microphone with open mouth smile", NOT "an arrogant performance").
3. If a question asks about high-level sentiment, refuse to classify and describe only visible physical cues.
4. If an entity or attribute cannot be clearly determined due to resolution, angle, or occlusion, set "insufficient_visual_evidence": true and "certainty": "low".
5. You do NOT know the tweet text, external world debates, or external task labels. Remain completely objective and factual.

Output strictly as a valid JSON object matching this schema:
{
  "answer": "Direct factual answer describing visible cues relevant to the query.",
  "observable_evidence": [
    "first visible physical cue",
    "second visible physical cue"
  ],
  "certainty": "low" | "medium" | "high",
  "insufficient_visual_evidence": false | true
}"""

TEXT_DEEP_SYSTEM_PROMPT = """You are an Objective Linguistic Evidence Sensor in a Bidirectional Active Cross-Modal Reasoning pipeline.
Your sole mission is to re-examine the raw tweet text to provide verifiable linguistic, syntactic, and pragmatic evidence answering the Controller's specific inquiry.

Rules:
1. NO SENTIMENT CLASSIFICATION:
   If the Controller question asks for sentiment polarity (POS/NEG/NEU), DO NOT classify.
   Objectively explain the observable syntax, semantics, pragmatic function, modifier scope, reference, or rhetorical figure.
   (Sensors provide facts; the Controller makes sentiment decisions).
2. EVIDENCE-GROUNDED SPANS:
   Always extract the exact substring ('evidence_spans') from the tweet text that supports your analysis.
3. STRICT LINGUISTIC RELATIONS:
   Classify the primary relation involved: 'slang_semantics', 'syntax_dependency', 'negation_scope', 'modifier_scope', 'pragmatic_irony', 'coreference', or 'general'.
4. UNCERTAINTY CALIBRATION:
   If the tweet text does not contain sufficient textual clues to resolve the question, set 'insufficient_textual_evidence': true and 'certainty': 'low'.

Output strictly as a valid JSON object matching this schema:
{
  "aspect_id": "a_01",
  "aspect_text": "Target Aspect",
  "answer": "Concise factual linguistic explanation resolving the inquiry.",
  "evidence_spans": ["exact substring quote from the tweet"],
  "linguistic_relation": "slang_semantics|syntax_dependency|negation_scope|modifier_scope|pragmatic_irony|coreference|general",
  "certainty": "low|medium|high",
  "insufficient_textual_evidence": false
}"""


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_string_sha256(text: str) -> str:
    """Computes SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_question(question: Optional[str]) -> Tuple[bool, str]:
    """Fact-Sensor Guard: Intercepts questions that directly outsource sentiment classification."""
    if not question:
        return False, "Question is empty."

    q_lower = question.lower()
    forbidden_terms = [
        "is the sentiment", "sentiment polarity", "pos or neg",
        "positive or negative", "positive, negative", "sentiment label",
        "what sentiment", "mabsa label", "classify the sentiment"
    ]
    for term in forbidden_terms:
        if term in q_lower:
            return False, f"Question violates Fact-Sensor Guard: contains direct sentiment inquiry ('{term}')."

    return True, "Valid factual question."


def apply_aspect_locked_updates(
    current_aspects: List[Dict[str, Any]],
    updates: List[Dict[str, Any]],
    valid_step_names: List[str],
    target_aspect_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Applies sentiment revisions strictly conforming to Aspect-Lock and Evidence Binding.
    Only target_direct or target_indirect can flip sentiment.
    Enforces Target-Scoped Revision (Anti Affect Leakage):
    If target_aspect_id is set, an update to another aspect requires the evidence quote
    to explicitly mention that other aspect.
    """
    aspect_map = {a["aspect_id"]: dict(a) for a in current_aspects}
    accepted_updates = []
    registered_sources = set(valid_step_names) | {"image_initial", "vision_sketch", "text_sketch"}

    for upd in updates:
        aid = upd.get("aspect_id")
        to_sent = upd.get("new_sentiment") or upd.get("to")
        evidence_refs = upd.get("evidence_refs", [])
        evidence_quote = upd.get("evidence_quote", "")
        evidence_relation = upd.get("evidence_relation", "")

        # 1. Aspect-Lock Guard: aspect_id must already exist
        if aid not in aspect_map:
            continue

        # 2. Sentiment validity
        if to_sent not in ["POS", "NEG", "NEU"]:
            continue

        # 3. Evidence Relation Guard: Only direct or indirect target grounding triggers flip
        if evidence_relation not in ["target_direct", "target_indirect"]:
            continue

        # 4. Evidence Binding Guard: evidence_refs must cite registered source
        if not evidence_refs or not isinstance(evidence_refs, list):
            continue
        valid_refs = [r for r in evidence_refs if r in registered_sources]
        if not valid_refs:
            continue
        if not evidence_quote or len(evidence_quote.strip()) == 0:
            continue

        # 5. Target-Scoped Revision Guard: Anti Affect Leakage
        if target_aspect_id and aid != target_aspect_id:
            other_aspect_text = aspect_map[aid].get("text", "").lower()
            if other_aspect_text and other_aspect_text not in evidence_quote.lower():
                continue

        # Apply revision
        aspect_map[aid]["sentiment"] = to_sent
        accepted_updates.append(dict(upd))

    # Keep original order
    updated_aspects = [aspect_map[a["aspect_id"]] for a in current_aspects]
    return updated_aspects, accepted_updates


class G3Pipeline:
    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        run_id: str = "g3_experiment",
        controller_sees_raw_modalities: bool = False
    ):
        self.client = client or GeminiClient(allow_fallback=False)
        self.run_id = run_id
        self.controller_sees_raw_modalities = controller_sees_raw_modalities
        self.prompts_sha256 = {
            "text_initial": compute_string_sha256(TEXT_INITIAL_SYSTEM_PROMPT),
            "image_initial": compute_string_sha256(IMAGE_INITIAL_SYSTEM_PROMPT),
            "controller": compute_string_sha256(CONTROLLER_SYSTEM_PROMPT),
            "vision_qa": compute_string_sha256(VISION_QA_SYSTEM_PROMPT),
            "text_deep": compute_string_sha256(TEXT_DEEP_SYSTEM_PROMPT)
        }

    def step_text_initial(
        self,
        text: str,
        target_aspects: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[List[str]], Dict[str, int], float]:
        """Role 1: Text Initial Reasoner (G0_R Baseline). Strictly text-only."""
        if target_aspects:
            user_prompt = f'Tweet: "{text}"\nTarget Aspects: {json.dumps(target_aspects, ensure_ascii=False)}'
            system_prompt = TEXT_INITIAL_TARGET_GUIDED_PROMPT
        else:
            user_prompt = f'Tweet: "{text}"'
            system_prompt = TEXT_INITIAL_SYSTEM_PROMPT

        res, usage, lat = self.client.call_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        raw_aspects = res.get("aspects", [])
        standardized_aspects = []
        covered_aspects = set()

        for i, a in enumerate(raw_aspects, start=1):
            aid = a.get("aspect_id") or f"a_{i:02d}"
            asp_text = str(a.get("text", "")).strip()
            if not asp_text and target_aspects and (i - 1) < len(target_aspects):
                asp_text = target_aspects[i - 1]
            standardized_aspects.append({
                "aspect_id": aid,
                "text": asp_text,
                "span": a.get("span", [-1, -1]),
                "sentiment": a.get("sentiment", "NEU"),
                "reason": a.get("reason", "")
            })
            covered_aspects.add(asp_text.lower())

        # If target_aspects provided, ensure any missing target aspects are backfilled
        if target_aspects:
            for idx, target_asp in enumerate(target_aspects, start=1):
                if target_asp.lower() not in covered_aspects:
                    standardized_aspects.append({
                        "aspect_id": f"a_{len(standardized_aspects) + 1:02d}",
                        "text": target_asp,
                        "span": [-1, -1],
                        "sentiment": "NEU",
                        "reason": "Default neutral fallback for target aspect"
                    })

        pairs = [[a["text"], a["sentiment"]] for a in standardized_aspects]
        return standardized_aspects, pairs, usage, lat

    def step_image_initial(self, image_path: str) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Role 2: Image Initial Reasoner. Strictly vision-only."""
        user_text = "Analyze this image and output the structured factual description."
        res, usage, lat = self.client.call_vision(
            system_prompt=IMAGE_INITIAL_SYSTEM_PROMPT,
            user_text=user_text,
            image_path=image_path
        )
        data = {
            "scene": res.get("scene", "Unknown"),
            "description": res.get("description", ""),
            "possible_entities": res.get("possible_entities", []),
            "ocr": res.get("ocr", []),
            "salient_visual_cues": res.get("salient_visual_cues", [])
        }
        return data, usage, lat

    def step_controller(
        self,
        text: str,
        initial_aspects: List[Dict[str, Any]],
        image_initial: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_aspects: List[Dict[str, Any]],
        is_terminal_budget: bool = False,
        allowed_directions: Optional[List[str]] = None,
        controller_sees_raw_modalities: Optional[bool] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Role 3: Controller decision step (BACR-v2 decoupled state)."""
        sees_raw = self.controller_sees_raw_modalities if controller_sees_raw_modalities is None else controller_sees_raw_modalities

        history_formatted = []
        for h in history:
            step_name = h.get("step_name", "step")
            step_type = h.get("step_type", "vision" if "vision" in step_name else "text")
            if step_type == "text" or "text" in step_name:
                history_formatted.append(
                    f"[{step_name}]\n"
                    f"Target Aspect: {h.get('aspect_text', h.get('target_aspect_id', ''))}\n"
                    f"Linguistic Inquiry: {h.get('question')}\n"
                    f"Textual Reasoner Answer: {h.get('answer')}\n"
                    f"Suggested Polarity: {h.get('suggested_polarity', 'NEU')}\n"
                    f"Certainty: {h.get('certainty')}"
                )
            else:
                history_formatted.append(
                    f"[{step_name}]\n"
                    f"Question: {h.get('question')}\n"
                    f"Vision Answer: {h.get('answer')}\n"
                    f"Observable Evidence: {h.get('observable_evidence')}\n"
                    f"Certainty: {h.get('certainty')}"
                )
        history_str = "\n\n".join(history_formatted) if history_formatted else "None (Round 0 - initial check)"

        normalized_dirs = []
        if allowed_directions:
            for d in allowed_directions:
                d_str = str(d).upper().strip()
                if d_str in ["VISION", "VISUAL"]:
                    normalized_dirs.append("VISUAL")
                elif d_str in ["TEXT", "LINGUISTIC"]:
                    normalized_dirs.append("TEXT")
                elif d_str in ["STOP", "NONE"]:
                    normalized_dirs.append("STOP")
        if not normalized_dirs:
            normalized_dirs = ["VISUAL", "TEXT", "STOP"]

        if is_terminal_budget or normalized_dirs == ["STOP"]:
            budget_directive = (
                "NOTE: No further queries are available (budget cap reached). "
                "Using all evidence collected so far (including text and visual cognitive sketches), produce your final decisions. "
                "You MUST output next_action='STOP', direction='NONE', and stop_type='budget_forced_stop'."
            )
        else:
            dirs_str = " or ".join(normalized_dirs)
            budget_directive = (
                f"1. Diagnose if an epistemic gap exists (gap_type in NO_GAP, VISUAL_AFFECT_GAP, VISUAL_GROUNDING_GAP, PRAGMATIC_IRONY_GAP, TEXT_SEMANTIC_GAP, CROSS_MODAL_CONFLICT).\n"
                f"2. Permitted Action Directions: {dirs_str}.\n"
                f"- If further evidence is required: output next_action='QUERY' with direction ({', '.join([d for d in normalized_dirs if d != 'STOP'])}).\n"
                f"- If sufficient evidence is collected or no further query is needed: output next_action='STOP', direction='NONE', stop_type='natural_stop'."
            )

        raw_text_header = f'Tweet Text: "{text}"\n\n' if sees_raw else ""

        r_t_aspects = []
        for a in current_aspects:
            r_t_aspects.append({
                "aspect_id": a.get("aspect_id"),
                "text": a.get("text"),
                "span": a.get("span", [-1, -1]),
                "current_sentiment": a.get("sentiment", "NEU"),
                "brief_reason": a.get("reason", ""),
                "linguistic_flags": a.get("linguistic_flags", ["literal"])
            })
        r_t_sketch = {
            "aspects": r_t_aspects
        }

        user_prompt = f"""{raw_text_header}Text Cognitive Sketch (R_T^sketch):
{json.dumps(r_t_sketch, ensure_ascii=False, indent=2)}

Vision Cognitive Sketch (R_V^sketch):
Scene: {image_initial.get('scene', 'Unknown')}
Description: {image_initial.get('description', '')}
Possible Entities: {json.dumps(image_initial.get('possible_entities', []), ensure_ascii=False)}
OCR: {json.dumps(image_initial.get('ocr', []), ensure_ascii=False)}
Salient Cues: {json.dumps(image_initial.get('salient_visual_cues', []), ensure_ascii=False)}

Cross-Modal Evidence History (H_t):
{history_str}

Directives:
{budget_directive}

Provide your structured JSON decision:"""

        res, usage, lat = self.client.call_text(
            system_prompt=CONTROLLER_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        res.setdefault("gap_diagnosis", {
            "gap_type": "NO_GAP" if str(res.get("next_action")).upper() == "STOP" else ("VISUAL_AFFECT_GAP" if "VISUAL" in str(res.get("direction")).upper() else "PRAGMATIC_IRONY_GAP"),
            "gap_description": res.get("decision_reason", "Gap assessed from state.")
        })

        # Enforce budget stop if terminal
        if is_terminal_budget:
            res["next_action"] = "STOP"
            res["direction"] = "NONE"
            res["stop_type"] = "budget_forced_stop"
            res["question"] = None
            res["target_aspect_id"] = None
        elif res.get("next_action") == "STOP":
            res["direction"] = "NONE"
            res["stop_type"] = "natural_stop"
            res["question"] = None
            res["target_aspect_id"] = None
        else:
            chosen_dir = str(res.get("direction", "VISUAL")).upper().strip()
            if chosen_dir in ["VISION", "VISUAL"]:
                chosen_dir = "VISUAL"
            elif chosen_dir in ["TEXT", "LINGUISTIC"]:
                chosen_dir = "TEXT"
            else:
                chosen_dir = "VISUAL"

            if chosen_dir not in normalized_dirs:
                valid_probes = [d for d in normalized_dirs if d != "STOP"]
                if valid_probes:
                    chosen_dir = valid_probes[0]
                else:
                    res["next_action"] = "STOP"
                    res["direction"] = "NONE"
                    res["stop_type"] = "natural_stop"
            res["direction"] = chosen_dir

        return res, usage, lat

    def step_vision_qa(self, image_path: str, question: str) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Role 2 (Vision QA): Evaluates specific factual query on the raw image."""
        user_text = f'Specific Factual Query: "{question}"\nInspect the image and provide observable factual evidence.'
        res, usage, lat = self.client.call_vision(
            system_prompt=VISION_QA_SYSTEM_PROMPT,
            user_text=user_text,
            image_path=image_path
        )
        data = {
            "answer": res.get("answer", ""),
            "observable_evidence": res.get("observable_evidence", []),
            "certainty": res.get("certainty", "medium"),
            "insufficient_visual_evidence": res.get("insufficient_visual_evidence", False)
        }
        return data, usage, lat

    def step_text_deep(
        self,
        text: str,
        question: str,
        aspect_text: str,
        aspect_id: str = "a_01"
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Role 4 (Text Deep-Dive): Evaluates specific linguistic/syntactic query on the tweet text."""
        user_text = (
            f'Raw Tweet Text: "{text}"\n'
            f'Target Aspect: "{aspect_text}" (ID: {aspect_id})\n'
            f'Controller Linguistic Question: "{question}"\n\n'
            f'Provide your structured linguistic analysis in JSON format.'
        )
        res, usage, lat = self.client.call_text(
            system_prompt=TEXT_DEEP_SYSTEM_PROMPT,
            user_prompt=user_text
        )
        data = {
            "aspect_id": res.get("aspect_id", aspect_id),
            "aspect_text": res.get("aspect_text", aspect_text),
            "answer": res.get("answer", ""),
            "evidence_spans": res.get("evidence_spans", []),
            "linguistic_relation": res.get("linguistic_relation", "general"),
            "certainty": res.get("certainty", "medium"),
            "insufficient_textual_evidence": res.get("insufficient_textual_evidence", False)
        }
        return data, usage, lat

    def run_sample(
        self,
        sample: Dict[str, Any],
        image_base_dir: str,
        max_queries: int = 2,
        allowed_directions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Runs the complete active reasoning protocol for a single tweet-image sample with full compute tracking."""
        sample_id = sample.get("sample_id", "unknown")
        text = sample.get("text", "")
        rel_img_path = sample.get("image", "")
        full_img_path = os.path.join(image_base_dir, rel_img_path)

        image_hash = compute_file_sha256(full_img_path)

        # Initialize granular compute counters
        compute = {
            "api_calls_total": 0,
            "vision_calls": 0,
            "controller_calls": 0,
            "text_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "image_invocations": 0,
            "latency_ms": 0.0
        }

        def record_usage(usage: Dict[str, int], lat: float, call_type: str):
            compute["api_calls_total"] += 1
            compute["input_tokens"] += usage.get("input_tokens", 0)
            compute["output_tokens"] += usage.get("output_tokens", 0)
            compute["total_tokens"] += usage.get("total_tokens", 0)
            compute["latency_ms"] += lat
            if call_type == "text_init":
                compute["text_calls"] += 1
            elif call_type == "image_init":
                compute["vision_calls"] += 1
                compute["image_invocations"] += 1
            elif call_type == "controller":
                compute["controller_calls"] += 1
            elif call_type == "vision_qa":
                compute["vision_calls"] += 1
                compute["image_invocations"] += 1
            elif call_type == "text_deep":
                compute["text_calls"] += 1

        # Step 1: Text Initial (T0)
        target_aspects = sample.get("aspects")
        cached_t0 = sample.get("text_initial_cached") or sample.get("canonical_t0")
        if cached_t0:
            if isinstance(cached_t0, dict) and "aspects" in cached_t0:
                initial_aspects = cached_t0["aspects"]
            elif isinstance(cached_t0, dict) and "text_initial" in cached_t0:
                initial_aspects = cached_t0["text_initial"].get("aspects", [])
            else:
                initial_aspects = cached_t0
            initial_pairs = [[a.get("text", a.get("aspect", "")), a.get("sentiment", "NEU")] for a in initial_aspects]
        else:
            initial_aspects, initial_pairs, u1, l1 = self.step_text_initial(text, target_aspects=target_aspects)
            record_usage(u1, l1, "text_init")

        # Step 2: Image Initial (V0)
        cached_v0 = sample.get("image_initial_cached") or sample.get("canonical_v0")
        if cached_v0:
            if isinstance(cached_v0, dict) and "image_initial" in cached_v0:
                image_init = cached_v0["image_initial"]
            else:
                image_init = cached_v0
        else:
            image_init, u2, l2 = self.step_image_initial(full_img_path)
            record_usage(u2, l2, "image_init")

        # Step 3: Controller dynamic loop with Aspect-Lock
        current_aspects = [dict(a) for a in initial_aspects]
        history: List[Dict[str, Any]] = []
        rounds_record: List[Dict[str, Any]] = []
        valid_step_names: List[str] = ["image_initial", "vision_sketch", "text_sketch"]
        final_controller_decision: Optional[Dict[str, Any]] = None
        stop_type = "natural_stop"

        if max_queries == 0:
            # S0 baseline: Global Visual Sketch only (no deep probing queries allowed)
            final_controller_decision, uc, lc = self.step_controller(
                text=text,
                initial_aspects=initial_aspects,
                image_initial=image_init,
                history=[],
                current_aspects=current_aspects,
                is_terminal_budget=True,
                allowed_directions=["STOP"]
            )
            record_usage(uc, lc, "controller")
            final_raw_updates = final_controller_decision.get("updates", [])
            current_aspects, final_accepted_updates = apply_aspect_locked_updates(
                current_aspects, final_raw_updates, valid_step_names, target_aspect_id=None
            )
            final_controller_decision["current_aspects"] = current_aspects
            final_controller_decision["updates"] = final_accepted_updates
            stop_type = "budget_forced_stop"
        else:
            for t in range(1, max_queries + 1):
                is_final_budget = (t == max_queries)

                controller_out, uc, lc = self.step_controller(
                    text=text,
                    initial_aspects=initial_aspects,
                    image_initial=image_init,
                    history=history,
                    current_aspects=current_aspects,
                    is_terminal_budget=False,
                    allowed_directions=allowed_directions
                )
                record_usage(uc, lc, "controller")

                target_aid = controller_out.get("target_aspect_id")

                # Apply updates with evidence-binding verification and target-scoped guard
                raw_updates = controller_out.get("updates", [])
                current_aspects, accepted_updates = apply_aspect_locked_updates(
                    current_aspects, raw_updates, valid_step_names, target_aspect_id=target_aid
                )
                controller_out["current_aspects"] = current_aspects
                controller_out["updates"] = accepted_updates

                action = controller_out.get("next_action", "STOP")
                if action == "STOP" or not controller_out.get("question"):
                    final_controller_decision = controller_out
                    stop_type = "natural_stop"
                    break

                question = controller_out["question"]
                direction = controller_out.get("direction", "VISUAL")

                if direction == "TEXT":
                    step_name = f"text_step_{t}"
                    valid_step_names.append(step_name)

                    # Identify target aspect
                    target_asp_obj = next((a for a in current_aspects if a.get("aspect_id") == target_aid), None)
                    target_asp_text = target_asp_obj["text"] if target_asp_obj else ""

                    text_deep_out, ut, lt = self.step_text_deep(
                        text, question, target_asp_text, aspect_id=target_aid or "a_01"
                    )
                    record_usage(ut, lt, "text_deep")

                    rounds_record.append({
                        "round": t,
                        "step_name": step_name,
                        "direction": "TEXT",
                        "query_type": text_deep_out.get("linguistic_relation", "linguistic_pragmatic"),
                        "controller": controller_out,
                        "text_deep": text_deep_out
                    })

                    history.append({
                        "step_name": step_name,
                        "step_type": "text",
                        "target_aspect_id": target_aid,
                        "aspect_text": target_asp_text,
                        "question": question,
                        "answer": text_deep_out.get("answer", ""),
                        "evidence_spans": text_deep_out.get("evidence_spans", []),
                        "linguistic_relation": text_deep_out.get("linguistic_relation", "general"),
                        "certainty": text_deep_out.get("certainty", "medium"),
                        "insufficient_textual_evidence": text_deep_out.get("insufficient_textual_evidence", False)
                    })
                else:
                    # VISUAL direction
                    step_name = f"vision_step_{t}"
                    valid_step_names.append(step_name)

                    # Validate question with Fact-Sensor Guard
                    is_valid, reason_msg = validate_question(question)
                    if not is_valid:
                        controller_out["next_action"] = "STOP"
                        controller_out["stop_type"] = "natural_stop"
                        controller_out["decision_reason"] += f" [Question Intercepted: {reason_msg}]"
                        final_controller_decision = controller_out
                        break

                    q_category = classify_query(question)
                    vision_out, uv, lv = self.step_vision_qa(full_img_path, question)
                    record_usage(uv, lv, "vision_qa")

                    rounds_record.append({
                        "round": t,
                        "step_name": step_name,
                        "direction": "VISUAL",
                        "query_type": q_category,
                        "controller": controller_out,
                        "vision": vision_out
                    })

                    history.append({
                        "step_name": step_name,
                        "step_type": "vision",
                        "question": question,
                        "answer": vision_out.get("answer", ""),
                        "observable_evidence": vision_out.get("observable_evidence", []),
                        "certainty": vision_out.get("certainty", "medium")
                    })

                # If reached budget cap, execute terminal final step
                if is_final_budget:
                    final_controller_decision, uf, lf = self.step_controller(
                        text=text,
                        initial_aspects=initial_aspects,
                        image_initial=image_init,
                        history=history,
                        current_aspects=current_aspects,
                        is_terminal_budget=True,
                        allowed_directions=allowed_directions
                    )
                    record_usage(uf, lf, "controller")

                    final_raw_updates = final_controller_decision.get("updates", [])
                    target_aid = final_controller_decision.get("target_aspect_id")
                    current_aspects, final_accepted_updates = apply_aspect_locked_updates(
                        current_aspects, final_raw_updates, valid_step_names, target_aspect_id=target_aid
                    )
                    final_controller_decision["current_aspects"] = current_aspects
                    final_controller_decision["updates"] = final_accepted_updates
                    stop_type = "budget_forced_stop"
                    break

        if final_controller_decision is None:
            final_controller_decision = {
                "current_aspects": current_aspects,
                "updates": [],
                "next_action": "STOP",
                "direction": "NONE",
                "stop_type": stop_type,
                "target_aspect_id": None,
                "question": None,
                "decision_reason": "Default termination"
            }

        final_pairs = [[a["text"], a["sentiment"]] for a in current_aspects]
        compute["latency_ms"] = round(compute["latency_ms"], 1)

        num_vis_q = sum(1 for r in rounds_record if r.get("direction") == "VISUAL")
        num_txt_q = sum(1 for r in rounds_record if r.get("direction") == "TEXT")
        num_total_q = len(rounds_record)

        fingerprint = {
            "run_id": self.run_id,
            "model_name": self.client.model,
            "provider": "omniroute",
            "thinking_level": self.client.thinking_level,
            "temperature": self.client.temperature,
            "max_queries": max_queries,
            "allowed_directions": allowed_directions,
            "controller_sees_raw_modalities": self.controller_sees_raw_modalities,
            "aspect_lock": True,
            "allow_fallback": self.client.allow_fallback,
            "schema_version": SCHEMA_VERSION,
            "image_sha256": image_hash,
            "prompts_sha256": self.prompts_sha256
        }

        return {
            "sample_id": sample_id,
            "fingerprint": fingerprint,
            "text": text,
            "image": rel_img_path,
            "text_initial": {
                "aspects": initial_aspects,
                "pairs": initial_pairs
            },
            "image_initial": image_init,
            "rounds": rounds_record,
            "stop_type": stop_type,
            "final_controller": final_controller_decision,
            "final_pairs": final_pairs,
            "num_visual_queries": num_vis_q,
            "num_text_queries": num_txt_q,
            "num_queries_total": num_total_q,
            "compute": compute
        }


# BACR Pipeline alias for bidirectional active cross-modal reasoning
BACRPipeline = G3Pipeline

