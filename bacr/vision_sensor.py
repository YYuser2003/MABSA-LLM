"""Objective Visual Sensor (I) for BACR-v3.

Provides purely physically observable visual evidence:
- I0: Global Visual Perception (scene, participants, OCR, salient cues)
- ID: Targeted Deep Visual Probe (factual query verification on raw pixels)
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

from bacr.client import BaseClient

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts", "v3")


def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class VisionSensor:
    def __init__(self, client: BaseClient):
        self.client = client
        try:
            self.prompt_global = load_prompt("vision_global.md")
        except Exception:
            self.prompt_global = load_prompt("vision_initial.md")
            
        try:
            self.prompt_deep = load_prompt("vision_probe.md")
        except Exception:
            self.prompt_deep = load_prompt("vision_deep.md")

    def perceive_global(
        self,
        image_path: str
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """I0: Analyzes raw image pixels to produce the Global Visual Opportunity Map V0."""
        user_prompt = "Examine the provided image and generate the structured Global Visual Opportunity Map in valid JSON."
        res, usage, lat = self.client.call_vision(
            system_prompt=self.prompt_global,
            user_text=user_prompt,
            image_path=image_path
        )
        data = {
            "scene": res.get("scene", "Unknown scene"),
            "description": res.get("description", ""),
            "observable_entities": res.get("observable_entities", res.get("possible_entities", [])),
            "possible_entities": res.get("possible_entities", res.get("observable_entities", [])),
            "visual_information_map": res.get("visual_information_map", {
                "person_identity_available": True,
                "facial_expression_available": True,
                "object_state_available": False,
                "ocr_available": bool(res.get("ocr")),
                "relationship_available": True
            }),
            "aspect_relevance": res.get("aspect_relevance", {}),
            "limitations": res.get("limitations", ["cannot infer sentiment"]),
            "ocr": res.get("ocr", []),
            "salient_visual_cues": res.get("salient_visual_cues", [])
        }
        return data, usage, lat

    def probe_deep(
        self,
        image_path: str,
        question: str
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """ID: Answers a targeted factual inquiry using raw image pixels without sentiment speculation."""
        user_prompt = (
            f'Targeted Inquiry: "{question}"\n'
            f'Inspect the raw image pixels and describe ONLY directly observable physical facts in valid JSON.'
        )
        res, usage, lat = self.client.call_vision(
            system_prompt=self.prompt_deep,
            user_text=user_prompt,
            image_path=image_path
        )

        raw_obs = res.get("observations", [])
        raw_evidence = res.get("observable_evidence", [])
        if not raw_evidence and raw_obs and isinstance(raw_obs, list):
            raw_evidence = [o.get("fact", "") for o in raw_obs if isinstance(o, dict) and o.get("fact")]

        ans = res.get("answer", "")
        if not ans and raw_evidence:
            ans = "; ".join(raw_evidence)

        data = {
            "question": question,
            "answer": ans,
            "observable_evidence": raw_evidence,
            "observations": raw_obs,
            "unsupported_claims": res.get("unsupported_claims", []),
            "certainty": res.get("certainty", "medium"),
            "insufficient_visual_evidence": res.get("insufficient_visual_evidence", False)
        }
        return data, usage, lat
