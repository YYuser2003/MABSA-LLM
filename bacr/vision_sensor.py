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
        """I0: Analyzes raw image pixels to produce the Global Visual Sketch V0."""
        user_prompt = "Examine the provided image and generate the structured Global Visual Sketch in valid JSON."
        res, usage, lat = self.client.call_vision(
            system_prompt=self.prompt_global,
            user_text=user_prompt,
            image_path=image_path
        )
        data = {
            "scene": res.get("scene", "Unknown scene"),
            "description": res.get("description", ""),
            "possible_entities": res.get("possible_entities", []),
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
        data = {
            "answer": res.get("answer", ""),
            "observable_evidence": res.get("observable_evidence", []),
            "certainty": res.get("certainty", "medium"),
            "insufficient_visual_evidence": res.get("insufficient_visual_evidence", False)
        }
        return data, usage, lat
