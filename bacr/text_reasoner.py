"""Deliberative Text Reasoner (T) for BACR-v3 Minimal Teacher.

First-order linguistic, syntactic, and pragmatic reasoner:
- TR: Text Re-deliberation from Controller critique (strictly without image access).
- TF: Evidence Fusion Reasoner incorporating verified visual evidence from the Firewall.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

from bacr.client import BaseClient
from bacr.schemas_v3 import CandidatePrediction

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts", "v3")


def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class TextReasoner:
    def __init__(self, client: BaseClient):
        self.client = client
        self.prompt_tr = load_prompt("text_rethink.md")
        self.prompt_tf = load_prompt("text_evidence_fusion.md")

    def rethink_text(
        self,
        text: str,
        aspect: str,
        anchor: Dict[str, Any],
        critique: str
    ) -> Tuple[CandidatePrediction, Dict[str, int], float]:
        """TR: Text Re-deliberation based on Controller critique (WITHOUT IMAGE ACCESS).
        Strictly linguistic domain: returns CandidatePrediction.
        """
        anchor_sent = anchor.get("sentiment", "NEU")
        anchor_reason = anchor.get("reason", anchor.get("rationale", ""))

        user_prompt = (
            f'Raw Tweet Text: "{text}"\n\n'
            f'Target Aspect: "{aspect}"\n\n'
            f'Initial Text Baseline (T0):\n'
            f'- Sentiment: {anchor_sent}\n'
            f'- Reason: {anchor_reason}\n\n'
            f'Meta-Controller Linguistic Critique:\n"{critique}"\n\n'
            f'Re-deliberate the aspect sentiment addressing the critique and output valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_tr,
            user_prompt=user_prompt
        )

        candidate = CandidatePrediction.validate_or_fallback(
            res,
            default_aspect=aspect,
            default_sentiment=anchor_sent
        )
        return candidate, usage, lat

    def fuse_evidence(
        self,
        text: str,
        aspect: str,
        anchor: Dict[str, Any],
        verified_evidence: Dict[str, Any]
    ) -> Tuple[CandidatePrediction, Dict[str, int], float]:
        """TF: Evidence Fusion Reasoner incorporating verified visual evidence from Firewall.
        Strictly evidence-grounded: returns CandidatePrediction.
        """
        anchor_sent = anchor.get("sentiment", "NEU")
        anchor_reason = anchor.get("reason", anchor.get("rationale", ""))
        usable = verified_evidence.get("usable_evidence", [])

        user_prompt = (
            f'Raw Tweet Text: "{text}"\n\n'
            f'Target Aspect: "{aspect}"\n\n'
            f'Initial Text Baseline (T0):\n'
            f'- Sentiment: {anchor_sent}\n'
            f'- Reason: {anchor_reason}\n\n'
            f'Verified Visual Proof from Firewall:\n'
            f'{json.dumps(usable, ensure_ascii=False, indent=2)}\n\n'
            f'Synthesize the tweet semantics with verified visual proof and output valid JSON.'
        )

        res, usage, lat = self.client.call_text(
            system_prompt=self.prompt_tf,
            user_prompt=user_prompt
        )

        candidate = CandidatePrediction.validate_or_fallback(
            res,
            default_aspect=aspect,
            default_sentiment=anchor_sent
        )
        return candidate, usage, lat


