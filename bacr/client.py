"""Unified Inference Client Module for BACR.

Supports:
- GeminiClient (OmniRoute / Google Gemini API)
- QwenClient (Local HuggingFace / vLLM Qwen3-VL-8B)
- Strict Model Lock & Retry with Exponential Backoff
"""

import os
import sys
import json
import time
import random
import base64
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import io
import requests

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "sk-8155e3017561267e-0a40fe-db0a138d")
DEFAULT_BASE_URL = os.environ.get("GEMINI_BASE_URL", "http://10.60.80.119:20128")
PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini_3.8")
FALLBACK_MODEL = "antigravity/gemini-3.8-flash-tiered"


def clean_json_response(content: str) -> str:
    """Strips markdown code blocks, backticks, and whitespace, extracting outer JSON braces."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Locate first { and last }
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        content = content[first_brace : last_brace + 1]
    return content.strip()


class BaseClient(ABC):
    """Abstract Base Class for LLM / MLLM Inference Clients."""

    @abstractmethod
    def call_text(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        user_text: Optional[str] = None,
        max_retries: int = 8
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        pass

    @abstractmethod
    def call_vision(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        image_path: str = "",
        user_text: Optional[str] = None,
        max_retries: int = 8
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        pass


class GeminiClient(BaseClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        thinking_level: str = "high",
        temperature: float = 0.1,
        allow_fallback: bool = False,
        timeout: int = 90
    ):
        self.api_key = api_key or DEFAULT_API_KEY
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or PRIMARY_MODEL
        self.thinking_level = thinking_level
        self.temperature = temperature
        self.allow_fallback = allow_fallback
        self.fallback_model = FALLBACK_MODEL
        self.timeout = timeout
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Reads local image and converts to base64 JPEG data URL."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"

    def call_text(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        user_text: Optional[str] = None,
        max_retries: int = 8
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Executes text-only generation with JSON output constraint."""
        prompt = user_prompt if user_prompt is not None else (user_text or "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self._call_api_with_retry(messages, max_retries=max_retries)

    def call_vision(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        image_path: str = "",
        user_text: Optional[str] = None,
        max_retries: int = 8
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """Executes multimodal generation with image and text prompt."""
        prompt = user_prompt if user_prompt is not None else (user_text or "")
        image_data_url = self.encode_image(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    }
                ]
            }
        ]
        return self._call_api_with_retry(messages, max_retries=max_retries)

    def _call_api_with_retry(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 8
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        target_model = self.model
        attempt = 0
        backoff = 2.0

        while attempt < max_retries:
            attempt += 1
            payload = {
                "model": target_model,
                "messages": messages,
                "temperature": self.temperature,
                "thinking_level": self.thinking_level,
                "response_format": {"type": "json_object"}
            }

            t0 = time.time()
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                latency_ms = (time.time() - t0) * 1000

                if resp.status_code == 200:
                    resp_json = resp.json()
                    choice = resp_json.get("choices", [{}])[0]
                    raw_content = choice.get("message", {}).get("content", "")
                    cleaned_content = clean_json_response(raw_content)
                    parsed = json.loads(cleaned_content)

                    usage = resp_json.get("usage", {})
                    metrics = {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                    return parsed, metrics, latency_ms

                elif resp.status_code in [429, 500, 502, 503, 504]:
                    sleep_time = backoff + random.uniform(0.5, 2.0)
                    time.sleep(sleep_time)
                    backoff = min(backoff * 1.6, 60.0)
                else:
                    sleep_time = backoff + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                    backoff = min(backoff * 1.5, 45.0)

            except (requests.RequestException, json.JSONDecodeError):
                sleep_time = backoff + random.uniform(0.5, 2.0)
                time.sleep(sleep_time)
                backoff = min(backoff * 1.6, 60.0)

        raise RuntimeError(f"Exhausted {max_retries} retries for model {target_model}")


class QwenClient(BaseClient):
    """Stub / Interface for Local Qwen3-VL-8B (HuggingFace / vLLM)."""

    def __init__(
        self,
        model_path: str = "/data2/models/Qwen3-VL-8B-Instruct",
        device: str = "cuda:0",
        temperature: float = 0.1
    ):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self._model = None
        self._processor = None

    def call_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        raise NotImplementedError("Local QwenClient text inference is active during Stage C SFT.")

    def call_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        max_retries: int = 3
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        raise NotImplementedError("Local QwenClient vision inference is active during Stage C SFT.")
