"""
LLM client for AutoDL-deployed Qwen (OpenAI-compatible API).

AutoDL vLLM exposes:  POST http://<host>:<port>/v1/chat/completions
Configure LLM_CONFIG in config.py or pass overrides to QwenClient().
"""

import json
import os
import time
from typing import Any, Dict, Generator, List, Optional

import requests

# ---------------------------------------------------------------------------
# Default config – override via environment variables or config.py
# ---------------------------------------------------------------------------

LLM_CONFIG = {
    # AutoDL public URL or SSH-forwarded local port
    # e.g. "http://region-<id>.seetacloud.com:<port>" or "http://localhost:8000"
    "base_url": os.getenv("AUTODL_LLM_URL", "http://localhost:8000"),
    "api_key":  os.getenv("AUTODL_LLM_KEY", "token-abc123"),  # vLLM default
    "model":    os.getenv("AUTODL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    "temperature": 0.7,
    "max_tokens": 1024,
    "timeout": 60,
}

_SYSTEM_PROMPT = (
    "你是一个产品概念设计领域的知识图谱问答助手。"
    "请根据提供的背景知识回答用户问题，回答要简洁准确。"
    "如果背景知识中没有相关信息，请如实说明，不要编造。"
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class QwenClient:
    """
    Thin wrapper around the OpenAI-compatible chat completion endpoint.
    Works with vLLM, Ollama, and any OpenAI-API-compatible backend.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or LLM_CONFIG
        self.base_url  = cfg["base_url"].rstrip("/")
        self.api_key   = cfg["api_key"]
        self.model     = cfg["model"]
        self.temperature = cfg.get("temperature", 0.7)
        self.max_tokens  = cfg.get("max_tokens", 1024)
        self.timeout     = cfg.get("timeout", 60)

        self._endpoint = f"{self.base_url}/v1/chat/completions"
        self._headers  = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens:  Optional[int]   = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}

        Returns:
            Assistant reply string.
        """
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens":  max_tokens  if max_tokens  is not None else self.max_tokens,
        }
        resp = requests.post(
            self._endpoint,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def simple_query(self, user_text: str, system: Optional[str] = None) -> str:
        """Single-turn query with optional system prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        return self.chat(messages)

    def health_check(self) -> bool:
        """Verify the endpoint is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers,
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False
