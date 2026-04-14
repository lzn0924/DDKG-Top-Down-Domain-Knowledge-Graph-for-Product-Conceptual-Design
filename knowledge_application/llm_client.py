"""
HTTP client for AutoDL-deployed Qwen via OpenAI-compatible API.

Supports vLLM, LMDeploy, and Ollama serving Qwen.
Configure LLM_CONFIG in config.py or set environment variables:
  AUTODL_LLM_URL   – e.g. https://region-3.autodl.pro:12345
  AUTODL_LLM_MODEL – e.g. Qwen/Qwen2.5-7B-Instruct
  AUTODL_API_KEY   – API key (use "EMPTY" for unauthenticated deployments)
"""

import json
from typing import Dict, Iterator, List, Optional

import requests

from config import LLM_CONFIG


class QwenClient:
    """
    Thin wrapper around /v1/chat/completions (OpenAI-compatible).

    Usage::
        client = QwenClient()
        reply = client.chat([{"role": "user", "content": "你好"}])
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or LLM_CONFIG
        self.base_url  = cfg["base_url"].rstrip("/")
        self.model     = cfg["model"]
        self._headers  = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {cfg.get('api_key', 'EMPTY')}",
        }
        self._timeout      = cfg.get("timeout", 60)
        self._temperature  = cfg.get("temperature", 0.3)
        self._max_tokens   = cfg.get("max_tokens", 2048)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Send a request and return the full reply string."""
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens":  max_tokens  if max_tokens  is not None else self._max_tokens,
            **kwargs,
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield reply tokens as they arrive (server-sent events)."""
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens":  max_tokens  if max_tokens  is not None else self._max_tokens,
            "stream":      True,
        }
        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=self._timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                chunk_str = line[6:].strip()
                if chunk_str == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk_str)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError):
                    continue

    def health_check(self) -> bool:
        """Return True if the endpoint is reachable."""
        try:
            r = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers,
                timeout=5,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False
