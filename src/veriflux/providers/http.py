"""Generic OpenAI-compatible HTTP inference provider.

Communicates with any compliant endpoint (vLLM, Ollama, LM Studio, OpenAI, Groq)
using only Python standard library urllib.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from veriflux._types import TokenUsage
from veriflux.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from veriflux.providers.base import Provider, ProviderResponse


class HTTPProvider(Provider):
    """OpenAI-compatible HTTP provider built on stdlib urllib."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return f"http:{self.model}"

    def _request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "veriflux/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_bytes = resp.read()
                    return json.loads(resp_bytes.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 401 or exc.code == 403:
                    raise ProviderAuthError(f"Authentication failed: {exc}", provider=self.name, status_code=exc.code) from exc
                if exc.code == 429:
                    if attempt < self.max_retries:
                        time.sleep(1.0 * (2 ** attempt))
                        continue
                    raise ProviderRateLimitError("Rate limit exceeded", provider=self.name, status_code=429) from exc
                if attempt == self.max_retries:
                    raise ProviderError(f"HTTP request failed: {exc}", provider=self.name, status_code=exc.code) from exc
                time.sleep(0.5 * (2 ** attempt))
            except urllib.error.URLError as exc:
                if "timed out" in str(exc).lower() and attempt == self.max_retries:
                    raise ProviderTimeoutError(f"Request timed out: {exc}", provider=self.name) from exc
                if attempt == self.max_retries:
                    raise ProviderError(f"Network error: {exc}", provider=self.name) from exc
                time.sleep(0.5 * (2 ** attempt))

        raise ProviderError("Exhausted retries", provider=self.name)

    def complete(self, messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ProviderResponse:
        """Call /chat/completions endpoint."""
        start_t = time.perf_counter()
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": list(messages),
            **{k: v for k, v in kwargs.items() if k != "model"},
        }
        data = self._request("chat/completions", payload)
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        choices = data.get("choices", [{}])
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {})
        content = message.get("content", "")
        finish_reason = first_choice.get("finish_reason", "stop")

        usage = data.get("usage", {})
        in_toks = usage.get("prompt_tokens", 0)
        out_toks = usage.get("completion_tokens", 0)
        tot_toks = usage.get("total_tokens", in_toks + out_toks)

        return ProviderResponse(
            content=content,
            model=data.get("model", self.model),
            tokens=TokenUsage(input_tokens=in_toks, output_tokens=out_toks, total_tokens=tot_toks),
            latency_ms=round(latency_ms, 2),
            finish_reason=finish_reason,
            raw_response=data,
        )

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Call /embeddings endpoint."""
        payload = {
            "model": "text-embedding-3-small",
            "input": list(texts),
        }
        data = self._request("embeddings", payload)
        items = data.get("data", [])
        return [item.get("embedding", []) for item in sorted(items, key=lambda x: x.get("index", 0))]

