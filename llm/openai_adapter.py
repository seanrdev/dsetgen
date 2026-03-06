"""
OpenAI-compatible LLM adapter.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from dsetgen.config import PipelineConfig
from dsetgen.core.registry import register_adapter
from dsetgen.exceptions import (
    LLMConnectionError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from dsetgen.llm.abstract_adapter import AbstractLLMAdapter

logger = logging.getLogger(__name__)

# Regex to extract the first JSON object from noisy LLM output.
_JSON_EXTRACT_RE = re.compile(r"\{.*\}", re.DOTALL)


@register_adapter("openai")
class OpenAICompatibleAdapter(AbstractLLMAdapter):
    """Async adapter for any server implementing the OpenAI chat completions API.

    Parameters
    ----------
    config:
        Reads ``ollama_base_url`` (treated as the base URL, e.g.
        ``http://10.250.250.163:8080/v1``), ``model_name``,
        ``llm_timeout_seconds``.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._base_url = config.ollama_base_url.rstrip("/")
        self._model = config.model_name
        self._timeout = config.llm_timeout_seconds
        self._client = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def startup(self) -> None:
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )
        logger.info(
            "OpenAICompatibleAdapter initialised — %s model=%s",
            self._base_url,
            self._model,
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        data = await self._post("/chat/completions", payload)

        # OpenAI response format: data.choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMMalformedResponseError(
                f"Unexpected response structure: {json.dumps(data)[:300]}"
            ) from exc

    async def generate_structured(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        raw = await self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_json(raw)

    async def health_check(self) -> bool:
        """Probe the server with GET /models (standard OpenAI endpoint)."""
        try:
            client = self._ensure_client()
            resp = await client.get("/models")
            if resp.status_code == 200:
                return True
            # Some servers don't implement /models — try a tiny completion.
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── Internal ───────────────────────────────────────────────────────────

    def _ensure_client(self):
        if self._client is None:
            raise LLMConnectionError(
                "OpenAICompatibleAdapter.startup() has not been called"
            )
        return self._client

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST with structured exception mapping."""
        import httpx

        client = self._ensure_client()

        try:
            resp = await client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM server did not respond within {self._timeout}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Cannot reach LLM server at {self._base_url}: {exc}"
            ) from exc

        if resp.status_code == 429:
            raise LLMRateLimitError("LLM server returned HTTP 429 — back off")

        if resp.status_code >= 400:
            raise LLMConnectionError(
                f"LLM server returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )

        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMMalformedResponseError(
                f"LLM server returned non-JSON body: {resp.text[:200]}"
            ) from exc

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from potentially noisy LLM output."""
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = _JSON_EXTRACT_RE.search(cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise LLMMalformedResponseError(
            f"Could not extract valid JSON from LLM output: {raw[:300]!r}"
        )
