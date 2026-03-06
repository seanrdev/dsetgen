"""Concrete LLM adapter for a local Ollama instance.

Ollama exposes a REST API at ``/api/generate`` (completion) and
``/api/chat`` (chat).  This adapter uses ``httpx.AsyncClient`` for
non-blocking I/O so that multiple chunks can be processed concurrently
via ``asyncio.Semaphore`` in the pipeline controller.

Error-Handling Strategy
-----------------------
* **Timeouts** → ``httpx.TimeoutException`` → ``LLMTimeoutError``
* **Connection refused / DNS** → ``httpx.ConnectError`` → ``LLMConnectionError``
* **HTTP 429** → ``LLMRateLimitError`` (caller implements backoff)
* **Malformed JSON from LLM** → ``LLMMalformedResponseError``
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

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


@register_adapter("ollama")
class OllamaAdapter(AbstractLLMAdapter):
    """Async adapter for a locally-running Ollama server.

    Parameters
    ----------
    config:
        Reads ``ollama_base_url``, ``model_name``, ``llm_timeout_seconds``.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._base_url = config.ollama_base_url.rstrip("/")
        self._model = config.model_name
        self._timeout = config.llm_timeout_seconds
        self._client = None  # Lazily created in startup()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Create the shared ``httpx.AsyncClient``."""
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )
        logger.info(
            "OllamaAdapter initialised — %s model=%s", self._base_url, self._model
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
        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        data = await self._post("/api/generate", payload)
        return data.get("response", "")

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
        try:
            resp = await self._ensure_client().get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    # ── Internal ───────────────────────────────────────────────────────────

    def _ensure_client(self):
        if self._client is None:
            raise LLMConnectionError(
                "OllamaAdapter.startup() has not been called"
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
                f"Ollama did not respond within {self._timeout}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self._base_url}: {exc}"
            ) from exc

        if resp.status_code == 429:
            raise LLMRateLimitError("Ollama returned HTTP 429 — back off")

        if resp.status_code >= 400:
            raise LLMConnectionError(
                f"Ollama returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )

        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMMalformedResponseError(
                f"Ollama returned non-JSON body: {resp.text[:200]}"
            ) from exc

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from potentially noisy LLM output.

        LLMs often wrap JSON in markdown fences or prepend conversational
        text.  We try, in order:
        1. Direct ``json.loads``.
        2. Regex extraction of the first ``{…}`` block.
        3. Raise ``LLMMalformedResponseError``.
        """
        # Strip markdown fences
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
