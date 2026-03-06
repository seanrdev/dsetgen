"""Strategy-pattern registry for ingestors, adapters, and formatters.

Concrete implementations register themselves at import time so that the
pipeline can look up the right handler by file extension or adapter name
without hard-coded ``if``/``elif`` chains.

Usage
-----
>>> from doc2dataset.core.registry import ingestor_registry
>>> handler_cls = ingestor_registry.get(".pdf")
>>> handler = handler_cls(config)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class _Registry:
    """A simple key → class mapping with duplicate-key warnings."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._store: Dict[str, Type[Any]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def register(self, key: str, cls: Type[Any]) -> None:
        """Bind *key* (e.g. ``".pdf"``) to *cls*.

        If *key* is already registered, the new binding wins and a warning
        is emitted — this lets downstream code override built-in handlers.
        """
        normalised = key.lower()
        if normalised in self._store:
            logger.warning(
                "%s: overriding %s → %s (was %s)",
                self._name,
                normalised,
                cls.__qualname__,
                self._store[normalised].__qualname__,
            )
        self._store[normalised] = cls

    def get(self, key: str) -> Optional[Type[Any]]:
        """Return the class bound to *key*, or ``None``."""
        return self._store.get(key.lower())

    def keys(self) -> list[str]:
        """Return all registered keys (sorted for deterministic logs)."""
        return sorted(self._store)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._store

    def __repr__(self) -> str:  # pragma: no cover
        entries = ", ".join(f"{k}={v.__qualname__}" for k, v in sorted(self._store.items()))
        return f"<{self._name} [{entries}]>"


def _register_decorator(registry: _Registry, key: str):
    """Decorator factory: ``@register_ingestor(".pdf")``."""
    def decorator(cls: Type[Any]) -> Type[Any]:
        registry.register(key, cls)
        return cls
    return decorator


# ── Singletons ─────────────────────────────────────────────────────────────

ingestor_registry = _Registry("IngestorRegistry")
adapter_registry = _Registry("AdapterRegistry")
formatter_registry = _Registry("FormatterRegistry")


def register_ingestor(extension: str):
    """Class decorator that registers an ingestor for *extension*."""
    return _register_decorator(ingestor_registry, extension)


def register_adapter(name: str):
    """Class decorator that registers an LLM adapter under *name*."""
    return _register_decorator(adapter_registry, name)


def register_formatter(name: str):
    """Class decorator that registers an output formatter under *name*."""
    return _register_decorator(formatter_registry, name)
