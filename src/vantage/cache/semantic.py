"""Exact and semantic response caches, and the backend wrapper that uses them.

A semantic cache is the highest-leverage optimisation available to a RAG
system, because real query traffic is extremely repetitive: the same question
arrives in twenty phrasings. It is also the easiest way to quietly destroy
answer quality, because a cache hit at too low a threshold returns a confident
answer to a question nobody asked.

So this module measures both sides. `hit_rate` is the number everyone reports.
`false_hit_rate` -- how often a hit returned an answer that does not match the
reference for the new question -- is the number that makes the first one
meaningful, and the threshold sweep in analysis.py is the deliverable.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vantage.backends.base import Backend, Health, Pricing, StreamChunk
from vantage.cache.embedders import Embedder, HashingEmbedder


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class CacheEntry:
    key: str
    value: str
    created_at: float
    meta: dict[str, Any] = field(default_factory=dict)
    hits: int = 0


@dataclass
class Lookup:
    hit: bool
    kind: str = "miss"  # "exact" | "semantic" | "miss"
    entry: CacheEntry | None = None
    similarity: float | None = None


class ExactCache:
    """Normalised-string LRU with TTL. Cheap, zero risk of a wrong answer, and
    on real traffic it catches a surprising share of hits on its own."""

    def __init__(self, capacity: int = 10_000, ttl_s: float | None = None):
        self.capacity = capacity
        self.ttl_s = ttl_s
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _expired(self, entry: CacheEntry) -> bool:
        return self.ttl_s is not None and (time.time() - entry.created_at) > self.ttl_s

    def lookup(self, text: str) -> Lookup:
        key = normalize(text)
        entry = self._store.get(key)
        if entry is None or self._expired(entry):
            if entry is not None:
                del self._store[key]
            self.misses += 1
            return Lookup(hit=False)
        self._store.move_to_end(key)
        entry.hits += 1
        self.hits += 1
        return Lookup(hit=True, kind="exact", entry=entry, similarity=1.0)

    def put(self, text: str, value: str, meta: dict[str, Any] | None = None) -> None:
        key = normalize(text)
        self._store[key] = CacheEntry(key=key, value=value, created_at=time.time(), meta=meta or {})
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "kind": "exact",
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


class SemanticCache:
    """Embedding cache with a cosine-similarity threshold.

    Brute-force numpy search: at the sizes a single-service cache reaches
    (tens of thousands of entries) a matrix-vector product is microseconds and
    an ANN index is unnecessary complexity. The interface is the same one FAISS
    or Qdrant would expose, so swapping it in later is a local change.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = 0.92,
        capacity: int = 10_000,
        ttl_s: float | None = None,
        exact_first: bool = True,
    ):
        self.embedder = embedder or HashingEmbedder()
        self.threshold = threshold
        self.capacity = capacity
        self.ttl_s = ttl_s
        self.exact = ExactCache(capacity=capacity, ttl_s=ttl_s) if exact_first else None

        self._entries: list[CacheEntry] = []
        self._matrix: np.ndarray | None = None
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0
        self.exact_hits = 0
        self.similarities: list[float] = []

    # -- internals ---------------------------------------------------------- #

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.embed([text])[0]

    def _rebuild(self) -> None:
        if not self._entries:
            self._matrix = None
            return
        self._matrix = np.vstack([e.meta["_vec"] for e in self._entries])

    def _drop_expired(self) -> None:
        if self.ttl_s is None:
            return
        now = time.time()
        keep = [e for e in self._entries if (now - e.created_at) <= self.ttl_s]
        if len(keep) != len(self._entries):
            self._entries = keep
            self._rebuild()

    # -- api ---------------------------------------------------------------- #

    def lookup(self, text: str) -> Lookup:
        if self.exact is not None:
            hit = self.exact.lookup(text)
            if hit.hit:
                self.hits += 1
                self.exact_hits += 1
                return hit

        self._drop_expired()
        if self._matrix is None or not self._entries:
            self.misses += 1
            return Lookup(hit=False)

        vec = self._embed(text)
        sims = self._matrix @ vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        self.similarities.append(score)

        if score >= self.threshold:
            entry = self._entries[best]
            entry.hits += 1
            self.hits += 1
            self.semantic_hits += 1
            return Lookup(hit=True, kind="semantic", entry=entry, similarity=score)

        self.misses += 1
        return Lookup(hit=False, similarity=score)

    def put(self, text: str, value: str, meta: dict[str, Any] | None = None) -> None:
        if self.exact is not None:
            self.exact.put(text, value, meta)
        entry = CacheEntry(
            key=normalize(text),
            value=value,
            created_at=time.time(),
            meta={**(meta or {}), "_vec": self._embed(text)},
        )
        self._entries.append(entry)
        while len(self._entries) > self.capacity:
            self._entries.pop(0)
        self._rebuild()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "kind": "semantic",
            "threshold": self.threshold,
            "embedder": getattr(self.embedder, "name", "unknown"),
            "size": len(self._entries),
            "hits": self.hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "mean_best_similarity": (
                float(np.mean(self.similarities)) if self.similarities else None
            ),
        }


class CachedBackend(Backend):
    """Wraps any backend with a cache, without the backend knowing.

    On a hit the cached text is returned as a single chunk, so TTFT collapses
    to the lookup time -- which is the entire point, and also the reason the
    cache must be measured on quality as well as speed.
    """

    kind = "cached"

    def __init__(self, inner: Backend, cache: SemanticCache | ExactCache, name: str | None = None):
        super().__init__(name=name or f"{inner.name}+cache", config_id=inner.config_id)
        self.inner = inner
        self.cache = cache
        self.last_lookup: Lookup | None = None

    async def _stream(self, req) -> AsyncIterator[StreamChunk]:
        key = req.meta.get("question") or req.prompt
        lookup = self.cache.lookup(key)
        self.last_lookup = lookup
        if lookup.hit and lookup.entry is not None:
            text = lookup.entry.value
            yield StreamChunk(
                text=text,
                prompt_tokens=lookup.entry.meta.get("prompt_tokens", 0),
                completion_tokens=lookup.entry.meta.get("completion_tokens", max(1, len(text.split()))),
                finish_reason="cache",
            )
            return

        pieces: list[str] = []
        prompt_tokens = completion_tokens = 0
        async for chunk in self.inner._stream(req):
            if chunk.text:
                pieces.append(chunk.text)
            if chunk.prompt_tokens is not None:
                prompt_tokens = chunk.prompt_tokens
            if chunk.completion_tokens is not None:
                completion_tokens = chunk.completion_tokens
            yield chunk
        self.cache.put(
            key,
            "".join(pieces),
            meta={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )

    async def generate(self, req, arrival_ts: float | None = None):
        result = await super().generate(req, arrival_ts=arrival_ts)
        lookup = self.last_lookup
        if lookup is not None and lookup.hit:
            result.cache_hit = True
            result.cache_kind = lookup.kind  # type: ignore[assignment]
            result.cache_similarity = lookup.similarity
        return result

    async def health(self) -> Health:
        return await self.inner.health()

    def pricing(self) -> Pricing:
        return self.inner.pricing()

    async def aclose(self) -> None:
        await self.inner.aclose()
