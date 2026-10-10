"""Embedders for the semantic cache.

The default is a dependency-free hashing embedder. It is not as good as a
trained sentence encoder, and that is deliberate: it means the whole cache
analysis runs offline, in CI, with no API key and no 400MB model download, and
the swap to a real encoder is one config line. Where the choice of embedder
changes a conclusion, the report says which one produced it.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int
    remote: bool

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Feature hashing over word unigrams, bigrams and character 4-grams.

    Near-duplicate questions ("what is the reference range for 25(OH)D?" vs
    "what's the reference range of 25(OH)D") land close together; unrelated
    questions do not. Cosine similarity on L2-normalised vectors, so the
    threshold semantics match any real encoder you swap in.
    """

    remote = False

    def __init__(self, dim: int = 512, name: str = "hashing"):
        self.dim = dim
        self.name = name

    def _features(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        joined = " ".join(words)
        feats += [joined[i : i + 4] for i in range(0, max(0, len(joined) - 3))]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class OpenAIEmbedder:  # pragma: no cover - requires network
    remote = True

    def __init__(self, model: str = "text-embedding-3-small", base_url: str | None = None):
        import httpx

        self.name = f"openai:{model}"
        self.model = model
        self.dim = 1536
        self._client = httpx.Client(
            base_url=(base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
            timeout=60.0,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self._client.post("/embeddings", json={"model": self.model, "input": texts})
        r.raise_for_status()
        vecs = np.asarray([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
        self.dim = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformerEmbedder:  # pragma: no cover - heavy optional dependency
    remote = False

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model}"
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


def build_embedder(kind: str | None = None, model: str | None = None) -> Embedder:
    kind = (kind or os.getenv("VANTAGE_EMBEDDER", "hashing")).lower()
    model = model or os.getenv("VANTAGE_EMBEDDING_MODEL", "text-embedding-3-small")
    if kind in {"hashing", "hash", "offline"}:
        return HashingEmbedder()
    if kind == "openai":
        return OpenAIEmbedder(model=model)
    if kind in {"sentence-transformers", "st", "local"}:
        return SentenceTransformerEmbedder(model=model)
    raise ValueError(f"unknown embedder '{kind}'")
