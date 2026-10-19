"""The workload: what we actually send at the server.

A serving benchmark is only as meaningful as its prompts. Benchmarking with
50-token prompts and 20-token answers tells you nothing about a RAG system,
because in RAG the prompt is enormous (retrieved context) and the answer is
short. Prefill cost dominates, prefix caching matters, and the throughput
numbers come out completely different. So the workload here is RAG-shaped by
construction and its token-length distribution is reported alongside results.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vantage.errors import ConfigError
from vantage.models import GenRequest

DEFAULT_SYSTEM = (
    "You answer questions using only the numbered sources provided. "
    "Cite every claim with the source number in square brackets, like [2]. "
    "If the sources do not contain the answer, say so plainly. Be concise."
)


@dataclass
class WorkloadItem:
    id: str
    question: str
    context: str
    reference_answer: str = ""
    reference_citations: list[int] = field(default_factory=list)
    source_ids: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    #: The exact user message the application sends, when the workload was
    #: exported from a real app. Takes precedence over the generic template,
    #: so the benchmark measures the prompt that actually runs in production.
    prompt_text: str = ""

    def prompt(self) -> str:
        if self.prompt_text:
            return self.prompt_text
        return f"Sources:\n{self.context}\n\nQuestion: {self.question}"


@dataclass
class Workload:
    items: list[WorkloadItem]
    system: str = DEFAULT_SYSTEM
    name: str = "workload"

    @classmethod
    def from_jsonl(cls, path: str | Path, system: str | None = None) -> Workload:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"workload file not found: {p}")
        items: list[WorkloadItem] = []
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{p}:{ln} is not valid JSON ({exc})") from exc
            missing = {"id", "question", "context"} - row.keys()
            if missing:
                raise ConfigError(f"{p}:{ln} is missing required field(s): {sorted(missing)}")
            items.append(
                WorkloadItem(
                    id=str(row["id"]),
                    question=row["question"],
                    context=row["context"],
                    reference_answer=row.get("reference_answer", ""),
                    reference_citations=row.get("reference_citations", []),
                    source_ids=row.get("source_ids", []),
                    meta=row.get("meta", {}),
                    prompt_text=row.get("prompt", ""),
                )
            )
        if not items:
            raise ConfigError(f"{p} contains no usable rows")
        # A workload exported from an application can ship that application's
        # system prompt beside it as `<name>.system.txt`. An explicit `system`
        # argument still wins.
        sidecar = p.with_suffix(".system.txt")
        if system is None and sidecar.exists():
            system = sidecar.read_text(encoding="utf-8").strip() or None
        return cls(items=items, system=system or DEFAULT_SYSTEM, name=p.stem)

    # -- request construction ---------------------------------------------- #

    def request(
        self,
        item: WorkloadItem,
        seq: int,
        max_tokens: int = 192,
        temperature: float = 0.0,
        want_reference_answer: bool = False,
    ) -> GenRequest:
        return GenRequest(
            id=f"{item.id}#{seq}",
            prompt=item.prompt(),
            system=self.system,
            max_tokens=max_tokens,
            temperature=temperature,
            meta={
                "item_id": item.id,
                "reference_answer": item.reference_answer,
                "reference_citations": item.reference_citations,
                "source_ids": item.source_ids,
                "question": item.question,
                "want_reference_answer": want_reference_answer,
            },
        )

    def sample(
        self,
        n: int,
        max_tokens: int = 192,
        seed: int = 7,
        shuffle: bool = True,
        want_reference_answer: bool = False,
    ) -> list[GenRequest]:
        """Draw n requests, cycling through the set if n exceeds its size.

        Cycling is deliberate and is recorded: a 60-item set driven at 2000
        requests means each item is seen ~33 times, which a server with prefix
        caching will exploit. That is realistic for production RAG traffic, but
        it must be visible in the report rather than hidden.
        """
        rng = random.Random(seed)
        order = list(self.items)
        if shuffle:
            rng.shuffle(order)
        out: list[GenRequest] = []
        for i in range(n):
            item = order[i % len(order)]
            out.append(
                self.request(
                    item,
                    seq=i,
                    max_tokens=max_tokens,
                    want_reference_answer=want_reference_answer,
                )
            )
        return out

    def eval_requests(self, max_tokens: int = 256) -> list[GenRequest]:
        """One request per item, in file order, asking the backend for a real
        answer. Used by the quality harness, never by the load generator."""
        return [
            self.request(item, seq=i, max_tokens=max_tokens, want_reference_answer=True)
            for i, item in enumerate(self.items)
            if item.reference_answer
        ]

    def stats(self) -> dict[str, Any]:
        words = [len(i.prompt().split()) for i in self.items]
        answers = [len(i.reference_answer.split()) for i in self.items if i.reference_answer]
        words.sort()
        return {
            "n_items": len(self.items),
            "prompt_words_p50": words[len(words) // 2],
            "prompt_words_min": words[0],
            "prompt_words_max": words[-1],
            "approx_prompt_tokens_p50": round(words[len(words) // 2] * 1.35),
            "n_with_reference": len(answers),
            "reference_words_mean": round(sum(answers) / len(answers), 1) if answers else 0,
        }
