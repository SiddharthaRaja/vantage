"""A simulated inference server.

Why a simulator exists in a serving benchmark:

1. You can run and debug the entire harness -- load generator, metrics,
   quality gate, cost model, charts, CI -- on a laptop with no GPU and no API
   key. Every test in tests/ runs against this.
2. It is a *model* of a continuous-batching server, so the shapes it produces
   are the shapes you are looking for on real hardware: throughput saturates,
   per-stream decode slows as the batch grows, and queueing delay explodes once
   offered load passes capacity. If your analysis code cannot find the knee
   here, it will not find it on an A100 either.

It is not a substitute for real measurement, and every report generated from it
is stamped "simulated". The numbers in your README should come from real
hardware; the simulator is how you make sure the pipeline that produces them is
correct before you start paying for GPU hours.

The model
---------
Per-sequence decode rate with b sequences in flight:

    r(b) = r1 * k / (k + b - 1)

so aggregate decode throughput is r1 * k * b / (k + b - 1), which rises
steeply, bends, and saturates at r1 * k tokens/s. This is the standard
saturating form of a batched decoder: adding a sequence to the batch is nearly
free while the GPU is memory-bandwidth-bound, and stops being free once it
isn't. `max_batch` caps how many sequences the engine will run at once;
everything beyond that waits in the engine queue, which is where server-side
queueing delay comes from.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from vantage.backends.base import Backend, Health, Pricing, StreamChunk
from vantage.errors import RequestFailed

# Filler tokens for load-testing requests, where only the token count matters.
_FILLER = [
    "serum", "concentration", "was", "measured", "across", "the", "validation",
    "cohort", "and", "the", "resulting", "distribution", "indicates", "that",
    "the", "assay", "performs", "within", "the", "stated", "tolerance", "for",
    "routine", "clinical", "use", "under", "the", "described", "conditions",
]


class SimulatedBackend(Backend):
    """A continuous-batching server you can run anywhere."""

    kind = "simulated"

    def __init__(
        self,
        name: str = "sim",
        model: str = "simulated-7b",
        decode_tokens_per_s: float = 320.0,
        batch_constant: float = 20.0,
        prefill_tokens_per_s: float = 24000.0,
        max_batch: int = 32,
        overhead_s: float = 0.012,
        jitter: float = 0.10,
        fail_rate: float = 0.0,
        prefix_cache: bool = False,
        prefix_cache_hit_rate: float = 0.0,
        prefix_cache_speedup: float = 4.0,
        fidelity: float = 1.0,
        style_noise: float = 0.4,
        seed: int = 11,
        pricing: dict[str, Any] | None = None,
        config_id: str | None = None,
        **params: Any,
    ):
        super().__init__(name=name, config_id=config_id, model=model, **params)
        self.model = model
        self.r1 = decode_tokens_per_s
        self.k = batch_constant
        self.prefill_rate = prefill_tokens_per_s
        self.max_batch = max_batch
        self.overhead_s = overhead_s
        self.jitter = jitter
        self.fail_rate = fail_rate
        self.prefix_cache = prefix_cache
        self.prefix_cache_hit_rate = prefix_cache_hit_rate
        self.prefix_cache_speedup = prefix_cache_speedup
        self.fidelity = fidelity
        self.style_noise = style_noise
        self.seed = seed
        self._pricing_cfg = pricing or {}

        self._slots = asyncio.Semaphore(max_batch)
        self._active = 0
        self._peak_batch = 0
        self._queue_waits: list[float] = []

    # -- the model ---------------------------------------------------------- #

    def per_seq_decode_rate(self, batch: int) -> float:
        b = max(1, batch)
        return self.r1 * self.k / (self.k + b - 1)

    def aggregate_throughput(self, batch: int) -> float:
        return self.per_seq_decode_rate(batch) * max(1, batch)

    def _rng(self, request_id: str) -> random.Random:
        return random.Random(f"{self.seed}:{request_id}")

    def _jittered(self, value: float, rng: random.Random) -> float:
        if self.jitter <= 0:
            return value
        return max(0.0, value * rng.gauss(1.0, self.jitter))

    # -- generation --------------------------------------------------------- #

    async def _stream(self, req) -> AsyncIterator[StreamChunk]:
        rng = self._rng(req.id)

        if self.fail_rate and rng.random() < self.fail_rate:
            await asyncio.sleep(self._jittered(0.02, rng))
            raise RequestFailed("simulated upstream failure", status=503, retryable=True)

        prompt_tokens = max(1, round(len(req.prompt.split()) * 1.35) + 8)
        words, completion_tokens = self._plan_output(req, rng)

        queued_at = time.perf_counter()
        async with self._slots:
            self._queue_waits.append(time.perf_counter() - queued_at)
            self._active += 1
            self._peak_batch = max(self._peak_batch, self._active)
            try:
                batch_now = self._active
                prefill_s = prompt_tokens / self.prefill_rate * (1 + 0.12 * (batch_now - 1))
                if self.prefix_cache and rng.random() < self.prefix_cache_hit_rate:
                    prefill_s /= self.prefix_cache_speedup
                await asyncio.sleep(self._jittered(prefill_s + self.overhead_s, rng))

                yield StreamChunk(text=words[0])
                for word in words[1:]:
                    rate = self.per_seq_decode_rate(self._active)
                    await asyncio.sleep(max(0.0, 1.0 / rate))
                    yield StreamChunk(text=" " + word)
            finally:
                self._active -= 1

        yield StreamChunk(
            text="",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )

    def _plan_output(self, req, rng: random.Random) -> tuple[list[str], int]:
        """Decide what the model 'says' and how many tokens that costs.

        Load-testing requests get filler of the requested length -- the content
        is irrelevant, the token count is not. Quality-harness requests (which
        set want_reference_answer) get the reference answer, degraded with
        probability 1 - fidelity. That degradation is what lets a quantised
        profile show up as a real quality drop in the report rather than as an
        assertion.
        """
        if req.meta.get("want_reference_answer") and req.meta.get("reference_answer"):
            answer = str(req.meta["reference_answer"])
            if rng.random() > self.fidelity:
                answer = _degrade(answer, rng)
            elif rng.random() < self.style_noise:
                # Even a correct answer is not word-identical to the reference.
                # Without this every faithful profile scores exactly 1.000,
                # which no real system does and which would make the quality
                # gate look like a rubber stamp.
                answer = _restyle(answer, rng)
            words = answer.split() or ["(empty)"]
            return words, max(1, round(len(words) * 1.3))

        n_words = max(1, round(req.max_tokens / 1.3))
        words = [_FILLER[i % len(_FILLER)] for i in range(n_words)]
        return words, req.max_tokens

    # -- misc --------------------------------------------------------------- #

    async def health(self) -> Health:
        return Health(
            ok=True,
            detail=(
                f"simulated: r1={self.r1:.0f} tok/s, k={self.k:.0f}, "
                f"max_batch={self.max_batch}, saturates ~{self.r1 * self.k:.0f} tok/s"
            ),
            model=self.model,
        )

    def pricing(self) -> Pricing:
        cfg = self._pricing_cfg
        if not cfg:
            return Pricing(kind="self_hosted", gpu_hourly_usd=0.75, n_gpus=1, note="default sim GPU")
        return Pricing(
            kind=cfg.get("kind", "self_hosted"),
            input_usd_per_mtok=cfg.get("input_usd_per_mtok"),
            output_usd_per_mtok=cfg.get("output_usd_per_mtok"),
            gpu_hourly_usd=cfg.get("gpu_hourly_usd"),
            n_gpus=cfg.get("n_gpus", 1),
            note=cfg.get("note", ""),
        )

    def engine_stats(self) -> dict[str, Any]:
        waits = self._queue_waits
        return {
            "peak_batch": self._peak_batch,
            "mean_engine_queue_s": (sum(waits) / len(waits)) if waits else 0.0,
            "saturation_tokens_per_s": self.r1 * self.k,
        }


_STYLES = [
    "Based on the sources, {}",
    "The reported value is as follows: {}",
    "According to the evidence provided, {}",
    "{} This is stated directly in the cited source.",
]


def _restyle(answer: str, rng: random.Random) -> str:
    """Reword a correct answer without touching its numbers or citations."""
    template = rng.choice(_STYLES)
    body = answer
    if template.startswith("{}") is False:
        body = body[0].lower() + body[1:] if body else body
    return template.format(body)


def _degrade(answer: str, rng: random.Random) -> str:
    """Make an answer worse in one of the ways a smaller or more aggressively
    quantised model actually gets worse: it drops its citation, it rounds or
    invents a number, or it truncates."""
    mode = rng.choice(["drop_citation", "perturb_number", "truncate"])
    if mode == "drop_citation":
        return re.sub(r"\s*\[\d+\]", "", answer)
    if mode == "perturb_number":
        def _bump(m: re.Match[str]) -> str:
            value = float(m.group(0))
            return f"{value * rng.choice([0.7, 1.4]):.1f}"

        return re.sub(r"\d+\.\d+|\d+", _bump, answer, count=1)
    words = answer.split()
    return " ".join(words[: max(3, len(words) // 2)])
