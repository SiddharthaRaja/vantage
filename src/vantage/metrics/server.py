"""What the server saw: scrape vLLM's Prometheus endpoint during a run.

Client-side numbers say *that* latency broke at some concurrency. The server
says *why*: whether the KV cache filled, whether requests sat in vLLM's own
waiting queue, whether running requests were preempted (evicted from the KV
cache and recomputed later), and how often the shared prompt prefix was reused.

Polled every second from a background task while the load runs. Gauges are
sampled; counters and histogram sums are differenced between the first and
last scrape, so the numbers describe this run only, not the server's lifetime.

Metric names changed between vLLM's V0 and V1 engines; both spellings are read.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# canonical name -> the Prometheus names it may appear under (first found wins)
GAUGES: dict[str, tuple[str, ...]] = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}
COUNTERS: dict[str, tuple[str, ...]] = {
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
    "prompt_tokens": ("vllm:prompt_tokens_total",),
    "generation_tokens": ("vllm:generation_tokens_total",),
    "ttft_sum": ("vllm:time_to_first_token_seconds_sum",),
    "ttft_count": ("vllm:time_to_first_token_seconds_count",),
    "queue_sum": ("vllm:request_queue_time_seconds_sum",),
    "queue_count": ("vllm:request_queue_time_seconds_count",),
    "prefill_sum": ("vllm:request_prefill_time_seconds_sum",),
    "prefill_count": ("vllm:request_prefill_time_seconds_count",),
}

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum every sample of each metric across its label sets.

    Labels here are the model name and engine index; one server serves one
    model, so summing is the right reduction (it also handles several engines).
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        if math.isnan(value):
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + value
    return out


def pick(sample: dict[str, float], names: tuple[str, ...]) -> float | None:
    for n in names:
        if n in sample:
            return sample[n]
    return None


@dataclass
class ServerTrace:
    """Scrapes taken during one run, and their reduction to a summary."""

    url: str
    times: list[float] = field(default_factory=list)
    samples: list[dict[str, float]] = field(default_factory=list)
    errors: int = 0

    def add(self, t: float, raw: dict[str, float]) -> None:
        row: dict[str, float] = {}
        for key, names in {**GAUGES, **COUNTERS}.items():
            v = pick(raw, names)
            if v is not None:
                row[key] = v
        self.times.append(t)
        self.samples.append(row)

    def _series(self, key: str) -> list[float]:
        return [s[key] for s in self.samples if key in s]

    def _delta(self, key: str) -> float | None:
        vals = self._series(key)
        return (vals[-1] - vals[0]) if len(vals) >= 2 else None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"url": self.url, "n_scrapes": len(self.samples), "scrape_errors": self.errors}
        if not self.samples:
            return out
        for key in GAUGES:
            vals = self._series(key)
            if vals:
                ordered = sorted(vals)
                out[f"{key}_mean"] = sum(vals) / len(vals)
                out[f"{key}_p95"] = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
                out[f"{key}_max"] = ordered[-1]
        preempt = self._delta("preemptions")
        if preempt is not None:
            out["preemptions"] = int(round(preempt))
        hits, queries = self._delta("prefix_hits"), self._delta("prefix_queries")
        if hits is not None and queries:
            out["prefix_cache_hit_rate"] = hits / queries
        for name in ("ttft", "queue", "prefill"):
            s, c = self._delta(f"{name}_sum"), self._delta(f"{name}_count")
            if s is not None and c:
                out[f"server_{name}_mean_s"] = s / c
        gen, span = self._delta("generation_tokens"), self.times[-1] - self.times[0]
        if gen is not None and span > 0:
            out["server_generation_tokens_per_s"] = gen / span
        return out

    def series(self) -> dict[str, list[float]]:
        """Compact time series for charts: seconds since start, per gauge."""
        t0 = self.times[0] if self.times else 0.0
        out: dict[str, list[float]] = {"t": [round(t - t0, 2) for t in self.times]}
        for key in GAUGES:
            out[key] = [s.get(key, float("nan")) for s in self.samples]
        return out


class ServerScraper:
    """Background poller. Use as an async context manager around the load:

        async with ServerScraper(url) as scraper:
            await generate_load(...)
        scraper.trace.summary()

    A scrape failure is counted, never raised: a benchmark must not die
    because its observer did.
    """

    def __init__(self, url: str, interval_s: float = 1.0, transport: httpx.AsyncBaseTransport | None = None):
        self.trace = ServerTrace(url=url)
        self.interval_s = interval_s
        self._client = httpx.AsyncClient(timeout=5.0, transport=transport)
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def scrape_once(self) -> None:
        try:
            r = await self._client.get(self.trace.url)
            r.raise_for_status()
            self.trace.add(time.perf_counter(), parse_prometheus(r.text))
        except Exception:  # noqa: BLE001
            self.trace.errors += 1

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await self.scrape_once()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)

    async def __aenter__(self) -> ServerScraper:
        await self.scrape_once()                 # baseline for the counters
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        if self._task:
            await self._task
        await self.scrape_once()                 # final values for the counters
        await self._client.aclose()


def metrics_url_for(base_url: str) -> str:
    """http://host:8000/v1 -> http://host:8000/metrics (vLLM serves both)."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + "/metrics"
