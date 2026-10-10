"""Load generation: closed loop and open loop.

The distinction is the single most important methodological choice in the
project, and it is the thing most benchmarks get wrong.

Closed loop  N virtual users. Each sends a request, waits for the answer, then
             sends the next. The system can never be offered more load than it
             can handle, because slow responses throttle the client. Answers
             "how does it behave at concurrency N".

Open loop    Requests arrive on a schedule, usually Poisson, regardless of
             whether earlier ones have finished. Answers "what happens at X
             requests per second", and is the only mode in which you can see
             the queue build and latency go vertical. Real users are open loop:
             they do not politely wait for your server to recover.

Both modes run a discarded warm-up phase first. Without it the first few
requests carry CUDA graph capture, lazy weight loading and cold caches, and
they will dominate a p99 computed over a short run.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from vantage.backends.base import Backend
from vantage.logging import info
from vantage.models import GenRequest, GenResult, LoadConfig


@dataclass
class LoadRun:
    results: list[GenResult]
    warmup_results: list[GenResult]
    window_start: float
    window_end: float
    offered_qps: float | None
    shed: int = 0

    @property
    def duration_s(self) -> float:
        return max(self.window_end - self.window_start, 1e-9)


async def _run_one(backend: Backend, req: GenRequest, arrival_ts: float | None = None) -> GenResult:
    return await backend.generate(req, arrival_ts=arrival_ts)


async def run_closed_loop(
    backend: Backend,
    requests: list[GenRequest],
    cfg: LoadConfig,
    progress: bool = False,
) -> LoadRun:
    queue: asyncio.Queue[GenRequest] = asyncio.Queue()
    for r in requests:
        queue.put_nowait(r)

    results: list[GenResult] = []
    deadline = (time.perf_counter() + cfg.duration_s) if cfg.duration_s else None

    async def worker() -> None:
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                return
            try:
                req = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await _run_one(backend, req))

    start = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(cfg.concurrency)])
    end = time.perf_counter()

    if progress:
        info(f"  closed loop c={cfg.concurrency}: {len(results)} requests in {end - start:.1f}s")
    return LoadRun(
        results=results,
        warmup_results=[],
        window_start=start,
        window_end=end,
        offered_qps=None,
    )


async def run_open_loop(
    backend: Backend,
    requests: list[GenRequest],
    cfg: LoadConfig,
    progress: bool = False,
) -> LoadRun:
    rate = float(cfg.rate_qps or 1.0)
    rng = random.Random(cfg.seed)

    # Pre-compute the arrival schedule so the generator itself never becomes
    # the bottleneck: the schedule exists before the first request is sent.
    arrivals: list[float] = []
    t = 0.0
    budget_n = cfg.n_requests or 10_000_000
    budget_t = cfg.duration_s or float("inf")
    while len(arrivals) < budget_n and t < budget_t:
        gap = rng.expovariate(rate) if cfg.distribution == "poisson" else 1.0 / rate
        t += gap
        if t >= budget_t:
            break
        arrivals.append(t)

    results: list[GenResult] = []
    in_flight: set[asyncio.Task[GenResult]] = set()
    shed = 0
    start = time.perf_counter()

    def _done(task: asyncio.Task[GenResult]) -> None:
        in_flight.discard(task)
        try:
            results.append(task.result())
        except Exception as exc:  # pragma: no cover - defensive
            results.append(
                GenResult(
                    request_id="unknown",
                    backend=backend.name,
                    config_id=backend.config_id,
                    ok=False,
                    error=str(exc),
                    error_kind="task_error",
                )
            )

    for i, offset in enumerate(arrivals):
        target = start + offset
        now = time.perf_counter()
        if target > now:
            await asyncio.sleep(target - now)
        scheduled = start + offset

        if len(in_flight) >= cfg.max_in_flight:
            # The client itself is saturated. Record it as load shedding rather
            # than quietly slowing the arrival process down -- a generator that
            # throttles itself has silently turned into a closed-loop test.
            shed += 1
            results.append(
                GenResult(
                    request_id=requests[i % len(requests)].id,
                    backend=backend.name,
                    config_id=backend.config_id,
                    ok=False,
                    error=f"client backlog exceeded max_in_flight={cfg.max_in_flight}",
                    error_kind="client_backlog",
                    arrival_ts=scheduled,
                    dispatch_ts=scheduled,
                )
            )
            continue

        task = asyncio.create_task(
            _run_one(backend, requests[i % len(requests)], arrival_ts=scheduled)
        )
        in_flight.add(task)
        task.add_done_callback(_done)

    if in_flight:
        await asyncio.gather(*list(in_flight), return_exceptions=True)
    end = time.perf_counter()

    if progress:
        info(f"  open loop {rate:g} qps: {len(results)} requests, {shed} shed, {end - start:.1f}s")
    return LoadRun(
        results=results,
        warmup_results=[],
        window_start=start,
        window_end=end,
        offered_qps=rate,
        shed=shed,
    )


async def warmup(backend: Backend, requests: list[GenRequest], n: int, concurrency: int) -> list[GenResult]:
    """Fire and discard. Results are returned so they can be inspected, but
    they never enter the summary."""
    if n <= 0:
        return []
    sub = [r.model_copy(update={"id": f"warmup-{r.id}", "is_warmup": True}) for r in requests[:n]]
    cfg = LoadConfig(mode="closed", concurrency=max(1, min(concurrency, n)), n_requests=n, warmup_requests=0)
    run = await run_closed_loop(backend, sub, cfg)
    for r in run.results:
        r.is_warmup = True
    return run.results


async def generate_load(
    backend: Backend,
    requests: list[GenRequest],
    cfg: LoadConfig,
    progress: bool = False,
) -> LoadRun:
    warm = await warmup(backend, requests, cfg.warmup_requests, cfg.concurrency)
    if cfg.mode == "closed":
        run = await run_closed_loop(backend, requests, cfg, progress=progress)
    else:
        run = await run_open_loop(backend, requests, cfg, progress=progress)
    run.warmup_results = warm
    return run
