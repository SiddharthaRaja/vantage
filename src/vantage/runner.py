"""Run one configuration end to end: build backend, drive load, summarise, cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vantage.backends.base import Backend
from vantage.backends.registry import build_backend
from vantage.cache.embedders import build_embedder
from vantage.cache.semantic import CachedBackend, SemanticCache
from vantage.logging import info
from vantage.metrics.collector import summarize_run
from vantage.metrics.cost import cost_for_summary
from vantage.metrics.server import ServerScraper
from vantage.models import SLO, GenResult, LoadConfig, RunSummary
from vantage.workload.dataset import Workload
from vantage.workload.generator import generate_load


@dataclass
class RunSpec:
    """One point in a sweep: a backend, a load pattern, and a label."""

    id: str
    backend: str | dict[str, Any]
    load: LoadConfig
    max_tokens: int = 192
    label: str = ""
    group: str = "default"
    backend_overrides: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    server: str | None = None          # a managed server (servers.py) this run needs


@dataclass
class RunOutcome:
    summary: RunSummary
    results: list[GenResult]
    backend_name: str
    engine_stats: dict[str, Any] = field(default_factory=dict)


def _n_requests_for(spec: RunSpec) -> int:
    if spec.load.n_requests:
        return spec.load.n_requests
    rate = spec.load.rate_qps or spec.load.concurrency
    duration = spec.load.duration_s or 10.0
    return max(32, int(rate * duration * 1.5) + 32)


async def run_spec(
    spec: RunSpec,
    workload: Workload,
    slo: SLO | None = None,
    progress: bool = True,
) -> RunOutcome:
    backend: Backend = build_backend(spec.backend, {**spec.backend_overrides, "config_id": spec.id})

    if spec.cache:
        cache = SemanticCache(
            embedder=build_embedder(spec.cache.get("embedder")),
            threshold=float(spec.cache.get("threshold", 0.92)),
            capacity=int(spec.cache.get("capacity", 10_000)),
            ttl_s=spec.cache.get("ttl_s"),
            exact_first=bool(spec.cache.get("exact_first", True)),
        )
        backend = CachedBackend(backend, cache)

    requests = workload.sample(
        n=_n_requests_for(spec),
        max_tokens=spec.max_tokens,
        seed=spec.load.seed,
    )

    if progress:
        info(f"[bold]{spec.id}[/bold] -> {spec.label or spec.backend}")

    inner = getattr(backend, "inner", backend)
    metrics_url = getattr(inner, "metrics_url", None)
    trace = None
    async with backend:
        if metrics_url:
            interval = float(getattr(inner, "metrics_interval_s", 1.0))
            async with ServerScraper(metrics_url, interval_s=interval) as scraper:
                run = await generate_load(backend, requests, spec.load, progress=progress)
            trace = scraper.trace
        else:
            run = await generate_load(backend, requests, spec.load, progress=progress)

    summary = summarize_run(
        run,
        config_id=spec.id,
        backend=getattr(backend, "name", str(spec.backend)),
        label=spec.label or spec.id,
        params={
            "group": spec.group,
            "mode": spec.load.mode,
            "concurrency": spec.load.concurrency,
            "rate_qps": spec.load.rate_qps,
            "max_tokens": spec.max_tokens,
            "cache": spec.cache or None,
            **spec.params,
        },
        slo=slo,
    )
    summary.cost = cost_for_summary(summary, backend.pricing())

    engine_stats = inner.engine_stats() if hasattr(inner, "engine_stats") else {}
    if trace is not None:
        summary.server = trace.summary()
        engine_stats["server"] = summary.server
        engine_stats["server_series"] = trace.series()
    if run.shed:
        summary.errors_by_kind.setdefault("client_backlog", run.shed)

    return RunOutcome(
        summary=summary,
        results=run.results,
        backend_name=getattr(backend, "name", "?"),
        engine_stats=engine_stats,
    )
