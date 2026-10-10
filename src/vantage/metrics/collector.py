"""Turn a finished load run into one RunSummary."""

from __future__ import annotations

from collections import Counter
from typing import Any

from vantage.metrics.stats import summarize_latency
from vantage.models import SLO, GenResult, RunSummary
from vantage.workload.generator import LoadRun


def summarize_run(
    run: LoadRun,
    config_id: str,
    backend: str,
    label: str = "",
    params: dict[str, Any] | None = None,
    slo: SLO | None = None,
) -> RunSummary:
    measured: list[GenResult] = [r for r in run.results if not r.is_warmup]
    ok = [r for r in measured if r.ok]
    failed = [r for r in measured if not r.ok]
    duration = run.duration_s

    slo = slo or SLO()
    met = sum(1 for r in measured if slo.request_meets(r))

    completion_tokens = sum(r.completion_tokens for r in ok)
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    cache_kinds = Counter(r.cache_kind for r in measured)

    summary = RunSummary(
        config_id=config_id,
        backend=backend,
        label=label or config_id,
        params=params or {},
        n_requests=len(measured),
        n_ok=len(ok),
        n_failed=len(failed),
        error_rate=(len(failed) / len(measured)) if measured else 0.0,
        errors_by_kind=dict(Counter(r.error_kind or "unknown" for r in failed)),
        duration_s=duration,
        offered_qps=run.offered_qps,
        achieved_qps=len(ok) / duration if duration > 0 else 0.0,
        output_tokens_per_s=completion_tokens / duration if duration > 0 else 0.0,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        token_counts_estimated=any(r.token_counts_estimated for r in ok),
        ttft=summarize_latency([r.ttft_s for r in ok if r.ttft_s is not None]),
        e2e=summarize_latency([r.e2e_s for r in ok]),
        total=summarize_latency([r.total_s for r in ok]),
        queue=summarize_latency([r.queue_s for r in ok]),
        itl=summarize_latency([r.itl_mean_s for r in ok if r.itl_mean_s is not None]),
        goodput_ratio=(met / len(measured)) if measured else 0.0,
        slo=slo,
        cache_hit_rate=(
            sum(1 for r in measured if r.cache_hit) / len(measured) if measured else 0.0
        ),
        cache_kinds=dict(cache_kinds),
    )
    return summary


def goodput_qps(summary: RunSummary) -> float:
    """Requests per second that actually met the SLO. The honest headline."""
    return summary.achieved_qps * summary.goodput_ratio if summary.n_requests else 0.0
