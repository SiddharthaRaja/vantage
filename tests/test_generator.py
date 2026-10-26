"""Load generation. The open/closed distinction is the methodological core of
the project, so it gets tested as behaviour, not just as code that runs."""

from __future__ import annotations

import asyncio

import pytest

from vantage.backends.simulated import SimulatedBackend
from vantage.metrics.collector import summarize_run
from vantage.models import SLO, LoadConfig
from vantage.workload.generator import generate_load, run_closed_loop, run_open_loop


def _requests(workload, n, max_tokens=16):
    return workload.sample(n=n, max_tokens=max_tokens, seed=3)


def test_closed_loop_runs_every_request_once(tiny_workload):
    sim = SimulatedBackend(max_batch=8)
    cfg = LoadConfig(mode="closed", concurrency=4, n_requests=12, warmup_requests=0)
    run = asyncio.run(run_closed_loop(sim, _requests(tiny_workload, 12), cfg))
    assert len(run.results) == 12
    assert {r.request_id for r in run.results} == {r.id for r in _requests(tiny_workload, 12)}
    assert all(r.ok for r in run.results)


def test_closed_loop_never_exceeds_its_concurrency(tiny_workload):
    sim = SimulatedBackend(max_batch=64, jitter=0.0)
    cfg = LoadConfig(mode="closed", concurrency=3, n_requests=15, warmup_requests=0)
    asyncio.run(run_closed_loop(sim, _requests(tiny_workload, 15), cfg))
    assert sim.engine_stats()["peak_batch"] <= 3


def test_higher_concurrency_raises_throughput_until_it_does_not(tiny_workload):
    def throughput(concurrency: int) -> float:
        sim = SimulatedBackend(max_batch=32)
        cfg = LoadConfig(mode="closed", concurrency=concurrency, n_requests=24, warmup_requests=0)
        run = asyncio.run(run_closed_loop(sim, _requests(tiny_workload, 24), cfg))
        return summarize_run(run, "c", "sim").output_tokens_per_s

    assert throughput(8) > throughput(1)


def test_open_loop_arrivals_are_independent_of_completions(tiny_workload):
    sim = SimulatedBackend(max_batch=2, decode_tokens_per_s=60)
    cfg = LoadConfig(mode="open", rate_qps=40, duration_s=1.0, n_requests=None,
                     warmup_requests=0, distribution="constant", max_in_flight=512)
    run = asyncio.run(run_open_loop(sim, _requests(tiny_workload, 64, max_tokens=24), cfg))
    # A closed-loop client with 2 slots could not have issued this many in 1s.
    assert len(run.results) > 20
    assert run.offered_qps == 40


def test_open_loop_sheds_rather_than_silently_throttling(tiny_workload):
    sim = SimulatedBackend(max_batch=1, decode_tokens_per_s=30)
    cfg = LoadConfig(mode="open", rate_qps=60, duration_s=0.8, n_requests=None,
                     warmup_requests=0, distribution="constant", max_in_flight=4)
    run = asyncio.run(run_open_loop(sim, _requests(tiny_workload, 64, max_tokens=24), cfg))
    assert run.shed > 0
    shed = [r for r in run.results if r.error_kind == "client_backlog"]
    assert shed and all(not r.ok for r in shed)


def test_warmup_results_are_excluded_from_the_summary(tiny_workload):
    sim = SimulatedBackend()
    cfg = LoadConfig(mode="closed", concurrency=2, n_requests=8, warmup_requests=4)
    run = asyncio.run(generate_load(sim, _requests(tiny_workload, 8), cfg))
    assert len(run.warmup_results) == 4
    summary = summarize_run(run, "c", "sim")
    assert summary.n_requests == 8  # warm-ups are not counted
    assert all(r.is_warmup for r in run.warmup_results)


def test_open_loop_requires_a_rate():
    with pytest.raises(ValueError):
        LoadConfig(mode="open", duration_s=1.0)


def test_goodput_falls_when_the_slo_is_impossible(tiny_workload):
    sim = SimulatedBackend(max_batch=1, decode_tokens_per_s=40)
    cfg = LoadConfig(mode="closed", concurrency=8, n_requests=16, warmup_requests=0)
    run = asyncio.run(run_closed_loop(sim, _requests(tiny_workload, 16, max_tokens=32), cfg))
    strict = summarize_run(run, "c", "sim", slo=SLO(ttft_p95_s=0.0001, total_p95_s=0.0001))
    loose = summarize_run(run, "c", "sim", slo=SLO(ttft_p95_s=60, total_p95_s=60))
    assert strict.goodput_ratio == 0.0
    assert loose.goodput_ratio == 1.0


def test_duration_alone_is_not_capped_by_the_request_default():
    """An open-loop point given only a duration runs for that duration.

    Regression: n_requests defaulted to 32 and won, so a 60 s point at 4 req/s
    stopped after 8 s and no queue ever formed.
    """
    cfg = LoadConfig(mode="open", rate_qps=4, duration_s=60)
    assert cfg.n_requests is None
    both = LoadConfig(mode="open", rate_qps=4, duration_s=60, n_requests=30)
    assert both.n_requests == 30          # an explicit count still caps
    closed = LoadConfig(mode="closed", concurrency=2)
    assert closed.n_requests == 32        # closed loop keeps its default
