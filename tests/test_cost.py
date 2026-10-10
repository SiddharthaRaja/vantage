"""The cost model is arithmetic, which means it can be tested exactly -- and
should be, because a wrong break-even number is a wrong decision."""

from __future__ import annotations

from vantage.backends.base import Pricing
from vantage.metrics.cost import api_cost_per_request, break_even_qps, cost_for_summary
from vantage.models import RunSummary

API = Pricing(kind="api", input_usd_per_mtok=0.15, output_usd_per_mtok=0.60)
HOST = Pricing(kind="self_hosted", gpu_hourly_usd=0.72, n_gpus=1)


def test_api_cost_per_request_is_linear_in_tokens():
    assert api_cost_per_request(API, 1_000_000, 0) == 0.15
    assert api_cost_per_request(API, 0, 1_000_000) == 0.60
    assert round(api_cost_per_request(API, 1000, 200), 8) == round(0.00015 + 0.00012, 8)


def test_self_hosted_cost_falls_as_utilisation_rises():
    def summary(qps: float) -> RunSummary:
        return RunSummary(
            config_id="c", backend="b", n_ok=100, n_requests=100,
            total_prompt_tokens=100_000, total_completion_tokens=20_000,
            achieved_qps=qps, output_tokens_per_s=qps * 200, duration_s=100,
        )

    cheap = cost_for_summary(summary(20.0), HOST)
    dear = cost_for_summary(summary(2.0), HOST)
    assert cheap.model == "self_hosted"
    assert cheap.usd_per_1k_requests < dear.usd_per_1k_requests
    # Exactly 10x the traffic for the same hourly bill is exactly 1/10 the cost.
    assert round(dear.usd_per_1k_requests / cheap.usd_per_1k_requests, 6) == 10.0


def test_break_even_is_where_the_two_lines_cross():
    be = break_even_qps(API, HOST, avg_prompt_tokens=1000, avg_completion_tokens=200)
    assert be is not None
    per_req = api_cost_per_request(API, 1000, 200)
    assert round(be.qps * per_req, 10) == round(0.72 / 3600, 10)
    assert round(be.requests_per_day) == round(be.qps * 86400)


def test_break_even_returns_none_without_pricing():
    assert break_even_qps(Pricing(), HOST, 100, 100) is None
    assert break_even_qps(API, Pricing(), 100, 100) is None


def test_cost_is_unknown_when_nothing_succeeded():
    empty = RunSummary(config_id="c", backend="b", n_ok=0)
    assert cost_for_summary(empty, API).model == "unknown"
