"""Cost modelling, and the break-even question.

Two regimes:

  API       you pay per token. Cost scales linearly with traffic and is
            independent of utilisation. Zero traffic costs zero.
  Self-host you pay per GPU-hour whether or not anyone is using it. Cost per
            request is (hourly rate) / (requests per second * 3600), so it
            falls as utilisation rises and is ruinous at low traffic.

Those two lines cross. Finding where they cross -- the break-even QPS -- is the
actual engineering question behind "should we self-host", and it takes about
fifteen lines of arithmetic that almost nobody does before buying a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from vantage.backends.base import Pricing
from vantage.models import CostBreakdown, RunSummary

SECONDS_PER_HOUR = 3600.0


def api_cost_per_request(pricing: Pricing, prompt_tokens: float, completion_tokens: float) -> float:
    pin = pricing.input_usd_per_mtok or 0.0
    pout = pricing.output_usd_per_mtok or 0.0
    return (prompt_tokens / 1e6) * pin + (completion_tokens / 1e6) * pout


def self_host_cost_per_hour(pricing: Pricing) -> float:
    return (pricing.gpu_hourly_usd or 0.0) * max(1, pricing.n_gpus)


def cost_for_summary(summary: RunSummary, pricing: Pricing) -> CostBreakdown:
    if summary.n_ok == 0:
        return CostBreakdown(model="unknown", notes="no successful requests")

    avg_in = summary.total_prompt_tokens / summary.n_ok
    avg_out = summary.total_completion_tokens / summary.n_ok

    if pricing.kind == "api" and (pricing.input_usd_per_mtok or pricing.output_usd_per_mtok):
        per_req = api_cost_per_request(pricing, avg_in, avg_out)
        per_mtok_out = (
            (per_req / avg_out) * 1e6 if avg_out else float("nan")
        )
        return CostBreakdown(
            model="api",
            usd_per_1k_requests=per_req * 1000,
            usd_per_1m_output_tokens=per_mtok_out,
            notes=(
                f"list pricing: ${pricing.input_usd_per_mtok}/Mtok in, "
                f"${pricing.output_usd_per_mtok}/Mtok out; "
                f"avg {avg_in:.0f} in / {avg_out:.0f} out per request"
            ),
        )

    if pricing.kind == "self_hosted" and pricing.gpu_hourly_usd:
        hourly = self_host_cost_per_hour(pricing)
        per_second = hourly / SECONDS_PER_HOUR
        per_req = per_second / summary.achieved_qps if summary.achieved_qps > 0 else float("inf")
        per_mtok_out = (
            per_second / summary.output_tokens_per_s * 1e6
            if summary.output_tokens_per_s > 0
            else float("inf")
        )
        return CostBreakdown(
            model="self_hosted",
            usd_per_1k_requests=per_req * 1000,
            usd_per_1m_output_tokens=per_mtok_out,
            notes=(
                f"${hourly:.2f}/hour for {max(1, pricing.n_gpus)} GPU(s) at the measured "
                f"{summary.achieved_qps:.2f} req/s. Cost per request is utilisation-dependent: "
                f"idle capacity is paid for."
            ),
        )

    return CostBreakdown(model="unknown", notes="no pricing configured for this backend")


@dataclass
class BreakEven:
    qps: float
    requests_per_day: float
    api_usd_per_request: float
    self_host_usd_per_hour: float
    explanation: str


def break_even_qps(
    api: Pricing,
    self_hosted: Pricing,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
) -> BreakEven | None:
    """The sustained request rate above which self-hosting is cheaper.

    Below it you are renting an idle GPU; above it the per-token bill wins.
    Note what this deliberately leaves out: engineering time, on-call, spare
    capacity for peaks, and the fact that you must provision for peak QPS while
    paying for mean QPS. The report says so rather than pretending the number
    is a decision on its own.
    """
    per_req = api_cost_per_request(api, avg_prompt_tokens, avg_completion_tokens)
    hourly = self_host_cost_per_hour(self_hosted)
    if per_req <= 0 or hourly <= 0:
        return None
    qps = (hourly / SECONDS_PER_HOUR) / per_req
    return BreakEven(
        qps=qps,
        requests_per_day=qps * 86400,
        api_usd_per_request=per_req,
        self_host_usd_per_hour=hourly,
        explanation=(
            f"API costs ${per_req:.5f}/request at {avg_prompt_tokens:.0f} in / "
            f"{avg_completion_tokens:.0f} out tokens. A ${hourly:.2f}/hour GPU costs "
            f"${hourly / SECONDS_PER_HOUR:.5f}/second regardless of load, so it pays for "
            f"itself above {qps:.2f} req/s sustained (~{qps * 86400:,.0f} requests/day) "
            f"-- provided the GPU can actually serve that rate within SLO."
        ),
    )
