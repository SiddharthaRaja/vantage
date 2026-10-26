"""Every value that crosses a module boundary is one of these types.

Why pydantic rather than dicts: a benchmark harness lives or dies on the
integrity of its numbers. If a backend forgets to report completion tokens, the
throughput figure silently becomes wrong. Typed boundaries turn that into a
loud validation error at the point it happens instead of a plausible-looking
chart three steps later.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Requests and results
# --------------------------------------------------------------------------- #


class GenRequest(BaseModel):
    """One unit of work sent to a backend."""

    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    system: str | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    stop: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    """Carried through untouched. Used for the reference answer, the gold
    citations, and anything else the quality harness needs later."""


class GenResult(BaseModel):
    """The outcome of one request, including every timing we can defend.

    Timing definitions (these are the industry-standard ones; stating them
    explicitly matters because half of published LLM benchmarks mean different
    things by the same words):

      queue_s  arrival time -> the moment the request was actually dispatched.
               Only non-zero in open-loop mode, where arrivals are independent
               of completions. This is client-side queueing and it is the first
               thing that explodes when you push past capacity.
      ttft_s   dispatch -> first token received by the client. Covers network,
               scheduling and prefill.
      itl_s    inter-token latency: the gaps between successive tokens during
               decode. Reported as mean and p95 rather than as a full list,
               because the list is large and uninteresting in aggregate.
      e2e_s    dispatch -> last token. ttft_s + decode time.
      total_s  arrival -> last token. queue_s + e2e_s. This is the number a
               user actually experiences, and the one most benchmarks omit.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    backend: str
    config_id: str

    ok: bool = True
    error: str | None = None
    error_kind: str | None = None

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_counts_estimated: bool = False
    """True when the server did not return a usage block and we estimated.
    Surfaced in the report, because an estimated throughput is not a measured
    one."""

    arrival_ts: float = 0.0
    dispatch_ts: float = 0.0
    first_token_ts: float | None = None
    end_ts: float = 0.0

    queue_s: float = 0.0
    ttft_s: float | None = None
    e2e_s: float = 0.0
    itl_mean_s: float | None = None
    itl_p95_s: float | None = None

    cache_hit: bool = False
    cache_kind: Literal["exact", "semantic", "miss"] = "miss"
    cache_similarity: float | None = None

    retries: int = 0
    is_warmup: bool = False

    @property
    def total_s(self) -> float:
        return self.queue_s + self.e2e_s

    @property
    def decode_s(self) -> float:
        if self.ttft_s is None:
            return 0.0
        return max(self.e2e_s - self.ttft_s, 0.0)

    @property
    def output_tokens_per_s(self) -> float | None:
        """Per-stream decode speed. Not the same thing as system throughput."""
        if self.completion_tokens <= 1 or self.decode_s <= 0:
            return None
        return (self.completion_tokens - 1) / self.decode_s


# --------------------------------------------------------------------------- #
# Load configuration
# --------------------------------------------------------------------------- #


class LoadConfig(BaseModel):
    """How to drive the backend.

    closed loop: a fixed number of virtual users, each sending the next request
    only when the previous one returns. This measures the system at a given
    concurrency and is what you want for "how fast is it when N users are
    hammering it".

    open loop: requests arrive on a schedule (Poisson by default) regardless of
    whether the server has kept up. This measures the system at a given offered
    load and is the only mode that exposes queueing collapse. Real traffic is
    open loop; nearly every student benchmark is closed loop and therefore
    cannot show the knee.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["closed", "open"] = "closed"
    concurrency: int = 1
    rate_qps: float | None = None
    distribution: Literal["poisson", "constant"] = "poisson"
    n_requests: int | None = 32
    duration_s: float | None = None
    warmup_requests: int = 4
    max_in_flight: int = 512
    seed: int = 7

    @model_validator(mode="after")
    def _check(self) -> LoadConfig:
        if self.mode == "open" and not self.rate_qps:
            raise ValueError("open-loop load requires rate_qps")
        if self.mode == "closed" and self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        # A duration without an explicit request count means "run for this
        # long". Without this, the n_requests default (32) silently ended every
        # open-loop point after 32 arrivals: 8 s at 4 req/s, before any queue
        # could build, which hid the saturation cliff the sweep exists to show.
        if self.duration_s is not None and "n_requests" not in self.model_fields_set:
            self.n_requests = None
        if self.n_requests is None and self.duration_s is None:
            raise ValueError("set n_requests or duration_s")
        return self


class SLO(BaseModel):
    """The service level objective a configuration is judged against.

    Goodput -- the share of requests that met the SLO -- is the metric that
    makes latency and throughput comparable. A config that doubles throughput
    while violating the SLO on 40% of requests has not doubled anything useful.
    """

    model_config = ConfigDict(extra="forbid")

    ttft_p95_s: float | None = 1.0
    total_p95_s: float | None = 8.0
    max_error_rate: float = 0.01

    def request_meets(self, r: GenResult) -> bool:
        if not r.ok:
            return False
        if self.ttft_p95_s is not None and (r.ttft_s is None or r.ttft_s > self.ttft_p95_s):
            return False
        return not (self.total_p95_s is not None and r.total_s > self.total_p95_s)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


class LatencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = 0
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["api", "self_hosted", "unknown"] = "unknown"
    usd_per_1k_requests: float | None = None
    usd_per_1m_output_tokens: float | None = None
    notes: str = ""


class RunSummary(BaseModel):
    """Everything we know about one configuration after measuring it."""

    model_config = ConfigDict(extra="forbid")

    config_id: str
    backend: str
    label: str = ""
    params: dict[str, Any] = Field(default_factory=dict)

    n_requests: int = 0
    n_ok: int = 0
    n_failed: int = 0
    error_rate: float = 0.0
    errors_by_kind: dict[str, int] = Field(default_factory=dict)

    duration_s: float = 0.0
    offered_qps: float | None = None
    achieved_qps: float = 0.0
    output_tokens_per_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    token_counts_estimated: bool = False

    ttft: LatencyStats = Field(default_factory=LatencyStats)
    e2e: LatencyStats = Field(default_factory=LatencyStats)
    total: LatencyStats = Field(default_factory=LatencyStats)
    queue: LatencyStats = Field(default_factory=LatencyStats)
    itl: LatencyStats = Field(default_factory=LatencyStats)

    goodput_ratio: float = 0.0
    slo: SLO | None = None

    cache_hit_rate: float = 0.0
    cache_kinds: dict[str, int] = Field(default_factory=dict)

    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    quality: dict[str, Any] | None = None


class QualitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_id: str
    n: int
    exact_match: float
    token_f1: float
    citation_grounding: float
    number_agreement: float
    composite: float
    composite_ci95: tuple[float, float] | None = None
    judge_score: float | None = None
    per_item: list[dict[str, Any]] = Field(default_factory=list)
