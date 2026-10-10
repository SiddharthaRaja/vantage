from __future__ import annotations

import asyncio

import httpx
import pytest

from vantage.backends.openai_compat import OpenAICompatBackend
from vantage.backends.registry import build_backend
from vantage.backends.simulated import SimulatedBackend
from vantage.errors import ConfigError
from vantage.models import GenRequest


def _req(i: int = 0, **kw) -> GenRequest:
    return GenRequest(id=f"r{i}", prompt="Sources: [1] text here\n\nQuestion: what?", **kw)


# -- simulated --------------------------------------------------------------- #


def test_simulated_backend_times_are_ordered_and_tokens_counted():
    result = asyncio.run(SimulatedBackend().generate(_req(max_tokens=20)))
    assert result.ok
    assert 0 < result.ttft_s < result.e2e_s
    assert result.completion_tokens == 20
    assert not result.token_counts_estimated
    assert result.itl_mean_s is not None


def test_simulated_throughput_saturates_as_the_batch_grows():
    sim = SimulatedBackend()
    single = sim.aggregate_throughput(1)
    many = sim.aggregate_throughput(64)
    assert many > single
    # Saturating, not linear: 64x the batch is far less than 64x the throughput.
    assert many < single * 64
    assert sim.per_seq_decode_rate(64) < sim.per_seq_decode_rate(1)


def test_simulated_backend_queues_beyond_max_batch():
    async def run(max_batch: int):
        sim = SimulatedBackend(max_batch=max_batch, jitter=0.0)
        results = await asyncio.gather(
            *[sim.generate(_req(i, max_tokens=16)) for i in range(16)]
        )
        return max(r.ttft_s for r in results), sim

    narrow_ttft, sim = asyncio.run(run(2))
    wide_ttft, _ = asyncio.run(run(16))
    assert narrow_ttft > wide_ttft
    assert sim.engine_stats()["peak_batch"] <= 2


def test_failures_are_recorded_not_raised():
    async def run():
        sim = SimulatedBackend(fail_rate=1.0)
        return await asyncio.gather(*[sim.generate(_req(i)) for i in range(4)])

    results = asyncio.run(run())
    assert all(not r.ok for r in results)
    assert all(r.error_kind == "http_503" for r in results)


def test_reference_answers_only_when_asked_for():
    plain = asyncio.run(SimulatedBackend().generate(_req(max_tokens=12)))
    cited = asyncio.run(
        SimulatedBackend(fidelity=1.0, style_noise=0.0).generate(
            _req(max_tokens=64, meta={"want_reference_answer": True,
                                      "reference_answer": "Recovery was 94.0% [1]."})
        )
    )
    assert "Recovery was 94.0% [1]." not in plain.text
    assert cited.text == "Recovery was 94.0% [1]."


# -- OpenAI-compatible streaming --------------------------------------------- #

SSE = (
    'data: {"choices":[{"delta":{"content":"Mean"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":" recovery"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":" was 94%"},"finish_reason":"stop"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":812,"completion_tokens":7}}\n\n'
    "data: [DONE]\n\n"
)


def _backend_with(handler) -> OpenAICompatBackend:
    backend = OpenAICompatBackend(name="mock", model="m", base_url="http://test/v1")
    backend._client = httpx.AsyncClient(
        base_url="http://test/v1", transport=httpx.MockTransport(handler)
    )
    return backend


def test_sse_stream_is_parsed_and_usage_is_preferred_over_estimates():
    async def run():
        backend = _backend_with(
            lambda request: httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"})
        )
        async with backend:
            return await backend.generate(_req())

    result = asyncio.run(run())
    assert result.ok
    assert result.text == "Mean recovery was 94%"
    assert (result.prompt_tokens, result.completion_tokens) == (812, 7)
    assert not result.token_counts_estimated
    assert result.ttft_s is not None and result.ttft_s <= result.e2e_s


def test_missing_usage_block_is_estimated_and_flagged():
    body = SSE.replace('data: {"choices":[],"usage":{"prompt_tokens":812,"completion_tokens":7}}\n\n', "")

    async def run():
        backend = _backend_with(lambda request: httpx.Response(200, text=body))
        async with backend:
            return await backend.generate(_req())

    result = asyncio.run(run())
    assert result.token_counts_estimated
    assert result.completion_tokens >= 3


def test_http_error_becomes_a_failed_result_with_the_status():
    async def run():
        backend = _backend_with(lambda request: httpx.Response(429, text="slow down"))
        async with backend:
            return await backend.generate(_req())

    result = asyncio.run(run())
    assert not result.ok
    assert result.error_kind == "http_429"
    assert "429" in (result.error or "")


def test_retries_are_off_by_default_and_counted_when_enabled():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text=SSE)

    async def run(max_retries: int):
        calls["n"] = 0
        backend = _backend_with(handler)
        backend.max_retries = max_retries
        backend.retry_backoff_s = 0.001
        async with backend:
            return await backend.generate(_req())

    assert not asyncio.run(run(0)).ok
    assert calls["n"] == 1
    assert asyncio.run(run(3)).ok
    assert calls["n"] == 3


# -- registry ---------------------------------------------------------------- #


def test_registry_builds_from_config_name():
    backend = build_backend("sim.fp16")
    assert isinstance(backend, SimulatedBackend)
    assert backend.r1 == 320


def test_registry_rejects_unknown_types_with_a_useful_message():
    with pytest.raises(ConfigError) as exc:
        build_backend({"type": "not-a-backend", "name": "x"})
    assert "not-a-backend" in str(exc.value)
