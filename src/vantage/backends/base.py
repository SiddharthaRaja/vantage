"""The backend interface.

Everything downstream -- the load generator, the metrics, the quality harness,
the report -- talks to this and only this. That is the whole reason the project
can compare an OpenAI endpoint, a local vLLM server and a fine-tuned model on
one chart: they are three implementations of one twelve-line interface.

Subclasses implement `_stream()`. The base class owns all timing, because
timing that is reimplemented per backend is timing you cannot trust.
"""

from __future__ import annotations

import abc
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from vantage.errors import RequestFailed
from vantage.models import GenRequest, GenResult


@dataclass
class StreamChunk:
    """One server-sent delta."""

    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


@dataclass
class Pricing:
    """Either API list pricing or an amortised self-hosting rate. Both are
    reduced to dollars in the cost model; see metrics/cost.py."""

    kind: str = "unknown"  # "api" | "self_hosted"
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None
    gpu_hourly_usd: float | None = None
    n_gpus: int = 1
    note: str = ""


@dataclass
class Health:
    ok: bool
    detail: str = ""
    model: str | None = None


class Backend(abc.ABC):
    """One way of running an LLM, behind one interface."""

    kind: str = "base"

    def __init__(self, name: str, config_id: str | None = None, **params: Any):
        self.name = name
        self.config_id = config_id or name
        self.params = params

    # -- subclass contract -------------------------------------------------- #

    @abc.abstractmethod
    def _stream(self, req: GenRequest) -> AsyncIterator[StreamChunk]:
        """Yield StreamChunks as they arrive. Must not swallow errors."""

    async def health(self) -> Health:
        return Health(ok=True, detail="no health check implemented")

    def pricing(self) -> Pricing:
        return Pricing()

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Backend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- timing, owned centrally -------------------------------------------- #

    async def generate(self, req: GenRequest, arrival_ts: float | None = None) -> GenResult:
        """Run one request and return a fully-timed result.

        Never raises for a per-request failure: a failed request is a data
        point, and a harness that crashes on the first HTTP 429 cannot measure
        a system under load. Failures are recorded and surface in error_rate.
        """
        dispatch = time.perf_counter()
        arrival = arrival_ts if arrival_ts is not None else dispatch
        result = GenResult(
            request_id=req.id,
            backend=self.name,
            config_id=self.config_id,
            arrival_ts=arrival,
            dispatch_ts=dispatch,
            queue_s=max(dispatch - arrival, 0.0),
        )

        first_token_ts: float | None = None
        token_times: list[float] = []
        pieces: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            async for chunk in self._stream(req):
                now = time.perf_counter()
                if chunk.text:
                    if first_token_ts is None:
                        first_token_ts = now
                    token_times.append(now)
                    pieces.append(chunk.text)
                if chunk.prompt_tokens is not None:
                    prompt_tokens = chunk.prompt_tokens
                if chunk.completion_tokens is not None:
                    completion_tokens = chunk.completion_tokens
        except RequestFailed as exc:
            end = time.perf_counter()
            result.ok = False
            result.error = str(exc)
            result.error_kind = f"http_{exc.status}" if exc.status else "request_failed"
            result.end_ts = end
            result.e2e_s = end - dispatch
            return result
        except Exception as exc:  # noqa: BLE001 - deliberately broad: see docstring
            end = time.perf_counter()
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"
            result.error_kind = type(exc).__name__
            result.end_ts = end
            result.e2e_s = end - dispatch
            return result

        end = time.perf_counter()
        text = "".join(pieces)

        estimated = False
        if completion_tokens is None:
            completion_tokens = max(len(token_times), estimate_tokens(text))
            estimated = True
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens((req.system or "") + req.prompt)
            estimated = True

        itl = [b - a for a, b in zip(token_times, token_times[1:], strict=False)]

        result.text = text
        result.prompt_tokens = prompt_tokens
        result.completion_tokens = completion_tokens
        result.token_counts_estimated = estimated
        result.first_token_ts = first_token_ts
        result.end_ts = end
        result.ttft_s = (first_token_ts - dispatch) if first_token_ts is not None else None
        result.e2e_s = end - dispatch
        if itl:
            arr = sorted(itl)
            result.itl_mean_s = sum(itl) / len(itl)
            result.itl_p95_s = arr[min(len(arr) - 1, int(0.95 * len(arr)))]
        return result


def estimate_tokens(text: str) -> int:
    """Rough token count used only when a server returns no usage block.

    Uses tiktoken when it is installed, else ~4 characters per token, which is
    close enough for English prose and wrong enough for code that we flag every
    estimated figure in the report rather than quietly charting it.
    """
    if not text:
        return 0
    try:  # pragma: no cover - optional dependency
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, round(len(text) / 4))
