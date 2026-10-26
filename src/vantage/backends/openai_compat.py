"""Backend for any server speaking the OpenAI chat-completions protocol.

That one adapter covers: OpenAI itself, vLLM (`--api-server`), SGLang, HF TGI's
OpenAI route, llama.cpp's server, Ollama (/v1), LM Studio, Together, Groq,
Fireworks and most managed endpoints. This is why the project can swap the
generation model without touching anything else.

Two deliberate choices worth defending in an interview:

1. Streaming is always on, even when nobody reads the stream. Without streaming
   there is no time-to-first-token, and TTFT is the single most important
   latency number for a chat-shaped workload.
2. Retries are OFF by default. A retry hides a failure and inflates the latency
   of the request that succeeded. If you want retries you must ask for them,
   and the retry count is recorded on the result.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from vantage.backends.base import Backend, Health, Pricing, StreamChunk
from vantage.errors import RequestFailed


class OpenAICompatBackend(Backend):
    kind = "openai_compat"

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 10.0,
        max_retries: int = 0,
        retry_backoff_s: float = 0.5,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        pricing: dict[str, Any] | None = None,
        config_id: str | None = None,
        scrape_metrics: bool = False,
        metrics_url: str | None = None,
        metrics_interval_s: float = 1.0,
        **params: Any,
    ):
        super().__init__(name=name, config_id=config_id, model=model, **params)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env, "") or "EMPTY"
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.extra_body = extra_body or {}
        # vLLM's Prometheus endpoint, polled during each run (metrics/server.py).
        self.metrics_url = (metrics_url or _metrics_url(self.base_url)) if scrape_metrics else None
        self.metrics_interval_s = float(metrics_interval_s)
        self._pricing_cfg = pricing or {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **(headers or {}),
            },
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
        )

    # ------------------------------------------------------------------ #

    def _payload(self, req) -> dict[str, Any]:
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": True,
            # Ask servers that support it to send a final usage block. vLLM and
            # OpenAI both do; servers that do not simply ignore the field and we
            # fall back to estimating, which the report flags.
            "stream_options": {"include_usage": True},
        }
        if req.stop:
            body["stop"] = req.stop
        body.update(self.extra_body)
        return body

    async def _stream(self, req) -> AsyncIterator[StreamChunk]:
        attempt = 0
        while True:
            try:
                async for chunk in self._stream_once(req):
                    yield chunk
                return
            except RequestFailed as exc:
                attempt += 1
                if not exc.retryable or attempt > self.max_retries:
                    raise
                import asyncio

                await asyncio.sleep(self.retry_backoff_s * (2 ** (attempt - 1)))

    async def _stream_once(self, req) -> AsyncIterator[StreamChunk]:
        async with self._client.stream(
            "POST", "/chat/completions", json=self._payload(req)
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")[:400]
                raise RequestFailed(
                    f"{response.status_code} from {self.base_url}: {body}",
                    status=response.status_code,
                    retryable=response.status_code in {408, 409, 429, 500, 502, 503, 504},
                )
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue

                usage = payload.get("usage") or {}
                choices = payload.get("choices") or []
                delta_text = ""
                finish = None
                if choices:
                    delta = choices[0].get("delta") or {}
                    delta_text = delta.get("content") or ""
                    finish = choices[0].get("finish_reason")

                yield StreamChunk(
                    text=delta_text,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=finish,
                )

    # ------------------------------------------------------------------ #

    async def health(self) -> Health:
        try:
            r = await self._client.get("/models", timeout=10.0)
            if r.status_code >= 400:
                return Health(ok=False, detail=f"GET /models -> {r.status_code}")
            names = [m.get("id") for m in (r.json().get("data") or [])]
            if names and self.model not in names:
                return Health(
                    ok=False,
                    detail=f"model '{self.model}' not served here; available: {names[:5]}",
                )
            return Health(ok=True, detail=f"{len(names)} model(s) served", model=self.model)
        except Exception as exc:  # noqa: BLE001
            return Health(ok=False, detail=f"{type(exc).__name__}: {exc}")

    def pricing(self) -> Pricing:
        cfg = self._pricing_cfg
        if not cfg:
            return Pricing()
        return Pricing(
            kind=cfg.get("kind", "api"),
            input_usd_per_mtok=cfg.get("input_usd_per_mtok"),
            output_usd_per_mtok=cfg.get("output_usd_per_mtok"),
            gpu_hourly_usd=cfg.get("gpu_hourly_usd"),
            n_gpus=cfg.get("n_gpus", 1),
            note=cfg.get("note", ""),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _metrics_url(base_url: str) -> str:
    # imported here: vantage.metrics imports the workload package, which imports backends
    from vantage.metrics.server import metrics_url_for

    return metrics_url_for(base_url)
