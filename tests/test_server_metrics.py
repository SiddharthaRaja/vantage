"""Server-side metrics and managed servers.

The end-to-end tests start tests/fake_vllm.py as a real subprocess, so the
whole path runs: the sweep starts a server, waits for health, drives load,
scrapes /metrics while it does, stops the server, starts the next one.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path

import httpx
import pytest

from vantage.errors import ConfigError
from vantage.metrics.server import ServerScraper, ServerTrace, metrics_url_for, parse_prometheus
from vantage.sweep import execute_sweep

ROOT = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).resolve().parent / "fake_vllm.py"

V1 = """
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 5.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.75
vllm:num_preemptions_total{engine="0",model_name="m"} 2.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 40.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="m"} 1.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 2.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 4.0
vllm:generation_tokens_total{engine="0",model_name="m"} 1000.0
"""


def test_parse_sums_label_sets_and_skips_comments_and_nan():
    text = 'a{x="1"} 2\na{x="2"} 3\n# a comment\nb 1e3\nc NaN\n'
    assert parse_prometheus(text) == {"a": 5.0, "b": 1000.0}


def test_trace_reads_v1_names_and_differences_counters():
    trace = ServerTrace(url="u")
    trace.add(0.0, parse_prometheus(V1))
    later = (V1.replace("num_preemptions_total{engine=\"0\",model_name=\"m\"} 2.0",
                        "num_preemptions_total{engine=\"0\",model_name=\"m\"} 7.0")
               .replace("hits_total{engine=\"0\",model_name=\"m\"} 40.0", "hits_total{engine=\"0\",model_name=\"m\"} 100.0")
               .replace("queries_total{engine=\"0\",model_name=\"m\"} 100.0", "queries_total{engine=\"0\",model_name=\"m\"} 200.0")
               .replace("seconds_sum{engine=\"0\",model_name=\"m\"} 2.0", "seconds_sum{engine=\"0\",model_name=\"m\"} 5.0")
               .replace("seconds_count{engine=\"0\",model_name=\"m\"} 4.0", "seconds_count{engine=\"0\",model_name=\"m\"} 10.0")
               .replace("generation_tokens_total{engine=\"0\",model_name=\"m\"} 1000.0",
                        "generation_tokens_total{engine=\"0\",model_name=\"m\"} 1500.0"))
    trace.add(10.0, parse_prometheus(later))
    s = trace.summary()
    assert s["waiting_max"] == 5.0 and s["kv_cache_usage_mean"] == 0.75
    assert s["preemptions"] == 5                       # this run only, not the server's lifetime
    assert s["prefix_cache_hit_rate"] == pytest.approx(0.6)
    assert s["server_ttft_mean_s"] == pytest.approx(0.5)  # (5 - 2) / (10 - 4)
    assert s["server_generation_tokens_per_s"] == pytest.approx(50.0)


def test_trace_reads_v0_names():
    trace = ServerTrace(url="u")
    trace.add(0.0, {"vllm:gpu_cache_usage_perc": 0.4, "vllm:num_preemptions": 1.0})
    trace.add(1.0, {"vllm:gpu_cache_usage_perc": 0.6, "vllm:num_preemptions": 3.0})
    s = trace.summary()
    assert s["kv_cache_usage_max"] == 0.6 and s["preemptions"] == 2


def test_metrics_url():
    assert metrics_url_for("http://localhost:8000/v1") == "http://localhost:8000/metrics"
    assert metrics_url_for("http://h:1/v1/") == "http://h:1/metrics"


def test_scraper_counts_failures_instead_of_raising():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500)
        return httpx.Response(200, text=V1)

    async def go() -> ServerTrace:
        async with ServerScraper("http://x/metrics", interval_s=0.01,
                                 transport=httpx.MockTransport(handler)) as sc:
            await asyncio.sleep(0.08)
        return sc.trace

    trace = asyncio.run(go())
    assert trace.errors == 1 and len(trace.samples) >= 3


# --------------------------------------------------------------------------- #
# End to end with a real (fake) server process
# --------------------------------------------------------------------------- #

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sweep(tmp_path: Path, port: int, servers: str, runs: str, extra: str = "") -> Path:
    cfg = f"""
name: managed
workload: {ROOT / "data" / "workload.sample.jsonl"}
cooldown_s: 0
slo: {{ ttft_p95_s: 5.0, total_p95_s: 20.0 }}
defaults:
  max_tokens: 6
  load: {{ mode: closed, concurrency: 6, n_requests: 12, warmup_requests: 0, seed: 1 }}
  backend:
    type: vllm
    name: fake
    model: fake-model
    api_key_env: VLLM_API_KEY
    base_url: http://127.0.0.1:{port}/v1
    scrape_metrics: true
    metrics_interval_s: 0.05
    pricing: {{ kind: self_hosted, gpu_hourly_usd: 0.35 }}
servers:
{servers}
runs:
{runs}
{extra}
"""
    path = tmp_path / "managed.yaml"
    path.write_text(cfg, encoding="utf-8")
    return path


def _server(name: str, port: int, flags: str) -> str:
    return (f"  {name}:\n"
            f"    command: {sys.executable} {FAKE} --port {port} --token-delay 0.04 {flags}\n"
            f"    health_url: http://127.0.0.1:{port}/v1/models\n"
            f"    startup_timeout_s: 20\n"
            f"    stop_grace_s: 0.2\n")


def test_sweep_restarts_server_per_profile_and_records_what_it_saw(tmp_path):
    port = _free_port()
    servers = _server("narrow", port, "--max-num-seqs 2") + _server("wide", port, "--max-num-seqs 8") \
        + _server("broken", port, "--fail")
    runs = ("  - { id: narrow, group: narrow, server: narrow }\n"
            "  - { id: wide, group: wide, server: wide }\n"
            "  - { id: broken, group: broken, server: broken }\n")
    result = asyncio.run(execute_sweep(_sweep(tmp_path, port, servers, runs), results_dir=tmp_path / "r"))

    assert [s.config_id for s in result.summaries] == ["narrow", "wide"]      # broken skipped, not fatal
    assert all(s.n_ok == 12 for s in result.summaries)
    narrow = result.engine_stats["narrow"]["server"]
    wide = result.engine_stats["wide"]["server"]
    assert narrow["running_max"] <= 2 < wide["running_max"]   # the cap took effect after the restart
    assert narrow["server_queue_mean_s"] > wide["server_queue_mean_s"]
    assert narrow["prefix_cache_hit_rate"] == pytest.approx(0.5, abs=0.05)
    assert result.summaries[0].params["server"] == "narrow"

    servers_json = json.loads((result.store.dir / "servers.json").read_text())
    assert set(servers_json["started"]) == {"narrow", "wide"}
    assert "broken" in servers_json["failed"]
    assert "--max-num-seqs 2" in servers_json["started"]["narrow"]["command"]
    assert (result.store.dir / "server_narrow.log").exists()
    # every server is stopped at the end: nothing answers on the port
    with socket.socket() as s, pytest.raises(ConnectionRefusedError):
        s.connect(("127.0.0.1", port))


def test_managed_servers_and_quality_gate_are_refused_together(tmp_path):
    port = _free_port()
    path = _sweep(tmp_path, port, _server("a", port, ""), "  - { id: a, server: a }\n",
                  extra="quality: { enabled: true, baseline: a }")
    with pytest.raises(ConfigError, match="quality gate"):
        asyncio.run(execute_sweep(path, results_dir=tmp_path / "r"))


def test_unknown_server_name_is_a_config_error(tmp_path):
    port = _free_port()
    path = _sweep(tmp_path, port, _server("a", port, ""), "  - { id: b, server: nope }\n")
    with pytest.raises(ConfigError, match="nope"):
        asyncio.run(execute_sweep(path, results_dir=tmp_path / "r"))


def test_report_shows_what_the_server_saw(tmp_path):
    from vantage.report.charts import render_all
    from vantage.report.render import write_report

    port = _free_port()
    servers = _server("capped", port, "--max-num-seqs 4") + _server("open", port, "--max-num-seqs 16")
    runs = ("  - { id: capped, group: capped, server: capped, matrix: { load.concurrency: [2, 4, 8] } }\n"
            "  - { id: open, group: open, server: open, matrix: { load.concurrency: [2, 4, 8] } }\n")
    result = asyncio.run(execute_sweep(_sweep(tmp_path, port, servers, runs), results_dir=tmp_path / "r"))
    charts = render_all(result.summaries, result.store.charts_dir)
    assert "server_state_vs_load" in charts
    md_path, _ = write_report(result.store.dir, result.manifest, result.summaries, charts)
    md = md_path.read_text()
    assert "## What the server saw" in md and "### Side by side" in md
    capped8 = next(s for s in result.summaries if s.config_id == "capped[concurrency=8]")
    assert capped8.server["waiting_max"] >= 2          # 8 users, 4 slots
    assert json.loads((result.store.dir / "summaries.json").read_text())[0]["server"]["n_scrapes"] > 0


def test_server_started_with_an_api_key_is_still_reached(tmp_path, monkeypatch):
    """Regression: VLLM_API_KEY set for the client is inherited by the server,
    which then requires it. The health check must send it, or it waits forever."""
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    port = _free_port()
    result = asyncio.run(execute_sweep(
        _sweep(tmp_path, port, _server("keyed", port, ""), "  - { id: keyed, server: keyed }\n"),
        results_dir=tmp_path / "r"))
    assert result.summaries[0].n_ok == 12          # health check and requests both authorised


def test_a_server_that_refuses_us_fails_fast(tmp_path, monkeypatch):
    """Up but answering 401 is reported at once, not after the start-up timeout."""
    import time as _time

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    port = _free_port()
    servers = _server("keyed", port, "").replace("    startup_timeout_s: 20\n",
                                                 "    startup_timeout_s: 60\n    env: { VLLM_API_KEY: s }\n")
    path = _sweep(tmp_path, port, servers, "  - { id: keyed, server: keyed }\n")
    # The profile's key reaches the server; hide it from the health check by
    # overriding after the fact: the client side has no key at all.
    from vantage import servers as srv
    original = srv.ManagedServer.start

    async def start_without_key(self):
        self.profile.env = {**self.profile.env}
        key = self.profile.env.pop("VLLM_API_KEY")
        monkeypatch.setenv("VANTAGE_TEST_SERVER_KEY", key)
        cmd = self.profile.command
        self.profile.command = f"env VLLM_API_KEY={key} {cmd}"
        return await original(self)

    monkeypatch.setattr(srv.ManagedServer, "start", start_without_key)
    t0 = _time.perf_counter()
    with pytest.raises(ConfigError, match="no server started"):
        asyncio.run(execute_sweep(path, results_dir=tmp_path / "r"))
    assert _time.perf_counter() - t0 < 30            # not the 60 s timeout
    failed = json.loads(next((tmp_path / "r").rglob("servers.json")).read_text())["failed"]
    assert "answers 401" in failed["keyed"]
