"""End to end: a YAML config in, a run directory with a report out.

This is the test that would catch the kind of breakage that matters -- a
sweep that runs but writes nothing reproducible."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from vantage.errors import ConfigError
from vantage.report.charts import render_all
from vantage.report.render import build_markdown, write_report
from vantage.sweep import execute_sweep, expand_runs

SWEEP = """
name: test-sweep
workload: {workload}
cooldown_s: 0
slo: {{ ttft_p95_s: 2.0, total_p95_s: 20.0 }}
defaults:
  max_tokens: 16
  load: {{ mode: closed, n_requests: 6, warmup_requests: 2, seed: 1 }}
runs:
  - id: fp16
    group: fp16
    label: fp16
    backend: {{ type: simulated, name: sim-a, decode_tokens_per_s: 900, max_batch: 8 }}
    matrix:
      load.concurrency: [1, 2, 4]
  - id: int4
    group: int4
    label: int4
    backend: {{ type: simulated, name: sim-b, decode_tokens_per_s: 1600, fidelity: 0.5, max_batch: 8 }}
    matrix:
      load.concurrency: [2]
quality:
  enabled: true
  baseline: fp16
  max_drop: 0.02
cache_analysis:
  enabled: true
  repeats: 3
  thresholds: [0.8, 0.95]
"""


def test_matrix_expansion_is_a_cartesian_product():
    specs = expand_runs(
        {
            "defaults": {"max_tokens": 32},
            "runs": [
                {
                    "id": "r",
                    "backend": "sim.fp16",
                    "matrix": {"load.concurrency": [1, 2], "backend.max_batch": [4, 8]},
                }
            ],
        }
    )
    assert len(specs) == 4
    assert {s.load.concurrency for s in specs} == {1, 2}
    assert {s.backend_overrides["max_batch"] for s in specs} == {4, 8}
    assert all("concurrency=" in s.id and "max_batch=" in s.id for s in specs)


def test_matrix_rejects_unscoped_keys():
    with pytest.raises(ConfigError):
        expand_runs({"runs": [{"backend": "sim.fp16", "matrix": {"concurrency.oops": [1]}}]})


def test_empty_sweep_is_an_error():
    with pytest.raises(ConfigError):
        expand_runs({"runs": []})


def test_full_sweep_writes_a_reproducible_run_directory(tmp_path, sample_workload, monkeypatch):
    workload_path = Path("data/workload.sample.jsonl").resolve()
    config = tmp_path / "sweep.yaml"
    config.write_text(SWEEP.format(workload=workload_path), encoding="utf-8")

    result = asyncio.run(
        execute_sweep(config, results_dir=tmp_path / "results", progress=False)
    )
    run_dir = result.store.dir

    assert len(result.summaries) == 4
    for name in ("manifest.json", "summaries.json", "raw.jsonl", "quality.json",
                 "quality_gate.json", "cache_sweep.json"):
        assert (run_dir / name).exists(), f"missing {name}"

    # Every measured request is in raw.jsonl, and nothing from the warm-up is.
    raw = [json.loads(line) for line in (run_dir / "raw.jsonl").read_text().splitlines()]
    assert len(raw) == sum(s.n_requests for s in result.summaries)
    assert not any(row["is_warmup"] for row in raw)

    # The deliberately broken config must be caught by the gate.
    verdicts = {c.candidate_id: c.verdict for c in result.comparisons}
    assert verdicts["int4"] == "fail"

    # And the report renders from what was saved.
    charts = render_all(result.summaries, run_dir / "charts",
                        cache_sweep=result.cache_sweep.to_dict())
    md, html = write_report(
        run_dir, result.manifest, result.summaries, charts,
        quality={k: v.model_dump() for k, v in result.quality.items()},
        gate=[c.to_dict() for c in result.comparisons],
        cache_sweep=result.cache_sweep.to_dict(),
    )
    assert md.exists() and html.exists()
    text = md.read_text()
    assert "## Findings" in text and "What this run does not establish" in text
    assert "<table>" in html.read_text()


def test_report_markdown_survives_missing_optional_sections(sample_workload):
    from vantage.models import RunSummary

    summaries = [RunSummary(config_id="a", backend="sim", label="a", n_ok=1, n_requests=1)]
    md = build_markdown({"name": "x"}, summaries, charts={})
    assert "# Vantage report: x" in md
