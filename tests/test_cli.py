"""The CLI is the only interface most people will ever touch, so it gets the
same treatment as the library: exit codes, not just output."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from vantage.cli import app

runner = CliRunner()


def test_doctor_succeeds_on_a_backend_that_is_up():
    result = runner.invoke(app, ["doctor", "-b", "sim.fp16"])
    assert result.exit_code == 0


def test_doctor_fails_loudly_on_a_backend_that_does_not_exist():
    result = runner.invoke(app, ["doctor", "-b", "nope.not-a-config"])
    assert result.exit_code == 1


def test_workload_prints_the_shape_of_the_data():
    result = runner.invoke(app, ["workload", "data/workload.sample.jsonl"])
    assert result.exit_code == 0
    stats = json.loads(result.stdout)
    assert stats["n_items"] == 60
    assert stats["n_with_reference"] == 60


def test_bench_runs_and_emits_a_summary():
    result = runner.invoke(
        app,
        ["bench", "-b", "sim.fp16", "-c", "4", "-n", "8", "--max-tokens", "16", "--warmup", "2"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["n_requests"] == 8
    assert payload["n_ok"] == 8
    assert payload["achieved_qps"] > 0


def test_quality_scores_the_holdout_set():
    result = runner.invoke(app, ["quality", "-b", "sim.fp16", "--limit", "6"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["n"] == 6
    assert 0.0 <= payload["composite"] <= 1.0


def test_cache_sim_recommends_a_threshold(tmp_path):
    out = tmp_path / "sweep.json"
    result = runner.invoke(app, ["cache-sim", "--repeats", "3", "--out", str(out)])
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["recommended_threshold"] is not None
    assert len(payload["points"]) >= 5
