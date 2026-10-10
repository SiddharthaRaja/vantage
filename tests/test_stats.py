"""Statistics are load-bearing: if these are wrong every chart is wrong."""

from __future__ import annotations

import numpy as np

from vantage.metrics.stats import (
    bootstrap_ci,
    knee_point,
    percentiles,
    relative_change,
    summarize_latency,
)


def test_percentiles_match_numpy_convention():
    values = list(range(1, 101))
    p = percentiles(values)
    assert p["p50"] == np.percentile(values, 50)
    assert p["p99"] == np.percentile(values, 99)


def test_percentiles_on_empty_input_are_nan_not_zero():
    # Zero would silently look like a very fast server.
    p = percentiles([])
    assert all(v != v for v in p.values())


def test_summarize_latency_reports_n_and_max():
    stats = summarize_latency([0.1, 0.2, 0.3, 10.0])
    assert stats.n == 4
    assert stats.max == 10.0
    assert stats.p50 < stats.p95 <= stats.max


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    values = rng.lognormal(mean=0.0, sigma=0.4, size=500).tolist()
    lo, hi = bootstrap_ci(values)
    assert lo < float(np.mean(values)) < hi
    assert (lo, hi) == bootstrap_ci(values)


def test_bootstrap_ci_degenerates_gracefully_on_tiny_samples():
    assert bootstrap_ci([1.0]) == (1.0, 1.0)


def test_knee_point_finds_the_bend_of_a_saturating_curve():
    x = [1, 2, 4, 8, 16, 32, 64]
    y = [100, 200, 390, 700, 900, 960, 975]
    k = knee_point(x, y)
    assert k is not None
    assert x[k] in (4, 8, 16)


def test_knee_point_needs_at_least_three_points():
    assert knee_point([1, 2], [1, 2]) is None


def test_relative_change():
    assert relative_change(110, 100) == 0.1
    assert relative_change(1, 0) != relative_change(1, 0)  # NaN
