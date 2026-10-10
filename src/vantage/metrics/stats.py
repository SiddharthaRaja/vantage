"""Summary statistics, with the two things most benchmark scripts skip:
a stated percentile convention, and uncertainty.

Percentiles use linear interpolation between order statistics (numpy's
default, and what wrk/k6/vegeta report). Stating that matters because the
"nearest rank" convention gives visibly different p99s on short runs, and a
reader cannot compare your numbers to anyone else's without knowing which you
used.

Confidence intervals are bootstrapped rather than assumed normal. Latency
distributions are heavy-tailed and right-skewed; a mean +/- 1.96 sigma on 200
samples of TTFT is not a meaningful interval. If two configurations' intervals
overlap, you have not shown a difference, and the report says so.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from vantage.models import LatencyStats


def percentiles(values: Sequence[float], qs: Sequence[float] = (50, 90, 95, 99)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {f"p{int(q)}": float("nan") for q in qs}
    return {f"p{int(q)}": float(np.percentile(arr, q)) for q in qs}


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 13,
) -> tuple[float, float]:
    """Percentile bootstrap. Deterministic given the seed, so a report
    regenerated from the same raw data produces identical intervals."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 3:
        v = float(stat(arr)) if arr.size else float("nan")
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = np.apply_along_axis(stat, 1, arr[idx])
    lo = float(np.percentile(stats, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(stats, 100 * (1 - (1 - confidence) / 2)))
    return (lo, hi)


def summarize_latency(values: Sequence[float], with_ci: bool = True) -> LatencyStats:
    vals = [v for v in values if v is not None]
    if not vals:
        return LatencyStats(n=0)
    arr = np.asarray(vals, dtype=float)
    p = percentiles(arr)
    ci = bootstrap_ci(arr) if with_ci and arr.size >= 3 else (None, None)
    return LatencyStats(
        n=int(arr.size),
        mean=float(arr.mean()),
        p50=p["p50"],
        p90=p["p90"],
        p95=p["p95"],
        p99=p["p99"],
        max=float(arr.max()),
        ci95_low=ci[0],
        ci95_high=ci[1],
    )


def knee_point(x: Sequence[float], y: Sequence[float]) -> int | None:
    """Index of the 'knee' of a saturating curve, by maximum distance from the
    straight line joining its endpoints (the standard Kneedle construction).

    This is how the report names the concurrency at which throughput stops
    improving -- the number you actually want when sizing a deployment, and the
    number you cannot get by eyeballing a chart in a slide.
    """
    if len(x) < 3 or len(x) != len(y):
        return None
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    xs_n = (xs - xs.min()) / (np.ptp(xs) or 1)
    ys_n = (ys - ys.min()) / (np.ptp(ys) or 1)
    p0 = np.array([xs_n[0], ys_n[0]])
    p1 = np.array([xs_n[-1], ys_n[-1]])
    line = p1 - p0
    norm = np.linalg.norm(line) or 1.0
    def _perp_distance(px: float, py: float) -> float:
        # 2-D cross product magnitude; numpy 2.0 removed np.cross for 2-D input.
        vx, vy = px - p0[0], py - p0[1]
        return float(abs(line[0] * vy - line[1] * vx) / norm)

    dists = [_perp_distance(px, py) for px, py in zip(xs_n, ys_n, strict=True)]
    return int(np.argmax(dists))


def relative_change(new: float, old: float) -> float:
    if old == 0:
        return float("nan")
    return (new - old) / old
