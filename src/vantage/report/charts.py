"""Charts.

Rules followed here, because a benchmark chart that misleads is worse than no
chart:

* One y-axis per chart, always. Two scales on one plot is the most common way
  to imply a relationship that is not in the data, and throughput and latency
  are exactly the pair people do it with.
* Series are distinguished by colour *and* by marker/line style *and* direct
  labels, so the chart survives greyscale printing and colour-blind readers.
  The palette is a fixed four-slot categorical set, validated for CVD
  separation rather than chosen by eye.
* Log x-axis for concurrency and load sweeps, because they are geometric.
* The knee of the throughput curve is computed and annotated, not eyeballed.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from vantage.metrics.stats import knee_point  # noqa: E402
from vantage.models import RunSummary  # noqa: E402

# Validated categorical palette (CVD-separated, fixed order -- never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "figure.dpi": 140,
    }
)


def _style(ax, title: str, xlabel: str, ylabel: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", pad=26 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color=INK_SOFT,
            va="bottom",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _group(summaries: list[RunSummary], mode: str | None = None) -> OrderedDict[str, list[RunSummary]]:
    out: OrderedDict[str, list[RunSummary]] = OrderedDict()
    for s in summaries:
        if mode and s.params.get("mode") != mode:
            continue
        key = str(s.params.get("group") or s.label or s.config_id)
        out.setdefault(key, []).append(s)
    return out


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #


def throughput_vs_concurrency(summaries: list[RunSummary], out: Path) -> Path | None:
    groups = {
        k: sorted([s for s in v if s.params.get("concurrency")], key=lambda s: s.params["concurrency"])
        for k, v in _group(summaries, mode="closed").items()
    }
    # A two-point "curve" is a line segment, and a line segment implies a
    # trend that was not measured. Curves need at least three points here.
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if not groups:
        return None

    label_inline = len(groups) <= 4
    all_x: set[int] = set()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, (name, runs) in enumerate(groups.items()):
        xs = [r.params["concurrency"] for r in runs]
        ys = [r.output_tokens_per_s for r in runs]
        all_x.update(xs)
        colour = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], markersize=6, linewidth=2,
                color=colour, label=name)
        if label_inline:
            ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, 0),
                        fontsize=9, color=colour, va="center")

        # A knee is only meaningful on a curve with enough points to have a
        # shape. On four points, "the knee" is an artefact of the sampling.
        k = knee_point(xs, ys) if len(xs) >= 5 else None
        if k is not None and 0 < k < len(xs) - 1:
            ax.scatter([xs[k]], [ys[k]], s=140, facecolors="none", edgecolors=colour,
                       linewidths=1.6, zorder=5)
            ax.annotate(
                f"knee c={xs[k]}",
                (xs[k], ys[k]),
                textcoords="offset points",
                xytext=(0, -24),
                fontsize=8.5,
                color=colour,
                ha="center",
            )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xticks(sorted(all_x))
    _style(
        ax,
        "Throughput saturates, then stops paying",
        "concurrent requests (log scale)",
        "output tokens / second",
        "closed loop; the circled point is the computed knee of each curve",
    )
    ax.legend(loc="upper left")
    ax.margins(x=0.14)
    return _save(fig, out)


def latency_vs_concurrency(summaries: list[RunSummary], out: Path) -> Path | None:
    groups = {
        k: sorted([s for s in v if s.params.get("concurrency")], key=lambda s: s.params["concurrency"])
        for k, v in _group(summaries, mode="closed").items()
    }
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    slo = next((s.slo for s in summaries if s.slo), None)
    all_x: set[int] = set()
    for i, (name, runs) in enumerate(groups.items()):
        xs = [r.params["concurrency"] for r in runs]
        all_x.update(xs)
        colour = SERIES[i % len(SERIES)]
        p50 = [r.total.p50 for r in runs]
        p95 = [r.total.p95 for r in runs]
        ax.plot(xs, p95, marker=MARKERS[i % len(MARKERS)], markersize=6, linewidth=2,
                color=colour, label=f"{name} p95")
        ax.plot(xs, p50, marker=MARKERS[i % len(MARKERS)], markersize=5, linewidth=1.6,
                linestyle="--", color=colour, alpha=0.75, label=f"{name} p50")

    if slo and slo.total_p95_s:
        ax.axhline(slo.total_p95_s, color="#e34948", linewidth=1.4, linestyle=":")
        ax.annotate(f"SLO p95 = {slo.total_p95_s:g}s", (ax.get_xlim()[0], slo.total_p95_s),
                    textcoords="offset points", xytext=(4, 5), fontsize=8.5, color="#e34948")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_x))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _style(
        ax,
        "Latency is what you pay for throughput",
        "concurrent requests (log scale)",
        "end-to-end latency, seconds (log scale)",
        "solid = p95, dashed = p50; both axes log",
    )
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, out)


def latency_vs_offered_load(summaries: list[RunSummary], out: Path) -> Path | None:
    runs = sorted(
        [s for s in summaries if s.params.get("mode") == "open" and s.offered_qps],
        key=lambda s: s.offered_qps or 0,
    )
    if len(runs) < 3:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = [r.offered_qps for r in runs]
    ax.plot(xs, [r.total.p50 for r in runs], marker="o", markersize=6, linewidth=1.8,
            linestyle="--", color=SERIES[0], label="p50")
    ax.plot(xs, [r.total.p95 for r in runs], marker="s", markersize=6, linewidth=2.2,
            color=SERIES[1], label="p95")
    ax.plot(xs, [r.total.p99 for r in runs], marker="^", markersize=6, linewidth=1.8,
            color=SERIES[3], label="p99")

    achieved = [r.achieved_qps for r in runs]
    saturated = next((x for x, a in zip(xs, achieved, strict=True) if a < 0.93 * x), None)
    if saturated:
        ax.axvline(saturated, color=INK_SOFT, linewidth=1.2, linestyle=":")
        ax.annotate(f"server stops keeping up\nat ~{saturated:g} req/s",
                    (saturated, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(-8, -30), ha="right", fontsize=8.5, color=INK_SOFT)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _style(
        ax,
        "Open loop: the cliff a concurrency sweep cannot show",
        "offered load (requests / second, Poisson arrivals)",
        "end-to-end latency, seconds (log scale)",
        "measured from arrival, so client-side queueing is included",
    )
    ax.legend(loc="upper left")
    return _save(fig, out)


def latency_percentiles(summaries: list[RunSummary], out: Path, concurrency: int | None = None) -> Path | None:
    import numpy as np

    runs = [s for s in summaries if s.params.get("mode") == "closed" and s.ttft.p50]
    if concurrency:
        runs = [s for s in runs if s.params.get("concurrency") == concurrency]
    else:
        counts: dict[Any, int] = {}
        for s in runs:
            counts[s.params.get("concurrency")] = counts.get(s.params.get("concurrency"), 0) + 1
        if not counts:
            return None
        concurrency = max(counts, key=lambda k: counts[k])
        runs = [s for s in runs if s.params.get("concurrency") == concurrency]
    if len(runs) < 2:
        return None

    labels = [s.label for s in runs]
    series = {"p50": [s.ttft.p50 for s in runs], "p95": [s.ttft.p95 for s in runs],
              "p99": [s.ttft.p99 for s in runs]}
    x = np.arange(len(runs))
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (name, values) in enumerate(series.items()):
        bars = ax.bar(x + (i - 1) * (width + 0.015), values, width, label=name,
                      color=SERIES[i], edgecolor=SURFACE, linewidth=1.5)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK_SOFT, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _style(
        ax,
        f"Time to first token at concurrency {concurrency}",
        "",
        "TTFT, seconds",
        "the tail is the number users notice; the mean is the number that flatters you",
    )
    ax.legend(loc="upper right")
    return _save(fig, out)


def cost_per_1k(summaries: list[RunSummary], out: Path) -> Path | None:
    seen: dict[str, RunSummary] = {}
    for s in summaries:
        if s.cost.usd_per_1k_requests is None:
            continue
        key = str(s.params.get("group") or s.label)
        best = seen.get(key)
        if best is None or (s.cost.usd_per_1k_requests or 9e9) < (best.cost.usd_per_1k_requests or 9e9):
            seen[key] = s
    if len(seen) < 2:
        return None

    rows = sorted(seen.items(), key=lambda kv: kv[1].cost.usd_per_1k_requests or 0)
    labels = [f"{k}" for k, _ in rows]
    values = [v.cost.usd_per_1k_requests or 0 for _, v in rows]

    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(rows) + 2.0))
    bars = ax.barh(labels, values, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, height=0.62)
    ax.bar_label(bars, fmt="$%.3f", fontsize=9, color=INK_SOFT, padding=4)
    ax.invert_yaxis()
    _style(
        ax,
        "Cost per 1,000 requests at the best measured utilisation",
        "USD per 1,000 requests",
        "",
        "self-hosted figures are utilisation-dependent: an idle GPU bills the same as a busy one",
    )
    ax.margins(x=0.16)
    return _save(fig, out)


def quality_vs_latency(summaries: list[RunSummary], out: Path) -> Path | None:
    """One point per configuration, at its best operating point.

    "Best" means the highest sustained request rate at which it still met the
    SLO for at least 95% of requests -- plotting every concurrency level would
    put six copies of the same model on the chart and say nothing extra.
    Colour follows the configuration, never the point's rank.
    """
    groups: dict[str, list[RunSummary]] = {}
    for s in summaries:
        if not s.quality or not s.total.p95 or s.params.get("mode") != "closed":
            continue
        groups.setdefault(str(s.params.get("group") or s.label), []).append(s)
    if len(groups) < 2:
        return None

    points: list[tuple[float, float, str, str]] = []
    for name, runs in groups.items():
        within = [r for r in runs if r.goodput_ratio >= 0.95]
        pick = max(within or runs, key=lambda r: r.achieved_qps)
        points.append(
            (
                pick.total.p95 or 0.0,
                pick.quality["composite"],
                f"{name}  ({pick.achieved_qps:.0f} req/s at c={pick.params.get('concurrency')})",
                name,
            )
        )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, (x, y, label, _) in enumerate(sorted(points, key=lambda p: p[3])):
        colour = SERIES[i % len(SERIES)]
        ax.scatter([x], [y], s=140, color=colour, marker=MARKERS[i % len(MARKERS)],
                   edgecolors=SURFACE, linewidths=1.6, zorder=4, label=label)
        # Alternate the label offset so two configurations that land on top of
        # each other stay readable.
        ax.annotate(label.split("  (")[0], (x, y), textcoords="offset points",
                    xytext=(10, 6) if i % 2 == 0 else (10, -14), fontsize=9, color=colour)

    frontier = []
    best_quality = -1.0
    for x, y, _, _ in sorted(points):
        if y > best_quality:
            frontier.append((x, y))
            best_quality = y
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], linewidth=1.4,
                linestyle="--", color=INK_SOFT, alpha=0.65, zorder=2)

    _style(
        ax,
        "The only comparison that matters: quality against latency",
        "p95 end-to-end latency at the best in-SLO operating point, seconds",
        "composite answer quality (0-1)",
        "up and to the left is better; the dashed line is the achievable frontier",
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.margins(x=0.22, y=0.28)
    return _save(fig, out)


def cache_threshold_sweep(sweep: dict[str, Any], out: Path) -> Path | None:
    points = sweep.get("points") or []
    if len(points) < 3:
        return None
    xs = [p["threshold"] for p in points]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(xs, [p["hit_rate"] for p in points], marker="o", markersize=6, linewidth=2,
            color=SERIES[0], label="cache hit rate")
    ax.plot(xs, [p["false_hit_rate"] for p in points], marker="s", markersize=6, linewidth=2,
            color=SERIES[1], label="false-hit rate (wrong answer served)")

    rec = sweep.get("recommended_threshold")
    if rec:
        ax.axvline(rec, color=INK_SOFT, linestyle=":", linewidth=1.3)
        ax.annotate(f"recommended\nthreshold {rec}", (rec, 0.9), textcoords="offset points",
                    xytext=(8, 0), fontsize=8.5, color=INK_SOFT)

    ax.set_ylim(-0.03, 1.03)
    _style(
        ax,
        "Choosing the semantic-cache threshold is a safety decision",
        "cosine similarity threshold",
        "share of queries",
        f"embedder: {sweep.get('embedder', '?')}; a hit is 'false' when the served answer "
        f"scores below {sweep.get('quality_floor', 0.75)} F1 against this query's reference",
    )
    ax.legend(loc="center left")
    return _save(fig, out)


def server_state_vs_load(summaries: list[RunSummary], out: Path) -> Path | None:
    """KV-cache use and vLLM's own waiting queue against concurrency.

    The latency chart shows where the curve bends; this shows why: the bend
    should line up with the KV cache filling and requests starting to wait.
    """
    groups = {
        k: sorted([s for s in v if s.params.get("concurrency") and s.server and s.server.get("n_scrapes")],
                  key=lambda s: s.params["concurrency"])
        for k, v in _group(summaries, mode="closed").items()
    }
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if not groups:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    all_x: set[int] = set()
    for i, (name, runs) in enumerate(groups.items()):
        xs = [r.params["concurrency"] for r in runs]
        all_x.update(xs)
        colour, marker = SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]
        kv = [(r.server.get("kv_cache_usage_mean") or 0.0) * 100 for r in runs]
        kv_max = [(r.server.get("kv_cache_usage_max") or 0.0) * 100 for r in runs]
        wait = [r.server.get("waiting_mean") or 0.0 for r in runs]
        ax1.plot(xs, kv, marker=marker, markersize=6, linewidth=2, color=colour, label=f"{name} mean")
        ax1.plot(xs, kv_max, marker=marker, markersize=4, linewidth=1.2, linestyle="--",
                 color=colour, alpha=0.7, label=f"{name} max")
        ax2.plot(xs, wait, marker=marker, markersize=6, linewidth=2, color=colour, label=name)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(all_x))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax1.set_ylim(0, 105)
    _style(ax1, "KV cache in use", "concurrent requests (log scale)", "% of KV cache",
           "solid = mean during the run, dashed = peak")
    _style(ax2, "Requests waiting inside vLLM", "concurrent requests (log scale)", "mean waiting requests",
           "queued by the engine, not by the client")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)
    return _save(fig, out)


def render_all(
    summaries: list[RunSummary],
    charts_dir: Path,
    cache_sweep: dict[str, Any] | None = None,
) -> dict[str, Path]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    plan = [
        ("throughput_vs_concurrency", lambda p: throughput_vs_concurrency(summaries, p)),
        ("latency_vs_concurrency", lambda p: latency_vs_concurrency(summaries, p)),
        ("latency_vs_offered_load", lambda p: latency_vs_offered_load(summaries, p)),
        ("latency_percentiles", lambda p: latency_percentiles(summaries, p)),
        ("server_state_vs_load", lambda p: server_state_vs_load(summaries, p)),
        ("cost_per_1k_requests", lambda p: cost_per_1k(summaries, p)),
        ("quality_vs_latency", lambda p: quality_vs_latency(summaries, p)),
    ]
    for name, fn in plan:
        path = fn(charts_dir / f"{name}.png")
        if path:
            made[name] = path
    if cache_sweep:
        path = cache_threshold_sweep(cache_sweep, charts_dir / "cache_threshold_sweep.png")
        if path:
            made["cache_threshold_sweep"] = path
    return made
