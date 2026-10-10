"""Command line interface.

    vantage doctor                     can I reach the backends I think I can?
    vantage workload                   what does my workload actually look like?
    vantage bench                      one configuration, measured
    vantage sweep -c <config.yaml>     a whole experiment, charted and written up
    vantage quality                    score answers, no load testing
    vantage cache-sim                  pick a semantic-cache threshold
    vantage report <run_dir>           rebuild the report from saved raw data
    vantage demo                       the whole thing on a simulated backend
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from vantage.backends.registry import build_backend
from vantage.cache.analysis import build_query_log, sweep_thresholds
from vantage.cache.embedders import build_embedder
from vantage.config import load_dotenv
from vantage.logging import error, good, info, warn
from vantage.models import SLO, LoadConfig, RunSummary
from vantage.quality.harness import run_quality_eval
from vantage.report.charts import render_all
from vantage.report.render import write_report
from vantage.runner import RunSpec, run_spec
from vantage.store import RunStore
from vantage.sweep import execute_sweep
from vantage.workload.dataset import Workload

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Measure what an LLM serving stack actually does: latency, throughput, cost, quality.",
)


@app.callback()
def _main() -> None:
    load_dotenv()


# --------------------------------------------------------------------------- #


@app.command()
def doctor(
    backend: list[str] = typer.Option(
        ["sim.fp16"], "--backend", "-b", help="backend config name; repeatable"
    ),
    workload: Path = typer.Option(Path("data/workload.sample.jsonl"), "--workload", "-w"),
) -> None:
    """Check that the backends respond and the workload parses.

    Worth thirty seconds before you spend half an hour benchmarking a server
    that was never up.
    """

    async def _run() -> int:
        failures = 0
        for name in backend:
            try:
                be = build_backend(name)
            except Exception as exc:  # noqa: BLE001
                error(f"{name}: could not be constructed -- {exc}")
                failures += 1
                continue
            async with be:
                health = await be.health()
            (good if health.ok else error)(
                f"{name}: {'ok' if health.ok else 'FAILED'} -- {health.detail}"
            )
            failures += 0 if health.ok else 1

        try:
            wl = Workload.from_jsonl(workload)
            good(f"workload {workload}: {json.dumps(wl.stats())}")
        except Exception as exc:  # noqa: BLE001
            error(f"workload {workload}: {exc}")
            failures += 1
        return failures

    raise typer.Exit(code=1 if asyncio.run(_run()) else 0)


@app.command()
def workload(path: Path = typer.Argument(Path("data/workload.sample.jsonl"))) -> None:
    """Print the shape of a workload file.

    The prompt length distribution is what decides whether a benchmark number
    means anything, so look at it before you quote one.
    """
    wl = Workload.from_jsonl(path)
    typer.echo(json.dumps(wl.stats(), indent=2))


@app.command()
def bench(
    backend: str = typer.Option("sim.fp16", "--backend", "-b"),
    workload_path: Path = typer.Option(Path("data/workload.sample.jsonl"), "--workload", "-w"),
    concurrency: int = typer.Option(8, "--concurrency", "-c"),
    requests: int = typer.Option(64, "--requests", "-n"),
    max_tokens: int = typer.Option(192, "--max-tokens"),
    rate_qps: float | None = typer.Option(None, "--rate", help="switch to open loop at this rate"),
    duration_s: float | None = typer.Option(None, "--duration"),
    warmup: int = typer.Option(8, "--warmup"),
    cache_threshold: float | None = typer.Option(None, "--cache", help="enable semantic cache"),
    save: bool = typer.Option(False, "--save", help="write a run directory"),
) -> None:
    """Measure one configuration and print the summary."""
    wl = Workload.from_jsonl(workload_path)
    load = LoadConfig(
        mode="open" if rate_qps else "closed",
        concurrency=concurrency,
        rate_qps=rate_qps,
        n_requests=None if duration_s else requests,
        duration_s=duration_s,
        warmup_requests=warmup,
    )
    spec = RunSpec(
        id=f"{backend}@{'open-' + str(rate_qps) if rate_qps else 'c' + str(concurrency)}",
        backend=backend,
        load=load,
        max_tokens=max_tokens,
        label=backend,
        group=backend,
        cache={"threshold": cache_threshold} if cache_threshold else None,
    )
    outcome = asyncio.run(run_spec(spec, wl, slo=SLO()))
    s = outcome.summary
    typer.echo(json.dumps(s.model_dump(exclude={"slo"}), indent=2, default=str))
    info(
        f"\n{s.label}: {s.achieved_qps:.2f} req/s, {s.output_tokens_per_s:,.0f} tok/s, "
        f"TTFT p95 {s.ttft.p95:.3f}s, total p95 {s.total.p95:.3f}s, "
        f"goodput {s.goodput_ratio:.0%}, errors {s.error_rate:.1%}"
    )
    if save:
        store = RunStore(f"bench-{backend}")
        store.write_json("summaries.json", [s.model_dump()])
        good(f"saved to {store.dir}")


@app.command("sweep")
def sweep_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="a sweep YAML from configs/sweeps/"),
    limit: int | None = typer.Option(None, "--limit", help="run only the first N configurations"),
    results_dir: Path | None = typer.Option(None, "--results-dir"),
    no_report: bool = typer.Option(False, "--no-report"),
) -> None:
    """Run a full experiment and write a report."""
    result = asyncio.run(execute_sweep(config, results_dir=results_dir, limit=limit))
    if no_report:
        return
    charts = render_all(
        result.summaries,
        result.store.charts_dir,
        cache_sweep=result.cache_sweep.to_dict() if result.cache_sweep else None,
    )
    md, html = write_report(
        result.store.dir,
        result.manifest,
        result.summaries,
        charts,
        quality={k: v.model_dump() for k, v in result.quality.items()} or None,
        gate=[c.to_dict() for c in result.comparisons] or None,
        cache_sweep=result.cache_sweep.to_dict() if result.cache_sweep else None,
        break_even=result.break_even,
    )
    good(f"report: {md}")
    good(f"report: {html}")


@app.command()
def quality(
    backend: str = typer.Option("sim.fp16", "--backend", "-b"),
    workload_path: Path = typer.Option(Path("data/eval.holdout.jsonl"), "--workload", "-w"),
    concurrency: int = typer.Option(4, "--concurrency", "-c"),
    limit: int | None = typer.Option(None, "--limit"),
    out: Path | None = typer.Option(None, "--out", help="write the full per-item JSON here"),
) -> None:
    """Score a backend's answers against the held-out reference set."""
    wl = Workload.from_jsonl(workload_path)

    async def _run():
        be = build_backend(backend)
        async with be:
            return await run_quality_eval(be, wl, config_id=backend, concurrency=concurrency, limit=limit)

    summary = asyncio.run(_run())
    typer.echo(
        json.dumps(summary.model_dump(exclude={"per_item"}), indent=2, default=str)
    )
    if out:
        out.write_text(json.dumps(summary.model_dump(), indent=2), encoding="utf-8")
        good(f"per-item scores -> {out}")


@app.command("cache-sim")
def cache_sim(
    workload_path: Path = typer.Option(Path("data/workload.sample.jsonl"), "--workload", "-w"),
    repeats: int = typer.Option(5, "--repeats", help="how many times the query log revisits each question"),
    paraphrase_rate: float = typer.Option(0.6, "--paraphrase-rate"),
    quality_floor: float = typer.Option(0.75, "--quality-floor"),
    max_false_hit_rate: float = typer.Option(0.02, "--max-false-hit-rate"),
    embedder: str = typer.Option("hashing", "--embedder"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Choose a semantic-cache similarity threshold.

    Prints hit rate and false-hit rate at each candidate threshold, so the
    trade-off is in front of you rather than behind you.
    """
    wl = Workload.from_jsonl(workload_path)
    pairs = [(i.question, i.reference_answer) for i in wl.items if i.reference_answer]
    log = build_query_log(pairs, repeats=repeats, paraphrase_rate=paraphrase_rate)
    sweep = sweep_thresholds(
        log,
        embedder=build_embedder(embedder),
        quality_floor=quality_floor,
        max_false_hit_rate=max_false_hit_rate,
    )
    for p in sweep.points:
        info(
            f"  threshold {p.threshold:.2f}  hit {p.hit_rate:6.1%}  "
            f"false-hit {p.false_hit_rate:6.1%}  served quality {p.mean_served_quality:.3f}"
        )
    good(f"recommended threshold: {sweep.recommended_threshold} -- {sweep.recommendation_reason}")
    if out:
        out.write_text(json.dumps(sweep.to_dict(), indent=2), encoding="utf-8")


@app.command("report")
def report_cmd(run_dir: Path = typer.Argument(..., help="a results/<sweep>/<timestamp> directory")) -> None:
    """Rebuild charts and the written report from a saved run directory."""
    run_dir = Path(run_dir)

    def _load(name: str):
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    manifest = _load("manifest.json") or {"name": run_dir.name}
    raw_summaries = _load("summaries.json")
    if not raw_summaries:
        error(f"no summaries.json in {run_dir}")
        raise typer.Exit(code=1)
    summaries = [RunSummary(**row) for row in raw_summaries]
    cache_sweep = _load("cache_sweep.json")

    charts = render_all(summaries, run_dir / "charts", cache_sweep=cache_sweep)
    md, html = write_report(
        run_dir,
        manifest,
        summaries,
        charts,
        quality=_load("quality.json"),
        gate=_load("quality_gate.json"),
        cache_sweep=cache_sweep,
        break_even=_load("break_even.json"),
    )
    good(f"report: {md}")
    good(f"report: {html}")


@app.command()
def demo(
    config: Path = typer.Option(Path("configs/sweeps/demo.yaml"), "--config", "-c"),
) -> None:
    """Run the full pipeline against the simulated backend. No GPU, no keys."""
    info("[bold]Vantage demo[/bold] -- simulated backend, about a minute.\n")
    sweep_cmd(config=config, limit=None, results_dir=None, no_report=False)
    warn("These numbers come from the simulator, not hardware. Point the configs at a real server before quoting them.")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
