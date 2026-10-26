"""Sweep execution: expand a YAML experiment into runs, execute them in a
defined order, and persist everything needed to rebuild the report.

Two things that look like details and are not:

* Runs execute sequentially, never in parallel. Two benchmarks sharing a GPU
  measure each other.
* A cool-down between configurations, because a saturated server does not
  become unsaturated the instant the last request returns, and the next
  configuration would inherit the queue.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vantage.backends.registry import build_backend
from vantage.cache.analysis import CacheSweep, build_query_log, sweep_thresholds
from vantage.cache.embedders import build_embedder
from vantage.config import load_yaml, merge
from vantage.errors import ConfigError
from vantage.logging import good, info, warn
from vantage.metrics.cost import break_even_qps
from vantage.models import SLO, LoadConfig, QualitySummary, RunSummary
from vantage.quality.harness import QualityComparison, compare_quality, run_quality_eval
from vantage.runner import RunSpec, run_spec
from vantage.servers import ManagedServer, ServerStartError, load_profiles
from vantage.store import RunStore, environment_fingerprint
from vantage.workload.dataset import Workload


@dataclass
class SweepResult:
    name: str
    store: RunStore
    summaries: list[RunSummary] = field(default_factory=list)
    quality: dict[str, QualitySummary] = field(default_factory=dict)
    comparisons: list[QualityComparison] = field(default_factory=list)
    cache_sweep: CacheSweep | None = None
    break_even: dict[str, Any] | None = None
    workload_stats: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    engine_stats: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Expansion
# --------------------------------------------------------------------------- #


def expand_runs(cfg: dict[str, Any]) -> list[RunSpec]:
    """Turn `runs:` (each optionally with a `matrix:`) into a flat list.

    A matrix key is a dotted path: `load.concurrency`, `backend.max_batch`,
    `max_tokens`. The cartesian product is taken per run entry, and the
    generated id records every varied value so a chart point can always be
    traced back to a configuration.
    """
    defaults = cfg.get("defaults", {}) or {}
    specs: list[RunSpec] = []

    for entry in cfg.get("runs", []) or []:
        entry = merge(defaults, entry)
        matrix: dict[str, list[Any]] = entry.pop("matrix", {}) or {}
        keys = list(matrix)
        combos = list(itertools.product(*[matrix[k] for k in keys])) if keys else [()]

        for combo in combos:
            resolved = dict(entry)
            load_cfg = dict(resolved.get("load", {}) or {})
            backend_overrides = dict(resolved.get("backend_overrides", {}) or {})
            extra: dict[str, Any] = {}

            for key, value in zip(keys, combo, strict=True):
                section, _, field_name = key.partition(".")
                if not field_name:
                    resolved[section] = value
                    extra[section] = value
                elif section == "load":
                    load_cfg[field_name] = value
                    extra[field_name] = value
                elif section == "backend":
                    backend_overrides[field_name] = value
                    extra[field_name] = value
                elif section == "cache":
                    resolved.setdefault("cache", {})[field_name] = value
                    extra[f"cache_{field_name}"] = value
                else:
                    raise ConfigError(f"matrix key '{key}' must start with load./backend./cache.")

            base_id = resolved.get("id") or str(resolved.get("backend", "run"))
            suffix = ",".join(f"{k}={v}" for k, v in extra.items())
            run_id = f"{base_id}[{suffix}]" if suffix else base_id

            specs.append(
                RunSpec(
                    id=run_id,
                    backend=resolved["backend"],
                    load=LoadConfig(**load_cfg) if load_cfg else LoadConfig(),
                    max_tokens=int(resolved.get("max_tokens", 192)),
                    label=resolved.get("label", base_id),
                    group=resolved.get("group", base_id),
                    backend_overrides=backend_overrides,
                    cache=resolved.get("cache"),
                    params={**extra, **({"server": resolved["server"]} if resolved.get("server") else {})},
                    server=resolved.get("server"),
                )
            )

    if not specs:
        raise ConfigError("sweep config contains no runs")
    return specs


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


async def execute_sweep(
    config_path: str | Path,
    results_dir: Path | None = None,
    limit: int | None = None,
    save_raw: bool = True,
    progress: bool = True,
) -> SweepResult:
    cfg = load_yaml(config_path)
    name = cfg.get("name") or Path(config_path).stem
    workload = Workload.from_jsonl(cfg["workload"])
    slo = SLO(**(cfg.get("slo") or {}))
    specs = expand_runs(cfg)
    if limit:
        specs = specs[:limit]

    store = RunStore(name, root=results_dir)
    result = SweepResult(name=name, store=store, workload_stats=workload.stats())

    result.manifest = {
        "name": name,
        "config_path": str(config_path),
        "config": cfg,
        "environment": environment_fingerprint(),
        "workload": {"path": cfg["workload"], **workload.stats()},
        "slo": slo.model_dump(),
        "n_configs": len(specs),
        "note": cfg.get("description", ""),
    }
    store.write_json("manifest.json", result.manifest)

    profiles = load_profiles(cfg)
    for spec in specs:
        if spec.server and spec.server not in profiles:
            raise ConfigError(f"run '{spec.id}' names server '{spec.server}', which is not under servers:")
    if profiles and (cfg.get("quality") or {}).get("enabled", False):
        raise ConfigError(
            "a sweep that manages servers cannot also run the quality gate: the gate runs after "
            "every load point, when the servers are already stopped. Gate in a separate sweep."
        )

    info(f"[bold]sweep '{name}'[/bold]: {len(specs)} configurations -> {store.dir}")

    cooldown = float(cfg.get("cooldown_s", 1.0))
    current: ManagedServer | None = None
    failed_servers: dict[str, str] = {}
    server_log: dict[str, Any] = {}
    try:
        for i, spec in enumerate(specs, start=1):
            info(f"({i}/{len(specs)}) {spec.id}")
            if spec.server and spec.server in failed_servers:
                warn(f"skipping {spec.id}: server '{spec.server}' did not start")
                continue
            if spec.server and (current is None or current.profile.name != spec.server):
                if current is not None:
                    await current.stop()
                current = ManagedServer(profiles[spec.server], store.dir / f"server_{spec.server}.log")
                info(f"starting server '{spec.server}': {current.profile.command}")
                try:
                    await current.start()
                except ServerStartError as exc:
                    failed_servers[spec.server] = str(exc)
                    warn(str(exc))
                    current = None
                    continue
                server_log[spec.server] = {
                    "command": current.profile.command,
                    "env": current.profile.env,
                    "startup_s": round(current.startup_s or 0.0, 1),
                }
                store.write_json("servers.json", {"started": server_log, "failed": failed_servers})
            await _one_run(spec, i, len(specs), workload, slo, progress, result, store, save_raw, cooldown)
    finally:
        if current is not None:
            await current.stop()
        if profiles:
            store.write_json("servers.json", {"started": server_log, "failed": failed_servers})

    if profiles and not result.summaries:
        raise ConfigError(f"no server started: {failed_servers}")

    await _after_runs(cfg, workload, specs, result, store)
    return result


async def _one_run(
    spec: RunSpec,
    i: int,
    n: int,
    workload: Workload,
    slo: SLO,
    progress: bool,
    result: SweepResult,
    store: RunStore,
    save_raw: bool,
    cooldown: float,
) -> None:
    outcome = await run_spec(spec, workload, slo=slo, progress=progress)
    result.summaries.append(outcome.summary)
    if outcome.engine_stats:
        result.engine_stats[spec.id] = outcome.engine_stats
    if save_raw:
        store.append_jsonl(
            "raw.jsonl",
            [
                {**r.model_dump(exclude={"text"}), "config_id": spec.id}
                for r in outcome.results
                if not r.is_warmup
            ],
        )
    store.write_json("summaries.json", [s.model_dump() for s in result.summaries])
    if cooldown and i < n:
        await asyncio.sleep(cooldown)


async def _after_runs(
    cfg: dict[str, Any],
    workload: Workload,
    specs: list[RunSpec],
    result: SweepResult,
    store: RunStore,
) -> None:
    # -- quality gate ------------------------------------------------------- #
    qcfg = cfg.get("quality") or {}
    if qcfg.get("enabled", False):
        await _run_quality(cfg, qcfg, workload, specs, result)

    # -- semantic cache threshold sweep -------------------------------------- #
    ccfg = cfg.get("cache_analysis") or {}
    if ccfg.get("enabled", False):
        pairs = [
            (item.question, item.reference_answer)
            for item in workload.items
            if item.reference_answer
        ]
        log = build_query_log(
            pairs,
            repeats=int(ccfg.get("repeats", 4)),
            paraphrase_rate=float(ccfg.get("paraphrase_rate", 0.6)),
            seed=int(ccfg.get("seed", 5)),
        )
        result.cache_sweep = sweep_thresholds(
            log,
            thresholds=ccfg.get("thresholds"),
            embedder=build_embedder(ccfg.get("embedder")),
            quality_floor=float(ccfg.get("quality_floor", 0.75)),
            max_false_hit_rate=float(ccfg.get("max_false_hit_rate", 0.02)),
        )
        store.write_json("cache_sweep.json", result.cache_sweep.to_dict())
        good(
            f"cache sweep: recommended threshold {result.cache_sweep.recommended_threshold} "
            f"({result.cache_sweep.recommendation_reason})"
        )

    # -- break-even ---------------------------------------------------------- #
    cost_cfg = cfg.get("cost") or {}
    if cost_cfg.get("api_reference") and cost_cfg.get("self_hosted_reference"):
        try:
            api_backend = build_backend(cost_cfg["api_reference"])
            host_backend = build_backend(cost_cfg["self_hosted_reference"])
            ok = [s for s in result.summaries if s.n_ok]
            avg_in = sum(s.total_prompt_tokens for s in ok) / max(1, sum(s.n_ok for s in ok))
            avg_out = sum(s.total_completion_tokens for s in ok) / max(1, sum(s.n_ok for s in ok))
            be = break_even_qps(api_backend.pricing(), host_backend.pricing(), avg_in, avg_out)
            if be:
                result.break_even = {
                    "qps": be.qps,
                    "requests_per_day": be.requests_per_day,
                    "api_usd_per_request": be.api_usd_per_request,
                    "self_host_usd_per_hour": be.self_host_usd_per_hour,
                    "explanation": be.explanation,
                    "avg_prompt_tokens": avg_in,
                    "avg_completion_tokens": avg_out,
                }
                store.write_json("break_even.json", result.break_even)
            await api_backend.aclose()
            await host_backend.aclose()
        except Exception as exc:  # noqa: BLE001
            warn(f"break-even analysis skipped: {exc}")

    store.write_json("summaries.json", [s.model_dump() for s in result.summaries])
    store.write_json("engine_stats.json", result.engine_stats)
    good(f"sweep complete: {store.dir}")


async def _run_quality(
    cfg: dict[str, Any],
    qcfg: dict[str, Any],
    workload: Workload,
    specs: list[RunSpec],
    result: SweepResult,
) -> None:
    """Score each distinct backend configuration once.

    Quality is a property of the model and its serving precision, not of the
    concurrency you drove it at, so scoring every point of a concurrency sweep
    would be 7x the cost for identical numbers. Configurations are deduplicated
    by backend + overrides.
    """
    # One quality score per group, not per load point: answer quality is a
    # property of the model and its serving precision, not of the concurrency
    # you drove it at. Scoring all 22 points of this sweep would cost 22x for
    # six distinct numbers.
    seen: dict[str, RunSpec] = {}
    for spec in specs:
        seen.setdefault(spec.group, spec)

    eval_workload = workload
    if qcfg.get("eval_file"):
        eval_workload = Workload.from_jsonl(qcfg["eval_file"])

    info(f"[bold]quality gate[/bold]: {len(seen)} distinct backend configs")
    for spec in seen.values():
        backend = build_backend(spec.backend, {**spec.backend_overrides, "config_id": spec.group})
        async with backend:
            summary = await run_quality_eval(
                backend,
                eval_workload,
                config_id=spec.group,
                concurrency=int(qcfg.get("concurrency", 4)),
                max_tokens=int(qcfg.get("max_tokens", 256)),
                limit=qcfg.get("limit"),
                progress=True,
            )
        result.quality[spec.group] = summary

    result.store.write_json(
        "quality.json", {k: v.model_dump() for k, v in result.quality.items()}
    )

    baseline_id = qcfg.get("baseline")
    if baseline_id and baseline_id in result.quality:
        baseline = result.quality[baseline_id]
        for config_id, candidate in result.quality.items():
            if config_id == baseline_id:
                continue
            comparison = compare_quality(
                baseline, candidate, max_drop=float(qcfg.get("max_drop", 0.02))
            )
            result.comparisons.append(comparison)
            line = f"  {config_id} vs {baseline_id}: {comparison.verdict.upper()} -- {comparison.detail}"
            good(line) if comparison.verdict == "pass" else warn(line)
        result.store.write_json(
            "quality_gate.json", [c.to_dict() for c in result.comparisons]
        )
    elif baseline_id:
        warn(f"quality baseline '{baseline_id}' was not among the evaluated configs")

    # Attach quality to the summaries so one table can show speed and quality.
    for summary in result.summaries:
        group = summary.params.get("group") or _group_of(summary.config_id, specs)
        if group and group in result.quality:
            q = result.quality[group]
            summary.quality = {
                "composite": q.composite,
                "token_f1": q.token_f1,
                "citation_grounding": q.citation_grounding,
                "number_agreement": q.number_agreement,
                "n": q.n,
            }


def _group_of(config_id: str, specs: list[RunSpec]) -> str | None:
    for spec in specs:
        if spec.id == config_id:
            return spec.group
    return None
