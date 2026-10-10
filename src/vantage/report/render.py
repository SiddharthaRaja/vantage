"""Turn a finished sweep into a written report: Markdown and a standalone HTML
page with the charts embedded.

The report is written to be read by someone who was not there: it states the
method, the environment, the SLO, what was measured, what the numbers mean and
-- in its own section -- what the run does *not* establish. The caveats section
is not modesty. It is the part that makes the rest of the document credible,
and it is the first thing a good interviewer will look for.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from vantage.metrics.stats import knee_point
from vantage.models import RunSummary


def _fmt(value: Any, spec: str = ".3f", dash: str = "--") -> str:
    if value is None:
        return dash
    try:
        if isinstance(value, float) and value != value:  # NaN
            return dash
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


def headline_findings(
    summaries: list[RunSummary],
    quality: dict[str, Any] | None,
    gate: list[dict[str, Any]] | None,
    cache_sweep: dict[str, Any] | None,
    break_even: dict[str, Any] | None,
) -> list[str]:
    """Generate the findings paragraph from the data rather than by hand.

    Everything here is derived, so it cannot drift away from the numbers in
    the tables underneath it -- which is the usual failure mode of a
    hand-written summary.
    """
    findings: list[str] = []
    closed = [s for s in summaries if s.params.get("mode") == "closed"]

    groups: dict[str, list[RunSummary]] = {}
    for s in closed:
        groups.setdefault(str(s.params.get("group") or s.label), []).append(s)

    for name, runs in groups.items():
        runs = sorted([r for r in runs if r.params.get("concurrency")], key=lambda r: r.params["concurrency"])
        if len(runs) < 3:
            continue
        xs = [r.params["concurrency"] for r in runs]
        ys = [r.output_tokens_per_s for r in runs]
        k = knee_point(xs, ys)
        if k is None:
            continue
        best = max(runs, key=lambda r: r.output_tokens_per_s)
        findings.append(
            f"**{name}** saturates at about {best.output_tokens_per_s:,.0f} output tokens/s. "
            f"The knee of the curve is at concurrency {xs[k]} "
            f"({ys[k]:,.0f} tok/s, p95 {_fmt(runs[k].total.p95, '.2f')}s): past that point "
            f"throughput rises {(ys[-1] / ys[k] - 1) * 100:.0f}% more while p95 latency rises "
            f"{(runs[-1].total.p95 or 0) / (runs[k].total.p95 or 1) * 100 - 100:.0f}%."
        )

    within = [s for s in closed if s.goodput_ratio >= 0.95 and s.n_ok]
    if within:
        best = max(within, key=lambda s: s.achieved_qps)
        findings.append(
            f"The highest load served inside the SLO was **{best.achieved_qps:.2f} req/s** "
            f"({best.label}, concurrency {best.params.get('concurrency')}), with "
            f"{best.goodput_ratio:.0%} of requests meeting it."
        )

    costed = [s for s in summaries if s.cost.usd_per_1k_requests]
    if len(costed) >= 2:
        cheapest = min(costed, key=lambda s: s.cost.usd_per_1k_requests or 9e9)
        dearest = max(costed, key=lambda s: s.cost.usd_per_1k_requests or 0)
        if (dearest.cost.usd_per_1k_requests or 0) > 0:
            ratio = (dearest.cost.usd_per_1k_requests or 0) / (cheapest.cost.usd_per_1k_requests or 1)
            findings.append(
                f"Cheapest configuration measured: **{cheapest.label}** at "
                f"${cheapest.cost.usd_per_1k_requests:.3f} per 1,000 requests, "
                f"{ratio:.1f}x cheaper than {dearest.label}."
            )

    if break_even:
        findings.append(
            f"**Break-even:** self-hosting beats the API above "
            f"**{break_even['qps']:.2f} req/s** sustained "
            f"({break_even['requests_per_day']:,.0f} requests/day)."
        )

    if gate:
        failures = [g for g in gate if g["verdict"] == "fail"]
        passes = [g for g in gate if g["verdict"] == "pass"]
        if failures:
            names = ", ".join(g["candidate"] for g in failures)
            findings.append(
                f"**Quality gate failed for: {names}.** Faster, but measurably worse on the "
                f"held-out set -- the speed-up is not free and the report does not pretend it is."
            )
        if passes:
            names = ", ".join(g["candidate"] for g in passes)
            findings.append(
                f"Quality gate passed for: {names} (no statistically significant regression "
                f"against the baseline on paired bootstrap)."
            )

    if cache_sweep and cache_sweep.get("recommended_threshold"):
        pts = {p["threshold"]: p for p in cache_sweep["points"]}
        rec = pts.get(cache_sweep["recommended_threshold"])
        if rec:
            findings.append(
                f"**Semantic cache:** at threshold {rec['threshold']} the cache serves "
                f"{rec['hit_rate']:.0%} of queries with a false-hit rate of "
                f"{rec['false_hit_rate']:.1%} -- {rec['estimated_calls_saved']} model calls "
                f"avoided out of {rec['n_queries']}."
            )

    return findings


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def results_table(summaries: list[RunSummary]) -> str:
    headers = [
        "config", "mode", "load", "n", "err", "TTFT p50", "TTFT p95",
        "E2E p95", "total p95", "tok/s", "req/s", "goodput", "$/1k", "quality",
    ]
    rows = []
    for s in summaries:
        if s.params.get("mode") == "closed":
            load = f"c={s.params.get('concurrency')}"
        elif s.offered_qps:
            load = f"{s.offered_qps:g} qps"
        else:
            load = "--"
        rows.append(
            [
                s.label,
                str(s.params.get("mode")),
                load,
                str(s.n_requests),
                f"{s.error_rate:.1%}",
                _fmt(s.ttft.p50, ".3f"),
                _fmt(s.ttft.p95, ".3f"),
                _fmt(s.e2e.p95, ".3f"),
                _fmt(s.total.p95, ".3f"),
                f"{s.output_tokens_per_s:,.0f}",
                f"{s.achieved_qps:.2f}",
                f"{s.goodput_ratio:.0%}",
                _fmt(s.cost.usd_per_1k_requests, ".3f"),
                _fmt((s.quality or {}).get("composite"), ".3f"),
            ]
        )
    return _table(headers, rows)


def quality_table(quality: dict[str, Any]) -> str:
    headers = ["config", "n", "composite", "token F1", "citation grounding", "number agreement"]
    rows = []
    for name, q in quality.items():
        rows.append(
            [
                name,
                str(q["n"]),
                _fmt(q["composite"], ".3f"),
                _fmt(q["token_f1"], ".3f"),
                _fmt(q["citation_grounding"], ".3f"),
                _fmt(q["number_agreement"], ".3f"),
            ]
        )
    return _table(headers, rows)


def gate_table(gate: list[dict[str, Any]]) -> str:
    headers = ["candidate", "baseline", "delta", "95% CI", "significant", "verdict"]
    rows = [
        [
            g["candidate"],
            g["baseline"],
            f"{g['delta']:+.3f}",
            f"{g['delta_ci95'][0]:+.3f} to {g['delta_ci95'][1]:+.3f}",
            "yes" if g["significant"] else "no",
            g["verdict"].upper(),
        ]
        for g in gate
    ]
    return _table(headers, rows)


def cache_table(sweep: dict[str, Any]) -> str:
    headers = ["threshold", "hit rate", "exact", "semantic", "false-hit rate", "mean served quality"]
    rows = [
        [
            f"{p['threshold']:.2f}",
            f"{p['hit_rate']:.1%}",
            f"{p['exact_hit_rate']:.1%}",
            f"{p['semantic_hit_rate']:.1%}",
            f"{p['false_hit_rate']:.1%}",
            f"{p['mean_served_quality']:.3f}",
        ]
        for p in sweep["points"]
    ]
    return _table(headers, rows)


CAVEATS = """\
### What this run does not establish

- **Numbers are specific to this hardware, this model and this workload.** A
  throughput figure without the GPU, the model, the quantisation, the prompt
  length distribution and the output length distribution beside it is not a
  result. All of those are recorded in `manifest.json`.
- **Prompt and output lengths drive everything.** This workload is RAG-shaped:
  long prompts, short answers, so prefill dominates. A chat workload with short
  prompts and long answers inverts that and will produce different conclusions
  from the same server.
- **Short runs have fat tails.** A p99 computed over a few hundred requests is
  an estimate with real uncertainty. Bootstrap intervals are given for the means;
  treat single-run p99 differences below about 20% as noise unless the run is long.
- **Cost figures are arithmetic on list prices, not invoices.** They exclude
  engineering time, idle capacity, redundancy, egress and the fact that you must
  provision for peak while paying for mean.
- **The quality score is a proxy.** Token F1, citation grounding and number
  agreement catch the failures that matter most in a cited-answer RAG system, but
  they are not a human reading the answers. A config that passes the gate has not
  been proven good -- it has been shown not to have regressed on these measures.
- **One client, one network path.** Client-side CPU, Python async scheduling and
  network latency are inside every measurement. For absolute numbers the load
  generator should run on a separate machine close to the server.
"""


def build_markdown(
    manifest: dict[str, Any],
    summaries: list[RunSummary],
    charts: dict[str, Path],
    quality: dict[str, Any] | None = None,
    gate: list[dict[str, Any]] | None = None,
    cache_sweep: dict[str, Any] | None = None,
    break_even: dict[str, Any] | None = None,
    chart_prefix: str = "charts",
) -> str:
    env = manifest.get("environment", {})
    wl = manifest.get("workload", {})
    simulated = any("sim" in (s.backend or "") for s in summaries)

    parts: list[str] = []
    parts.append(f"# Vantage report: {manifest.get('name', 'run')}\n")
    if manifest.get("note"):
        parts.append(f"{manifest['note'].strip()}\n")
    if simulated:
        parts.append(
            "> **Simulated backend.** At least one configuration in this run was the built-in\n"
            "> simulator, not real hardware. The shapes are right and the pipeline is real;\n"
            "> the absolute numbers are a model. Do not quote them as measurements.\n"
        )

    parts.append(
        "\n".join(
            [
                f"- Run: `{env.get('timestamp_utc', '?')}` UTC",
                f"- Commit: `{env.get('git_sha') or 'not a git checkout'}`"
                + (" (working tree dirty)" if env.get("git_dirty") else ""),
                f"- Host: {env.get('platform', '?')}, Python {env.get('python', '?')}",
                f"- GPUs: {', '.join(env.get('gpus') or []) or 'none detected'}",
                f"- Workload: `{wl.get('path', '?')}` -- {wl.get('n_items', '?')} items, "
                f"median prompt ~{wl.get('approx_prompt_tokens_p50', '?')} tokens",
                f"- SLO: {json.dumps(manifest.get('slo', {}))}",
            ]
        )
        + "\n"
    )

    findings = headline_findings(summaries, quality, gate, cache_sweep, break_even)
    if findings:
        parts.append("## Findings\n")
        parts.extend(f"- {f}" for f in findings)
        parts.append("")

    if break_even:
        parts.append("## Self-host vs API\n")
        parts.append(break_even["explanation"] + "\n")

    parts.append("## Results\n")
    parts.append(results_table(summaries) + "\n")
    parts.append(
        "`total p95` is measured from request arrival, so it includes queueing; `E2E p95` "
        "starts at dispatch. In closed-loop runs the two are the same by construction.\n"
    )

    order = [
        ("throughput_vs_concurrency", "Throughput vs concurrency"),
        ("latency_vs_concurrency", "Latency vs concurrency"),
        ("latency_vs_offered_load", "Latency vs offered load (open loop)"),
        ("latency_percentiles", "TTFT percentiles"),
        ("quality_vs_latency", "Quality vs latency"),
        ("cost_per_1k_requests", "Cost per 1,000 requests"),
        ("cache_threshold_sweep", "Semantic cache threshold sweep"),
    ]
    charted = [(k, t) for k, t in order if k in charts]
    if charted:
        parts.append("## Charts\n")
        for key, title in charted:
            parts.append(f"### {title}\n")
            parts.append(f"![{title}]({chart_prefix}/{Path(charts[key]).name})\n")

    if quality:
        parts.append("## Answer quality\n")
        parts.append(quality_table(quality) + "\n")
        parts.append(
            "Composite = 0.45 x token F1 + 0.25 x citation grounding + 0.25 x number "
            "agreement + 0.05 x exact match. Weights are a judgement call and are stated "
            "here so a reader can re-weight them.\n"
        )
    if gate:
        parts.append("### Quality gate\n")
        parts.append(gate_table(gate) + "\n")
        parts.append(
            "Deltas are paired bootstrap over per-item composite scores (2,000 resamples). "
            "A difference whose interval spans zero is not treated as a difference.\n"
        )

    if cache_sweep:
        parts.append("## Semantic cache\n")
        parts.append(cache_table(cache_sweep) + "\n")
        parts.append(f"**Recommendation:** {cache_sweep['recommendation_reason']}\n")

    parts.append(CAVEATS)
    parts.append(
        "\n### Reproducing this\n\n```bash\n"
        f"vantage sweep -c {manifest.get('config_path', 'configs/sweeps/demo.yaml')}\n"
        "```\n\nRaw per-request records are in `raw.jsonl`; every chart and every number "
        "above is derived from that file plus `summaries.json`.\n"
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_HTML_CSS = """
:root { color-scheme: light dark;
  --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink-soft:#52514e;
  --line:#e4e3df; --accent:#2a78d6; --warn:#eb6834; }
@media (prefers-color-scheme: dark) { :root {
  --surface:#1a1a19; --panel:#232322; --ink:#f0efec; --ink-soft:#c3c2b7;
  --line:#3a3a38; --accent:#3987e5; --warn:#d95926; } }
* { box-sizing:border-box; }
body { margin:0; padding:48px 20px 96px; background:var(--surface); color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
main { max-width:980px; margin:0 auto; }
h1 { font-size:30px; line-height:1.2; margin:0 0 8px; letter-spacing:-0.02em; }
h2 { font-size:21px; margin:40px 0 12px; padding-top:18px; border-top:1px solid var(--line); letter-spacing:-0.01em; }
h3 { font-size:16px; margin:26px 0 8px; color:var(--ink-soft); text-transform:uppercase;
  letter-spacing:0.06em; font-weight:600; }
p, li { color:var(--ink); }
blockquote { margin:18px 0; padding:12px 16px; border-left:3px solid var(--warn);
  background:color-mix(in srgb, var(--warn) 8%, transparent); border-radius:0 8px 8px 0; }
table { border-collapse:collapse; width:100%; margin:14px 0 6px; font-size:13.5px;
  display:block; overflow-x:auto; }
th, td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--ink-soft); font-weight:600; font-size:12px; text-transform:uppercase;
  letter-spacing:0.04em; }
tbody tr:hover { background:color-mix(in srgb, var(--accent) 7%, transparent); }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:0.9em;
  background:color-mix(in srgb, var(--ink) 7%, transparent); padding:1px 5px; border-radius:4px; }
pre { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; overflow-x:auto; }
pre code { background:none; padding:0; }
img { max-width:100%; display:block; margin:10px 0 24px; border:1px solid var(--line);
  border-radius:10px; background:#fcfcfb; }
.badge { display:inline-block; font-size:12px; padding:2px 8px; border-radius:999px;
  border:1px solid var(--line); color:var(--ink-soft); }
"""


def _md_to_html(md: str, charts: dict[str, Path]) -> str:
    """A deliberately small Markdown subset renderer.

    Enough for headings, tables, lists, code fences, images and inline code --
    which is all this report uses. Avoiding a Markdown dependency keeps the
    report generator importable anywhere, and the HTML is produced from the
    same source as the .md so the two can never disagree.
    """
    import html as _html
    import re

    lines = md.splitlines()
    out: list[str] = []
    in_table = False
    in_code = False
    in_list = False

    def inline(text: str) -> str:
        text = _html.escape(text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
        return text

    def close_blocks() -> None:
        nonlocal in_table, in_list
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in lines:
        line = raw.rstrip()

        if line.startswith("```"):
            close_blocks()
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(_html.escape(line))
            continue

        img = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if img:
            close_blocks()
            key = Path(img.group(2)).stem
            path = charts.get(key)
            if path and Path(path).exists():
                data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
                out.append(f'<img alt="{_html.escape(img.group(1))}" src="data:image/png;base64,{data}">')
            else:
                out.append(f'<img alt="{_html.escape(img.group(1))}" src="{img.group(2)}">')
            continue

        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                close_blocks()
                out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells)
                           + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table and not line.startswith("|"):
            out.append("</tbody></table>")
            in_table = False

        if line.startswith("> "):
            close_blocks()
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
            continue
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list and not line.startswith("- "):
            out.append("</ul>")
            in_list = False

        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif not line:
            out.append("")
        else:
            out.append(f"<p>{inline(line)}</p>")

    close_blocks()
    body = "\n".join(out)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Vantage report</title>"
        f"<style>{_HTML_CSS}</style></head><body><main>{body}"
        f"<p class='badge'>generated by Vantage at {stamp}</p>"
        "</main></body></html>"
    )


def write_report(
    out_dir: Path,
    manifest: dict[str, Any],
    summaries: list[RunSummary],
    charts: dict[str, Path],
    quality: dict[str, Any] | None = None,
    gate: list[dict[str, Any]] | None = None,
    cache_sweep: dict[str, Any] | None = None,
    break_even: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    md = build_markdown(manifest, summaries, charts, quality, gate, cache_sweep, break_even)
    md_path = Path(out_dir) / "REPORT.md"
    md_path.write_text(md, encoding="utf-8")
    html_path = Path(out_dir) / "report.html"
    html_path.write_text(_md_to_html(md, charts), encoding="utf-8")
    return md_path, html_path
