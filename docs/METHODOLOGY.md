# Methodology

Everything here is a decision that changes the numbers. They are written down
so a reader can disagree with a specific choice rather than distrusting the
whole report.

## 1. The four latencies

| name | from | to | what it tells you |
|---|---|---|---|
| `queue_s` | arrival | dispatch | client-side backlog; non-zero only in open loop with a cap |
| `ttft_s` | dispatch | first token | network + scheduling + prefill. The number a user feels first |
| `itl_s` | token *i* | token *i+1* | how fast the text appears once it starts. Degrades as the batch grows |
| `e2e_s` | dispatch | last token | the full generation |
| `total_s` | arrival | last token | `queue_s + e2e_s`. The only one that is the user's experience |

Streaming is always on, even when nothing reads the stream, because TTFT does
not exist without it.

## 2. Closed loop vs open loop

**Closed loop** holds *N* virtual users. Each sends a request, waits, sends the
next. Offered load is therefore a function of the server's speed: if the server
slows down, the client slows down with it. This is the right model for a
batch job or a fixed worker pool, and it is what nearly every student benchmark
runs.

**Open loop** generates arrivals on a schedule — Poisson by default, because
independent users produce Poisson arrivals — regardless of whether earlier
requests have finished. This is the right model for user traffic, and it is the
only one in which a queue can build without bound. The characteristic result is
that latency is flat and unremarkable up to about 80% of capacity and then goes
vertical. A closed-loop test cannot produce that curve, which is why systems
that benchmarked beautifully fall over in production.

The generator pre-computes its arrival schedule before sending anything, so the
client's own work can never become the thing that shapes the arrival process.
If the client cannot keep up (more than `max_in_flight` outstanding), it records
the request as **shed** rather than delaying the arrival — a load generator that
quietly throttles itself has turned into a closed-loop test without saying so.

## 3. Warm-up and ordering

- Warm-up requests run at the configured concurrency and are discarded. The
  first requests to a fresh server carry CUDA graph capture, lazy weight loads
  and cold caches; on a 200-request run they would own the p99.
- Configurations run strictly sequentially with a cool-down between them. A
  saturated server does not drain the instant the last response returns, and
  the next configuration would inherit the queue.
- Within a run, the workload is sampled with a fixed seed, so two
  configurations see the same prompts in the same order.

## 4. Failures

A failed request is a data point. Dropping failures is the single easiest way
to produce an excellent benchmark for a broken server: if 60% of requests
return HTTP 500 in 3ms and you only summarise successes, the p95 looks superb.
So:

- Failures are recorded with their status and counted by kind.
- `error_rate` appears in every summary and `max_error_rate` is part of the SLO.
- Retries are **off** by default. With retries on, a 429 followed by a success
  is scored as one slow success — the failure disappears and the latency
  distribution is corrupted. When retries are enabled the count is stored per
  request.

## 5. Statistics

- Percentiles use linear interpolation between order statistics (numpy's
  default, matching wrk, k6 and vegeta). The nearest-rank convention gives a
  visibly different p99 on a few hundred samples.
- Confidence intervals are **bootstrapped**, not normal-theory. Latency is
  right-skewed and heavy-tailed; mean ± 1.96σ is not meaningful on 200 samples.
- Quality comparisons use a **paired** bootstrap over per-item scores, because
  both configurations answered the same questions. Pairing is free variance
  reduction and discarding it would widen every interval for no reason.
- **A difference whose 95% interval includes zero is not reported as a
  difference.** This rule is why the gate says "pass" for small negative deltas
  rather than inventing a regression.

## 6. Throughput, goodput, and the knee

- `output_tokens_per_s` is the system's aggregate decode rate: total completion
  tokens over the measurement window. It is not the per-stream rate, which
  falls as the batch grows — both are reported because they answer different
  questions ("how much can this box serve" vs "how fast does text appear").
- `goodput_ratio` is the share of requests meeting the SLO. `achieved_qps ×
  goodput_ratio` is the honest headline number.
- The **knee** of a throughput curve is computed by maximum perpendicular
  distance from the chord joining the curve's endpoints (the Kneedle
  construction), not by eye, and only for curves with at least five points —
  on four points "the knee" is an artefact of where you happened to sample.

## 7. Cost

Two regimes, reduced to the same unit:

```
API         usd/request = in_tokens/1e6 * price_in + out_tokens/1e6 * price_out
self-host   usd/request = (gpu_hourly * n_gpus / 3600) / achieved_qps
```

The second depends on utilisation; the first does not. Setting them equal gives

```
break-even qps = (gpu_hourly * n_gpus / 3600) / api_usd_per_request
```

Below that rate you are renting an idle GPU. Above it the token bill wins. What
the figure deliberately excludes, and what the report says it excludes:
engineering and on-call time, redundancy, the gap between peak and mean traffic
(you provision for peak, you pay for peak, you use mean), egress, and the fact
that list prices change.

## 8. Quality

Four scorers, all deterministic and cheap enough to run in CI:

| scorer | catches |
|---|---|
| `token_f1` | general agreement with the reference, robust to phrasing |
| `exact_match` | strict agreement, useful only for short extractive answers |
| `citation_grounding` | answers that cite nothing, or cite a source number that was never in the prompt — a machine-checkable hallucination signal |
| `number_agreement` | numbers in the answer that do not appear in the reference. In a biomedical corpus this is the most dangerous failure, and token F1 barely notices it |

Composite = `0.45·F1 + 0.25·grounding + 0.25·numbers + 0.05·EM`. The weights are
a judgement call, stated in every report so a reader can re-weight them.

An LLM judge is supported and off by default: a gate you cannot afford to run
on every commit is not a gate.

**What this does not measure:** fluency, helpfulness, tone, whether the answer
is useful to a clinician. A configuration that passes has not been shown to be
good; it has been shown not to have regressed on these measures.

## 9. Reproducibility

Each run directory contains the resolved config, the git SHA and dirty flag,
the host and GPU description, and every individual request. The report is
generated from those files, and `vantage report <dir>` regenerates it. A number
that cannot be traced back to `raw.jsonl` does not belong in the report.

## 10. Known limitations

- The load generator is Python asyncio. Above roughly 2,000 requests/second on
  one process the client itself becomes a measurable share of the latency; for
  rates beyond that, run several client processes or use a Go/Rust generator and
  treat this harness as the analysis layer.
- The client should not share a machine with the server. `docker-compose.yml`
  separates them for this reason.
- The simulator is a model, not hardware. It reproduces the shapes (saturating
  throughput, batch-dependent decode, queueing collapse) and nothing else.
