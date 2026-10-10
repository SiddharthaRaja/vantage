# Vantage

**A serving-performance and answer-quality benchmark harness for LLM inference.**

Most "I deployed an LLM" projects stop at a working endpoint. Vantage is the
part that comes after: it drives real load at a serving stack, measures what it
actually does under that load, prices it, checks that the optimisations did not
quietly make the answers worse, and writes the whole thing up.

```bash
pip install -e . && vantage demo        # no GPU, no API key, ~2 minutes
```

That produces a run directory with per-request records, per-configuration
summaries, six charts and a written report with its own caveats section.

---

## What it measures, and why those things

| | |
|---|---|
| **TTFT, ITL, end-to-end, total** | Four different latencies that benchmarks routinely conflate. `total` is measured from request *arrival*, so queueing is inside it — the number a user experiences. |
| **Closed-loop and open-loop load** | Closed loop answers "how does it behave at concurrency N". Open loop (Poisson arrivals) answers "what happens at N requests per second" — and is the only mode that can show the saturation cliff, because a closed-loop client throttles itself whenever the server slows down. |
| **Goodput, not just throughput** | Requests per second that *met the SLO*. A config that doubles throughput while blowing the SLO on 40% of requests has not doubled anything. |
| **Cost per 1,000 requests, and break-even** | API cost scales with tokens; self-hosting costs the same whether the GPU is busy or idle. Those two lines cross somewhere, and finding where is the actual engineering question behind "should we self-host". |
| **Answer quality, as a gate** | Every optimisation here trades quality for speed somewhere. Token F1, citation grounding and number agreement run on a held-out set, compared by paired bootstrap, and a config that regresses beyond budget is marked FAIL no matter how fast it is. |

## What it can point at

One adapter covers every OpenAI-compatible server, which is nearly all of them:
**vLLM**, **SGLang**, **TGI**, **llama.cpp**, **Ollama**, **LM Studio**, the
**OpenAI API** and most hosted providers. Plus a **simulated backend** that
models a continuous-batching engine, so the entire harness — including CI —
runs with no GPU at all.

## The optimisations it implements and measures

- **Semantic caching** — embedding-similarity lookup with a tunable threshold,
  shipped with the analysis that chooses the threshold (below).
- **Prefix caching**, **quantisation** (fp16 / int8 / int4) and
  **batching/concurrency** — configured, swept and compared.
- **A fine-tuned small model** — `finetune/` distils a teacher, QLoRA-trains a
  4B student, and runs it through this same harness so it lands in the same
  table as everything else. See [finetune/README.md](finetune/README.md).

---

## The chart the project is really about

A semantic cache that returns a confident answer to a question nobody asked is
worse than no cache. So the threshold is chosen with the false-hit rate on
screen, not by picking 0.9 because it sounds safe:

| threshold | hit rate | false-hit rate |
|---|---|---|
| 0.70 | 85.7% | 17.5% |
| 0.80 | 76.3% | 10.5% |
| **0.90** | **49.7%** | **0.7%** |
| 0.96 | 30.7% | 0.0% |

A false hit is a cache hit whose served answer scores below 0.75 F1 against the
*new* question's reference. The sweep replays an ordered query log through a
cold cache at each threshold; order matters, because a shuffled log overstates
what a cold cache can do.

## Example findings from `vantage demo`

> These come from the **simulated** backend, so they demonstrate the analysis,
> not a GPU. Every report generated from the simulator says so in its header.

- fp16 saturates at ~3,400 output tok/s; the computed knee is at concurrency 16.
  Past it, throughput rises 25% while p95 latency rises 40%.
- int4 is 1.7x the throughput of fp16 and 42% cheaper per request — **and fails
  the quality gate** at −0.053 composite (95% CI −0.096 to −0.015). int8 gives
  most of the speed and passes.
- Prefix caching cut TTFT roughly in half at low concurrency and did **nothing**
  for throughput, because at 450-token prompts and 96-token answers this
  workload is decode-bound. A real finding, and one you only get by measuring
  prefill and decode separately.
- Break-even against a hosted API lands at **1.65 req/s sustained** (~142,000
  requests/day) for a $0.75/hour GPU at this prompt and answer length.

---

## Install and use

```bash
git clone <this repo> && cd vantage
pip install -e ".[dev]"

vantage doctor -b sim.fp16            # is everything reachable?
vantage demo                          # full pipeline, simulated
vantage workload data/workload.sample.jsonl

# against a real server
vllm serve Qwen/Qwen3-4B-Instruct --port 8000 --enable-prefix-caching
vantage doctor -b vllm.qwen3-4b
vantage sweep -c configs/sweeps/concurrency.yaml
vantage sweep -c configs/sweeps/offered-load.yaml

# one-off measurements
vantage bench -b vllm.qwen3-4b -c 32 -n 200
vantage quality -b vllm.qwen3-4b
vantage cache-sim --repeats 5

# rebuild a report from saved raw data, months later
vantage report results/concurrency/20261010-141630
```

Experiments are YAML. A concurrency sweep is six lines:

```yaml
runs:
  - id: vllm
    backend: vllm.qwen3-4b
    matrix:
      load.concurrency: [1, 2, 4, 8, 16, 32, 64, 128]
quality: { enabled: true, baseline: vllm }
```

## Layout

```
src/vantage/
  backends/     one interface; OpenAI-compatible adapter + simulator
  workload/     RAG-shaped prompts, closed- and open-loop generators
  metrics/      percentiles, bootstrap CIs, knee detection, cost model
  cache/        exact + semantic cache, threshold/false-hit analysis
  quality/      scorers and the paired-bootstrap gate
  report/       charts and the written report
configs/        backend definitions and sweep experiments
finetune/       distillation, QLoRA training, merge/export, comparison
tests/          59 tests, all runnable without a GPU
```

Every run directory is self-contained: `manifest.json` (resolved config, git
SHA, host, GPUs), `raw.jsonl` (every request), `summaries.json`, `quality.json`,
`quality_gate.json`, `cache_sweep.json`, `break_even.json`, charts, and both
`REPORT.md` and a standalone `report.html`. If a chart cannot be regenerated
from those files, it is not evidence.

## Methodology, in short

Full version in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

- Warm-up requests are run and discarded; cold-start costs do not enter a p99.
- Configurations run sequentially with a cool-down. Two benchmarks sharing a GPU
  measure each other.
- Retries are **off** by default. A retry hides a failure and inflates the
  latency of the request that succeeded.
- Failed requests are recorded, never dropped — otherwise a server returning 500
  to everything posts an excellent p95.
- Percentiles use linear interpolation (numpy/wrk/k6 convention), stated because
  the nearest-rank convention gives a visibly different p99 on short runs.
- Differences are reported with bootstrap confidence intervals. If the interval
  for a difference spans zero, the report does not call it a difference.
- Token counts come from the server's `usage` block where available; where they
  are estimated, the report says so.

## CI

Every push runs lint, the test suite, and a full sweep against the simulated
backend whose quality gate must behave as expected. A change that breaks the
measurement fails the build rather than surfacing three weeks later as a chart
that looks slightly wrong.

## Licence

MIT.
