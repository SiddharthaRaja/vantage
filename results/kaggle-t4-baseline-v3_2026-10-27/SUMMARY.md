# Vantage on Kaggle 2x T4 (kaggle_t4_2026-10-27_0902)

vLLM 0.32.0; total 1.90 h; dry run: False

| step | status | note |
|---|---|---|
| setup | ok | 2cb7094 Kaggle baseline v3: never hand the client's API key to vLLM |
| install_vllm | ok | vllm 0.32.0 (isolated Python 3.12) |
| servers | ok | vllm 0.32.0 (cuda graphs on) |
| sanity | ok | prompt tokens [1174, 1174] |
| kaggle-t4 | ok | gate pass: delta -0.012, 95% CI [-0.0327, 0.0096] |
| kaggle-t4-offered-load | ok | done |
| crosscheck | failed | RuntimeError: vllm bench serve failed with every flag set (see vllm_bench.log) |
| cache_sim | ok | 0.8 |

## Sanity check

```
question: Do the authors give any examples of major events which draw the public's attention and the impact they have on stock price?

Qwen/Qwen3-4B-Instruct-2507: prompt_tokens=1174 completion_tokens=127 7.0s
Yes, the authors provide an example of a specific news event and its impact on stock price. News event No. is correlated with the stock price rise on 07/15/2013, and its impact is observed to fade out for the next two trading days, reappearing on 07/18/2013 [6]. This illustrates how certain news events can influence stock prices in a time-dependent manner, with effects that emerge, diminish, and then reappear. However, the sources do not specify the nature or content of the news event beyond its correlation with price movements.

cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit: prompt_tokens=1174 completion_tokens=132 4.4s
The authors provide an example of a specific news event correlated with stock price movement: "news event No. has been correlated with the stock price rises on 07/15/2013, but for the next two trading days, its impact fades out. On 07/18/2013, the news event No. begins to show its impact" [6]. This illustrates how a particular news event influences stock prices on a given day and how its impact evolves over subsequent trading days. However, the sources do not specify the nature or identity of the news event beyond its correlation with price changes on those dates.

```

---

# Vantage report: kaggle-t4

Qwen3-4B-Instruct-2507 fp16 vs AWQ int4 on vLLM, one Kaggle T4 each, Vectora workload (qasper-300), quality-gated on 32 references.

- Run: `2026-10-27T09:08:04+00:00` UTC
- Commit: `2cb7094`
- Host: Linux-6.18.55+-x86_64-with-glibc2.39, Python 3.13.15
- GPUs: Tesla T4, 15360 MiB, 580.178.04, Tesla T4, 15360 MiB, 580.178.04
- Workload: `data/workload.vectora.jsonl` -- 365 items, median prompt ~663 tokens
- SLO: {"ttft_p95_s": 2.0, "total_p95_s": 15.0, "max_error_rate": 0.01}

## Findings

- **fp16** saturates at about 100 output tokens/s. The knee of the curve is at concurrency 16 (89 tok/s, p95 34.66s): past that point throughput rises 12% more while p95 latency rises 239%.
- **awq** saturates at about 120 output tokens/s. The knee of the curve is at concurrency 4 (102 tok/s, p95 9.00s): past that point throughput rises 18% more while p95 latency rises 1216%.
- The highest load served inside the SLO was **0.68 req/s** (Qwen3-4B AWQ int4 (T4), concurrency 4), with 100% of requests meeting it.
- Cheapest configuration measured: **Qwen3-4B AWQ int4 (T4)** at $0.122 per 1,000 requests, 5.5x cheaper than Qwen3-4B fp16 (T4).
- **Break-even:** self-hosting beats the API above **0.40 req/s** sustained (34,434 requests/day).
- Quality gate passed for: awq (no statistically significant regression against the baseline on paired bootstrap).

## Self-host vs API

API costs $0.00024/request at 1035 in / 148 out tokens. A $0.35/hour GPU costs $0.00010/second regardless of load, so it pays for itself above 0.40 req/s sustained (~34,434 requests/day) -- provided the GPU can actually serve that rate within SLO.

## Results

| config | mode | load | n | err | TTFT p50 | TTFT p95 | E2E p95 | total p95 | tok/s | req/s | goodput | $/1k | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B fp16 (T4) | closed | c=1 | 48 | 0.0% | 0.663 | 0.881 | 8.897 | 8.897 | 22 | 0.15 | 100% | 0.667 | 0.483 |
| Qwen3-4B fp16 (T4) | closed | c=2 | 48 | 0.0% | 0.711 | 0.932 | 10.700 | 10.700 | 37 | 0.24 | 100% | 0.397 | 0.483 |
| Qwen3-4B fp16 (T4) | closed | c=4 | 96 | 0.0% | 0.743 | 1.138 | 14.151 | 14.151 | 55 | 0.38 | 100% | 0.253 | 0.483 |
| Qwen3-4B fp16 (T4) | closed | c=8 | 96 | 0.0% | 0.830 | 1.447 | 21.397 | 21.397 | 74 | 0.52 | 48% | 0.187 | 0.483 |
| Qwen3-4B fp16 (T4) | closed | c=16 | 160 | 0.0% | 0.949 | 3.479 | 34.664 | 34.664 | 89 | 0.62 | 8% | 0.157 | 0.483 |
| Qwen3-4B fp16 (T4) | closed | c=32 | 160 | 0.0% | 1.639 | 11.270 | 64.446 | 64.446 | 100 | 0.70 | 2% | 0.140 | 0.483 |
| Qwen3-4B fp16 (T4) | closed | c=64 | 160 | 0.0% | 38.366 | 63.618 | 117.416 | 117.416 | 100 | 0.70 | 0% | 0.139 | 0.483 |
| Qwen3-4B AWQ int4 (T4) | closed | c=1 | 48 | 0.0% | 0.606 | 0.845 | 4.506 | 4.506 | 45 | 0.29 | 100% | 0.338 | 0.472 |
| Qwen3-4B AWQ int4 (T4) | closed | c=2 | 48 | 0.0% | 0.074 | 0.087 | 4.486 | 4.486 | 86 | 0.55 | 100% | 0.176 | 0.472 |
| Qwen3-4B AWQ int4 (T4) | closed | c=4 | 96 | 0.0% | 0.277 | 0.787 | 9.005 | 9.005 | 102 | 0.68 | 100% | 0.143 | 0.472 |
| Qwen3-4B AWQ int4 (T4) | closed | c=8 | 96 | 0.0% | 0.729 | 1.476 | 15.935 | 15.935 | 104 | 0.69 | 79% | 0.141 | 0.472 |
| Qwen3-4B AWQ int4 (T4) | closed | c=16 | 160 | 0.0% | 0.891 | 3.511 | 29.654 | 29.654 | 110 | 0.73 | 13% | 0.133 | 0.472 |
| Qwen3-4B AWQ int4 (T4) | closed | c=32 | 160 | 0.0% | 1.559 | 10.957 | 55.882 | 55.882 | 116 | 0.77 | 1% | 0.126 | 0.472 |
| Qwen3-4B AWQ int4 (T4) | closed | c=64 | 160 | 0.0% | 3.137 | 35.460 | 118.505 | 118.505 | 120 | 0.80 | 1% | 0.122 | 0.472 |

`total p95` is measured from request arrival, so it includes queueing; `E2E p95` starts at dispatch. In closed-loop runs the two are the same by construction.

### Side by side

| load | fp16 | awq |
|---|---|---|
| c=1 | 0.88 / 42 / 100% | 0.84 / 19 / 100% |
| c=2 | 0.93 / 51 / 100% | 0.09 / 23 / 100% |
| c=4 | 1.14 / 67 / 100% | 0.79 / 32 / 100% |
| c=8 | 1.45 / 102 / 48% | 1.48 / 73 / 79% |
| c=16 | 3.48 / 169 / 8% | 3.51 / 139 / 13% |
| c=32 | 11.27 / 304 / 2% | 10.96 / 266 / 1% |
| c=64 | 63.62 / 341 / 0% | 35.46 / 479 / 1% |

Each cell: TTFT p95 (s) / inter-token latency p50 (ms) / goodput.

## What the server saw

| config | load | KV mean | KV max | running max | waiting mean | waiting max | preempted | prefix hit | queue s (server) | TTFT s (server) | TTFT s (client) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B fp16 (T4) | c=1 | 4% | 5% | 1 | 0.0 | 0 | 0 | 39% | 0.000 | 0.622 | 0.610 |
| Qwen3-4B fp16 (T4) | c=2 | 6% | 8% | 2 | 0.0 | 0 | 0 | 39% | 0.000 | 0.692 | 0.664 |
| Qwen3-4B fp16 (T4) | c=4 | 11% | 14% | 4 | 0.0 | 0 | 0 | 35% | 0.000 | 0.796 | 0.743 |
| Qwen3-4B fp16 (T4) | c=8 | 21% | 26% | 8 | 0.0 | 4 | 0 | 35% | 0.044 | 1.085 | 0.874 |
| Qwen3-4B fp16 (T4) | c=16 | 41% | 50% | 16 | 0.1 | 4 | 0 | 33% | 0.051 | 1.447 | 1.342 |
| Qwen3-4B fp16 (T4) | c=32 | 77% | 94% | 32 | 0.5 | 15 | 0 | 33% | 0.679 | 2.740 | 2.714 |
| Qwen3-4B fp16 (T4) | c=64 | 85% | 100% | 39 | 20.4 | 47 | 22 | 33% | 30.710 | 33.350 | 34.881 |
| Qwen3-4B AWQ int4 (T4) | c=1 | 2% | 3% | 1 | 0.0 | 0 | 0 | 39% | 0.000 | 0.571 | 0.560 |
| Qwen3-4B AWQ int4 (T4) | c=2 | 3% | 4% | 2 | 0.0 | 0 | 0 | 99% | 0.000 | 0.069 | 0.073 |
| Qwen3-4B AWQ int4 (T4) | c=4 | 6% | 7% | 4 | 0.0 | 0 | 0 | 68% | 0.000 | 0.369 | 0.395 |
| Qwen3-4B AWQ int4 (T4) | c=8 | 11% | 12% | 8 | 0.1 | 4 | 0 | 35% | 0.047 | 1.002 | 0.780 |
| Qwen3-4B AWQ int4 (T4) | c=16 | 20% | 25% | 16 | 0.1 | 4 | 0 | 33% | 0.037 | 1.423 | 1.323 |
| Qwen3-4B AWQ int4 (T4) | c=32 | 38% | 47% | 32 | 0.6 | 15 | 0 | 33% | 0.675 | 2.716 | 2.686 |
| Qwen3-4B AWQ int4 (T4) | c=64 | 69% | 91% | 64 | 4.2 | 49 | 0 | 33% | 5.304 | 8.654 | 8.955 |

Scraped from vLLM's `/metrics` every second during each run. **KV** is the share of the KV cache in use: when it nears 100%, new requests wait (**waiting**) and running ones can be **preempted** (evicted and recomputed later), which is where tail latency comes from. **prefix hit** is the share of prompt tokens served from cached prefixes (the shared system prompt). Server TTFT starts when vLLM receives the request; the client's also includes the client and the HTTP hop, so the gap between them is overhead outside the engine.

## Charts

### Throughput vs concurrency

![Throughput vs concurrency](charts/throughput_vs_concurrency.png)

### Latency vs concurrency

![Latency vs concurrency](charts/latency_vs_concurrency.png)

### TTFT percentiles

![TTFT percentiles](charts/latency_percentiles.png)

### What the server saw

![What the server saw](charts/server_state_vs_load.png)

### Quality vs latency

![Quality vs latency](charts/quality_vs_latency.png)

### Cost per 1,000 requests

![Cost per 1,000 requests](charts/cost_per_1k_requests.png)

## Answer quality

| config | n | composite | token F1 | citation grounding | number agreement |
|---|---|---|---|---|---|
| fp16 | 32 | 0.483 | 0.267 | 1.000 | 0.452 |
| awq | 32 | 0.472 | 0.262 | 1.000 | 0.415 |

Composite = 0.45 x token F1 + 0.25 x citation grounding + 0.25 x number agreement + 0.05 x exact match. Weights are a judgement call and are stated here so a reader can re-weight them.

### Quality gate

| candidate | baseline | delta | 95% CI | significant | verdict |
|---|---|---|---|---|---|
| awq | fp16 | -0.012 | -0.033 to +0.010 | no | PASS |

Deltas are paired bootstrap over per-item composite scores (2,000 resamples). A difference whose interval spans zero is not treated as a difference.

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


### Reproducing this

```bash
vantage sweep -c /kaggle/working/vantage_repo/configs/sweeps/kaggle-t4.yaml
```

Raw per-request records are in `raw.jsonl`; every chart and every number above is derived from that file plus `summaries.json`.


---

# Vantage report: kaggle-t4-offered-load

Open-loop (Poisson) offered load, Qwen3-4B fp16 vs AWQ int4 on vLLM, one Kaggle T4 each, Vectora workload.

- Run: `2026-10-27T10:03:38+00:00` UTC
- Commit: `2cb7094` (working tree dirty)
- Host: Linux-6.18.55+-x86_64-with-glibc2.39, Python 3.13.15
- GPUs: Tesla T4, 15360 MiB, 580.178.04, Tesla T4, 15360 MiB, 580.178.04
- Workload: `data/workload.vectora.jsonl` -- 365 items, median prompt ~663 tokens
- SLO: {"ttft_p95_s": 2.0, "total_p95_s": 15.0, "max_error_rate": 0.01}

## Findings

- Cheapest configuration measured: **Qwen3-4B AWQ int4 (T4), Poisson arrivals** at $0.094 per 1,000 requests, 3.4x cheaper than Qwen3-4B fp16 (T4), Poisson arrivals.
- **Break-even:** self-hosting beats the API above **0.40 req/s** sustained (34,535 requests/day).

## Self-host vs API

API costs $0.00024/request at 1033 in / 147 out tokens. A $0.35/hour GPU costs $0.00010/second regardless of load, so it pays for itself above 0.40 req/s sustained (~34,535 requests/day) -- provided the GPU can actually serve that rate within SLO.

## Results

| config | mode | load | n | err | TTFT p50 | TTFT p95 | E2E p95 | total p95 | tok/s | req/s | goodput | $/1k | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B fp16 (T4), Poisson arrivals | open | 0.25 qps | 30 | 0.0% | 0.725 | 1.432 | 17.026 | 17.027 | 44 | 0.31 | 80% | 0.315 | -- |
| Qwen3-4B fp16 (T4), Poisson arrivals | open | 0.5 qps | 30 | 0.0% | 0.184 | 0.248 | 15.050 | 15.051 | 80 | 0.56 | 93% | 0.175 | -- |
| Qwen3-4B fp16 (T4), Poisson arrivals | open | 1 qps | 71 | 0.0% | 1.488 | 17.488 | 59.676 | 59.677 | 107 | 0.74 | 18% | 0.132 | -- |
| Qwen3-4B fp16 (T4), Poisson arrivals | open | 1.5 qps | 99 | 0.0% | 31.362 | 60.869 | 94.408 | 94.409 | 101 | 0.71 | 1% | 0.137 | -- |
| Qwen3-4B fp16 (T4), Poisson arrivals | open | 2 qps | 143 | 0.0% | 49.922 | 115.493 | 140.071 | 140.072 | 102 | 0.71 | 0% | 0.136 | -- |
| Qwen3-4B fp16 (T4), Poisson arrivals | open | 3 qps | 195 | 0.0% | 90.098 | 193.453 | 220.926 | 220.927 | 100 | 0.70 | 0% | 0.140 | -- |
| Qwen3-4B fp16 (T4), Poisson arrivals | open | 4 qps | 247 | 0.0% | 126.116 | 253.110 | 286.778 | 286.779 | 102 | 0.71 | 0% | 0.137 | -- |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | open | 0.25 qps | 30 | 0.0% | 0.625 | 1.195 | 8.510 | 8.511 | 50 | 0.33 | 100% | 0.299 | -- |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | open | 0.5 qps | 30 | 0.0% | 0.097 | 0.137 | 7.835 | 7.836 | 94 | 0.62 | 100% | 0.157 | -- |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | open | 1 qps | 71 | 0.0% | 1.019 | 3.111 | 50.266 | 50.267 | 134 | 0.86 | 30% | 0.113 | -- |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | open | 1.5 qps | 99 | 0.0% | 0.930 | 6.586 | 65.713 | 65.714 | 155 | 1.04 | 3% | 0.094 | -- |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | open | 2 qps | 143 | 0.0% | 13.266 | 83.365 | 126.946 | 126.947 | 122 | 0.81 | 1% | 0.120 | -- |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | open | 3 qps | 195 | 0.0% | 76.434 | 159.505 | 202.341 | 202.342 | 116 | 0.77 | 0% | 0.126 | -- |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | open | 4 qps | 247 | 0.0% | 107.507 | 216.571 | 265.950 | 265.951 | 117 | 0.77 | 0% | 0.126 | -- |

`total p95` is measured from request arrival, so it includes queueing; `E2E p95` starts at dispatch. In closed-loop runs the two are the same by construction.

### Side by side

| load | fp16 | awq |
|---|---|---|
| 0.25 qps | 1.43 / 67 / 80% | 1.20 / 27 / 100% |
| 0.5 qps | 0.25 / 71 / 93% | 0.14 / 32 / 100% |
| 1 qps | 17.49 / 249 / 18% | 3.11 / 195 / 30% |
| 1.5 qps | 60.87 / 321 / 1% | 6.59 / 319 / 3% |
| 2 qps | 115.49 / 336 / 0% | 83.37 / 532 / 1% |
| 3 qps | 193.45 / 345 / 0% | 159.51 / 592 / 0% |
| 4 qps | 253.11 / 343 / 0% | 216.57 / 597 / 0% |

Each cell: TTFT p95 (s) / inter-token latency p50 (ms) / goodput.

## What the server saw

| config | load | KV mean | KV max | running max | waiting mean | waiting max | preempted | prefix hit | queue s (server) | TTFT s (server) | TTFT s (client) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B fp16 (T4), Poisson arrivals | 0.25 qps | 8% | 22% | 8 | 0.0 | 0 | 0 | 45% | 0.000 | 0.703 | 0.706 |
| Qwen3-4B fp16 (T4), Poisson arrivals | 0.5 qps | 10% | 30% | 11 | 0.0 | 0 | 0 | 99% | 0.000 | 0.161 | 0.185 |
| Qwen3-4B fp16 (T4), Poisson arrivals | 1 qps | 46% | 100% | 36 | 1.7 | 13 | 7 | 63% | 2.875 | 4.109 | 4.571 |
| Qwen3-4B fp16 (T4), Poisson arrivals | 1.5 qps | 62% | 100% | 39 | 12.7 | 45 | 10 | 35% | 22.865 | 24.967 | 26.935 |
| Qwen3-4B fp16 (T4), Poisson arrivals | 2 qps | 71% | 100% | 39 | 28.4 | 89 | 13 | 33% | 47.797 | 49.888 | 52.647 |
| Qwen3-4B fp16 (T4), Poisson arrivals | 3 qps | 78% | 100% | 39 | 52.2 | 141 | 30 | 32% | 85.886 | 88.179 | 91.777 |
| Qwen3-4B fp16 (T4), Poisson arrivals | 4 qps | 81% | 100% | 39 | 77.1 | 193 | 40 | 32% | 121.854 | 124.062 | 128.067 |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | 0.25 qps | 2% | 8% | 6 | 0.0 | 0 | 0 | 45% | 0.000 | 0.584 | 0.568 |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | 0.5 qps | 4% | 10% | 7 | 0.0 | 0 | 0 | 99% | 0.000 | 0.083 | 0.097 |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | 1 qps | 26% | 63% | 46 | 0.0 | 1 | 0 | 63% | 0.002 | 1.050 | 1.171 |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | 1.5 qps | 49% | 100% | 76 | 0.2 | 3 | 3 | 66% | 0.111 | 1.780 | 1.901 |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | 2 qps | 68% | 100% | 80 | 21.2 | 67 | 12 | 37% | 28.401 | 31.989 | 33.755 |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | 3 qps | 77% | 100% | 80 | 46.7 | 119 | 21 | 32% | 64.832 | 68.425 | 71.219 |
| Qwen3-4B AWQ int4 (T4), Poisson arrivals | 4 qps | 81% | 100% | 80 | 72.6 | 172 | 31 | 32% | 99.347 | 102.964 | 106.293 |

Scraped from vLLM's `/metrics` every second during each run. **KV** is the share of the KV cache in use: when it nears 100%, new requests wait (**waiting**) and running ones can be **preempted** (evicted and recomputed later), which is where tail latency comes from. **prefix hit** is the share of prompt tokens served from cached prefixes (the shared system prompt). Server TTFT starts when vLLM receives the request; the client's also includes the client and the HTTP hop, so the gap between them is overhead outside the engine.

## Charts

### Latency vs offered load (open loop)

![Latency vs offered load (open loop)](charts/latency_vs_offered_load.png)

### Cost per 1,000 requests

![Cost per 1,000 requests](charts/cost_per_1k_requests.png)

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


### Reproducing this

```bash
vantage sweep -c /kaggle/working/vantage_repo/configs/sweeps/kaggle-t4-offered-load.yaml
```

Raw per-request records are in `raw.jsonl`; every chart and every number above is derived from that file plus `summaries.json`.


---

# Semantic cache

```

$ vantage cache-sim -w data/workload.vectora.jsonl --embedder st --repeats 5 --out /kaggle/working/kaggle_t4_2026-10-27_0902/cache_sim.json
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 2809.29it/s]
/kaggle/working/vantage_repo/src/vantage/cache/embedders.py:99: FutureWarning: The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.
  self.dim = self._model.get_sentence_embedding_dimension()
  threshold 0.70  hit  80.0%  false-hit   6.6%  served quality 0.950
  threshold 0.75  hit  78.2%  false-hit   3.6%  served quality 0.973
  threshold 0.80  hit  76.1%  false-hit   1.3%  served quality 0.990
  threshold 0.84  hit  74.0%  false-hit   0.9%  served quality 0.993
  threshold 0.88  hit  71.5%  false-hit   0.1%  served quality 0.999
  threshold 0.90  hit  70.2%  false-hit   0.0%  served quality 1.000
  threshold 0.92  hit  69.3%  false-hit   0.0%  served quality 1.000
  threshold 0.94  hit  67.3%  false-hit   0.0%  served quality 1.000
  threshold 0.96  hit  63.6%  false-hit   0.0%  served quality 1.000
  threshold 0.98  hit  50.0%  false-hit   0.0%  served quality 1.000
recommended threshold: 0.8 -- highest hit rate (76.1%) among thresholds whose 
false-hit rate stays at or below 2.0%; at this setting 1388 of 1825 model calls 
are avoided

```
