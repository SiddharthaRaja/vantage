# Kaggle 2x T4, tuning v3 (2026-10-27)

Notebook `vantage_t4_tuning_v3`, Vantage `e228ab0`, vLLM 0.32.0 (isolated
Python 3.12). Qwen3-4B-Instruct-2507 on one T4 each, the 365-prompt Vectora
workload (about 1,030 prompt tokens and 145 answer tokens per request). Six
servers, each started by the sweep with one setting changed
(`configs/sweeps/kaggle-t4-tune.yaml`; exact commands in
`kaggle-t4-tune/servers.json`), every load point preceded by a prefix-cache
reset (`prefix_cache_reset: true` on all 30 points). Then `vllm bench serve`
against the baseline server. 2.8 hours.

SLO: TTFT p95 ≤ 2 s and total p95 ≤ 15 s. Cost at an assumed $0.35/h per T4.

## Where fp16 breaks, and why

| c | TTFT p95 | per-token | total p95 | tok/s | goodput | KV mean / max | waiting | preempted |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.87 s | 43 ms | 9.0 s | 21 | 100% | 3% / 5% | 0 | 0 |
| 4 | 1.21 s | 67 ms | 14.2 s | 55 | 99% | 10% / 12% | 0 | 0 |
| 8 | 1.64 s | 101 ms | 21.2 s | 75 | 47% | 19% / 23% | 0 | 0 |
| 16 | 3.63 s | 168 ms | 34.7 s | 90 | 6% | 36% / 44% | 0.1 | 0 |
| 32 | 11.37 s | 301 ms | 63.5 s | 101 | 2% | 67% / 82% | 0.6 | 0 |
| 64 | 61.74 s | 375 ms | 122.1 s | 103 | 0% | 83% / 100% | 17.2 | 19 |

The SLO breaks at concurrency 8 with the KV cache under a quarter full and
nothing waiting: the cost of each decode step grows with the batch (43 → 301
ms per token) because the T4 is compute-bound for this model (no
FlashAttention on compute capability 7.5). Memory becomes the limit only at
64, where the cache fills, 17 requests wait inside vLLM and 19 are preempted.

## One setting at a time (fp16)

| variant | change | c=16 TTFT p95 | c=32 tok/s | c=64 preempted | verdict |
|---|---|---|---|---|---|
| base | none | 3.63 s | 101 | 19 | reference |
| noprefix | prefix caching off | **6.80 s** | **86** | 17 (15 already at c=32) | worse everywhere: reusing the shared system prompt (≈32% of prompt tokens) is worth ~15% throughput |
| mem95 | `--gpu-memory-utilization 0.95` | 3.64 s | 101 | 16 | helps only past saturation (52 running at c=64 vs 45) |
| seqs32 | `--max-num-seqs 32` | 3.60 s | 101 | **0** | at c=64: no preemption and faster tokens (305 vs 375 ms) for those running; the rest queue |
| chunk512 | `--max-num-batched-tokens 512` | **2.86 s** | 105 | 15 | best fp16 variant: −21% TTFT tail at c=16, +4% throughput; near run-to-run noise |

No setting moves the SLO capacity (concurrency 4 for every variant), which is
what a compute-bound GPU predicts.

## fp16 vs AWQ int4, cold cache

| | fp16 | AWQ int4 |
|---|---|---|
| per-token latency at c=1 | 43 ms | **19 ms** |
| goodput at c=8 | 47% | **76%** |
| throughput at c=64 | 103 tok/s | **119 tok/s** |
| c=64: running / waiting / preempted | 45 / 17.2 / 19 | 64 / 4.5 / 0 |
| $ per 1,000 requests at c=4 (inside the SLO) | $0.253 | **$0.172** |

AWQ reads a quarter of the weight bytes per token, which is what a
single-stream decode is limited by, and its smaller weights leave twice the
KV cache (60,272 vs 29,728 tokens). At c=64 its tokens are slower (491 vs 375
ms) because it runs all 64 requests at once where fp16 can hold 45. Quality
gate (baseline v3, 32 references): no regression detectable, delta −0.012,
95% CI [−0.033, +0.010].

## Cross-check against `vllm bench serve`

| c | TTFT p50 | per-token latency vs TPOT | output tok/s |
|---|---|---|---|
| 1 | 0.98 | 1.00 | 1.01 |
| 8 | 0.97 | 0.99 | 1.05 |
| 32 | 1.02 | 1.00 | 1.02 |

Ratios Vantage / vLLM. Medians agree within 3% and per-token latency within
1%; TTFT p99 does not (0.20 to 1.30), because at 48 to 160 requests p99 is
one or two requests and vllm bench has no warm-up. The table the notebook
wrote (`CROSSCHECK_as_run.md`) compared Vantage's per-request mean gap with
vLLM's pooled ITL, a different statistic; `CROSSCHECK.md` compares like with
like.

## Repeatability

The `base` curve repeats baseline v3's fp16 curve: throughput within 1–2% at
every point, TTFT p95 within about 15% (c=8: 1.64 s vs 1.45 s). Single runs
per configuration, so differences smaller than that are not findings.

## What changed from baseline v3

Cold cache per point: AWQ's c=2 TTFT p50 is 0.65 s (v3: 0.07 s, warm cache),
c=4 0.67 s (v3: 0.28 s), and AWQ inside the SLO costs $0.172 per 1,000 requests
(v3: $0.143). fp16 was barely affected (its smaller cache had already evicted
the earlier points' prompts).
