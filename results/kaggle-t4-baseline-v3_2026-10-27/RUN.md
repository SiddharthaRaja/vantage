# Kaggle 2x T4, baseline v3 (2026-10-27)

Notebook `vantage_t4_baseline_v3`, Vantage `2cb7094`, vLLM 0.32.0 in an
isolated Python 3.12 environment. Qwen3-4B-Instruct-2507 fp16 on T4 #0, AWQ
int4 (cyankiwi) on T4 #1, each a whole GPU. Workload: 365 Vectora prompts from
qasper-300 (1,174 prompt tokens read by both servers in the sanity check).
1.9 hours.

| step | result |
|---|---|
| kaggle-t4 (closed loop, c = 1..64, both precisions) | done, 55.6 min |
| quality gate (32 references) | AWQ passes: delta −0.012, 95% CI [−0.033, +0.010] |
| kaggle-t4-offered-load (Poisson, 0.25..4 req/s) | done, 49 min |
| cross-check vs `vllm bench serve` | **failed**: vLLM's bench extras (pandas) were not installed |
| cache-sim (CPU) | recommended threshold 0.80: 76% hits, 1.3% false hits |

## Headline numbers

| | fp16 | AWQ int4 |
|---|---|---|
| model memory / KV cache | 7.64 GiB / 29,728 tokens | 3.29 GiB / 60,272 tokens |
| ITL p50 at c=1 | 42 ms | 19 ms |
| peak throughput | ~100 tok/s (0.70 req/s) | ~120 tok/s (0.80 req/s) |
| highest concurrency with 100% goodput (TTFT p95 ≤ 2 s, total p95 ≤ 15 s) | 4 | 4 |
| c=64: running / waiting / preempted (server metrics) | 39 / 20 / 22 | 64 / 4 / 0 |
| open-loop saturation | ~0.75 req/s | ~1.0 req/s |
| $ per 1,000 requests at the last point inside the SLO ($0.35/h assumed) | 0.253 (c=4) | 0.143 (c=4) |

Self-hosting beats gpt-4o-mini list price above 0.40 req/s, and one T4 holds
the SLO only up to about 0.68 req/s (AWQ).

## Caveat: warm prefix cache at low load

Every load point drew its prompts in the same seeded order, so a point
re-sent prompts the previous point had just processed. The server metrics
show it: prefix-cache hit rate 99% at AWQ c=2 and 68% at c=4 (fp16, with
half the KV cache, had evicted them: 39%). The AWQ c=2 and c=4 first-token
times (p50 0.07 s and 0.28 s) and the 0.5 req/s open-loop points are
flattered. High-load points settle at 33%, the share of each prompt that is
the shared system prompt, as expected. Fixed afterwards (cache cleared before
every point); the tuning run re-measures both curves.
