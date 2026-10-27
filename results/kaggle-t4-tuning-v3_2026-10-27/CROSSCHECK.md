# Cross-check: Vantage vs `vllm bench serve` (vLLM 0.32.0, flags ['--custom-output-len', '192', '--skip-chat-template'])

Same server (fp16, baseline settings), same input tokens, prefix cache cleared before each run. Vantage's per-token latency is each request's mean gap between tokens, compared with vLLM's TPOT (the same definition); the table the notebook wrote compared it with vLLM's pooled ITL, a different statistic, and is kept as CROSSCHECK_as_run.md. TTFT p99 is the single worst request or two at these sample sizes, and vllm bench has no warm-up, so tails differ more than medians.

| concurrency | metric | Vantage | vllm bench | Vantage / vllm |
|---|---|---|---|---|
| 1 | TTFT p50 (ms) | 659.6 | 672.0 | 0.98 |
| 1 | TTFT p99 (ms) | 1,225.8 | 939.7 | 1.30 |
| 1 | per-token latency p50 (ms) vs TPOT | 42.7 | 42.5 | 1.00 |
| 1 | output tok/s | 21.4 | 21.2 | 1.01 |
| 8 | TTFT p50 (ms) | 840.7 | 868.4 | 0.97 |
| 8 | TTFT p99 (ms) | 1,916.7 | 9,661.9 | 0.20 |
| 8 | per-token latency p50 (ms) vs TPOT | 100.8 | 101.3 | 0.99 |
| 8 | output tok/s | 74.8 | 70.9 | 1.05 |
| 32 | TTFT p50 (ms) | 1,665.3 | 1,629.7 | 1.02 |
| 32 | TTFT p99 (ms) | 15,407.4 | 25,363.1 | 0.61 |
| 32 | per-token latency p50 (ms) vs TPOT | 301.1 | 300.3 | 1.00 |
| 32 | output tok/s | 100.8 | 98.5 | 1.02 |
