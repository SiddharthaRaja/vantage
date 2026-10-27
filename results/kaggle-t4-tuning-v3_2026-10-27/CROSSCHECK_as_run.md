# Cross-check: Vantage vs `vllm bench serve` (vLLM 0.32.0, flags ['--custom-output-len', '192', '--skip-chat-template'])

Same server, same input tokens. Throughput can differ where the two tools stop answers at different lengths; per-token latency should not.

| concurrency | metric | Vantage | vllm bench | Vantage / vllm |
|---|---|---|---|---|
| 1 | TTFT p50 (ms) | 659.6 | 672.0 | 0.98 |
| 1 | TTFT p99 (ms) | 1,225.8 | 939.7 | 1.30 |
| 1 | ITL p50 (ms) | 42.7 | 42.5 | 1.01 |
| 1 | output tok/s | 21.4 | 21.2 | 1.01 |
| 8 | TTFT p50 (ms) | 840.7 | 868.4 | 0.97 |
| 8 | TTFT p99 (ms) | 1,916.7 | 9,661.9 | 0.20 |
| 8 | ITL p50 (ms) | 100.8 | 73.5 | 1.37 |
| 8 | output tok/s | 74.8 | 70.9 | 1.05 |
| 32 | TTFT p50 (ms) | 1,665.3 | 1,629.7 | 1.02 |
| 32 | TTFT p99 (ms) | 15,407.4 | 25,363.1 | 0.61 |
| 32 | ITL p50 (ms) | 301.1 | 175.2 | 1.72 |
| 32 | output tok/s | 100.8 | 98.5 | 1.02 |
