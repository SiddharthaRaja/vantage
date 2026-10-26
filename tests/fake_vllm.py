"""A stand-in for `vllm serve`, for tests: no GPU, standard library only.

Speaks the parts of vLLM's API Vantage uses: GET /v1/models, streaming POST
/v1/chat/completions (with a usage block), and GET /metrics in vLLM V1's
Prometheus names. A batch cap (--max-num-seqs) makes requests beyond it wait,
so the waiting gauge and the queue-time histogram move like the real thing.

    python tests/fake_vllm.py --port 8123 --max-num-seqs 4 --tokens 6
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="fake-model")
ap.add_argument("--max-num-seqs", type=int, default=8)
ap.add_argument("--tokens", type=int, default=6)
ap.add_argument("--token-delay", type=float, default=0.005)
ap.add_argument("--startup-delay", type=float, default=0.3)
ap.add_argument("--fail", action="store_true", help="exit during start-up")
args, _ = ap.parse_known_args()      # accept (and ignore) the rest of vllm serve's flags

LOCK = threading.Lock()
SLOTS = threading.Semaphore(args.max_num_seqs)
STATE = {"running": 0, "waiting": 0, "prompt_tokens": 0, "generation_tokens": 0,
         "queue_sum": 0.0, "queue_count": 0, "ttft_sum": 0.0, "ttft_count": 0,
         "prefix_hits": 0, "prefix_queries": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, json.dumps({"data": [{"id": args.model}]}))
        elif self.path == "/metrics":
            with LOCK:
                s = dict(STATE)
            usage = min(1.0, s["running"] / args.max_num_seqs)
            lines = [
                "# HELP vllm:num_requests_running running",
                f'vllm:num_requests_running{{model_name="{args.model}"}} {s["running"]}',
                f'vllm:num_requests_waiting{{model_name="{args.model}"}} {s["waiting"]}',
                f'vllm:kv_cache_usage_perc{{model_name="{args.model}"}} {usage}',
                f'vllm:num_preemptions_total{{model_name="{args.model}"}} 0.0',
                f'vllm:prefix_cache_hits_total{{model_name="{args.model}"}} {s["prefix_hits"]}',
                f'vllm:prefix_cache_queries_total{{model_name="{args.model}"}} {s["prefix_queries"]}',
                f'vllm:prompt_tokens_total{{model_name="{args.model}"}} {s["prompt_tokens"]}',
                f'vllm:generation_tokens_total{{model_name="{args.model}"}} {s["generation_tokens"]}',
                f'vllm:request_queue_time_seconds_sum{{model_name="{args.model}"}} {s["queue_sum"]}',
                f'vllm:request_queue_time_seconds_count{{model_name="{args.model}"}} {s["queue_count"]}',
                f'vllm:time_to_first_token_seconds_sum{{model_name="{args.model}"}} {s["ttft_sum"]}',
                f'vllm:time_to_first_token_seconds_count{{model_name="{args.model}"}} {s["ttft_count"]}',
            ]
            self._send(200, "\n".join(lines) + "\n", "text/plain")
        else:
            self._send(404, "{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        prompt = " ".join(m.get("content", "") for m in body.get("messages", []))
        n_prompt = max(1, len(prompt.split()))
        t0 = time.perf_counter()
        with LOCK:
            STATE["waiting"] += 1
        SLOTS.acquire()
        queued = time.perf_counter() - t0
        with LOCK:
            STATE["waiting"] -= 1
            STATE["running"] += 1
            STATE["queue_sum"] += queued
            STATE["queue_count"] += 1
            STATE["prompt_tokens"] += n_prompt
            STATE["prefix_queries"] += n_prompt
            STATE["prefix_hits"] += n_prompt // 2
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            n = min(int(body.get("max_tokens", args.tokens)), args.tokens)
            for i in range(n):
                time.sleep(args.token_delay)
                if i == 0:
                    with LOCK:
                        STATE["ttft_sum"] += time.perf_counter() - t0
                        STATE["ttft_count"] += 1
                chunk = {"choices": [{"delta": {"content": f"tok{i} "}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            final = {"choices": [{"delta": {}, "finish_reason": "length"}],
                     "usage": {"prompt_tokens": n_prompt, "completion_tokens": n}}
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            with LOCK:
                STATE["generation_tokens"] += n
        finally:
            with LOCK:
                STATE["running"] -= 1
            SLOTS.release()


if __name__ == "__main__":
    time.sleep(args.startup_delay)
    if args.fail:
        raise SystemExit(3)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
