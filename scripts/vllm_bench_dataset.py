"""
Convert a Vantage workload into a dataset for vLLM's own benchmark.

The cross-check only means something if both tools send the *same tokens*.
Vantage sends chat messages (system + user) to /v1/chat/completions, where
the server applies the model's chat template. ``vllm bench serve`` with
``--dataset-name custom`` sends plain prompts. So this applies the same chat
template here, with the model's own tokenizer, and the benchmark is run with
``--skip-chat-template`` against /v1/completions: identical input tokens.

Run on the GPU box (it needs the tokenizer, ``pip install transformers``):

    python scripts/vllm_bench_dataset.py --model Qwen/Qwen3-4B-Instruct-2507 \\
        --workload data/workload.vectora.jsonl --out data/vllm_bench.vectora.jsonl

It prints input-token statistics; compare them with the prompt tokens Vantage
reports for the same workload (they should match to within a token or two).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def conversations(workload: Path) -> list[list[dict]]:
    """The exact messages Vantage sends, built by Vantage's own code."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from vantage.workload.dataset import Workload

    wl = Workload.from_jsonl(workload)
    out = []
    for item in wl.items:
        msgs = [{"role": "system", "content": wl.system}] if wl.system else []
        out.append(msgs + [{"role": "user", "content": item.prompt()}])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="HF id of the served model (for its tokenizer)")
    ap.add_argument("--workload", type=Path, default=Path("data/workload.vectora.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/vllm_bench.vectora.jsonl"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    convs = conversations(args.workload)
    lengths = []
    with args.out.open("w", encoding="utf-8") as fh:
        for msgs in convs:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))
            fh.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")
    lengths.sort()
    print(f"wrote {len(convs)} prompts to {args.out}")
    print(f"input tokens: min {lengths[0]}, median {int(statistics.median(lengths))}, "
          f"p95 {lengths[int(len(lengths) * 0.95) - 1]}, max {lengths[-1]}")


if __name__ == "__main__":
    main()
