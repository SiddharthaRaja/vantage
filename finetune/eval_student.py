#!/usr/bin/env python3
"""The experiment the whole fine-tuning branch exists to run.

Three models, one workload, one eval set, one piece of hardware:

    teacher   the hosted model that produced the training data
    base      the small open model, untouched
    student   the same small model with our adapter

For each: answer quality on the held-out split, serving latency and throughput
under identical load, and cost per thousand requests. The claim worth making is
not "I fine-tuned a model" -- it is "the fine-tuned 4B recovers N% of the
teacher's quality on this task at 1/Xth the cost, and here is the measurement".

A fine-tune that loses to the base model is also a result, and this script will
tell you so rather than letting you quietly not mention it.

Usage
-----
    # serve base and adapter from one vLLM process:
    vllm serve Qwen/Qwen3-4B-Instruct --enable-lora \
        --lora-modules vantage-ft=outputs/qwen3-4b-vantage-lora --port 8000

    python finetune/eval_student.py \
        --teacher openai.gpt-4o-mini \
        --base vllm.qwen3-4b \
        --student vllm.qwen3-4b-lora \
        --eval finetune/data/test.questions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vantage.backends.registry import build_backend  # noqa: E402
from vantage.config import load_dotenv  # noqa: E402
from vantage.metrics.collector import summarize_run  # noqa: E402
from vantage.metrics.cost import cost_for_summary  # noqa: E402
from vantage.models import SLO, LoadConfig  # noqa: E402
from vantage.quality.harness import compare_quality, run_quality_eval  # noqa: E402
from vantage.workload.dataset import Workload  # noqa: E402
from vantage.workload.generator import generate_load  # noqa: E402


async def measure(name: str, backend_name: str, workload: Workload, args) -> dict:
    backend = build_backend(backend_name, {"config_id": name})
    async with backend:
        health = await backend.health()
        if not health.ok:
            print(f"  {name}: backend unhealthy -- {health.detail}")
            return {"name": name, "backend": backend_name, "error": health.detail}

        quality = await run_quality_eval(
            backend, workload, config_id=name, concurrency=args.quality_concurrency,
            max_tokens=args.max_tokens, progress=True,
        )
        requests = workload.sample(n=args.requests, max_tokens=args.max_tokens, seed=7)
        load = LoadConfig(mode="closed", concurrency=args.concurrency,
                          n_requests=args.requests, warmup_requests=args.warmup)
        run = await generate_load(backend, requests, load, progress=True)
        summary = summarize_run(run, config_id=name, backend=backend_name, label=name, slo=SLO())
        summary.cost = cost_for_summary(summary, backend.pricing())

    return {"name": name, "backend": backend_name, "quality": quality, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default="")
    parser.add_argument("--base", default="vllm.qwen3-4b")
    parser.add_argument("--student", default="vllm.qwen3-4b-lora")
    parser.add_argument("--eval", default="finetune/data/test.questions.jsonl")
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--quality-concurrency", type=int, default=4)
    parser.add_argument("--out", default="results/finetune-compare.json")
    args = parser.parse_args()

    load_dotenv()
    workload = Workload.from_jsonl(args.eval)
    print(f"evaluation set: {args.eval} -- {json.dumps(workload.stats())}\n")

    targets = [("base", args.base), ("student", args.student)]
    if args.teacher:
        targets.insert(0, ("teacher", args.teacher))

    results = [asyncio.run(measure(name, backend, workload, args)) for name, backend in targets]
    usable = [r for r in results if "error" not in r]
    if not usable:
        sys.exit("no backend was reachable; nothing to compare")

    print("\n" + "=" * 92)
    header = f"{'model':<12}{'quality':>9}{'F1':>8}{'cite':>8}{'p95 s':>9}{'tok/s':>10}{'req/s':>9}{'$/1k':>10}"
    print(header)
    print("-" * 92)
    for r in usable:
        q, s = r["quality"], r["summary"]
        print(
            f"{r['name']:<12}{q.composite:>9.3f}{q.token_f1:>8.3f}{q.citation_grounding:>8.3f}"
            f"{(s.total.p95 or 0):>9.3f}{s.output_tokens_per_s:>10,.0f}{s.achieved_qps:>9.2f}"
            f"{(s.cost.usd_per_1k_requests or float('nan')):>10.3f}"
        )
    print("=" * 92 + "\n")

    by_name = {r["name"]: r for r in usable}
    comparisons = []
    if "base" in by_name and "student" in by_name:
        c = compare_quality(by_name["base"]["quality"], by_name["student"]["quality"], max_drop=0.0)
        comparisons.append(("student vs base", c))
        direction = "beats" if c.delta > 0 else "loses to"
        print(f"student {direction} base: {c.detail}")
        if c.delta <= 0:
            print(
                "  The fine-tune did not improve on the base model. Before training again, check:\n"
                "   - is the serving prompt identical to the training prompt?\n"
                "   - did the adapter actually load (vLLM logs the lora-modules it registered)?\n"
                "   - is the training set large and clean enough (how many examples survived filtering)?\n"
                "   - is the task one a 4B model can do at all, or does it need the capability the teacher has?"
            )
    if "teacher" in by_name and "student" in by_name:
        c = compare_quality(by_name["teacher"]["quality"], by_name["student"]["quality"], max_drop=0.03)
        comparisons.append(("student vs teacher", c))
        recovered = by_name["student"]["quality"].composite / max(by_name["teacher"]["quality"].composite, 1e-9)
        print(f"student recovers {recovered:.1%} of teacher quality -- {c.detail}")
        t_cost = by_name["teacher"]["summary"].cost.usd_per_1k_requests
        s_cost = by_name["student"]["summary"].cost.usd_per_1k_requests
        if t_cost and s_cost:
            print(f"at {t_cost / s_cost:.1f}x lower cost per 1,000 requests")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "eval_set": args.eval,
                "load": {"requests": args.requests, "concurrency": args.concurrency},
                "models": [
                    {
                        "name": r["name"],
                        "backend": r["backend"],
                        "quality": r["quality"].model_dump(exclude={"per_item"}),
                        "summary": r["summary"].model_dump(exclude={"slo"}),
                    }
                    for r in usable
                ],
                "comparisons": [{"name": n, **c.to_dict()} for n, c in comparisons],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
