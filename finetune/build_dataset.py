#!/usr/bin/env python3
"""Build a supervised fine-tuning set by distilling a teacher model.

The idea: a large hosted model already answers your task well, and you pay for
it on every request forever. Have it answer a few thousand examples once, keep
only the answers that survive automated checks, and train a small open model on
those. You then own a model that does one job nearly as well, runs on hardware
you control, and costs a fraction per token.

The pipeline:

    corpus (question + retrieved context)
      -> teacher answers each one
      -> reject answers that fail grounding checks      <- this step is the work
      -> deduplicate near-identical examples
      -> split train / validation / test by hashed id   <- this step is the rigour
      -> write chat-format JSONL

Two rules that decide whether the result is worth anything:

1. **The test split is drawn before anything else happens and is never used for
   training, filtering thresholds, or prompt tweaking.** It is hashed from the
   item id, so re-running this script with different settings cannot leak an
   example across the boundary.
2. **Teacher output is not ground truth.** A frontier model hallucinates
   citations and numbers too. Everything it produces goes through the same
   grounding checks the quality harness uses, and the rejection rate is
   reported -- if it is high, the prompt is wrong, not the filter.

Usage
-----
    python finetune/build_dataset.py \
        --workload data/workload.sample.jsonl \
        --teacher openai.gpt-4o-mini \
        --out finetune/data \
        --limit 2000
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vantage.backends.registry import build_backend  # noqa: E402
from vantage.cache.embedders import HashingEmbedder  # noqa: E402
from vantage.config import load_dotenv  # noqa: E402
from vantage.quality.scorers import citation_grounding, number_agreement  # noqa: E402
from vantage.workload.dataset import Workload  # noqa: E402

SPLIT_WEIGHTS = {"train": 0.80, "val": 0.10, "test": 0.10}


def split_for(item_id: str, seed: str = "vantage-v1") -> str:
    """Deterministic split from a hash of the id.

    Hashing rather than shuffling means the split survives re-runs, added data
    and changed ordering: an item that was in test yesterday is in test today.
    """
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name, weight in SPLIT_WEIGHTS.items():
        cumulative += weight
        if position < cumulative:
            return name
    return "test"


def accept(answer: str, context: str, source_ids: list[int]) -> tuple[bool, str]:
    """Automated acceptance checks on a teacher answer.

    These are the same checks the serving quality gate applies, which is the
    point: the student is trained on exactly the behaviour it will later be
    measured on.
    """
    text = answer.strip()
    if len(text) < 10:
        return False, "too_short"
    if len(text.split()) > 220:
        return False, "too_long"
    if citation_grounding(text, source_ids, reference="[1]") < 1.0:
        return False, "citation_missing_or_invented"
    if number_agreement(text, context) < 1.0:
        return False, "number_not_in_context"
    refusal_markers = ("i cannot", "as an ai", "i'm sorry", "i do not have access")
    if any(marker in text.lower() for marker in refusal_markers):
        return False, "refusal"
    return True, "ok"


def deduplicate(records: list[dict], threshold: float = 0.97) -> tuple[list[dict], int]:
    """Drop near-duplicate prompts.

    Duplicates do not teach the model anything new; they just reweight the loss
    towards whatever the corpus happened to repeat, and they inflate the
    apparent size of the dataset.
    """
    if not records:
        return records, 0
    embedder = HashingEmbedder()
    vectors = embedder.embed([r["question"] for r in records])
    keep: list[int] = []
    for i in range(len(records)):
        if keep:
            sims = vectors[keep] @ vectors[i]
            if float(sims.max()) >= threshold:
                continue
        keep.append(i)
    return [records[i] for i in keep], len(records) - len(keep)


def to_chat(record: dict, system: str) -> dict:
    """Chat format. The student is trained on the *same* prompt template the
    serving stack uses -- a fine-tune trained on one prompt and served with
    another is the most common way to lose most of the gain."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["answer"]},
        ],
        "meta": {"item_id": record["item_id"], "teacher": record["teacher"]},
    }


async def generate(args: argparse.Namespace) -> list[dict]:
    workload = Workload.from_jsonl(args.workload)
    items = [i for i in workload.items if split_for(i.id) != "test" or args.include_test]
    if args.limit:
        items = items[: args.limit]

    backend = build_backend(args.teacher)
    semaphore = asyncio.Semaphore(args.concurrency)
    rejected: Counter[str] = Counter()
    records: list[dict] = []

    async def one(item) -> None:
        req = workload.request(item, seq=0, max_tokens=args.max_tokens, want_reference_answer=False)
        async with semaphore:
            result = await backend.generate(req)
        if not result.ok:
            rejected["request_failed"] += 1
            return
        ok, reason = accept(result.text, item.context, item.source_ids)
        if not ok:
            rejected[reason] += 1
            return
        records.append(
            {
                "item_id": item.id,
                "question": item.question,
                "prompt": req.prompt,
                "answer": result.text.strip(),
                "teacher": args.teacher,
                "split": split_for(item.id),
            }
        )

    async with backend:
        await asyncio.gather(*[one(i) for i in items])

    print(f"teacher produced {len(records) + sum(rejected.values())} answers")
    print(f"  accepted: {len(records)}")
    for reason, n in rejected.most_common():
        print(f"  rejected [{reason}]: {n}")
    if rejected and len(records) < 0.5 * (len(records) + sum(rejected.values())):
        print(
            "\n  WARNING: more than half of the teacher's answers were rejected.\n"
            "  That is a prompt problem, not a filter problem. Fix the prompt before training.\n"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", default="data/workload.sample.jsonl")
    parser.add_argument("--teacher", default="openai.gpt-4o-mini", help="a backend config name")
    parser.add_argument("--out", default="finetune/data")
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dedupe-threshold", type=float, default=0.97)
    parser.add_argument("--include-test", action="store_true",
                        help="DANGER: lets held-out items into generation. Only for inspection.")
    args = parser.parse_args()

    load_dotenv()
    workload = Workload.from_jsonl(args.workload)
    records = asyncio.run(generate(args))
    records, dropped = deduplicate(records, args.dedupe_threshold)
    print(f"  dropped as near-duplicates: {dropped}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    handles = {name: (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in SPLIT_WEIGHTS}
    try:
        for record in records:
            split = record["split"]
            if split == "test" and not args.include_test:
                continue
            handles[split].write(json.dumps(to_chat(record, workload.system)) + "\n")
            counts[split] += 1
    finally:
        for fh in handles.values():
            fh.close()

    # The test split is written as *questions only*: it is an evaluation set for
    # the serving harness, not training data, and it must never contain a
    # teacher answer that could be copied.
    test_items = [i for i in workload.items if split_for(i.id) == "test"]
    with (out_dir / "test.questions.jsonl").open("w", encoding="utf-8") as fh:
        for item in test_items:
            fh.write(json.dumps({
                "id": item.id,
                "question": item.question,
                "context": item.context,
                "reference_answer": item.reference_answer,
                "reference_citations": item.reference_citations,
                "source_ids": item.source_ids,
            }) + "\n")

    print(f"\nwrote to {out_dir}:")
    for name in SPLIT_WEIGHTS:
        print(f"  {name}.jsonl: {counts[name]} examples")
    print(f"  test.questions.jsonl: {len(test_items)} held-out items (no teacher answers)")
    print("\nNext: python finetune/train_lora.py --config finetune/configs/qwen3-4b-qlora.yaml")


if __name__ == "__main__":
    main()
