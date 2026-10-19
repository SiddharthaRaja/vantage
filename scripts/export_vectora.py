"""
Export a Vantage workload from a Vectora corpus.

Vantage's bundled workload is synthetic. This replaces it with the real thing:
every question in a Vectora golden set, the chunks Vectora's own retriever
returns for it, and the *exact* prompt Vectora sends to its LLM — system prompt,
user message, source formatting and all. A serving benchmark that sends a
different prompt from the application is measuring a system nobody runs.

Run it with Vectora's environment (it imports `vectora`), from anywhere:

    python scripts/export_vectora.py --vectora ../Vectora --corpus qasper-sentiment

Writes, next to each other in data/:

    workload.vectora.jsonl        every question — the load generator's input
    workload.vectora.system.txt   Vectora's answer system prompt (picked up automatically)
    eval.vectora.jsonl            the subset used by the quality gate
    eval.vectora.system.txt

Choices worth knowing:

* Retrieval runs in Vectora's `fast` mode (no LLM calls), so the export needs
  no GPU and no API key, and the context is deterministic.
* Golden queries of the form `In the paper "<title>": <question>` are stripped
  to the question. Run 2 of Vectora showed the title does most of retrieval's
  work; real users do not type it.
* QASPER's placeholder tokens (BIBREF12, INLINEFORM0, TABREF3, ...) are
  cleaned out of the context, matching Vectora's run-2 clean corpus.
* Reference answers are the QASPER annotators' answers (from `notes`). They are
  short spans, so absolute token-F1 against a full prose answer is low; the
  quality gate compares configurations against each other, which is what it
  needs.
* The eval subset is chosen by hashing the id, so it is stable across exports,
  and only keeps questions whose answering paper was actually retrieved —
  otherwise the reference is not in the prompt and every config scores ~0.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

TITLE_PREFIX = re.compile(r'^In the paper "[^"]*":\s*')
GOLD_PREFIX = re.compile(r"^\s*Gold answer:\s*", re.I)
PLACEHOLDERS = [
    (re.compile(r"\s*BIBREF\d+(\s*,\s*BIBREF\d+)*"), ""),
    (re.compile(r"(INLINEFORM|DISPLAYFORM)\d+"), "[formula]"),
    (re.compile(r"TABREF\d+"), "the table"),
    (re.compile(r"FIGREF\d+"), "the figure"),
    (re.compile(r"SECREF\d+"), "a later section"),
]


def clean(text: str) -> str:
    for rx, rep in PLACEHOLDERS:
        text = rx.sub(rep, text)
    return text


def strip_title(query: str) -> str:
    return TITLE_PREFIX.sub("", query).strip()


def gold_answer(notes: str) -> str:
    """The annotators' answer, from a QASPER golden case's notes field."""
    if not notes or not GOLD_PREFIX.match(notes):
        return ""
    return clean(GOLD_PREFIX.sub("", notes)).strip()


def in_eval(item_id: str, fraction: float) -> bool:
    h = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)
    return (h % 10_000) / 10_000 < fraction


async def export(args: argparse.Namespace) -> None:
    vectora_dir = Path(args.vectora).resolve()
    out_dir = Path(args.out_dir).resolve()
    golden = Path(args.golden).resolve() if args.golden else None

    # Vectora reads .env and its corpora relative to its own folder.
    os.chdir(vectora_dir)
    sys.path.insert(0, str(vectora_dir))
    from evaluation.run_eval import load_dataset
    from vectora.api.deps import build_container
    from vectora.config import SearchMode, get_settings
    from vectora.generation.pipeline import build_context
    from vectora.generation.prompts import ANSWER_SYSTEM, PROMPT_VERSION, answer_user_message
    from vectora.generation.router import classify

    settings = get_settings()
    container = await asyncio.to_thread(build_container, settings, warm=False, corpus=args.corpus)
    golden = golden or container.corpus.golden
    cases, _ = load_dataset(golden)
    print(f"{len(cases)} questions from {golden}")

    rows, hits = [], 0
    try:
        for case in cases:
            question = strip_title(case.query)
            query_type, _ = await classify(question, llm=None, settings=settings)
            chunks, _ = await container.engine.retriever.retrieve(
                question, mode=SearchMode.FAST, query_type=query_type, top_k=args.top_k
            )
            context = clean(build_context(chunks))
            prompt = answer_user_message(
                query=question,
                context=context,
                prompt_hint=settings.route(query_type).prompt_hint,
            )
            papers = [rc.chunk.paper_id for rc in chunks]
            relevant = set(case.relevant_papers)
            ref_cites = [i for i, p in enumerate(papers, start=1) if p in relevant]
            hits += bool(ref_cites)
            rows.append({
                "id": case.id,
                "question": question,
                "context": context,
                "prompt": prompt,
                "reference_answer": gold_answer(case.notes),
                "reference_citations": ref_cites,
                "source_ids": list(range(1, len(chunks) + 1)),
                "meta": {
                    "source": f"vectora:{args.corpus}",
                    "query_type": query_type.value,
                    "relevant_papers": sorted(relevant),
                    "retrieved_papers": papers,
                    "answer_paper_retrieved": bool(ref_cites),
                    "retrieval_mode": "fast",
                    "top_k": args.top_k,
                    "prompt_version": PROMPT_VERSION,
                },
            })
    finally:
        await container.aclose()

    out_dir.mkdir(parents=True, exist_ok=True)
    workload = out_dir / "workload.vectora.jsonl"
    evalset = out_dir / "eval.vectora.jsonl"
    eval_rows = [
        r for r in rows
        if r["reference_answer"] and r["meta"]["answer_paper_retrieved"] and in_eval(r["id"], args.eval_fraction)
    ]
    for path, data in ((workload, rows), (evalset, eval_rows)):
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n", "utf-8")
        path.with_suffix(".system.txt").write_text(ANSWER_SYSTEM + "\n", "utf-8")

    words = sorted(len((r["prompt"] + ANSWER_SYSTEM).split()) for r in rows)
    print(f"wrote {workload}  ({len(rows)} items; answering paper retrieved for {hits})")
    print(f"wrote {evalset}  ({len(eval_rows)} items for the quality gate)")
    print(f"prompt words incl. system: min {words[0]}, median {words[len(words) // 2]}, max {words[-1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vectora", required=True, help="Path to the Vectora repo")
    ap.add_argument("--corpus", default="qasper-sentiment")
    ap.add_argument("--golden", default=None, help="Golden set (default: the corpus's own)")
    ap.add_argument("--top-k", type=int, default=6, help="Chunks per prompt (Vectora fast mode uses 6)")
    ap.add_argument("--eval-fraction", type=float, default=0.6)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    asyncio.run(export(ap.parse_args()))


if __name__ == "__main__":
    main()
