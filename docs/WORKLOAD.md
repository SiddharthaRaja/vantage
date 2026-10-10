# Workloads: how to point this at your own system

The bundled `data/workload.sample.jsonl` is synthetic. It exists so the repo
runs for anyone who clones it. Your numbers only mean something once the
workload is yours.

## The format

One JSON object per line:

```json
{
  "id": "vd-017",
  "question": "What limit of detection did the LC-MS/MS method achieve in serum?",
  "context": "[1] Study 1: ...\n[2] Study 2: ...\n[3] Study 3: ...",
  "reference_answer": "The method achieved a limit of detection of 0.42 ng/mL in serum [2].",
  "reference_citations": [2],
  "source_ids": [1, 2, 3],
  "meta": {"anything": "you like"}
}
```

| field | required | used for |
|---|---|---|
| `id` | yes | request ids, pairing across configs, split hashing |
| `question` | yes | the cache key, and the question in the prompt |
| `context` | yes | retrieved chunks, numbered `[n]` |
| `reference_answer` | for quality | scoring; omit it and the item is load-only |
| `source_ids` | for grounding | which citation numbers are legitimate |

Prompts are built by `Workload.request()`; the system prompt lives in
`workload/dataset.py` and should match the one your application actually sends.
**If the benchmark prompt differs from the production prompt, the benchmark is
measuring a system you do not run.**

## Getting real data out of a RAG app

The honest source is your own logs: real questions, with the context your
retriever actually returned.

```python
# inside your app, next to where you call the model
log.write(json.dumps({
    "id": request_id,
    "question": user_question,
    "context": "\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(retrieved)),
    "source_ids": list(range(1, len(retrieved) + 1)),
}) + "\n")
```

Reference answers are the expensive part. In rough order of quality:

1. **Hand-written by someone who knows the domain.** 50 items is enough to be
   useful and takes an afternoon. This is the best thing you can do for the
   project.
2. **A strong model's answers, hand-checked.** Generate with the largest model
   you can afford, read every one, fix or delete the wrong ones. Record that
   this is what you did.
3. **A strong model's answers, filtered automatically.** What
   `finetune/build_dataset.py` does. Cheapest, weakest, and the one to be most
   careful about claiming things from.

Never use the model under test to write its own references.

## Length matters more than you expect

Prompt and answer length decide which part of inference dominates, and
therefore which optimisation helps:

| workload | prompt | answer | dominated by | what helps |
|---|---|---|---|---|
| RAG question answering | 1k–4k | 50–200 | prefill | prefix caching, chunked prefill, bigger batches |
| chat | 100–1k | 200–800 | decode | quantisation, speculative decoding, tensor parallel |
| summarisation | 4k–32k | 200–600 | prefill, hard | long-context kernels, KV cache capacity |
| classification | 200–2k | 1–10 | prefill | batching, a much smaller model |

`vantage workload <file>` prints the distribution. Put it in the report: a
throughput number without the length distribution beside it is not comparable
to anything.

## Query logs for the cache analysis

The semantic-cache sweep needs *repetition* — real traffic asks the same thing
many ways. With a production log, pass it straight in. Without one,
`build_query_log()` synthesises repeats and paraphrases from your Q/A set, and
the report labels the result as synthetic. Do not quote a hit rate from
synthetic traffic as if it were a production figure; quote the *shape* of the
hit-rate/false-hit trade-off, which is what transfers.
