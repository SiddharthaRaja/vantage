# Vantage: the whole thing explained from scratch

You asked me to build this in one pass and explain everything afterwards,
assuming you know nothing about it. That is what this document is. It is long
because it is meant to replace a week of reading, and it is ordered so you can
stop at any point and still have something useful.

- **Part 1** — what this project is and why it is the right one to have built
- **Part 2** — every concept you need, from zero
- **Part 3** — what the code actually is, file by file
- **Part 4** — how to run it, exactly, on your machine
- **Part 5** — the fine-tuning branch
- **Part 6** — what to do next, in order, to turn this into a resume line
- **Part 7** — how to talk about it, and the questions you will be asked
- **Part 8** — what I could not do, and what is still open

---

# Part 1 — What this is, and why

## The one-line version

Vantage is a tool that measures how an LLM serving system behaves under load:
how fast it answers, how many requests per second it can take before latency
falls apart, how much each request costs, and — the part almost nobody does —
whether the things you did to make it faster quietly made the answers worse.

## Why this project rather than another one

Your portfolio before this was entirely *inference-side*. Vectora calls models,
orchestrates retrieval, evaluates answers. All good work, but every project
answers the same question: can you use a model well?

There are two obvious holes an interviewer can poke at:

1. *"Have you ever run anything under load, or do you just have a demo?"*
2. *"Have you ever trained a model, or do you only call APIs?"*

This project closes the first hole completely and the second one partially (the
`finetune/` branch is a real training pipeline, and it attaches to the same
harness instead of being a second disconnected project).

It also stacks on Vectora instead of sitting beside it. You are not presenting
two unrelated side projects; you are presenting one system with a measured
performance story: *here is my RAG stack, here is what it costs and how fast it
is, here is what I changed, here is the measurement that proves the change
helped and did not break the answers.*

That reads as an engineer. Two tutorials read as a student.

## What makes it different from the thousand "I benchmarked an LLM" repos

Four things, and they are the four things to say out loud in an interview:

1. **Open-loop load generation.** Nearly every amateur benchmark is closed-loop
   (a fixed number of workers, each waiting for its reply before sending the
   next). That design can never show you the cliff where a server stops coping,
   because when the server slows down, the client slows down too. Vantage does
   both, and the open-loop chart is the one that shows the cliff.
2. **A quality gate.** Every speed-up trades something. This harness scores
   answers on a held-out set and marks a configuration FAIL if it got measurably
   worse, even if it got much faster. In the demo run, 4-bit quantisation is
   1.7× faster and 42% cheaper — and fails the gate.
3. **A cost model with a break-even point.** Self-hosting costs the same whether
   the GPU is busy or idle; an API costs per token. Those two lines cross, and
   the crossing point is the actual engineering decision. Vantage computes it.
4. **Honest statistics.** Bootstrap confidence intervals, paired comparisons,
   and an explicit rule: if the interval for a difference includes zero, the
   report does not call it a difference.

---

# Part 2 — Every concept, from zero

Skip anything you already know. Nothing here assumes the previous section.

## 2.1 What happens when a model answers you

A language model generates text one token at a time (a token is roughly
three-quarters of a word). Serving a request has two distinct phases, and
almost every performance question comes down to which one dominates:

**Prefill.** The model reads your entire prompt at once and builds an internal
representation of it. This is one big matrix multiplication over the whole
prompt, so it is *compute-bound* and very parallel. A 2,000-token prompt is
processed in a single pass.

**Decode.** The model then produces output tokens one at a time, and each new
token depends on all the previous ones. This cannot be parallelised across
tokens, and each step reads the entire model's weights from GPU memory. So
decode is *memory-bandwidth-bound* — the GPU is mostly waiting for memory, not
calculating.

Why this matters: the two phases respond to completely different optimisations.
If your workload is RAG (huge prompt, short answer), prefill dominates and
caching the prompt helps. If it is chat (short prompt, long answer), decode
dominates and quantising the weights helps. Optimising the wrong phase does
nothing, which is exactly what the demo run shows for prefix caching.

**KV cache.** During decode the model would otherwise recompute its view of
every earlier token at every step. Instead it stores those intermediate values —
the "keys and values" — in GPU memory and reuses them. This is the KV cache. It
makes decode feasible, and it is usually what limits how many requests a GPU can
serve at once: each in-flight request holds its own KV cache, proportional to
its length. When people say "we ran out of GPU memory at 40 concurrent
requests", they mean the KV cache.

**Continuous batching.** The GPU is idle most of the time during decode, so
modern servers (vLLM, SGLang, TGI) run many requests in the same batch and add
or remove requests from the batch at every token step rather than waiting for a
whole batch to finish. The effect: as you add concurrent requests, total
throughput rises steeply at first, then flattens — and each individual stream
gets slower. That curve, and where it bends, is the single most important thing
this harness measures.

## 2.2 The four latencies (and why benchmarks lie about them)

- **TTFT — time to first token.** From sending the request to the first word
  appearing. Dominated by queueing plus prefill. This is what a user perceives
  as "did it respond".
- **ITL — inter-token latency.** The gap between successive tokens. Determines
  how fast text streams out. Grows as the batch grows.
- **End-to-end.** From sending the request to the last token.
- **Total.** From the request *arriving* to the last token — so it includes
  time spent waiting in a queue before anything was even sent.

Most published benchmarks report one number called "latency" without saying
which of these it is, and usually measure from dispatch rather than arrival,
which hides all the queueing. Vantage measures all four and defines each in the
report.

## 2.3 Closed loop vs open loop — the big one

This is the methodological point that makes the project credible. Learn it
properly; you will be asked.

**Closed loop:** you have N virtual users. Each sends a request, waits for the
answer, then sends the next. Load is self-limiting: if the server slows down,
your client sends fewer requests. It answers "how does the system behave at
concurrency N".

**Open loop:** requests arrive on a schedule — say 20 per second, with random
gaps (a Poisson process, which is what independent users actually produce) —
whether or not the previous ones finished. Load does not back off when the
server struggles. It answers "what happens at 20 requests per second".

Why it matters: in a closed-loop test, latency rises smoothly and nothing ever
looks catastrophic. In an open-loop test, latency is boring and flat up to about
80% of capacity and then goes **vertical**, because a queue that grows faster
than it drains grows without bound. That cliff is where production systems die,
and a closed-loop benchmark structurally cannot show it to you.

In the demo run you can see it: offered 40 req/s, the server delivers 38.7 and
p95 TTFT is 0.3s. Offered 52 req/s, it delivers 41 — it is at capacity — and p95
TTFT jumps to 1.13s, with 14% of requests missing the SLO.

One detail that makes the implementation honest: if the *client* can't keep up,
it records those requests as "shed" instead of slowing its own arrival schedule.
A load generator that quietly throttles itself has silently turned into a
closed-loop test, and that is a real bug in several popular tools.

## 2.4 Percentiles, and why the average is useless

If 99 requests take 0.2s and one takes 10s, the mean is 0.3s and looks fine.
The user who waited 10 seconds does not care about your mean.

- **p50 (median)** — half of requests were faster than this.
- **p95** — 95% were faster; 1 in 20 was slower.
- **p99** — 1 in 100 was slower. On a page that makes 10 model calls, roughly
  one in ten page loads hits it.

Tails are where user experience lives. Vantage reports p50/p90/p95/p99 and the
max for every latency, and the SLO is defined on p95.

There is also a convention detail that sounds pedantic and is not: percentiles
can be computed by "nearest rank" or by linear interpolation between the two
nearest samples. On a few hundred samples these give visibly different p99s.
Vantage uses interpolation (what numpy, wrk, k6 and vegeta do) and says so, so
its numbers are comparable to other tools'.

## 2.5 Goodput

Throughput is requests per second. **Goodput is requests per second that met
your service level objective.** The distinction matters because you can always
increase throughput by accepting more requests and serving them all slowly.

Vantage takes an SLO (e.g. "TTFT p95 under 1 second, total p95 under 8 seconds,
error rate under 1%") and reports what fraction of requests met it. The honest
headline for a deployment is `achieved_qps × goodput_ratio`.

## 2.6 Confidence intervals and the bootstrap

You measured 200 requests and config A's mean was 0.41s while config B's was
0.39s. Is B faster, or did you get lucky?

The **bootstrap** answers this without assuming anything about the shape of the
distribution: resample your 200 measurements with replacement 2,000 times,
compute the statistic each time, and look at the spread of those 2,000 answers.
The middle 95% of them is your confidence interval.

This matters for latency specifically because latency distributions are
right-skewed with heavy tails — the usual "mean ± 1.96 × standard deviation"
formula assumes a bell curve and is simply wrong here.

For quality comparisons Vantage uses a **paired** bootstrap: both configurations
answered the same questions, so it resamples the per-question *differences*
rather than the two score sets independently. Pairing removes the variance that
comes from some questions being harder than others, which makes the test much
more sensitive for free.

And the rule that follows: **if the interval for a difference includes zero,
the report does not call it a difference.** This is why the demo marks a −0.018
quality drop as PASS (interval −0.047 to 0.000, spans zero) and a −0.053 drop as
FAIL (interval −0.096 to −0.015, entirely below zero).

## 2.7 Quantisation

Model weights are normally 16-bit floating point numbers. **Quantisation**
stores them in fewer bits — 8-bit integers, or 4-bit with formats like AWQ,
GPTQ or NF4.

Why it makes things faster: decode is memory-bandwidth-bound, so halving the
size of the weights roughly halves the amount of memory traffic per token. You
also fit more KV cache in the leftover memory, so you can batch more requests.

What it costs: accuracy. Usually a little at 8-bit, sometimes a lot at 4-bit,
and *how much* depends on the model and the task in ways nobody can predict
reliably. Which is the entire argument for the quality gate: you cannot reason
your way to the answer, you have to measure it.

## 2.8 Prefix caching

In a RAG system every request starts with the same long system prompt, and
often shares retrieved chunks with other requests. **Prefix caching** means the
server recognises a shared prefix and reuses the KV cache it already computed
for it, skipping that part of prefill.

In the demo it cut TTFT roughly in half at low concurrency and did *nothing*
for throughput — because with 450-token prompts and 96-token answers that
workload is decode-bound, and prefix caching only helps prefill. That is a
genuine, slightly counter-intuitive finding, and it is exactly the kind of thing
that is worth putting in a report.

## 2.9 Semantic caching

An **exact cache** stores answers keyed on the exact question text. Cheap,
perfectly safe, and misses whenever a user rephrases.

A **semantic cache** embeds the question into a vector, and on a new question
finds the nearest stored vector by cosine similarity. If it is closer than some
threshold, serve the stored answer.

The danger is obvious once stated: set the threshold too low and you serve the
answer to question A when someone asked question B — confidently, instantly, and
wrongly. Almost every blog post about semantic caching reports a hit rate and
never mentions this.

So Vantage measures the **false-hit rate**: it replays a query log at each
candidate threshold and checks whether the answer the cache served actually
matches the reference answer for the *new* question. The result is the table in
the README: at threshold 0.70 you get an 85.7% hit rate and 17.5% of those hits
are wrong; at 0.90 you get 49.7% hits and 0.7% wrong. Now the threshold is an
engineering decision with numbers behind it.

(An embedding, by the way, is just a list of numbers — a few hundred of them —
that represents a piece of text, arranged so that similar texts get similar
lists. "Cosine similarity" is the standard way to measure how close two such
lists point in the same direction: 1.0 means identical direction, 0 means
unrelated.)

## 2.10 The cost model

```
API:        cost per request = (input tokens / 1M) × price_in
                             + (output tokens / 1M) × price_out
Self-host:  cost per request = (GPU $/hour ÷ 3600) ÷ requests per second
```

The API line is flat per request and independent of how busy you are. The
self-hosted line falls as utilisation rises, because you pay for the GPU whether
or not anyone is using it.

Set them equal and solve:

```
break-even req/s = (GPU $/hour ÷ 3600) ÷ (API cost per request)
```

In the demo: a $0.75/hour GPU and gpt-4o-mini pricing at this prompt length
gives **1.65 req/s sustained**, about 142,000 requests a day. Below that, the
API is cheaper. Above it, the GPU is.

The report also says what the number excludes — engineering time, on-call,
redundancy, and the fact that you must provision for peak traffic while paying
for peak capacity all day. Stating the limits of your own analysis is what makes
the analysis trustworthy rather than naive.

## 2.11 LoRA and QLoRA (for Part 5)

Fine-tuning normally means updating all of a model's weights, which needs
enormous GPU memory. **LoRA** (Low-Rank Adaptation) freezes the original weights
and learns two small matrices per layer instead; the model computes
`W + B·A` where B and A together have maybe 0.5% as many parameters as W. You
train the small matrices, and you ship a ~100MB "adapter" rather than a new 8GB
model.

**QLoRA** does the same while holding the frozen base model in 4-bit, so a 4B
model fine-tunes on a single 16GB GPU — the free tier of Colab, at the slow end.

**Distillation** here means: use a big expensive model (the "teacher") to
produce good answers to a few thousand of your questions, then train a small
model (the "student") on those answers. You end up owning a small model that
does your one job nearly as well as the big one, at a fraction of the cost.

---

# Part 3 — What the code is, file by file

The repo has 91 files. These are the ones that matter, in the order a reader
should meet them.

## The shape of it

```
src/vantage/
  models.py        every value that crosses a module boundary, typed
  backends/        "a way of running an LLM", behind one interface
  workload/        what we send, and at what arrival pattern
  metrics/         what we measure and what it costs
  cache/           an optimisation we implement and measure
  quality/         the guard rail
  report/          charts and the written report
  runner.py        run one configuration
  sweep.py         run an experiment of many configurations
  store.py         where results go, and how they stay reproducible
  cli.py           the commands you type
configs/           backend definitions + sweep experiments (YAML)
finetune/          the training branch
tests/             59 tests, none needing a GPU
docs/              methodology and how to plug in your own data
```

## `models.py` — typed boundaries

Every request, result and summary is a Pydantic model. A benchmark harness
lives or dies on the integrity of its numbers: if a backend forgets to report
how many tokens it generated, the throughput figure becomes silently wrong and
you find out from a chart three steps later. Typed boundaries turn that into a
loud error at the point it happens.

The interesting one is `GenResult`, which carries all four timings plus
`token_counts_estimated` — a flag set whenever the server did not report token
usage and we had to guess. The report surfaces it, because an estimated
throughput is not a measured one.

## `backends/` — one interface, everything behind it

`base.py` defines a twelve-line interface. A subclass implements `_stream()`;
the base class owns *all* the timing, because timing reimplemented per backend
is timing you cannot trust.

`openai_compat.py` speaks the OpenAI chat-completions protocol, which means one
adapter covers vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, OpenAI and most
hosted providers. Two deliberate choices in it:

- Streaming is always on, even when nothing reads the stream, because TTFT does
  not exist without it.
- Retries are **off** by default. A retry hides a failure and inflates the
  latency of the request that eventually succeeded. If you want them you must
  ask, and the count is recorded.

`simulated.py` is a model of a continuous-batching server. It is why you can
run and debug the entire harness on a laptop with no GPU, and why CI can test
the measurement pipeline on every commit. The core of it is one formula: with
*b* sequences in flight, each one decodes at

```
r(b) = r1 · k / (k + b − 1)
```

so aggregate throughput rises, bends and saturates at `r1 · k` tokens/second.
That is the standard saturating shape of a batched decoder. The simulator also
models the engine queue (`max_batch`), prefill cost proportional to prompt
length, optional prefix caching, failure injection, and — importantly — a
`fidelity` parameter that makes a lower-precision profile produce measurably
worse answers, so the quality gate has something real to catch.

Every report generated from the simulator is stamped "simulated" in its header.
Be strict about this. The moment you quote a simulated number as a measurement,
the whole project becomes untrustworthy.

## `workload/` — the prompts and the arrival pattern

`dataset.py` loads JSONL of `(question, retrieved context, reference answer,
valid source ids)` and builds RAG-shaped prompts. The bundled sample is 60
synthetic items in the shape of your vitamin D corpus — invented numbers, no
copyrighted text, so the repo runs for anyone who clones it. `docs/WORKLOAD.md`
explains how to replace it with your own logged traffic, which is the thing that
makes your numbers mean something.

`generator.py` implements both load modes (Part 2.3), the discarded warm-up
phase, and the shed accounting.

## `metrics/` — statistics and money

`stats.py`: percentiles with a stated convention, bootstrap CIs, and
`knee_point()` — the Kneedle construction, which finds the bend in a saturating
curve by maximum perpendicular distance from the chord joining its endpoints.
That is how the report can say "the knee is at concurrency 16" instead of
"as you can see from the chart".

`collector.py`: turns a finished run into a `RunSummary`.

`cost.py`: the two cost regimes and the break-even calculation from Part 2.10.

## `cache/` — the optimisation, and the analysis that justifies it

`embedders.py` ships a dependency-free hashing embedder (deterministic, no
network, runs in CI) alongside OpenAI and sentence-transformers options.
`semantic.py` is the cache itself plus `CachedBackend`, a wrapper that puts a
cache in front of *any* backend without that backend knowing. `analysis.py` is
the threshold sweep that produces the hit-rate/false-hit-rate table.

## `quality/` — the gate

`scorers.py` has four deterministic scorers:

| scorer | catches |
|---|---|
| token F1 | general agreement, robust to phrasing |
| exact match | strict agreement; only useful for short answers |
| citation grounding | an answer that cites nothing, or cites source [7] when only [1]–[3] were given — a machine-checkable hallucination |
| number agreement | numbers in the answer that are not in the reference. In a biomedical corpus this is the most dangerous failure, and token F1 barely notices it |

`harness.py` runs an evaluation set through a backend, scores it, and
`compare_quality()` does the paired bootstrap and returns PASS / WARN / FAIL.

## `report/` — charts and prose

`charts.py` produces six charts with matplotlib, using a fixed categorical
colour palette checked for colour-blind separation, line styles and direct
labels so nothing depends on colour alone, and one y-axis per chart always
(dual-axis charts are the most common way to imply a relationship that is not
in the data).

`render.py` writes `REPORT.md` and a standalone `report.html` with the charts
embedded. The findings paragraph at the top is **generated from the data**, not
written by hand, so it cannot drift away from the tables underneath it. And
every report ends with a section titled *"What this run does not establish"*.
That section is not modesty — it is the part that makes the rest credible, and
it is the first thing a good reviewer looks for.

## `sweep.py` and `store.py` — experiments and reproducibility

A sweep is a YAML file listing runs, each optionally with a `matrix:` that gets
expanded into a cartesian product. Runs execute strictly sequentially with a
cool-down, because two benchmarks sharing a GPU measure each other, and a
saturated server does not drain the instant the last response returns.

Every run writes a directory containing the resolved config, the git SHA and
whether the tree was dirty, the host and GPU description, every individual
request in `raw.jsonl`, the summaries, the quality scores, the gate verdicts,
the cache sweep, the break-even figures, the charts and both reports.
`vantage report <dir>` rebuilds everything from those files. The rule: if a
chart cannot be regenerated from the raw data six weeks later, it is not
evidence.

## `tests/` — 59 of them

They test behaviour, not just that the code runs: that higher concurrency
raises throughput, that the open-loop generator really does issue more requests
than a closed-loop client could, that failures are recorded rather than raised,
that the quality gate passes on noise and fails on a real regression, that
tightening the cache threshold monotonically reduces the hit rate, that the SSE
streaming parser handles a missing usage block and flags the estimate, and that
a full sweep writes a run directory you can rebuild a report from.

The CI workflow runs lint, the tests, and a complete simulated sweep on every
push, then uploads the report as a build artefact.

---

# Part 4 — Running it, exactly

You are on Windows. These commands assume PowerShell; on Mac or Linux the only
difference is the virtualenv activation line.

## First run — 5 minutes, no GPU, no API key

```powershell
cd path\to\vantage
python -m venv .venv
.venv\Scripts\Activate.ps1          # mac/linux: source .venv/bin/activate
pip install -e ".[dev]"

vantage doctor -b sim.fp16          # should print "ok"
vantage demo                        # ~2 minutes
```

Then open the `report.html` it printed the path to. That is the whole pipeline.

## The commands

| command | what it does |
|---|---|
| `vantage doctor -b <backend>` | checks a backend responds and the workload parses, before you waste half an hour |
| `vantage workload <file>` | prints the prompt-length distribution of a workload |
| `vantage bench -b <backend> -c 32 -n 200` | measures one configuration |
| `vantage bench -b <backend> --rate 20 --duration 60` | same, open loop at 20 req/s |
| `vantage sweep -c configs/sweeps/<x>.yaml` | runs a whole experiment and writes the report |
| `vantage quality -b <backend>` | scores answers only, no load testing |
| `vantage cache-sim --repeats 5` | the semantic cache threshold sweep |
| `vantage report <run_dir>` | rebuilds charts and report from saved raw data |
| `vantage demo` | everything, simulated |

## The experiments that ship

| file | question it answers |
|---|---|
| `demo.yaml` | everything at once, simulated — the showcase |
| `concurrency.yaml` | how many concurrent users can this deployment take? |
| `offered-load.yaml` | at what requests/second does it fall over? |
| `quantization.yaml` | does fp16 → int8 → int4 pay for itself? |
| `cache.yaml` | where should the semantic cache threshold sit? |
| `finetune-compare.yaml` | teacher vs base vs fine-tuned student |

## Pointing it at something real

The simulated numbers demonstrate the analysis. They are not results. To get
results you need a GPU for an hour or two — rent one (RunPod, Lambda, Vast.ai;
an A10G or 4090 is about $0.40–$0.80/hour, so a full afternoon is a few
dollars):

```bash
pip install vllm
vllm serve Qwen/Qwen3-4B-Instruct --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching

# in another terminal, on a DIFFERENT machine if you can
vantage doctor -b vllm.qwen3-4b
vantage sweep -c configs/sweeps/concurrency.yaml
vantage sweep -c configs/sweeps/offered-load.yaml
```

Then edit `configs/backends/vllm.qwen3-4b.yaml` so `gpu_hourly_usd` is what you
actually paid, and re-run. The cost numbers are only as honest as that line.

Two things to get right or the numbers are worthless:

1. **Run the client on a different machine from the server if you possibly
   can.** A load generator sharing a CPU with the server it measures is
   measuring itself as well. `docker-compose.yml` separates them.
2. **Use your own prompts.** `docs/WORKLOAD.md` explains the format and how to
   export real traffic out of Vectora. A benchmark on synthetic prompts measures
   a system you do not run.

---

# Part 5 — The fine-tuning branch

Everything above is inference-side. `finetune/` is the part that makes a model
rather than measuring one, and it is deliberately small and attached to the
harness instead of being a second project.

The claim at the end is one sentence: *a 4B model fine-tuned on distilled
answers recovers N% of the teacher's quality on this task while serving it at
X times lower cost per request* — measured by the same harness, on the same
held-out set, under the same load.

## The four scripts

**`build_dataset.py` — distillation.** Takes your questions plus retrieved
context, has a strong hosted model answer each one, and keeps only the answers
that pass automated grounding checks. Three things in it are the actual
engineering:

- *The test split is hashed from the item id, not shuffled.* That means the
  split survives re-runs and added data: an item in test yesterday is in test
  today. The test file is written as **questions only**, with no teacher answer
  beside it, so it cannot leak into training by accident.
- *Teacher output is not ground truth.* Frontier models invent citations and
  numbers too. Every answer goes through the same `citation_grounding` and
  `number_agreement` checks the serving gate uses, and rejections are counted
  by reason. If more than ~30% are rejected, the prompt is wrong and training on
  what survived would bake the problem in.
- *Near-duplicates are dropped.* Duplicates do not teach the model anything;
  they reweight the loss toward whatever your corpus happened to repeat.

**`train_lora.py` — QLoRA training.** Config-driven (`configs/qwen3-4b-qlora.yaml`),
runs on one 16GB GPU in 20–60 minutes for ~2,000 examples. The two settings that
matter most:

- *Train on the serving prompt.* A model fine-tuned on one prompt format and
  served with another loses most of the gain. It is the commonest silent
  failure in student fine-tunes.
- *Loss on the answer only* (`assistant_only_loss`). RAG prompts are enormous
  and the answer is short; training on every token means ~95% of the gradient
  teaches the model to reproduce retrieved documents instead of to answer.

**`merge_and_export.py`** folds the adapter into the base weights when you want
a standalone model, and prints exact recipes for GGUF (llama.cpp / Ollama) and
AWQ 4-bit export. It also tells you when *not* to merge: if you serve with
vLLM's `--enable-lora`, base and fine-tune run in the same process on the same
hardware, which removes every hardware confound from the comparison. For
benchmarking, that is what you want.

**`eval_student.py`** is the experiment: teacher, base and student, one
workload, one eval set, one GPU. It prints a table of quality, latency,
throughput and cost, runs the paired bootstrap, and — if the student lost to
the base model — prints the checklist of things that are usually wrong
(prompt mismatch, adapter not actually loaded, too little surviving data, task
beyond a 4B model).

## The rule to keep repeating to yourself

**A falling training loss is not a result.** It means the model learned the
distribution of your training data. Whether it answers held-out questions
better than the model you started from is a different question, and only
`eval_student.py` answers it. If you remember one thing from this section,
remember that — it is also the thing interviewers probe for.

---

# Part 6 — What to do next, in order

The code is done. Here is the sequence that turns it into a resume line, with
the highest-value step first. Everything up to step 3 is one weekend.

### Step 1 — Run it against real hardware (half a day, ~$5)

This is the single highest-value thing and nothing else matters as much. Rent
an A10G or 4090 by the hour, start vLLM, and run `concurrency.yaml` and
`offered-load.yaml`. Set `gpu_hourly_usd` to what you actually paid. Now you
have real numbers, and the README's example section can be rewritten with
measurements instead of simulation.

While you are there, run `quantization.yaml` against a real AWQ build of the
same model. Whether real int4 passes or fails your gate is a genuinely
interesting result either way.

### Step 2 — Swap in Vectora's real traffic (half a day)

Export `(question, retrieved context)` from Vectora into the JSONL format in
`docs/WORKLOAD.md`. Hand-write 40–60 reference answers yourself — it takes an
afternoon and it is the single biggest upgrade to the project's credibility,
because then the quality gate is scoring your real system on your real domain.

### Step 3 — Publish (an hour)

Push to GitHub. Put the real report's charts in the README. Make sure the
caveats section is intact — a reader who sees you state the limits of your own
measurement will trust the measurement.

### Step 4 — Wire the gate into Vectora's CI (an hour)

Add a GitHub Action to the Vectora repo that runs `vantage quality` against a
small eval set on every push. "Evals in CI" is a genuinely senior-sounding
thing for an intern to have done, and it costs an hour once this exists.

### Step 5 — The fine-tuning branch (a weekend)

Only after steps 1–3. Distil, train, serve the adapter alongside the base in
one vLLM process, run `finetune-compare.yaml`. If the student wins, you have a
second headline result. If it loses, you have an honest write-up of why — which
is also worth having, and far better than silence.

### Step 6 — Write the 600-word post

One page: the problem, the method, the four charts, the surprising finding
(prefix caching doing nothing for a decode-bound workload is a good one), and
what you would do differently. Link it from the README and your CV. This is
what turns a repo into something people actually read.

### Running in parallel, as agreed

DSA practice daily, and applications going out now rather than when the project
is polished. The project is a conversation-starter in interviews you already
have, not a prerequisite for getting them.

---

# Part 7 — How to talk about it

## Resume bullets

Use these once step 1 is done and the bracketed numbers are real. Do not ship
them with simulated figures in them.

> **LLM inference benchmarking harness** — Built an open-source benchmark for
> LLM serving stacks measuring TTFT, inter-token latency, p50/p95/p99 end-to-end
> latency, throughput and SLO goodput under both closed-loop and open-loop
> (Poisson) load against vLLM, TGI, Ollama and hosted APIs through one adapter.

> Identified the saturation point of a [Qwen3-4B on A10G] deployment at
> [N] req/s, beyond which p95 latency rose [X]× for [Y]% additional throughput;
> computed the API-vs-self-host break-even at [Z] req/s sustained.

> Implemented a semantic response cache and selected its similarity threshold
> by measuring false-hit rate alongside hit rate — [49.7]% of queries served
> from cache at a [0.7]% rate of serving a mismatched answer.

> Built a paired-bootstrap quality gate (token F1, citation grounding,
> hallucinated-number detection) that blocks any serving optimisation causing a
> statistically significant answer-quality regression; it caught a [−0.053]
> composite drop from 4-bit quantisation that a latency-only benchmark would
> have shipped.

> *(after step 5)* Fine-tuned [Qwen3-4B] with QLoRA on [N] distilled, grounding-
> filtered examples; the student recovered [X]% of teacher quality at [Y]× lower
> cost per request, measured on a held-out split through the same harness.

Notice the shape of every one: a specific thing you built, a specific number,
and what the number means. "Built a RAG chatbot" has none of those.

## The questions you will be asked

**"Why open loop and closed loop?"** — Closed loop measures behaviour at a
given concurrency; it cannot overload the server because slow responses throttle
the client. Open loop measures behaviour at a given arrival rate and is the only
one that shows queueing collapse. Real traffic is open loop. *(Part 2.3.)*

**"What's the difference between throughput and goodput?"** — Goodput only
counts requests that met the SLO. You can always raise throughput by serving
everyone slowly.

**"How do you know a difference is real?"** — Paired bootstrap over per-item
scores; if the 95% interval includes zero I don't call it a difference. I use
paired rather than independent because both configs answered the same questions.

**"Why is your p99 noisy?"** — Because a p99 over a few hundred requests is an
estimate with real uncertainty. That is why the report gives bootstrap intervals
and warns against reading single-run p99 differences under ~20%.

**"Where does prefill versus decode matter?"** — Prefill is compute-bound and
parallel; decode is memory-bandwidth-bound and sequential. RAG is prefill-heavy,
chat is decode-heavy, and they reward different optimisations. In my run, prefix
caching halved TTFT and did nothing for throughput because that workload was
decode-bound.

**"What would you do differently?"** — The load generator is Python asyncio, so
above roughly 2,000 req/s the client becomes part of what I am measuring; I'd
move the generator to Go or run several processes. I'd also want human-rated
answers rather than only automated scorers — my quality gate can prove a
regression but cannot prove an answer is good.

**"What did you get wrong?"** — Worth having a real answer here. A good one:
the first version of the quality gate compared independent means and called a
noise-level difference a regression; switching to a paired bootstrap fixed it.

## The one-sentence pitch

> "I built a benchmark harness for LLM serving that measures latency,
> throughput, cost and answer quality together — so you can't ship a
> 'performance win' that's actually a quality regression — and used it to find
> the saturation point and the self-hosting break-even for my own RAG system."

---

# Part 8 — What I could not do, and what is open

Said plainly, because you should not discover these in an interview.

**Everything in the demo report is simulated.** I had no GPU and no API access
in the environment I built this in. The harness is real, the analysis is real,
the tests are real; the *numbers* in the example run come from the simulator.
Step 1 of Part 6 fixes this and nothing else does.

**The sample workload is synthetic.** 60 invented items in the shape of your
vitamin D corpus. Real prompts change the conclusions — prompt length decides
which optimisation helps.

**The fine-tuning scripts have not been executed.** They are written carefully
against current TRL/PEFT APIs and they parse and dry-run, but no GPU has run
them. Expect one or two small API adjustments; the libraries move fast. The
logic — the split hashing, the grounding filters, the completion-only loss, the
comparison — is the part that matters and is correct.

**No LLM judge is wired up by default.** The scorers are deterministic ones.
An LLM judge adds nuance and costs money per run; the hook is there.

**The cost model is list-price arithmetic.** It excludes your time, redundancy,
and the peak-versus-mean provisioning gap. The report says so.

**I could not reach your Vectora folder.** No folder was connected, so Vantage
is standalone. It is designed to point at Vectora rather than live inside it,
which is the right structure anyway — but connecting the two (step 2) is manual
work I could not start.

## If something breaks

Run `vantage doctor -b <backend>` first; it catches most of it. Then
`pytest -q` — if the tests pass, the harness is fine and the problem is
configuration. The three most likely issues:

1. *A backend config points at a server that isn't running.* `doctor` says so.
2. *`vantage: command not found`* — the virtualenv isn't activated, or
   `pip install -e .` wasn't run.
3. *A sweep runs but the report has no charts* — a chart is skipped when there
   isn't enough data for it (a curve needs three or more points). That is
   intentional, not a failure.

Bring me the error and the run directory next session and we'll go through it.
