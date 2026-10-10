# Roadmap

Ordered by value, not by ease. Each item is a commit (or a short branch and a
pull request). Nothing here is cosmetic: every entry is work the project
genuinely needs, and the reason the list exists is that a repository's history
should show iteration rather than one heap of code.

## v0.2 — turn the demonstration into a result

- [ ] **Measure against real hardware.** Rent an A10G or 4090 for an hour, run
      `concurrency.yaml` and `offered-load.yaml` against a live vLLM server,
      commit the run directories, and rewrite the README's findings with
      measurements. Set `gpu_hourly_usd` to what was actually paid — the cost
      chart is only as honest as that line. *Nothing else on this list is worth
      as much as this one item.*
- [ ] **Replace the synthetic workload with real traffic.** Export
      `(question, retrieved context)` pairs from Vectora into the JSONL shape in
      `docs/WORKLOAD.md` and hand-write 40–60 reference answers. The quality
      gate then scores the real system on its real domain.
- [ ] **Report per-stream decode rate separately from aggregate throughput.**
      Aggregate tok/s rises with batch size while each user's stream slows down;
      only the first is charted today. Both belong on the page.
- [ ] **Bound memory on long open-loop runs.** Results accumulate in a list for
      the whole run; at 50 req/s for an hour that is 180,000 objects. Stream
      them to `raw.jsonl` as they complete.

## v0.3 — depth

- [ ] **Optional LLM-judge scorer behind a flag.** The four deterministic
      scorers cannot see fluency or helpfulness. A judge can, costs money per
      run, and must stay out of CI's default path. Report both and note where
      they disagree.
- [ ] **Per-stage latency breakdown (queue / prefill / decode).** The timings
      are already captured; stacking them turns "it got slower" into "prefill
      got slower".
- [ ] **Methodology note on client overhead.** Measure how much latency the
      Python client itself contributes at high rates and publish the ceiling
      above which these numbers stop being trustworthy.
- [ ] **Fail CI on a quality-gate regression**, not only on test failure.

## v0.4 — the fine-tuning branch, end to end

- [ ] **Distil, train and benchmark a Qwen3-4B student** through the existing
      harness. Serve the adapter alongside the base model in one vLLM process so
      the comparison carries no hardware confound. Publish the result whichever
      way it goes; a documented negative result beats silence.
- [ ] **RESULTS.md** — one page, four charts, every number traced to a
      committed run directory.

## Known limitations (carried until fixed)

- Everything in the current example run comes from the **simulated backend**.
  The analysis is real; the hardware is a model.
- The load generator is Python asyncio. Above roughly 2,000 req/s on one
  process the client becomes a measurable share of the latency.
- The quality scorers are proxies. They can prove a regression; they cannot
  prove an answer is good.
- Cost figures are arithmetic on list prices. They exclude engineering time,
  redundancy, and the gap between peak and mean provisioning.

## Working habits for this repo

Open an issue before fixing something non-trivial, so the noticing is visible
as well as the fixing. Use a branch and a PR for anything substantial, even
solo — the PR description is where the trade-off gets explained. Keep commits
proportionate to their change. Commit run directories alongside the code change
that produced them: a number with a dated run behind it cannot be faked.
