# The real GPU run — runbook

One rented 24 GB GPU (RTX 4090 or A10G), a few hours, a few dollars. Everything
that can be done for free is done before the meter starts.

**What comes out:** concurrency and offered-load curves for Qwen3-4B on vLLM
(bf16 and AWQ int4) on the real Vectora workload, measured from a separate
client machine; a cross-check against vLLM's own benchmark on the same box; and
a quality-gated bf16-vs-AWQ comparison against hand-written references.

---

## Part 1 — free, on your laptop, before renting anything

**1. Export the Vectora workload** (needs `qasper-300` ingested and indexed in
Vectora). From the Vectora folder, with Vectora's venv active:

```powershell
python ..\Vantage\scripts\export_vectora.py --vectora . --corpus qasper-300
```

Writes `Vantage/data/workload.vectora.jsonl` and `eval.vectora.jsonl` (+ the
`.system.txt` sidecars).

**2. Write the reference answers** (~1.5 hours). From the Vantage folder:

```powershell
python scripts\references.py export
```

Open `data\references_todo.csv`, fill `your_reference` for each row (rules at
the top of `scripts/references.py`: 1–3 sentences from the sources only, cite
by `[n]`, `SKIP` if the sources don't answer it). Then:

```powershell
python scripts\references.py merge
git add data/workload.vectora.* data/eval.vectora.* data/eval.handwritten.* data/references_todo.csv
git commit -m "Vectora workload and hand-written references"
git push
```

**3. Dry-run every sweep on the simulator** so a config typo doesn't cost GPU
time: `vantage doctor -b sim.fp16 -w data/workload.vectora.jsonl`.

---

## Part 2 — rent the GPU

Any provider with hourly 24 GB GPUs works (RunPod, Vast.ai, Lambda…). Pick a
PyTorch/CUDA image, an RTX 4090 or A10G, ~50 GB disk. **Write down the hourly
price** — it goes into every cost number.

The client stays on your laptop. Reach the box over an SSH tunnel (simplest,
works through any firewall):

```powershell
ssh -N -L 8000:localhost:8000 -L 8001:localhost:8001 root@<box-ip> -p <ssh-port>
```

Leave that window open. On the laptop, `localhost:8000` is now the box.
(The tunnel adds your network round trip to TTFT; that is what a remote user
sees. The cross-check in Part 4 runs on the box to remove it.)

On the box:

```bash
pip install vllm transformers
git clone https://github.com/SiddharthaRaja/Vantage.git && cd Vantage && pip install -e .
export VLLM_API_KEY=pick-anything
```

---

## Part 3 — the measurements (from the laptop)

Laptop terminal, Vantage folder, venv active:

```powershell
$env:VLLM_API_KEY="pick-anything"      # same as the box
$env:GPU_HOURLY_USD="0.69"             # the real price you wrote down
```

**bf16.** On the box:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --port 8000 --api-key $VLLM_API_KEY
```

Wait for "Application startup complete", then on the laptop:

```powershell
vantage doctor -b vllm.qwen3-4b-2507 -w data/workload.vectora.jsonl
vantage sweep -c configs/sweeps/gpu-concurrency-bf16.yaml
vantage sweep -c configs/sweeps/gpu-offered-load.yaml
```

**AWQ int4.** Stop the bf16 server (Ctrl-C), then on the box:

```bash
vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --port 8001 --api-key $VLLM_API_KEY
```

```powershell
vantage doctor -b vllm.qwen3-4b-2507-awq -w data/workload.vectora.jsonl
vantage sweep -c configs/sweeps/gpu-concurrency-awq.yaml
```

**Quality, bf16 vs AWQ.** Run both servers at once with memory split (commands
in `configs/sweeps/gpu-quantization.yaml`), then:

```powershell
vantage sweep -c configs/sweeps/gpu-quantization.yaml
```

Take speed from the concurrency sweeps, quality from this one.

---

## Part 4 — cross-check against vLLM's own benchmark (on the box)

Same box, same server (bf16 on 8000), same tokens. Copy the workload over
(`scp data/workload.vectora.* root@<box-ip>:Vantage/data/`), then on the box:

```bash
python scripts/vllm_bench_dataset.py --model Qwen/Qwen3-4B-Instruct-2507
for c in 1 8 32; do
  vllm bench serve --backend vllm --model Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
    --endpoint /v1/completions --dataset-name custom \
    --dataset-path data/vllm_bench.vectora.jsonl --skip-chat-template \
    --num-prompts 160 --max-concurrency $c \
    --save-result --result-dir results/vllm-bench --result-filename c$c.json
  vantage bench -b vllm.qwen3-4b-2507 -w data/workload.vectora.jsonl \
    -c $c -n 160 --save
done
```

(Check `vllm bench serve --help` for the output-length flag in your vLLM
version and set it to 192 to match Vantage's `max_tokens`.)

Compare **TTFT p50/p99** and **TPOT/ITL** at each concurrency. Per-token
latency is robust to the two tools stopping at slightly different lengths.
Agreement within ~10% means Vantage measures what vLLM's own tool measures;
a bigger gap is a finding to explain, not to hide.

---

## Part 5 — before you stop the instance

```powershell
scp -P <ssh-port> -r root@<box-ip>:Vantage/results/vllm-bench results/
```

Check every `results/<sweep>/<timestamp>/report.md` exists on the laptop, then
**stop the instance** (stopped instances still bill for disk on some
providers; delete it if you are done). Commit the results:

```powershell
git add results/gpu-* results/vllm-bench
git commit -m "Real GPU run: Qwen3-4B on vLLM, bf16 vs AWQ, cross-checked"
git push
```

Then the README's simulated numbers get replaced with these.
