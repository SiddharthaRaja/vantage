"""
Build the two Kaggle 2x T4 notebooks from one set of cells:

    notebooks/vantage_kaggle.ipynb       the baseline: capacity, open loop, quality
                                         gate, cross-check, cache
    notebooks/vantage_kaggle_tune.ipynb  tuning: vLLM settings one at a time, with
                                         server metrics (configs/sweeps/kaggle-t4-tune)

    python notebooks/build_kaggle.py

The notebooks are plain Python (no shell magics), so the same cells also run
outside Kaggle in a dry-run mode: the baseline against the simulator, the
tuning run against tests/fake_vllm.py, which Vantage starts and stops itself:

    VANTAGE_DRY_RUN=1 VANTAGE_WORK=/tmp/vk VANTAGE_REPO_URL=. python notebooks/build_kaggle.py --run
    VANTAGE_DRY_RUN=1 VANTAGE_WORK=/tmp/vk VANTAGE_REPO_URL=. python notebooks/build_kaggle.py --run --tune

which is how they are tested before they cost GPU time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {"base": HERE / "vantage_kaggle.ipynb", "tune": HERE / "vantage_kaggle_tune.ipynb"}

# (kind, text, only): only=None means the cell is in both notebooks
CELLS: list[tuple[str, str, str | None]] = []


def md(text: str, only: str | None = None) -> None:
    CELLS.append(("markdown", text.strip("\n"), only))


def code(text: str, only: str | None = None) -> None:
    CELLS.append(("code", text.strip("\n"), only))


md("""
# Vantage on real hardware: Kaggle 2x T4

Every Vantage number so far came from its simulator. This run replaces them with measurements of
**Qwen3-4B-Instruct-2507** served by **vLLM**, driven by **Vectora's real workload** (365 prompts
exported from qasper-300, each sent exactly as Vectora sends it).

| GPU | model | port |
|---|---|---|
| T4 #0 | Qwen3-4B fp16 (the T4 has no bf16) | 8000 |
| T4 #1 | the same model, AWQ int4 | 8001 |

| step | question | output |
|---|---|---|
| sweep `kaggle-t4` | capacity: concurrency 1 to 64 per precision; cost per 1,000 requests; **quality gate** on the 32 reference answers | `kaggle-t4/REPORT.md` |
| sweep `kaggle-t4-offered-load` | where is the saturation cliff under Poisson arrivals? | `kaggle-t4-offered-load/REPORT.md` |
| cross-check | does Vantage measure what vLLM's own `vllm bench serve` measures? | `CROSSCHECK.md` |
| cache-sim (CPU) | is a semantic cache safe for this workload? | `cache_sim.json` |

**To run:** Settings: Accelerator **GPU T4 x2**, Internet **on**. Then **Save Version, Save & Run All
(Commit)**, so it runs unattended and the output is kept. Expect about 3 hours.
Everything lands in `/kaggle/working/<run>/` and `vantage_kaggle_results.zip`; send me the zip.

Each step records ok / failed / skipped in `steps.json`; one failure never stops the steps that do
not depend on it.
""", only="base")

md("""
# Vantage tuning: Kaggle 2x T4

The baseline run (`vantage_kaggle.ipynb`) measured where latency breaks. This run explains why and
tries to move it. Vantage starts vLLM with one setting changed at a time, drives the same load
points, and scrapes vLLM's `/metrics` every second: KV-cache use, requests waiting inside the
engine, preemptions, prefix-cache hits.

| server | change | question |
|---|---|---|
| `base` | none (baseline settings), concurrency 1 to 64 | where does the KV cache fill, and does that match the knee? |
| `noprefix` | prefix caching off | what is reusing the shared system prompt worth? |
| `mem95` | `--gpu-memory-utilization 0.95` | does more KV cache move the knee? |
| `seqs32` | `--max-num-seqs 32` | does capping the batch protect latency past the knee? |
| `chunk512` | `--max-num-batched-tokens 512` | do smaller prefill chunks cut inter-token latency spikes? |
| `awq` | AWQ int4 on T4 #1 | why AWQ is faster or slower than fp16 here |

All settings are in `configs/sweeps/kaggle-t4-tune.yaml`; the exact command of every server is
written to `servers.json`.

**To run:** Accelerator **GPU T4 x2**, Internet **on**, then **Save Version, Save & Run All
(Commit)**. About 2.5 hours. Send me `vantage_kaggle_tune_results.zip`.
""", only="tune")

md("## 1. Settings and helpers")
code(r'''
import datetime, glob, json, os, shutil, subprocess, sys, time, traceback, urllib.request

NOTEBOOK = "__NOTEBOOK__"                              # "base" or "tune"; set by build_kaggle.py
DRY_RUN = os.environ.get("VANTAGE_DRY_RUN") == "1"     # no GPUs: simulator / fake server (testing)
WORK = os.environ.get("VANTAGE_WORK", "/kaggle/working")
REPO_URL = os.environ.get("VANTAGE_REPO_URL", "https://github.com/SiddharthaRaja/vantage.git")
# Not ".../vantage": a folder named like the package in the working directory
# shadows the installed package when Python looks there first.
REPO = f"{WORK}/vantage_repo"
RUN = ("kaggle_t4_tune_" if NOTEBOOK == "tune" else "kaggle_t4_") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H%M")
OUT = f"{WORK}/{RUN}"
os.makedirs(OUT, exist_ok=True)

FP16 = "Qwen/Qwen3-4B-Instruct-2507"
AWQ = "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"
VLLM_CANDIDATES = ["vllm", "vllm==0.10.2", "vllm==0.9.2"]   # first one that serves on a T4 wins
CROSSCHECK = [(1, 48), (8, 96), (32, 160)]                  # (concurrency, requests), as in the sweep

os.environ.update(VLLM_API_KEY="EMPTY", PYTHONUNBUFFERED="1",
                  GPU_HOURLY_USD=os.environ.get("GPU_HOURLY_USD", "0.35"))
STEPS = {}
T0 = time.time()


def sh(cmd, log=None, timeout=None, cwd=None, env=None):
    """Run a shell command, stream it to a log file, return the exit code (never raises)."""
    log = log or f"{OUT}/shell.log"
    with open(log, "a") as fh:
        fh.write(f"\n$ {cmd}\n"); fh.flush()
        try:
            p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=fh,
                               stderr=subprocess.STDOUT, timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"TIMEOUT after {timeout}s\n")
            return 124


def tail(path, n=25):
    try:
        print("".join(open(path, errors="replace").readlines()[-n:]))
    except OSError:
        print(f"(no {path})")


def run_step(name, fn, needs=()):
    """Run one step; record ok / failed / skipped; never raise."""
    missing = [d for d in needs if STEPS.get(d, {}).get("status") != "ok"]
    t = time.time()
    if missing:
        STEPS[name] = {"status": "skipped", "why": f"needs {', '.join(missing)}"}
    else:
        try:
            note = fn()
            STEPS[name] = {"status": "ok", "note": note, "minutes": round((time.time() - t) / 60, 1)}
        except Exception as e:  # noqa: BLE001
            STEPS[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}",
                           "minutes": round((time.time() - t) / 60, 1)}
            traceback.print_exc()
    json.dump(STEPS, open(f"{OUT}/steps.json", "w"), indent=2, default=str)
    print(f"[{name}] {STEPS[name]['status']}  {STEPS[name].get('note') or STEPS[name].get('error') or STEPS[name].get('why') or ''}")


print("run ->", OUT, "| dry run" if DRY_RUN else "")
''')

md("## 2. GPUs, Vantage, workload")
code(r'''
def setup():
    if not DRY_RUN:
        sh("nvidia-smi", log=f"{OUT}/nvidia_smi.txt")
        n = subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip()
        if n != "2":
            raise RuntimeError(f"expected 2 GPUs, found {n}: set Accelerator to GPU T4 x2")
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    if sh(f"git clone -q {REPO_URL} {REPO}") != 0:
        raise RuntimeError("git clone failed")
    sh("git log --oneline -3", log=f"{OUT}/commit.txt", cwd=REPO)
    if sh(f"{sys.executable} -m pip install -q -e {REPO}", timeout=900) != 0:
        raise RuntimeError("pip install of Vantage failed (see shell.log)")
    for f in ["data/workload.vectora.jsonl", "data/eval.references.jsonl"]:
        if not os.path.exists(f"{REPO}/{f}"):
            raise RuntimeError(f"missing {f}")
    sh("vantage workload data/workload.vectora.jsonl", log=f"{OUT}/workload_shape.txt", cwd=REPO)
    commits = [l for l in open(f"{OUT}/commit.txt").read().splitlines() if l.strip() and not l.startswith("$")]
    return commits[0][:70] if commits else "cloned"

run_step("setup", setup)
''')

md("""
## 3. vLLM

vLLM goes into **its own Python 3.12 environment** (made with `uv`), not Kaggle's. Kaggle's image is
Python 3.13 with hundreds of preinstalled packages; installing vLLM into it once crashed vLLM on
start-up inside `transformers`' audio imports, and older vLLM versions have no Python 3.13 builds at
all. Vantage only talks to vLLM over HTTP, so the two never need to share packages. The
environment's `bin` goes first on `PATH`, so every `vllm` command below is the isolated one.

The newest vLLM is tried first; if it cannot serve on a T4 (compute capability 7.5), the next
candidate is installed into a fresh environment in step 4. The version used goes into the report.
""")
code(r'''
VLLM_VERSION = None

VENV = "/tmp/vllm-venv"     # outside /kaggle/working, so it is not saved as output

def install_vllm(spec):
    """A fresh, isolated Python 3.12 environment with `spec` installed; its bin first on PATH."""
    global VLLM_VERSION
    if DRY_RUN:
        VLLM_VERSION = "dry-run"; return VLLM_VERSION
    log = f"{OUT}/vllm_install.log"
    if sh(f"{sys.executable} -m pip install -q uv", log=log, timeout=600) != 0:
        raise RuntimeError("pip install uv failed")
    shutil.rmtree(VENV, ignore_errors=True)
    if sh(f"{sys.executable} -m uv venv -q --python 3.12 {VENV}", log=log, timeout=900) != 0:
        raise RuntimeError("could not create the Python 3.12 environment (see vllm_install.log)")
    if sh(f"{sys.executable} -m uv pip install -q --python {VENV}/bin/python '{spec}'", log=log, timeout=1800) != 0:
        raise RuntimeError(f"installing {spec} failed (see vllm_install.log)")
    if not os.environ["PATH"].startswith(f"{VENV}/bin:"):
        os.environ["PATH"] = f"{VENV}/bin:" + os.environ["PATH"]
    VLLM_VERSION = subprocess.run([f"{VENV}/bin/python", "-c", "import vllm; print(vllm.__version__)"],
                                  capture_output=True, text=True).stdout.strip()
    if not VLLM_VERSION:
        raise RuntimeError(f"{spec} installed but does not import (see vllm_install.log)")
    sh(f"{VENV}/bin/python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())'",
       log=log)
    return f"vllm {VLLM_VERSION} (isolated Python 3.12)"

run_step("install_vllm", lambda: install_vllm(VLLM_CANDIDATES[0]), needs=["setup"])
''')

md("""
## 4. Two servers, one per GPU

`--max-model-len 4096`: Vectora prompts are ~900 tokens with the system prompt, answers at most 256,
so nothing is truncated; without a limit vLLM would reserve memory for the model's full 262k context.
If a server does not come up, it is retried with `--enforce-eager`, then with the next vLLM version.
""", only="base")

md("""
## 4. Pre-flight: which vLLM serves on a T4?

Starts the fp16 server once, with the same fallbacks as the baseline run (`--enforce-eager`, then
older vLLM versions), and stops it again. Whatever flag it needed is passed to every server in the
tuning sweep through `VLLM_EXTRA`, so all variants differ only in the setting under test.
""", only="tune")
code(r'''
SERVERS = {}
VLLM_EXTRA = ""

def _up(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3)
        return True
    except Exception:
        return False


def launch(model, gpu, port, extra=()):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"{OUT}/server_{port}.log", "a")
    cmd = ["vllm", "serve", model, "--dtype", "half", "--max-model-len", "4096",
           "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--port", str(port), *extra]
    log.write("\n$ " + " ".join(cmd) + "\n"); log.flush()
    SERVERS[port] = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_up(ports, limit_s=1500):
    t = time.time()
    while time.time() - t < limit_s:
        if all(_up(p) for p in ports):
            return True
        if any(SERVERS[p].poll() is not None for p in ports):
            return False
        time.sleep(10)
    return False


def stop_servers():
    for p in SERVERS.values():
        p.terminate()
    time.sleep(10)
    for p in SERVERS.values():
        if p.poll() is None:
            p.kill()
    SERVERS.clear()
    sh("pkill -f 'vllm serve' || true")
    time.sleep(5)


def start_servers(models=((None, 0, 8000), (None, 1, 8001))):
    """Start one server per (model, gpu, port); fall back until one vLLM version serves."""
    global VLLM_EXTRA
    models = [(m or (FP16 if gpu == 0 else AWQ), gpu, port) for m, gpu, port in models]
    if DRY_RUN:
        VLLM_EXTRA = ""
        return "simulator"
    attempts = [(VLLM_CANDIDATES[0], ())] + [(VLLM_CANDIDATES[0], ("--enforce-eager",))] + \
               [(v, ("--enforce-eager",)) for v in VLLM_CANDIDATES[1:]]
    current = VLLM_CANDIDATES[0]
    for spec, extra in attempts:
        if spec != current:
            current = spec
            try:
                install_vllm(spec)
            except RuntimeError as e:
                print(f"{spec}: {e}")
                continue
        for model, gpu, port in models:
            launch(model, gpu, port, extra)
        ports = [port for _, _, port in models]
        if wait_up(ports):
            VLLM_EXTRA = " ".join(extra)
            return f"vllm {VLLM_VERSION} {VLLM_EXTRA or '(cuda graphs on)'}"
        print(f"servers did not start with {spec} {extra}; last lines:")
        for port in ports:
            tail(f"{OUT}/server_{port}.log", 15)
        stop_servers()
    raise RuntimeError("no vLLM version served on the T4s (see server_*.log)")
''')

code(r'''
run_step("servers", start_servers, needs=["install_vllm"])
''', only="base")

code(r'''
def preflight():
    note = start_servers(models=[(FP16, 0, 8000)])
    stop_servers()
    os.environ["VLLM_EXTRA"] = VLLM_EXTRA        # read by ${VLLM_EXTRA:-} in kaggle-t4-tune.yaml
    return note

run_step("preflight", preflight, needs=["install_vllm"])
''', only="tune")

md("""
## 5. Is the whole prompt reaching the model?

One real Vectora prompt to each server. The server's own count of prompt tokens should be ~900, and
the two precisions should read the same number of tokens.
""", only="base")
code(r'''
def ask(port, model, system, user, max_tokens=192):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"], time.time() - t, r["choices"][0]["message"]["content"]


def sanity():
    if DRY_RUN:
        return "skipped in dry run"
    row = json.loads(open(f"{REPO}/data/workload.vectora.jsonl").readline())
    system = open(f"{REPO}/data/workload.vectora.system.txt").read()
    lines, counts = [], []
    for port, model in [(8000, FP16), (8001, AWQ)]:
        pt, ct, s, text = ask(port, model, system, row["prompt"])
        counts.append(pt)
        lines.append(f"{model}: prompt_tokens={pt} completion_tokens={ct} {s:.1f}s\n{text}\n")
    open(f"{OUT}/sanity_check.txt", "w").write(f"question: {row['question']}\n\n" + "\n".join(lines))
    print(open(f"{OUT}/sanity_check.txt").read())
    if min(counts) < 400:
        raise RuntimeError(f"prompt tokens {counts}: the prompt is being cut")
    return f"prompt tokens {counts}"

run_step("sanity", sanity, needs=["servers"])
''', only="base")

md("""
## 6. The sweeps

Load points run one at a time. Raw per-request records, charts and `REPORT.md` are copied into the
run folder as each sweep finishes.
""")
code(r'''
SIM = {"vllm.qwen3-4b-2507-awq-t4": "sim.int4", "vllm.qwen3-4b-2507-t4": "sim.fp16"}

FAKE = {"vllm serve Qwen/Qwen3-4B-Instruct-2507": f"{sys.executable} {REPO}/tests/fake_vllm.py --model Qwen/Qwen3-4B-Instruct-2507",
        "vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit": f"{sys.executable} {REPO}/tests/fake_vllm.py --model cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"}

def sweep_config(name):
    path = f"{REPO}/configs/sweeps/{name}.yaml"
    if not DRY_RUN:
        return path
    text = open(path).read()                       # dry run: no GPU, tiny loads
    if "servers:" in text:                         # managed servers: a fake vLLM that Vantage starts
        for real, fake in FAKE.items():
            text = text.replace(real, fake)
    else:                                          # servers started by the notebook: the simulator
        for real, sim in SIM.items():
            text = text.replace(real, sim)
    for a, b in [("n_requests: 160", "n_requests: 12"), ("n_requests: 96", "n_requests: 8"),
                 ("n_requests: 48", "n_requests: 4"), ("n_requests: 30", "n_requests: 3"),
                 ("duration_s: 60", "duration_s: 2"), ("cooldown_s: 10", "cooldown_s: 0"),
                 ("cooldown_s: 5", "cooldown_s: 0"), ("[0.25, 0.5]", "[2, 4]")]:
        text = text.replace(a, b)
    os.makedirs(f"{OUT}/dry_configs", exist_ok=True)
    open(f"{OUT}/dry_configs/{name}.yaml", "w").write(text)
    return f"{OUT}/dry_configs/{name}.yaml"


def sweep(name, timeout_h=4):
    log = f"{OUT}/{name}.log"
    rc = sh(f"vantage sweep -c {sweep_config(name)}", log=log, cwd=REPO, timeout=int(timeout_h * 3600))
    runs = sorted(glob.glob(f"{REPO}/results/{name}/*"))
    if runs:
        shutil.copytree(runs[-1], f"{OUT}/{name}", dirs_exist_ok=True)
    tail(log, 12)
    if rc != 0 or not os.path.exists(f"{OUT}/{name}/REPORT.md"):
        raise RuntimeError(f"sweep exited {rc} (see {name}.log)")
    gate = f"{OUT}/{name}/quality_gate.json"
    if os.path.exists(gate):
        g = json.load(open(gate))[0]
        return f"gate {g['verdict']}: delta {g['delta']:+.3f}, 95% CI {g['delta_ci95']}"
    return "done"
''')
code(r'''
run_step("kaggle-t4", lambda: sweep("kaggle-t4"), needs=["sanity"] if not DRY_RUN else ["setup"])
''', only="base")
code(r'''
run_step("kaggle-t4-offered-load", lambda: sweep("kaggle-t4-offered-load", timeout_h=2),
         needs=["sanity"] if not DRY_RUN else ["setup"])
''', only="base")
code(r'''
run_step("kaggle-t4-tune", lambda: sweep("kaggle-t4-tune", timeout_h=5), needs=["preflight"])
''', only="tune")

md("""
## 7. Cross-check against vLLM's own benchmark

Same server (fp16, port 8000), same input tokens: `scripts/vllm_bench_dataset.py` applies the model's
chat template, so `vllm bench serve` sends exactly what Vantage sends. Compared at concurrency 1, 8
and 32 with the request counts the sweep used. Agreement within ~10% on TTFT and inter-token latency
means Vantage measures what vLLM measures; a bigger gap is a finding to explain.
""", only="base")
code(r'''
BENCH_FLAGS = [
    ["--custom-output-len", "192", "--skip-chat-template"],
    ["--custom-output-len", "192", "--custom-skip-chat-template"],
    ["--skip-chat-template"],
]

def vllm_bench(c, n):
    out = f"{OUT}/vllm-bench"
    os.makedirs(out, exist_ok=True)
    base = (f"vllm bench serve --backend vllm --model {FP16} --base-url http://127.0.0.1:8000 "
            f"--endpoint /v1/completions --dataset-name custom --dataset-path data/vllm_bench.vectora.jsonl "
            f"--num-prompts {n} --max-concurrency {c} --save-result --result-dir {out} --result-filename c{c}.json")
    for flags in BENCH_FLAGS:
        if sh(base + " " + " ".join(flags), log=f"{OUT}/vllm_bench.log", cwd=REPO, timeout=3600) == 0:
            return json.load(open(f"{out}/c{c}.json")), flags
    raise RuntimeError("vllm bench serve failed with every flag set (see vllm_bench.log)")


def crosscheck():
    if DRY_RUN:
        return "skipped in dry run"
    rc = sh(f"{sys.executable} scripts/vllm_bench_dataset.py --model {FP16}", log=f"{OUT}/vllm_bench.log", cwd=REPO)
    if rc != 0:                                    # Kaggle's transformers failed: try the vLLM environment's
        sh(f"{sys.executable} -m uv pip install -q --python {VENV}/bin/python pyyaml httpx pydantic",
           log=f"{OUT}/vllm_bench.log")
        rc = sh(f"{VENV}/bin/python scripts/vllm_bench_dataset.py --model {FP16}", log=f"{OUT}/vllm_bench.log", cwd=REPO)
    if rc != 0:
        raise RuntimeError("could not build the vllm bench dataset")
    ours = {s["config_id"]: s for s in json.load(open(f"{OUT}/kaggle-t4/summaries.json"))}
    rows = ["| concurrency | metric | Vantage | vllm bench | Vantage / vllm |", "|---|---|---|---|---|"]
    used = None
    for c, n in CROSSCHECK:
        v, used = vllm_bench(c, n)
        s = ours.get(f"fp16[concurrency={c}]")
        if not s:
            continue
        ms = lambda x: x * 1000 if x is not None else None  # noqa: E731
        num = lambda x: f"{x:,.1f}" if x is not None else "n/a"  # noqa: E731
        pairs = [("TTFT p50 (ms)", ms(s["ttft"]["p50"]), v.get("median_ttft_ms")),
                 ("TTFT p99 (ms)", ms(s["ttft"]["p99"]), v.get("p99_ttft_ms")),
                 ("ITL p50 (ms)", ms(s["itl"]["p50"]), v.get("median_itl_ms")),
                 ("output tok/s", s["output_tokens_per_s"], v.get("output_throughput"))]
        for metric, a, b in pairs:
            ratio = f"{a / b:.2f}" if a and b else "n/a"
            rows.append(f"| {c} | {metric} | {num(a)} | {num(b)} | {ratio} |")
    text = (f"# Cross-check: Vantage vs `vllm bench serve` (vLLM {VLLM_VERSION}, flags {used})\n\n"
            "Same server, same input tokens. Throughput can differ where the two tools stop answers at "
            "different lengths; per-token latency should not.\n\n" + "\n".join(rows) + "\n")
    open(f"{OUT}/CROSSCHECK.md", "w").write(text)
    print(text)
    return "written"

run_step("crosscheck", crosscheck, needs=["kaggle-t4"])
''', only="base")

md("""
## 8. Semantic cache (CPU)

Replays Vectora's questions, with paraphrases, through a cache at each similarity threshold. A false
hit serves one question's answer to a different question; title-free QASPER questions like "What
are the baselines?" repeat across papers with different answers.
""", only="base")
code(r'''
def cache_sim():
    embedder = "hashing" if DRY_RUN else "st"
    if not DRY_RUN:
        sh(f"{sys.executable} -m pip install -q sentence-transformers", timeout=900)
        os.environ["VANTAGE_EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    rc = sh(f"vantage cache-sim -w data/workload.vectora.jsonl --embedder {embedder} --repeats 5 "
            f"--out {OUT}/cache_sim.json", log=f"{OUT}/cache_sim.log", cwd=REPO, timeout=3600)
    tail(f"{OUT}/cache_sim.log", 14)
    if rc != 0:
        raise RuntimeError("cache-sim failed")
    return json.load(open(f"{OUT}/cache_sim.json")).get("recommended_threshold")

run_step("cache_sim", cache_sim, needs=["setup"])
''', only="base")

md("## 9. Summary and the zip")
code(r'''
def package():
    if SERVERS:
        stop_servers()
    sh("nvidia-smi", log=f"{OUT}/nvidia_smi_end.txt")
    title = "Vantage tuning on Kaggle 2x T4" if NOTEBOOK == "tune" else "Vantage on Kaggle 2x T4"
    parts = [f"# {title} ({RUN})", "",
             f"vLLM {VLLM_VERSION}; total {(time.time() - T0) / 3600:.2f} h; dry run: {DRY_RUN}", "",
             "| step | status | note |", "|---|---|---|"]
    for k, v in STEPS.items():
        parts.append(f"| {k} | {v['status']} | {v.get('note') or v.get('error') or v.get('why') or ''} |")
    for f in ["sanity_check.txt"]:
        if os.path.exists(f"{OUT}/{f}"):
            parts += ["", "## Sanity check", "", "```", open(f"{OUT}/{f}").read()[:3000], "```"]
    for name in ["kaggle-t4", "kaggle-t4-offered-load", "kaggle-t4-tune"]:
        if os.path.exists(f"{OUT}/{name}/REPORT.md"):
            parts += ["", "---", "", open(f"{OUT}/{name}/REPORT.md").read()]
    if os.path.exists(f"{OUT}/CROSSCHECK.md"):
        parts += ["", "---", "", open(f"{OUT}/CROSSCHECK.md").read()]
    if os.path.exists(f"{OUT}/cache_sim.log"):
        parts += ["", "---", "", "# Semantic cache", "", "```", open(f"{OUT}/cache_sim.log").read()[-3000:], "```"]
    open(f"{OUT}/SUMMARY.md", "w").write("\n".join(parts) + "\n")
    stem = "vantage_kaggle_tune_results" if NOTEBOOK == "tune" else "vantage_kaggle_results"
    zip_path = shutil.make_archive(f"{WORK}/{stem}", "zip", WORK, RUN)
    return f"{zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)"

run_step("package", package)
print(open(f"{OUT}/SUMMARY.md").read()[:4000])
''')


def cells_for(name: str) -> list[tuple[str, str]]:
    return [(kind, text.replace("__NOTEBOOK__", name)) for kind, text, only in CELLS if only in (None, name)]


def build(name: str) -> dict:
    cells = []
    for i, (kind, text) in enumerate(cells_for(name)):
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {},
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                       "isGpuEnabled": True, "language": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    for name, path in NOTEBOOKS.items():
        nb = build(name)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(nb['cells'])} cells)")
    if "--run" in sys.argv:          # execute the code cells in order (dry-run testing)
        scope: dict = {"__name__": "__main__"}
        for kind, text in cells_for("tune" if "--tune" in sys.argv else "base"):
            if kind == "code":
                exec(compile(text, "<notebook>", "exec"), scope)
