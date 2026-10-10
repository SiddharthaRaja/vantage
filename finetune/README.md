# The fine-tuning branch

The serving harness measures models. This directory makes one.

The claim at the end is a single sentence: *a 4B model fine-tuned on distilled
answers recovers N% of the teacher's quality on this task while serving it at
X times lower cost per request* — measured by the same harness, on the same
held-out set, under the same load.

## Why this is here at all

Everything else in Vantage is inference-side: it calls models, measures them,
caches them, costs them. None of it demonstrates that you can train one. This
branch closes that gap without becoming a second project, because it reuses the
harness's backends, its eval set and its quality scorers. The fine-tuned model
is just one more row in the same table.

## The sequence

```bash
# 0. install training deps (not in the base install -- the harness must stay
#    importable on a laptop with no CUDA)
pip install -e ".[finetune]"

# 1. distil: let the teacher answer your corpus, keep only answers that pass
#    the same grounding checks the serving gate uses
python finetune/build_dataset.py \
    --workload data/workload.sample.jsonl \
    --teacher openai.gpt-4o-mini \
    --out finetune/data

# 2. train: QLoRA, ~20-60 min on one 16GB GPU
python finetune/train_lora.py --config finetune/configs/qwen3-4b-qlora.yaml

# 3. serve base and adapter from ONE vLLM process, so the comparison has no
#    hardware confound
vllm serve Qwen/Qwen3-4B-Instruct --enable-lora \
    --lora-modules vantage-ft=outputs/qwen3-4b-vantage-lora \
    --port 8000 --max-model-len 8192

# 4. measure: quality, latency, throughput, cost, all three models
python finetune/eval_student.py \
    --teacher openai.gpt-4o-mini \
    --base vllm.qwen3-4b \
    --student vllm.qwen3-4b-lora

# 5. and in the full harness, with charts
vantage sweep -c configs/sweeps/finetune-compare.yaml
```

## The parts that decide whether it works

**Lock the test split before you generate anything.** `build_dataset.py` hashes
the item id to assign splits, so the boundary survives re-runs and added data.
The test split is written as questions only — no teacher answer is ever stored
next to a held-out question, so it cannot leak into training by accident.

**Filter the teacher.** A frontier model invents citations and numbers too.
Every generated answer goes through `citation_grounding` and
`number_agreement` from the serving quality harness; rejects are counted and
reported by reason. If the rejection rate is above ~30%, the prompt is wrong
and training on what survives will bake the problem in.

**Train on the serving prompt.** The template used in training is the one the
serving stack sends. A model fine-tuned on one prompt format and served with
another loses most of the gain, and it is the single most common silent failure
in student fine-tunes.

**Loss on the answer only.** RAG prompts are enormous and the answer is short.
Training on every token means ~95% of the gradient teaches the model to
reproduce retrieved documents. `assistant_only_loss` masks the prompt.

**Eval loss is not a result.** A falling loss curve means the model learned the
distribution of your training data. Whether it answers held-out questions
better than the model you started from is a different question, and
`eval_student.py` is the one that answers it.

## What "it didn't work" looks like, and what to do

| Symptom | Likely cause | Check |
|---|---|---|
| Student scores below base | prompt mismatch, or adapter never loaded | diff the training prompt against the serving prompt; check vLLM's startup log for the registered `lora-modules` |
| Student matches base exactly | adapter not applied at inference | request the adapter's model name, not the base model's |
| Loss falls, quality doesn't | memorising a too-small or too-repetitive set | how many examples survived filtering and dedup? Under ~500 is usually not enough |
| Citations degrade | completion-only masking off, or the teacher's own citations were weak | inspect 20 training examples by hand; it takes ten minutes and catches most of this |
| Good on test, bad in the app | the eval set is not your traffic | rebuild the eval set from real logged questions |

## Hardware and cost, honestly

| | GPU | Time | Rough cost |
|---|---|---|---|
| Distillation (2k examples, teacher) | none | ~10 min | a few dollars of API spend |
| QLoRA, 4B, 2 epochs, 2k examples | 16GB (T4 / A10G / 4090) | 20-60 min | under $1 rented, free on Colab |
| Merge + export | CPU, 32GB RAM | ~5 min | free |
| Benchmark both models | the serving GPU | ~15 min | pennies |

The whole branch is a weekend. The harness is what makes it one: the eval set,
the scorers, the load generator and the cost model already exist.
