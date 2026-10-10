#!/usr/bin/env python3
"""QLoRA supervised fine-tune.

What the pieces are, since the acronyms hide simple ideas:

  LoRA    Instead of updating a weight matrix W, learn two small matrices A and
          B and use W + BA. A and B have a tiny fraction of W's parameters, so
          you train ~0.5% of the model and ship a ~100MB adapter instead of a
          new 8GB checkpoint.
  QLoRA   The same, with the frozen base model held in 4-bit. The adapters stay
          in bf16, so quality is close to LoRA-on-fp16, and a 4B model trains
          on one consumer GPU.
  SFT     Supervised fine-tuning: plain next-token prediction on (prompt,
          answer) pairs. No reward model, no RL.

Completion-only loss
--------------------
By default a causal LM is trained to predict *every* token, including the
prompt. For RAG that is actively harmful: the prompt is 95% of the sequence and
is retrieved text, so the model spends its capacity learning to reproduce
source documents instead of learning to answer. This script masks the prompt
and computes loss only on the assistant's reply.

Run
---
    pip install -e ".[finetune]"
    python finetune/train_lora.py --config finetune/configs/qwen3-4b-qlora.yaml

Expect roughly 20-60 minutes for 2 epochs over ~2,000 examples on one 16GB GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def require(package: str, extra: str = "finetune") -> None:
    try:
        __import__(package)
    except ImportError:
        sys.exit(
            f"missing dependency '{package}'.\n"
            f"Install the training extras:  pip install -e \".[{extra}]\"\n"
            "These are deliberately not part of the base install -- the benchmark "
            "harness must stay importable on a laptop with no CUDA."
        )


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="finetune/configs/qwen3-4b-qlora.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg, qcfg, lcfg, tcfg, dcfg = (
        cfg["model"], cfg["quantization"], cfg["lora"], cfg["train"], cfg["data"]
    )

    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return

    for package in ("torch", "transformers", "trl", "peft", "datasets"):
        require(package)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible. This will be unusably slow on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base"], trust_remote_code=mcfg.get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        # Padding with EOS and masking it in the labels is the standard fix for
        # models that ship without a pad token.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant = None
    if qcfg.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
        )

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base"],
        quantization_config=quant,
        dtype=torch.bfloat16 if tcfg.get("bf16", True) else torch.float16,
        attn_implementation=mcfg.get("attn_implementation", "sdpa"),
        trust_remote_code=mcfg.get("trust_remote_code", False),
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=tcfg.get("gradient_checkpointing", True)
        )

    peft_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg.get("dropout", 0.05),
        bias=lcfg.get("bias", "none"),
        target_modules=lcfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} of {total:,} ({trainable / total:.3%})")

    dataset = load_dataset(
        "json",
        data_files={"train": dcfg["train_file"], "validation": dcfg["val_file"]},
    )
    print(f"train: {len(dataset['train'])} examples, validation: {len(dataset['validation'])}")

    sft_config = SFTConfig(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg.get("epochs", 2),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        weight_decay=tcfg.get("weight_decay", 0.0),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        logging_steps=tcfg.get("logging_steps", 10),
        eval_strategy=tcfg.get("eval_strategy", "steps"),
        eval_steps=tcfg.get("eval_steps", 50),
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_steps=tcfg.get("save_steps", 50),
        save_total_limit=tcfg.get("save_total_limit", 2),
        load_best_model_at_end=tcfg.get("load_best_model_at_end", True),
        metric_for_best_model=tcfg.get("metric_for_best_model", "eval_loss"),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=tcfg.get("seed", 17),
        report_to=tcfg.get("report_to", "none"),
        max_length=mcfg.get("max_seq_len", 4096),
        packing=False,  # packing concatenates examples; with completion-only
                        # masking on long RAG prompts it saves little and
                        # complicates the loss mask
        # Mask everything except the assistant turn. TRL reads the chat
        # template to find the boundary.
        assistant_only_loss=dcfg.get("completion_only", True),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(tcfg["output_dir"])
    tokenizer.save_pretrained(tcfg["output_dir"])

    metrics = trainer.evaluate()
    Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(tcfg["output_dir"]) / "train_summary.json").write_text(
        json.dumps({"config": cfg, "final_eval": metrics, "trainable_params": trainable}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(
        f"\nadapter saved to {tcfg['output_dir']}\n"
        "Next:\n"
        f"  vllm serve {mcfg['base']} --enable-lora --lora-modules vantage-ft={tcfg['output_dir']} --port 8000\n"
        "  python finetune/eval_student.py --base vllm.qwen3-4b --student vllm.qwen3-4b-lora\n"
        "\nA falling eval loss is not a result. The result is what eval_student.py prints."
    )


if __name__ == "__main__":
    main()
