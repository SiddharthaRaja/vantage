#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model, and optionally quantise it.

When to merge, and when not to:

  Don't merge  if you are serving with vLLM's --enable-lora. One process holds
               the base weights and swaps adapters per request, so base and
               fine-tune run on identical hardware in the same server. For the
               benchmark that is ideal: it removes every hardware confound from
               the comparison.
  Do merge     to ship a standalone model, to quantise the fine-tuned weights
               (AWQ/GPTQ operate on merged weights), to export GGUF for
               llama.cpp/Ollama, or to upload one artefact to the Hub.

Merging is lossless with respect to the adapter: W_merged = W + BA, exactly the
computation that happened at inference anyway. Quantising afterwards is not
lossless, which is why the quality gate runs again on the quantised model.

Usage
-----
    python finetune/merge_and_export.py \
        --base Qwen/Qwen3-4B-Instruct \
        --adapter outputs/qwen3-4b-vantage-lora \
        --out outputs/qwen3-4b-vantage-merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

GGUF_NOTE = """\
To export GGUF for llama.cpp or Ollama (no Python API; use llama.cpp's script):

    git clone https://github.com/ggerganov/llama.cpp
    python llama.cpp/convert_hf_to_gguf.py {out} --outfile {name}-f16.gguf --outtype f16
    ./llama.cpp/llama-quantize {name}-f16.gguf {name}-q4_k_m.gguf Q4_K_M

Then serve it with Ollama and benchmark it through the same harness:

    ollama create {name} -f Modelfile
    vantage bench -b ollama.{name} -c 8
"""

AWQ_NOTE = """\
To quantise to 4-bit AWQ (needs a calibration set -- use your own prompts, not
wikitext, or you will calibrate for the wrong distribution):

    pip install autoawq
    python - <<'EOF'
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    model = AutoAWQForCausalLM.from_pretrained("{out}")
    tok = AutoTokenizer.from_pretrained("{out}")
    calib = [l for l in open("finetune/data/train.jsonl")][:128]
    model.quantize(tok, quant_config={{"w_bit": 4, "q_group_size": 128,
                                       "zero_point": True, "version": "GEMM"}})
    model.save_quantized("{out}-awq")
    tok.save_pretrained("{out}-awq")
    EOF

Then re-run the quality gate. Quantisation after fine-tuning can undo a
meaningful part of what the fine-tune bought, and the only way to know is to
measure it:

    vantage sweep -c configs/sweeps/quantization.yaml
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--push-to-hub", default="", help="repo id, e.g. your-name/qwen3-4b-vantage")
    parser.add_argument("--print-quantisation-notes", action="store_true")
    args = parser.parse_args()

    name = Path(args.out).name
    if args.print_quantisation_notes:
        print(GGUF_NOTE.format(out=args.out, name=name))
        print(AWQ_NOTE.format(out=args.out))
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        sys.exit(f"missing dependency ({exc}). Install with: pip install -e \".[finetune]\"")

    print(f"loading base model {args.base} in {args.dtype} (not quantised -- merging into 4-bit weights is lossy)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=getattr(torch, args.dtype), device_map="cpu"
    )
    print(f"applying adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"merged model written to {args.out}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"pushed to https://huggingface.co/{args.push_to_hub}")
        print(
            "Put the eval numbers in the model card. A published model with no "
            "evaluation is a liability, not a portfolio piece."
        )

    print("\n" + GGUF_NOTE.format(out=args.out, name=name))


if __name__ == "__main__":
    main()
