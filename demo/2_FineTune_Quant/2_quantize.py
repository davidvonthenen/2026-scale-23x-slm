#!/usr/bin/env python3
"""
Quantize & serialize a merged Qwen SQL model for CPU or Apple MPS inference.

What this script does:
- AutoAWQ's fast path is GPU-centric; CPU inference relies on IPEX (Intel-only),
  and MPS isn't the intended target. So we switch to Optimum-Quanto, which is
  device-agnostic (CPU/MPS/CUDA) and safetensors-friendly.

Outputs:
- A standard HF model directory (default: ./qwen_sql_quanto_int4)
  containing config + quantized weights (safetensors) + tokenizer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optimum-Quanto (device-agnostic quantization)
from optimum.quanto import QuantizedModelForCausalLM, qint4, qint8

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


SYSTEM_PROMPT = "You are a database engineer. Generate valid SQL for the given schema."

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize Qwen SQL model with Optimum-Quanto for CPU/MPS inference.")
    p.add_argument("--model_dir", type=str, default="./merged_qwen_sql", help="Path to merged HF model dir.")
    p.add_argument("--output_dir", type=str, default="./qwen_sql_quanto_int8", help="Where to write quantized HF dir.")
    p.add_argument("--weights", type=str, choices=["int4", "int8"], default="int8", help="Weight quantization dtype.")
    p.add_argument(
        "--torch_dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Base dtype used when loading the FP model before quantizing.",
    )
    p.add_argument(
        "--exclude",
        type=str,
        default="lm_head",
        help='Comma-separated module name patterns to exclude from quantization (default: "lm_head").',
    )
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    return p.parse_args()


def _dtype_from_str(s: str):
    if s == "auto":
        return "auto"
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    torch_dtype = _dtype_from_str(args.torch_dtype)

    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=args.trust_remote_code,
        fix_mistral_regex=True,  # keeps parity with your earlier workaround
    )

    print(f"Loading base model from: {model_dir}")
    # Note: Keep it on CPU for quantization for maximum portability.
    # low_cpu_mem_usage helps reduce peak RAM while loading large models.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,  # keep simple/portable; loads on CPU
    )
    model.eval()

    weights_dtype = qint4 if args.weights == "int4" else qint8
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]

    print(f"Quantizing with Optimum-Quanto (weights={args.weights}, exclude={exclude})...")
    qmodel = QuantizedModelForCausalLM.quantize(
        model,
        weights=weights_dtype,
        exclude=exclude,  # keep lm_head in higher precision by default
    )
    qmodel.eval()

    # Save a standard HF directory with safetensors weights.
    print(f"Saving quantized model to: {out_dir}")
    qmodel.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    # Optional: save a tiny metadata breadcrumb for humans
    (out_dir / "quantization_meta.txt").write_text(
        f"backend=optimum-quanto\nweights={args.weights}\nexclude={exclude}\n",
        encoding="utf-8",
    )

    print("\n✅ Quantization complete.")
    print(f"✅ Quantized HF model dir: {out_dir.resolve()}")
    print("✅ This directory can be loaded on CPU or Apple MPS.\n")


if __name__ == "__main__":
    main()
