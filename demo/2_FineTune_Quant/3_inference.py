#!/usr/bin/env python3
"""
Qwen SQL inference (Optimum-Quanto int8) with an MPS-safe GQA workaround.

This avoids the head-dimension mismatch (e.g., 28 vs 4 heads) that triggers the crash.
"""

from __future__ import annotations

import builtins
import logging
import os
import time
import warnings
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Dependency Hotfix: Diffusers 'logger' NameError Bypass
# -----------------------------------------------------------------------------
if getattr(builtins, "logger", None) is None:
    builtins.logger = logging.getLogger("diffusers_runtime_patch")

from optimum.quanto import QuantizedModelForCausalLM
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# Warnings / env
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------------------------------------------------------------
# MPS-safe SDPA(GQA) patch
# -----------------------------------------------------------------------------

def _patch_sdpa_for_mps_gqa() -> None:
    """Patch torch SDPA so `enable_gqa=True` does not crash on MPS."""

    if not torch.backends.mps.is_available():
        return

    original_sdpa = F.scaled_dot_product_attention

    # Parse PyTorch version to safely handle the enable_gqa parameter
    # The enable_gqa flag is supported in PyTorch >= 2.4
    torch_version = torch.__version__.split("+")[0]
    major, minor = map(int, torch_version.split(".")[:2])
    supports_enable_gqa = major > 2 or (major == 2 and minor >= 4)

    def sdpa_mps_gqa_safe(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if enable_gqa and query.device.type == "mps":
            # query: (..., Hq, L, E)
            # key/value: (..., Hkv, S, E)
            hq = query.size(-3)
            hk = key.size(-3)
            hv = value.size(-3)

            if hk != hv:
                raise RuntimeError(
                    f"GQA requires key/value head counts to match; got hk={hk}, hv={hv}"
                )
            if hk == 0:
                raise RuntimeError("Invalid key head dimension (0)")
            if hq % hk != 0:
                raise RuntimeError(
                    f"GQA requires query_heads % kv_heads == 0; got hq={hq}, hk={hk}"
                )

            repeat = hq // hk
            key = key.repeat_interleave(repeat, dim=-3).contiguous()
            value = value.repeat_interleave(repeat, dim=-3).contiguous()
            
            # Since we manually expanded the tensors, we instruct the backend
            # to bypass its internal GQA logic.
            enable_gqa = False

        kwargs: Dict[str, Any] = {
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if scale is not None:
            kwargs["scale"] = scale
            
        if supports_enable_gqa:
            kwargs["enable_gqa"] = enable_gqa

        return original_sdpa(query, key, value, **kwargs)

    F.scaled_dot_product_attention = sdpa_mps_gqa_safe  # type: ignore[assignment]


_patch_sdpa_for_mps_gqa()


# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        print(">> Using Apple-Silicon MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(">> Using CUDA GPU")
        return torch.device("cuda")
    print(">> Using CPU")
    return torch.device("cpu")


device = _select_device()
compute_dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32


# -----------------------------------------------------------------------------
# Model + tokenizer
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a database engineer. Generate valid SQL for the given schema."
model_dir = "./qwen_sql_quanto_int8"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    fix_mistral_regex=True,
)

# Ensure a pad token exists for generation APIs.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model
model_kwargs: Dict[str, Any] = {
    "trust_remote_code": True,
    "torch_dtype": compute_dtype,
}
if device.type == "mps":
    # Avoid SDPA(GQA) on MPS if the model class supports this switch.
    model_kwargs["attn_implementation"] = "eager"

try:
    model = QuantizedModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
except TypeError:
    # Some model classes may not accept torch_dtype / attn_implementation.
    model = QuantizedModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

model.to(device)
model.eval()

# Make generation config explicit to avoid warnings.
if getattr(model, "generation_config", None) is not None:
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id


# -----------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------

def generate_sql(schema: str, question: str, max_new_tokens: int = 256) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema: {schema}\nQuestion: {question}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    start_time = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    end_time = time.time()

    gen_tokens = out[0, inputs["input_ids"].shape[-1] :]
    sql = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    print(f"Generation time: {end_time - start_time:.2f} seconds")
    return sql


# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------

def main() -> None:
    examples = [
        (
            "Example 1",
            "CREATE TABLE users(id INT, name TEXT);",
            "List all user names.",
        ),
        (
            "Example 2",
            "[Create table Employees...]",
            "Who is the highest paid employee?",
        ),
    ]

    for title, schema, question in examples:
        print("-" * 30)
        print(f"{title}: {question}")
        print()
        sql = generate_sql(schema=schema, question=question)
        print(sql)
        print("-" * 30)
        print("\n")


if __name__ == "__main__":
    main()
