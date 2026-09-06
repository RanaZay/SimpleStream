#!/usr/bin/env python3
"""Run official-style LLaMA-3 judging for StreamBench-v0.3 outputs.

StreamBench evaluates open-ended predictions with an LLM judge.  This runner
uses the same input fields and output fields used by the official protocol:
``question``, ``label``, ``predict`` -> ``llama_pred`` and ``score``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of "
    "generative outputs for question-answer pairs. Your task is to compare the "
    "predicted answer with the correct answer and determine if they are semantically "
    "consistent. The answer may be correct even if wording differs. Reply only "
    "with 'yes' or 'no'."
)


def _build_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    content = (
        f"Question: {row.get('question', '')}\n"
        f"Correct Answer: {row.get('label', '')}\n"
        f"Predicted Answer: {row.get('predict', '')}\n\n"
        "Is the predicted answer semantically correct?"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _parse_score(text: str) -> int:
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    if normalized.startswith("yes") or normalized == "y":
        return 1
    if normalized.startswith("no") or normalized == "n":
        return 0
    if "yes" in normalized.split():
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit > 0:
        rows = rows[: args.limit]

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=dtype,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.device != "auto":
        model.to(args.device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"judge {args.input.name}"):
            messages = _build_prompt(row)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            judge_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            judged = dict(row)
            judged["llama_pred"] = judge_text
            judged["score"] = _parse_score(judge_text)
            f.write(json.dumps(judged, ensure_ascii=False) + "\n")
            f.flush()

    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
