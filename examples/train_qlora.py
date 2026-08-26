"""Reference QLoRA training script with the guards wired in.

This is not runnable as-is: you need a GPU, a base model, and a JSONL dataset.
It's here to show where each helper goes in a real run. The parts that matter
are marked. Ranks, LR, and data loading are yours to fill in.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    python examples/train_qlora.py --base <model> --data train.jsonl --out ./out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qlora_kit.preflight import preflight, PreflightError
from qlora_kit.vram import free_vram_mib, choose_max_len, VramGateError
from qlora_kit.resume import remaining_steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model id or path")
    ap.add_argument("--data", required=True, help="training JSONL")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--planned-steps", type=int, default=1000)
    ap.add_argument("--resume-from", help="checkpoint dir to re-init from")
    ap.add_argument("--resume-done", type=int, default=0, help="steps already completed")
    args = ap.parse_args()

    # 1. Preflight. Refuse to start if the allocator setting is wrong; warn on
    #    other GPU tenants. Two seconds now saves an hour later.
    try:
        for w in preflight(require_alloc_conf=True):
            print(f"[warn] {w}")
    except PreflightError as exc:
        print(f"[stop] {exc}")
        return 2

    # 2. VRAM gate. Pick sequence length from what's actually free, or abort
    #    with a sentence instead of OOM'ing an hour in.
    try:
        free = free_vram_mib()
        max_len = choose_max_len(free)
        print(f"[vram] {free} MiB free -> max_len {max_len}")
    except VramGateError as exc:
        print(f"[stop] {exc}")
        return 2

    # ---- heavy imports after the guards pass ----
    import torch  # noqa: F401
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(args.base)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="auto")
    base = prepare_model_for_kbit_training(base)

    # 3. Resume path: re-init from checkpoint, do NOT use resume_from_checkpoint.
    if args.resume_from:
        model = PeftModel.from_pretrained(base, args.resume_from, is_trainable=True)
        max_steps = remaining_steps(total=args.planned_steps, done=args.resume_done)
        print(f"[resume] re-init from {args.resume_from}, {max_steps} steps left")
    else:
        lora = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base, lora)
        max_steps = args.planned_steps

    # ... build your dataset with tokenizer truncation=max_len, then:
    targs = TrainingArguments(
        output_dir=args.out, max_steps=max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        optim="paged_adamw_8bit", save_steps=100, logging_steps=1, bf16=True,
    )

    # 4. Log to progress.jsonl, not just the training log. After a resume the
    #    log's step numbers get confusing; a jsonl you append is the truth.
    progress = Path(args.out) / "progress.jsonl"
    progress.parent.mkdir(parents=True, exist_ok=True)

    class ProgressCb:
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                with open(progress, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": state.global_step, "loss": logs["loss"]}) + "\n")

    # trainer = Trainer(model=model, args=targs, train_dataset=..., callbacks=[ProgressCb()])
    # trainer.train()   # no resume_from_checkpoint, ever
    print("[note] fill in dataset + Trainer, then train()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
