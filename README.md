# qlora-single-gpu-playbook

Fine-tuning a 27B model with QLoRA on a single 32GB GPU, and the things that went wrong doing it.

This is not another "QLoRA in 10 lines" tutorial. Those work until the run is 8 hours long on one consumer card, and then you meet the failures nobody writes down: the allocator that quietly strangles itself, the resume that hangs forever at the same step, the checkpoint that won't save after a power cut. This repo is the notes I wish I'd had, plus the small helpers that encode them so you don't have to remember.

Everything here is model-agnostic. The numbers are from a 27B run on an RTX 5090 (32GB), but the failures are about the tooling, not the model.

## The short version

- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. With it `True` the allocator bloated to zero free memory and steps went from normal to 485 seconds. This one setting was the difference between a training run and a slideshow.
- Don't use `resume_from_checkpoint`. It hung at the same step every time (GPU at 100% compute, 0% memory movement). Re-initialize the adapter from the checkpoint with a fresh optimizer instead. `qlora_kit/resume.py` does the math.
- Gate on free VRAM before you start, not after the OOM. `qlora_kit/vram.py` reads free memory and picks a sequence length, or aborts with a message instead of crashing an hour in.
- Kill everything else on the GPU first. A background inference server or a desktop app holding 2GB is enough to tip a run that fit in a dry run into OOM.
- Watch `progress.jsonl`, not the training log. The log lies about where you are after a resume.

## What's in here

```
qlora_kit/
  vram.py         free-VRAM query + a gate that picks max_len or aborts
  resume.py       remaining-steps math + the re-init pattern that replaces resume_from_checkpoint
  preflight.py    environment checks (alloc conf, stray GPU processes)
  healthcheck.py  a short dry run that fails fast if loss isn't moving
examples/
  train_qlora.py  a reference training script with all of the above wired in
```

The helpers are plain functions with no heavy deps, so you can lift `vram.py` or `resume.py` into your own trainer without adopting the whole thing.

## The failures, in detail

### The allocator strangles itself

Symptom: free VRAM slowly falls to near zero over the first few hundred steps, step time climbs from normal to 400+ seconds, no OOM, just crawling.

Cause: `expandable_segments:True` fragments and holds memory it won't give back under this workload.

Fix:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```
`qlora_kit/preflight.py` checks this is set before you start and refuses to run if it isn't, because finding out 300 steps later is expensive.

### resume_from_checkpoint hangs

Symptom: you restart after a pause, and the run sits at one step forever. `nvidia-smi` shows the GPU pinned at 100% compute with no memory activity.

Cause: for this setup `resume_from_checkpoint` reliably deadlocks. I never found a config that fixed it.

Fix: reload the adapter as a trainable model, give it a fresh optimizer and learning-rate schedule, and set `max_steps` to only the remaining steps. Take it out of `trainer.train()` entirely.
```python
from qlora_kit.resume import remaining_steps
# done=400 of a planned 742
steps_left = remaining_steps(total=742, done=400)   # -> 342
# load PeftModel.from_pretrained(ckpt, is_trainable=True), fresh optimizer, max_steps=steps_left
```
`examples/train_qlora.py` shows the full reload.

### Checkpoint won't save after a crash

Symptom: after a power cut you try to resume from the last checkpoint and get a `KeyError` on save.

Cause: the saved `trainer_state.json` is missing the `stateful_callbacks.TrainerControl` block newer Transformers expects.

Fix: synthesize a minimal valid `trainer_state.json`. You can drop the optimizer state, the adapter weights (`adapter_model.safetensors`) are intact and are what you actually need. `qlora_kit/resume.py` has a `repair_trainer_state()` helper.

### It fit in the dry run, then OOM'd

Symptom: a short test run fits comfortably, the real run OOMs early.

Cause: something else grew on the GPU. A background inference server waking up, a desktop app, a previous run's leaked process.

Fix: `qlora_kit/preflight.py` lists non-Python processes holding GPU memory so you can clear them. On a 32GB card the margin at 27B is thin enough that 1-2GB elsewhere decides the run.

## VRAM gate

The single most useful habit: decide sequence length from free memory before loading anything.

```python
from qlora_kit.vram import free_vram_mib, choose_max_len

free = free_vram_mib()
max_len = choose_max_len(free)      # >=30000 -> 2048, 28000-29999 -> 1536, else abort
```

The defaults come from the 27B/32GB run (2048 covered 98.8% of examples, 1536 covered 95.4%). Override the thresholds for your model. The point is to fail with a sentence, not a stack trace.

## Health check

Before committing 8 hours, run a short dry pass. `qlora_kit/healthcheck.py` runs a handful of steps and checks that loss is actually moving down and VRAM sits under your ceiling, then exits. If loss is flat or memory is already near the edge, better to know in two minutes.

## What this is not

- Not a trainer. It's a set of guards around whatever trainer you use (this was built around Transformers + PEFT + bitsandbytes).
- Not tuned advice on ranks and learning rates. Those depend on your data. This is about surviving the run mechanically.
- Not tested on multi-GPU. All of this is single-card experience.

## License

MIT.

Notes from building [CastelOS](https://github.com/CastelDazur/castelos-public). Pairs well with [gpu-memory-guard](https://github.com/CastelDazur/gpu-memory-guard) for the pre-load VRAM check.
