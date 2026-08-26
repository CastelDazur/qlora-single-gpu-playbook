"""Resuming a run without the hang, and repairing a checkpoint after a crash.

Two hard-won pieces:

1. `resume_from_checkpoint` deadlocked at the same step every time on this
   setup (GPU pinned at 100% compute, 0% memory movement). The reliable
   alternative is to reload the adapter as trainable, hand it a fresh optimizer
   and schedule, and set max_steps to only what's left. `remaining_steps` is
   the arithmetic; the reload is shown in examples/train_qlora.py.

2. After a power cut, the saved trainer_state.json can be missing the
   stateful_callbacks block newer Transformers wants, and the next save dies
   with a KeyError. `repair_trainer_state` writes a minimal valid one. The
   adapter weights themselves are fine; it's only the bookkeeping that breaks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def remaining_steps(total: int, done: int) -> int:
    """Steps left to run. The value you pass as max_steps on a re-init resume."""
    if total < 0 or done < 0:
        raise ValueError("total and done must be non-negative")
    return max(0, total - done)


def repair_trainer_state(path: str | Path, *, global_step: int | None = None) -> dict[str, Any]:
    """Ensure a trainer_state.json has the fields a save will look for.

    Adds a minimal `stateful_callbacks.TrainerControl` block if it's missing,
    which is what causes the KeyError-on-save after an unclean stop. Returns the
    repaired dict and writes it back. Does not touch adapter weights.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "trainer_state.json"

    state: dict[str, Any] = {}
    if p.exists():
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}

    if global_step is not None:
        state["global_step"] = global_step
    state.setdefault("global_step", 0)

    callbacks = state.setdefault("stateful_callbacks", {})
    # The block Transformers expects to reload/save cleanly.
    callbacks.setdefault("TrainerControl", {
        "args": {
            "should_training_stop": False,
            "should_epoch_stop": False,
            "should_save": False,
            "should_evaluate": False,
            "should_log": False,
        }
    })

    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


REINIT_RECIPE = """\
# Instead of trainer.train(resume_from_checkpoint=ckpt):
from peft import PeftModel
from qlora_kit.resume import remaining_steps

model = PeftModel.from_pretrained(base_model, ckpt_dir, is_trainable=True)
steps_left = remaining_steps(total=PLANNED_STEPS, done=CHECKPOINT_STEP)

# Build a FRESH optimizer and LR scheduler (do not restore the old ones),
# set training_args.max_steps = steps_left, and call trainer.train() with
# no resume_from_checkpoint. The adapter picks up where it left off; the
# optimizer starts clean, which is what avoids the deadlock.
"""
