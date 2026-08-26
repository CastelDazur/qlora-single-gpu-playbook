"""qlora_kit: guards and helpers for QLoRA on a single GPU.

The helpers are plain functions with no heavy dependencies. Import what you
need into your own trainer; you do not have to adopt the whole thing.
"""

from qlora_kit.vram import free_vram_mib, choose_max_len, VramGateError
from qlora_kit.resume import remaining_steps, repair_trainer_state
from qlora_kit.preflight import check_alloc_conf, gpu_processes, preflight
from qlora_kit.healthcheck import loss_trend_ok

__version__ = "0.1.0"
__all__ = [
    "free_vram_mib", "choose_max_len", "VramGateError",
    "remaining_steps", "repair_trainer_state",
    "check_alloc_conf", "gpu_processes", "preflight",
    "loss_trend_ok",
]
