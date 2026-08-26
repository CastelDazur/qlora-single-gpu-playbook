"""Decide sequence length from free VRAM before loading anything.

The lesson behind this file: on a 32GB card at 27B the margin is thin, and the
difference between a run that fits and an OOM an hour in is often a few hundred
MiB and the sequence length you picked. So check first, pick a length that
fits, and if nothing fits, say so in a sentence instead of crashing later.

Reads `nvidia-smi`. No torch dependency, so you can call this before importing
the heavy stack.
"""

from __future__ import annotations

import shutil
import subprocess


class VramGateError(RuntimeError):
    """Not enough free VRAM to start, or VRAM could not be read."""


# (min_free_mib, max_len). Checked high to low; first match wins.
# Defaults from a 27B/32GB run: 2048 covered 98.8% of examples, 1536 covered 95.4%.
DEFAULT_GATES: list[tuple[int, int]] = [
    (30000, 2048),
    (28000, 1536),
]
ABORT_BELOW_MIB = 28000


def parse_nvidia_smi_free(output: str) -> list[int]:
    """Parse `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`."""
    vals = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        # Tolerate a stray unit if the caller didn't pass nounits.
        token = line.split()[0].replace(",", "")
        try:
            vals.append(int(token))
        except ValueError:
            continue
    return vals


def free_vram_mib(device: int = 0) -> int:
    """Free VRAM on `device` in MiB. Raises VramGateError if nvidia-smi is missing."""
    if shutil.which("nvidia-smi") is None:
        raise VramGateError("nvidia-smi not found; cannot read free VRAM")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise VramGateError(f"nvidia-smi failed: {exc}") from exc
    vals = parse_nvidia_smi_free(out)
    if device >= len(vals):
        raise VramGateError(f"device {device} not found; nvidia-smi reported {len(vals)} GPU(s)")
    return vals[device]


def choose_max_len(
    free_mib: int,
    gates: list[tuple[int, int]] | None = None,
    abort_below: int = ABORT_BELOW_MIB,
) -> int:
    """Pick a sequence length for the free memory, or raise with a clear message."""
    if free_mib < abort_below:
        raise VramGateError(
            f"only {free_mib} MiB free, need at least {abort_below}. "
            f"Free the GPU (see qlora_kit.preflight.gpu_processes) and retry."
        )
    for min_free, max_len in (gates or DEFAULT_GATES):
        if free_mib >= min_free:
            return max_len
    # Above abort floor but below the lowest gate: use the lowest gate's length.
    return min((g[1] for g in (gates or DEFAULT_GATES)), default=1536)


def gate(device: int = 0, gates: list[tuple[int, int]] | None = None) -> int:
    """Read free VRAM and return a max_len, or raise VramGateError. The one-liner."""
    return choose_max_len(free_vram_mib(device), gates=gates)
