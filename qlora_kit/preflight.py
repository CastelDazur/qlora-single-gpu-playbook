"""Checks to run before an 8-hour job, so it fails in two seconds instead.

Two things that quietly killed runs:
  - PYTORCH_CUDA_ALLOC_CONF not set to expandable_segments:False (allocator
    strangles itself, step time explodes).
  - Something else on the GPU (an inference server, a desktop app) eating the
    couple of GB that decides whether a 27B run fits on a 32GB card.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


REQUIRED_ALLOC = "expandable_segments:False"


def check_alloc_conf() -> tuple[bool, str]:
    """True if PYTORCH_CUDA_ALLOC_CONF disables expandable_segments."""
    val = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    ok = "expandable_segments:false" in val.lower()
    if ok:
        return True, f"PYTORCH_CUDA_ALLOC_CONF={val}"
    return False, (
        f"PYTORCH_CUDA_ALLOC_CONF is '{val or 'unset'}'. "
        f"Set it to '{REQUIRED_ALLOC}' or the allocator can bloat to zero free "
        f"memory and step time explodes (seen: normal -> 485s/step)."
    )


@dataclass
class GpuProc:
    pid: int
    name: str
    used_mib: int


def gpu_processes() -> list[GpuProc]:
    """Processes holding GPU memory, most memory first. Empty list if none/unavailable."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    procs: list[GpuProc] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            procs.append(GpuProc(pid=int(parts[0]), name=parts[1], used_mib=int(parts[2])))
        except ValueError:
            continue
    return sorted(procs, key=lambda p: p.used_mib, reverse=True)


class PreflightError(RuntimeError):
    pass


def preflight(*, require_alloc_conf: bool = True, warn_other_gpu_mib: int = 1000) -> list[str]:
    """Run the checks. Returns warnings; raises PreflightError on a hard stop.

    Hard stop: alloc conf wrong (when required). Warning: other processes
    holding more than `warn_other_gpu_mib` on the GPU.
    """
    warnings: list[str] = []

    ok, msg = check_alloc_conf()
    if not ok:
        if require_alloc_conf:
            raise PreflightError(msg)
        warnings.append(msg)

    for proc in gpu_processes():
        if proc.used_mib >= warn_other_gpu_mib:
            warnings.append(
                f"pid {proc.pid} ({proc.name}) is holding {proc.used_mib} MiB on the GPU; "
                f"free it before a large run"
            )
    return warnings
