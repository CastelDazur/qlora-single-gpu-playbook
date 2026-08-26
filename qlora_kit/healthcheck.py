"""Fail fast before committing hours.

The idea: run a handful of real steps, look at the loss, and bail if it isn't
moving. A flat loss in the first few steps almost always means something is
wrong (wrong target modules, frozen adapter, a data bug), and it's much cheaper
to catch it now than at hour six.

`loss_trend_ok` is the decision, kept pure so it's testable without a GPU.
`run_healthcheck` is the thin wrapper you'd point at a real trainer.
"""

from __future__ import annotations

from typing import Callable, Sequence


def loss_trend_ok(losses: Sequence[float], min_rel_drop: float = 0.02) -> bool:
    """True if loss dropped by at least `min_rel_drop` (relative) start to end.

    Compares the mean of the first third to the mean of the last third, so one
    noisy step doesn't decide it. Needs at least 3 points.
    """
    pts = [float(x) for x in losses]
    if len(pts) < 3:
        return False
    k = max(1, len(pts) // 3)
    head = sum(pts[:k]) / k
    tail = sum(pts[-k:]) / k
    if head <= 0:
        return tail < head or tail <= 0
    return (head - tail) / head >= min_rel_drop


def run_healthcheck(
    step_fn: Callable[[], float],
    n_steps: int = 6,
    min_rel_drop: float = 0.02,
) -> tuple[bool, list[float]]:
    """Run `step_fn` n_steps times, collecting loss, and judge the trend.

    `step_fn` should run one training step and return its loss. This stays
    trainer-agnostic on purpose; wire it to whatever you use. Returns
    (ok, losses).
    """
    losses = [float(step_fn()) for _ in range(n_steps)]
    return loss_trend_ok(losses, min_rel_drop=min_rel_drop), losses
