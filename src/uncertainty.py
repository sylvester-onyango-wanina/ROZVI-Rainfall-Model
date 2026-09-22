"""
uncertainty.py
Lightweight uncertainty helpers for the MVP.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


def add_input_noise(
    arr: np.ndarray,
    sigma: float = 0.05,
    clip_range: tuple[float, float] = (0.0, None),
    random_seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0.0, sigma, size=arr.shape)
    out = arr + noise
    lo, hi = clip_range
    if lo is not None:
        out = np.maximum(out, lo)
    if hi is not None:
        out = np.minimum(out, hi)
    return out


def monte_carlo_depth_uncertainty(
    depth_func,
    n_iterations: int = 50,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Generic wrapper: call depth_func() many times (caller is responsible for
    perturbing inputs inside the function) and return mean / std.
    """
    rng = np.random.default_rng(random_seed)
    samples = []
    for i in range(n_iterations):
        # depth_func should accept a seed or use global RNG
        samples.append(depth_func(seed=int(rng.integers(0, 1_000_000))))
    stack = np.stack(samples, axis=0)
    return {
        "mean": np.nanmean(stack, axis=0),
        "std": np.nanstd(stack, axis=0),
        "p05": np.nanpercentile(stack, 5, axis=0),
        "p95": np.nanpercentile(stack, 95, axis=0),
    }
