"""
rainfall_indices.py
ETCCDI extreme-precipitation indices from a daily (time, y, x) stack.
Adapted and hardened from the original rainfall_model project.
"""
from __future__ import annotations

import numpy as np
from .utils import get_logger, qc_check_array

logger = get_logger(__name__)

WET_DAY_MM = 1.0


def _rolling_sum(arr: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    csum = np.cumsum(arr, axis=axis)
    csum = np.insert(csum, 0, 0, axis=axis)
    slicer_hi = [slice(None)] * arr.ndim
    slicer_lo = [slice(None)] * arr.ndim
    slicer_hi[axis] = slice(window, None)
    slicer_lo[axis] = slice(None, -window)
    return csum[tuple(slicer_hi)] - csum[tuple(slicer_lo)]


def mean_annual_precip(daily_precip: np.ndarray, n_years: float) -> np.ndarray:
    qc_check_array(daily_precip, "daily_precip", valid_range=(0, 2000))
    return np.nansum(daily_precip, axis=0) / n_years


def rx1day(daily_precip: np.ndarray) -> np.ndarray:
    return np.nanmax(daily_precip, axis=0)


def rx5day(daily_precip: np.ndarray) -> np.ndarray:
    if daily_precip.shape[0] < 5:
        raise ValueError("Rx5day requires at least 5 daily time steps.")
    roll5 = _rolling_sum(daily_precip, window=5, axis=0)
    return np.nanmax(roll5, axis=0)


def percentile_threshold_index(
    daily_precip: np.ndarray,
    percentile: float,
    wet_days_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    wet_mask = daily_precip >= WET_DAY_MM if wet_days_only else np.isfinite(daily_precip)
    masked = np.where(wet_mask, daily_precip, np.nan)
    threshold = np.nanpercentile(masked, percentile, axis=0)
    exceed = np.where(daily_precip >= threshold[np.newaxis, ...], daily_precip, 0.0)
    total_exceed = np.nansum(exceed, axis=0)
    return threshold, total_exceed


def r10mm(daily_precip: np.ndarray) -> np.ndarray:
    return np.sum(daily_precip >= 10.0, axis=0).astype("float64")


def r20mm(daily_precip: np.ndarray) -> np.ndarray:
    return np.sum(daily_precip >= 20.0, axis=0).astype("float64")


def _max_consecutive_run(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(int)
    out_shape = mask.shape
    run = np.zeros(out_shape[1:], dtype=int)
    best = np.zeros(out_shape[1:], dtype=int)
    for t in range(out_shape[0]):
        run = np.where(mask[t] == 1, run + 1, 0)
        best = np.maximum(best, run)
    return best.astype("float64")


def cdd(daily_precip: np.ndarray) -> np.ndarray:
    return _max_consecutive_run(daily_precip < WET_DAY_MM)


def cwd(daily_precip: np.ndarray) -> np.ndarray:
    return _max_consecutive_run(daily_precip >= WET_DAY_MM)


def sdii(daily_precip: np.ndarray) -> np.ndarray:
    """Simple Daily Intensity Index: mean precipitation on wet days."""
    wet = daily_precip >= WET_DAY_MM
    total = np.nansum(np.where(wet, daily_precip, 0.0), axis=0)
    count = np.sum(wet, axis=0)
    return np.where(count > 0, total / count, np.nan)


def compute_all_indices(
    daily_precip: np.ndarray,
    n_years: float,
    extreme_percentile: float = 95.0,
) -> dict[str, np.ndarray]:
    logger.info(
        "Computing rainfall indices over %d time steps (%.1f years).",
        daily_precip.shape[0], n_years,
    )
    _, r95p = percentile_threshold_index(daily_precip, 95.0)
    _, r99p = percentile_threshold_index(daily_precip, 99.0)

    return {
        "mean_annual_precip": mean_annual_precip(daily_precip, n_years),
        "rx1day": rx1day(daily_precip),
        "rx5day": rx5day(daily_precip),
        "r95p": r95p,
        "r99p": r99p,
        "r10mm": r10mm(daily_precip),
        "r20mm": r20mm(daily_precip),
        "cdd": cdd(daily_precip),
        "cwd": cwd(daily_precip),
        "sdii": sdii(daily_precip),
    }
