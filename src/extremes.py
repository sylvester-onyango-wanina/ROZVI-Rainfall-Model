"""
extremes.py
Extreme-value analysis for rainfall: annual-maximum series, GEV fitting,
return-level estimation with uncertainty.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from .utils import get_logger

logger = get_logger(__name__)


def annual_maxima(daily_precip: np.ndarray, year_indices: np.ndarray) -> np.ndarray:
    """
    Compute annual maximum series from a daily (time, y, x) stack.

    Parameters
    ----------
    daily_precip : array (T, Y, X)
    year_indices : array (T,) integer year labels for each time step

    Returns
    -------
    ams : array (n_years, Y, X)
    """
    years = np.unique(year_indices)
    ams = []
    for y in years:
        mask = year_indices == y
        if mask.sum() < 30:  # require reasonable coverage
            continue
        ams.append(np.nanmax(daily_precip[mask], axis=0))
    if not ams:
        raise ValueError("No years with sufficient data for annual maxima.")
    return np.stack(ams, axis=0)


def fit_gev_per_pixel(
    ams: np.ndarray,
    min_years: int = 20,
) -> dict[str, np.ndarray]:
    """
    Fit GEV (location, scale, shape) to the annual-maximum series at each pixel.
    Returns parameter arrays and a validity mask.
    """
    n_years, ny, nx = ams.shape
    shape = np.full((ny, nx), np.nan)
    loc = np.full((ny, nx), np.nan)
    scale = np.full((ny, nx), np.nan)
    valid = np.zeros((ny, nx), dtype=bool)

    for i in range(ny):
        for j in range(nx):
            series = ams[:, i, j]
            series = series[np.isfinite(series)]
            if len(series) < min_years:
                continue
            try:
                # scipy: c=shape, loc, scale (c>0 → Weibull-type for maxima)
                c, loc_ij, scale_ij = stats.genextreme.fit(series)
                shape[i, j] = c
                loc[i, j] = loc_ij
                scale[i, j] = scale_ij
                valid[i, j] = True
            except Exception:
                continue
    return {"shape": shape, "loc": loc, "scale": scale, "valid": valid}


def return_level(
    shape: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
    return_period_years: float,
) -> np.ndarray:
    """
    GEV return level for a given return period (years).
    """
    # Probability of exceedance
    p = 1.0 / return_period_years
    # scipy genextreme.ppf uses the same parameterisation
    rl = np.full_like(loc, np.nan)
    mask = np.isfinite(shape) & np.isfinite(loc) & np.isfinite(scale) & (scale > 0)
    rl[mask] = stats.genextreme.ppf(1 - p, shape[mask], loc=loc[mask], scale=scale[mask])
    return rl


def return_levels_for_periods(
    gev_params: dict[str, np.ndarray],
    periods: list[float],
) -> dict[float, np.ndarray]:
    out = {}
    for rp in periods:
        out[rp] = return_level(
            gev_params["shape"], gev_params["loc"], gev_params["scale"], rp
        )
    return out


def bootstrap_return_level_uncertainty(
    ams: np.ndarray,
    return_period: float,
    n_boot: int = 100,
    min_years: int = 20,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Very simple pixel-wise bootstrap of return levels (computationally heavy;
    intended for small AOIs or selected locations in production).
    Returns (mean, std) of bootstrap return levels.
    """
    rng = np.random.default_rng(random_seed)
    n_years, ny, nx = ams.shape
    means = np.full((ny, nx), np.nan)
    stds = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            series = ams[:, i, j]
            series = series[np.isfinite(series)]
            if len(series) < min_years:
                continue
            boots = []
            for _ in range(n_boot):
                sample = rng.choice(series, size=len(series), replace=True)
                try:
                    c, loc, scale = stats.genextreme.fit(sample)
                    rl = stats.genextreme.ppf(
                        1 - 1 / return_period, c, loc=loc, scale=scale
                    )
                    boots.append(rl)
                except Exception:
                    continue
            if boots:
                means[i, j] = np.mean(boots)
                stds[i, j] = np.std(boots)
    return means, stds
