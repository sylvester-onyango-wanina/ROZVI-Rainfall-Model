"""
validation.py
Basic continuous and categorical metrics + station sampling.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from rasterio.transform import rowcol

from .utils import get_logger

logger = get_logger(__name__)


def extract_at_points(
    raster: np.ndarray,
    transform,
    points_xy: list[tuple[float, float]],
) -> np.ndarray:
    values = []
    for x, y in points_xy:
        try:
            row, col = rowcol(transform, x, y)
            if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
                values.append(raster[row, col])
            else:
                values.append(np.nan)
        except Exception:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def continuous_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr

    mask = np.isfinite(observed) & np.isfinite(predicted)
    obs, pred = observed[mask], predicted[mask]
    if len(obs) < 3:
        return {"n": int(len(obs)), "error": "insufficient pairs"}

    pearson_r, pearson_p = pearsonr(obs, pred)
    spearman_r, spearman_p = spearmanr(obs, pred)
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    mae = float(np.mean(np.abs(pred - obs)))
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "n": int(len(obs)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": float(np.mean(pred - obs)),
    }


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    mask = np.isfinite(observed) & np.isfinite(predicted)
    obs, pred = observed[mask], predicted[mask]
    if len(obs) < 2:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / (np.std(obs) + 1e-12)
    beta = np.mean(pred) / (np.mean(obs) + 1e-12)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
