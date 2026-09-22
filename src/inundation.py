"""
inundation.py
Convert hydrological quantities into flood extent / depth proxies.
Primary national method: HAND-based depth estimation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import RasterLayer, get_logger, make_provenance

logger = get_logger(__name__)


def hand_to_depth(
    hand: np.ndarray,
    water_level_m: float,
) -> np.ndarray:
    """
    Simple HAND inundation: depth = max(0, water_level - HAND).
    water_level_m is the assumed water-surface elevation relative to the
    drainage network (can be derived from a rating curve or a design storm).
    """
    depth = water_level_m - hand
    return np.clip(depth, 0, None)


def depth_from_return_period_rainfall(
    hand: np.ndarray,
    runoff_mm: np.ndarray,
    scaling_factor: float = 0.001,
) -> np.ndarray:
    """
    Extremely simplified conversion of runoff volume to an indicative depth
    modulated by HAND.  This is a SCREENING proxy only.

    Production systems replace this with:
      - volume redistribution along the drainage network, or
      - 1D/2D hydraulic simulation (LISFLOOD-FP, HEC-RAS 2D, etc.)
    """
    # Higher runoff and lower HAND → higher indicative depth
    # scaling_factor converts mm runoff into a rough depth contribution
    indicative = runoff_mm * scaling_factor * np.exp(-hand / 5.0)
    return np.clip(indicative, 0, None)


def classify_flood_depth(
    depth: np.ndarray,
    breaks_m: list[float] = (0.1, 0.5, 1.0, 2.0),
) -> np.ndarray:
    """Classify continuous depth into ordinal hazard classes (0 = dry)."""
    classes = np.zeros_like(depth, dtype="int16")
    for i, b in enumerate(sorted(breaks_m), start=1):
        classes = np.where(depth >= b, i, classes)
    classes = np.where(np.isfinite(depth), classes, -1)
    return classes


def build_hazard_layer(
    depth: np.ndarray,
    transform,
    crs,
    return_period: float,
    scenario: str,
    peril: str = "pluvial_fluvial_combined",
    model_version: str = "0.1.0-mvp",
) -> RasterLayer:
    prov = make_provenance(
        source_datasets=["DEM", "rainfall_extremes", "HAND"],
        processing_steps=["hand_proxy", "runoff_to_depth_proxy"],
        model_version=model_version,
        extra={
            "return_period_years": return_period,
            "scenario": scenario,
            "peril": peril,
            "note": "Screening-level depth proxy; replace with hydraulic model for asset decisions",
        },
    )
    return RasterLayer(
        data=depth,
        transform=transform,
        crs=crs,
        name=f"flood_depth_rp{int(return_period)}_{scenario}",
        description=f"Indicative flood depth (m) for RP={return_period} yr, scenario={scenario}",
        units="m",
        provenance=prov,
    )
