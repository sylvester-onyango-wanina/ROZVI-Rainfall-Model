"""
hydrology.py
Terrain derivatives, flow accumulation, simple SCS-CN runoff, HAND.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from rasterio.transform import xy

from .utils import RasterLayer, get_logger, qc_check_array

logger = get_logger(__name__)


def compute_slope_degrees(dem: np.ndarray, cellsize_m: float) -> np.ndarray:
    """Simple finite-difference slope in degrees."""
    dy, dx = np.gradient(dem, cellsize_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)


def compute_flow_direction_d8(dem: np.ndarray) -> np.ndarray:
    """
    Very basic D8 flow direction (encoded as 1-8 or 0 for sinks).
    Production code should use WhiteboxTools / pysheds / TauDEM for
    depression filling and proper routing.
    """
    # Placeholder: return zeros; real implementation required for production
    logger.warning(
        "compute_flow_direction_d8 is a stub. "
        "Use WhiteboxTools or pysheds for production hydrology."
    )
    return np.zeros_like(dem, dtype="uint8")


def compute_hand(
    dem: np.ndarray,
    flow_accumulation: Optional[np.ndarray] = None,
    stream_threshold: float = 1000.0,
    transform=None,
) -> np.ndarray:
    """
    Height Above Nearest Drainage (HAND) — simplified implementation.

    True HAND requires:
      1. Depression-filled DEM
      2. Flow direction
      3. Stream network (accumulation > threshold)
      4. For every cell, elevation difference to the stream cell it drains to

    This MVP version returns a relative-elevation proxy suitable for testing
    the downstream pipeline. Replace with a robust HAND implementation
    (e.g. via WhiteboxTools or the HAND algorithms in literature) before
    operational use.
    """
    logger.warning(
        "compute_hand is a simplified proxy. "
        "Replace with a full HAND implementation (Whitebox / literature) "
        "for scientifically defensible floodplain mapping."
    )
    # Simple relative elevation from local minimum in a moving window
    # (NOT true HAND — only for pipeline testing)
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(dem, size=15)
    hand_proxy = dem - local_min
    hand_proxy = np.clip(hand_proxy, 0, None)
    return hand_proxy


def scs_cn_runoff(
    precip_mm: np.ndarray,
    cn: np.ndarray,
    amc: str = "II",
) -> np.ndarray:
    """
    SCS / NRCS Curve Number runoff depth (mm).

    Q = (P - Ia)^2 / (P - Ia + S)   for P > Ia
    S = 25400/CN - 254
    Ia = 0.2 * S   (standard assumption)
    """
    cn = np.clip(cn, 1, 100)
    S = 25400.0 / cn - 254.0
    Ia = 0.2 * S

    # Simple AMC adjustment (very approximate)
    if amc.upper() == "I":
        cn = cn / (2.281 - 0.01281 * cn)
    elif amc.upper() == "III":
        cn = cn / (0.427 + 0.00573 * cn)
    cn = np.clip(cn, 1, 100)
    S = 25400.0 / cn - 254.0
    Ia = 0.2 * S

    Q = np.where(
        precip_mm > Ia,
        (precip_mm - Ia) ** 2 / (precip_mm - Ia + S),
        0.0,
    )
    return np.clip(Q, 0, None)


def default_cn_from_slope_landcover_proxy(
    slope_deg: np.ndarray,
    built_up_fraction: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fallback CN map when proper soil + land-cover layers are unavailable.
    Higher slope and higher built-up → higher CN.
    This is a pragmatic placeholder, not a substitute for SoilGrids + WorldCover.
    """
    base = 70.0
    slope_term = np.clip(slope_deg / 30.0, 0, 1) * 15.0
    urban_term = 0.0
    if built_up_fraction is not None:
        urban_term = np.clip(built_up_fraction, 0, 1) * 20.0
    cn = base + slope_term + urban_term
    return np.clip(cn, 40, 98)
