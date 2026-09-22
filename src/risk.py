"""
risk.py
Combine hazard, exposure and vulnerability into asset-level risk metrics
and optional client-facing 1–10 scores that remain traceable to physical quantities.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def compute_asset_risk(
    assets: pd.DataFrame,
    depth_col: str = "max_depth_m",
    damage_frac_col: str = "damage_fraction",
    return_period: float = 100.0,
    scenario: str = "historical",
) -> pd.DataFrame:
    """
    Minimal risk table.  In a full implementation this would also carry
    probability, duration, velocity, and multi-peril aggregation.
    """
    df = assets.copy()
    df["return_period_years"] = return_period
    df["scenario"] = scenario
    df["hazard_depth_m"] = df[depth_col]
    df["vulnerability"] = df[damage_frac_col]

    # Simple relative risk score (0–1) = damage fraction for the chosen scenario.
    # This is deliberately transparent; financial AAL would require a full
    # event-set or integration across the frequency curve.
    df["relative_risk"] = df["vulnerability"].clip(0, 1)

    # Optional client-facing 1–10 band (calibrated, not arbitrary weights)
    df["client_score_1_10"] = _relative_to_client_score(df["relative_risk"])

    return df


def _relative_to_client_score(relative: pd.Series) -> pd.Series:
    """
    Map continuous relative risk [0,1] onto integer 1–10 bands.
    Thresholds should be reviewed with domain experts and validated
    against historical loss experience when available.
    """
    bins = [-0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90, 1.01]
    labels = list(range(1, 11))
    return pd.cut(relative, bins=bins, labels=labels, include_lowest=True).astype(float)


def attach_confidence(
    assets: pd.DataFrame,
    rainfall_uncertainty: Optional[float] = None,
    dem_uncertainty_m: Optional[float] = None,
    station_density_ok: bool = True,
) -> pd.DataFrame:
    """
    Attach a simple confidence label.  Production systems propagate
    full uncertainty distributions.
    """
    df = assets.copy()
    confidence = np.full(len(df), "MEDIUM", dtype=object)

    if not station_density_ok:
        confidence[:] = "LOW"
    if dem_uncertainty_m is not None and dem_uncertainty_m > 5.0:
        confidence[:] = "LOW"
    if rainfall_uncertainty is not None and rainfall_uncertainty > 0.3:
        confidence = np.where(confidence == "LOW", "LOW", "MEDIUM")

    # Local depth-based adjustment: very high depths from a screening model
    # should be flagged as higher uncertainty
    if "hazard_depth_m" in df.columns:
        high_depth = df["hazard_depth_m"].fillna(0) > 3.0
        confidence = np.where(high_depth & (confidence != "LOW"), "MEDIUM", confidence)

    df["confidence"] = confidence
    return df
