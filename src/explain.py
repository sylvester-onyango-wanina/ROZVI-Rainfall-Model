"""
explain.py
Generate plain-language, variable-driven explanations for each asset.
Every sentence must be derivable from actual model columns.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def explain_asset(row: pd.Series) -> str:
    """
    Build a short, non-technical explanation from the columns present on the row.
    """
    parts = []

    asset_id = row.get("asset_id", row.get("chainage_id", "unknown"))
    asset_type = str(row.get("asset_type", "asset")).lower()
    depth = row.get("hazard_depth_m", row.get("max_depth_m"))
    rp = row.get("return_period_years")
    scenario = row.get("scenario", "selected")
    score = row.get("client_score_1_10")
    conf = row.get("confidence", "unknown")

    # Opening
    if pd.notna(depth) and depth > 0.05:
        parts.append(
            f"This {asset_type} (ID {asset_id}) is estimated to experience "
            f"indicative flooding of approximately {depth:.2f} m"
        )
        if pd.notna(rp):
            parts[-1] += f" under the 1-in-{int(rp)}-year rainfall scenario"
        parts[-1] += f" ({scenario} climate)."
    else:
        parts.append(
            f"This {asset_type} (ID {asset_id}) shows negligible modelled flood depth "
            f"under the selected scenario ({scenario})."
        )

    # Drivers (only if columns exist)
    if "hand_m" in row and pd.notna(row["hand_m"]):
        parts.append(
            f"It lies approximately {row['hand_m']:.1f} m above the nearest drainage network."
        )
    if "upstream_area_km2" in row and pd.notna(row["upstream_area_km2"]):
        parts.append(
            f"Upstream contributing area is roughly {row['upstream_area_km2']:.1f} km²."
        )
    if "distance_to_river_m" in row and pd.notna(row["distance_to_river_m"]):
        parts.append(
            f"Distance to the nearest mapped channel is about {row['distance_to_river_m']:.0f} m."
        )

    if pd.notna(score):
        parts.append(f"Client-facing risk score: {int(score)}/10 (confidence: {conf}).")

    parts.append(
        "This is a screening-level estimate derived from rainfall extremes, "
        "terrain (HAND proxy), and literature vulnerability curves. "
        "It is not a substitute for detailed hydraulic modelling or site survey."
    )

    return " ".join(parts)


def add_explanations(assets: pd.DataFrame) -> pd.DataFrame:
    df = assets.copy()
    df["explanation"] = df.apply(explain_asset, axis=1)
    return df
