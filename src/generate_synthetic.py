"""
generate_synthetic.py
Create a small, physically plausible synthetic dataset so the full pipeline
can be executed without external data downloads.
THIS IS NOT REAL OBSERVATIONAL DATA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import geopandas as gpd
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def generate(out_dir: Path, seed: int = 42, size: int = 40, n_days: int = 365 * 3):
    """
    Generate a tiny synthetic AOI (~ size x size pixels at ~1 km) with:
      - DEM (gentle slope + channel)
      - daily precipitation (seasonal + orographic enhancement)
      - a few fake stations
      - a simple road and a few building points
    """
    out_dir = Path(out_dir)
    raw = out_dir / "raw"
    val = out_dir / "validation"
    raw.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    res_m = 1000.0
    transform = from_origin(25.0, -17.0, 0.01, 0.01)  # roughly Zimbabwe-ish
    crs = "EPSG:4326"

    # DEM: higher in the "north", with a central valley
    y = np.linspace(0, 1, size)
    x = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    dem = 1200 - 400 * yy + 80 * np.sin(4 * np.pi * xx) - 150 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.6) ** 2) / 0.05)
    dem += rng.normal(0, 5, dem.shape)

    # Daily precip: wet season + orographic boost on higher ground
    t = np.arange(n_days)
    seasonal = 5 + 15 * np.sin(2 * np.pi * (t % 365) / 365 - 1.5) ** 2
    seasonal = np.clip(seasonal, 0, None)
    orog = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
    daily = np.zeros((n_days, size, size))
    for i in range(n_days):
        base = seasonal[i] * (0.6 + 0.8 * orog)
        noise = rng.exponential(1.0, (size, size)) * (0.3 + 0.7 * (seasonal[i] > 5))
        daily[i] = np.clip(base + noise - 2, 0, 150)

    # Write DEM
    dem_path = raw / "dem.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs=crs, transform=transform, nodata=-9999,
    ) as dst:
        dst.write(dem.astype("float32"), 1)

    # Write daily precip as npy (MVP convenience)
    np.save(raw / "daily_precip.npy", daily.astype("float32"))

    # Study area
    minx, miny = transform * (0, size)
    maxx, maxy = transform * (size, 0)
    gdf = gpd.GeoDataFrame(
        {"name": ["synthetic_aoi"]},
        geometry=[box(minx, miny, maxx, maxy)],
        crs=crs,
    )
    gdf.to_file(raw / "study_area.geojson", driver="GeoJSON")

    # Fake stations
    stations = []
    for i in range(8):
        r, c = rng.integers(2, size - 2, size=2)
        x_pt, y_pt = transform * (c, r)
        stations.append({
            "station_id": f"SYN_{i:03d}",
            "x": x_pt,
            "y": y_pt,
            "source": "station",
            "observed_value": float(np.nanmean(daily[:, r, c]) * 365),
            "observed_event": int(np.nanmax(daily[:, r, c]) > 40),
        })
    pd.DataFrame(stations).to_csv(val / "stations.csv", index=False)

    # Simple buildings (points)
    bld_rows = []
    for i in range(15):
        r, c = rng.integers(1, size - 1, size=2)
        x_pt, y_pt = transform * (c, r)
        bld_rows.append({
            "asset_id": f"BLD_{i:04d}",
            "asset_type": "residential" if i % 3 else "commercial",
            "geometry": gpd.points_from_xy([x_pt], [y_pt])[0],
        })
    bld = gpd.GeoDataFrame(bld_rows, crs=crs)
    bld.to_file(raw / "buildings.geojson", driver="GeoJSON")

    # Simple road (one line across the valley)
    from shapely.geometry import LineString
    road = gpd.GeoDataFrame(
        {"asset_id": ["ROAD_001"], "asset_type": ["road"]},
        geometry=[LineString([(minx + 0.01, (miny + maxy) / 2), (maxx - 0.01, (miny + maxy) / 2)])],
        crs=crs,
    )
    road.to_file(raw / "roads.geojson", driver="GeoJSON")

    meta = {
        "transform": transform,
        "crs": crs,
        "size": size,
        "resolution_m": res_m,
        "n_years": n_days / 365.0,
        "n_days": n_days,
        "note": "SYNTHETIC DATA — NOT REAL OBSERVATIONS",
    }
    logger.info("Synthetic data written to %s (SYNTHETIC — not real)", raw)
    return meta
