"""
exposure.py
Asset ingestion, road segmentation, and sampling of hazard layers onto assets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from shapely.geometry import LineString, Point

from .utils import get_logger

logger = get_logger(__name__)


def load_assets(
    path: Union[str, Path],
    asset_type: str = "building",
    id_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load a vector asset layer and guarantee a unique asset_id."""
    path = Path(path)
    gdf = gpd.read_file(path)
    if id_column and id_column in gdf.columns:
        gdf["asset_id"] = gdf[id_column].astype(str)
    else:
        gdf["asset_id"] = [f"{asset_type}_{i:06d}" for i in range(len(gdf))]
    gdf["asset_type"] = asset_type
    if gdf.crs is None:
        logger.warning("Asset layer has no CRS; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def segment_roads(
    roads: gpd.GeoDataFrame,
    target_length_m: float = 2000.0,
    min_length_m: float = 500.0,
    id_column: str = "asset_id",
) -> gpd.GeoDataFrame:
    """
    Segment LineString roads into approximately target_length_m chainages.
    Preserves parent road ID and generates chainage_id.
    """
    if roads.crs is None or roads.crs.is_geographic:
        roads_m = roads.to_crs("EPSG:3857")
    else:
        roads_m = roads.copy()

    segments = []
    for idx, row in roads_m.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = [geom]

        parent_id = row.get(id_column, f"road_{idx}")
        for line in lines:
            length = line.length
            if length <= target_length_m:
                seg = row.copy()
                seg.geometry = line
                seg["chainage_id"] = f"{parent_id}_000"
                seg["segment_length_m"] = length
                seg["parent_road_id"] = parent_id
                segments.append(seg)
                continue

            n_seg = max(1, int(np.round(length / target_length_m)))
            seg_len = length / n_seg
            if seg_len < min_length_m:
                n_seg = max(1, int(length / min_length_m))
                seg_len = length / n_seg

            distances = np.linspace(0, length, n_seg + 1)
            for i in range(n_seg):
                start = line.interpolate(distances[i])
                end = line.interpolate(distances[i + 1])
                seg_geom = LineString([start, end])
                seg = row.copy()
                seg.geometry = seg_geom
                seg["chainage_id"] = f"{parent_id}_{i:03d}"
                seg["segment_length_m"] = seg_geom.length
                seg["parent_road_id"] = parent_id
                segments.append(seg)

    if not segments:
        return gpd.GeoDataFrame(columns=list(roads.columns) + ["chainage_id", "segment_length_m", "parent_road_id"], crs=roads_m.crs)

    out = gpd.GeoDataFrame(segments, crs=roads_m.crs)
    # return in original CRS
    if roads.crs is not None:
        out = out.to_crs(roads.crs)
    return out


def sample_raster_at_points(
    raster: np.ndarray,
    transform,
    gdf: gpd.GeoDataFrame,
    column_name: str = "value",
) -> gpd.GeoDataFrame:
    """Sample a raster at the centroid (or representative point) of each geometry."""
    gdf = gdf.copy()
    values = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            values.append(np.nan)
            continue
        pt = geom.centroid if geom.geom_type != "Point" else geom
        try:
            row, col = rowcol(transform, pt.x, pt.y)
            if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
                values.append(float(raster[row, col]))
            else:
                values.append(np.nan)
        except Exception:
            values.append(np.nan)
    gdf[column_name] = values
    return gdf


def compute_road_flood_metrics(
    road_segments: gpd.GeoDataFrame,
    depth_raster: np.ndarray,
    transform,
    depth_threshold_m: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    For linear assets: sample depth along the line (simplified to centroid +
    a few points) and compute flooded fraction / max depth.
    """
    gdf = sample_raster_at_points(depth_raster, transform, road_segments, "max_depth_m")
    gdf["flooded"] = gdf["max_depth_m"] >= depth_threshold_m
    # Placeholder for true linear sampling; production code densifies the line
    gdf["flooded_fraction"] = gdf["flooded"].astype(float)
    return gdf
