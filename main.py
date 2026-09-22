#!/usr/bin/env python3
"""
main.py — Zimbabwe Asset-Level Flood Hazard & Risk Model (MVP orchestrator)

Usage:
    python main.py --config config/config.yaml --generate-synthetic-data
    python main.py --config config/config.yaml

This MVP demonstrates the full scientific chain with synthetic data:
  rainfall indices → extremes (simplified) → HAND proxy → indicative depth
  → asset exposure → vulnerability → risk → explanation

Replace synthetic inputs and proxy methods with real data and full
hydrology/hydraulics before any operational or commercial use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from src.utils import load_config, get_logger, RasterLayer, write_geotiff, make_provenance
from src.generate_synthetic import generate as generate_synthetic
from src import rainfall_indices as ri
from src import hydrology as hydro
from src import inundation as inund
from src import exposure as expo
from src import vulnerability as vuln
from src import risk as risk_mod
from src import explain as explain_mod
from src import validation as val

logger = get_logger("main")


def run(config_path: str, generate_data: bool = False):
    cfg = load_config(config_path)
    paths = cfg["paths"]
    seed = cfg["project"]["random_seed"]
    np.random.seed(seed)

    raw_dir = Path(paths["raw_dir"])
    out_raster = Path(paths["output_raster_dir"])
    out_asset = Path(paths["output_asset_dir"])
    out_report = Path(paths["output_report_dir"])

    # ------------------------------------------------------------------
    # 0. Data
    # ------------------------------------------------------------------
    if generate_data or not (raw_dir / "daily_precip.npy").exists():
        logger.info("Generating SYNTHETIC example dataset (NOT real data)...")
        meta = generate_synthetic(out_dir=Path("data"), seed=seed)
    else:
        import rasterio
        with rasterio.open(raw_dir / "dem.tif") as src:
            meta = {
                "transform": src.transform,
                "crs": str(src.crs),
                "size": src.width,
                "n_years": 3.0,
            }

    transform = meta["transform"]
    crs = meta["crs"]
    n_years = meta.get("n_years", 3.0)

    daily_precip = np.load(raw_dir / "daily_precip.npy")
    dem_layer = __import__("src.utils", fromlist=["read_geotiff"]).read_geotiff(raw_dir / "dem.tif")
    dem = dem_layer.data

    # ------------------------------------------------------------------
    # 1. Rainfall indices
    # ------------------------------------------------------------------
    logger.info("STEP 1 — Rainfall indices (ETCCDI)")
    indices = ri.compute_all_indices(daily_precip, n_years=n_years)

    # ------------------------------------------------------------------
    # 2. Terrain / HAND proxy
    # ------------------------------------------------------------------
    logger.info("STEP 2 — Terrain derivatives & HAND proxy")
    # Approximate cell size in metres (crude for geographic CRS)
    cellsize_m = 1000.0
    slope = hydro.compute_slope_degrees(dem, cellsize_m)
    hand = hydro.compute_hand(dem)
    cn = hydro.default_cn_from_slope_landcover_proxy(slope)

    # ------------------------------------------------------------------
    # 3. Indicative flood depth for a design rainfall (Rx1day scaled)
    # ------------------------------------------------------------------
    logger.info("STEP 3 — Indicative inundation (HAND + runoff proxy)")
    # Use Rx1day as a simple design storm magnitude for the MVP
    design_rain = indices["rx1day"]
    runoff = hydro.scs_cn_runoff(design_rain, cn, amc=cfg["hydrology"]["antecedent_moisture"])
    depth = inund.depth_from_return_period_rainfall(hand, runoff, scaling_factor=0.008)

    depth_layer = inund.build_hazard_layer(
        depth, transform, crs,
        return_period=100,
        scenario="historical_synthetic",
        model_version=cfg["project"]["version"],
    )
    write_geotiff(
        depth_layer,
        out_raster / "flood_depth_indicative_rp100.tif",
        description=depth_layer.description,
    )

    # Also write supporting layers
    for name, arr, desc, units in [
        ("rx1day", indices["rx1day"], "Max 1-day precipitation (mm)", "mm"),
        ("hand_proxy", hand, "HAND proxy (m) — NOT true HAND", "m"),
        ("runoff_mm", runoff, "SCS-CN runoff from Rx1day (mm)", "mm"),
    ]:
        lyr = RasterLayer(arr, transform, crs, name=name, description=desc, units=units)
        write_geotiff(lyr, out_raster / f"{name}.tif", description=desc)

    # ------------------------------------------------------------------
    # 4. Exposure
    # ------------------------------------------------------------------
    logger.info("STEP 4 — Asset exposure")
    assets_list = []

    bld_path = raw_dir / "buildings.geojson"
    if bld_path.exists():
        buildings = expo.load_assets(bld_path, asset_type="building")
        buildings = expo.sample_raster_at_points(depth, transform, buildings, "max_depth_m")
        buildings["hand_m"] = expo.sample_raster_at_points(hand, transform, buildings, "hand_m")["hand_m"]
        assets_list.append(buildings)

    road_path = raw_dir / "roads.geojson"
    if road_path.exists():
        roads = expo.load_assets(road_path, asset_type="road")
        roads_seg = expo.segment_roads(
            roads,
            target_length_m=cfg["exposure"]["road_segment_length_m"],
            min_length_m=cfg["exposure"]["min_segment_length_m"],
        )
        roads_seg = expo.compute_road_flood_metrics(roads_seg, depth, transform)
        roads_seg["hand_m"] = expo.sample_raster_at_points(hand, transform, roads_seg, "hand_m")["hand_m"]
        assets_list.append(roads_seg)

    if not assets_list:
        logger.warning("No asset files found; creating a few synthetic points for demonstration")
        # Fallback already handled by generator
        pass

    assets = gpd.GeoDataFrame(pd.concat(assets_list, ignore_index=True), crs=crs)

    # ------------------------------------------------------------------
    # 5. Vulnerability
    # ------------------------------------------------------------------
    logger.info("STEP 5 — Vulnerability")
    assets = vuln.apply_vulnerability(assets, depth_column="max_depth_m", type_column="asset_type")

    # ------------------------------------------------------------------
    # 6. Risk + confidence + explanation
    # ------------------------------------------------------------------
    logger.info("STEP 6 — Risk, confidence, explanation")
    assets = risk_mod.compute_asset_risk(
        assets, depth_col="max_depth_m", damage_frac_col="damage_fraction",
        return_period=100, scenario="historical_synthetic",
    )
    assets = risk_mod.attach_confidence(
        assets, rainfall_uncertainty=0.15, dem_uncertainty_m=3.0, station_density_ok=False
    )
    assets = explain_mod.add_explanations(assets)

    # Save asset results (CSV first for robustness)
    csv_path = out_asset / "asset_risk.csv"
    assets.drop(columns="geometry", errors="ignore").to_csv(csv_path, index=False)
    logger.info("Wrote asset CSV to %s", csv_path)
    try:
        out_gpkg = out_asset / "asset_risk.gpkg"
        # remove stale journal if present
        for p in out_asset.glob("asset_risk.gpkg*"):
            p.unlink(missing_ok=True)
        assets.to_file(out_gpkg, driver="GPKG")
        logger.info("Wrote asset GeoPackage to %s", out_gpkg)
    except Exception as exc:
        logger.warning("Could not write GeoPackage (%s); CSV is available", exc)

    # ------------------------------------------------------------------
    # 7. Minimal validation against synthetic stations
    # ------------------------------------------------------------------
    logger.info("STEP 7 — Validation snapshot")
    stations = pd.read_csv(cfg["rainfall"]["station_data_path"])
    pts = list(zip(stations["x"], stations["y"]))
    pred_depth = val.extract_at_points(depth, transform, pts)
    # Compare Rx1day at stations vs observed_value (very rough)
    pred_rx1 = val.extract_at_points(indices["rx1day"], transform, pts)
    report = {
        "n_stations": len(stations),
        "rx1day_vs_station_annual": val.continuous_metrics(
            stations["observed_value"].values / 365.0, pred_rx1  # crude daily-scale comparison
        ),
        "note": "Synthetic data only — metrics are illustrative, not scientific validation",
    }
    with open(out_report / "validation_snapshot.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("MVP pipeline complete. Review outputs/ and the explanations in asset_risk.csv")
    return {
        "depth": depth,
        "assets": assets,
        "indices": indices,
        "report": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zimbabwe Flood Risk Model (MVP)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--generate-synthetic-data", action="store_true")
    args = parser.parse_args()
    run(args.config, args.generate_synthetic_data)
