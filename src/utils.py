"""
utils.py — Configuration, logging, raster I/O, QC helpers, provenance.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.io import DatasetReader
import rasterio

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Ensure output directories exist
    for key in ("output_raster_dir", "output_stats_dir", "output_fig_dir",
                "output_report_dir", "output_asset_dir", "log_dir",
                "processed_dir"):
        p = cfg.get("paths", {}).get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# RasterLayer dataclass-like container
# ---------------------------------------------------------------------------
class RasterLayer:
    """Lightweight container for a 2-D array + geospatial metadata + provenance."""

    def __init__(
        self,
        data: np.ndarray,
        transform: Affine,
        crs: Union[str, CRS],
        nodata: float = -9999.0,
        name: str = "",
        description: str = "",
        units: str = "",
        provenance: Optional[dict] = None,
    ):
        self.data = np.asarray(data, dtype="float64")
        self.transform = transform
        self.crs = CRS.from_user_input(crs) if not isinstance(crs, CRS) else crs
        self.nodata = nodata
        self.name = name
        self.description = description
        self.units = units
        self.provenance = provenance or {}
        self.provenance.setdefault("created_utc", datetime.now(timezone.utc).isoformat())

    @property
    def shape(self):
        return self.data.shape

    def mask_nodata(self) -> np.ndarray:
        return np.where(
            (self.data == self.nodata) | ~np.isfinite(self.data),
            np.nan,
            self.data,
        )


def read_geotiff(path: Union[str, Path]) -> RasterLayer:
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read(1).astype("float64")
        nodata = src.nodata if src.nodata is not None else -9999.0
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        return RasterLayer(
            data=data,
            transform=src.transform,
            crs=src.crs,
            nodata=nodata if nodata is not None else -9999.0,
            name=path.stem,
        )


def write_geotiff(
    layer: RasterLayer,
    path: Union[str, Path],
    description: str = "",
    compress: str = "lzw",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": layer.data.shape[0],
        "width": layer.data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": layer.crs,
        "transform": layer.transform,
        "nodata": layer.nodata,
        "compress": compress,
    }
    data = layer.data.astype("float32")
    data = np.where(np.isfinite(data), data, layer.nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            description=description or layer.description,
            units=layer.units,
            name=layer.name,
            provenance=json.dumps(layer.provenance),
        )
    logger.info("Wrote %s", path)
    return path


def qc_check_array(
    arr: np.ndarray,
    name: str,
    valid_range: Optional[tuple[float, float]] = None,
    allow_nan: bool = True,
) -> None:
    """Basic physical sanity checks."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        logger.warning("QC: %s contains no finite values", name)
        return
    if valid_range is not None:
        lo, hi = valid_range
        if finite.min() < lo or finite.max() > hi:
            logger.warning(
                "QC: %s values outside expected range [%.3g, %.3g] "
                "(observed [%.3g, %.3g])",
                name, lo, hi, finite.min(), finite.max(),
            )


def ensure_same_grid(
    layers: dict[str, RasterLayer],
    reference: Optional[RasterLayer] = None,
) -> dict[str, RasterLayer]:
    """Placeholder for reproject/resample/align. Real implementation uses
    rasterio.warp.reproject. For MVP we assume inputs are already aligned."""
    if reference is None:
        reference = next(iter(layers.values()))
    for name, lyr in layers.items():
        if lyr.shape != reference.shape:
            raise ValueError(
                f"Layer '{name}' shape {lyr.shape} does not match reference "
                f"{reference.shape}. Run preprocessing.reproject_resample_align first."
            )
    return layers


def make_provenance(
    source_datasets: list[str],
    processing_steps: list[str],
    model_version: str,
    config_hash: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    prov = {
        "source_datasets": source_datasets,
        "processing_steps": processing_steps,
        "model_version": model_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if config_hash:
        prov["config_hash"] = config_hash
    if extra:
        prov.update(extra)
    return prov
