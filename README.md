# Zimbabwe Asset-Level Flood Hazard & Risk Model (MVP)

**Scientifically oriented, modular pipeline** that progresses from rainfall extremes through hydrology and indicative inundation to asset-level exposure, vulnerability, risk and plain-language explanation.

> **This is an MVP / research scaffold.**  
> The current inundation method uses a HAND *proxy* and a simplified runoff-to-depth conversion.  
> It is suitable for pipeline testing, architecture demonstration and rapid screening experiments.  
> **It is not yet a production-grade flood model** and must not be used for insurance pricing, engineering design or high-stakes decisions without replacing the proxy modules with calibrated hydrology and hydraulic simulation, real gauge data, and proper validation.

## Key design principles

- Clear separation of rainfall hazard → hydrological response → flood hazard → exposure → vulnerability → risk
- Configuration-driven (no magic numbers in source)
- Provenance metadata on every raster
- Asset-level outputs with unique IDs and human-readable explanations
- Explicit uncertainty / confidence flags
- Synthetic data generator so the full chain can be executed immediately

## Quick start

```bash
cd zimbabwe_flood_risk
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run with synthetic data (no external downloads required)
python main.py --config config/config.yaml --generate-synthetic-data
```

Outputs appear under `outputs/`:

- `rasters/` — indicative flood depth, Rx1day, HAND proxy, runoff
- `assets/asset_risk.gpkg` and `.csv` — per-asset depth, damage fraction, risk score, confidence, explanation
- `reports/validation_snapshot.json`

## Project layout

```
zimbabwe_flood_risk/
├── config/config.yaml          # ALL parameters
├── data/{raw,processed,validation}/
├── src/
│   ├── utils.py                # config, raster I/O, provenance
│   ├── rainfall_indices.py     # ETCCDI indices
│   ├── extremes.py             # GEV / return levels (ready for real AMS)
│   ├── hydrology.py            # slope, SCS-CN, HAND proxy
│   ├── inundation.py           # depth from HAND + runoff
│   ├── exposure.py             # asset load, road segmentation, sampling
│   ├── vulnerability.py        # literature depth-damage curves
│   ├── risk.py                 # risk metrics + 1–10 client score
│   ├── explain.py              # plain-language driver text
│   ├── uncertainty.py
│   ├── validation.py
│   └── generate_synthetic.py
├── main.py
├── requirements.txt
└── tests/
```

## What is deliberately still a proxy / stub

| Component              | Current state                          | Production replacement                          |
|------------------------|----------------------------------------|-------------------------------------------------|
| HAND                   | Local-minimum proxy                    | Full HAND (Whitebox / literature algorithm)     |
| Flow direction / accum | Stub                                   | WhiteboxTools / pysheds / TauDEM                |
| Bias correction        | Not yet wired                          | Quantile mapping + residual kriging vs gauges   |
| Extreme-value analysis | Functions ready, not called in main    | Annual maxima → GEV → return levels + bootstrap |
| Hydraulic modelling    | None                                   | LISFLOOD-FP / HEC-RAS 2D for Tier-3 areas       |
| Gauge network          | Synthetic only                         | GHCN + MSD + ZINWA partnership                  |
| Vulnerability curves   | Generic literature placeholders        | Local damage surveys / insurance curves         |

## Next development steps (recommended order)

1. **Ingest real Zimbabwe rainfall stations** (GHCN/GSOD + request MSD) and implement the bias-correction module.
2. Replace the HAND proxy with a proper implementation and condition the DEM.
3. Wire `extremes.py` into the main pipeline and produce return-period rainfall surfaces.
4. Add a proper national hydrography layer (HydroSHEDS / MERIT Hydro).
5. Introduce a Tier-3 path that can launch a 2-D hydraulic model on selected catchments.
6. Expand the asset schema and attach real replacement values / criticality scores.
7. Full uncertainty cascade and independent flood-extent validation (Sentinel-1).

## Licence & citation

This scaffold is provided for research and development.  
When using any component in a publication or commercial product, cite the underlying scientific sources (ETCCDI, SCS-CN, HAND literature, DEM providers, etc.) and clearly state the limitations of the screening methods.

## Relationship to the earlier rainfall susceptibility model

The previous `rainfall_model` project is retained as the conceptual ancestor for the rainfall-index engine.  
All weighted-overlay susceptibility, AHP and Random-Forest weighting logic has been **removed** from the critical path; those techniques may still be used as optional diagnostics but are no longer presented as flood hazard or risk.
