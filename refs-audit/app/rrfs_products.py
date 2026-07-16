"""RRFS operational product registry.

Registers RRFS's own deterministic output as its own full model ("rrfs" in
the Model dropdown, mirroring REPS's "own model" pattern rather than
folding into REFS/HREF) -- see app/rrfs_data.py's module docstring for why
REFS/HREF's own ensemble-post files can't supply most of these fields at
all (they're either missing entirely, like smoke/dust/AOD, or computed
differently, like REFS's own ensemble CAPE/UH/shear).

Phase 1 (this file) covers Severe + Storm Attributes + Fire + the existing
Air Quality products -- Synoptic (upper-air/prslev), Satellite, and
Precipitation are an explicit, deferred follow-up.

Every product here uses ONE of two recipes:
  - "rrfs_field": generic single-record pull. The idx substring is built
    from `rrfs_idx_tmpl` (a template with `{fhr}`/`{fhrm1}` placeholders)
    at render time -- see refs_core.py's PlotJob._rrfs_field. This covers
    everything, including the 4 original Air Quality products (previously
    each had its own Python loader function; now they're just template
    strings like every other product here).
  - "rrfs_shear": bulk shear magnitude, needs 2 records (U/V components)
    combined -- see PlotJob._rrfs_shear / rrfs_fields.load_rrfs_shear.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import refs_core as core  # noqa: E402
from .rrfs_fields import aqi_from_pm25  # noqa: E402

_KGM3_TO_UGM3 = lambda x: x * 1e9   # noqa: E731 -- smoke/PM2.5 mass density
_KGM2_TO_MGM2 = lambda x: x * 1e6   # noqa: E731 -- column-integrated smoke mass
_M_TO_KFT = lambda x: x * 0.00328084   # noqa: E731 -- echo top, m -> kft
_MS_TO_KT = lambda x: x * 1.94384   # noqa: E731 -- shear/wind, m/s -> kt
_PM25_KGM3_TO_AQI = lambda x: aqi_from_pm25(x * 1e9)   # noqa: E731

RRFS_FHR_STEP = 1

PRODUCTS = {
    # ---- Air Quality (existing 4, migrated from the old "rrfs_aq" source
    # onto this model; same fields, same idx substrings, same conversions).
    "rrfs_smoke_sfc": dict(
        cat="Air Quality", name="Surface Smoke", recipe="rrfs_field",
        rrfs_field_key="smoke_sfc",
        rrfs_idx_tmpl=("MASSDEN:8 m above ground:{fhr} hour fcst:"
                        "aerosol=Particulate organic matter dry:aerosol_size <2.5e-06"),
        cmap="smoke_sfc", units="ug/m3", convert=_KGM3_TO_UGM3,
        spc_title="RRFS near-surface smoke (PM2.5) — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_vi_smoke": dict(
        cat="Air Quality", name="Column Smoke (VI)", recipe="rrfs_field",
        rrfs_field_key="vi_smoke",
        rrfs_idx_tmpl=("COLMD:entire atmosphere (considered as a single layer):{fhr} hour fcst:"
                        "aerosol=Particulate organic matter dry:aerosol_size <2.5e-06"),
        cmap="smoke_vi", units="mg/m2", convert=_KGM2_TO_MGM2,
        spc_title="RRFS column-integrated smoke (PM2.5) — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_aod": dict(
        cat="Air Quality", name="Aerosol Optical Depth", recipe="rrfs_field",
        rrfs_field_key="aod",
        rrfs_idx_tmpl="AOTK:entire atmosphere (considered as a single layer):{fhr} hour fcst:",
        cmap="aod", units="",
        spc_title="RRFS total aerosol optical depth — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_aqi": dict(
        cat="Air Quality", name="Air Quality Index (PM2.5)", recipe="rrfs_field",
        rrfs_field_key="pm25_total_mean",
        rrfs_idx_tmpl=("MASSDEN:8 m above ground:{fhrm1}-{fhr} hour ave fcst:"
                        "aerosol=Total aerosol:aerosol_size <2.5e-06"),
        cmap="aqi_epa", units="AQI", convert=_PM25_KGM3_TO_AQI,
        cbar_tick_positions=[25, 75, 125, 175, 250, 400],
        cbar_tick_labels=["Good", "Moderate", "USG", "Unhealthy", "V. Unhealthy", "Hazardous"],
        spc_title="RRFS Air Quality Index (EPA PM2.5 breakpoints)",
        fhr_stride=RRFS_FHR_STEP, min_fhr=1, source="rrfs",
    ),

    # ---- Severe -------------------------------------------------------
    "rrfs_cape_sfc": dict(
        cat="RRFS Severe", name="Surface-Based CAPE", recipe="rrfs_field",
        rrfs_field_key="cape_sfc", rrfs_idx_tmpl="CAPE:surface:{fhr} hour fcst:",
        cmap="cape", units="J/kg",
        spc_title="RRFS surface-based CAPE — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_cin_sfc": dict(
        cat="RRFS Severe", name="Surface-Based CIN", recipe="rrfs_field",
        rrfs_field_key="cin_sfc", rrfs_idx_tmpl="CIN:surface:{fhr} hour fcst:",
        cmap="cin", units="J/kg",
        spc_title="RRFS surface-based CIN — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_cape_ml": dict(
        cat="RRFS Severe", name="Mixed-Layer CAPE", recipe="rrfs_field",
        rrfs_field_key="cape_ml", rrfs_idx_tmpl="CAPE:90-0 mb above ground:{fhr} hour fcst:",
        cmap="cape", units="J/kg",
        spc_title="RRFS 90mb mixed-layer CAPE — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_cin_ml": dict(
        cat="RRFS Severe", name="Mixed-Layer CIN", recipe="rrfs_field",
        rrfs_field_key="cin_ml", rrfs_idx_tmpl="CIN:90-0 mb above ground:{fhr} hour fcst:",
        cmap="cin", units="J/kg",
        spc_title="RRFS 90mb mixed-layer CIN — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_cape_mu": dict(
        cat="RRFS Severe", name="Most-Unstable CAPE", recipe="rrfs_field",
        rrfs_field_key="cape_mu", rrfs_idx_tmpl="CAPE:255-0 mb above ground:{fhr} hour fcst:",
        cmap="mucape", units="J/kg",
        spc_title="RRFS 255mb most-unstable CAPE — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_cin_mu": dict(
        cat="RRFS Severe", name="Most-Unstable CIN", recipe="rrfs_field",
        rrfs_field_key="cin_mu", rrfs_idx_tmpl="CIN:255-0 mb above ground:{fhr} hour fcst:",
        cmap="cin", units="J/kg",
        spc_title="RRFS 255mb most-unstable CIN — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_dcape": dict(
        cat="RRFS Severe", name="Downdraft CAPE", recipe="rrfs_field",
        rrfs_field_key="dcape", rrfs_idx_tmpl="DCAPE:400-0 mb above ground:{fhr} hour fcst:",
        cmap="dcape", units="J/kg",
        spc_title="RRFS downdraft CAPE (400mb layer) — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_srh_01": dict(
        cat="RRFS Severe", name="0-1km Storm-Relative Helicity", recipe="rrfs_field",
        rrfs_field_key="srh_01", rrfs_idx_tmpl="HLCY:1000-0 m above ground:{fhr} hour fcst:",
        cmap="srh", units="m2/s2",
        spc_title="RRFS 0-1km SRH — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_srh_03": dict(
        cat="RRFS Severe", name="0-3km Storm-Relative Helicity", recipe="rrfs_field",
        rrfs_field_key="srh_03", rrfs_idx_tmpl="HLCY:3000-0 m above ground:{fhr} hour fcst:",
        cmap="srh", units="m2/s2",
        spc_title="RRFS 0-3km SRH — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_shear_01": dict(
        cat="RRFS Severe", name="0-1km Bulk Shear", recipe="rrfs_shear",
        rrfs_shear_layer="0-1000 m above ground",
        cmap="shear", units="kt", convert=_MS_TO_KT,
        spc_title="RRFS 0-1km bulk shear — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_shear_06": dict(
        cat="RRFS Severe", name="0-6km Bulk Shear", recipe="rrfs_shear",
        rrfs_shear_layer="0-6000 m above ground",
        cmap="shear", units="kt", convert=_MS_TO_KT,
        spc_title="RRFS 0-6km bulk shear — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),

    # ---- Storm Attributes ---------------------------------------------
    "rrfs_refc": dict(
        cat="RRFS Storm Attributes", name="Composite Reflectivity", recipe="rrfs_field",
        rrfs_field_key="refc",
        rrfs_idx_tmpl="REFC:entire atmosphere (considered as a single layer):{fhr} hour fcst:",
        cmap="refc", units="dBZ",
        spc_title="RRFS composite reflectivity — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_uh_25": dict(
        cat="RRFS Storm Attributes", name="Max UH 2-5km", recipe="rrfs_field",
        rrfs_field_key="uh_25",
        rrfs_idx_tmpl="MXUPHL:5000-2000 m above ground:{fhrm1}-{fhr} hour max fcst:",
        cmap="uh", units="m2/s2",
        spc_title="RRFS max updraft helicity 2-5km — deterministic run",
        fhr_stride=RRFS_FHR_STEP, min_fhr=1, source="rrfs",
    ),
    "rrfs_uh_03": dict(
        cat="RRFS Storm Attributes", name="Max UH 0-3km", recipe="rrfs_field",
        rrfs_field_key="uh_03",
        rrfs_idx_tmpl="MXUPHL:3000-0 m above ground:{fhrm1}-{fhr} hour max fcst:",
        cmap="uh", units="m2/s2",
        spc_title="RRFS max updraft helicity 0-3km — deterministic run",
        fhr_stride=RRFS_FHR_STEP, min_fhr=1, source="rrfs",
    ),
    "rrfs_retop": dict(
        cat="RRFS Storm Attributes", name="Echo Top", recipe="rrfs_field",
        rrfs_field_key="retop",
        rrfs_idx_tmpl="RETOP:entire atmosphere (considered as a single layer):{fhr} hour fcst:",
        cmap="retop_kft", units="kft", convert=_M_TO_KFT,
        spc_title="RRFS radar echo top — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_vil": dict(
        cat="RRFS Storm Attributes", name="Vertically Integrated Liquid", recipe="rrfs_field",
        rrfs_field_key="vil",
        rrfs_idx_tmpl="VIL:entire atmosphere (considered as a single layer):{fhr} hour fcst:",
        cmap="vil", units="kg/m2",
        spc_title="RRFS vertically integrated liquid — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),
    "rrfs_gust": dict(
        cat="RRFS Storm Attributes", name="Max Wind Gust", recipe="rrfs_field",
        rrfs_field_key="gust", rrfs_idx_tmpl="GUST:surface:{fhr} hour fcst:",
        cmap="gust", units="kt", convert=_MS_TO_KT,
        spc_title="RRFS surface wind gust — deterministic run",
        fhr_stride=RRFS_FHR_STEP, source="rrfs",
    ),

    # ---- Fire -----------------------------------------------------------
    "rrfs_wfirepot": dict(
        cat="RRFS Fire", name="Hourly Wildfire Potential", recipe="rrfs_field",
        rrfs_field_key="wfirepot",
        rrfs_idx_tmpl="WFIREPOT:surface:{fhrm1}-{fhr} hour ave fcst:",
        cmap="wfire", units="",
        spc_title="RRFS hourly wildfire potential — deterministic run",
        fhr_stride=RRFS_FHR_STEP, min_fhr=1, source="rrfs",
    ),
}


def register() -> None:
    for pid, prod in PRODUCTS.items():
        core.PRODUCTS[pid] = prod
