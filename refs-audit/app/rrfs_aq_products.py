"""RRFS air-quality product registry.

Registers smoke/dust/AOD/AQI products into refs_core.PRODUCTS as their own
"Air Quality" tab (see app/catalog.py), a genuinely independent data
source from REFS/HREF/REPS -- RRFS's raw deterministic output, not an
ensemble product. See app/rrfs_aq_data.py's module docstring for why
REFS/HREF's own ensemble-post files can't supply this (they don't carry
smoke/dust fields at all).

v1 is deliberately scoped to the wildfire-smoke-relevant fields: surface
smoke, column (vertically-integrated) smoke, total AOD, and a computed
EPA AQI. Dust variants (RRFS also publishes coarse/fine dust MASSDEN/
COLMD the same way) are a natural, low-effort follow-up using the exact
same recipe -- not included here to keep the first batch focused.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import refs_core as core  # noqa: E402

_KGM3_TO_UGM3 = lambda x: x * 1e9   # noqa: E731 -- smoke/PM2.5 mass density
_KGM2_TO_MGM2 = lambda x: x * 1e6   # noqa: E731 -- column-integrated smoke mass

RRFS_AQ_FHR_STEP = 1

PRODUCTS = {
    "rrfs_smoke_sfc": dict(
        cat="Air Quality", name="Surface Smoke", recipe="rrfs_aq_field",
        rrfs_field="smoke_sfc",
        cmap="smoke_sfc", units="ug/m3", convert=_KGM3_TO_UGM3,
        spc_title="RRFS near-surface smoke (PM2.5) — deterministic run",
        fhr_stride=RRFS_AQ_FHR_STEP, source="rrfs_aq",
    ),
    "rrfs_vi_smoke": dict(
        cat="Air Quality", name="Column Smoke (VI)", recipe="rrfs_aq_field",
        rrfs_field="vi_smoke",
        cmap="smoke_vi", units="mg/m2", convert=_KGM2_TO_MGM2,
        spc_title="RRFS column-integrated smoke (PM2.5) — deterministic run",
        fhr_stride=RRFS_AQ_FHR_STEP, source="rrfs_aq",
    ),
    "rrfs_aod": dict(
        cat="Air Quality", name="Aerosol Optical Depth", recipe="rrfs_aq_field",
        rrfs_field="aod",
        cmap="aod", units="",
        spc_title="RRFS total aerosol optical depth — deterministic run",
        fhr_stride=RRFS_AQ_FHR_STEP, source="rrfs_aq",
    ),
    "rrfs_aqi": dict(
        cat="Air Quality", name="Air Quality Index (PM2.5)", recipe="rrfs_aqi",
        cmap="aqi_epa", units="AQI",
        cbar_tick_positions=[25, 75, 125, 175, 250, 400],
        cbar_tick_labels=["Good", "Moderate", "USG", "Unhealthy", "V. Unhealthy", "Hazardous"],
        spc_title="RRFS Air Quality Index (EPA PM2.5 breakpoints)",
        fhr_stride=RRFS_AQ_FHR_STEP, min_fhr=1, source="rrfs_aq",
    ),
}


def register() -> None:
    for pid, prod in PRODUCTS.items():
        core.PRODUCTS[pid] = prod
