"""Catalog: groups REFS PRODUCTS into SPC-HREF-style tabs."""
from __future__ import annotations

import sys
from pathlib import Path

# refs_core.py lives at the repo root, alongside the `app/` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import refs_core as core  # noqa: E402

# Register Phase D composite products and snow probability sweep.
from . import extra_products as _extras   # noqa: E402
_extras.register()

# Register REPS (Environment Canada Regional Ensemble Prediction System) --
# a genuinely independent data source/pipeline from REFS/HREF, given its
# own top-level tab rather than folded into the REFS/HREF model toggle.
from . import reps_products as _reps      # noqa: E402
_reps.register()


# Map refs_core category strings → SPC-HREF top-tab buckets.
TAB_ORDER = [
    "SPC Guidance",
    "Synoptic",
    "Severe",
    "Winter",
    "Fire",
    "Precipitation",
    "Storm Attributes",
    "Member Viewer",
    "Ensemble Spread",
    "REPS",
]

CATEGORY_TO_TAB = {
    # SPC Guidance now holds only the SPC HREF calibrated guidance products,
    # mirroring the real SPC experimental HREF page. The former occupants
    # (REFS PMM reflectivity + UH) moved to Storm Attributes.
    "Reflectivity (PMM Series)":   "Storm Attributes",
    "Updraft Helicity (2-5 km)":   "Storm Attributes",
    "Calibrated Severe (4-h)":     "SPC Guidance",
    "Calibrated Severe (24-h)":    "SPC Guidance",
    "Calibrated Thunder":          "SPC Guidance",
    "Lightning Density (4-h)":     "SPC Guidance",
    "Synoptic / Surface":          "Synoptic",
    "Synoptic / Moisture":         "Synoptic",
    "Synoptic / Upper":            "Synoptic",
    "Kinematics":                  "Synoptic",
    "Surface":                     "Synoptic",
    "Severe Probabilities":        "Severe",
    "Thermodynamics":              "Severe",
    "QPF (PMM)":                   "Precipitation",
    "QPF (LPMM)":                  "Precipitation",
    "QPF (Mean)":                  "Precipitation",
    "QPF (Prob)":                  "Precipitation",
    "QPF (EAS scale-aware)":       "Precipitation",
    "Flash Flood Threat":          "Precipitation",
    "Member Plots (RRFS_A)":       "Member Viewer",
    "Ensemble Spread":             "Ensemble Spread",
    "Satellite (Simulated)":       "Storm Attributes",
    "Fire Weather":                "Fire",
    "REPS":                        "REPS",
    "REPS Derived":                "REPS",
    "REPS Spread":                 "REPS",
    "REPS Precip Type":            "REPS",
    "REPS Precipitation":          "REPS",
    "REPS Flooding":               "REPS",
    "REPS Severe Proxy":           "REPS",
    "REPS Extremes":               "REPS",
    "REPS Synoptic":               "REPS",
    "REPS Members":                "REPS",
}

# A few products move into Winter / Fire when their name implies it.
WINTER_KEYS = {
    "snow_24h_pmmn", "snow_24h_mean",
    # Phase D snow additions
    "snow_6h_pmmn",
    "snow_6h_prob_025", "snow_6h_prob_076",
    "snow_6h_prob_152", "snow_6h_prob_304",
}
FIRE_KEYS: set[str] = set()  # placeholder for future fire-wx products


def tab_for(pid: str, prod: dict) -> str:
    if pid in WINTER_KEYS:
        return "Winter"
    if pid in FIRE_KEYS:
        return "Fire"
    return CATEGORY_TO_TAB.get(prod["cat"], "Storm Attributes")


def catalog() -> dict:
    """Return the full product catalog grouped by tab → category → products."""
    tabs: dict[str, dict[str, list[dict]]] = {t: {} for t in TAB_ORDER}
    for pid, prod in core.PRODUCTS.items():
        t = tab_for(pid, prod)
        cat = prod["cat"]
        tabs[t].setdefault(cat, []).append({
            "pid": pid,
            "name": prod["name"],
            "ftype": prod.get("ftype") or prod.get("recipe") or "mean",
            "spc_title": prod.get("spc_title", prod["name"]),
            # Units string ("%", "in", "dBZ", …). The frontend uses "%" to
            # detect probability products so it can auto-select the SPC Ramp
            # palette for them (and Default for everything else).
            "units": prod.get("units", ""),
            # Earliest forecast hour where this product can have data
            # (n-hour accumulations need fhr ≥ n; instantaneous products = 0).
            "min_fhr": int(prod.get("min_fhr", 0)),
            # Forecast-hour alignment: an n-hour accumulation is only
            # available at fhrs that are multiples of n. Frontend uses this
            # to disable mis-aligned timeline cells.
            "fhr_stride": int(prod.get("fhr_stride", 1)),
            # Data source. Default products come from the REFS/HREF buckets;
            # SPC calibrated guidance (source='spc_post') uses its own data
            # source + availability, so the frontend handles it specially.
            "source": prod.get("source", ""),
        })
    # Drop empty tabs.
    return {t: tabs[t] for t in TAB_ORDER if tabs[t]}


def regions() -> list[dict]:
    out = []
    for key, r in core.REGIONS.items():
        out.append({
            "key": key,
            "name": r["name"],
            "lon": list(r["lon"]),
            "lat": list(r["lat"]),
        })
    return out


PALETTES = list(core.PALETTES)
THEMES = list(core.THEMES.keys())
APP_VERSION = core.APP_VERSION
