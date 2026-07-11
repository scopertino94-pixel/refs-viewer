"""REPS v1 product registry.

Registers Environment Canada REPS products into refs_core.PRODUCTS as its
own "REPS" tab (see app/catalog.py's TAB_ORDER/CATEGORY_TO_TAB), reusing
the recipe='reps_mean' dispatch added to PlotJob.render() in refs_core.py.

v1 is deliberately scoped to ensemble-MEAN products only. REPS also
publishes "-Prob" files (e.g. TMP-Prob, WIND-Prob), but those turned out
to be multi-message percentile/threshold-probability bundles (9 messages
per file: 5 percentiles + 4 unlabelled probability messages) rather than
a single clean field -- decoding them correctly needs its own validation
pass, deferred to a later addition rather than guessed at here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import refs_core as core  # noqa: E402

_K_TO_F = lambda k: (k - 273.15) * 9 / 5 + 32   # noqa: E731
_MS_TO_KT = lambda ms: ms * 1.94384              # noqa: E731
_PA_TO_HPA = lambda pa: pa / 100.0               # noqa: E731
_M_TO_DAM = lambda m: m / 10.0                   # noqa: E731 -- geopotential height, decameters
_FRAC_TO_PCT = lambda f: f * 100.0               # noqa: E731 -- volumetric soil moisture 0-1 -> %
# Spread (std-dev) conversions are DELTAS, not absolute values -- a Kelvin
# std-dev converts to a Fahrenheit std-dev by scaling only (no +32/-273.15
# offset, which would be meaningless applied to a spread rather than a value).
_K_TO_F_DELTA = lambda k: k * 9.0 / 5.0           # noqa: E731

REPS_MAX_FHOUR = 72
REPS_FHR_STEP = 3

PRODUCTS = {
    "reps_t2m_mean": dict(
        cat="REPS", name="2m Temperature (Mean)", recipe="reps_mean",
        reps_var="TMP", reps_level="AGL-2m",
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS 2m temperature — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_rh2m_mean": dict(
        cat="REPS", name="2m Relative Humidity (Mean)", recipe="reps_mean",
        reps_var="RH", reps_level="AGL-2m",
        cmap="rh", units="%",
        spc_title="REPS 2m relative humidity — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind10m_mean": dict(
        cat="REPS", name="10m Wind Speed (Mean)", recipe="reps_mean",
        reps_var="WIND", reps_level="AGL-10m",
        cmap="wind_sfc", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 10m wind speed — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_mslp_mean": dict(
        cat="REPS", name="MSLP (Mean)", recipe="reps_mean",
        reps_var="PRMSL", reps_level="MSL",
        cmap="mslp", units="hPa", convert=_PA_TO_HPA,
        spc_title="REPS mean sea-level pressure — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    # ---- v1.5: pressure-level / soil / cloud additions ----------------
    "reps_hgt500_mean": dict(
        cat="REPS", name="500mb Heights (Mean)", recipe="reps_mean",
        reps_var="HGT", reps_level="ISBL-0500",
        cmap="hgt500", units="dam", convert=_M_TO_DAM,
        spc_title="REPS 500-mb geopotential height — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_temp850_mean": dict(
        cat="REPS", name="850mb Temperature (Mean)", recipe="reps_mean",
        reps_var="TMP", reps_level="ISBL-0850",
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS 850-mb temperature — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_rh700_mean": dict(
        cat="REPS", name="700mb Relative Humidity (Mean)", recipe="reps_mean",
        reps_var="RH", reps_level="ISBL-0700",
        cmap="rh", units="%",
        spc_title="REPS 700-mb relative humidity — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind500_mean": dict(
        cat="REPS", name="500mb Wind Speed (Mean)", recipe="reps_wind_level_mean",
        reps_level="ISBL-0500",
        cmap="wind500", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 500-mb wind speed — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_soilt_mean": dict(
        cat="REPS", name="Soil Temperature 10cm (Mean)", recipe="reps_mean",
        reps_var="TSOIL", reps_level="DBS-10cm",
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS 10-cm soil temperature — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_soilm_mean": dict(
        cat="REPS", name="Soil Moisture 10cm (Mean)", recipe="reps_mean",
        reps_var="VSOILM", reps_level="DBS-10cm",
        cmap="rh", units="%", convert=_FRAC_TO_PCT,
        spc_title="REPS 10-cm volumetric soil moisture — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_tcdc_mean": dict(
        cat="REPS", name="Total Cloud Cover (Mean)", recipe="reps_mean",
        reps_var="TCDC", reps_level="SFC",
        cmap="clouds", units="%",
        spc_title="REPS total cloud cover — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    # ---- v2: creative derived/spread/precip-type products -------------
    "reps_heat_index_mean": dict(
        cat="REPS Derived", name="Heat Index (Mean)", recipe="reps_heat_index_mean",
        cmap="t2m", units="degF",
        spc_title="REPS heat index — 21-member ensemble mean (per-member Rothfusz, then averaged)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind_chill_mean": dict(
        cat="REPS Derived", name="Wind Chill (Mean)", recipe="reps_wind_chill_mean",
        cmap="t2m", units="degF",
        spc_title="REPS wind chill — 21-member ensemble mean (per-member NWS formula, then averaged)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_freezing_level": dict(
        cat="REPS Derived", name="Freezing Level (0°C Height)", recipe="reps_freezing_level",
        cmap="frzlvl", units="m",
        spc_title="REPS freezing level — height of 0°C isotherm from ensemble-mean T/Z profile",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_ptype_dominant": dict(
        cat="REPS Precip Type", name="Dominant Precip Type", recipe="reps_ptype_dominant",
        cmap="ptype_dom", units="",
        cbar_tick_positions=[0.5, 1.5, 2.5, 3.5],
        cbar_tick_labels=["Rain", "Snow", "Ice Pellets", "Frz Rain"],
        spc_title="REPS dominant precipitation type — winner among ensemble-mean rain/snow/ice pellets/freezing rain accumulation",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_spread_t2m": dict(
        cat="REPS Spread", name="2m Temperature Spread", recipe="reps_spread",
        reps_var="TMP", reps_level="AGL-2m",
        cmap="sptemp2m", units="degF", convert=_K_TO_F_DELTA,
        spc_title="REPS 2m temperature — ensemble spread (std dev across 21 members)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_spread_wind10m": dict(
        cat="REPS Spread", name="10m Wind Speed Spread", recipe="reps_spread",
        reps_var="WIND", reps_level="AGL-10m",
        cmap="spwind10m", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 10m wind speed — ensemble spread (std dev across 21 members)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
}


def register() -> None:
    for pid, prod in PRODUCTS.items():
        core.PRODUCTS[pid] = prod
