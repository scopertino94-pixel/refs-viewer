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
}


def register() -> None:
    for pid, prod in PRODUCTS.items():
        core.PRODUCTS[pid] = prod
