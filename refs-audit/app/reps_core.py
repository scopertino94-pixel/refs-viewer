"""REPS field loading: decode, crop, ensemble statistics.

REPS files bundle all 21 members (1 control + 20 perturbed) as separate
GRIB2 messages with mismatched `dataType` -- cfgrib refuses to open them as
one dataset, but opens cleanly when filtered by dataType (validated against
a live file: filter_by_keys={'dataType':'cf'} -> control, {'dataType':'pf'}
-> 20 perturbed members, `number` 1-20).

REPS's 10km grid is a rotated-pole grid covering "Canada and the United
States" -- the domain center unrotates correctly to true lat/lon, but the
domain's far corners (near the pole) unrotate to wild true-coordinates
(one measured corner landed near Japan). Rendering the full array with
pcolormesh would produce wraparound streaks across the map, so every load
is cropped to a real-world bounding box first.

Rendering reuses refs_core.PlotManager.shaded(data, lats, lons, prod,
region, run_dt, fhr) directly -- that function is already grid-agnostic
(just needs 2D data/lat/lon arrays), so no changes to refs_core.py's
rendering engine are needed for REPS.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from .reps_grib import ensure_reps_file_cached

# Real-world bounding box REPS renders are cropped to before plotting.
# Generous padding around the advertised "Canada + US" domain; the far
# corners of the native rotated grid are excluded entirely.
CROP_LAT = (15.0, 72.0)
CROP_LON = (-145.0, -45.0)

_decoded_cache: dict[tuple, tuple] = {}
_DECODED_CACHE_MAX = 64


def _crop_bbox(lat2d: np.ndarray, lon2d: np.ndarray):
    """Return (row_slice, col_slice) bounding the crop box.

    Uses the bounding row/col range of matching points rather than a
    scattered mask, so the result stays a proper rectangular sub-array
    (required for pcolormesh) -- a few cells outside the exact box may be
    included, which is harmless.
    """
    mask = ((lat2d >= CROP_LAT[0]) & (lat2d <= CROP_LAT[1]) &
            (lon2d >= CROP_LON[0]) & (lon2d <= CROP_LON[1]))
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("REPS crop bbox matched no grid points")
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def _decode_members_sync(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blocking cfgrib decode -- run via asyncio.to_thread from callers.

    Returns (members[21,ny,nx], lat2d[ny,nx], lon2d[ny,nx]), already
    cropped to CROP_LAT/CROP_LON and ordered [control, perturbed 1..20].
    """
    import cfgrib

    ds_cf = cfgrib.open_dataset(str(path), filter_by_keys={"dataType": "cf"})
    ds_pf = cfgrib.open_dataset(str(path), filter_by_keys={"dataType": "pf"})

    # Exactly one data variable per REPS file (the file is named after it).
    varname_cf = list(ds_cf.data_vars)[0]
    varname_pf = list(ds_pf.data_vars)[0]

    lat2d = ds_cf.latitude.values
    lon2d = ds_cf.longitude.values
    rs, cs = _crop_bbox(lat2d, lon2d)

    control = ds_cf[varname_cf].values[rs, cs]
    perturbed = ds_pf[varname_pf].values[:, rs, cs]   # (20, ny, nx)

    members = np.concatenate([control[None, ...], perturbed], axis=0)  # (21,ny,nx)
    return members, lat2d[rs, cs], lon2d[rs, cs]


async def load_reps_members(
    cache_dir: Path, date: str, run: int, var: str, level: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fetch (if needed) + decode a REPS file, returning the 21-member array.

    Decoded (already-cropped) results are cached in-process by file identity
    so repeat requests against the same (date,run,var,level,fhr) skip the
    cfgrib decode entirely -- the expensive step after the download itself.
    """
    ckey = (date, run, var, level, fhr)
    if ckey in _decoded_cache:
        return _decoded_cache[ckey]

    path = await ensure_reps_file_cached(cache_dir, date, run, var, level, fhr)
    if path is None:
        return None

    members, lat2d, lon2d = await asyncio.to_thread(_decode_members_sync, path)

    if len(_decoded_cache) >= _DECODED_CACHE_MAX:
        _decoded_cache.pop(next(iter(_decoded_cache)))
    _decoded_cache[ckey] = (members, lat2d, lon2d)
    return members, lat2d, lon2d


async def load_reps_mean(
    cache_dir: Path, date: str, run: int, var: str, level: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ensemble mean across all 21 members. Returns (mean2d, lat2d, lon2d)."""
    result = await load_reps_members(cache_dir, date, run, var, level, fhr)
    if result is None:
        return None
    members, lat2d, lon2d = result
    return members.mean(axis=0), lat2d, lon2d


async def load_reps_wind_speed_mean(
    cache_dir: Path, date: str, run: int, level: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ensemble-mean wind SPEED at a level, from separate UGRD/VGRD files.

    REPS publishes wind components (not a pre-combined speed field) at
    pressure levels, unlike the AGL-10m "WIND" file which is speed
    already. Speed is computed per-member (sqrt(u^2+v^2)) BEFORE
    averaging -- averaging components first would understate speed in a
    spread-out ensemble (vector mean vs. mean of magnitudes).
    """
    u = await load_reps_members(cache_dir, date, run, "UGRD", level, fhr)
    v = await load_reps_members(cache_dir, date, run, "VGRD", level, fhr)
    if u is None or v is None:
        return None
    u_members, lat2d, lon2d = u
    v_members, _, _ = v
    speed_members = np.sqrt(u_members**2 + v_members**2)
    return speed_members.mean(axis=0), lat2d, lon2d


async def load_reps_spread(
    cache_dir: Path, date: str, run: int, var: str, level: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ensemble spread (std dev across all 21 members) -- an uncertainty
    map: where members disagree most is where forecast confidence is
    lowest. Same member array load_reps_mean already does, one extra
    numpy call, no extra fetch cost since the decoded array is cached."""
    result = await load_reps_members(cache_dir, date, run, var, level, fhr)
    if result is None:
        return None
    members, lat2d, lon2d = result
    return members.std(axis=0), lat2d, lon2d


def _rothfusz_heat_index(t_f: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """NWS Rothfusz regression. Unreliable below ~80 degF -- falls back
    to plain temperature there (same convention refs_core.py's REFS
    _heat_index recipe already uses)."""
    hi = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f * t_f
        - 0.05481717 * rh * rh
        + 0.00122874 * t_f * t_f * rh
        + 0.00085282 * t_f * rh * rh
        - 0.00000199 * t_f * t_f * rh * rh
    )
    return np.where(t_f < 80.0, t_f, hi)


async def load_reps_heat_index_mean(
    cache_dir: Path, date: str, run: int, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ensemble-mean heat index from 2m temp + 2m RH (REPS publishes RH
    directly, unlike REFS which derives it from dewpoint). Computed
    per-member BEFORE averaging -- heat index is a strongly nonlinear
    function of T and RH, so heat-index-of-the-mean would understate
    peak values in a spread-out ensemble, same reasoning as wind speed
    above.
    """
    t = await load_reps_members(cache_dir, date, run, "TMP", "AGL-2m", fhr)
    rh = await load_reps_members(cache_dir, date, run, "RH", "AGL-2m", fhr)
    if t is None or rh is None:
        return None
    t_members, lat2d, lon2d = t
    rh_members, _, _ = rh
    t_f_members = (t_members - 273.15) * 9.0 / 5.0 + 32.0
    hi_members = _rothfusz_heat_index(t_f_members, rh_members)
    return hi_members.mean(axis=0), lat2d, lon2d


def _wind_chill(t_f: np.ndarray, wind_mph: np.ndarray) -> np.ndarray:
    """NWS wind chill formula. Only valid for T<=50 degF and wind>3 mph --
    elsewhere returns plain temperature (chill has no meaning in warm/
    calm conditions)."""
    v16 = np.power(np.maximum(wind_mph, 0.0), 0.16)
    wc = 35.74 + 0.6215 * t_f - 35.75 * v16 + 0.4275 * t_f * v16
    return np.where((t_f <= 50.0) & (wind_mph > 3.0), wc, t_f)


async def load_reps_wind_chill_mean(
    cache_dir: Path, date: str, run: int, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ensemble-mean wind chill from 2m temp + 10m wind speed, computed
    per-member before averaging (same reasoning as heat index)."""
    t = await load_reps_members(cache_dir, date, run, "TMP", "AGL-2m", fhr)
    wind = await load_reps_members(cache_dir, date, run, "WIND", "AGL-10m", fhr)
    if t is None or wind is None:
        return None
    t_members, lat2d, lon2d = t
    wind_members, _, _ = wind
    t_f_members = (t_members - 273.15) * 9.0 / 5.0 + 32.0
    wind_mph_members = wind_members * 2.23694
    wc_members = _wind_chill(t_f_members, wind_mph_members)
    return wc_members.mean(axis=0), lat2d, lon2d


# Precip-type accumulation variables REPS publishes as separate fields
# (rain / snow-water-equiv / ice pellets / freezing rain), each an
# ensemble-mean-able ACCUMULATION -- REFS/HREF don't partition precip
# this way, so a "which type dominates" composite is genuinely unique
# to what REPS makes available. Order fixes the categorical index used
# by the colormap in reps_products.py -- keep them in sync.
PTYPE_VARS = ("ARAIN", "ASNOW", "AICEP", "AFRAIN")
PTYPE_LEVEL = "SFC"


async def load_reps_dominant_ptype(
    cache_dir: Path, date: str, run: int, fhr: int, min_total_mm: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Categorical "which precip type dominates" map: at each gridpoint,
    whichever of rain/snow/ice-pellets/freezing-rain has the highest
    ensemble-mean accumulation wins that cell. Cells with negligible
    total accumulation (< min_total_mm summed across all 4 types) are
    masked out entirely rather than picking an arbitrary "winner" among
    near-zero noise.

    Returns (category2d, lat2d, lon2d) where category is 0=rain,
    1=snow, 2=ice pellets, 3=freezing rain, NaN=no meaningful precip.
    """
    means = []
    lat2d = lon2d = None
    for var in PTYPE_VARS:
        result = await load_reps_mean(cache_dir, date, run, var, PTYPE_LEVEL, fhr)
        if result is None:
            return None
        mean, la, lo = result
        means.append(np.clip(mean, 0.0, None))   # accumulations are non-negative
        if lat2d is None:
            lat2d, lon2d = la, lo
    stack = np.stack(means, axis=0)   # (4, ny, nx)
    total = stack.sum(axis=0)
    dominant = np.argmax(stack, axis=0).astype(float)
    dominant = np.where(total >= min_total_mm, dominant, np.nan)
    return dominant, lat2d, lon2d


# Pressure levels (hPa, matching REPS's ISBL tokens) used for freezing-
# level interpolation, surface-up. REPS's full ISBL set goes further,
# but this range comfortably brackets any 0 degC crossing in the lower/
# mid troposphere without fetching levels that would rarely matter.
FREEZING_LEVEL_HPA = (1000, 925, 850, 700, 500)


async def load_reps_freezing_level(
    cache_dir: Path, date: str, run: int, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Height (m) of the 0 degC isotherm, from ensemble-mean temperature
    profiles at 1000/925/850/700/500 mb. Not published directly by
    REPS -- derived by linearly interpolating HGT against TMP across
    the bracketing pair of levels where the mean temperature profile
    crosses 273.15 K, computed on the ALREADY ensemble-meaned T/HGT
    profiles (unlike heat index/wind chill/wind speed above, freezing
    level is a level-finding operation, not a nonlinear function of the
    instantaneous fields, so there's no real precision cost to meaning
    first here -- and doing it once instead of per-member is 21x cheaper).

    Surface (AGL-2m) temperature is included as the lowest "level" using
    each column's own surface height... except REPS doesn't publish
    terrain height directly, so this uses the 1000-mb level as the
    effective lowest bracket instead. Columns entirely above or below
    freezing across the whole profile return NaN (no crossing to find).
    """
    t_profiles = []
    h_profiles = []
    lat2d = lon2d = None
    for hpa in FREEZING_LEVEL_HPA:
        level = f"ISBL-{hpa:04d}"
        t_result = await load_reps_mean(cache_dir, date, run, "TMP", level, fhr)
        h_result = await load_reps_mean(cache_dir, date, run, "HGT", level, fhr)
        if t_result is None or h_result is None:
            return None
        t_mean, la, lo = t_result
        h_mean, _, _ = h_result
        t_profiles.append(t_mean)
        h_profiles.append(h_mean)
        if lat2d is None:
            lat2d, lon2d = la, lo

    t_stack = np.stack(t_profiles, axis=0)   # (nlev, ny, nx), surface->aloft
    h_stack = np.stack(h_profiles, axis=0)
    freezing_k = 273.15
    nlev, ny, nx = t_stack.shape
    result = np.full((ny, nx), np.nan)

    # Walk adjacent level pairs looking for a sign change in (T - 0C).
    # First crossing found (from the surface up) wins -- if a profile has
    # a low-level inversion punching back above freezing there could be
    # more than one crossing; the lowest one is what matters for a
    # rain/snow line.
    for i in range(nlev - 1):
        t0, t1 = t_stack[i], t_stack[i + 1]
        h0, h1 = h_stack[i], h_stack[i + 1]
        crosses = ((t0 - freezing_k) * (t1 - freezing_k) <= 0) & (t0 != t1)
        still_unset = np.isnan(result)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(t1 != t0, (freezing_k - t0) / (t1 - t0), 0.0)
        interp_h = h0 + frac * (h1 - h0)
        take = crosses & still_unset
        result = np.where(take, interp_h, result)

    return result, lat2d, lon2d
