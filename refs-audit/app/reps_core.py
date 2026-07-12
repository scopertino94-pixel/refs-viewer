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


async def load_reps_windowed_mean(
    cache_dir: Path, date: str, run: int, var: str, level: str, fhr: int,
    window_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ensemble mean of a window_h-hour total for a CUMULATIVE-SINCE-INIT
    field (APCP, SFCWRO -- confirmed via direct GRIB inspection: stepRange
    is always '0-N', growing with fhr, not a sliding window). A window
    total is the DIFFERENCE of two cumulative ensemble means -- valid
    because subtraction is linear, so this equals the true windowed
    ensemble mean exactly, no extra per-member math needed.

    Negative residue from independent decode rounding is clipped to 0.
    """
    end = await load_reps_mean(cache_dir, date, run, var, level, fhr)
    if end is None:
        return None
    end_mean, lat2d, lon2d = end
    if fhr <= window_h:
        return np.clip(end_mean, 0.0, None), lat2d, lon2d
    start = await load_reps_mean(cache_dir, date, run, var, level, fhr - window_h)
    if start is None:
        return None
    start_mean, _, _ = start
    window = end_mean - start_mean
    return np.clip(window, 0.0, None), lat2d, lon2d


async def load_reps_olr_window(
    cache_dir: Path, date: str, run: int, fhr: int, window_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ensemble-mean outgoing longwave radiation (W/m^2) averaged over a
    window_h-hour window ending at fhr. ULWRF_NTAT is published as a
    time-MEAN flux over 0-N (confirmed empirically: the 0-N mean is
    ~stable near 242 W/m^2 across N=3..24, i.e. an average, not an
    accumulation whose value would grow with N). To recover a sub-window
    mean from two cumulative means, un-integrate:
        avg(t1,t2) = (t2*avg(0,t2) - t1*avg(0,t1)) / (t2 - t1)
    so late forecast hours aren't smeared by early convection the way
    the raw 0-N average is. Low OLR = deep, cold convective cloud tops.
    """
    end = await load_reps_mean(cache_dir, date, run, "ULWRF", "NTAT", fhr)
    if end is None:
        return None
    end_avg, lat2d, lon2d = end
    if fhr <= window_h:
        return end_avg, lat2d, lon2d
    start = await load_reps_mean(cache_dir, date, run, "ULWRF", "NTAT", fhr - window_h)
    if start is None:
        return None
    start_avg, _, _ = start
    t1 = fhr - window_h
    window_avg = (fhr * end_avg - t1 * start_avg) / window_h
    return window_avg, lat2d, lon2d


async def load_reps_olr_window_members(
    cache_dir: Path, date: str, run: int, fhr: int, window_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-MEMBER window-averaged OLR (W/m^2) -- the member-resolved
    analog of load_reps_olr_window, for the OLR stamp panel + member
    browser. Same un-integration formula, applied per member so each
    member's own convective spread is preserved. Returns the full
    (members[21,ny,nx], lat2d, lon2d) array."""
    end = await load_reps_members(cache_dir, date, run, "ULWRF", "NTAT", fhr)
    if end is None:
        return None
    end_m, lat2d, lon2d = end
    if fhr <= window_h:
        return end_m, lat2d, lon2d
    start = await load_reps_members(cache_dir, date, run, "ULWRF", "NTAT", fhr - window_h)
    if start is None:
        return None
    start_m, _, _ = start
    t1 = fhr - window_h
    window = (fhr * end_m - t1 * start_m) / window_h
    return window, lat2d, lon2d


async def load_reps_windowed_members(
    cache_dir: Path, date: str, run: int, var: str, level: str, fhr: int,
    window_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-MEMBER window_h-hour total for a cumulative-since-init field
    (APCP) -- the member-resolved analog of load_reps_windowed_mean, for
    the stamp-panel renderer. Each member's window total is the
    difference of that same member's two cumulative fields (end minus
    start), so the per-member spread is preserved (unlike the mean,
    which could be differenced after averaging). Returns the full
    (members[21,ny,nx], lat2d, lon2d) array. Negative decode-rounding
    residue clipped to 0.
    """
    end = await load_reps_members(cache_dir, date, run, var, level, fhr)
    if end is None:
        return None
    end_m, lat2d, lon2d = end
    if fhr <= window_h:
        return np.clip(end_m, 0.0, None), lat2d, lon2d
    start = await load_reps_members(cache_dir, date, run, var, level, fhr - window_h)
    if start is None:
        return None
    start_m, _, _ = start
    return np.clip(end_m - start_m, 0.0, None), lat2d, lon2d


def _decode_qpf_prob_sync(path: Path, thresh_mm: float):
    """Blocking cfgrib decode of a single probability-of-exceedance
    message from a REPS "-Prob" accumulation file -- run via
    asyncio.to_thread from callers.

    These files bundle 3 message families in one GRIB2 file (confirmed
    via direct eccodes message inspection): 13-14 threshold messages
    (productDefinitionTemplateNumber=9, "probability of event above
    lower limit", thresholds 0.2-100mm), 5 percentile messages (PDT10),
    and 4 derived-stat messages (PDT12: mean/spread/min/max). Filtering
    cfgrib to the exact (PDT, scaledValueOfLowerLimit, scaleFactorOfLowerLimit)
    triple isolates exactly one clean 2D field -- no ensemble math needed,
    this is a genuine pre-computed probability (%), not something we
    derive ourselves.
    """
    import cfgrib

    ds = cfgrib.open_dataset(str(path), filter_by_keys={
        "productDefinitionTemplateNumber": 9,
        "scaledValueOfLowerLimit": int(round(thresh_mm)),
        "scaleFactorOfLowerLimit": 0,
    })
    varname = list(ds.data_vars)[0]
    lat2d = ds.latitude.values
    lon2d = ds.longitude.values
    rs, cs = _crop_bbox(lat2d, lon2d)
    data = ds[varname].values[rs, cs]
    return data, lat2d[rs, cs], lon2d[rs, cs]


async def load_reps_qpf_prob(
    cache_dir: Path, date: str, run: int, fhr: int, thresh_mm: float,
    window_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Probability (%) that window_h-hour total precip exceeds thresh_mm,
    decoded directly from REPS's own TPRATE-Accum{window_h}h-Prob file.
    Only published at fhrs where fhr % window_h == 0 (e.g. the 6h-window
    file exists at F06/F12/F18... not F03/F09) -- callers must gate on
    that via the product's fhr_stride.
    """
    var = f"TPRATE-Accum{window_h}h-Prob"
    ckey = (date, run, var, thresh_mm, fhr)
    if ckey in _decoded_cache:
        return _decoded_cache[ckey]

    path = await ensure_reps_file_cached(cache_dir, date, run, var, "SFC", fhr)
    if path is None:
        return None

    result = await asyncio.to_thread(_decode_qpf_prob_sync, path, thresh_mm)

    if len(_decoded_cache) >= _DECODED_CACHE_MAX:
        _decoded_cache.pop(next(iter(_decoded_cache)))
    _decoded_cache[ckey] = result
    return result


async def load_reps_shear_mean(
    cache_dir: Path, date: str, run: int, level_lo: str, level_hi: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Bulk shear vector magnitude between two pressure levels, computed
    per-member (vector subtract, then magnitude) BEFORE averaging --
    same reasoning as wind speed: magnitude is a nonlinear function of
    the vector components, so the mean of per-member magnitudes differs
    from (and is more physically correct than) the magnitude of the
    mean-vector difference in a spread-out ensemble."""
    u_lo = await load_reps_members(cache_dir, date, run, "UGRD", level_lo, fhr)
    v_lo = await load_reps_members(cache_dir, date, run, "VGRD", level_lo, fhr)
    u_hi = await load_reps_members(cache_dir, date, run, "UGRD", level_hi, fhr)
    v_hi = await load_reps_members(cache_dir, date, run, "VGRD", level_hi, fhr)
    if u_lo is None or v_lo is None or u_hi is None or v_hi is None:
        return None
    u_lo_m, lat2d, lon2d = u_lo
    v_lo_m, _, _ = v_lo
    u_hi_m, _, _ = u_hi
    v_hi_m, _, _ = v_hi
    du = u_hi_m - u_lo_m
    dv = v_hi_m - v_lo_m
    shear_members = np.sqrt(du**2 + dv**2)
    return shear_members.mean(axis=0), lat2d, lon2d


async def load_reps_lapse_rate_mean(
    cache_dir: Path, date: str, run: int, level_lo: str, level_hi: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Lapse rate (K/km) between two pressure levels, from ensemble-MEAN
    T/HGT profiles (mean-first -- same cost/precision trade-off as
    freezing level: not strongly nonlinear, and 21x cheaper than
    per-member). Positive = temperature decreasing with height (the
    normal/steep-instability sense; matches refs_core's existing
    cmap_lapse_rate convention)."""
    t_lo = await load_reps_mean(cache_dir, date, run, "TMP", level_lo, fhr)
    h_lo = await load_reps_mean(cache_dir, date, run, "HGT", level_lo, fhr)
    t_hi = await load_reps_mean(cache_dir, date, run, "TMP", level_hi, fhr)
    h_hi = await load_reps_mean(cache_dir, date, run, "HGT", level_hi, fhr)
    if t_lo is None or h_lo is None or t_hi is None or h_hi is None:
        return None
    t_lo_m, lat2d, lon2d = t_lo
    h_lo_m, _, _ = h_lo
    t_hi_m, _, _ = t_hi
    h_hi_m, _, _ = h_hi
    dz_km = (h_hi_m - h_lo_m) / 1000.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lapse = -(t_hi_m - t_lo_m) / dz_km
    return lapse, lat2d, lon2d


def _decode_prob_bundle_stat_sync(path: Path, pdt: int, match_key: str, match_val):
    """Blocking cfgrib decode of a single percentile or derived-stat
    message from a REPS TMP-Prob/WIND-Prob/HEATX-Prob/WCF-Prob file --
    run via asyncio.to_thread from callers.

    These 4 files are a DIFFERENT, smaller bundle shape than the
    precip-rate "-Prob" files (confirmed via direct eccodes message
    inspection): only 9 messages -- 5 percentile messages
    (productDefinitionTemplateNumber=6, percentileValue=10/25/50/75/90)
    + 4 derived-stat messages (PDT=2, derivedForecast per WMO code table
    4.7: 0=mean, 4=spread, 8=min, 9=max). NO threshold-exceedance
    messages exist in this family -- "probability of extreme heat/cold/
    wind" is not available the way precip-rate exceedance probability
    is; percentile/mean/spread/min/max are the only things to decode.
    """
    import cfgrib

    ds = cfgrib.open_dataset(str(path), filter_by_keys={
        "productDefinitionTemplateNumber": pdt,
        match_key: match_val,
    })
    varname = list(ds.data_vars)[0]
    lat2d = ds.latitude.values
    lon2d = ds.longitude.values
    rs, cs = _crop_bbox(lat2d, lon2d)
    data = ds[varname].values[rs, cs]
    return data, lat2d[rs, cs], lon2d[rs, cs]


async def load_reps_prob_bundle_stat(
    cache_dir: Path, date: str, run: int, var: str, level: str, fhr: int,
    pdt: int, match_key: str, match_val,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Decode one percentile or derived-stat field from a REPS
    TMP-Prob/WIND-Prob/HEATX-Prob/WCF-Prob bundle. `var` is the file
    token (e.g. "HEATX-Prob"); `pdt`/`match_key`/`match_val` select the
    message -- pdt=6 + match_key='percentileValue' + match_val in
    {10,25,50,75,90} for a percentile, or pdt=2 +
    match_key='derivedForecast' + match_val in {0,4,8,9} for
    mean/spread/min/max.
    """
    ckey = (date, run, var, pdt, match_key, match_val, fhr)
    if ckey in _decoded_cache:
        return _decoded_cache[ckey]

    path = await ensure_reps_file_cached(cache_dir, date, run, var, level, fhr)
    if path is None:
        return None

    result = await asyncio.to_thread(_decode_prob_bundle_stat_sync, path, pdt, match_key, match_val)

    if len(_decoded_cache) >= _DECODED_CACHE_MAX:
        _decoded_cache.pop(next(iter(_decoded_cache)))
    _decoded_cache[ckey] = result
    return result


def _grid_metrics(lat2d: np.ndarray, lon2d: np.ndarray):
    """Local grid-step distances (meters) in the array's row (y) and
    column (x) index directions, computed directly from the actual
    lat/lon arrays via central differences.

    This is deliberately NOT the "assume a uniform dlat/dlon sampled
    from one grid-point pair" shortcut refs_core.py's REFS/HREF
    _divergence helper uses -- that's fine for REFS/HREF's own grid,
    but REPS's grid is rotated-pole (see module docstring): after
    cropping to a real-world bounding box, a row/col index step does
    NOT correspond to a fixed true lat/lon step size across the whole
    cropped domain, and near the crop edges a "col" step isn't even
    purely east-west. Computing dx/dy from the full 2D coordinate
    arrays at every point is correct regardless of that distortion.

    Returns (dx, dy), each shaped like lat2d.
    """
    R = 6371000.0
    lat_r = np.deg2rad(lat2d)
    lon_r = np.deg2rad(lon2d)
    dlat_dy, dlat_dx = np.gradient(lat_r)
    dlon_dy, dlon_dx = np.gradient(lon_r)
    dy = R * np.sqrt(dlat_dy**2 + (dlon_dy * np.cos(lat_r))**2)
    dx = R * np.sqrt(dlat_dx**2 + (dlon_dx * np.cos(lat_r))**2)
    return dx, dy


async def load_reps_divergence_mean(
    cache_dir: Path, date: str, run: int, level: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Horizontal divergence (du/dx + dv/dy), x10^-5/s, at a pressure
    level. Divergence is a LINEAR operator on u,v, so computing it from
    the ensemble-MEAN wind field is mathematically exact -- unlike heat
    index/wind chill, there's no per-member-first cost/precision
    tradeoff here; averaging first changes nothing."""
    u = await load_reps_mean(cache_dir, date, run, "UGRD", level, fhr)
    v = await load_reps_mean(cache_dir, date, run, "VGRD", level, fhr)
    if u is None or v is None:
        return None
    u_m, lat2d, lon2d = u
    v_m, _, _ = v
    dx, dy = _grid_metrics(lat2d, lon2d)
    du_dx = np.gradient(u_m, axis=1) / dx
    dv_dy = np.gradient(v_m, axis=0) / dy
    div = (du_dx + dv_dy) * 1e5
    return div, lat2d, lon2d


async def load_reps_vorticity_mean(
    cache_dir: Path, date: str, run: int, level: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Absolute vorticity (relative + planetary), x10^-5/s, at a
    pressure level. Relative vorticity (dv/dx - du/dy) is linear in
    u,v -- same exactness argument as divergence above. Planetary
    vorticity f=2*Omega*sin(lat) depends only on latitude, not the
    ensemble at all."""
    u = await load_reps_mean(cache_dir, date, run, "UGRD", level, fhr)
    v = await load_reps_mean(cache_dir, date, run, "VGRD", level, fhr)
    if u is None or v is None:
        return None
    u_m, lat2d, lon2d = u
    v_m, _, _ = v
    dx, dy = _grid_metrics(lat2d, lon2d)
    dv_dx = np.gradient(v_m, axis=1) / dx
    du_dy = np.gradient(u_m, axis=0) / dy
    f = 2.0 * 7.2921e-5 * np.sin(np.deg2rad(lat2d))
    abs_vort = (dv_dx - du_dy + f) * 1e5
    return abs_vort, lat2d, lon2d


async def load_reps_temp_advection_mean(
    cache_dir: Path, date: str, run: int, level: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Horizontal temperature advection -(u*dT/dx + v*dT/dy), in K per
    3 hours (matches the usual synoptic advection-map convention), at
    a pressure level, from ensemble-MEAN T/U/V.

    Computed mean-first: unlike divergence/vorticity above, advection
    is a PRODUCT of wind and a temperature gradient (bilinear, not
    linear), so mean-first is an approximation, not an exact identity
    -- same cost/precision tradeoff already accepted for freezing level
    and lapse rate (see those docstrings). Per-member would cost 21x
    the fetch/decode for a refinement that matters far less here than
    it does for heat index/wind chill's much more sharply nonlinear
    formulas.
    """
    u = await load_reps_mean(cache_dir, date, run, "UGRD", level, fhr)
    v = await load_reps_mean(cache_dir, date, run, "VGRD", level, fhr)
    t = await load_reps_mean(cache_dir, date, run, "TMP", level, fhr)
    if u is None or v is None or t is None:
        return None
    u_m, lat2d, lon2d = u
    v_m, _, _ = v
    t_m, _, _ = t
    dx, dy = _grid_metrics(lat2d, lon2d)
    dT_dx = np.gradient(t_m, axis=1) / dx
    dT_dy = np.gradient(t_m, axis=0) / dy
    advection_per_s = -(u_m * dT_dx + v_m * dT_dy)
    advection_per_3h = advection_per_s * 3600.0 * 3.0
    return advection_per_3h, lat2d, lon2d


async def load_reps_thickness_mean(
    cache_dir: Path, date: str, run: int, level_lo: str, level_hi: str, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Thickness (dam) between two pressure levels, from ensemble-mean
    heights -- e.g. 1000-500mb thickness, the classic rain/snow-line
    field (540 dam is the standard threshold)."""
    h_lo = await load_reps_mean(cache_dir, date, run, "HGT", level_lo, fhr)
    h_hi = await load_reps_mean(cache_dir, date, run, "HGT", level_hi, fhr)
    if h_lo is None or h_hi is None:
        return None
    h_lo_m, lat2d, lon2d = h_lo
    h_hi_m, _, _ = h_hi
    thickness_dam = (h_hi_m - h_lo_m) / 10.0
    return thickness_dam, lat2d, lon2d
