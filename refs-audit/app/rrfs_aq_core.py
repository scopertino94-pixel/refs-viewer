"""RRFS air-quality field loading: decode + EPA AQI conversion.

REFS/HREF's own ensemble-post files carry NO smoke/dust/AOD fields
(confirmed by direct .idx inspection -- only VIS). RRFS, the deterministic
model REFS's members are drawn from, DOES publish them in its own "2dfld"
CONUS output (see app/rrfs_aq_data.py's module docstring for the file-
family gotcha -- always the "3km...conus" file, never the "2p5km...hi/pr"
Hawaii/Puerto Rico nests).

Every field here is a SINGLE deterministic value (no ensemble, no member
loop) decoded from one byte-range-fetched GRIB2 record -- the simplest
loader shape of any data source in this app so far.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from .rrfs_aq_grib import ensure_rrfs_aq_record_cached

_decoded_cache: dict[tuple, tuple] = {}
_DECODED_CACHE_MAX = 64


def _decode_sync(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blocking cfgrib decode of a single-message GRIB2 file -- run via
    asyncio.to_thread from callers. No filter_by_keys needed: the byte
    range already isolates exactly one record."""
    import cfgrib

    ds = cfgrib.open_dataset(str(path))
    varname = list(ds.data_vars)[0]
    data = ds[varname].values
    lat2d = ds.latitude.values
    lon2d = ds.longitude.values
    if lat2d.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon2d, lat2d)
    return data, lat2d, lon2d


async def _load_record(
    cache_dir: Path, date: str, run: int, fhr: int, field_key: str, idx_substring: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ckey = (date, run, fhr, field_key)
    if ckey in _decoded_cache:
        return _decoded_cache[ckey]

    path = await ensure_rrfs_aq_record_cached(cache_dir, date, run, fhr, field_key, idx_substring)
    if path is None:
        return None

    result = await asyncio.to_thread(_decode_sync, path)

    if len(_decoded_cache) >= _DECODED_CACHE_MAX:
        _decoded_cache.pop(next(iter(_decoded_cache)))
    _decoded_cache[ckey] = result
    return result


async def load_rrfs_smoke_surface(
    cache_dir: Path, date: str, run: int, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Near-surface (8m AGL) smoke mass density (kg/m^3), instantaneous at
    fhr -- organic-matter-dry aerosol, PM2.5 size bin. This is the direct
    "surface smoke" signature (wildfire smoke specifically, not dust or
    other aerosol species)."""
    sub = (f"MASSDEN:8 m above ground:{fhr} hour fcst:"
           f"aerosol=Particulate organic matter dry:aerosol_size <2.5e-06")
    return await _load_record(cache_dir, date, run, fhr, "smoke_sfc", sub)


async def load_rrfs_pm25_total_mean(
    cache_dir: Path, date: str, run: int, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Near-surface TOTAL aerosol PM2.5 (kg/m^3), the (fhr-1)-to-fhr HOURLY
    AVERAGE RRFS itself publishes (not an instant value, and not something
    we average ourselves -- this is genuinely a sliding hourly mean field,
    unlike REPS's OLR which needed de-averaging math). "Total aerosol"
    (not just organic-matter smoke) is what the EPA AQI formula is meant to
    operate on -- during a smoke event the two are numerically close (smoke
    dominates total aerosol), but total is the physically correct input.
    Requires fhr >= 1 (no averaging window before the run starts).
    """
    if fhr < 1:
        return None
    sub = (f"MASSDEN:8 m above ground:{fhr - 1}-{fhr} hour ave fcst:"
           f"aerosol=Total aerosol:aerosol_size <2.5e-06")
    return await _load_record(cache_dir, date, run, fhr, "pm25_total_mean", sub)


async def load_rrfs_vi_smoke(
    cache_dir: Path, date: str, run: int, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Column-integrated (vertically-summed) smoke mass (kg/m^2),
    instantaneous at fhr -- organic-matter-dry aerosol, PM2.5 size bin."""
    sub = (f"COLMD:entire atmosphere (considered as a single layer):{fhr} hour fcst:"
           f"aerosol=Particulate organic matter dry:aerosol_size <2.5e-06")
    return await _load_record(cache_dir, date, run, fhr, "vi_smoke", sub)


async def load_rrfs_aod(
    cache_dir: Path, date: str, run: int, fhr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Total aerosol optical thickness/depth (dimensionless), instantaneous
    at fhr, column-integrated across all aerosol species."""
    sub = f"AOTK:entire atmosphere (considered as a single layer):{fhr} hour fcst:"
    return await _load_record(cache_dir, date, run, fhr, "aod", sub)


# EPA PM2.5 AQI breakpoints (24-hr-average table, applied here to RRFS's
# modeled hourly-mean PM2.5 -- the same convention NOAA's own experimental
# smoke/AQI viewers use for a model "instantaneous AQI" display, not a true
# 24-hr NowCast). (C_lo, C_hi, AQI_lo, AQI_hi) in ug/m3.
_AQI_BREAKPOINTS = [
    (0.0,   9.0,   0,   50),
    (9.1,   35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]


def aqi_from_pm25(pm25_ugm3: np.ndarray) -> np.ndarray:
    """Vectorized EPA piecewise-linear PM2.5 -> AQI conversion. Values
    above the top breakpoint (325.4 ug/m3, the worst official category)
    are extrapolated on the same last-segment slope rather than clipped,
    so extreme smoke cores still show a meaningfully higher AQI instead of
    all pinning at 500 -- the colormap's `extend='max'` communicates
    "beyond the standard scale" for those pixels."""
    c = np.clip(pm25_ugm3, 0.0, None)
    aqi = np.full_like(c, np.nan, dtype=float)
    for c_lo, c_hi, aqi_lo, aqi_hi in _AQI_BREAKPOINTS:
        mask = (c >= c_lo) & (c <= c_hi)
        aqi = np.where(mask, aqi_lo + (aqi_hi - aqi_lo) / (c_hi - c_lo) * (c - c_lo), aqi)
    c_lo, c_hi, aqi_lo, aqi_hi = _AQI_BREAKPOINTS[-1]
    beyond = c > c_hi
    aqi = np.where(beyond, aqi_lo + (aqi_hi - aqi_lo) / (c_hi - c_lo) * (c - c_lo), aqi)
    return aqi
