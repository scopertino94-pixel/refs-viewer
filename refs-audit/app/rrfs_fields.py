"""RRFS operational field loading: generic single-record pull + a few
derived (multi-record) fields, plus the EPA AQI conversion.

REFS/HREF's own ensemble-post files carry a much narrower field set than
RRFS's own raw deterministic output (confirmed by direct .idx inspection --
e.g. no MASSDEN/COLMD/AOTK smoke/dust/AOD records at all, and RRFS's own
CAPE/CIN/UH/shear/etc. layers don't match REFS's ensemble-post shapes
either). See app/rrfs_data.py's module docstring for the file-family
gotcha -- always the "3km...conus" file, never the "2p5km...hi/pr"
Hawaii/Puerto Rico nests.

Most fields here are a SINGLE deterministic value (no ensemble, no member
loop) decoded from one byte-range-fetched GRIB2 record -- the simplest
loader shape of any data source in this app. `load_rrfs_generic` covers
that whole class (used by refs_core.py's `_rrfs_field` recipe, which
builds the idx substring from the product's own `rrfs_idx_tmpl`, so no new
loader function is needed per field -- new severe/storm-attribute/fire
products are just new product-registry entries, not new Python). A couple
of fields (bulk shear) need 2 records combined and get their own loader.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from .rrfs_grib import ensure_rrfs_record_cached

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
    family: str = "2dfld",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ckey = (date, run, fhr, field_key)
    if ckey in _decoded_cache:
        return _decoded_cache[ckey]

    path = await ensure_rrfs_record_cached(cache_dir, date, run, fhr, field_key, idx_substring, family)
    if path is None:
        return None

    result = await asyncio.to_thread(_decode_sync, path)

    if len(_decoded_cache) >= _DECODED_CACHE_MAX:
        _decoded_cache.pop(next(iter(_decoded_cache)))
    _decoded_cache[ckey] = result
    return result


async def load_rrfs_generic(
    cache_dir: Path, date: str, run: int, fhr: int, field_key: str, idx_substring: str,
    family: str = "2dfld",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Public entry point for the generic single-record `rrfs_field` recipe
    -- every Severe/Storm Attributes/Fire product goes through this, with
    the product registry supplying `field_key`/`idx_substring`/`family`."""
    return await _load_record(cache_dir, date, run, fhr, field_key, idx_substring, family)


async def load_rrfs_shear(
    cache_dir: Path, date: str, run: int, fhr: int, layer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Bulk shear magnitude (m/s) for a given AGL layer (e.g. "0-1000 m
    above ground" or "0-6000 m above ground") -- RRFS publishes the U/V
    components (VUCSH/VVCSH) separately, so this fetches both and combines
    sqrt(u**2 + v**2). Lat/lon grid is identical between the two records
    (same file, same domain) so either's is used for both."""
    u_sub = f"VUCSH:{layer}:{fhr} hour fcst:"
    v_sub = f"VVCSH:{layer}:{fhr} hour fcst:"
    u_key = f"vucsh_{layer.split(' ')[0]}"
    v_key = f"vvcsh_{layer.split(' ')[0]}"
    u = await _load_record(cache_dir, date, run, fhr, u_key, u_sub)
    if u is None:
        return None
    v = await _load_record(cache_dir, date, run, fhr, v_key, v_sub)
    if v is None:
        return None
    u_data, lat2d, lon2d = u
    v_data, _, _ = v
    return np.sqrt(u_data ** 2 + v_data ** 2), lat2d, lon2d


# Smoke/VI-smoke/AOD/AQI (the original 4 Air Quality products) now go
# through the same generic `rrfs_field`/load_rrfs_generic path as every
# other product below -- their idx substrings live in app/rrfs_products.py
# as `rrfs_idx_tmpl` strings instead of one-off Python functions here.
# AQI's PM2.5->AQI step is just its product's `convert` lambda calling
# aqi_from_pm25() below, same mechanism every other product's unit
# conversion (K->F, m->kft, etc.) already uses.

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
