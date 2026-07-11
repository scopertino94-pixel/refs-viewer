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
