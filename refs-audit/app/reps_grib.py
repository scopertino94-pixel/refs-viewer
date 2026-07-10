"""Full-file GRIB2 fetch/cache for REPS from Environment Canada's Datamart.

REPS publishes one multi-message GRIB2 file per (variable, level, forecast
hour) -- NOT per-member, and with NO .idx sidecar -- so unlike REFS/HREF's
byte-range fetch (grib_range.py), the only option here is a full-file
download, decoded once and cached. This mirrors the shape of REFS's
decoded-field disk cache (field_persist.py) but the fetch primitive itself
is a plain full download rather than a byte-range record fetch. Files are
never partial (unlike grib_range.py's partial-cache-plus-sidecar scheme) --
either the whole file is on disk, or it isn't.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from .reps_data import DD_BASE, client


def reps_grib_url(date: str, run: int, var: str, level: str, fhr: int) -> str:
    return (f"{DD_BASE}/{run:02d}/{fhr:03d}/"
            f"{date}T{run:02d}Z_MSC_REPS_{var}_{level}_RLatLon0.09x0.09_PT{fhr:03d}H.grib2")


def reps_grib_path(cache_dir: Path, date: str, run: int,
                    var: str, level: str, fhr: int) -> Path:
    fname = f"MSC_REPS_{var}_{level}_PT{fhr:03d}H.grib2"
    return cache_dir / "reps" / date / f"{run:02d}" / fname


_download_locks: dict[Path, asyncio.Lock] = {}


def _lock_for(path: Path) -> asyncio.Lock:
    lock = _download_locks.get(path)
    if lock is None:
        lock = asyncio.Lock()
        _download_locks[path] = lock
    return lock


async def ensure_reps_file_cached(
    cache_dir: Path, date: str, run: int, var: str, level: str, fhr: int,
) -> Path | None:
    """Download the full multi-message file once, cache to disk."""
    p = reps_grib_path(cache_dir, date, run, var, level, fhr)
    if p.exists() and p.stat().st_size > 0:
        return p

    p.parent.mkdir(parents=True, exist_ok=True)
    async with _lock_for(p):
        if p.exists() and p.stat().st_size > 0:      # re-check post-lock
            return p
        url = reps_grib_url(date, run, var, level, fhr)
        tmp = p.with_suffix(p.suffix + ".tmp")
        try:
            async with client().stream("GET", url, timeout=90.0) as r:
                if r.status_code != 200:
                    return None
                with open(tmp, "wb") as f:
                    async for chunk in r.aiter_bytes(1 << 16):
                        f.write(chunk)
        except httpx.HTTPError:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(p)
        return p
