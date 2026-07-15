"""Byte-range GRIB2 fetch for RRFS air-quality fields, by .idx text match.

Same idea as app/grib_range.py (REFS) and app/reps_grib.py (REPS), but a
third, again-different shape: RRFS's CONUS "2dfld" file is a SINGLE big
GRIB2 file (no member concept, no ensemble) with ~150+ records: surface
fields, smoke/dust mass density at several aerosol-type/size bins, column-
integrated smoke/dust, and AOD, all in one file per (date, run, fhr).

Rather than filtering by GRIB2 keys (aerosolType / aerosol size interval),
this fetches by matching the .idx line's own descriptive text -- the idx
already spells out the aerosol type and size bin in plain text (e.g.
"aerosol=Particulate organic matter dry:aerosol_size <2.5e-06"), so a
substring match against that text unambiguously selects the record we
want, then a single HTTP Range request pulls just those bytes. The result
is a complete, valid single-message GRIB2 file cfgrib can open directly
with no filter_by_keys needed.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from . import rrfs_aq_data as data

# Deliberately NOT rrfs_aq_data.client()'s module-level singleton: each
# recipe method wraps its work in its own asyncio.run() (a fresh event
# loop every call), but a render worker is long-lived and handles many
# renders over its life. A client bound to whichever loop created it
# first silently breaks on every later asyncio.run() call in that same
# process -- the exact bug already hit and fixed for REPS
# (app/reps_grib.py). rrfs_aq_data.client() itself is fine as-is; it's
# only ever used from the main FastAPI event loop (persistent for the
# app's life) for cycle discovery, not from these worker-process calls.


async def _fetch_idx_lines(date: str, run: int, fhr: int) -> list[str] | None:
    url = data.grib_idx_url(date, run, fhr)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=8.0),
                                     follow_redirects=True) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return None
        return r.text.strip().splitlines()
    except Exception:
        return None


def _find_record_range(lines: list[str], substring: str) -> tuple[int, int | None] | None:
    """Return (start_byte, end_byte_exclusive_or_None) for the first idx
    line whose descriptive text (VAR:LEVEL:FCST_DESC[:extras]) contains
    `substring`. end is None for the last record in the file (fetch to EOF)."""
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 4:
            continue
        offset = int(parts[1])
        rest = ":".join(parts[3:])
        if substring in rest:
            if i + 1 < len(lines):
                next_parts = lines[i + 1].split(":")
                next_offset = int(next_parts[1])
            else:
                next_offset = None
            return offset, next_offset
    return None


async def ensure_rrfs_aq_record_cached(
    cache_dir: Path, date: str, run: int, fhr: int,
    field_key: str, idx_substring: str,
) -> Path | None:
    """Fetch (if needed) the single GRIB2 record matching idx_substring,
    caching it at cache_dir/rrfs_aq/{date}/{run}/{field_key}_f{fhr:03d}.grib2.
    Returns the cached path, or None if the cycle/fhr/field doesn't exist.
    """
    dest = cache_dir / "rrfs_aq" / date / f"{run:02d}" / f"{field_key}_f{fhr:03d}.grib2"
    if dest.exists():
        return dest

    lines = await _fetch_idx_lines(date, run, fhr)
    if lines is None:
        return None
    rec = _find_record_range(lines, idx_substring)
    if rec is None:
        print(f"[rrfs_aq_grib] no idx match for {field_key!r} "
              f"(substring={idx_substring!r}) at {date} {run:02d}z F{fhr:03d}",
              flush=True)
        return None
    start, end = rec

    url = f"{data.S3_BASE}/{data.S3_PREFIX.format(date=date, run=run)}/{data.FNAME_T.format(run=run, fhr=fhr)}"
    headers = {"Range": f"bytes={start}-{end - 1}"} if end is not None else {"Range": f"bytes={start}-"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0),
                                     follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
        if r.status_code not in (200, 206):
            return None
    except Exception as e:
        print(f"[rrfs_aq_grib] fetch failed for {field_key} F{fhr:03d}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return dest
