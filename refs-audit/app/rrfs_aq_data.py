"""RRFS air-quality metadata helpers: cycle discovery, fhour availability.

RRFS (the deterministic model REFS's ensemble is built from) publishes its
own smoke/dust/AOD fields -- REFS/HREF's ensemble-post files do NOT carry
them (confirmed by direct .idx inspection: REFS's mean/prob/pmmn/lpmm/sprd
files have no MASSDEN/COLMD/AOTK records at all, only VIS). So air quality
needs RRFS's raw deterministic output as its own data source, same as REPS
is its own source -- not folded into the REFS/HREF model toggle.

RRFS runs HOURLY (00-23z), unlike REFS's 6-hourly cadence, with a run-
dependent max forecast hour: 84h at the four synoptic hours (00/06/12/18z),
18h at every other hour (confirmed empirically against the live S3 bucket).

Files live at (same S3 bucket as REFS, different prefix):
  https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.{date}/{run:02d}/
     rrfs.t{run:02d}z.2dfld.3km.f{fhr:03d}.conus.grib2[.idx]

IMPORTANT gotcha (found the hard way): the SAME "2dfld" file family also
publishes small Hawaii ("hi") and Puerto Rico ("pr") nest files at 2.5km
resolution under a *different* filename pattern
(`2dfld.2p5km.f{fhr}.hi/pr.grib2`) -- these decode fine and look
superficially plausible (valid MASSDEN/COLMD/AOTK records) but cover the
wrong domain entirely (Hawaii's lat/lon box, tiny ~321x225 grid). The
correct CONUS file is `2dfld.3km.f{fhr}.conus.grib2` (Lambert grid,
1799x1059, the same domain HRRR/RAP use) -- always use that one.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import httpx

S3_BASE = "https://noaa-rrfs-pds.s3.amazonaws.com"
S3_PREFIX = "rrfs_a/rrfs.{date}/{run:02d}"
FNAME_T = "rrfs.t{run:02d}z.2dfld.3km.f{fhr:03d}.conus.grib2"

SYNOPTIC_RUNS = (0, 6, 12, 18)
MAX_FHOUR_SYNOPTIC = 84
MAX_FHOUR_OFFHOUR = 18
FHOUR_STEP = 1

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=8.0),
            limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
            follow_redirects=True,
        )
    return _client


def max_fhour_for_run(run: int) -> int:
    return MAX_FHOUR_SYNOPTIC if run in SYNOPTIC_RUNS else MAX_FHOUR_OFFHOUR


def grib_idx_url(date: str, run: int, fhr: int) -> str:
    return (f"{S3_BASE}/{S3_PREFIX.format(date=date, run=run)}/"
            f"{FNAME_T.format(run=run, fhr=fhr)}.idx")


def cycle_label(date: str, run: int) -> str:
    dt = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
    return f"{dt.strftime('%a %d %b')} {run:02d}Z"


async def _probe(date: str, run: int) -> bool:
    """HEAD the F000 .idx -- cheap existence check for the whole cycle."""
    try:
        r = await client().head(grib_idx_url(date, run, 0), timeout=6.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


_cycles_cache: tuple[float, list[dict]] | None = None
_CYCLES_TTL = 90


async def list_recent_cycles(n_back: int = 8) -> list[dict]:
    """Most recent n available RRFS cycles (HEAD-probed). Cached 90s."""
    global _cycles_cache
    now = time.monotonic()
    if _cycles_cache and now - _cycles_cache[0] < _CYCLES_TTL:
        return _cycles_cache[1]

    cur = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    candidates: list[tuple[str, int]] = []
    for i in range(n_back + 6):
        c = cur - timedelta(hours=i)
        candidates.append((c.strftime("%Y%m%d"), c.hour))

    sem = asyncio.Semaphore(6)

    async def bounded(date, run):
        async with sem:
            ok = await _probe(date, run)
            return (date, run, ok)

    results = await asyncio.gather(*(bounded(d, r) for d, r in candidates))
    available = [(d, r) for d, r, ok in results if ok][:n_back]
    out = [
        {"date": d, "run": r, "label": cycle_label(d, r),
         "max_fhour": max_fhour_for_run(r)}
        for d, r in available
    ]
    _cycles_cache = (now, out)
    return out


async def find_latest_run() -> tuple[str, int] | None:
    cycles = await list_recent_cycles(n_back=2)
    return (cycles[0]["date"], cycles[0]["run"]) if cycles else None
