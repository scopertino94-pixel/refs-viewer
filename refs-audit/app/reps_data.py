"""REPS metadata helpers: cycle discovery, fhour availability.

Probes Environment Canada's public Datamart for available REPS cycles,
mirroring the structure of href_data.py. REPS is a genuinely independent
model from REFS/HREF -- different agency (ECCC not NOAA), different grid
(10km rotated-pole, Canada+US), different variable set (2m temp/RH/wind,
precip-type, soil, radiation -- no convective-allowing fields at all).

REPS files live at:
  https://dd.weather.gc.ca/today/ensemble/reps/10km/grib2/{run:02d}/{fhr:03d}/
     {date}T{run:02d}Z_MSC_REPS_{var}_{level}_RLatLon0.09x0.09_PT{fhr:03d}H.grib2

Unlike REFS/HREF, there are no per-member files and no .idx sidecars --
each file bundles all 21 ensemble members (1 control + 20 perturbed) as
separate GRIB2 messages. See reps_grib.py for the fetch/cache strategy.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import httpx

DD_BASE = "https://dd.weather.gc.ca/today/ensemble/reps/10km/grib2"
REPS_RUNS = (0, 6, 12, 18)
MAX_FHOUR = 72
FHOUR_STEP = 3

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=8.0),
            follow_redirects=True,
        )
    return _client


def cycle_label(date: str, run: int) -> str:
    dt = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
    return f"{dt.strftime('%a %d %b')} {run:02d}Z"


# A variable confirmed present at every published forecast hour -- cheap
# (HEAD-only, no body transfer) existence probe for a candidate cycle.
_PROBE_VAR = "TMP_AGL-2m"


def _probe_url(date: str, run: int, fhr: int = 6) -> str:
    return (f"{DD_BASE}/{run:02d}/{fhr:03d}/"
            f"{date}T{run:02d}Z_MSC_REPS_{_PROBE_VAR}_RLatLon0.09x0.09_PT{fhr:03d}H.grib2")


async def _probe(date: str, run: int) -> bool:
    try:
        r = await client().head(_probe_url(date, run), timeout=10.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


_cycles_cache: tuple[float, list[dict]] | None = None
_CYCLES_TTL = 300  # REPS updates 4x/day -- cache longer than REFS/HREF's churn


async def list_recent_cycles(n_back: int = 4) -> list[dict]:
    """Most recent n available REPS cycles (HEAD-probed). Cached for 300 s.

    ECCC's /today/ path typically only retains the most recent cycle or two
    (unlike REFS/HREF's rolling multi-day window), so this may return fewer
    than n_back even when the probe logic is correct.
    """
    global _cycles_cache
    now = time.monotonic()
    if _cycles_cache and now - _cycles_cache[0] < _CYCLES_TTL:
        return _cycles_cache[1]

    import logging
    _log = logging.getLogger("refs-viewer.reps")

    cur = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cur = cur.replace(hour=(cur.hour // 6) * 6)

    candidates: list[tuple[str, int]] = []
    for i in range(n_back + 4):
        c = cur - timedelta(hours=6 * i)
        candidates.append((c.strftime("%Y%m%d"), c.hour))

    sem = asyncio.Semaphore(4)

    async def bounded(date: str, run: int):
        async with sem:
            ok = await _probe(date, run)
            return (date, run, ok)

    results = await asyncio.gather(*(bounded(d, r) for d, r in candidates))
    available = [(d, r) for d, r, ok in results if ok][:n_back]
    _log.info("reps cycles: found %d available: %s",
              len(available),
              [(d, f"{r:02d}z") for d, r in available[:4]])
    out = [
        {"date": d, "run": r, "label": cycle_label(d, r), "max_fhour": MAX_FHOUR}
        for d, r in available
    ]
    _cycles_cache = (now, out)
    return out


async def find_latest_run() -> tuple[str, int] | None:
    cycles = await list_recent_cycles(n_back=2)
    return (cycles[0]["date"], cycles[0]["run"]) if cycles else None
