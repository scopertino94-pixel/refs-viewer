"""FastAPI app for the REFS-Viewer.

Endpoints:
  GET /                                         → static frontend
  GET /api/cycles                               → recent REFS cycles
  GET /api/catalog                              → product tabs + regions + palettes
  GET /api/cycle-status/{date}/{run}            → fhours available for a cycle
  GET /api/tile/{date}/{run}/{pid}/{fhr}.png    → rendered PNG (cached)
  GET /api/meta/{date}/{run}/{pid}/{fhr}        → meta for current tile
  GET /api/latest                               → newest published cycle
"""
from __future__ import annotations

import asyncio
import io
import logging
import multiprocessing as _mp
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone

# RENDER_WORKERS controls the render pool size + flavor:
#   - 1 (default fallback)        : single-threaded — guaranteed stable
#   - 2+ (default when env unset) : spawn-based ProcessPoolExecutor
# Set BEFORE creating any executor and BEFORE any concurrent.futures fork
# happens. force=True lets us call this even if a default was set earlier.
RENDER_WORKERS = int(os.environ.get("RENDER_WORKERS", "2"))
if RENDER_WORKERS > 1:
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set elsewhere; spawn was the intent
        pass
from pathlib import Path

import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import refs_core as core              # noqa: E402
from . import cache, refs_data        # noqa: E402
from . import cache_persist           # noqa: E402
from . import field_persist           # noqa: E402
from . import href_data               # noqa: E402
from . import spc_data                # noqa: E402
from . import reps_data                # noqa: E402
from . import reps_products as _reps_products  # noqa: E402
from . import rrfs_aq_data              # noqa: E402
from . import rrfs_aq_products as _rrfs_aq_products  # noqa: E402
from .catalog import (APP_VERSION, PALETTES, THEMES, catalog,    # noqa: E402
                      regions)
from .grib_range import (ensure_partial_cached,  # noqa: E402
                          ensure_member_partial_cached)
from .href_grib import ensure_href_partial_cached  # noqa: E402
from .idx_match import match_for, member_match_for  # noqa: E402
from .needed_records import needed_matches      # noqa: E402
from .render import render as render_png  # noqa: E402
from .render import probe_value as _probe_value_worker  # noqa: E402
from .render import probe_series as _probe_series_worker  # noqa: E402
from .render import unproject_box as _unproject_worker  # noqa: E402

# refs_core's downloaded-grib cache path (matches DEFAULT_LOCAL)
_REFS_CACHE = Path(core.DEFAULT_LOCAL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
# httpx/httpcore log every single S3/NOMADS byte-range request at INFO. A
# member-mean tile alone fires ~40 of them, so this flooded the Space logs and
# pushed real startup/prewarm lines out of view within seconds. Quiet them to
# WARNING — the app's own INFO logs (prewarm, cache persist, render timings)
# stay visible. Override with HTTPX_LOG_LEVEL if you ever need the firehose.
_httpx_level = os.environ.get("HTTPX_LOG_LEVEL", "WARNING").upper()
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(_httpx_level)
log = logging.getLogger("refs-viewer")

def _compute_render_hash() -> str:
    """Hash files whose contents change pixel output, so the tile-cache-bust
    ``?v=`` token only flips when renders would actually differ — unrelated
    deploys (CSS tweaks, etc.) leave the browser tile cache intact."""
    h = hashlib.sha256()
    # extra_products.py defines product params (levels, smoothing, colors,
    # overlays) that change pixel output — it MUST be part of this hash or
    # product-def tweaks serve stale cached tiles forever.
    for rel in ("refs_core.py", "app/render.py", "app/extra_products.py"):
        try:
            h.update((ROOT / rel).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


def _compute_asset_hash() -> str:
    """Hash files used for the frontend page itself. Used to bust the
    browser cache for app.js / style.css when the UI changes, without
    invalidating the (much larger) tile cache. Includes the render-hash
    inputs too so a render-code change still flips JS/CSS in case the JS
    references a new endpoint shape."""
    h = hashlib.sha256()
    for rel in ("refs_core.py", "app/render.py",
                "static/app.js", "static/style.css", "static/index.html"):
        try:
            h.update((ROOT / rel).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


BUILD_ID = _compute_render_hash() or uuid.uuid4().hex[:8]
ASSET_ID = _compute_asset_hash() or BUILD_ID
# Stamp the on-disk tile cache with the render-code version so a render change
# (like the SPC eccodes fix) invalidates stale tiles on persistent /data
# instead of serving images produced by the old code.
cache.VERSION = BUILD_ID
STATIC = ROOT / "static"
STARTED_AT = int(time.time())

app = FastAPI(title="REFS-Viewer", docs_url="/api/docs")
# Compress text payloads (JSON catalog, app.js, style.css, index HTML) only.
# A content-type-aware middleware — the stock GZipMiddleware re-gzipped the
# already-compressed WebP tiles, wasting CPU on every serve for ~0% gain.
from .gzip_mw import SelectiveGZipMiddleware  # noqa: E402
app.add_middleware(SelectiveGZipMiddleware, minimum_size=1024)


@app.get("/api/health")
async def health():
    return {"ok": True, "build": BUILD_ID, "version": APP_VERSION}


@app.get("/api/version")
async def version():
    return {"build": BUILD_ID, "version": APP_VERSION, "started_at": STARTED_AT}


@app.get("/api/catalog")
async def catalog_endpoint():
    return {
        "tabs": catalog(),
        "regions": regions(),
        "palettes": PALETTES,
        "themes": THEMES,
        "max_fhour": refs_data.MAX_FHOUR,
        "href_max_fhour": href_data.MAX_FHOUR,
        "runs": list(refs_data.REFS_RUNS),
        "href_runs": list(href_data.HREF_RUNS),
    }


@app.get("/api/cycles")
async def cycles_endpoint():
    try:
        return await refs_data.list_recent_cycles()
    except Exception as e:
        log.exception("cycles endpoint failed")
        raise HTTPException(502, f"S3 listing failed: {type(e).__name__}: {e}")


@app.get("/api/latest")
async def latest_endpoint():
    r = await refs_data.find_latest_run()
    if not r:
        raise HTTPException(404, "No recent REFS run found")
    date, run = r
    return {"date": date, "run": run,
            "label": refs_data.cycle_label(date, run)}


@app.get("/api/href-cycles")
async def href_cycles_endpoint():
    try:
        return await href_data.list_recent_cycles()
    except Exception as e:
        log.exception("href-cycles endpoint failed")
        raise HTTPException(502, f"NOMADS listing failed: {type(e).__name__}: {e}")


@app.get("/api/href-latest")
async def href_latest_endpoint():
    r = await href_data.find_latest_run()
    if not r:
        raise HTTPException(404, "No recent HREF run found")
    date, run = r
    return {"date": date, "run": run,
            "label": href_data.cycle_label(date, run)}


@app.get("/api/cycle-status/{date}/{run}")
async def cycle_status(date: str, run: int):
    try:
        fhrs = await refs_data.list_available_fhours(date, run)
    except Exception as e:
        log.exception("cycle-status failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    # REFS publishes hourly forecast files F001..F060 — there is no F000
    # analysis file. So "expected" must not blindly include F000, otherwise
    # the count tops out at 60/61 and the cycle never reads as complete.
    # The run is done once the terminal hour (MAX_FHOUR) is posted; REFS
    # writes sequentially, so that is the reliable completion signal.
    mx = max(fhrs) if fhrs else -1
    has_f000 = 0 in fhrs
    expected = refs_data.MAX_FHOUR + (1 if has_f000 else 0)
    return {
        "date": date, "run": run,
        "available": fhrs,
        "max_fhour": refs_data.MAX_FHOUR,
        "count": len(fhrs),
        "expected": expected,
        "complete": mx >= refs_data.MAX_FHOUR,
    }


@app.get("/api/spc-status/{date}/{run}/{pid}")
async def spc_status(date: str, run: int, pid: str):
    """Available forecast hours for one SPC calibrated-guidance product in
    the given cycle. The frontend uses this to drive the timeline for SPC
    products (whose fhr ranges differ from REFS/HREF and vary by product)."""
    prod = core.PRODUCTS.get(pid)
    if not prod or prod.get("source") != "spc_post":
        raise HTTPException(404, "not an SPC product")
    try:
        subdir, token = prod["ftype"].split("|", 1)
    except ValueError:
        raise HTTPException(500, "malformed SPC ftype")
    try:
        fhrs = await spc_data.list_available_fhours(date, run, subdir, token)
    except Exception as e:
        log.exception("spc-status failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    return {
        "date": date, "run": run, "pid": pid,
        "available": fhrs,
        "count": len(fhrs),
        "min_fhr": int(prod.get("min_fhr", 0)),
    }


@app.get("/api/reps-cycles")
async def reps_cycles():
    """Recent available REPS cycles. Separate from /api/cycles (REFS) --
    REPS is a genuinely independent data source (Environment Canada, not
    NOAA) with its own cycle availability, unrelated to REFS/HREF's."""
    try:
        cycles = await reps_data.list_recent_cycles()
    except Exception as e:
        log.exception("reps-cycles failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    return {"cycles": cycles}


@app.get("/api/reps-status/{date}/{run}/{pid}")
async def reps_status(date: str, run: int, pid: str):
    """Available forecast hours for a REPS product in the given cycle.

    REPS publishes atomically per cycle (000-072 by 3h, all at once) rather
    than incrementally like REFS -- so this just checks whether the cycle
    itself is currently available and, if so, returns the full expected
    fhr list, unlike /api/spc-status's per-product probing.
    """
    prod = core.PRODUCTS.get(pid)
    if not prod or prod.get("source") != "reps":
        raise HTTPException(404, "not a REPS product")
    try:
        cycles = await reps_data.list_recent_cycles()
    except Exception as e:
        log.exception("reps-status failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    exists = any(c["date"] == date and c["run"] == run for c in cycles)
    fhrs = list(range(0, reps_data.MAX_FHOUR + 1, reps_data.FHOUR_STEP)) if exists else []
    return {
        "date": date, "run": run, "pid": pid,
        "available": fhrs,
        "count": len(fhrs),
        "min_fhr": 0,
    }


@app.get("/api/rrfsaq-cycles")
async def rrfsaq_cycles():
    """Recent available RRFS air-quality cycles. RRFS is hourly (unlike
    REFS's 6-hourly / REPS's 4x-daily cadence) with a run-dependent max
    forecast hour -- see app/rrfs_aq_data.py."""
    try:
        cycles = await rrfs_aq_data.list_recent_cycles()
    except Exception as e:
        log.exception("rrfsaq-cycles failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    return {"cycles": cycles}


@app.get("/api/rrfsaq-status/{date}/{run}/{pid}")
async def rrfsaq_status(date: str, run: int, pid: str):
    """Available forecast hours for an RRFS-AQ product in the given cycle.

    Like REPS's status endpoint, this checks whether the cycle exists and
    returns the full expected fhr list for that run (18h off-hour, 84h at
    synoptic hours) rather than per-fhr probing -- RRFS posts hours in
    order fairly quickly once the run starts, so the small risk of showing
    a not-yet-posted final hour as "available" is an acceptable tradeoff
    for a much simpler endpoint.
    """
    prod = core.PRODUCTS.get(pid)
    if not prod or prod.get("source") != "rrfs_aq":
        raise HTTPException(404, "not an RRFS-AQ product")
    try:
        cycles = await rrfs_aq_data.list_recent_cycles()
    except Exception as e:
        log.exception("rrfsaq-status failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    match = next((c for c in cycles if c["date"] == date and c["run"] == run), None)
    max_fhour = match["max_fhour"] if match else 0
    min_fhr = int(prod.get("min_fhr", 0))
    fhrs = list(range(min_fhr, max_fhour + 1, rrfs_aq_data.FHOUR_STEP)) if match else []
    return {
        "date": date, "run": run, "pid": pid,
        "available": fhrs,
        "count": len(fhrs),
        "min_fhr": min_fhr,
    }


# Render pool. With RENDER_WORKERS>1 we use a spawn-based ProcessPoolExecutor
# so each worker is a fresh Python interpreter (avoids the fork-inherited
# threading-lock deadlock that hung us on the first parallel-render attempt).
# Set RENDER_WORKERS=1 in HF Space settings as an emergency rollback to the
# single-thread path.
if RENDER_WORKERS > 1:
    _render_executor = ProcessPoolExecutor(max_workers=RENDER_WORKERS)
    log.info("render pool: ProcessPoolExecutor(spawn, max_workers=%d)", RENDER_WORKERS)
else:
    _render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="refs-render")
    log.info("render pool: ThreadPoolExecutor(max_workers=1) — single-threaded mode")
_render_sem = asyncio.Semaphore(max(1, RENDER_WORKERS))

# Hard timeout per render (sec). If a worker hangs, we surface it to the
# client as a 504 instead of letting the request stall forever.
RENDER_TIMEOUT_SECS = float(os.environ.get("RENDER_TIMEOUT_SECS", "180"))

# How long a "No data" placeholder tile is served before re-attempting the
# render. Short on purpose: the usual cause is GRIB records not posted *yet*.
NODATA_TTL_SECS = float(os.environ.get("NODATA_TTL_SECS", "300"))

# Timestamp of the most recent *user-initiated* tile/gif/png request. The
# prewarm render loop pauses while users are active so a 48-fhr warm pass
# never steals a render worker from a live click.
_last_user_activity = 0.0


def _mark_user_activity() -> None:
    global _last_user_activity
    _last_user_activity = time.time()


# Seconds of quiet required before a prewarm render proceeds.
PREWARM_IDLE_SECS = float(os.environ.get("PREWARM_IDLE_SECS", "8"))

# The cache key a DEFAULT user tile request resolves to: _serve_tile appends
# the counties/cities/regions toggle suffix (all off by default in the UI).
# Prewarm MUST render under this exact key — the bare "CONUS" key it used
# before never matched a real request, so every prewarmed tile was orphaned.
_PREWARM_CACHE_KEY = "CONUS_c0y0r0"


async def _prefetch_member_records(date: str, run: int, fhr: int,
                                   pid: str, prod: dict) -> None:
    """Byte-range fetch the records each member needs for a `member_mean` /
    `storm_motion` recipe. Each member is its own GRIB file at a separate
    bucket key, so we issue parallel partial fetches one per member.
    """
    recipe = prod.get('recipe')
    member_product = prod.get('member_product', '2dfld')
    n = prod.get('n_members', 5)
    # Determine the list of .idx substrings we need from each member's file.
    if recipe == 'storm_motion':
        var_keys = ('ustm_6km', 'vstm_6km')
    else:
        var_keys = (prod.get('var'),)
    # Resolve per-fhr step (fixed `step` or callable `step_from_fhr`).
    step = prod.get('step')
    if step is None and prod.get('step_from_fhr'):
        try:
            step = prod['step_from_fhr'](fhr)
        except Exception:
            step = None
    # `step` here is the *period length* in hours for accumulation/averages
    # (e.g. step=1 → ``5-6 hour ave fcst``). If the product carries an
    # 'A-B' string form, parse out the period length.
    if isinstance(step, str) and '-' in step:
        try:
            a, b = (int(x) for x in step.split('-'))
            step = b - a
        except ValueError:
            step = None
    acc_type = prod.get('acc_type', 'acc')
    # Windowed products (window_h, SPC "4-hr max" convention) need the hourly
    # record from each contributing fhr's member file, not just the frame's.
    window = int(prod.get('window_h', 1) or 1)
    fhrs = [w for w in range(fhr - window + 1, fhr + 1) if w >= 1] or [fhr]
    tasks = []
    for w in fhrs:
        matches: list[str] = []
        for v in var_keys:
            if not v:
                continue
            # member_match_for knows which 2dfld vars are hourly-max records
            # (WIND/HAIL/MXUPHL/...) — plain match_for builds an instantaneous
            # marker for those, matches nothing, and the partial file ends up
            # without the record the renderer needs.
            m = member_match_for(v, level=prod.get('level'), fhr=w,
                                 step=step,
                                 acc_type=acc_type if step is not None else None)
            if m is not None:
                matches.append(m)
        if not matches:
            continue
        tasks += [
            ensure_member_partial_cached(_REFS_CACHE, date, run, mem,
                                          member_product, w, matches)
            for mem in range(1, n + 1)
        ]
    if not tasks:
        return
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if not isinstance(r, Exception) and r is not None)
    if ok < len(tasks):
        log.info("prefetch members: %s F%03d %d/%d member files fetched",
                 pid, fhr, ok, len(tasks))


async def _prefetch_records(date: str, run: int, fhr: int, pid: str,
                            model: str = "refs") -> None:
    """Byte-range fetch only the GRIB records this product needs.

    Silent no-op on any failure or unmapped variable — refs_core's full-file
    download path is the safety net.
    """
    try:
        prod = core.PRODUCTS[pid]
        recipe = prod.get('recipe')
        if model == "href":
            # HREF has no member files; skip member-based products silently.
            if recipe in ('member_mean', 'storm_motion', 'member_prob'):
                return
            matches = needed_matches(prod, fhr)
            if not matches:
                return
            tasks = [
                ensure_href_partial_cached(_REFS_CACHE, date, run, ftype, fhr, ms)
                for ftype, ms in matches.items()
                if ms
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.warning("href prefetch: %s F%03d: %s",
                                pid, fhr, type(r).__name__)
        else:
            # Member-based recipes have their own bucket layout — fetch from the
            # per-member files in parallel. Paintballs/stamps are member-based
            # too: without this they'd land on whatever partial member file an
            # earlier product left behind (which lacks their record) and render
            # as "No data". Their prob overlays come from the standard enspost
            # ftypes, so byte-range those as well (needed_matches returns them).
            if recipe in ('member_mean', 'storm_motion', 'member_prob',
                          'paintball', 'stamps'):
                member_task = _prefetch_member_records(date, run, fhr, pid, prod)
                ov_tasks = []
                if recipe in ('paintball', 'stamps'):
                    # Windowed overlays (window_h) draw on hourly prob records
                    # from each contributing fhr's enspost file.
                    _w = int(prod.get('window_h', 1) or 1)
                    _whrs = ([w for w in range(fhr - _w + 1, fhr + 1) if w >= 1]
                             or [fhr])
                    for w in _whrs:
                        ov_matches = needed_matches(prod, w)
                        ov_tasks += [
                            ensure_partial_cached(_REFS_CACHE, date, run,
                                                  ftype, w, ms)
                            for ftype, ms in (ov_matches or {}).items() if ms
                        ]
                results = await asyncio.gather(member_task, *ov_tasks,
                                               return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        log.warning("member prefetch: %s F%03d: %s",
                                    pid, fhr, type(r).__name__)
                return
            # qpf_sum needs the 6-hr APCP from N contributing fhrs, not just
            # the current one — prefetch all of them plus overlay records
            # in parallel, then return (needed_matches is not used).
            if recipe == 'qpf_sum':
                period = prod.get('sum_period', 6)
                n_steps = prod.get('n_steps', 4)
                total_h = period * n_steps
                tasks = []
                if fhr >= total_h and fhr % period == 0:
                    for i in range(n_steps):
                        f = fhr - period * (n_steps - 1 - i)
                        tasks.append(ensure_partial_cached(
                            _REFS_CACHE, date, run, 'mean', f,
                            [f'APCP:surface:{f - period}-{f} hour acc fcst']
                        ))
                # Also prefetch overlay records at the current fhr.
                ov_matches = needed_matches(prod, fhr)
                if ov_matches:
                    tasks += [
                        ensure_partial_cached(_REFS_CACHE, date, run, ft, fhr, ms)
                        for ft, ms in ov_matches.items() if ms
                    ]
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for r in results:
                        if isinstance(r, Exception):
                            log.warning("qpf_sum prefetch: %s F%03d: %s",
                                        pid, fhr, type(r).__name__)
                return
            # Windowed NP products need the hourly prob record from each
            # contributing fhr's enspost file, not just the frame's.
            if recipe == 'prob_window':
                _w = int(prod.get('window_h', 4) or 1)
                whrs = ([w for w in range(fhr - _w + 1, fhr + 1) if w >= 1]
                        or [fhr])
            else:
                whrs = [fhr]
            tasks = []
            for w in whrs:
                matches = needed_matches(prod, w)
                tasks += [
                    ensure_partial_cached(_REFS_CACHE, date, run, ftype, w, ms)
                    for ftype, ms in (matches or {}).items()
                    if ms
                ]
            if not tasks:
                return
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.warning("prefetch: %s F%03d: %s",
                                pid, fhr, type(r).__name__)
    except Exception:
        log.exception("prefetch failed for %s F%03d (model=%s)", pid, fhr, model)


def _validate(pid: str, region: str, palette: str, theme: str, fhr: int):
    if pid not in core.PRODUCTS:
        raise HTTPException(404, f"Unknown product: {pid}")
    if region not in core.REGIONS:
        raise HTTPException(404, f"Unknown region: {region}")
    if palette not in core.PALETTES:
        raise HTTPException(400, f"Unknown palette: {palette}")
    if theme not in core.THEMES:
        raise HTTPException(400, f"Unknown theme: {theme}")
    if fhr < 0 or fhr > refs_data.MAX_FHOUR:
        raise HTTPException(404, f"fhr {fhr} out of range")


def _parse_bbox(s: str) -> tuple[float, float, float, float] | None:
    """Parse 'lon_min,lat_min,lon_max,lat_max'. Returns None if invalid."""
    if not s:
        return None
    try:
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 4:
            return None
        lon_min, lat_min, lon_max, lat_max = parts
        if not (-180 <= lon_min < lon_max <= 180):
            return None
        if not (-90 <= lat_min < lat_max <= 90):
            return None
        return (lon_min, lat_min, lon_max, lat_max)
    except (TypeError, ValueError):
        return None


def _resolve_sector(region: str, bbox_str: str, sector_name: str):
    """Return (region_arg, bbox_or_None, sector_label, cache_key).

    If bbox is valid, the render bypasses preset regions; otherwise we
    enforce that `region` matches a preset.
    """
    bbox = _parse_bbox(bbox_str)
    if bbox is not None:
        label = (sector_name or "Custom").strip()[:48] or "Custom"
        # cache_key embeds the bbox so different rectangles don't collide
        cache_key = f"bbox:{','.join(f'{v:.3f}' for v in bbox)}"
        return ("CONUS", bbox, label, cache_key)
    if region not in core.REGIONS:
        raise HTTPException(404, f"Unknown region: {region}")
    return (region, None, region, region)


async def _speculative_prefetch(date: str, run: int, pid: str, fhr: int,
                                model: str = "refs") -> None:
    """After a cache miss on fhr, silently prefetch GRIB records for the next
    few frames so those renders only pay plot time, not download + plot time.
    Fires-and-forgets — any failure is intentionally swallowed.
    """
    max_fhr = href_data.MAX_FHOUR if model == "href" else refs_data.MAX_FHOUR
    # Prefetch the next 2 forecast hours (or fewer near the end of the run).
    # Kept small + concurrency-gated so look-ahead never starves the click the
    # user is actually waiting on.
    ahead = [h for h in range(fhr + 1, min(fhr + 3, max_fhr + 1))]
    if not ahead:
        return
    tasks = [_bg_prefetch_records(date, run, h, pid, model=model) for h in ahead]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        pass  # speculative — never let this surface to the caller


async def _speculative_render(date, run, pid, fhr, region, palette, theme,
                              bbox, sector_label, cache_key,
                              show_counties, show_cities, show_regions,
                              model, member: int = -1):
    """Render the next frames ahead of the client's sequential preload —
    but only while a render worker is actually idle. The per-tile lock in
    _render_to_cache dedupes against the client's own request for the same
    frame, and cache hits return instantly, so over-firing is cheap."""
    max_fhr = href_data.MAX_FHOUR if model == "href" else refs_data.MAX_FHOUR
    for h in (fhr + 1, fhr + 2):
        if h > max_fhr:
            return
        # Never steal a worker from a live click: bail unless the pool has
        # a free slot right now.
        if getattr(_render_sem, "_value", 0) <= 0:
            return
        try:
            await _render_to_cache(date, run, pid, h, region, palette, theme,
                                   bbox, sector_label, cache_key,
                                   show_counties=show_counties,
                                   show_cities=show_cities,
                                   show_regions=show_regions,
                                   model=model, member=member)
        except Exception:
            return  # speculative — never surface


async def _render_to_cache(date, run, pid, fhr, region, palette, theme,
                           bbox, sector_label, cache_key,
                           show_counties: bool = False,
                           show_cities: bool = False,
                           show_regions: bool = False,
                           model: str = "refs", member: int = -1):
    """Internal: ensure a rendered tile is in the cache.

    Returns (path, is_no_data). Real renders are cached permanently;
    "No data" placeholder tiles go to a side path with a short TTL so the
    product recovers as soon as the missing GRIB records get posted —
    a permanently cached placeholder would mask that for the whole run.
    """
    # Prefix cache key with model so REFS and HREF tiles never collide.
    keyed = f"{model}:{cache_key}" if model != "refs" else cache_key
    p = cache.png_path(date, run, pid, keyed, palette, theme, fhr)
    if p.exists():
        return p, False
    nd = p.with_name(f"nodata-{p.name}")
    try:
        if nd.exists() and time.time() - nd.stat().st_mtime < NODATA_TTL_SECS:
            return nd, True
    except OSError:
        pass
    lock = cache.get_lock(date, run, pid, keyed, palette, theme, fhr)
    async with lock:
        if p.exists():
            return p, False
        await _prefetch_records(date, run, fhr, pid, model=model)
        # Speculatively prefetch the next few frames' GRIB records in the
        # background while this frame renders — by the time the user clicks
        # forward the download is already done.
        asyncio.create_task(
            _speculative_prefetch(date, run, pid, fhr, model=model))
        async with _render_sem:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(
                _render_executor,
                render_png, pid, date, run, fhr, region, palette, theme,
                bbox, sector_label, show_counties, show_cities, show_regions,
                model, member,
            )
            try:
                result = await asyncio.wait_for(fut, timeout=RENDER_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                log.error("render timeout %s F%03d after %ss",
                          pid, fhr, RENDER_TIMEOUT_SECS)
                raise HTTPException(504, f"Render timeout for {pid} F{fhr:03d}")
        if result is None:
            raise HTTPException(404, f"Render returned no data for {pid} F{fhr:03d}")
        png, meta = result
        if meta.get("no_data"):
            nd.write_bytes(png)
            return nd, True
        meta = dict(meta)
        meta["sector_label"] = sector_label
        if bbox: meta["bbox"] = list(bbox)
        cache.write(date, run, pid, keyed, palette, theme, fhr, png, meta)
        asyncio.create_task(asyncio.to_thread(cache.prune))
    return p, False


async def _serve_tile(date: str, run: int, pid: str, fhr: int,
                      region: str, palette: str, theme: str,
                      bbox: str, sector_name: str,
                      counties: int = 0, cities: int = 0,
                      regions: int = 0,
                      model: str = "refs", member: int = -1):
    _mark_user_activity()
    if pid not in core.PRODUCTS:
        raise HTTPException(404, f"Unknown product: {pid}")
    if palette not in core.PALETTES:
        raise HTTPException(400, f"Unknown palette: {palette}")
    if theme not in core.THEMES:
        raise HTTPException(400, f"Unknown theme: {theme}")
    max_fhr = href_data.MAX_FHOUR if model == "href" else refs_data.MAX_FHOUR
    if fhr < 0 or fhr > max_fhr:
        raise HTTPException(404, f"fhr {fhr} out of range")
    region_arg, bbox_t, sector_label, cache_key = _resolve_sector(
        region, bbox, sector_name)
    show_counties = bool(counties)
    show_cities = bool(cities)
    show_regions_flag = bool(regions)
    # Encode toggle state into the cache key so toggling counties, cities, or
    # regions produces distinct on-disk tiles instead of stomping a shared file.
    cache_key = f"{cache_key}_c{int(show_counties)}y{int(show_cities)}r{int(show_regions_flag)}"
    # Member browser: a single-member full-size view is a distinct tile from
    # the grid and from every other member, so it needs its own cache slot.
    member = int(member)
    if member >= 0:
        cache_key = f"{cache_key}_m{member:02d}"
    p, no_data = await _render_to_cache(date, run, pid, fhr, region_arg, palette,
                                theme, bbox_t, sector_label, cache_key,
                                show_counties=show_counties,
                                show_cities=show_cities,
                                show_regions=show_regions_flag,
                                model=model, member=member)
    # Stay one step ahead of the client's frame-by-frame preload: render the
    # next frames on the idle worker while this one ships. Fire-and-forget.
    asyncio.create_task(_speculative_render(
        date, run, pid, fhr, region_arg, palette, theme, bbox_t,
        sector_label, cache_key, show_counties, show_cities,
        show_regions_flag, model, member))
    if no_data:
        # Placeholder tile — must not stick in the browser cache, or the
        # product stays "broken" client-side even after the data arrives.
        return FileResponse(p, media_type="image/webp",
                            headers={"Cache-Control": "no-store"})
    # Tile URLs include ?v=BUILD_ID (= content hash of render code); a render-
    # code change invalidates the browser cache automatically. So mark each
    # tile immutable for a year — already-loaded frames are served entirely
    # from browser memory.
    return FileResponse(
        p, media_type="image/webp",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# Tiles are WebP now. Keep the legacy .png route as an alias so any in-flight
# bookmarks / open tabs from before the switch still resolve.
@app.get("/api/tile/{date}/{run}/{pid}/{fhr}.webp")
async def tile_webp(date: str, run: int, pid: str, fhr: int,
                    region: str = "CONUS",
                    palette: str = "Default",
                    theme: str = "dark",
                    bbox: str = "",
                    sector_name: str = "",
                    counties: int = 0, cities: int = 0,
                    regions: int = 0,
                    model: str = "refs", member: int = -1):
    return await _serve_tile(date, run, pid, fhr, region, palette, theme,
                             bbox, sector_name,
                             counties=counties, cities=cities,
                             regions=regions, model=model, member=member)


@app.get("/api/tile/{date}/{run}/{pid}/{fhr}.png")
async def tile_png_alias(date: str, run: int, pid: str, fhr: int,
                         region: str = "CONUS",
                         palette: str = "Default",
                         theme: str = "dark",
                         bbox: str = "",
                         sector_name: str = "",
                         counties: int = 0, cities: int = 0,
                         regions: int = 0,
                         model: str = "refs", member: int = -1):
    return await _serve_tile(date, run, pid, fhr, region, palette, theme,
                             bbox, sector_name,
                             counties=counties, cities=cities,
                             regions=regions, model=model, member=member)


# --------------------------------------------------------------------------
# MRMS overlay (observed reflectivity contours).
#
# Critical perf notes:
#  * The overlay is an *observation* — it does not depend on the REFS
#    product / palette. We cache by (snapshot_minute, region, theme,
#    product) only, so many fhrs (any fhrs whose valid time rounds to the
#    same 2-min MRMS snapshot) share the same on-disk tile.
#  * Render goes through `_render_executor` (the same process pool the
#    REFS tiles use, with the *same* semaphore). This means matplotlib
#    work never holds the FastAPI event-loop's GIL, and MRMS competes
#    fairly with REFS rather than starving it via `asyncio.to_thread`.
# --------------------------------------------------------------------------

# Re-entrant in-flight dedup so a burst of fhr scrubs that all map to the
# same snapshot collapses to a single render.
_MRMS_INFLIGHT: dict[str, asyncio.Future] = {}


def _mrms_render_worker(style, region_key, data, lats, lons,
                        bbox, sector_label, theme,
                        title_product="", valid_dt=None):
    """Process-pool entry point — imported on the worker so this module
    stays cheap to import in the parent. ``style`` selects which render
    flavor to produce:
      contours        — line contours, transparent BG (Filled mode overlay
                        uses REFS underneath for basemap)
      filled          — full NWS palette, transparent BG, no basemap
      filled_basemap  — full NWS palette + standard basemap baked in
                        (Compare / MRMS-only modes use this for a
                        self-contained tile)
    """
    if style == "filled":
        from .mrms import render_overlay_filled
        return render_overlay_filled(region_key, data, lats, lons,
                                     bbox, sector_label, theme,
                                     with_basemap=False)
    if style == "filled_basemap":
        from .mrms import render_overlay_filled
        return render_overlay_filled(region_key, data, lats, lons,
                                     bbox, sector_label, theme,
                                     with_basemap=True,
                                     title_product=title_product,
                                     valid_dt=valid_dt)
    from .mrms import render_overlay
    return render_overlay(region_key, data, lats, lons,
                          bbox, sector_label, theme)


@app.get("/api/mrms/{date}/{run}/{fhr}.webp")
async def mrms_overlay(date: str, run: int, fhr: int,
                       region: str = "CONUS",
                       bbox: str = "",
                       sector_name: str = "",
                       theme: str = "dark",
                       product: str = "refc",
                       style: str = "contours"):
    """Transparent MRMS observed-reflectivity overlay for the given
    forecast frame's valid time.

    ``style`` controls the visual form:
      contours = thin dBZ contour lines (25/40/55) on a transparent BG
      filled   = full NWS dBZ palette, transparent below the bottom stop
    """
    if fhr < 0 or fhr > refs_data.MAX_FHOUR:
        raise HTTPException(404, f"fhr {fhr} out of range")
    region_arg, bbox_t, sector_label, cache_key = _resolve_sector(
        region, bbox, sector_name)
    if theme not in core.THEMES:
        theme = "dark"

    from .mrms import (render_overlay, get_mrms_data, snap_to_2min,
                       PRODUCTS as _MRMS_PRODUCTS)
    if product not in _MRMS_PRODUCTS:
        product = "refc"
    if style not in ("contours", "filled", "filled_basemap"):
        style = "contours"

    from datetime import datetime, timedelta, timezone
    init = datetime.strptime(date, "%Y%m%d").replace(
        hour=run, tzinfo=timezone.utc)
    valid = init + timedelta(hours=fhr)
    now_utc = datetime.now(timezone.utc)

    # Cache key is keyed by the SNAPSHOT minute, not the (date, fhr) pair.
    # All fhrs whose valid time rounds to the same MRMS snapshot share
    # a single on-disk tile — typical 60-h loop drops from ~60 to ~30 tiles.
    snapshot = snap_to_2min(valid) if valid <= now_utc else None
    safe_region = cache_key.replace("/", "_").replace(" ", "_")
    if snapshot is not None:
        snap_tag = snapshot.strftime("%Y%m%dT%H%M")
        fname = f"mrms_{product}_{style}_{snap_tag}_{safe_region}_{theme}.webp"
    else:
        # Future frame — produces an empty overlay; key on the valid minute
        # so the "nothing here yet" answer caches cheaply but expires when
        # obs eventually land.
        fname = (f"mrms_{product}_{style}_future_"
                 f"{valid.strftime('%Y%m%dT%H%M')}_{safe_region}_{theme}.webp")
    cpath = cache.CACHE_DIR / fname
    # Verified if the snapshot is at least 30 min in the past.
    verified = (snapshot is not None
                and (now_utc - snapshot) >= timedelta(minutes=30))

    if cpath.exists():
        return FileResponse(
            cpath, media_type="image/webp",
            headers={"Cache-Control":
                     "public, max-age=31536000, immutable" if verified
                     else "public, max-age=120"},
        )

    # Dedup concurrent requests for the same snapshot+region+theme so a
    # scrub-burst doesn't queue N copies of the same render.
    inflight_key = fname
    pending = _MRMS_INFLIGHT.get(inflight_key)
    if pending is not None:
        overlay = await pending
    else:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        _MRMS_INFLIGHT[inflight_key] = fut
        try:
            if snapshot is None:
                data = lats = lons = None
            else:
                t0 = time.monotonic()
                try:
                    loaded = await get_mrms_data(snapshot, _REFS_CACHE, product)
                except Exception as e:
                    log.warning("MRMS fetch failed snap=%s product=%s: %s",
                                snapshot, product, e)
                    loaded = None
                t_fetch = time.monotonic() - t0
                if loaded is None:
                    data = lats = lons = None
                    log.info("MRMS snap=%s product=%s region=%s: no data (fetch=%.2fs)",
                             snapshot, product, cache_key, t_fetch)
                else:
                    data, lats, lons = loaded
                    log.info("MRMS snap=%s product=%s region=%s: loaded shape=%s (fetch=%.2fs)",
                             snapshot, product, cache_key, data.shape, t_fetch)

            # Route render through the same process pool as REFS tiles
            # so matplotlib work doesn't hold the event-loop GIL and
            # competes fairly under the same semaphore.
            # Friendly product labels for the side-by-side title row.
            _MRMS_LABELS = {
                "refc": "MRMS Composite Reflectivity (QC)",
                "hsr":  "MRMS Seamless Hybrid-Scan Reflectivity",
                "base": "MRMS Base Reflectivity",
            }
            title_product = (_MRMS_LABELS.get(product, "")
                             if style == "filled_basemap" else "")
            title_valid = (valid if style == "filled_basemap" else None)

            t1 = time.monotonic()
            async with _render_sem:
                exec_fut = loop.run_in_executor(
                    _render_executor, _mrms_render_worker,
                    style, region_arg, data, lats, lons,
                    bbox_t, sector_label, theme,
                    title_product, title_valid,
                )
                try:
                    overlay = await asyncio.wait_for(
                        exec_fut, timeout=RENDER_TIMEOUT_SECS)
                except asyncio.TimeoutError:
                    log.error("MRMS render timeout snap=%s product=%s",
                              snapshot, product)
                    fut.set_exception(HTTPException(504, "MRMS render timeout"))
                    raise
            t_render = time.monotonic() - t1
            if overlay is None:
                log.warning("MRMS render returned None snap=%s product=%s",
                            snapshot, product)
                fut.set_exception(HTTPException(500, "MRMS render failed"))
                raise HTTPException(500, "MRMS render failed")
            log.info("MRMS snap=%s product=%s: %d bytes (render=%.2fs)",
                     snapshot, product, len(overlay), t_render)
            if verified:
                try:
                    cpath.write_bytes(overlay)
                except OSError:
                    pass
            fut.set_result(overlay)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            _MRMS_INFLIGHT.pop(inflight_key, None)

    from fastapi.responses import Response
    return Response(
        content=overlay, media_type="image/webp",
        headers={"Cache-Control":
                 "public, max-age=31536000, immutable" if verified
                 else "public, max-age=120"},
    )


# --------------------------------------------------------------------------
# REFS forecast comp-ref CONTOUR overlay (transparent). Distinct from the
# main REFS tile because we want only the line contours, transparent BG,
# sized to MAP_BOX so the frontend can stack it on the MRMS panel in
# compare mode. Cache key is (date, run, fhr, region, theme) — the data
# is fully determined by those, no observation lookups involved.
# --------------------------------------------------------------------------

_REFS_CONTOUR_INFLIGHT: dict[str, asyncio.Future] = {}


def _refs_contour_render_worker(date, run, fhr, region, theme,
                                bbox, sector_label):
    from .render import render_refs_contours
    return render_refs_contours(date, run, fhr, region, theme,
                                bbox, sector_label)


@app.get("/api/refs_contours/{date}/{run}/{fhr}.webp")
async def refs_contour_overlay(date: str, run: int, fhr: int,
                               region: str = "CONUS",
                               bbox: str = "",
                               sector_name: str = "",
                               theme: str = "dark"):
    """REFS composite-reflectivity contour overlay for compare/verify mode.

    Returns a transparent WebP with dashed contour lines at 20/35/50 dBZ
    drawn over the REFS forecast field for the given frame. Sized to the
    standard MAP_BOX so it stacks pixel-for-pixel on the MRMS panel.
    """
    if fhr < 0 or fhr > refs_data.MAX_FHOUR:
        raise HTTPException(404, f"fhr {fhr} out of range")
    region_arg, bbox_t, sector_label, cache_key = _resolve_sector(
        region, bbox, sector_name)
    if theme not in core.THEMES:
        theme = "dark"

    safe_region = cache_key.replace("/", "_").replace(" ", "_")
    fname = (f"refsc_{date}_{run:02d}_F{fhr:03d}_"
             f"{safe_region}_{theme}.webp")
    cpath = cache.CACHE_DIR / fname

    if cpath.exists():
        return FileResponse(
            cpath, media_type="image/webp",
            headers={"Cache-Control":
                     "public, max-age=31536000, immutable"},
        )

    inflight_key = fname
    pending = _REFS_CONTOUR_INFLIGHT.get(inflight_key)
    if pending is not None:
        overlay = await pending
    else:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        _REFS_CONTOUR_INFLIGHT[inflight_key] = fut
        try:
            await _prefetch_records(date, run, fhr, "refc_pmmn_series")
            async with _render_sem:
                exec_fut = loop.run_in_executor(
                    _render_executor, _refs_contour_render_worker,
                    date, run, fhr, region_arg, theme,
                    bbox_t, sector_label,
                )
                try:
                    overlay = await asyncio.wait_for(
                        exec_fut, timeout=RENDER_TIMEOUT_SECS)
                except asyncio.TimeoutError:
                    log.error("REFS contour render timeout %s F%03d", date, fhr)
                    fut.set_exception(HTTPException(504,
                        "REFS contour render timeout"))
                    raise
            if overlay is None:
                # The forecast frame might not have a PMM file (yet), or
                # the field is empty. Return a 1x1 transparent webp rather
                # than 404 so the frontend can render a "no contours"
                # answer with the same DOM stacking logic.
                from PIL import Image
                _empty = io.BytesIO()
                Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(
                    _empty, format="WEBP", quality=80)
                overlay = _empty.getvalue()
            try:
                cpath.write_bytes(overlay)
            except OSError:
                pass
            fut.set_result(overlay)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            _REFS_CONTOUR_INFLIGHT.pop(inflight_key, None)

    from fastapi.responses import Response
    return Response(
        content=overlay, media_type="image/webp",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# --------------------------------------------------------------------------
# Paintball overlay — per-member threshold contours, transparent WebP.
# --------------------------------------------------------------------------

_PAINTBALL_INFLIGHT: dict[str, asyncio.Future] = {}


def _paintball_render_worker(date, run, fhr, var, thresh,
                              region, theme, bbox, sector_label):
    from .render import render_paintball_overlay
    return render_paintball_overlay(date, run, fhr, var, thresh,
                                    region, theme, bbox, sector_label)


@app.get("/api/paintball-overlay/{date}/{run}/{fhr}.webp")
async def paintball_overlay_endpoint(
        date: str, run: int, fhr: int,
        var: str = "refc", thresh: float = 40.0,
        region: str = "CONUS", theme: str = "dark",
        bbox: str = "", sector_name: str = ""):
    """Per-member threshold contours (paintball) as a transparent WebP overlay.

    Each REFS member gets a distinct color. Returns a 1×1 transparent WebP
    when no member data is available rather than 404, so the frontend can
    skip rendering without special-casing HTTP errors.
    """
    if fhr < 0 or fhr > refs_data.MAX_FHOUR:
        raise HTTPException(404, f"fhr {fhr} out of range")
    region_arg, bbox_t, sector_label, ck = _resolve_sector(
        region, bbox, sector_name)
    if theme not in core.THEMES:
        theme = "dark"
    safe = ck.replace("/", "_").replace(" ", "_")
    # BUILD_ID in the name: render-code changes must invalidate these on-disk
    # overlays the same way they invalidate the main tile cache (pb_* files
    # survive deploys via the cache-persist snapshot).
    fname = (f"pb_{BUILD_ID}_{var}_{thresh:.0f}_{date}_{run:02d}_F{fhr:03d}_"
             f"{safe}_{theme}.webp")
    cpath = cache.CACHE_DIR / fname

    if cpath.exists():
        return FileResponse(
            cpath, media_type="image/webp",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    inflight_key = fname
    pending = _PAINTBALL_INFLIGHT.get(inflight_key)
    if pending is not None:
        data = await pending
    else:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        _PAINTBALL_INFLIGHT[inflight_key] = fut
        try:
            # Member files store WIND/HAIL/MXUPHL as 1-h max records; the
            # member-aware match is required or the byte-range fetch finds
            # nothing and the overlay renders empty. The overlay renders a
            # 4-hr max (SPC convention), so fetch all 4 contributing fhrs.
            tasks = []
            for w in range(max(1, fhr - 3), fhr + 1):
                m = member_match_for(var, fhr=w)
                if m:
                    tasks += [
                        ensure_member_partial_cached(
                            _REFS_CACHE, date, run, mem, "2dfld", w, [m])
                        for mem in range(1, 6)
                    ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            async with _render_sem:
                exec_fut = loop.run_in_executor(
                    _render_executor, _paintball_render_worker,
                    date, run, fhr, var, thresh,
                    region_arg, theme, bbox_t, sector_label,
                )
                try:
                    data = await asyncio.wait_for(
                        exec_fut, timeout=RENDER_TIMEOUT_SECS)
                except asyncio.TimeoutError:
                    log.error("paintball render timeout %s F%03d", date, fhr)
                    fut.set_exception(HTTPException(504,
                        "Paintball render timeout"))
                    raise
            if data is None:
                from PIL import Image
                _empty = io.BytesIO()
                Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(
                    _empty, format="WEBP", quality=80)
                data = _empty.getvalue()
                # No member data — likely transient (records not posted yet).
                # Serve the blank but do NOT cache it on disk or in the
                # browser; an immutable empty overlay would keep the
                # paintball "broken" for the rest of the run.
                fut.set_result(data)
                from fastapi.responses import Response
                return Response(content=data, media_type="image/webp",
                                headers={"Cache-Control": "no-store"})
            try:
                cpath.write_bytes(data)
            except OSError:
                pass
            fut.set_result(data)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            _PAINTBALL_INFLIGHT.pop(inflight_key, None)

    from fastapi.responses import Response
    # A 1×1 blank (no member data) may also reach here via the in-flight
    # future when a concurrent request rendered it — never cache those.
    _cc = ("no-store" if len(data) < 100
           else "public, max-age=31536000, immutable")
    return Response(
        content=data, media_type="image/webp",
        headers={"Cache-Control": _cc},
    )


@app.get("/api/meteogram/{date}/{run}/{pid}")
async def meteogram_endpoint(date: str, run: int, pid: str,
                             fx: float, fy: float,
                             region: str = "CONUS", bbox: str = "",
                             sector_name: str = "", model: str = "refs"):
    """Point time series: the product's value at one map location across
    every valid forecast hour of the run. Powers the click-meteogram."""
    _mark_user_activity()
    if pid not in core.PRODUCTS:
        raise HTTPException(404, f"Unknown product: {pid}")
    prod = core.PRODUCTS[pid]
    _is_rrfs_aq = prod.get("source") == "rrfs_aq"
    if not _is_rrfs_aq and prod.get("recipe") not in (None, "prob_window"):
        return {"ok": True, "series": None,
                "reason": "not supported for member/multi-field products"}
    if _is_rrfs_aq:
        max_fhr = rrfs_aq_data.max_fhour_for_run(run)
    else:
        max_fhr = href_data.MAX_FHOUR if model == "href" else refs_data.MAX_FHOUR
    stride = max(1, int(prod.get("fhr_stride", 1)))
    fmin = max(1, int(prod.get("min_fhr", 0)))
    fhrs = [h for h in range(fmin, max_fhr + 1) if h % stride == 0] or [fmin]
    region_arg, bbox_t, sector_label, _ck = _resolve_sector(
        region, bbox, sector_name)
    # Byte-range prefetch every fhr's records first (batched) so the worker
    # only pays decode — and usually not even that, via the field cache.
    B = 8
    for i in range(0, len(fhrs), B):
        await asyncio.gather(
            *(_prefetch_records(date, run, h, pid, model=model)
              for h in fhrs[i:i + B]),
            return_exceptions=True)
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        _render_executor, _probe_series_worker,
        pid, date, run, fhrs, fx, fy, region_arg, bbox_t, sector_label, model)
    try:
        res = await asyncio.wait_for(fut, timeout=150)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Meteogram timed out")
    return {"ok": True, "series": res}


@app.get("/api/unproject")
async def unproject_endpoint(fx1: float, fy1: float, fx2: float, fy2: float,
                            region: str = "CONUS", bbox: str = "",
                            sector_name: str = ""):
    """Invert two drawn-box corners (image fractions, fy from the top) to a
    lon/lat bounding box using the tile's real Lambert projection. Used by the
    'draw a custom sector' flow so the drawn box maps to the right geography.
    Projection geometry is independent of model/date/fhr — only the current
    view's region/bbox matters."""
    region_arg, bbox_t, sector_label, _ck = _resolve_sector(
        region, bbox, sector_name)
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        _render_executor, _unproject_worker,
        [fx1, fx2], [fy1, fy2], region_arg, bbox_t, sector_label)
    try:
        pts = await asyncio.wait_for(fut, timeout=15)
    except asyncio.TimeoutError:
        return {"ok": False}
    if not pts or len(pts) < 2 or pts[0] is None or pts[1] is None:
        return {"ok": False}
    (lon1, lat1), (lon2, lat2) = pts[0], pts[1]
    return {
        "ok": True,
        "lon_min": min(lon1, lon2), "lat_min": min(lat1, lat2),
        "lon_max": max(lon1, lon2), "lat_max": max(lat1, lat2),
    }


@app.get("/api/probe/{date}/{run}/{pid}/{fhr}")
async def probe_endpoint(date: str, run: int, pid: str, fhr: int,
                         fx: float, fy: float,
                         region: str = "CONUS", bbox: str = "",
                         sector_name: str = "", model: str = "refs"):
    """Value-under-cursor readout. (fx, fy) are fractions of the tile image
    (fy from the top). Returns {ok, value, units, lat, lon}; value is null
    for unsupported recipes or off-map positions."""
    if pid not in core.PRODUCTS:
        raise HTTPException(404, f"Unknown product: {pid}")
    max_fhr = href_data.MAX_FHOUR if model == "href" else refs_data.MAX_FHOUR
    if fhr < 0 or fhr > max_fhr:
        raise HTTPException(404, f"fhr {fhr} out of range")
    region_arg, bbox_t, sector_label, _ck = _resolve_sector(
        region, bbox, sector_name)
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        _render_executor, _probe_value_worker,
        pid, date, run, fhr, fx, fy, region_arg, bbox_t, sector_label, model)
    try:
        res = await asyncio.wait_for(fut, timeout=25)
    except asyncio.TimeoutError:
        return {"ok": False, "value": None}
    if res is None:
        return {"ok": True, "value": None}
    return {"ok": True, **res}


# --------------------------------------------------------------------------
# LSR overlay (storm reports). Same caching philosophy as MRMS — observation,
# so cache by valid-minute + window + region only, not by (date, fhr). Two
# different REFS runs whose F-hours land at the same UTC minute share a tile.
# Render goes through the process pool with the same semaphore.
# --------------------------------------------------------------------------

_LSR_INFLIGHT: dict[str, asyncio.Future] = {}


def _lsr_render_worker(region_key, lsrs, bbox, sector_label):
    from .lsr import render_overlay
    return render_overlay(region_key, lsrs, bbox, sector_label)


@app.get("/api/lsr/{date}/{run}/{fhr}.webp")
async def lsr_overlay(date: str, run: int, fhr: int,
                      region: str = "CONUS",
                      bbox: str = "",
                      sector_name: str = "",
                      window_min: int = 30,
                      types: str = ""):
    """Transparent LSR overlay for the given forecast frame's valid time.

    `types` is a comma-separated list of IEM single-letter LSR codes
    (e.g. "T,H,G,D,O" for severe; "F,E" for the flood/flash-flood layer).
    Empty / missing → server default (severe). The set is included in the
    cache key so the severe and flood layers cache independently.
    """
    from .lsr import DEFAULT_TYPES, ALL_KNOWN_TYPES
    if fhr < 0 or fhr > refs_data.MAX_FHOUR:
        raise HTTPException(404, f"fhr {fhr} out of range")
    window_min = max(5, min(180, window_min))
    # Parse + canonicalise the requested type set so the cache key is stable
    # regardless of caller's order / whitespace.
    if types:
        wanted = tuple(sorted({
            t.strip().upper() for t in types.split(",")
            if t.strip().upper() in ALL_KNOWN_TYPES
        }))
        if not wanted:
            wanted = DEFAULT_TYPES
    else:
        wanted = DEFAULT_TYPES
    types_tag = "".join(wanted) or "none"

    region_arg, bbox_t, sector_label, cache_key = _resolve_sector(
        region, bbox, sector_name)

    from datetime import datetime, timedelta, timezone
    init = datetime.strptime(date, "%Y%m%d").replace(
        hour=run, tzinfo=timezone.utc)
    valid = init + timedelta(hours=fhr)
    now_utc = datetime.now(timezone.utc)
    verified = (now_utc - valid) >= timedelta(minutes=window_min)

    safe_region = cache_key.replace("/", "_").replace(" ", "_")
    # Key by valid-minute + window + type-set — independent of which
    # (date, fhr) pair produced this valid time.
    valid_tag = valid.strftime("%Y%m%dT%H%M")
    fname = f"lsr_{valid_tag}_w{window_min:03d}_t{types_tag}_{safe_region}.webp"
    cpath = cache.CACHE_DIR / fname

    if cpath.exists():
        return FileResponse(
            cpath, media_type="image/webp",
            headers={"Cache-Control":
                     "public, max-age=31536000, immutable" if verified
                     else "public, max-age=120"},
        )

    inflight_key = fname
    pending = _LSR_INFLIGHT.get(inflight_key)
    if pending is not None:
        overlay = await pending
    else:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        _LSR_INFLIGHT[inflight_key] = fut
        try:
            if valid > now_utc:
                lsrs = []
            else:
                from .lsr import fetch_lsrs
                try:
                    lsrs = await fetch_lsrs(
                        valid - timedelta(minutes=window_min),
                        valid + timedelta(minutes=window_min),
                        types=wanted,
                    )
                except Exception as e:
                    log.warning("IEM LSR fetch failed valid=%s w=%d: %s",
                                valid_tag, window_min, e)
                    lsrs = []

            async with _render_sem:
                exec_fut = loop.run_in_executor(
                    _render_executor, _lsr_render_worker,
                    region_arg, lsrs, bbox_t, sector_label,
                )
                try:
                    overlay = await asyncio.wait_for(
                        exec_fut, timeout=RENDER_TIMEOUT_SECS)
                except asyncio.TimeoutError:
                    log.error("LSR render timeout valid=%s", valid_tag)
                    fut.set_exception(HTTPException(504, "LSR render timeout"))
                    raise
            if overlay is None:
                fut.set_exception(HTTPException(500, "LSR render failed"))
                raise HTTPException(500, "LSR render failed")
            if verified:
                try:
                    cpath.write_bytes(overlay)
                except OSError:
                    pass
            fut.set_result(overlay)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            _LSR_INFLIGHT.pop(inflight_key, None)

    from fastapi.responses import Response
    return Response(
        content=overlay, media_type="image/webp",
        headers={"Cache-Control":
                 "public, max-age=31536000, immutable" if verified
                 else "public, max-age=120"},
    )


# =============================================================================
#  WxWorks region-boundary overlay
# =============================================================================

_REGIONS_INFLIGHT: dict[str, asyncio.Future] = {}


def _regions_render_worker(region_key, bbox, sector_label):
    from .regions_overlay import render_overlay
    return render_overlay(region_key, bbox, sector_label)


@app.get("/api/regions-overlay.webp")
async def regions_overlay_endpoint(region: str = "CONUS",
                                   bbox: str = "",
                                   sector_name: str = ""):
    """Transparent WebP overlay of WxWorks forecast-area boundaries.

    Region boundaries are static (not date/run dependent) so the cache key
    is just the region/sector string.  Immutable once rendered.
    """
    region_arg, bbox_t, sector_label, cache_key = _resolve_sector(
        region, bbox, sector_name)

    safe_key = cache_key.replace("/", "_").replace(" ", "_")
    fname = f"wxworks_regions_{safe_key}.webp"
    cpath = cache.CACHE_DIR / fname

    if cpath.exists():
        from fastapi.responses import FileResponse
        return FileResponse(
            cpath, media_type="image/webp",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    inflight_key = fname
    pending = _REGIONS_INFLIGHT.get(inflight_key)
    if pending is not None:
        overlay = await pending
    else:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        _REGIONS_INFLIGHT[inflight_key] = fut
        try:
            async with _render_sem:
                exec_fut = loop.run_in_executor(
                    _render_executor, _regions_render_worker,
                    region_arg, bbox_t, sector_label,
                )
                try:
                    overlay = await asyncio.wait_for(
                        exec_fut, timeout=RENDER_TIMEOUT_SECS)
                except asyncio.TimeoutError:
                    log.error("Regions overlay render timeout region=%s", region_arg)
                    fut.set_exception(HTTPException(504, "Regions render timeout"))
                    raise HTTPException(504, "Regions render timeout")
            if overlay:
                cpath.write_bytes(overlay)
            fut.set_result(overlay)
        except HTTPException:
            raise
        except Exception as e:
            log.error("Regions overlay render error: %s", e)
            if not fut.done():
                fut.set_exception(e)
            raise HTTPException(500, str(e))
        finally:
            _REGIONS_INFLIGHT.pop(inflight_key, None)

    if not overlay:
        raise HTTPException(404, "No regions data")

    from fastapi.responses import Response
    return Response(
        content=overlay, media_type="image/webp",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/download/{date}/{run}/{pid}/{fhr}.png")
async def download_png(date: str, run: int, pid: str, fhr: int,
                       region: str = "CONUS",
                       palette: str = "Default",
                       theme: str = "dark",
                       bbox: str = "",
                       sector_name: str = "",
                       counties: int = 0, cities: int = 0,
                       regions: int = 0,
                       model: str = "refs"):
    """Return a true PNG of the requested tile.

    Internally we render once to the WebP cache; this endpoint decodes that
    cached WebP and re-encodes it as PNG on demand so operational users who
    want a PNG (for reports/slides) still get one.
    """
    _mark_user_activity()
    if pid not in core.PRODUCTS:
        raise HTTPException(404, f"Unknown product: {pid}")
    if palette not in core.PALETTES:
        raise HTTPException(400, f"Unknown palette: {palette}")
    if theme not in core.THEMES:
        raise HTTPException(400, f"Unknown theme: {theme}")
    max_fhr = href_data.MAX_FHOUR if model == "href" else refs_data.MAX_FHOUR
    if fhr < 0 or fhr > max_fhr:
        raise HTTPException(404, f"fhr {fhr} out of range")
    region_arg, bbox_t, sector_label, cache_key = _resolve_sector(
        region, bbox, sector_name)
    show_counties = bool(counties)
    show_cities = bool(cities)
    show_regions_flag = bool(regions)
    cache_key = f"{cache_key}_c{int(show_counties)}y{int(show_cities)}r{int(show_regions_flag)}"
    p, _no_data = await _render_to_cache(date, run, pid, fhr, region_arg, palette,
                                theme, bbox_t, sector_label, cache_key,
                                show_counties=show_counties,
                                show_cities=show_cities,
                                show_regions=show_regions_flag,
                                model=model)

    def _to_png() -> bytes:
        from PIL import Image
        import io as _io
        with Image.open(p) as im:
            buf = _io.BytesIO()
            im.save(buf, format="PNG", optimize=False, compress_level=1)
            return buf.getvalue()

    data = await asyncio.to_thread(_to_png)
    from fastapi.responses import Response
    model_tag = model.upper()
    fname = f"{model_tag}_{date}_{run:02d}z_{pid}_{cache_key.replace('/', '_')}_F{fhr:03d}.png"
    return Response(content=data, media_type="image/png", headers={
        "Content-Disposition": f'attachment; filename="{fname}"',
        "Cache-Control": "private, max-age=300",
    })


@app.get("/api/export/gif/{date}/{run}/{pid}.gif")
async def export_gif(date: str, run: int, pid: str,
                     region: str = "CONUS",
                     palette: str = "Default",
                     theme: str = "dark",
                     bbox: str = "",
                     sector_name: str = "",
                     fmin: int = 0, fmax: int = 60, step: int = 1,
                     duration_ms: int = 500, max_width: int = 1500,
                     counties: int = 0, cities: int = 0,
                     regions: int = 0,
                     model: str = "refs"):
    """Build an animated GIF of the rendered frames in [fmin, fmax] by step."""
    _mark_user_activity()
    if pid not in core.PRODUCTS:
        raise HTTPException(404, f"Unknown product: {pid}")
    if palette not in core.PALETTES:
        raise HTTPException(400, f"Unknown palette: {palette}")
    if theme not in core.THEMES:
        raise HTTPException(400, f"Unknown theme: {theme}")
    step = max(1, min(24, step))
    fmin = max(0, fmin)
    cap_fhr = href_data.MAX_FHOUR if model == "href" else refs_data.MAX_FHOUR
    fmax = min(cap_fhr, fmax)
    if fmax < fmin:
        raise HTTPException(400, "fmax < fmin")
    duration_ms = max(80, min(2000, duration_ms))
    max_width = max(640, min(2400, max_width))
    region_arg, bbox_t, sector_label, cache_key = _resolve_sector(
        region, bbox, sector_name)
    show_counties = bool(counties)
    show_cities = bool(cities)
    show_regions_flag = bool(regions)
    cache_key = f"{cache_key}_c{int(show_counties)}y{int(show_cities)}r{int(show_regions_flag)}"
    frames: list = []
    for h in range(fmin, fmax + 1, step):
        try:
            p, nd = await _render_to_cache(date, run, pid, h, region_arg,
                                       palette, theme, bbox_t,
                                       sector_label, cache_key,
                                       show_counties=show_counties,
                                       show_cities=show_cities,
                                       show_regions=show_regions_flag,
                                       model=model)
            if nd:
                continue  # don't bake placeholder frames into the GIF
            frames.append(p)
        except HTTPException:
            continue
        except Exception as e:
            log.warning("gif: skipping F%03d: %s", h, e)
            continue
    if not frames:
        raise HTTPException(404, "No frames available to encode")

    def encode() -> bytes:
        from PIL import Image
        import io as _io
        imgs = []
        for fp in frames:
            im = Image.open(fp).convert("RGB")
            if im.width > max_width:
                ratio = max_width / im.width
                im = im.resize((max_width, int(im.height * ratio)), Image.LANCZOS)
            imgs.append(im.quantize(colors=256, dither=Image.FLOYDSTEINBERG))
        buf = _io.BytesIO()
        imgs[0].save(buf, format="GIF", save_all=True,
                     append_images=imgs[1:],
                     duration=duration_ms, loop=0, optimize=False,
                     disposal=2)
        return buf.getvalue()

    data = await asyncio.to_thread(encode)
    fname = f"REFS_{date}_{run:02d}z_{pid}_{cache_key.replace('/', '_')}.gif"
    from fastapi.responses import Response
    return Response(content=data, media_type="image/gif", headers={
        "Content-Disposition": f'attachment; filename="{fname}"',
        "Cache-Control": "private, max-age=60",
    })


@app.get("/api/meta/{date}/{run}/{pid}/{fhr}")
async def meta_endpoint(date: str, run: int, pid: str, fhr: int,
                        region: str = "CONUS",
                        palette: str = "Default",
                        theme: str = "dark"):
    m = cache.read_meta(date, run, pid, region, palette, theme, fhr)
    if m is None:
        raise HTTPException(404, "Meta not cached (render the tile first)")
    return m


# --- Background prewarm of latest run ------------------------------------
# Two-phase strategy:
#
# Phase 1 — GRIB prefetch (fast, concurrent): byte-range fetch the records
#   for a broad set of products across all forecast hours.  No render step;
#   just gets the raw data onto disk so the first user click only pays the
#   ~3-5 s render cost instead of 20-60 s download + render.
#
# Phase 2 — Render (slow, sequential): actually render tiles for the most-
#   watched products at key forecast hours so they come back instantly even
#   on the very first click, with no plot wait at all.
#
# PREWARM_RENDER_PRODUCTS / PREWARM_RENDER_FHOURS drive Phase 2.
# PREWARM_FETCH_PRODUCTS drives Phase 1 (superset of render list).
#
# All lists are tunable via env vars for quick iteration without a redeploy:
#   PREWARM_FETCH_PRODUCTS  comma-separated product IDs
#   PREWARM_RENDER_PRODUCTS comma-separated product IDs
#   PREWARM_RENDER_FHOURS   comma-separated integers

def _env_list(name: str, default: tuple) -> tuple:
    v = os.environ.get(name, "")
    if v.strip():
        return tuple(x.strip() for x in v.split(",") if x.strip())
    return default

# Products to pre-render (full tile) — high-traffic staples only; render is
# sequential so keep this list short (~10-15 entries).
_DEFAULT_RENDER_PRODUCTS = (
    "refc_pmmn_series", "uh25_pmmn",
    "qpf_6h_lpmm_series",
    "sbcape_mean_series", "srh_03km_mean",
    "t2m_combo", "pwat_refc_combo",
    # Operationally critical severe-wind combo (paintball + joint NHP).
    # Member records are byte-range fetched, so this is cheap to warm.
    "paintball_wind10_30kt_nhp",
    "wind10_prob_30kt_4h",
)
# Forecast hours at which to pre-render the above products. Every fhr: the
# box idles ~5 h between cycles and warm tiles are the single biggest
# loading-speed lever on 2 vCPUs. The render loop yields to live user
# traffic (see _prewarm_worker), so the breadth costs idle CPU only.
_DEFAULT_RENDER_FHOURS = tuple(range(1, 49))

# Products to GRIB-prefetch only (no render). Previously this was the ENTIRE
# catalog (~100 products) × every available fhour, which generated a large,
# sustained background S3 fan-out that competed with interactive renders for
# the shared httpx pool and CPU. Trim to a curated set of high-traffic
# products (the pre-render staples + the most-opened products across the main
# tabs) so background fetching stays light. First click on anything outside
# this list still works — it just pays the one-time download on that click.
# Override with PREWARM_FETCH_PRODUCTS to widen/narrow without a redeploy.
_DEFAULT_FETCH_PRODUCTS = (
    # Pre-render staples (Phase 2 renders these; fetch must cover them).
    "refc_pmmn_series", "uh25_pmmn",
    "qpf_6h_lpmm_series", "qpf_3h_pmmn_series",
    "sbcape_mean_series", "srh_03km_mean", "t2m_combo", "pwat_refc_combo",
    # Other frequently-opened products across the tabs.
    "qpf_1h_prob_001", "qpf_1h_prob_100",
    "refd_pmmn", "wind_10m_mean", "td2m_combo",
    "paintball_wind10_30kt_nhp",
    "wind10_prob_30kt_4h",
)

PREWARM_RENDER_PRODUCTS = _env_list("PREWARM_RENDER_PRODUCTS", _DEFAULT_RENDER_PRODUCTS)
PREWARM_RENDER_FHOURS   = tuple(int(h) for h in _env_list(
    "PREWARM_RENDER_FHOURS", tuple(str(h) for h in _DEFAULT_RENDER_FHOURS)))
PREWARM_FETCH_PRODUCTS  = _env_list("PREWARM_FETCH_PRODUCTS", _DEFAULT_FETCH_PRODUCTS)

# Back-compat alias so existing ENABLE_PREWARM env var still works.
ENABLE_PREWARM = os.environ.get("ENABLE_PREWARM", "1") == "1"

# Keep the old names as aliases so nothing that references them breaks.
PREWARM_PRODUCTS = PREWARM_RENDER_PRODUCTS
PREWARM_FHOURS   = PREWARM_RENDER_FHOURS

_PREFETCH_BATCH = 6    # concurrent HTTP requests per prewarm batch (gentle on cpu-basic)
_PREWARM_BATCH_PAUSE = 0.4   # seconds between batches so users keep the pool/event loop

# Background fetches (prewarm Phase-1 + speculative look-ahead) share the SAME
# httpx connection pool as user-initiated tile renders. On a 2-vCPU box an
# unbounded background fan-out exhausts the pool and starves real requests.
# This semaphore caps *background* concurrency to a few connections, always
# leaving the bulk of the pool free for interactive renders. User renders call
# _prefetch_records directly and are NOT gated.
_bg_fetch_sem: "asyncio.Semaphore | None" = None


def _bg_sem() -> asyncio.Semaphore:
    global _bg_fetch_sem
    if _bg_fetch_sem is None:
        _bg_fetch_sem = asyncio.Semaphore(
            int(os.environ.get("BG_FETCH_CONCURRENCY", "3")))
    return _bg_fetch_sem


async def _bg_prefetch_records(date: str, run: int, fhr: int, pid: str,
                               model: str = "refs") -> None:
    """Concurrency-limited wrapper around _prefetch_records for background work
    (prewarm + speculative), so it can never starve user-initiated renders."""
    async with _bg_sem():
        await _prefetch_records(date, run, fhr, pid, model=model)


async def _prewarm_one(date: str, run: int, pid: str, fhr: int) -> None:
    """Fetch GRIB records AND render the default-CONUS/dark/SPC-Classic tile,
    so the first user click on a prewarm product hits a fully-warm cache.
    """
    # Already warm (e.g. restored from the cache snapshot): skip the
    # prefetch entirely instead of paying idx fetches for nothing.
    if cache.png_path(date, run, pid, _PREWARM_CACHE_KEY,
                      "Default", "dark", fhr).exists():
        return
    try:
        await _prefetch_records(date, run, fhr, pid)
    except Exception as e:
        log.debug("prewarm prefetch %s F%03d: %s", pid, fhr, e)
        return
    try:
        await _render_to_cache(
            date, run, pid, fhr,
            region="CONUS", palette="Default", theme="dark",
            bbox=None, sector_label="CONUS", cache_key=_PREWARM_CACHE_KEY,
        )
    except HTTPException as e:
        log.debug("prewarm render %s F%03d: %s", pid, fhr, e.detail)
    except Exception as e:
        log.debug("prewarm render %s F%03d: %s", pid, fhr, e)


async def _prewarm_fetch_all(date: str, run: int,
                             available_fhrs: list[int]) -> None:
    """Phase 1: GRIB byte-range prefetch for all products × available fhours.

    Runs in batches so we don't open hundreds of S3 connections at once.
    Only fetches records whose on-disk tile doesn't exist yet — if a persistent
    cache volume is mounted the vast majority will already be present.

    ``available_fhrs`` must be the verified list from S3 — we never try to
    fetch fhours that aren't published yet, which avoids flooding the event
    loop with 404s when a run is still being ingested.
    """
    import refs_core as _core
    pairs = [
        (pid, fhr)
        for pid in PREWARM_FETCH_PRODUCTS
        for fhr in available_fhrs
        if pid in _core.PRODUCTS
        # Skip if tile already cached — nothing to download.
        and not cache.png_path(date, run, pid, _PREWARM_CACHE_KEY,
                               "Default", "dark", fhr).exists()
    ]
    log.info("prewarm fetch: %d (product, fhr) pairs to prefetch for %s %02dZ "
             "(%d fhrs available)", len(pairs), date, run, len(available_fhrs))
    for i in range(0, len(pairs), _PREFETCH_BATCH):
        batch = pairs[i:i + _PREFETCH_BATCH]
        tasks = [_bg_prefetch_records(date, run, fhr, pid) for pid, fhr in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errs = sum(1 for r in results if isinstance(r, Exception))
        if errs:
            log.debug("prewarm fetch batch %d: %d errors", i // _PREFETCH_BATCH, errs)
        # Real pause (not sleep(0)) so interactive renders keep CPU + connections.
        await asyncio.sleep(_PREWARM_BATCH_PAUSE)
    log.info("prewarm fetch: done for %s %02dZ", date, run)


async def _prewarm_worker() -> None:
    """Find latest run; Phase 1 GRIB prefetch then Phase 2 render."""
    await asyncio.sleep(30)           # let the server serve first user clicks before warming
    seen: set[tuple[str, int]] = set()
    while True:
        try:
            latest = await refs_data.find_latest_run()
        except Exception:
            latest = None
        if latest and latest not in seen:
            date, run = latest
            log.info("prewarm: warming %s %02dZ", date, run)

            # Check how many fhours are actually published before doing
            # anything — if the run is still ingesting (common for the first
            # 1-2 hours after init time) we skip both phases so we don't
            # flood the event loop with hundreds of 404 requests.
            try:
                available_fhrs = await refs_data.list_available_fhours(date, run)
            except Exception:
                available_fhrs = []

            if len(available_fhrs) < 6:
                log.info("prewarm: skipping %s %02dZ — only %d fhrs on S3, "
                         "will retry next cycle", date, run, len(available_fhrs))
                # Don't add to `seen` so we re-check on next poll.
                await asyncio.sleep(300)
                continue

            # Phase 1: concurrent GRIB prefetch across full product catalog.
            try:
                await _prewarm_fetch_all(date, run, available_fhrs)
            except Exception:
                log.exception("prewarm fetch phase failed")

            # Phase 2: sequential render for top products at key fhours.
            # Sequential so prewarm doesn't saturate the render pool and block
            # real user requests. fhr-major ordering: early frames of EVERY
            # product warm first (that's where users land on a fresh cycle).
            # Before each render, wait for PREWARM_IDLE_SECS of user quiet so
            # the warm pass never competes with live clicks.
            render_fhrs = [h for h in PREWARM_RENDER_FHOURS if h in set(available_fhrs)]
            for fhr in render_fhrs:
                for pid in PREWARM_RENDER_PRODUCTS:
                    while time.time() - _last_user_activity < PREWARM_IDLE_SECS:
                        await asyncio.sleep(5)
                    await _prewarm_one(date, run, pid, fhr)

            seen.add(latest)
            log.info("prewarm: done %s %02dZ", date, run)
            if len(seen) > 8:
                # Forget the oldest entries so a new cycle re-warms.
                for old in list(seen)[:-4]:
                    seen.discard(old)
        await asyncio.sleep(300)       # re-check every 5 min for new cycles


# Nothing pruned the GRIB-partial cache (or the new decoded-field cache)
# before — ephemeral /tmp resets masked it, but the expanded prewarm and the
# field cache grow disk much faster. Sweep anything older than this hourly.
GRIB_CACHE_MAX_AGE_H = float(os.environ.get("GRIB_CACHE_MAX_AGE_H", "72"))
# Decoded fields are ~7.6 MB each and only useful while a cycle is being
# actively browsed — expire them faster than the GRIBs.
FIELD_CACHE_MAX_AGE_H = float(os.environ.get("FIELD_CACHE_MAX_AGE_H", "36"))


def _janitor_sweep() -> None:
    import re as _re
    import shutil as _shutil
    cutoff = time.time() - GRIB_CACHE_MAX_AGE_H * 3600
    field_cutoff = time.time() - FIELD_CACHE_MAX_AGE_H * 3600
    base = _REFS_CACHE
    if not base.exists():
        return
    removed = 0
    for d in base.iterdir():
        try:
            if d.is_dir() and _re.fullmatch(r"\d{8}", d.name):
                # Cycle dir (YYYYMMDD) — mtime stops moving once the day's
                # cycles are complete, so age-based removal is safe.
                if d.stat().st_mtime < cutoff:
                    _shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            elif d.is_dir() and d.name == "_fields":
                for f in d.iterdir():
                    if f.stat().st_mtime < field_cutoff:
                        f.unlink(missing_ok=True)
        except OSError:
            continue
    if removed:
        log.info("janitor: removed %d expired cycle dir(s)", removed)


async def _janitor_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(_janitor_sweep)
        except Exception:
            log.exception("janitor sweep failed")
        await asyncio.sleep(3600)


@app.on_event("startup")
async def _on_startup() -> None:
    # Restore the on-disk tile cache from the last snapshot so a redeploy /
    # cold-start serves warm tiles immediately instead of re-rendering. Runs
    # in the background (and prewarm's 30 s initial sleep overlaps it) so a
    # large restore never delays the Space's readiness probe.
    async def _restore() -> None:
        try:
            await asyncio.to_thread(
                cache_persist.restore_on_startup, cache.CACHE_DIR)
        except Exception:
            log.exception("cache persist: restore task failed")
    asyncio.create_task(_restore())
    asyncio.create_task(cache_persist.snapshot_loop(cache.CACHE_DIR))
    asyncio.create_task(_janitor_loop())

    # Remote decoded-field cache warm (PERSIST_FIELDS=1 to enable).
    # Downloads previously saved .npz fields so GRIB decode is skipped on the
    # first render after a cold start / restart.
    async def _field_warm() -> None:
        try:
            fields_dir = _REFS_CACHE / "_fields"
            await asyncio.to_thread(field_persist.warm_on_startup, fields_dir)
        except Exception:
            log.exception("field persist: warm task failed")
    asyncio.create_task(_field_warm())
    asyncio.create_task(field_persist.prune_old_fields_async())

    if not ENABLE_PREWARM:
        log.info("prewarm: disabled (set ENABLE_PREWARM=1 to enable)")
        return
    asyncio.create_task(_prewarm_worker())
    log.info("prewarm: worker scheduled")


def _build_index_html() -> str:
    """Read index.html and inject ?v=BUILD_ID onto the JS + CSS references so
    browsers refetch them whenever the render code (or any other file used to
    derive BUILD_ID) changes. Without this, app.js sits in the browser cache
    indefinitely and users keep running the old JS even after a deploy."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    # ASSET_ID covers static-file contents so editing app.js or style.css
    # alone is enough to bust the browser cache, even when render code
    # (and thus BUILD_ID, used for tiles) hasn't changed.
    return (html
            .replace('href="./style.css"', f'href="./style.css?v={ASSET_ID}"')
            .replace('src="./app.js"', f'src="./app.js?v={ASSET_ID}"'))


_INDEX_HTML_CACHE = _build_index_html()


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def index():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(
        _INDEX_HTML_CACHE,
        # Keep the document itself uncacheable so a refresh always gets the
        # current ?v=BUILD_ID tokens; the tagged static assets it points at
        # can still be aggressively cached because their URL changes.
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# Static assets (everything except the root HTML, which the route above
# serves with cache-busting tokens injected).
app.mount("/", StaticFiles(directory=str(STATIC), html=False), name="static")
