"""Thin wrapper around refs_core.PlotJob → PNG bytes.

CRITICAL: refs_core (and through it, eccodes / pygrib / cartopy / matplotlib)
must be imported on the SAME thread it's used from. eccodes binds thread-local
state at module import time; using it from a different thread later causes a
hard segfault.

The render() function below is invoked from a single dedicated
ThreadPoolExecutor worker thread (see app.main._render_executor). We defer
all refs_core-touching imports until the first call so init happens on that
worker thread, not the main asyncio thread.
"""
from __future__ import annotations

import io
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_initialized = False
_job_refs = None           # PlotJob backed by REFSDataProcessor
_job_href = None           # PlotJob backed by HREFDataProcessor
_job_spc = None            # PlotJob backed by SPCPostDataProcessor (SPC guidance)
_job_reps = None           # PlotJob for REPS (recipe='reps_mean' never touches
                           # self.proc -- it does its own async fetch/decode --
                           # so no dedicated DataProcessor is needed, unlike
                           # the other three jobs above).
_job_rrfs = None           # PlotJob for RRFS operational (same shape as REPS --
                           # recipes do their own async fetch/decode, no processor).
_core = None               # refs_core module reference
_init_lock = threading.Lock()


def _ensure_initialized():
    """Import refs_core on the calling thread and build PlotJobs for both models."""
    global _initialized, _job_refs, _job_href, _job_spc, _job_reps, _job_rrfs, _core
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        sys.path.insert(0, str(_ROOT))
        import refs_core as core  # noqa: WPS433 -- deferred on purpose
        # Each ProcessPoolExecutor worker is a fresh spawn — `app.catalog`
        # isn't auto-imported here, so the extras (PRODUCTS additions and
        # the COMPOSITES function registry) aren't populated unless we
        # explicitly trigger them. Without this, requests for any product
        # added in extra_products.py 404 with KeyError on PRODUCTS[pid].
        try:
            from . import extra_products as _extras  # noqa: WPS433
            _extras.register()
        except Exception as e:
            print(f"[render] extras registration failed: "
                  f"{type(e).__name__}: {e}", flush=True)
        # Same gotcha applies to REPS's product registry.
        try:
            from . import reps_products as _reps_products  # noqa: WPS433
            _reps_products.register()
        except Exception as e:
            print(f"[render] reps_products registration failed: "
                  f"{type(e).__name__}: {e}", flush=True)
        # ...and to RRFS's.
        try:
            from . import rrfs_products as _rrfs_products  # noqa: WPS433
            _rrfs_products.register()
        except Exception as e:
            print(f"[render] rrfs_products registration failed: "
                  f"{type(e).__name__}: {e}", flush=True)
        _core = core
        _job_refs = core.PlotJob(
            core.REFSDataProcessor(local_dir=core.DEFAULT_LOCAL),
            core.PlotManager(),
        )
        _job_href = core.PlotJob(
            core.HREFDataProcessor(local_dir=core.DEFAULT_LOCAL),
            core.PlotManager(),
        )
        _job_spc = core.PlotJob(
            core.SPCPostDataProcessor(local_dir=core.DEFAULT_LOCAL),
            core.PlotManager(),
        )
        _job_reps = core.PlotJob(None, core.PlotManager())
        _job_rrfs = core.PlotJob(None, core.PlotManager())
        _initialized = True
        print(f"[render] refs_core initialized on thread "
              f"{threading.current_thread().name}", flush=True)


def render(pid: str, date: str, run: int, fhr: int,
           region: str, palette: str, theme: str,
           bbox: tuple | None = None, sector_name: str = "",
           show_counties: bool = False, show_cities: bool = False,
           show_regions: bool = False,
           model: str = "refs", member: int = -1):
    """Render one product/frame. Returns (png_bytes, meta) or None.

    Must be called from a single dedicated worker thread.

    If ``bbox`` is provided as (lon_min, lat_min, lon_max, lat_max), the
    region is temporarily overridden in refs_core.REGIONS so refs_core
    projects to that custom domain. Renders are serialized so global
    mutation is safe.

    ``show_counties`` and ``show_cities`` flip overlay layers on a per-
    render basis — defaults OFF for the minimal AG-WX-style look.
    """
    _ensure_initialized()
    core = _core
    # SPC post-processed calibrated guidance and REPS are each their own
    # data source, routed to a dedicated job regardless of the REFS/HREF
    # model toggle.
    _prod = core.PRODUCTS.get(pid)
    _is_spc = bool(_prod and _prod.get("source") == "spc_post")
    _is_reps = bool(_prod and _prod.get("source") == "reps")
    _is_rrfs = bool(_prod and _prod.get("source") == "rrfs")
    if _is_spc:
        job = _job_spc
    elif _is_reps:
        job = _job_reps
    elif _is_rrfs:
        job = _job_rrfs
    elif model == "href":
        job = _job_href
    else:
        job = _job_refs
    _CUSTOM = "__custom__"
    custom_region_installed = False
    effective_region = region
    try:
        core.apply_palette(palette)
        core.PlotManager.theme = core.THEMES.get(theme, core.THEMES["dark"])
        core.PlotManager.show_counties = bool(show_counties)
        core.PlotManager.show_cities   = bool(show_cities)
        core.PlotManager.show_regions  = bool(show_regions)
        core.PlotManager.model_label   = (
            "SPC HREF" if _is_spc
            else "REPS" if _is_reps
            else "RRFS" if _is_rrfs
            else "HREF v3" if model == "href" else "REFS")
        # Member browser: which single member the stamp recipes should
        # render full-size (-1 = 21-panel grid). Reset in finally so it
        # never leaks into the next serialized render.
        core.PlotManager.member_view = int(member)
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            core.REGIONS[_CUSTOM] = dict(
                lon=(lon_min, lon_max),
                lat=(lat_min, lat_max),
                proj_lon=(lon_min + lon_max) / 2.0,
                name=(sector_name or "Custom"),
            )
            effective_region = _CUSTOM
            custom_region_installed = True
        run_dt = datetime.strptime(date, "%Y%m%d").replace(
            hour=run, tzinfo=timezone.utc)
        fig = job.render(pid, date, run, fhr, effective_region, run_dt,
                         lambda m: None)
        # Render returned None — most often because the underlying GRIB record
        # doesn't exist at this fhour (multi-hour accumulations early in the
        # run, member files not yet posted, etc.). Surface that to the user
        # as a captioned placeholder tile rather than a 404.
        is_no_data = False
        if fig is None:
            prod = core.PRODUCTS.get(pid)
            if prod is not None:
                fig = job.pm.no_data(prod, effective_region, run_dt, fhr)
                is_no_data = True
    except Exception as e:
        print(f"[render] {pid} {date} {run:02d}z F{fhr:03d}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None
    finally:
        if custom_region_installed:
            core.REGIONS.pop(_CUSTOM, None)  # noqa: render-finally
        core.PlotManager.member_view = -1    # never leak into next render
    if fig is None:
        return None
    buf = io.BytesIO()
    # WebP is ~60-70% smaller than PNG at q=92 with no perceptible loss for
    # these plots, and Pillow encodes it faster than PNG's default level-6
    # zlib pass. method=4 is Pillow's "balanced" effort.
    fig.savefig(buf, format="webp", dpi=core.PlotManager.DPI,
                bbox_inches=None,
                pil_kwargs={"quality": 92, "method": 4})
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    meta = {
        "pid": pid,
        "date": date,
        "run": run,
        "fhr": fhr,
        "region": region,
        "palette": palette,
        "theme": theme,
        "model": model,
        "product": core.PRODUCTS[pid]["name"],
        "spc_title": core.PRODUCTS[pid].get("spc_title", ""),
    }
    # Placeholder tiles must not enter the permanent cache: the condition
    # ("member file not posted yet") clears within minutes, but an immutable
    # cached placeholder would mask the recovery for the rest of the run.
    if is_no_data:
        meta["no_data"] = True
    return buf.getvalue(), meta


def _probe_lonlat(core, region: str, fx: float, fy: float):
    """Image-fraction (fx, fy from top) → (lon, lat), or None if off-map.
    Replicates cartopy set_extent's boundary-sampled projection bbox."""
    import numpy as np
    import cartopy.crs as ccrs
    pm = core.PlotManager
    r = core.REGIONS.get(region)
    if r is None:
        return None
    left, bot, w, h = pm.MAP_BOX
    figy = 1.0 - fy
    if not (left <= fx <= left + w and bot <= figy <= bot + h):
        return None
    axx = (fx - left) / w
    axy = (figy - bot) / h
    proj = pm._projection(region)
    n = 50
    lon0, lon1 = r["lon"]; lat0, lat1 = r["lat"]
    blons = np.concatenate([np.linspace(lon0, lon1, n), np.full(n, lon1),
                            np.linspace(lon1, lon0, n), np.full(n, lon0)])
    blats = np.concatenate([np.full(n, lat0), np.linspace(lat0, lat1, n),
                            np.full(n, lat1), np.linspace(lat1, lat0, n)])
    pts = proj.transform_points(ccrs.PlateCarree(), blons, blats)
    xmin, xmax = np.nanmin(pts[:, 0]), np.nanmax(pts[:, 0])
    ymin, ymax = np.nanmin(pts[:, 1]), np.nanmax(pts[:, 1])
    x = xmin + axx * (xmax - xmin)
    y = ymin + axy * (ymax - ymin)
    lonp, latp = ccrs.PlateCarree().transform_point(x, y, src_crs=proj)
    if not (np.isfinite(lonp) and np.isfinite(latp)):
        return None
    return float(lonp), float(latp)


def _probe_load_field_rrfs(prod, date: str, run: int, fhr: int):
    """Load one fhr's RRFS field for point sampling -- same data path as
    PlotJob._rrfs_field / _rrfs_shear in refs_core.py, minus the plotting."""
    import asyncio
    from app import rrfs_fields as rf
    core = _core
    cache_dir = Path(core.DEFAULT_LOCAL)
    recipe = prod.get("recipe")
    try:
        if recipe == "rrfs_shear":
            layer = prod.get("rrfs_shear_layer")
            if not layer:
                return None, None, None
            result = asyncio.run(rf.load_rrfs_shear(cache_dir, date, run, fhr, layer))
        else:
            tmpl = prod.get("rrfs_idx_tmpl")
            field_key = prod.get("rrfs_field_key")
            if not tmpl or not field_key:
                return None, None, None
            sub = tmpl.format(fhr=fhr, fhrm1=fhr - 1)
            family = prod.get("rrfs_family", "2dfld")
            result = asyncio.run(rf.load_rrfs_generic(cache_dir, date, run, fhr, field_key, sub, family))
        if result is None:
            return None, None, None
        data, la, lo = result
        if "convert" in prod:
            data = prod["convert"](data)
        return data, la, lo
    except Exception as e:
        print(f"[probe-rrfs] {prod.get('name')} F{fhr:03d}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None, None, None


def _probe_load_field(job, prod, date: str, run: int, fhr: int):
    """Load the same 2-D field a tile render would, for sampling.
    Supports recipe None and prob_window; returns (data, lats, lons)."""
    import numpy as np
    recipe = prod.get("recipe")
    if recipe == "prob_window":
        fhrs = job._window_fhrs(fhr, prod.get("window_h", 4))
        data = la = lo = None
        for wh in fhrs:
            try:
                f = job.proc.find_or_fetch(date, run, wh, prod["ftype"],
                                           lambda m: None)
            except Exception:
                continue
            d, la2, lo2 = job.proc.load_var(
                f, prod["var"], level=prod.get("level"),
                thresh=prod.get("thresh"),
                below=prod.get("prob_below", False))
            if d is None:
                continue
            data = d.astype(float) if data is None else np.fmax(data, d)
            if la is None:
                la, lo = la2, lo2
        return data, la, lo
    try:
        f = job.proc.find_or_fetch(date, run, fhr, prod["ftype"],
                                   lambda m: None)
    except Exception:
        return None, None, None
    step = prod.get("step") or (prod["step_from_fhr"](fhr)
                                if "step_from_fhr" in prod else None)
    data, la, lo = job.proc.load_var(
        f, prod["var"], level=prod.get("level"), step=step,
        thresh=prod.get("thresh"),
        below=prod.get("prob_below", False))
    if data is not None and "convert" in prod:
        data = prod["convert"](data)
    return data, la, lo


def _nearest_idx(la, lo, lonp, latp):
    """Nearest grid index for a lon/lat point, or None if off-grid."""
    import numpy as np
    if la.ndim == 1:
        lo2d, la2d = np.meshgrid(lo, la)
    else:
        lo2d, la2d = lo, la
    lon_pt = lonp % 360 if np.nanmax(lo2d) > 180 else lonp
    d2 = ((la2d - latp) ** 2
          + ((lo2d - lon_pt) * np.cos(np.radians(latp))) ** 2)
    iy, ix = np.unravel_index(np.nanargmin(d2), d2.shape)
    if d2[iy, ix] > 0.1 ** 2:
        return None
    return iy, ix


def probe_value(pid: str, date: str, run: int, fhr: int,
                fx: float, fy: float,
                region: str = "CONUS", bbox: tuple | None = None,
                sector_name: str = "", model: str = "refs") -> dict | None:
    """Sample the rendered field's value at an image-fraction position.

    (fx, fy) are fractions of the full tile image (fy from the TOP). We map
    them through the fixed MAP_BOX axes geometry and the region's Lambert
    projection to a lat/lon, then nearest-neighbor sample the same GRIB
    field the tile was rendered from.

    Supported: standard shaded products (recipe None) and prob_window.
    Other recipes (paintball/stamps/member composites) return None — the
    frontend simply shows no tooltip.
    """
    _ensure_initialized()
    core = _core
    prod = core.PRODUCTS.get(pid)
    if prod is None:
        return None
    recipe = prod.get("recipe")
    if recipe not in (None, "prob_window"):
        return None
    _is_spc = prod.get("source") == "spc_post"
    if _is_spc:
        job = _job_spc
    elif model == "href":
        job = _job_href
    else:
        job = _job_refs

    import numpy as np

    _CUSTOM = "__probe__"
    installed = False
    try:
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            core.REGIONS[_CUSTOM] = dict(
                lon=(lon_min, lon_max), lat=(lat_min, lat_max),
                proj_lon=(lon_min + lon_max) / 2.0,
                name=(sector_name or "Custom"),
            )
            region = _CUSTOM
            installed = True
        pt = _probe_lonlat(core, region, fx, fy)
        if pt is None:
            return None
        lonp, latp = pt
        data, la, lo = _probe_load_field(job, prod, date, run, fhr)
        if data is None:
            return None
        idx = _nearest_idx(la, lo, lonp, latp)
        if idx is None:
            return None
        val = data[idx]
        if not np.isfinite(val):
            return None
        return {
            "value": round(float(val), 2),
            "units": prod.get("units", ""),
            "lat": round(float(latp), 3),
            "lon": round(float(lonp if lonp <= 180 else lonp - 360), 3),
        }
    except Exception as e:
        print(f"[probe] {pid} F{fhr:03d}: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        if installed:
            core.REGIONS.pop(_CUSTOM, None)


def probe_series(pid: str, date: str, run: int, fhrs: list[int],
                 fx: float, fy: float,
                 region: str = "CONUS", bbox: tuple | None = None,
                 sector_name: str = "", model: str = "refs") -> dict | None:
    """Point meteogram: sample the product's field at one map position
    across many forecast hours. The decoded-field disk cache makes the
    per-fhr loads cheap whenever the product has been rendered (or
    prewarmed) for this cycle.

    Returns {lat, lon, units, name, points: [{fhr, value|null}, ...]}.
    """
    _ensure_initialized()
    core = _core
    prod = core.PRODUCTS.get(pid)
    if prod is None:
        return None
    _is_rrfs = prod.get("source") == "rrfs"
    if not _is_rrfs and prod.get("recipe") not in (None, "prob_window"):
        return None
    _is_spc = prod.get("source") == "spc_post"
    if _is_spc:
        job = _job_spc
    elif model == "href":
        job = _job_href
    else:
        job = _job_refs

    import numpy as np

    _CUSTOM = "__probe__"
    installed = False
    try:
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            core.REGIONS[_CUSTOM] = dict(
                lon=(lon_min, lon_max), lat=(lat_min, lat_max),
                proj_lon=(lon_min + lon_max) / 2.0,
                name=(sector_name or "Custom"),
            )
            region = _CUSTOM
            installed = True
        pt = _probe_lonlat(core, region, fx, fy)
        if pt is None:
            return None
        lonp, latp = pt
        idx = None              # nearest-index computed once; grid is fixed
        points = []
        for fhr in fhrs:
            val = None
            try:
                if _is_rrfs:
                    data, la, lo = _probe_load_field_rrfs(prod, date, run, fhr)
                else:
                    data, la, lo = _probe_load_field(job, prod, date, run, fhr)
                if data is not None:
                    if idx is None:
                        idx = _nearest_idx(la, lo, lonp, latp)
                        if idx is None:
                            return None     # off-grid: no series possible
                    v = data[idx]
                    if np.isfinite(v):
                        val = round(float(v), 2)
            except Exception:
                pass
            points.append({"fhr": int(fhr), "value": val})
        return {
            "lat": round(float(latp), 3),
            "lon": round(float(lonp if lonp <= 180 else lonp - 360), 3),
            "units": prod.get("units", ""),
            "name": prod.get("name", pid),
            "points": points,
        }
    except Exception as e:
        print(f"[probe-series] {pid}: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        if installed:
            core.REGIONS.pop(_CUSTOM, None)


def unproject_box(fxs: list[float], fys: list[float],
                  region: str = "CONUS", bbox: tuple | None = None,
                  sector_name: str = "") -> list:
    """Invert image-fraction points to (lon, lat) using the SAME Lambert
    projection + MAP_BOX axes geometry the tiles are rendered with.

    This is what the "draw a custom sector" flow uses: the frontend can't
    invert the Lambert Conformal projection itself (a naive linear/equirect
    inverse lands hundreds of km off, especially on CONUS-scale views), so it
    sends the drawn corners as image fractions and we do the real inverse here.

    Returns a list of (lon, lat) tuples aligned with the inputs; an entry is
    None if that point fell outside the map axes.
    """
    _ensure_initialized()
    core = _core
    _CUSTOM = "__unproj__"
    installed = False
    try:
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            core.REGIONS[_CUSTOM] = dict(
                lon=(lon_min, lon_max), lat=(lat_min, lat_max),
                proj_lon=(lon_min + lon_max) / 2.0,
                name=(sector_name or "Custom"),
            )
            region = _CUSTOM
            installed = True
        # Clamp the drawn corners to the map axes (MAP_BOX) so a drag that
        # spills a few px into the title/colorbar margins snaps to the map edge
        # instead of failing to invert. fy is measured from the TOP.
        left, bot, w, h = core.PlotManager.MAP_BOX
        fx_lo, fx_hi = left, left + w
        fy_lo, fy_hi = 1.0 - (bot + h), 1.0 - bot   # top-origin bounds
        out = []
        for fx, fy in zip(fxs, fys):
            cfx = min(max(fx, fx_lo), fx_hi)
            cfy = min(max(fy, fy_lo), fy_hi)
            out.append(_probe_lonlat(core, region, cfx, cfy))
        return out
    except Exception as e:
        print(f"[unproject] {type(e).__name__}: {e}", flush=True)
        return []
    finally:
        if installed:
            core.REGIONS.pop(_CUSTOM, None)


# REFS forecast composite-reflectivity contours for the verification panel.
# Contour levels mirror the MRMS observation overlay (25/40/55 dBZ) but use
# slightly different stops (20/35/50 dBZ) so the forecast curves don't bury
# the observation curves when both are visible. Dashed line style further
# distinguishes forecast from observed at a glance.
REFS_CONTOUR_LEVELS  = (20.0, 35.0, 50.0)
REFS_CONTOUR_WIDTHS  = (0.9, 1.3, 1.7)


def render_refs_contours(date: str, run: int, fhr: int,
                         region: str, theme: str = "dark",
                         bbox: tuple | None = None,
                         sector_name: str = "") -> bytes | None:
    """Render REFS composite-reflectivity (PMM) contour lines as a
    transparent WebP sized to the standard MAP_BOX.

    Used in side-by-side ("Compare") mode to overlay the forecast REFC
    contours on top of the MRMS observation panel — the eye can then
    judge whether the forecast storm shape matched what actually happened
    without flipping between products.

    Data source: same as the 'refc_pmmn_series' catalog entry
    (ftype='pmmn', var='refc'). Returns WebP bytes, or None if the GRIB
    record is unavailable for this fhour (e.g. PMM file not yet posted).
    """
    _ensure_initialized()
    core = _core
    job = _job_refs   # contours always come from REFS
    _CUSTOM = "__custom__"
    custom_region_installed = False
    effective_region = region
    try:
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            core.REGIONS[_CUSTOM] = dict(
                lon=(lon_min, lon_max),
                lat=(lat_min, lat_max),
                proj_lon=(lon_min + lon_max) / 2.0,
                name=(sector_name or "Custom"),
            )
            effective_region = _CUSTOM
            custom_region_installed = True
        if effective_region not in core.REGIONS:
            return None

        # Fetch raw REFS comp-ref PMM field. Don't go through PlotJob.render
        # because we want the raw 2D array, not a fully-rendered tile.
        try:
            f = job.proc.find_or_fetch(date, run, fhr, "pmmn", lambda m: None)
        except Exception as e:
            print(f"[refs_contours] cannot fetch pmmn for {date} {run:02d}z "
                  f"F{fhr:03d}: {type(e).__name__}: {e}", flush=True)
            return None
        if f is None:
            return None
        try:
            data, lats, lons = job.proc.load_var(f, "refc")
        except Exception as e:
            print(f"[refs_contours] load_var refc failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            return None
        if data is None:
            return None

        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import cartopy.crs as ccrs

        pm = core.PlotManager
        r = core.REGIONS[effective_region]
        proj = pm._projection(effective_region)
        fig = Figure(figsize=(pm.FIG_W, pm.FIG_H), dpi=pm.DPI)
        fig.patch.set_alpha(0)
        ax = fig.add_axes(pm.MAP_BOX, projection=proj)
        ax.set_extent([r["lon"][0], r["lon"][1], r["lat"][0], r["lat"][1]],
                      crs=ccrs.PlateCarree())
        # Strip everything that would draw a non-transparent background, so
        # the WebP stacks cleanly on top of the MRMS tile.
        ax.set_facecolor((0, 0, 0, 0))
        ax.patch.set_alpha(0)
        for s in ax.spines.values():
            s.set_visible(False)
        try:
            ax.spines["geo"].set_visible(False)
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([])

        # Theme-aware contour color ramp. Cool/cyan for the 20 dBZ rim,
        # neutral for 35, and a warm tone for the 50 dBZ severe core — all
        # tuned to read against either the dark or light MRMS palette
        # underneath without colliding with the MRMS contour colors.
        if theme == "light":
            colors = ("#0d8181", "#1c1c1c", "#a8005e")
        else:
            colors = ("#9defff", "#ffffff", "#ffaad4")

        if np.any(np.isfinite(data)):
            if lats.ndim == 1:
                lons2d, lats2d = np.meshgrid(lons, lats)
            else:
                lons2d, lats2d = lons, lats
            ax.contour(
                lons2d, lats2d, data,
                levels=list(REFS_CONTOUR_LEVELS),
                colors=list(colors),
                linewidths=list(REFS_CONTOUR_WIDTHS),
                linestyles="dashed",
                transform=ccrs.PlateCarree(),
                zorder=10,
            )

        FigureCanvasAgg(fig).draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="webp", transparent=True,
                    pil_kwargs={"quality": 92, "method": 4})
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        return buf.getvalue()
    except Exception as e:
        print(f"[refs_contours] {date} {run:02d}z F{fhr:03d}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None
    finally:
        if custom_region_installed:
            core.REGIONS.pop(_CUSTOM, None)  # noqa: contours-finally


def render_paintball_overlay(date: str, run: int, fhr: int,
                              var: str, thresh: float,
                              region: str, theme: str = "dark",
                              bbox: tuple | None = None,
                              sector_name: str = "") -> bytes | None:
    """Render per-member threshold contours as a transparent WebP overlay.

    Each of the 5 REFS members gets a distinct color from cmap_paintball().
    REFS members only — HREF has no individual member files.
    Returns None when no member data is available for this fhour.
    """
    _ensure_initialized()
    core = _core
    job  = _job_refs
    _CUSTOM = "__custom__"
    custom_region_installed = False
    effective_region = region
    try:
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            core.REGIONS[_CUSTOM] = dict(
                lon=(lon_min, lon_max),
                lat=(lat_min, lat_max),
                proj_lon=(lon_min + lon_max) / 2.0,
                name=(sector_name or "Custom"),
            )
            effective_region = _CUSTOM
            custom_region_installed = True
        if effective_region not in core.REGIONS:
            return None

        # window_h=4 matches the SPC HREF 4-hr-max convention used by the
        # full paintball products (REFS member records are 1-h).
        prod = dict(var=var, thresh=thresh, member_product='2dfld',
                    window_h=4)
        mems = job._load_members(prod, date, run, fhr, lambda m: None)
        if not mems:
            return None

        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import cartopy.crs as ccrs

        pm   = core.PlotManager
        r    = core.REGIONS[effective_region]
        proj = pm._projection(effective_region)
        fig  = Figure(figsize=(pm.FIG_W, pm.FIG_H), dpi=pm.DPI)
        fig.patch.set_alpha(0)
        ax = fig.add_axes(pm.MAP_BOX, projection=proj)
        ax.set_extent([r["lon"][0], r["lon"][1], r["lat"][0], r["lat"][1]],
                      crs=ccrs.PlateCarree())
        ax.set_facecolor((0, 0, 0, 0))
        ax.patch.set_alpha(0)
        for s in ax.spines.values():
            s.set_visible(False)
        try:
            ax.spines["geo"].set_visible(False)
        except Exception:
            pass
        ax.set_xticks([])
        ax.set_yticks([])

        colors    = core.cmap_paintball()
        any_drawn = False
        for i, (mem, data, lats, lons) in enumerate(mems):
            c = colors[i % len(colors)]
            if lats.ndim == 1:
                lons2d, lats2d = np.meshgrid(lons, lats)
            else:
                lons2d, lats2d = lons, lats
            if np.any(np.isfinite(data) & (data >= thresh)):
                ax.contour(
                    lons2d, lats2d, data,
                    levels=[float(thresh)],
                    colors=[c],
                    linewidths=[1.4],
                    transform=ccrs.PlateCarree(),
                    zorder=10 + i,
                )
                any_drawn = True

        if not any_drawn:
            return None

        FigureCanvasAgg(fig).draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="webp", transparent=True,
                    pil_kwargs={"quality": 92, "method": 4})
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        return buf.getvalue()
    except Exception as e:
        print(f"[paintball_overlay] {var}>={thresh} {date} {run:02d}z F{fhr:03d}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None
    finally:
        if custom_region_installed:
            core.REGIONS.pop(_CUSTOM, None)
