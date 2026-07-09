#!/usr/bin/env python3
# =============================================================================
#  REFS Ensemble Plotter (SPC HREF Style) v3.0
#  ---------------------------------------------------------------------------
#  Evaluation tool for NOAA's Rapid Ensemble Forecast System (REFS / RRFS)
#  Builds SPC HREF-style plots from the pre-computed REFS ensemble products
#  (mean, pmmn, lpmm, prob, avrg) hosted on the NOAA RRFS public S3 bucket.
#
#  Bucket layout (auto-detected from .idx files alongside grib2):
#    https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/refs.YYYYMMDD/HH/
#       refs.tHHz.mean.fXX.conus.grib2     <- ensemble mean
#       refs.tHHz.pmmn.fXX.conus.grib2     <- probability-matched mean
#       refs.tHHz.lpmm.fXX.conus.grib2     <- long-period PMM (multi-hour QPF)
#       refs.tHHz.prob.fXX.conus.grib2     <- neighborhood probabilities
#       refs.tHHz.avrg.fXX.conus.grib2     <- arithmetic mean (APCP)
#
#  Required packages (conda-forge):
#    cfgrib eccodes cartopy xarray requests scipy matplotlib pillow
#  Optional:
#    pygrib  (faster random-access load if eccodes records are sparse)
# =============================================================================

import os
import re
import sys
import time
import json
import threading
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
# Do NOT force ECCODES_DEFINITION_PATH -- let conda's eccodes auto-discover
if os.environ.get('ECCODES_DEFINITION_PATH') == '':
    del os.environ['ECCODES_DEFINITION_PATH']

import xarray as xr
import cfgrib

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pygrib
    HAS_PYGRIB = True
except ImportError:
    HAS_PYGRIB = False
    print("[refs_core] pygrib not available; using cfgrib (slower for "
          "multi-variable products). To enable pygrib, install a wheel "
          "compatible with libeccodes on the host.", flush=True)

# =============================================================================
#  Constants & defaults
# =============================================================================
APP_TITLE     = "REFS Viewer"
APP_VERSION   = "3.0"
S3_BASE_URL   = "https://noaa-rrfs-pds.s3.amazonaws.com"
S3_PREFIX_T   = "rrfs_public/refs.{date}/{run:02d}/enspost/"
S3_FNAME_T    = "refs.t{run:02d}z.{ftype}.f{fhr:02d}.conus.grib2"
# Individual member files (rrfs_a/rrfsens.YYYYMMDD/HH/m00X/)
S3_MEM_PREFIX_T = "rrfs_a/rrfsens.{date}/{run:02d}/m{mem:03d}/"
S3_MEM_FNAME_T  = "rrfs.t{run:02d}z.m{mem:03d}.{product}.3km.f{fhr:03d}.conus.grib2"
N_MEMBERS = 5    # currently 5 in the public test bucket; auto-detected at run time

# HREFv3 on NOMADS HTTPS
HREF_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/v3.1"
HREF_PREFIX_T = "href.{date}/ensprod/"
HREF_FNAME_T  = "href.t{run:02d}z.conus.{ftype}.f{fhr:02d}.grib2"

DEFAULT_LOCAL = os.environ.get("REFS_CACHE_DIR", "/tmp/refs_cache")
SETTINGS_FILE = Path.home() / ".refs_viewer_settings.json"


def find_latest_run(max_back_hours: int = 30):
    """Probe NOAA's bucket for the most recently published REFS run.

    Walks backwards from "now" in 6-hour steps and HEADs a known small product
    file (mean F006) to test availability. Returns (date, hour) on success
    or None if nothing in the last ``max_back_hours`` hours is available.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cur_cycle_hour = (now.hour // 6) * 6
    candidate = now.replace(hour=cur_cycle_hour)
    for _ in range(max_back_hours // 6 + 1):
        date_str = candidate.strftime("%Y%m%d")
        url = (
            f"{S3_BASE_URL}/"
            f"{S3_PREFIX_T.format(date=date_str, run=candidate.hour)}"
            f"refs.t{candidate.hour:02d}z.mean.f006.conus.grib2"
        )
        try:
            r = requests.head(url, timeout=6)
            if r.status_code == 200:
                return candidate.date(), candidate.hour
        except requests.RequestException:
            pass
        candidate -= timedelta(hours=6)
    return None

# =============================================================================
#  Theme tokens
# =============================================================================
THEMES = {
    'light': dict(
        bg='#EEEEEE', panel='#EEEEEE', card='#FAFAFA',
        fg='#1a1a1a', muted='#444444', accent='#1a3a6a',
        header_bg='#3a5a7a', header_fg='white', header_btn_fg='#cfe6ff',
        entry_bg='white', entry_fg='#1a1a1a',
        ok_bg='#2E8B57', ok_active='#3aa86b', ok_fg='white',
        danger_bg='#8B2222', danger_fg='white',
        player_bg='#1a2a3a', player_fg='white', player_btn='#3a5a7a',
        topbar_bg='#E6ECF2', topbar_fg='#1a3a6a',
        scroll_track='#d0d0d0',
        fig_face='white', fig_axes='white', map_land='#FAFAFA',
        map_ocean='#E8F0F7', map_state='#444444', map_border='#222222',
        map_county='#9aa0a8',
        # Contour colors – dark lines on the light figure background. No casing
        # (cnt_halo=None) so light mode matches its original clean appearance.
        cnt_h='#1a1a1a',     # geopotential height / isobar lines
        cnt_p='#1a1a1a',     # MSLP isobars
        cnt_prob='#222222',  # neighborhood probability contours
        cnt_spd='#2a2a2a',   # spread / secondary contours
        cnt_halo=None,       # light mode draws no contour casing
    ),
    'dark': dict(
        bg='#1e2329', panel='#262b33', card='#2a2f38',
        fg='#e6e8eb', muted='#9aa3b0', accent='#7fb2ff',
        header_bg='#2c3e5a', header_fg='#dde6f5', header_btn_fg='#8fb8ff',
        entry_bg='#1a1f26', entry_fg='#e6e8eb',
        ok_bg='#2E8B57', ok_active='#3aa86b', ok_fg='white',
        danger_bg='#a83b3b', danger_fg='white',
        player_bg='#11161d', player_fg='#dde6f5', player_btn='#2c3e5a',
        topbar_bg='#1a2330', topbar_fg='#cfe6ff',
        scroll_track='#1a1f26',
        fig_face='#1e2329', fig_axes='#262b33', map_land='#2f343d',
        map_ocean='#22303d', map_state='#a6b3c4', map_border='#cfd8e3',
        map_county='#7a8493',
        # Contour colors – white core + an opaque BLACK casing (cnt_halo,
        # applied by _halo()). The casing is the contrasting key: over bright
        # fills (yellow/red reflectivity, green QPF, white precip) the black
        # outline carries the line; over the dark map base the white core does.
        # One of the two always reads, on any background.
        cnt_h='#ffffff',     # geopotential height / isobar lines
        cnt_p='#ffffff',     # MSLP isobars
        cnt_prob='#ffffff',  # neighborhood probability contours
        cnt_spd='#f0f0f0',   # spread / secondary contours (near-white)
        cnt_halo='#000000',  # opaque black casing behind every contour line
    ),
}

# =============================================================================
#  Colormaps (SPC-style)
# =============================================================================
def _cmap(colors, levels, name='custom'):
    cm = ListedColormap(colors, name=name)
    return cm, BoundaryNorm(levels, cm.N), levels

def cmap_refc():
    # NWS reflectivity, 5 dBZ bins from 5 to 75
    cs = ['#04E9E7','#019FF4','#0300F4','#02FD02','#01C501','#008E00',
          '#FDF802','#E5BC00','#FD9500','#FD0000','#D40000','#BC0000',
          '#F800FD','#9854C6']
    lv = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    return _cmap(cs, lv, 'refc')

def cmap_uh():
    cs = ['#a8e6a3','#3cb371','#1e7b3a','#ffeb6b','#ff9a3c','#ff3b3b',
          '#c81e1e','#a020f0','#5a1b9a']
    lv = [25,50,75,100,150,200,250,300,400,500]
    return _cmap(cs, lv, 'uh')

def cmap_cape():
    cs = ['#e8e8e8','#bababa','#8a8a8a','#5fa8d3','#1f5fc7',
          '#a6f0a6','#3cb44b','#ffe66d','#ff8c1a','#ff3838',
          '#bf1a8a','#7a0eb0']
    lv = [100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000]
    return _cmap(cs, lv, 'cape')

def cmap_mucape():
    return cmap_cape()

def cmap_cin():
    cs = ['#430050','#7b0a8a','#b03ec5','#e07ff0','#ffb3f0',
          '#ffd9b3','#ffb060','#ff7a1f','#d63a1f','#7a0000']
    lv = [-800,-500,-300,-200,-150,-100,-75,-50,-25,-10,-1]
    return _cmap(cs, lv, 'cin')

def cmap_qpf():
    # SPC QPF palette (inches)
    cs = ['#a0f0a0','#3eb43e','#1f8a1f','#1054a8','#3a9aff','#33d6ee',
          '#15f0f0','#9c75ff','#7a1ddc','#a80000','#d40000','#ff5a1f',
          '#ff9a3c','#c97a00','#ffc828','#ffff64','#ffd0e0','#ffffff']
    lv = [0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
          4.0,5.0,6.0,8.0,10.0,15.0]
    return _cmap(cs, lv, 'qpf')

def cmap_prob():
    cs = ['#cfdfff','#9bbcff','#5b8aff','#1f55e8','#0033bf',
          '#5a18a8','#9015c5','#c01fd6','#e72ff5','#ff00ff']
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(cs, lv, 'prob')

# SPC's neighborhood-probability fill convention: gray → blue → green →
# yellow → orange → red → magenta. Same level edges as the default prob
# ramp so colorbars and contour labels line up product-to-product; only
# the colors change. Selected via the 'SPC Ramp' palette.
def cmap_prob_spc():
    cs = ['#bdbdbd','#7da7d9','#3d76e0','#2ca02c','#8cd17d',
          '#ffe600','#ffaa00','#ff5500','#e00000','#c800c8']
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(cs, lv, 'prob_spc')

def cmap_wind500():
    cs = ['#d8d8d8','#a8a8a8','#7fbfff','#1f7be8','#0033bf',
          '#9bd87a','#2a8a2a','#ffe66d','#d68a30','#ff9a9a','#a01fa0']
    lv = [30,40,50,60,70,80,100,120,140,160,180]
    return _cmap(cs, lv, 'w500')

def cmap_wind250():
    cs = ['#d0d0d0','#a8a8a8','#7fbfff','#1f7be8','#0033bf',
          '#9bd87a','#2a8a2a','#ffe66d','#d68a30','#ff5a5a','#a01fa0','#5a0a7a']
    lv = [50,70,90,110,130,150,170,190,210,230,250,280]
    return _cmap(cs, lv, 'w250')

def cmap_pwat():
    # Legacy mm levels — kept so any caller that still references 'pwat'
    # gets the original behavior. New PWAT products use 'pwat_in'.
    cs = ['#d8b890','#c6a070','#bfe6bf','#62b262','#1f7a1f',
          '#8a3aff','#c91fc9','#ff1f80','#ff5a1f']
    lv = [10,15,20,25,30,40,50,60,75,90]
    return _cmap(cs, lv, 'pwat')

def cmap_pwat_in():
    # PWAT in inches. Tan (dry) → green (moist) → blue (very moist) →
    # purple/magenta (tropical). Levels chosen to highlight the
    # convectively-meaningful 1.0-2.0" band that PWAT is mostly read for.
    cs = ['#c9a97a','#dccaa0','#bfe6bf','#62b262','#1f7a1f',
          '#5b8aff','#1f3fbf','#8a3aff','#c91fc9','#ff1f80']
    lv = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
    return _cmap(cs, lv, 'pwat_in')

def cmap_retop_kft():
    # Echo-top heights, kft. Cold blues at low heights → warm magenta at
    # overshooting-top altitudes. Deliberately distinct from the dBZ
    # reflectivity palette so viewers don't misread height as intensity.
    cs = ['#cfe4ff','#7fbfff','#1f7be8','#3cb43c','#ffe66d',
          '#ff9a3c','#d63a1f','#a01fa0','#5a0a7a']
    lv = [5, 10, 20, 30, 40, 50, 55, 60, 65, 70]
    return _cmap(cs, lv, 'retop_kft')

def cmap_srh():
    # 0-3 km storm-relative helicity, m²/s². Standard severe-wx thresholds:
    # 100 = supercell-favorable, 200 = strong tornadoes possible, 400+ = sig
    # tornado environment. Bins resolve operational decision space.
    cs = ['#e8e8e8','#cfe4ff','#a6f0a6','#3cb43c','#ffe66d',
          '#ff9a3c','#d63a1f','#a01fa0']
    lv = [50, 100, 150, 200, 300, 400, 500, 750]
    return _cmap(cs, lv, 'srh')

def cmap_t2m():
    # cold→warm continuous-ish, degF
    cs = ['#5a0094','#3b1fb0','#1f50d4','#1f8de8','#33d6ee',
          '#a6f0a6','#3cb43c','#ffe66d','#ff9a3c','#ff3838',
          '#bf1a8a','#7a0eb0']
    lv = [-20,0,10,20,32,40,50,60,70,80,90,100,110]
    return _cmap(cs, lv, 't2m')

def cmap_td2m():
    cs = ['#6b3210','#8a5a2a','#bda06a','#dbc999','#e0e0e0',
          '#a6e6c8','#3cb471','#1f8a4b','#0a5a2a','#9aff9a']
    lv = [-20,0,20,30,40,50,55,60,65,70,80]
    return _cmap(cs, lv, 'td2m')

# --- Alternate palettes -----------------------------------------------------
# Returned by the same level set as the SPC defaults so swapping is harmless.
def _mpl_palette(name, levels, under=None):
    """Build a discrete cmap from an mpl colormap sampled at len(levels)+1."""
    import matplotlib.cm as mcm
    base = mcm.get_cmap(name, len(levels))
    cs = [matplotlib.colors.to_hex(base(i)) for i in range(len(levels))]
    if under: cs[0] = under
    return cs

def cmap_refc_modern():
    lv = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    return _cmap(_mpl_palette('turbo', lv), lv, 'refc_mod')

def cmap_refc_cb():
    lv = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    return _cmap(_mpl_palette('cividis', lv), lv, 'refc_cb')

def cmap_prob_modern():
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(_mpl_palette('viridis', lv), lv, 'prob_mod')

def cmap_prob_cb():
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(_mpl_palette('cividis', lv), lv, 'prob_cb')

def cmap_cape_modern():
    lv = [100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000]
    return _cmap(_mpl_palette('plasma', lv), lv, 'cape_mod')

def cmap_cape_cb():
    lv = [100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000]
    return _cmap(_mpl_palette('cividis', lv), lv, 'cape_cb')

def cmap_qpf_modern():
    lv = [0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
          4.0,5.0,6.0,8.0,10.0,15.0]
    return _cmap(_mpl_palette('viridis', lv), lv, 'qpf_mod')

def cmap_qpf_cb():
    lv = [0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
          4.0,5.0,6.0,8.0,10.0,15.0]
    return _cmap(_mpl_palette('cividis', lv), lv, 'qpf_cb')

# Current palette (module-level; PaletteManager mutates this and the registry).
PALETTE = 'SPC Classic'
def _palette_from(mname):
    """Factory: build refc/prob/cape/qpf cmaps from a single mpl colormap."""
    def mk(levels, name):
        return lambda: _cmap(_mpl_palette(mname, levels), levels, name)
    return dict(
        refc=mk([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75], 'refc_'+mname),
        prob=mk([5,10,20,30,40,50,60,70,80,90,100],            'prob_'+mname),
        cape=mk([100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000],
                                                                'cape_'+mname),
        qpf =mk([0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
                 4.0,5.0,6.0,8.0,10.0,15.0], 'qpf_'+mname),
    )

PALETTES = ('Default',
            'SPC Ramp',
            'High Contrast',
            'Spectral')


# ---- High Contrast family --------------------------------------------------
# A set of minimal, bold palettes inspired by modern weather visualizations
# (AG-WX, Pivotal, GR2Analyst): a few saturated stops with clean transitions
# rather than a continuous ramp. They pair well with the white-halo contours
# we use for overlays and the minimal basemap toggles default OFF.
#
# Variants:
#   High Contrast        — balanced full-spectrum (default high-contrast look)
#   High Contrast Warm   — sunset / fire ramp; emphasizes intensity over hue
#   High Contrast Cool   — arctic / ocean ramp; calmer, document-friendly
#   Neon Storm           — saturated neon stops on dark; dramatic, dark-theme
def cmap_refc_hicon():
    cs = ['#7fc4f5','#4ea3e8','#2c7fcf','#5dc972','#2e9b3f',
          '#1a6b29','#fdd744','#f7a523','#ef5a1e','#d4271a',
          '#a8132f','#7b1043','#aa3aa5','#7c2a8a','#4a1265']
    lv = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    return _cmap(cs, lv, 'refc_hicon')

def cmap_prob_hicon():
    cs = ['#e6f1ff','#b9d4ff','#7faaff','#3d76e0','#1a4ac4',
          '#f8a73a','#ef5a1e','#c41a2e','#a01a9a','#5d0e7a']
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(cs, lv, 'prob_hicon')

def cmap_cape_hicon():
    cs = ['#eef0f3','#cfdce8','#9fbcd6','#3f80c7','#1f4ea3',
          '#5fc472','#1f8a3a','#fdd744','#f3892c','#d4271a',
          '#a8132f','#5d0e7a']
    lv = [100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000]
    return _cmap(cs, lv, 'cape_hicon')

def cmap_qpf_hicon():
    cs = ['#bfe6d2','#5fc472','#1f8a3a','#5fa5f0','#2c7fcf',
          '#1a4ac4','#7fb8e8','#4ea3e8','#aa3aa5','#7b1043',
          '#a8132f','#d4271a','#ef5a1e','#f7a523','#fdd744',
          '#fef3a8','#fae8e8','#ffffff']
    lv = [0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
          4.0,5.0,6.0,8.0,10.0,15.0]
    return _cmap(cs, lv, 'qpf_hicon')


# --- High Contrast Warm — single-direction intensity ramp -------------------
# Deep purple → magenta → red → orange → yellow → cream. Reads as a
# "heat" gradient regardless of which field you apply it to; great for
# making one signal jump out without the rainbow ambiguity.
def cmap_refc_hcwarm():
    cs = ['#330033','#5a1657','#7a1f64','#9c2667','#c0265d',
          '#dc3d4e','#ef5a1e','#f7872f','#f7a523','#fdd744',
          '#ffe98a','#fff2b5','#fff7d6','#fffae8','#ffffff']
    lv = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    return _cmap(cs, lv, 'refc_hcwarm')

def cmap_prob_hcwarm():
    cs = ['#2a0a3a','#5a1657','#7a1f64','#9c2667','#c0265d',
          '#dc3d4e','#ef5a1e','#f7872f','#f7a523','#fdd744','#ffffff']
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(cs, lv, 'prob_hcwarm')

def cmap_cape_hcwarm():
    cs = ['#1f0a26','#2a0a3a','#5a1657','#7a1f64','#9c2667',
          '#c0265d','#dc3d4e','#ef5a1e','#f7872f','#f7a523',
          '#fdd744','#ffe98a','#ffffff']
    lv = [100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000]
    return _cmap(cs, lv, 'cape_hcwarm')

def cmap_qpf_hcwarm():
    cs = ['#fff2b5','#fdd744','#f7a523','#f7872f','#ef5a1e',
          '#dc3d4e','#c0265d','#9c2667','#7a1f64','#5a1657',
          '#3a0e4a','#2a0a3a','#1f0a26','#3d1a4e','#5a1657',
          '#9c2667','#dc3d4e','#ffffff']
    lv = [0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
          4.0,5.0,6.0,8.0,10.0,15.0]
    return _cmap(cs, lv, 'qpf_hcwarm')


# --- High Contrast Cool — calm, document-friendly ---------------------------
# Deep navy → blue → cyan → teal → mint → cream. A "cool" intensity ramp
# that prints well and stays calm against busy basemaps. Good default for
# presentations and PDFs.
def cmap_refc_hccool():
    cs = ['#0a1a3a','#142a5a','#1d3f80','#2659a4','#2f74c0',
          '#3f8fd0','#5aa9d8','#7fc5d4','#a8dcc8','#c5e8b8',
          '#dceeae','#ecf3b0','#f5f7c6','#fafce4','#ffffff']
    lv = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    return _cmap(cs, lv, 'refc_hccool')

def cmap_prob_hccool():
    cs = ['#0a1a3a','#142a5a','#1d3f80','#2659a4','#2f74c0',
          '#3f8fd0','#5aa9d8','#7fc5d4','#a8dcc8','#c5e8b8','#ffffff']
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(cs, lv, 'prob_hccool')

def cmap_cape_hccool():
    cs = ['#050d1f','#0a1a3a','#142a5a','#1d3f80','#2659a4',
          '#2f74c0','#3f8fd0','#5aa9d8','#7fc5d4','#a8dcc8',
          '#c5e8b8','#dceeae','#ffffff']
    lv = [100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000]
    return _cmap(cs, lv, 'cape_hccool')

def cmap_qpf_hccool():
    cs = ['#fafce4','#dceeae','#c5e8b8','#a8dcc8','#7fc5d4',
          '#5aa9d8','#3f8fd0','#2f74c0','#2659a4','#1d3f80',
          '#142a5a','#0a1a3a','#050d1f','#152040','#2f74c0',
          '#5aa9d8','#a8dcc8','#ffffff']
    lv = [0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
          4.0,5.0,6.0,8.0,10.0,15.0]
    return _cmap(cs, lv, 'qpf_hccool')


# --- Neon Storm — saturated stops, intense on dark themes -------------------
# Electric blue → neon green → neon yellow → hot pink → magenta. Pure pop;
# best with dark theme. Use for showcase / social-share aesthetics.
def cmap_refc_neon():
    cs = ['#00d4ff','#00ffea','#00ff88','#7dff2a','#c8ff1f',
          '#fff200','#ffb000','#ff6a00','#ff2222','#ff1a8c',
          '#ff00d4','#c800ff','#7a00ff','#ff66ff','#ffffff']
    lv = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    return _cmap(cs, lv, 'refc_neon')

def cmap_prob_neon():
    cs = ['#001f3a','#00d4ff','#00ff88','#c8ff1f','#fff200',
          '#ffb000','#ff6a00','#ff2222','#ff00d4','#c800ff','#ffffff']
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(cs, lv, 'prob_neon')

def cmap_cape_neon():
    cs = ['#001f3a','#003a5a','#00d4ff','#00ffea','#00ff88',
          '#7dff2a','#c8ff1f','#fff200','#ffb000','#ff6a00',
          '#ff2222','#ff00d4','#ffffff']
    lv = [100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000]
    return _cmap(cs, lv, 'cape_neon')

def cmap_qpf_neon():
    cs = ['#00ff88','#7dff2a','#c8ff1f','#00d4ff','#00ffea',
          '#fff200','#ffb000','#ff6a00','#ff1a8c','#ff00d4',
          '#c800ff','#7a00ff','#ff2222','#ff6a00','#ffb000',
          '#fff200','#7dff2a','#ffffff']
    lv = [0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
          4.0,5.0,6.0,8.0,10.0,15.0]
    return _cmap(cs, lv, 'qpf_neon')


# ---- Twilight Storm --------------------------------------------------------
# Moody storm-sky aesthetic: deep navy → indigo → magenta → amber → cream.
# Distinct from the existing mpl "Twilight" entry (which is a cyclic ramp).
def cmap_refc_twstorm():
    cs = ['#1c2541','#2e3a72','#4b3b8a','#704a9c','#9954a0',
          '#bf5a8e','#dc6275','#ec7257','#f08c3c','#f0a82c',
          '#ecc342','#e7d785','#dfe6c4','#ffffff','#fbe9ff']
    lv = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    return _cmap(cs, lv, 'refc_twstorm')

def cmap_prob_twstorm():
    cs = ['#1c2541','#2e3a72','#4b3b8a','#704a9c','#9954a0',
          '#bf5a8e','#dc6275','#ec7257','#f08c3c','#f0a82c','#fbe9ff']
    lv = [5,10,20,30,40,50,60,70,80,90,100]
    return _cmap(cs, lv, 'prob_twstorm')

def cmap_cape_twstorm():
    cs = ['#0e1a30','#1c2541','#2e3a72','#4b3b8a','#704a9c',
          '#9954a0','#bf5a8e','#dc6275','#ec7257','#f08c3c',
          '#f0a82c','#ecc342','#fbe9ff']
    lv = [100,250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,8000]
    return _cmap(cs, lv, 'cape_twstorm')

def cmap_qpf_twstorm():
    cs = ['#1c2541','#2e3a72','#3b5099','#4b6fb5','#5d8ec8',
          '#7aabd6','#4b3b8a','#704a9c','#9954a0','#bf5a8e',
          '#dc6275','#ec7257','#f08c3c','#f0a82c','#ecc342',
          '#e7d785','#dfe6c4','#ffffff']
    lv = [0.01,0.10,0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,
          4.0,5.0,6.0,8.0,10.0,15.0]
    return _cmap(cs, lv, 'qpf_twstorm')

def apply_palette(name):
    """Swap palette-aware entries in the _CMAPS dispatch dict in-place."""
    global PALETTE
    PALETTE = name
    mapping = {
        'Default':                    {'refc': cmap_refc, 'prob': cmap_prob,
                                       'cape': cmap_cape, 'qpf':  cmap_qpf},
        # Back-compat aliases (old saved URLs / localStorage)
        'SPC Classic':                {'refc': cmap_refc, 'prob': cmap_prob,
                                       'cape': cmap_cape, 'qpf':  cmap_qpf},
        # SPC's NP page color convention — only the prob ramp changes, so
        # REFS NP tiles compare 1:1 against SPC HREF imagery.
        'SPC Ramp':                   {'refc': cmap_refc, 'prob': cmap_prob_spc,
                                       'cape': cmap_cape, 'qpf':  cmap_qpf},
        'High Contrast':              {'refc': cmap_refc_hicon, 'prob': cmap_prob_hicon,
                                       'cape': cmap_cape_hicon, 'qpf':  cmap_qpf_hicon},
        'High Contrast Warm':         {'refc': cmap_refc_hcwarm, 'prob': cmap_prob_hcwarm,
                                       'cape': cmap_cape_hcwarm, 'qpf':  cmap_qpf_hcwarm},
        'High Contrast Cool':         {'refc': cmap_refc_hccool, 'prob': cmap_prob_hccool,
                                       'cape': cmap_cape_hccool, 'qpf':  cmap_qpf_hccool},
        'Neon Storm':                 {'refc': cmap_refc_neon, 'prob': cmap_prob_neon,
                                       'cape': cmap_cape_neon, 'qpf':  cmap_qpf_neon},
        'Twilight Storm':             {'refc': cmap_refc_twstorm, 'prob': cmap_prob_twstorm,
                                       'cape': cmap_cape_twstorm, 'qpf':  cmap_qpf_twstorm},
        'Modern (Viridis/Turbo)':     {'refc': cmap_refc_modern, 'prob': cmap_prob_modern,
                                       'cape': cmap_cape_modern, 'qpf':  cmap_qpf_modern},
        'Colorblind Safe (Cividis)':  {'refc': cmap_refc_cb, 'prob': cmap_prob_cb,
                                       'cape': cmap_cape_cb, 'qpf':  cmap_qpf_cb},
        'Magma':    _palette_from('magma'),
        'Plasma':   _palette_from('plasma'),
        'Inferno':  _palette_from('inferno'),
        'Twilight': _palette_from('twilight_shifted'),
        'Coolwarm': _palette_from('coolwarm'),
        'Spectral': _palette_from('Spectral_r'),
    }
    for k, v in mapping.get(name, mapping['Default']).items():
        _CMAPS[k] = v

def cmap_spread_cape():
    cs = ['#f4f4f4','#d0e7d0','#9bd87a','#3a9a3a','#ffe66d',
          '#ff9a3c','#d63a1f','#7a0000','#3a005a']
    lv = [50,100,200,300,500,750,1000,1500,2000]
    return _cmap(cs, lv, 'spcape')

def cmap_spread_hgt():
    cs = ['#f4f4f4','#d6e6f7','#a8c8e8','#5b8aff','#1f55e8',
          '#9015c5','#c01fd6','#ff00ff']
    lv = [3,5,8,12,16,20,30,40,60]
    return _cmap(cs, lv, 'sphgt')

def cmap_spread_mslp():
    cs = ['#f4f4f4','#e8e8d8','#d2c9a8','#bf8a3a','#a04a1a',
          '#7a1a1a','#3a005a']
    lv = [1,2,3,4,5,7,10,15]
    return _cmap(cs, lv, 'spmslp')

def cmap_clouds():
    cs = ['#f6f6f6','#dedede','#c4c4c4','#a8a8a8','#8c8c8c',
          '#707070','#545454','#383838']
    lv = [10,20,30,40,50,60,70,80,90]
    return _cmap(cs, lv, 'clouds')

def cmap_vis():
    cs = ['#7a0000','#bf1a1a','#ff7a3c','#ffce5a','#fff36d',
          '#9bd87a','#3a9a3a','#a8d8e8','#e0e0e0']
    lv = [0.25,0.5,1,2,3,4,6,8,10,15]
    return _cmap(cs, lv, 'vis')

def cmap_snow():
    cs = ['#c8e6ff','#9bbfff','#5b8aff','#1f55e8','#0033bf',
          '#5a18a8','#9015c5','#c01fd6','#e72ff5','#ff00ff','#ffbfff']
    lv = [0.1,0.5,1,2,4,6,8,10,15,20,30,48]
    return _cmap(cs, lv, 'snow')

def cmap_paintball():
    return ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
            '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

# --- Composite-parameter palettes ----------------------------------------
# These cover the SPC-mesoanalysis-style composite indices (SCP, STP, EHI,
# SHIP). All share a single shape: faint → green → yellow → orange →
# magenta → deep red so any value ≥ 1 (the "significant" threshold for
# STP / EHI / SHIP) jumps out visually.
def cmap_composite():
    cs = ['#f3f3f3','#d6efd6','#9bd87a','#3a9a3a','#ffe66d',
          '#ff9a3c','#d63a1f','#a01fa0','#5a0a7a','#1a0030']
    lv = [0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30]
    return _cmap(cs, lv, 'composite')

# 700-500 mb lapse rate in K/km. Most environments are 5-7 K/km; steep
# lapse rates (>7) are convectively/hail-supportive and >8 is dry-adiabatic.
def cmap_lapse_rate():
    cs = ['#cfdfff','#bfe0bf','#9bd87a','#3a9a3a','#ffe66d',
          '#ff9a3c','#d63a1f','#a01fa0','#5a0a7a']
    lv = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0]
    return _cmap(cs, lv, 'lapse')

# Divergence palette (10^-5 / s). Diverging around 0 — blue = convergence,
# red = divergence. Useful for the 250-mb storm-top divergence proxy.
def cmap_divergence():
    cs = ['#0a005a','#1f55e8','#7fbfff','#cfe4ff','#f4f4f4',
          '#ffd9b3','#ff7a3c','#d63a1f','#7a0000']
    lv = [-30, -20, -10, -5, -1, 1, 5, 10, 20, 30]
    return _cmap(cs, lv, 'div')

# Absolute vorticity (10^-5 / s) — matches NAM vorticity chart style.
# Green→yellow→orange→red→black ramp, positive only (cyclonic).
def cmap_vorticity():
    cs = ['#d4f5d4','#7edc7e','#00bb00','#ffff00',
          '#ffbf00','#ff6600','#e00000','#880000','#000000','#5a00aa','#1a0050']
    lv = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80]
    return _cmap(cs, lv, 'vort')

# Hail diameter in inches. NWS severe = 1.0", significant severe = 2.0".
# Member 2dfld field is in m (units of meters of equivalent ice diameter),
# converted to inches at product level.
def cmap_hail():
    cs = ['#cfe4ff','#7fbfff','#3a9aff','#9bd87a','#3cb43c',
          '#ffe66d','#ff9a3c','#d63a1f','#a01fa0','#5a0a7a']
    lv = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    return _cmap(cs, lv, 'hail')

# Vertically Integrated Liquid (kg/m²). VIL > 30 is severe-storm signature,
# > 50 = significant hail signature.
def cmap_vil():
    cs = ['#9bd87a','#3cb43c','#1f7a1f','#ffe66d','#ff9a3c',
          '#d63a1f','#a01fa0','#5a0a7a','#1a0030']
    lv = [0.5, 1, 5, 10, 20, 30, 40, 50, 60, 75]
    return _cmap(cs, lv, 'vil')

# Surface wind gust in knots.
def cmap_gust():
    cs = ['#d8d8d8','#a8d8e8','#5b8aff','#1f55e8','#9bd87a',
          '#3cb43c','#ffe66d','#ff9a3c','#d63a1f','#a01fa0','#5a0a7a']
    lv = [10, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
    return _cmap(cs, lv, 'gust')

# DCAPE (downdraft CAPE), J/kg. Higher = stronger wet-microburst potential.
def cmap_dcape():
    cs = ['#f3f3f3','#cfe4ff','#7fbfff','#1f55e8','#9bd87a',
          '#3cb43c','#ffe66d','#ff9a3c','#d63a1f','#a01fa0']
    lv = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500]
    return _cmap(cs, lv, 'dcape')

# Mass convergence (column-integrated). Diverging palette: red = convergence
# (where storms form), blue = divergence. Stored as 10⁻⁵ / s after scaling.
#
# Levels widened to ±200 because raw native-grid mconv routinely exceeds the
# old ±50 cap (everything saturated to the end colors). Products that use
# this cmap should also set `smooth_sigma` to suppress salt-and-pepper, and
# `mask_below_abs` to hide the meteorologically-noisy near-zero band.
def cmap_mconv():
    cs = ['#0a3a8a','#1f55e8','#7fbfff','#cfe4ff','#ffffff',
          '#ffd9b3','#ff7a3c','#d63a1f','#a01a1a','#5a0030']
    lv = [-200, -100, -50, -20, -5, 5, 20, 50, 100, 200]
    return _cmap(cs, lv, 'mconv')

# IR-enhanced simulated satellite (brightness temperature, °C).
# Mirrors the NCEP/SPC convention used on operational GOES IR products:
# warm surface = dark, lower-mid clouds = grey, then rainbow + magenta
# enhancements through the convective cloud-top temperature range. The
# product converts the raw Kelvin field to Celsius before mapping.
def cmap_ir_satellite():
    cs = [
        '#9b3a8b',   # -90 to -80 °C : magenta (overshooting-top cores)
        '#dcdcdc',   # -80 to -70 °C : light grey (extreme cold anvil)
        '#0a0a0a',   # -70 to -60 °C : near-black highlight band
        '#dc143c',   # -60 to -50 °C : red
        '#ff8c00',   # -50 to -40 °C : orange
        '#ffff00',   # -40 to -30 °C : yellow
        '#00c800',   # -30 to -20 °C : green
        '#00d2ff',   # -20 to -10 °C : cyan
        '#aaaaaa',   # -10 to   0 °C : light grey (mid/high clouds)
        '#666666',   #   0 to  10 °C : medium grey (low cloud)
        '#333333',   #  10 to  20 °C : dark grey (surface, cool)
        '#1a1a1a',   #  20 to  30 °C : near black
        '#000000',   #  30 to  40 °C : black (warm surface)
    ]
    lv = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40]
    return _cmap(cs, lv, 'ir')

# Wildfire potential (NCEP "WFIREPOT", dimensionless 0-1 probability scale).
def cmap_wfirepot():
    cs = ['#f3f3e6','#fff0a0','#ffd060','#ff9a30','#ef5a20',
          '#c91f1f','#7a0000','#3a0000']
    lv = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 0.95]
    return _cmap(cs, lv, 'wfire')

# Storm motion speed in knots — vector arrows overlaid on top.
def cmap_smotion():
    cs = ['#f6f6f6','#dedede','#a8d8e8','#5b8aff','#1f55e8',
          '#9bd87a','#3cb43c','#ffe66d','#ff9a3c','#d63a1f']
    lv = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
    return _cmap(cs, lv, 'smot')

# Registry of composite-parameter computation functions, populated by
# app/extra_products.py at startup. Each entry maps a string key
# (referenced by PRODUCTS[pid]['composite_fn']) to a callable
#     f(ingredients: dict[str, ndarray], **kw) -> 2-D ndarray
# where ``ingredients`` holds the loaded 2-D fields keyed by the names
# declared in PRODUCTS[pid]['ingredients'].
COMPOSITES: dict = {}

# =============================================================================
#  Variable specs
#  ---------------------------------------------------------------------------
#  REFS uses NCEP local GRIB tables.  Many variables resolve to shortName
#  'unknown' in cfgrib because eccodes' default tables lack them.  We address
#  every variable by (parameterCategory, parameterNumber, typeOfLevel,
#  level[, topLevel/bottomLevel]) which is unambiguous.
#
#  Scaling: probability messages use scaledValueOfUpperLimit / 10^scaleFactor;
#  REFS uses scaleFactor=3 throughout, so the divisor is 1000.
# =============================================================================
VARSPEC = {
    # name: dict(cat, parm, tlev[, lev, top, bot, ...])
    'cape_sfc':   dict(cat=7, parm=6, tlev='surface'),
    'cape_ml':    dict(cat=7, parm=6, tlev='pressureFromGroundLayer', lev=9000),
    'cape_mu':    dict(cat=7, parm=6, tlev='pressureFromGroundLayer', lev=18000),
    'cin_sfc':    dict(cat=7, parm=7, tlev='surface'),
    'cin_ml':     dict(cat=7, parm=7, tlev='pressureFromGroundLayer', lev=9000),
    'srh_3km':    dict(cat=7, parm=8, tlev='heightAboveGroundLayer', lev=3000, top=3000, bot=0),
    # vwsh_06km, maxuvv_lyr: cfgrib's `level` attribute for these layer-type
    # records doesn't equal the lev= values we used to pass (cfgrib picks one
    # endpoint of the layer, not always the one we expect), so the filter was
    # rejecting every record. cat+parm+tlev is unique within the file — rely
    # on that. top/bot kept for documentation only.
    'vwsh_sfc':   dict(cat=2, parm=192, tlev='surface'),
    'vwsh_06km':  dict(cat=2, parm=192, tlev='heightAboveGroundLayer', top=6000, bot=0),
    'mxuphl_03':  dict(cat=7, parm=199, tlev='heightAboveGroundLayer', top=3000, bot=0),
    'maxuvv_lyr': dict(cat=2, parm=220, tlev='isobaricLayer', top=10000, bot=100000),
    'mxuphl_25':  dict(cat=7, parm=199, tlev='heightAboveGroundLayer', lev=5000, top=5000, bot=2000),
    'refc':       dict(cat=16, parm=5,  tlev='atmosphereSingleLayer'),
    'retop':      dict(cat=16, parm=3,  tlev='atmosphereSingleLayer'),
    'refd_1km':   dict(cat=16, parm=4,  tlev='heightAboveGround', lev=1000),
    'maxref_1km': dict(cat=16, parm=198,tlev='heightAboveGround', lev=1000),
    'tp_sfc':     dict(cat=1, parm=8,   tlev='surface'),
    'asnow':      dict(cat=1, parm=29,  tlev='surface'),
    't_2m':       dict(cat=0, parm=0,   tlev='heightAboveGround', lev=2),
    'd_2m':       dict(cat=0, parm=6,   tlev='heightAboveGround', lev=2),
    't_lvl':      dict(cat=0, parm=0,   tlev='isobaricInhPa'),
    'dpt_lvl':    dict(cat=0, parm=6,   tlev='isobaricInhPa'),
    'rh_lvl':     dict(cat=1, parm=1,   tlev='isobaricInhPa'),
    'u_lvl':      dict(cat=2, parm=2,   tlev='isobaricInhPa'),
    'v_lvl':      dict(cat=2, parm=3,   tlev='isobaricInhPa'),
    'gh_lvl':     dict(cat=3, parm=5,   tlev='isobaricInhPa'),
    'wind_lvl':   dict(cat=2, parm=1,   tlev='isobaricInhPa'),  # WIND speed magnitude
    'u_10m':      dict(cat=2, parm=2,   tlev='heightAboveGround', lev=10),
    'v_10m':      dict(cat=2, parm=3,   tlev='heightAboveGround', lev=10),
    'si_10m':     dict(cat=2, parm=1,   tlev='heightAboveGround', lev=10),
    'mslp':       dict(cat=3, parm=192, tlev='meanSea'),         # MSLET
    'pwat':       dict(cat=1, parm=3,   tlev='atmosphereSingleLayer'),
    'ltng':       dict(cat=17, parm=192, tlev='atmosphere'),
    'gh_sfc':     dict(cat=3, parm=5,   tlev='surface'),
    # Geometric vertical velocity (dz/dt, m/s). Positive = upward.
    # Only published in the REFS mean file at 700 mb.
    'dzdt_lvl':   dict(cat=2, parm=9,   tlev='isobaricInhPa'),
    # Categorical precipitation type flags (binary ensemble-mean probabilities).
    'crain':      dict(cat=1, parm=192, tlev='surface'),
    'csnow':      dict(cat=1, parm=195, tlev='surface'),
    'cfrzr':      dict(cat=1, parm=193, tlev='surface'),
    'cicep':      dict(cat=1, parm=194, tlev='surface'),
    'tcdc':       dict(cat=6, parm=1,   tlev='atmosphereSingleLayer'),
    'lcdc':       dict(cat=6, parm=3,   tlev='lowCloudLayer'),
    'mcdc':       dict(cat=6, parm=4,   tlev='middleCloudLayer'),
    'hcdc':       dict(cat=6, parm=5,   tlev='highCloudLayer'),
    'vis':        dict(cat=19,parm=0,   tlev='surface'),
    # PPFFG: NCEP-local "Probability of Precipitation > Flash Flood Guidance".
    # Resolve via shortName (the cat=1 parm=194 we used before was the CICEP
    # code — completely wrong, would never find any record).
    'ppffg':      dict(shortname='ppffg', tlev='surface'),

    # Member-only variables (live in 2dfld files, accessed via member byte-range).
    # For NCEP-local fields whose cat/parm guesses don't match eccodes' table
    # entries, filter by cfgrib `shortName` instead — eccodes maps wgrib2's
    # uppercase names ("VIL", "DCAPE", "MCONV") to lower-case shortNames.
    'vil':        dict(shortname='vil',   tlev='atmosphereSingleLayer'),
    'gust':       dict(cat=2,  parm=22,   tlev='surface'),
    'dcape':      dict(shortname='dcape', tlev='pressureFromGroundLayer'),
    'mconv':      dict(shortname='mconv', tlev='atmosphereSingleLayer'),
    'ustm_6km':   dict(cat=2,  parm=27,   tlev='heightAboveGroundLayer'),
    'vstm_6km':   dict(cat=2,  parm=28,   tlev='heightAboveGroundLayer'),

    # Member-only severe-wx fields published in 2dfld files.
    'hail':       dict(shortname='hail', tlev='surface'),
    # Simulated satellite brightness temperature (Kelvin). Member-only.
    # NCEP-local; rely on the tlev='nominalTop' fallback in _cfgrib_load
    # to pick the single record from the byte-range partial.
    'brtemp':     dict(shortname='brtemp',   tlev='nominalTop'),
    'sbta1613':   dict(shortname='SBTA1613', tlev='nominalTop'),
    # Wildfire potential (0-1). 1-hour averaged forecast — see acc_type='ave'.
    'wfirepot':   dict(shortname='wfirepot', tlev='surface'),
}

# =============================================================================
#  Product registry
#  ---------------------------------------------------------------------------
#  Each product is a recipe specifying which file types and GRIB messages
#  to extract.  Common fields:
#     cat        : GUI category
#     name       : display name
#     ftype      : 'mean' | 'pmmn' | 'lpmm' | 'prob' | 'avrg'
#     short      : GRIB shortName (cfgrib lowercase)
#     level_type : cfgrib typeOfLevel
#     level      : numeric level (when applicable)
#     step       : stepRange string (e.g. '5-6' for 1-h APCP at FHR 6)
#     cmap       : colormap key
#     units      : display units
#     convert    : optional fn applied to data array
#     overlay    : optional list of overlay dicts {ftype, short, ...,
#                  style:'contour'|'barbs', levels:[..], colors:..}
#     spc_title  : caption row 2
#     valid_step : 'inst' | 'accum_1h' | 'accum_3h' | 'accum_6h' | 'accum_24h'
# =============================================================================

def _step_for_acc(fhr, n):
    """Return stepRange string for an n-hour accumulation ending at fhr."""
    return f"{max(0,fhr-n)}-{fhr}"

def _ov_prob(var, thresh, color='#3a1f00'):
    return dict(ftype='prob', var=var, thresh=thresh, style='contour',
                levels=[10,30,50,70,90], smooth=2.0, colors=color,
                linewidths=[0.8,1.2,1.6,2.0,2.4])

PRODUCTS = {
    # ---- Reflectivity / convection -----------------------------------------
    'refc_pmmn_series': dict(
        cat='Reflectivity (PMM Series)', name='Comp. Refl. PMM + prob >40 dBZ',
        ftype='pmmn', var='refc', cmap='refc', units='dBZ',
        overlay=[_ov_prob('refc', 40)],
        spc_title='Composite reflectivity (dBZ; shaded, PMM), neighborhood prob >40 dBZ (contours)'),

    'refd_pmmn': dict(
        cat='Reflectivity (PMM Series)', name='1km AGL Refl. PMM',
        ftype='pmmn', var='refd_1km', cmap='refc', units='dBZ',
        overlay=[_ov_prob('refd_1km', 40)],
        spc_title='1 km AGL reflectivity (dBZ; shaded, PMM), prob >40 dBZ (contours)'),

    'maxref_pmmn': dict(
        cat='Reflectivity (PMM Series)', name='Hourly Max 1km Refl. PMM',
        ftype='pmmn', var='maxref_1km', cmap='refc', units='dBZ',
        overlay=[_ov_prob('maxref_1km', 40)],
        spc_title='Hourly maximum 1 km reflectivity (dBZ; shaded, PMM), prob >40 dBZ (contours)'),

    # (retop_pmmn removed 2026-06-11 — aviation-flavored; echo-top signal
    #  lives on as the contour overlay in refc_echotop_combo.)

    # ---- Updraft helicity --------------------------------------------------
    'uh25_pmmn': dict(
        cat='Updraft Helicity (2-5 km)', name='UH 2-5km PMM + prob >75',
        ftype='pmmn', var='mxuphl_25', cmap='uh', units='m^2/s^2',
        overlay=[_ov_prob('mxuphl_25', 75)],
        spc_title='2-5 km UH (m^2/s^2; shaded, PMM), neighborhood prob >75 (contours)'),

    'uh25_prob25':  dict(cat='Updraft Helicity (2-5 km)', name='Prob UH 2-5km > 25',
        ftype='prob', var='mxuphl_25', thresh=25, cmap='prob', units='%',
        spc_title='Neighborhood probability of 2-5 km UH > 25 m^2/s^2'),
    'uh25_prob75':  dict(cat='Updraft Helicity (2-5 km)', name='Prob UH 2-5km > 75',
        ftype='prob', var='mxuphl_25', thresh=75, cmap='prob', units='%',
        spc_title='Neighborhood probability of 2-5 km UH > 75 m^2/s^2'),
    'uh25_prob150': dict(cat='Updraft Helicity (2-5 km)', name='Prob UH 2-5km > 150',
        ftype='prob', var='mxuphl_25', thresh=150, cmap='prob', units='%',
        spc_title='Neighborhood probability of 2-5 km UH > 150 m^2/s^2'),

    # ---- SPC HREF calibrated guidance (NOMADS spc_post) -------------------
    # Native-rendered SPC experimental HREF calibrated probabilities. These
    # use a separate data source (source='spc_post'); render.py routes them to
    # an SPCPostDataProcessor-backed job. Forecast hours / cycles differ from
    # REFS: 4-h GEFS-calibrated severe runs F016–F048 on the 00/12Z cycles.
    'spc_cal_tor_4h': dict(
        cat='Calibrated Severe (4-h)', name='4-h Tornado (HREF/GEFS)',
        source='spc_post', recipe='spc_prob',
        ftype='severe|href_cal_gefs_tor_{run:02d}.4hr', var='torprob',
        cmap='spc_tor', units='%', min_fhr=16, fhr_stride=1,
        spc_title='4-hr neighborhood probability of a tornado (r=40 km), HREF/GEFS-calibrated'),
    'spc_cal_hail_4h': dict(
        cat='Calibrated Severe (4-h)', name='4-h Hail (HREF/GEFS)',
        source='spc_post', recipe='spc_prob',
        ftype='severe|href_cal_gefs_hail_{run:02d}.4hr', var='hailprob',
        cmap='spc_hailwind', units='%', min_fhr=16, fhr_stride=1,
        spc_title='4-hr neighborhood probability of severe hail (r=40 km), HREF/GEFS-calibrated'),
    'spc_cal_wind_4h': dict(
        cat='Calibrated Severe (4-h)', name='4-h Wind (HREF/GEFS)',
        source='spc_post', recipe='spc_prob',
        ftype='severe|href_cal_gefs_wind_{run:02d}.4hr', var='windprob',
        cmap='spc_hailwind', units='%', min_fhr=16, fhr_stride=1,
        spc_title='4-hr neighborhood probability of severe wind (r=40 km), HREF/GEFS-calibrated'),

    'spc_cal_tor_24h': dict(
        cat='Calibrated Severe (24-h)', name='24-h Tornado (HREF/GEFS)',
        source='spc_post', recipe='spc_prob',
        ftype='severe|href_cal_gefs_tor_{run:02d}.24hr', var='torprob',
        cmap='spc_tor', units='%', min_fhr=36, fhr_stride=1,
        spc_title='24-hr neighborhood probability of a tornado (r=40 km), HREF/GEFS-calibrated'),
    'spc_cal_hail_24h': dict(
        cat='Calibrated Severe (24-h)', name='24-h Hail (HREF/GEFS)',
        source='spc_post', recipe='spc_prob',
        ftype='severe|href_cal_gefs_hail_{run:02d}.24hr', var='hailprob',
        cmap='spc_hailwind', units='%', min_fhr=36, fhr_stride=1,
        spc_title='24-hr neighborhood probability of severe hail (r=40 km), HREF/GEFS-calibrated'),
    'spc_cal_wind_24h': dict(
        cat='Calibrated Severe (24-h)', name='24-h Wind (HREF/GEFS)',
        source='spc_post', recipe='spc_prob',
        ftype='severe|href_cal_gefs_wind_{run:02d}.24hr', var='windprob',
        cmap='spc_hailwind', units='%', min_fhr=36, fhr_stride=1,
        spc_title='24-hr neighborhood probability of severe wind (r=40 km), HREF/GEFS-calibrated'),

    'spc_thunder_1h': dict(
        cat='Calibrated Thunder', name='1-h Thunder',
        source='spc_post', recipe='spc_prob',
        ftype='thunder|hrefct_1hr', var='tstm',
        cmap='spc_thunder', units='%', min_fhr=1, fhr_stride=1,
        spc_title='1-hr HREF calibrated thunderstorm probability'),
    'spc_thunder_4h': dict(
        cat='Calibrated Thunder', name='4-h Thunder',
        source='spc_post', recipe='spc_prob',
        ftype='thunder|hrefct_4hr', var='tstm',
        cmap='spc_thunder', units='%', min_fhr=4, fhr_stride=1,
        spc_title='4-hr HREF calibrated thunderstorm probability'),
    'spc_thunder_full': dict(
        cat='Calibrated Thunder', name='Period-total Thunder',
        source='spc_post', recipe='spc_prob',
        ftype='thunder|hrefct_full', var='tstm',
        cmap='spc_thunder', units='%', min_fhr=24, fhr_stride=1,
        spc_title='Period-total HREF calibrated thunderstorm probability'),

    'spc_ltg_4h_25': dict(
        cat='Lightning Density (4-h)', name='4-h P(≥25 flashes)',
        source='spc_post', recipe='spc_prob',
        ftype='ltgdensity|hrefld_4hr', var='tstm', thresh=25,
        cmap='spc_ltg', units='%', min_fhr=4, fhr_stride=1,
        spc_title='4-hr probability of ≥25 lightning flashes per grid box'),
    'spc_ltg_4h_100': dict(
        cat='Lightning Density (4-h)', name='4-h P(≥100 flashes)',
        source='spc_post', recipe='spc_prob',
        ftype='ltgdensity|hrefld_4hr', var='tstm', thresh=100,
        cmap='spc_ltg', units='%', min_fhr=4, fhr_stride=1,
        spc_title='4-hr probability of ≥100 lightning flashes per grid box'),

    # ---- QPF ---------------------------------------------------------------
    'qpf_1h_pmmn_series': dict(
        cat='QPF (PMM)', name='1-h QPF PMM + prob >0.50"',
        ftype='pmmn', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,1),
        cmap='qpf', units='in', convert=lambda x: x/25.4,
        overlay=[dict(ftype='prob', var='tp_sfc',
                      step_from_fhr=lambda f: _step_for_acc(f,1),
                      thresh=12.7, style='contour',
                      levels=[10,30,50,70,90], smooth=2.0,
                      colors='#3a1f00', linewidths=[0.8,1.2,1.6,2.0,2.4])],
        spc_title='1-hr QPF (in; PMM), neighborhood prob >0.50 in (contours)'),

    'qpf_3h_pmmn_series': dict(
        cat='QPF (PMM)', name='3-h QPF PMM + prob >1" + >3"',
        ftype='pmmn', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,3),
        cmap='qpf', units='in', convert=lambda x: x/25.4,
        # Mirrors the SPC HREF "3-hr QPF (PMM) + NP[QPF>1"] + NP[QPF>3"]" plot:
        #   brown contours = neighborhood prob of >1" (25.4 mm)
        #   red contours   = neighborhood prob of >3" (76.2 mm)
        # Both thresholds are published in the enspost/NOMADS `prob` file at
        # the 3-h accumulation window (verified: 21-24 h acc fcst carries
        # prob >25.4 and prob >76.2). They land in the same partial GRIB and
        # _cfgrib_load's threshold disambiguation selects each one.
        overlay=[dict(ftype='prob', var='tp_sfc',
                      step_from_fhr=lambda f: _step_for_acc(f,3),
                      thresh=25.4, style='contour',
                      levels=[10,30,50,70,90], smooth=2.0,
                      colors='#000000', linewidths=[0.8,1.2,1.6,2.0,2.4]),
                 dict(ftype='prob', var='tp_sfc',
                      step_from_fhr=lambda f: _step_for_acc(f,3),
                      thresh=76.2, style='contour',
                      levels=[10,30,50,70,90], smooth=2.0,
                      colors='#c81e1e', linewidths=[1.1,1.5,1.9,2.3,2.7])],
        spc_title=('3-hr QPF (in; PMM), neighborhood prob >1.00 in (brown) '
                   'and >3.00 in (red) contours')),

    'qpf_6h_lpmm_series': dict(
        cat='QPF (LPMM)', name='6-h QPF LPMM',
        ftype='lpmm', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,6),
        cmap='qpf', units='in', convert=lambda x: x/25.4,
        spc_title='6-hr QPF (in; LPMM)'),

    # ---- QPF LPMM native accumulation windows (1/3-h) ----------------------
    # NOTE: REFS/HREF publish deterministic QPF only at 1-h and 3-h in the
    # pmmn/lpmm files (REFS lpmm additionally carries a native 6-h window —
    # see qpf_6h_lpmm_series above). 6/12/24-h PMM and 12/24-h LPMM were
    # removed 2026-07-06: no backing records exist, so they rendered "No data"
    # (and, before the cfgrib stepRange fix, silently fell back to the 3-h
    # record). Longer accumulations remain available as neighborhood
    # probabilities (prob file DOES publish 6/12/24-h windows) and as the
    # 6-h / 24-h ensemble MEAN (additive, so summing is valid).
    'qpf_1h_lpmm_series': dict(
        cat='QPF (LPMM)', name='1-h QPF LPMM',
        ftype='lpmm', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,1),
        cmap='qpf', units='in', convert=lambda x: x/25.4,
        spc_title='1-hr QPF (in; LPMM)'),

    'qpf_3h_lpmm_series': dict(
        cat='QPF (LPMM)', name='3-h QPF LPMM',
        ftype='lpmm', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,3),
        cmap='qpf', units='in', convert=lambda x: x/25.4,
        spc_title='3-hr QPF (in; LPMM)'),

    'qpf_1h_prob_50':  dict(cat='QPF (Prob)', name='Prob 1-h QPF > 0.50"',
        ftype='prob', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,1),
        thresh=12.7, cmap='prob', units='%',
        spc_title='Neighborhood probability of 1-hr QPF > 0.50 in'),
    'qpf_1h_prob_100': dict(cat='QPF (Prob)', name='Prob 1-h QPF > 1.00"',
        ftype='prob', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,1),
        thresh=25.4, cmap='prob', units='%',
        spc_title='Neighborhood probability of 1-hr QPF > 1.00 in'),
    'qpf_1h_prob_200': dict(cat='QPF (Prob)', name='Prob 1-h QPF > 2.00"',
        ftype='prob', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,1),
        thresh=50.8, cmap='prob', units='%',
        spc_title='Neighborhood probability of 1-hr QPF > 2.00 in'),
    'qpf_1h_prob_300': dict(cat='QPF (Prob)', name='Prob 1-h QPF > 3.00"',
        ftype='prob', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,1),
        thresh=76.2, cmap='prob', units='%',
        spc_title='Neighborhood probability of 1-hr QPF > 3.00 in'),

    # SPC HREF "1-h QPF, ens P(>0.01")" — probability of measurable precip.
    # The enspost/NOMADS prob file's smallest published APCP threshold is
    # 12.7 mm (0.5"), so 0.01" (0.254 mm) is NOT available as a ready-made
    # neighborhood-probability record. Compute it from the 5 member files'
    # 1-h APCP instead (member_prob recipe), with a neighborhood radius so it
    # reads like SPC's smoothed field rather than a blocky 5-member fraction.
    'qpf_1h_prob_001': dict(cat='QPF (Prob)', name='1-h QPF ens P(>0.01")',
        recipe='member_prob', member_product='2dfld', n_members=5,
        var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,1),
        thresh=0.254, nbrhd_km=40.0, cmap='prob', units='%',
        spc_title='Ensemble probability of 1-hr QPF > 0.01 in (measurable; r=40 km)'),


    # ---- 3-h QPF probabilities (neighborhood) ------------------------------
    'qpf_3h_prob_100': dict(cat='QPF (Prob)', name='Prob 3-h QPF > 1.00"',
        ftype='prob', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,3),
        thresh=25.4, cmap='prob', units='%',
        spc_title='Neighborhood probability of 3-hr QPF > 1.00 in'),
    'qpf_3h_prob_200': dict(cat='QPF (Prob)', name='Prob 3-h QPF > 2.00"',
        ftype='prob', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,3),
        thresh=50.8, cmap='prob', units='%',
        spc_title='Neighborhood probability of 3-hr QPF > 2.00 in'),
    'qpf_3h_prob_300': dict(cat='QPF (Prob)', name='Prob 3-h QPF > 3.00"',
        ftype='prob', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,3),
        thresh=76.2, cmap='prob', units='%',
        spc_title='Neighborhood probability of 3-hr QPF > 3.00 in'),
    'qpf_3h_prob_500': dict(cat='QPF (Prob)', name='Prob 3-h QPF > 5.00"',
        ftype='prob', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,3),
        thresh=127, cmap='prob', units='%',
        spc_title='Neighborhood probability of 3-hr QPF > 5.00 in'),

    # ---- 6-h QPF probabilities (strict: no data before F06) ----------------
    'qpf_6h_prob_200': dict(cat='QPF (Prob)', name='Prob 6-h QPF > 2.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-6}-{f}" if f >= 6 else None,
        min_fhr=6,
        thresh=50.8, cmap='prob', units='%',
        spc_title='Neighborhood probability of 6-hr QPF > 2.00 in'),
    'qpf_6h_prob_300': dict(cat='QPF (Prob)', name='Prob 6-h QPF > 3.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-6}-{f}" if f >= 6 else None,
        min_fhr=6,
        thresh=76.2, cmap='prob', units='%',
        spc_title='Neighborhood probability of 6-hr QPF > 3.00 in'),
    'qpf_6h_prob_500': dict(cat='QPF (Prob)', name='Prob 6-h QPF > 5.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-6}-{f}" if f >= 6 else None,
        min_fhr=6,
        thresh=127, cmap='prob', units='%',
        spc_title='Neighborhood probability of 6-hr QPF > 5.00 in'),

    # ---- 12-h QPF probabilities (valid at F12, F24, F36) -------------------
    'qpf_12h_prob_300': dict(cat='QPF (Prob)', name='Prob 12-h QPF > 3.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-12}-{f}" if f >= 12 else None,
        min_fhr=12,
        thresh=76.2, cmap='prob', units='%',
        spc_title='Neighborhood probability of 12-hr QPF > 3.00 in'),
    'qpf_12h_prob_500': dict(cat='QPF (Prob)', name='Prob 12-h QPF > 5.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-12}-{f}" if f >= 12 else None,
        min_fhr=12,
        thresh=127, cmap='prob', units='%',
        spc_title='Neighborhood probability of 12-hr QPF > 5.00 in'),
    'qpf_12h_prob_800': dict(cat='QPF (Prob)', name='Prob 12-h QPF > 8.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-12}-{f}" if f >= 12 else None,
        min_fhr=12,
        thresh=203.2, cmap='prob', units='%',
        spc_title='Neighborhood probability of 12-hr QPF > 8.00 in'),

    # ---- 24-h QPF probabilities (valid at F24 in the F01-F36 window) -------
    # REFS encodes the 24h running total as "0-1 day acc fcst" in the .idx;
    # idx_match.fhr_marker handles this when step == fhr == 24.
    'qpf_24h_prob_200': dict(cat='QPF (Prob)', name='Prob 24-h QPF > 2.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-24}-{f}" if f >= 24 else None,
        min_fhr=24,
        thresh=50.8, cmap='prob', units='%',
        spc_title='Neighborhood probability of 24-hr QPF > 2.00 in'),
    'qpf_24h_prob_300': dict(cat='QPF (Prob)', name='Prob 24-h QPF > 3.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-24}-{f}" if f >= 24 else None,
        min_fhr=24,
        thresh=76.2, cmap='prob', units='%',
        spc_title='Neighborhood probability of 24-hr QPF > 3.00 in'),
    'qpf_24h_prob_500': dict(cat='QPF (Prob)', name='Prob 24-h QPF > 5.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-24}-{f}" if f >= 24 else None,
        min_fhr=24,
        thresh=127, cmap='prob', units='%',
        spc_title='Neighborhood probability of 24-hr QPF > 5.00 in'),
    'qpf_24h_prob_800': dict(cat='QPF (Prob)', name='Prob 24-h QPF > 8.00"',
        ftype='prob', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-24}-{f}" if f >= 24 else None,
        min_fhr=24,
        thresh=203.2, cmap='prob', units='%',
        spc_title='Neighborhood probability of 24-hr QPF > 8.00 in'),

    # ---- 6-h QPF ensemble mean + prob contour (from mean file) -------------
    'qpf_6h_mean_series': dict(
        cat='QPF (Mean)', name='6-h QPF mean + prob >1.0"',
        ftype='mean', var='tp_sfc',
        step_from_fhr=lambda f: f"{f-6}-{f}" if f >= 6 else None,
        min_fhr=6,
        cmap='qpf', units='in', convert=lambda x: x/25.4,
        overlay=[dict(ftype='prob', var='tp_sfc',
                      step_from_fhr=lambda f: f"{f-6}-{f}" if f >= 6 else None,
        min_fhr=6,
                      thresh=25.4, style='contour',
                      levels=[10,30,50,70,90], smooth=2.0,
                      colors='#3a1f00', linewidths=[0.8,1.2,1.6,2.0,2.4])],
        spc_title='6-hr QPF (in; ens mean shaded), neighborhood prob >1.00 in (contours)'),

    # ---- Thermodynamics ----------------------------------------------------
    'sbcape_mean_series': dict(
        cat='Thermodynamics', name='SBCAPE mean + prob ML CAPE >1000',
        ftype='mean', var='cape_sfc', cmap='cape', units='J/kg',
        overlay=[dict(ftype='prob', var='cape_ml', thresh=1000, style='contour',
                      levels=[10,30,50,70,90], smooth=2.0,
                      colors='#222222', linewidths=[0.8,1.2,1.6,2.0,2.4])],
        spc_title='Surface CAPE (J/kg; shaded, ens mean), prob ML CAPE >1000 (contours)'),
    'mlcape_mean':  dict(cat='Thermodynamics', name='ML (90-0mb) CAPE mean',
        ftype='mean', var='cape_ml', cmap='cape', units='J/kg',
        spc_title='Mixed-layer (90-0 mb) CAPE (J/kg), ensemble mean'),
    'mucape_mean':  dict(cat='Thermodynamics', name='MU (180-0mb) CAPE mean',
        ftype='mean', var='cape_mu', cmap='cape', units='J/kg',
        spc_title='Most unstable (180-0 mb) CAPE (J/kg), ensemble mean'),
    'sbcin_mean':   dict(cat='Thermodynamics', name='SBCIN mean',
        ftype='mean', var='cin_sfc', cmap='cin', units='J/kg',
        spc_title='Surface CIN (J/kg), ensemble mean'),
    'mlcin_mean':   dict(cat='Thermodynamics', name='ML (90-0mb) CIN mean',
        ftype='mean', var='cin_ml', cmap='cin', units='J/kg',
        spc_title='Mixed-layer (90-0 mb) CIN (J/kg), ensemble mean'),
    'pwat_mean':    dict(cat='Thermodynamics', name='PWAT mean (in)',
        ftype='mean', var='pwat', cmap='pwat_in', units='in',
        convert=lambda x: x/25.4,
        spc_title='Precipitable water (in), ensemble mean'),
    'srh_03km_mean': dict(cat='Thermodynamics', name='0-3km SRH mean',
        ftype='mean', var='srh_3km', cmap='srh', units='m^2/s^2',
        spc_title='0-3 km storm-relative helicity (m^2/s^2), ensemble mean'),

    # ---- Kinematics --------------------------------------------------------
    'wind_250_mean': dict(cat='Kinematics', name='250mb wind + heights + divergence',
        recipe='wind_level', level=250, cmap='wind250',
        spc_title='250 mb wind speed (kt; shaded), heights (dam), divergence contours (×10⁻⁵ s⁻¹), ensemble mean'),
    'wind_500_mean': dict(cat='Kinematics', name='500mb wind + heights',
        recipe='wind_level', level=500, cmap='wind500',
        spc_title='500 mb wind speed (kt; shaded) and heights (dam), ensemble mean'),
    'vort_500_mean': dict(cat='Kinematics', name='500mb abs vorticity + heights',
        recipe='vort_level', level=500, cmap='vort',
        spc_title='500 mb absolute vorticity (×10⁻⁵ s⁻¹; shaded), heights (dam), wind barbs, ensemble mean'),
    'wind_700_mean': dict(cat='Kinematics', name='700mb wind + RH + heights + omega',
        recipe='wind_700_rh', level=700, cmap='wind500',
        spc_title='700 mb wind (kt), heights (dam), RH >70% (green shading), vertical motion contours'),
    'wind_850_mean': dict(cat='Kinematics', name='850mb wind + RH + temp + heights',
        recipe='wind_850_temp', level=850, cmap='wind500',
        spc_title='850 mb wind (kt), heights (dam), temperature (°C, contours), RH >80% (green shading)'),
    'wind_925_mean': dict(cat='Kinematics', name='925mb wind + RH + temp + heights',
        recipe='wind_850_temp', level=925, cmap='wind500',
        spc_title='925 mb wind (kt), heights (dam), temperature (°C, contours), RH >80% (green shading)'),
    'wind_10m_mean': dict(cat='Kinematics', name='10m wind mean',
        recipe='wind_10m', cmap='wind500',
        spc_title='10 m wind (kt), ensemble mean'),
    'shear_06km_mean': dict(cat='Kinematics', name='0-6 km bulk shear mean (kt)',
        ftype='mean', var='vwsh_06km', cmap='wind500', units='kt',
        convert=lambda x: x*1.94384,
        spc_title='0-6 km bulk wind shear (kt), ensemble mean -- supercell discriminator'),

    # ---- Surface ----------------------------------------------------------
    # (mslp_mean [MSLP + 850-500 thickness] removed 2026-06-11 per user —
    #  MSLP lives on in the combo_sfc products.)
    'ptype_mslp_mean': dict(cat='Surface', name='Precip type + MSLP',
        recipe='ptype_mslp', cmap=None,
        spc_title='Ensemble-mean precipitation type (rain/snow/FZRA/IP) and MSLP (hPa)'),
    't2m_mean':  dict(cat='Surface', name='2-m temperature mean (degF)',
        ftype='mean', var='t_2m', cmap='t2m', units='degF',
        convert=lambda x: (x-273.15)*9/5+32,
        spc_title='2-m temperature (degF), ensemble mean'),
    'td2m_mean': dict(cat='Surface', name='2-m dewpoint mean (degF)',
        ftype='mean', var='d_2m', cmap='td2m', units='degF',
        convert=lambda x: (x-273.15)*9/5+32,
        spc_title='2-m dewpoint (degF), ensemble mean'),
    'heat_index': dict(cat='Surface',
        name='Heat index (apparent T, degF)',
        recipe='heat_index', cmap='t2m', units='degF',
        spc_title='Heat index (degF) -- NWS Rothfusz formula from T2m and Td2m'),
    # snow_24h_pmmn removed: PMMN files don't carry ASNOW at all.
    # snow_24h_mean removed: mean ASNOW exists as "0-1 day acc fcst" but the
    # idx matcher emits "0-24 hour" notation; needs a dedicated step encoder
    # before this can come back reliably.

    # ---- Severe probabilities ---------------------------------------------
    'refc_prob_40': dict(cat='Severe Probabilities', name='Prob Comp. Refl. > 40 dBZ',
        ftype='prob', var='refc', thresh=40, cmap='prob', units='%',
        spc_title='Neighborhood probability of composite reflectivity > 40 dBZ'),
    'refc_prob_50': dict(cat='Severe Probabilities', name='Prob Comp. Refl. > 50 dBZ',
        ftype='prob', var='refc', thresh=50, cmap='prob', units='%',
        spc_title='Neighborhood probability of composite reflectivity > 50 dBZ'),
    'ltng_prob': dict(cat='Severe Probabilities', name='Prob Lightning > 0.08',
        ftype='prob', var='ltng', thresh=0.08, cmap='prob', units='%',
        spc_title='Neighborhood probability of lightning > 0.08 fl/km^2/min'),

    # 0-6 km bulk shear -- supercell threshold tiers
    'shear_06km_prob_30kt': dict(cat='Severe Probabilities',
        name='Prob 0-6 km Shear > 30 kt',
        ftype='prob', var='vwsh_06km', thresh=15.4,
        cmap='prob', units='%',
        spc_title='Neighborhood probability of 0-6 km bulk shear > 30 kt'),
    'shear_06km_prob_40kt': dict(cat='Severe Probabilities',
        name='Prob 0-6 km Shear > 40 kt (supercell)',
        ftype='prob', var='vwsh_06km', thresh=20.6,
        cmap='prob', units='%',
        spc_title='Neighborhood probability of 0-6 km bulk shear > 40 kt -- supercell threshold'),
    # (shear_06km_prob_50kt removed 2026-06-11 per user — 30/40 kt tiers kept.)

    # Max updraft velocity (100-1000 mb) -- convective vigor
    'maxuvv_prob_10': dict(cat='Severe Probabilities',
        name='Prob Max Updraft > 10 m/s',
        ftype='prob', var='maxuvv_lyr', thresh=10,
        cmap='prob', units='%',
        spc_title='Neighborhood probability of max upward vertical velocity (100-1000 mb) > 10 m/s'),
    'maxuvv_prob_20': dict(cat='Severe Probabilities',
        name='Prob Max Updraft > 20 m/s (vigorous)',
        ftype='prob', var='maxuvv_lyr', thresh=20,
        cmap='prob', units='%',
        spc_title='Neighborhood probability of max upward vertical velocity > 20 m/s -- vigorous convection'),

    # (wind10_prob_40kt [hourly], mlcin_prob_50, t2m_prob_freezing removed
    #  2026-06-11 per user. retop_prob_* and vis_prob_* removed earlier same
    #  day — aviation-flavored. The 4-hr wind NPs carry the wind signal.)

    # ---- CAPE / CIN threshold probabilities (SPC Severe -> threshold probs)
    'mlcape_prob_500':  dict(cat='Severe Probabilities', name='Prob ML CAPE > 500',
        ftype='prob', var='cape_ml', thresh=500, cmap='prob', units='%',
        spc_title='Neighborhood probability of 90-0 mb CAPE > 500 J/kg'),
    'mlcape_prob_1000': dict(cat='Severe Probabilities', name='Prob ML CAPE > 1000',
        ftype='prob', var='cape_ml', thresh=1000, cmap='prob', units='%',
        spc_title='Neighborhood probability of 90-0 mb CAPE > 1000 J/kg'),
    'mlcape_prob_1500': dict(cat='Severe Probabilities', name='Prob ML CAPE > 1500',
        ftype='prob', var='cape_ml', thresh=1500, cmap='prob', units='%',
        spc_title='Neighborhood probability of 90-0 mb CAPE > 1500 J/kg'),
    'mlcape_prob_2000': dict(cat='Severe Probabilities', name='Prob ML CAPE > 2000',
        ftype='prob', var='cape_ml', thresh=2000, cmap='prob', units='%',
        spc_title='Neighborhood probability of 90-0 mb CAPE > 2000 J/kg'),
    'mlcape_prob_3000': dict(cat='Severe Probabilities', name='Prob ML CAPE > 3000',
        ftype='prob', var='cape_ml', thresh=3000, cmap='prob', units='%',
        spc_title='Neighborhood probability of 90-0 mb CAPE > 3000 J/kg'),
    'mlcin_prob_100':   dict(cat='Severe Probabilities', name='Prob ML CIN < -100',
        ftype='prob', var='cin_ml',  thresh=-100, cmap='prob', units='%',
        prob_below=True,
        spc_title='Neighborhood probability of 90-0 mb CIN < -100 J/kg'),

    # ---- SRH probabilities (Severe -> Shear/threshold probs) ---------------
    'srh_03km_prob_100': dict(cat='Severe Probabilities', name='Prob 0-3km SRH > 100',
        ftype='prob', var='srh_3km', thresh=100, cmap='prob', units='%',
        spc_title='Neighborhood probability of 0-3 km SRH > 100 m^2/s^2'),
    'srh_03km_prob_200': dict(cat='Severe Probabilities', name='Prob 0-3km SRH > 200',
        ftype='prob', var='srh_3km', thresh=200, cmap='prob', units='%',
        spc_title='Neighborhood probability of 0-3 km SRH > 200 m^2/s^2'),
    'srh_03km_prob_400': dict(cat='Severe Probabilities', name='Prob 0-3km SRH > 400',
        ftype='prob', var='srh_3km', thresh=400, cmap='prob', units='%',
        spc_title='Neighborhood probability of 0-3 km SRH > 400 m^2/s^2'),

    # ---- Synoptic -> Surface composites (T or Td + MSLP + 10 m barbs) ------
    't2m_combo': dict(
        cat='Synoptic / Surface',
        name='2-m Temperature + MSLP + 10m wind',
        recipe='combo_sfc', var='t_2m', cmap='t2m', units='degF',
        convert=lambda x: (x-273.15)*9/5+32,
        spc_title='2-m temperature (degF), MSLP (hPa), 10-m wind (kt), ensemble mean'),
    'td2m_combo': dict(
        cat='Synoptic / Surface',
        name='2-m Dewpoint + MSLP + 10m wind',
        recipe='combo_sfc', var='d_2m', cmap='td2m', units='degF',
        convert=lambda x: (x-273.15)*9/5+32,
        spc_title='2-m dewpoint (degF), MSLP (hPa), 10-m wind (kt), ensemble mean'),

    # ---- Synoptic -> Clouds & Moisture -------------------------------------
    # (cloud_lmh tri-layer combo removed 2026-06-11 — synoptic fluff for a
    #  severe-wx audience; the single-layer cloud means below remain.)
    'tcdc_mean': dict(cat='Synoptic / Moisture', name='Total Cloud Cover mean',
        ftype='mean', var='tcdc', cmap='clouds', units='%',
        spc_title='Total cloud cover (%), ensemble mean'),
    'lcdc_mean': dict(cat='Synoptic / Moisture', name='Low Cloud Cover mean',
        ftype='mean', var='lcdc', cmap='clouds', units='%',
        spc_title='Low cloud cover (%), ensemble mean'),
    'mcdc_mean': dict(cat='Synoptic / Moisture', name='Mid Cloud Cover mean',
        ftype='mean', var='mcdc', cmap='clouds', units='%',
        spc_title='Mid cloud cover (%), ensemble mean'),
    'hcdc_mean': dict(cat='Synoptic / Moisture', name='High Cloud Cover mean',
        ftype='mean', var='hcdc', cmap='clouds', units='%',
        spc_title='High cloud cover (%), ensemble mean'),

    'pwat_prob_15in': dict(cat='Synoptic / Moisture', name='Prob PWAT > 1.5"',
        ftype='prob', var='pwat', thresh=37.5, cmap='prob', units='%',
        spc_title='Neighborhood probability of PWAT > 37.5 mm (~1.5 in)'),
    'pwat_prob_2in':  dict(cat='Synoptic / Moisture', name='Prob PWAT > 2.0"',
        ftype='prob', var='pwat', thresh=50,   cmap='prob', units='%',
        spc_title='Neighborhood probability of PWAT > 50 mm (~2.0 in)'),

    'td_prob_60F': dict(cat='Synoptic / Moisture', name='Prob 2-m Dewpoint > 60F',
        ftype='prob', var='d_2m', thresh=288.71, cmap='prob', units='%',
        spc_title='Neighborhood probability of 2-m dewpoint > 60 degF'),
    'td_prob_65F': dict(cat='Synoptic / Moisture', name='Prob 2-m Dewpoint > 65F',
        ftype='prob', var='d_2m', thresh=291.48, cmap='prob', units='%',
        spc_title='Neighborhood probability of 2-m dewpoint > 65 degF'),
    'td_prob_70F': dict(cat='Synoptic / Moisture', name='Prob 2-m Dewpoint > 70F',
        ftype='prob', var='d_2m', thresh=294.26, cmap='prob', units='%',
        spc_title='Neighborhood probability of 2-m dewpoint > 70 degF'),


    # ---- Flash Flood Recurrence Index (FFRI / PPFFG) -----------------------
    'ppffg_1h': dict(cat='Flash Flood Threat', name='P(precip > FFG), 1-h',
        ftype='ffri', var='ppffg', step_from_fhr=lambda f: _step_for_acc(f,1),
        thresh=1, cmap='prob', units='%',
        spc_title='Neighborhood probability 1-hr precip > Flash Flood Guidance'),
    'ppffg_3h': dict(cat='Flash Flood Threat', name='P(precip > FFG), 3-h',
        ftype='ffri', var='ppffg', step_from_fhr=lambda f: _step_for_acc(f,3),
        thresh=3, cmap='prob', units='%',
        spc_title='Neighborhood probability 3-hr precip > Flash Flood Guidance'),
    'ppffg_6h': dict(cat='Flash Flood Threat', name='P(precip > FFG), 6-h',
        ftype='ffri', var='ppffg', step_from_fhr=lambda f: _step_for_acc(f,6),
        thresh=6, cmap='prob', units='%',
        spc_title='Neighborhood probability 6-hr precip > Flash Flood Guidance'),
    'ffri_qpf_6h_2in': dict(cat='Flash Flood Threat', name='FFRI 6-h QPF p>2"',
        ftype='ffri', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,6),
        thresh=50, cmap='prob', units='%',
        spc_title='6-hr QPF probability > 2 in (FFRI)'),
    'ffri_qpf_6h_4in': dict(cat='Flash Flood Threat', name='FFRI 6-h QPF p>4"',
        ftype='ffri', var='tp_sfc', step_from_fhr=lambda f: _step_for_acc(f,6),
        thresh=100, cmap='prob', units='%',
        spc_title='6-hr QPF probability > 4 in (FFRI)'),

    # (Entire 'QPF (EAS scale-aware)' category — 8 products — removed
    #  2026-06-11: niche; the standard neighborhood QPF probabilities
    #  cover the same operational decisions.)

    # ---- Individual-member products (rrfs_a/rrfsens/) ----------------------
    'paintball_refc_40': dict(cat='Member Plots (RRFS_A)',
        name='Paintball: Comp. Refl. >40 dBZ',
        recipe='paintball', member_product='2dfld',
        var='refc', thresh=40,
        spc_title='Composite reflectivity > 40 dBZ, ensemble paintball'),
    'paintball_refc_50': dict(cat='Member Plots (RRFS_A)',
        name='Paintball: Comp. Refl. >50 dBZ',
        recipe='paintball', member_product='2dfld',
        var='refc', thresh=50,
        spc_title='Composite reflectivity > 50 dBZ, ensemble paintball'),
    # window_h=4: SPC HREF severe paintballs (UH/hail/gust/wind) are 4-hr max
    # products; REFS member records are 1-h, so take the max of the 4 hourly
    # records ending at the frame hour to match.
    'paintball_uh_75': dict(cat='Member Plots (RRFS_A)',
        name='Paintball: 4-hr max UH 2-5km >75 m^2/s^2',
        recipe='paintball', member_product='2dfld',
        var='mxuphl_25', thresh=75, window_h=4,
        spc_title='4-hr max 2-5 km UH > 75 m^2/s^2, ensemble paintball'),
    'paintball_uh_150': dict(cat='Member Plots (RRFS_A)',
        name='Paintball: 4-hr max UH 2-5km >150 m^2/s^2',
        recipe='paintball', member_product='2dfld',
        var='mxuphl_25', thresh=150, window_h=4,
        spc_title='4-hr max 2-5 km UH > 150 m^2/s^2, ensemble paintball'),

    # Stamps -- multi-panel grid, one per member
    'stamps_refc': dict(cat='Member Plots (RRFS_A)',
        name='Stamps: Composite Reflectivity',
        recipe='stamps', member_product='2dfld',
        var='refc', cmap='refc', units='dBZ',
        spc_title='Composite reflectivity (dBZ)'),
    'stamps_uh25': dict(cat='Member Plots (RRFS_A)',
        name='Stamps: UH 2-5 km',
        recipe='stamps', member_product='2dfld',
        var='mxuphl_25', cmap='uh', units='m^2/s^2',
        spc_title='2-5 km updraft helicity (m^2/s^2)'),

    # ---- New 2dfld stamps: tornado layer / hail / gust -------------------
    'stamps_uh03': dict(cat='Member Plots (RRFS_A)',
        name='Stamps: UH 0-3 km (tornado layer)',
        recipe='stamps', member_product='2dfld',
        var='mxuphl_03', cmap='uh', units='m^2/s^2',
        spc_title='0-3 km updraft helicity (m^2/s^2) -- low-level rotation / tornado layer'),
    'stamps_hail': dict(cat='Member Plots (RRFS_A)',
        name='Stamps: Hail Size (in)',
        recipe='stamps', member_product='2dfld',
        var='hail', cmap='hail', units='in',
        convert=lambda x: x * 39.37,    # m -> in
        spc_title='Forecast max hail size per member (in)'),
    'stamps_gust': dict(cat='Member Plots (RRFS_A)',
        name='Stamps: Surface Wind Gust (kt)',
        recipe='stamps', member_product='2dfld',
        var='gust', cmap='gust', units='kt',
        convert=lambda x: x * 1.94384,
        # Ambient 10-20 kt gusts shade the whole domain; floor at 20 kt so
        # the panels show the convective signal.
        mask_below=20,
        spc_title='Surface wind gust per member (kt, shaded ≥20)'),

    # ---- New paintball products on 2dfld ---------------------------------
    'paintball_uh03_75': dict(cat='Member Plots (RRFS_A)',
        name='Paintball: 4-hr max UH 0-3km > 75 (tornado layer)',
        recipe='paintball', member_product='2dfld',
        var='mxuphl_03', thresh=75, window_h=4,
        spc_title='4-hr max 0-3 km UH > 75 m^2/s^2 -- low-level rotation paintball'),
    'paintball_hail_1in': dict(cat='Member Plots (RRFS_A)',
        name='Paintball: 4-hr max Hail > 1 inch (severe)',
        recipe='paintball', member_product='2dfld',
        var='hail', thresh=0.0254, window_h=4,        # m (1 in)
        spc_title='4-hr max forecast hail size > 1 in -- severe-hail paintball'),
    'paintball_gust_50kt': dict(cat='Member Plots (RRFS_A)',
        name='Paintball: 4-hr max Gust > 50 kt',
        recipe='paintball', member_product='2dfld',
        var='gust', thresh=25.72, window_h=4,         # m/s (50 kt)
        spc_title='4-hr max surface wind gust > 50 kt -- damaging-wind paintball'),

    # ---- Member-derived gust probabilities (true ensemble prob, not the
    #      enspost 10-m wind-speed proxy). _member_prob recipe counts the
    #      fraction of members exceeding the threshold at each grid point. -
    'gust_prob_30mph': dict(cat='Severe Probabilities',
        name='Member Prob Gust > 30 mph',
        recipe='member_prob', member_product='2dfld', n_members=5,
        var='gust', thresh=13.41,         # m/s (30 mph)
        cmap='prob', units='%',
        spc_title='Member-derived probability of surface gust > 30 mph'),
    'gust_prob_40mph': dict(cat='Severe Probabilities',
        name='Member Prob Gust > 40 mph',
        recipe='member_prob', member_product='2dfld', n_members=5,
        var='gust', thresh=17.88,         # m/s (40 mph)
        cmap='prob', units='%',
        spc_title='Member-derived probability of surface gust > 40 mph'),
    'gust_prob_50mph': dict(cat='Severe Probabilities',
        name='Member Prob Gust > 50 mph',
        recipe='member_prob', member_product='2dfld', n_members=5,
        var='gust', thresh=22.35,         # m/s (50 mph)
        cmap='prob', units='%',
        spc_title='Member-derived probability of surface gust > 50 mph'),
    'gust_prob_60mph': dict(cat='Severe Probabilities',
        name='Member Prob Gust > 60 mph (severe)',
        recipe='member_prob', member_product='2dfld', n_members=5,
        var='gust', thresh=26.82,         # m/s (60 mph) -- NWS severe gust
        cmap='prob', units='%',
        spc_title='Member-derived probability of surface gust > 60 mph -- NWS severe-gust tier'),
    'gust_prob_75mph': dict(cat='Severe Probabilities',
        name='Member Prob Gust > 75 mph (sig severe)',
        recipe='member_prob', member_product='2dfld', n_members=5,
        var='gust', thresh=33.53,         # m/s (75 mph)
        cmap='prob', units='%',
        spc_title='Member-derived probability of surface gust > 75 mph -- significant severe'),
}

REGIONS = {
    'CONUS':     dict(lon=(-122.0,-72.5), lat=(22.5, 49.5), proj_lon=-97.5,
                      name='CONUS'),
    'Northwest': dict(lon=(-126.0,-108.0), lat=(38.0, 50.0), proj_lon=-117.0,
                      name='Northwest'),
    'Southwest': dict(lon=(-125.0,-104.0), lat=(28.0, 42.0), proj_lon=-113.0,
                      name='Southwest'),
    'N. Plains': dict(lon=(-110.0, -90.0), lat=(38.0, 50.0), proj_lon=-100.0,
                      name='Northern Plains'),
    'S. Plains': dict(lon=(-108.0, -89.0), lat=(26.0, 40.0), proj_lon=-99.0,
                      name='Southern Plains'),
    'Midwest':   dict(lon=(-100.0, -78.0), lat=(36.5, 49.5), proj_lon=-89.0,
                      name='Midwest'),
    'Southeast': dict(lon=( -95.0, -73.0), lat=(24.0, 38.0), proj_lon=-85.0,
                      name='Southeast'),
    'Northeast': dict(lon=( -83.0, -65.0), lat=(37.0, 48.0), proj_lon=-74.0,
                      name='Northeast'),
    'Mid-Atl':   dict(lon=( -88.0, -72.0), lat=(33.0, 43.0), proj_lon=-80.0,
                      name='Mid-Atlantic'),
}

# =============================================================================
#  Data processor
# =============================================================================
class REFSDataProcessor:
    """Download + extract REFS GRIB2 variables.

    Lookup priority:
      1. <local>/refs.tHHz.<ftype>.fXX.conus.grib2     (flat -- your test set)
      2. <local>/YYYYMMDD/HH/refs.tHHz.<ftype>.fXX.conus.grib2
      3. S3 download into (2)
    """

    def __init__(self, local_dir=DEFAULT_LOCAL):
        self.local_dir = Path(local_dir)
        self._dataset_cache = {}  # path -> {keytuple: xarray}

    # ----- Member file acquisition ------------------------------------------
    def find_or_fetch_member(self, date_str, run, fhr, mem, product='2dfld',
                             status_cb=None):
        """Download an individual ensemble member file from rrfs_a/rrfsens/."""
        fname = S3_MEM_FNAME_T.format(run=run, mem=mem, product=product, fhr=fhr)
        for p in [self.local_dir / fname,
                  self.local_dir / date_str / f"{run:02d}" / f"m{mem:03d}" / fname]:
            if p.exists():
                return p
        target = self.local_dir / date_str / f"{run:02d}" / f"m{mem:03d}" / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{S3_BASE_URL}/{S3_MEM_PREFIX_T.format(date=date_str, run=run, mem=mem)}{fname}"
        if status_cb: status_cb(f"Downloading m{mem:03d} {product} F{fhr:03d}...")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(target, 'wb') as f:
                    for chunk in r.iter_content(1<<17):
                        f.write(chunk)
        except Exception as e:
            if target.exists(): target.unlink()
            raise RuntimeError(f"Could not fetch member {mem} {product}: {e}")
        return target

    def ensure_member_record(self, path, date_str, run, fhr, mem, match,
                             product='2dfld'):
        """Append `match`-ing GRIB records to a byte-range *partial* member
        file that is missing them.

        Partial member files (written by the API layer's byte-range prefetch)
        only contain the records some earlier product needed. When a later
        product hits the same file looking for a different variable, load_var
        returns None even though the record exists upstream. This sync helper
        lets the render worker pull just the missing record (~1-2 MB) instead
        of failing — or worse, re-downloading the 300+ MB full member file.

        Returns True if new bytes were appended (caller should retry the load).
        """
        path = Path(path)
        sidecar = Path(str(path) + ".records.json")
        if match is None or not sidecar.exists():
            return False   # full file (or no sidecar contract) — nothing to add
        try:
            have = set(json.loads(sidecar.read_text()))
        except (OSError, json.JSONDecodeError):
            have = set()
        if match in have:
            return False
        fname = S3_MEM_FNAME_T.format(run=run, mem=mem, product=product, fhr=fhr)
        url = (f"{S3_BASE_URL}/"
               f"{S3_MEM_PREFIX_T.format(date=date_str, run=run, mem=mem)}{fname}")
        try:
            r = requests.get(url + ".idx", timeout=20)
            r.raise_for_status()
            rows = []
            for ln in r.text.splitlines():
                parts = ln.split(":", 2)
                if len(parts) < 3:
                    continue
                try:
                    rows.append((int(parts[1]), ln))
                except ValueError:
                    continue
            wanted = []
            for i, (off, raw) in enumerate(rows):
                if match in raw:
                    end = rows[i + 1][0] - 1 if i + 1 < len(rows) else None
                    wanted.append((off, end))
            if not wanted:
                return False
            with open(path, "ab") as f:
                for off, end in wanted:
                    hdr = f"bytes={off}-" if end is None else f"bytes={off}-{end}"
                    rr = requests.get(url, headers={"Range": hdr}, timeout=45)
                    if rr.status_code not in (200, 206):
                        rr.raise_for_status()
                    f.write(rr.content)
            have.add(match)
            sidecar.write_text(json.dumps(sorted(have)))
            return True
        except Exception as e:
            print(f"[member-heal] m{mem:03d} F{fhr:03d} {match!r}: "
                  f"{type(e).__name__}: {e}", flush=True)
            return False

    # ----- File acquisition -------------------------------------------------
    def find_or_fetch(self, date_str, run, fhr, ftype, status_cb=None):
        fname = S3_FNAME_T.format(run=run, ftype=ftype, fhr=fhr)
        for p in [self.local_dir / fname,
                  self.local_dir / date_str / f"{run:02d}" / fname]:
            if p.exists():
                return p
        target = self.local_dir / date_str / f"{run:02d}" / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{S3_BASE_URL}/{S3_PREFIX_T.format(date=date_str, run=run)}{fname}"
        if status_cb: status_cb(f"Downloading {fname}...")
        try:
            with requests.get(url, stream=True, timeout=45) as r:
                r.raise_for_status()
                total = int(r.headers.get('Content-Length', 0))
                got = 0
                with open(target, 'wb') as f:
                    for chunk in r.iter_content(1<<17):
                        f.write(chunk); got += len(chunk)
                        if status_cb and total:
                            pct = 100.0*got/total
                            status_cb(f"Downloading {fname} ({pct:.0f}%)")
        except Exception as e:
            if target.exists(): target.unlink()
            raise RuntimeError(f"Could not fetch {fname}: {e}")
        return target

    # ----- Variable extraction (pygrib primary, cfgrib fallback) ------------
    def load_var(self, filepath, varkey, level=None, step=None, thresh=None,
                 below=False):
        """Load a 2D field.

        varkey -- a key into VARSPEC (or a fresh dict with cat/parm/tlev[/lev]).
        level  -- if varkey's spec lacks 'lev', supply it here (e.g. for wind_lvl).
        step   -- GRIB stepRange (e.g. '5-6').
        thresh -- numeric probability threshold (real-world units); we match
                  against scaledValueOfUpperLimit / 10**scaleFactorOfUpperLimit.
        Returns (data2d, lats, lons) or (None, None, None)."""

        if isinstance(varkey, dict):
            spec = dict(varkey)
        else:
            spec = dict(VARSPEC.get(varkey, {}))
        if level is not None:
            spec['lev'] = level

        # Decoded-field disk cache. GRIB decode is ~half the cold-render
        # cost, and the SAME field is re-decoded for every region / theme /
        # palette variant of a product (and again in the other render worker
        # — the in-memory _DS_CACHE is per-process). Persist the final 2-D
        # array once and serve repeats with a ~50 ms np.load.
        fcp = self._field_cache_path(filepath, spec, step, thresh, below)
        if fcp is not None and fcp.exists():
            try:
                z = np.load(fcp, allow_pickle=False)
                g = np.load(fcp.parent / str(z['grid']), allow_pickle=False)
                return (z['data'].astype(np.float64),
                        g['lats'].astype(np.float64),
                        g['lons'].astype(np.float64))
            except Exception:
                # Torn/expired grid file etc. — fall through to a fresh
                # decode, which re-saves both.
                try: fcp.unlink()
                except OSError: pass

        # L2 remote field cache (survives container restarts).
        # Checked only on local miss; adds ~300-500 ms vs 3-5 s for GRIB decode.
        if fcp is not None:
            try:
                from app.field_persist import stable_key, try_load_remote
                sk = stable_key(filepath, spec, step, thresh, below)
                if sk and try_load_remote(sk, fcp.parent):
                    # Remote copy landed in fcp.parent under the stable key name;
                    # re-check the local path (local key may differ on this host).
                    remote_fcp = fcp.parent / f"{sk}.npz"
                    if remote_fcp.exists():
                        try:
                            z = np.load(remote_fcp, allow_pickle=False)
                            g = np.load(fcp.parent / str(z['grid']), allow_pickle=False)
                            return (z['data'].astype(np.float64),
                                    g['lats'].astype(np.float64),
                                    g['lons'].astype(np.float64))
                        except Exception:
                            try: remote_fcp.unlink()
                            except OSError: pass
            except ImportError:
                pass
            except Exception:
                pass

        if HAS_PYGRIB:
            r = self._pygrib_load(filepath, spec, step, thresh, below)
            if r[0] is not None:
                self._field_cache_save(fcp, *r)
                self._field_persist_push(filepath, spec, step, thresh, below, fcp)
                return r
        r = self._cfgrib_load(filepath, spec, step, thresh, below=below)
        if r[0] is not None:
            self._field_cache_save(fcp, *r)
            self._field_persist_push(filepath, spec, step, thresh, below, fcp)
        return r

    @staticmethod
    def _field_persist_push(filepath, spec, step, thresh, below, fcp):
        """Fire-and-forget push of a freshly decoded field to remote cache."""
        try:
            from app.field_persist import stable_key, push_field_async, enabled
            if not enabled() or fcp is None:
                return
            sk = stable_key(filepath, spec, step, thresh, below)
            if not sk:
                return
            import threading, asyncio as _aio
            # Derive grid path from the saved field file.
            import numpy as _np
            z = _np.load(fcp, allow_pickle=False)
            gname = str(z['grid'])
            gpath = fcp.parent / gname
            gk = gname.removeprefix('grid_').removesuffix('.npz')
            def _push():
                try:
                    _aio.run(push_field_async(sk, fcp, grid_key=gk, grid_path=gpath))
                except Exception:
                    pass
            threading.Thread(target=_push, daemon=True).start()
        except Exception:
            pass

    _FIELD_CACHE_VERSION = 1

    def _field_cache_path(self, filepath, spec, step, thresh, below):
        """Stable .npz path for a decoded field, or None if uncacheable.
        Keyed by file mtime so partial-GRIB appends invalidate naturally."""
        try:
            mtime = int(os.path.getmtime(str(filepath)))
        except OSError:
            return None
        import hashlib
        key = repr((self._FIELD_CACHE_VERSION, str(filepath), mtime,
                    sorted((k, str(v)) for k, v in spec.items()),
                    step, thresh, below))
        h = hashlib.sha1(key.encode()).hexdigest()[:24]
        return self.local_dir / '_fields' / f'{h}.npz'

    @staticmethod
    def _field_cache_save(fcp, data, lats, lons):
        """Atomic writes so a concurrent reader in the other render worker
        never sees a torn file. The lat/lon grid is identical for every field
        on the same model grid, so it's stored ONCE per grid and referenced
        by name — fields are then ~7.6 MB of float32 data each."""
        if fcp is None or data is None:
            return
        try:
            import hashlib
            fcp.parent.mkdir(parents=True, exist_ok=True)
            gk = hashlib.sha1(repr((lats.shape, float(np.ravel(lats)[0]),
                                    float(np.ravel(lats)[-1]),
                                    float(np.ravel(lons)[0]),
                                    float(np.ravel(lons)[-1]))).encode()
                              ).hexdigest()[:16]
            gname = f'grid_{gk}.npz'
            gpath = fcp.parent / gname
            if not gpath.exists():
                gtmp = gpath.with_suffix(f'.{os.getpid()}.tmp')
                with open(gtmp, 'wb') as f:
                    np.savez(f, lats=lats.astype(np.float32),
                             lons=lons.astype(np.float32))
                os.replace(gtmp, gpath)
            tmp = fcp.with_suffix(f'.{os.getpid()}.tmp')
            with open(tmp, 'wb') as f:
                np.savez(f, data=data.astype(np.float32),
                         grid=np.array(gname))
            os.replace(tmp, fcp)
        except Exception as e:
            print(f"[field-cache] save failed: {type(e).__name__}: {e}",
                  flush=True)
            try: tmp.unlink()
            except Exception: pass

    # ----- pygrib path -----------------------------------------------------
    def _pygrib_load(self, filepath, spec, step, thresh, below=False):
        try:
            g = pygrib.open(str(filepath))
        except Exception as e:
            print(f"[pygrib] open failed: {e}")
            return None, None, None
        match = None
        try:
            for m in g:
                if spec.get('cat') is not None and m.parameterCategory != spec['cat']:
                    continue
                if spec.get('parm') is not None and m.parameterNumber != spec['parm']:
                    continue
                if spec.get('tlev') and m.typeOfLevel != spec['tlev']:
                    continue
                if spec.get('lev') is not None and m.level != spec['lev']:
                    continue
                if spec.get('top') is not None and getattr(m,'topLevel',None) != spec['top']:
                    continue
                if spec.get('bot') is not None and getattr(m,'bottomLevel',None) != spec['bot']:
                    continue
                if step is not None and m.stepRange != step:
                    continue
                if thresh is not None:
                    try:
                        if below:
                            sv = m.scaledValueOfLowerLimit
                            sf = m.scaleFactorOfLowerLimit
                        else:
                            sv = m.scaledValueOfUpperLimit
                            sf = m.scaleFactorOfUpperLimit
                        v = sv / (10**sf) if sf else float(sv)
                        if abs(v - thresh) > max(1e-3, abs(thresh)*0.01):
                            continue
                    except Exception:
                        continue
                match = m
                break
            if match is None:
                return None, None, None
            data = np.array(match.values, dtype=float)
            lats, lons = match.latlons()
            lons = np.where(lons > 180, lons - 360, lons)
            data = np.nan_to_num(data, nan=0.0)
            return data, lats, lons
        finally:
            g.close()

    # ----- cfgrib path -----------------------------------------------------
    # Module-level cache of opened xarray Datasets keyed by (filepath, fk).
    # Letting cfgrib write its on-disk .4.idx (we no longer pass indexpath='')
    # also avoids re-scanning the GRIB on every open.
    _DS_CACHE: dict = {}
    _DS_CACHE_MAX = 32   # ~few MB each; cap to bound worker memory
    # Sentinel cached for (filepath, mtime, fk, thresh, below) combinations that
    # resolve to NO matching record. Without this, products carrying an overlay
    # whose record doesn't exist in the file (e.g. the 3-h QPF >3" prob contour,
    # which REFS/HREF don't publish at the 3-h accumulation) re-ran the full
    # idx-fetch + up-to-4 cfgrib open_dataset scans on EVERY forecast frame.
    # Keyed by mtime, so appending records to a partial GRIB invalidates it.
    _NEG_DS = object()

    def _cfgrib_load(self, filepath, spec, step, thresh, below=False):
        fk = {}
        # Prefer cfgrib `shortName` when supplied — robust against NCEP-local
        # parm-code mismatches where we'd otherwise have to guess.
        if spec.get('shortname'): fk['shortName'] = spec['shortname']
        if spec.get('cat') is not None: fk['parameterCategory'] = spec['cat']
        if spec.get('parm') is not None: fk['parameterNumber'] = spec['parm']
        if spec.get('tlev'): fk['typeOfLevel'] = spec['tlev']
        if spec.get('lev') is not None: fk['level'] = spec['lev']
        if step is not None: fk['stepRange'] = step
        # Include mtime so appending to a partial GRIB invalidates stale opens.
        try:
            mtime = os.path.getmtime(str(filepath))
        except OSError:
            return None, None, None
        # Include thresh + below in the cache key — different threshold
        # requests against the same partial file must open distinct
        # filtered datasets, otherwise the second-threshold call would
        # hit the first-threshold cache entry and silently return data
        # for the wrong threshold. This was the root cause of dual-
        # overlay products like the heavy-rain composite drawing
        # identical contours.
        cache_key = (str(filepath), mtime, tuple(sorted(fk.items())),
                     thresh, below)
        ds = self._DS_CACHE.get(cache_key)
        if ds is self._NEG_DS:
            # Previously resolved to "no such record" for this exact file
            # revision — skip the idx-fetch + repeated cfgrib scans entirely.
            return None, None, None
        if ds is None:
            def _open(_fk):
                try:
                    # read_keys exposes the probability-threshold metadata
                    # as DataArray attrs so the safety-net verify below can
                    # actually check it. Without these, cfgrib only exposes
                    # a default set of keys and our threshold check
                    # silently falls through to "can't verify".
                    return xr.open_dataset(
                        str(filepath), engine='cfgrib',
                        backend_kwargs={
                            'filter_by_keys': _fk,
                            'read_keys': [
                                'scaledValueOfUpperLimit',
                                'scaleFactorOfUpperLimit',
                                'scaledValueOfLowerLimit',
                                'scaleFactorOfLowerLimit',
                            ],
                        },
                    )
                except Exception:
                    return None
            ds = None
            # Threshold-tightening pass: when `thresh` is supplied, ask
            # cfgrib to filter on `scaledValueOfUpperLimit` (or LowerLimit
            # for prob-below) directly. Without this, cfgrib silently
            # picks the FIRST matching record when several records differ
            # only in threshold — which is exactly how the dual-overlay
            # heavy-rain composite collapsed to identical contours. REFS
            # uses two scale-factor conventions:
            #   sf=3 for prob/eas files (thresh in mm × 1000)
            #   sf=0 for ffri files (thresh as a small integer)
            # Try both; the one whose encoded value hits a real record
            # opens with non-empty data_vars and wins.
            if thresh is not None:
                if below:
                    sv_key = 'scaledValueOfLowerLimit'
                    sf_key = 'scaleFactorOfLowerLimit'
                else:
                    sv_key = 'scaledValueOfUpperLimit'
                    sf_key = 'scaleFactorOfUpperLimit'
                for try_sf in (3, 0):
                    fk_t = dict(fk)
                    fk_t[sv_key] = int(round(thresh * (10 ** try_sf)))
                    fk_t[sf_key] = try_sf
                    cand = _open(fk_t)
                    if cand is not None and cand.data_vars:
                        ds = cand
                        break
            # Fall back to the un-tightened filter if the threshold pass
            # didn't pin a record, or if no thresh was requested at all.
            if ds is None or not ds.data_vars:
                ds = _open(fk)
            # Fallback 2: cfgrib's NCEP-local shortName / parm-code lookup
            # missed the record we want (common for NCEP-only fields like
            # VIL / DCAPE). Retry with just the typeOfLevel — safe because
            # byte-range partials contain only the records we requested.
            if ds is None or not ds.data_vars:
                loose_fk = {}
                if spec.get('tlev'):
                    loose_fk['typeOfLevel'] = spec['tlev']
                # Preserve the stepRange filter when an accumulation window was
                # explicitly requested. Dropping it here was safe for byte-range
                # REFS partials (they contain ONLY the records we fetched), but
                # HREF downloads WHOLE files — so a loose open with no step would
                # silently grab an arbitrary APCP window. That is exactly why the
                # HREF 6-hr and 12-hr QPF tiles rendered identical (both fell back
                # to the same available 1-/3-hr accumulation). Instantaneous
                # fields (VIL/DCAPE, the case this fallback exists for) pass
                # step=None, so their NCEP-local recovery is unaffected.
                if step is not None:
                    loose_fk['stepRange'] = step
                ds_loose = _open(loose_fk)
                if ds_loose is not None and ds_loose.data_vars:
                    ds = ds_loose
            # Evict oldest entry if over cap (simple FIFO; good enough).
            # Shared by the positive and negative-cache stores below.
            def _evict_if_full():
                if len(self._DS_CACHE) >= self._DS_CACHE_MAX:
                    old_key = next(iter(self._DS_CACHE))
                    old_ds = self._DS_CACHE.pop(old_key)
                    try: old_ds.close()        # sentinel has no .close() — guarded
                    except Exception: pass
            if ds is None or not ds.data_vars:
                # Cache the negative outcome so repeat frames / scrubs of an
                # absent overlay don't re-pay the idx-fetch + cfgrib scans.
                _evict_if_full()
                self._DS_CACHE[cache_key] = self._NEG_DS
                return None, None, None
            _evict_if_full()
            self._DS_CACHE[cache_key] = ds
        if not ds.data_vars: return None, None, None
        da = ds[list(ds.data_vars)[0]]
        # Threshold disambiguation
        # ------------------------
        # When a partial GRIB carries multiple "probability of exceedance"
        # records that differ ONLY in the threshold value (e.g. FFRI's
        # APCP:0-6 hour acc fcst:prob >50 and prob >100), cfgrib stacks
        # them onto an auxiliary dimension. The dim is typically named
        # `scaledValueOfUpperLimit` (or ...LowerLimit for prob-below), not
        # 'threshold' — so the previous substring check missed it and the
        # later squeeze + arr[0] silently returned the first record for
        # every threshold. That made every multi-thresh prob product
        # collapse to identical data.
        #
        # New approach: iterate any non-spatial dim with size > 1, try the
        # raw values and several common scale-factor exponents (REFS uses
        # scaleFactor=0 for FFRI/EAS files and scaleFactor=3 for enspost
        # prob files), and pick the index whose decoded value matches
        # `thresh` within tolerance. If no candidate matches, return None
        # instead of silently returning the wrong record.
        if thresh is not None:
            extra_dims = [d for d in da.dims
                          if d not in ('latitude', 'longitude', 'y', 'x')
                          and da.sizes[d] > 1]
            if extra_dims:
                matched = False
                for d in extra_dims:
                    try:
                        vals = np.asarray(ds[d].values, dtype=float)
                    except (KeyError, TypeError, ValueError):
                        continue
                    tol = max(1e-3, abs(thresh) * 0.01)
                    chosen_idx = None
                    for sf in (0, 1, 2, 3):
                        scaled = vals / (10.0 ** sf)
                        idx = int(np.argmin(np.abs(scaled - thresh)))
                        if abs(float(scaled[idx]) - thresh) <= tol:
                            chosen_idx = idx
                            break
                    if chosen_idx is not None:
                        da = da.isel({d: chosen_idx})
                        matched = True
                        break
                if not matched:
                    print(f"[cfgrib] threshold {thresh} not found in dims "
                          f"{extra_dims} for {filepath}", flush=True)
                    return None, None, None
                # When disambiguation ran, the matched coord value was
                # already checked against `thresh`. Skip the safety-net
                # verify below (which only handles the no-aux-dim case).
            else:
                # ------------------------------------------------------
                #  Safety-net verify for the no-aux-dim case
                # ------------------------------------------------------
                #  When the partial GRIB has only a SINGLE prob record at
                #  this (var, step) — common, since byte-range partials
                #  are sparse — cfgrib opens it as a plain 2-D array with
                #  no auxiliary dim. The disambiguation block above
                #  doesn't run, and the old code returned that record's
                #  values whether or not its encoded threshold matched.
                #  This was the bug behind qpf_6h_prob_050 / 200 / 300
                #  rendering identical tiles.
                #
                #  Fix: check the DataArray's GRIB threshold metadata
                #  against the requested `thresh`. If we can't verify
                #  (cfgrib didn't expose the attr AND no aux dim was
                #  there to have just disambiguated), fail loudly
                #  instead of serving wrong data silently.
                if below:
                    sv = da.attrs.get('GRIB_scaledValueOfLowerLimit')
                    sf = da.attrs.get('GRIB_scaleFactorOfLowerLimit')
                else:
                    sv = da.attrs.get('GRIB_scaledValueOfUpperLimit')
                    sf = da.attrs.get('GRIB_scaleFactorOfUpperLimit')
                if sv is not None:
                    try:
                        v = float(sv) / (10.0 ** int(sf or 0))
                        tol = max(1e-3, abs(thresh) * 0.01)
                        if abs(v - thresh) > tol:
                            # POSITIVE mismatch — we CAN verify and it's
                            # wrong. This is the case we must refuse to
                            # serve, because returning it would render
                            # the wrong threshold under the right title.
                            print(f"[cfgrib] threshold MISMATCH (no aux "
                                  f"dim): requested {thresh}, loaded {v} "
                                  f"(sv={sv}, sf={sf}) for {filepath} "
                                  f"step={step}", flush=True)
                            return None, None, None
                    except (TypeError, ValueError):
                        # Couldn't decode the metadata; fall through to
                        # the lenient warning + return below.
                        sv = None
                if sv is None:
                    # No metadata exposed by cfgrib (even with read_keys
                    # asking for it). We can't positively confirm match
                    # OR mismatch — log it, then trust the byte-range
                    # fetch + filter_by_keys to have selected the right
                    # record and return the data anyway. Refusing here
                    # would break legitimate single-overlay products like
                    # qpf_3h_pmmn_series' prob-contour overlay, where the
                    # partial file contains a single record that IS the
                    # one we asked for.
                    print(f"[cfgrib] cannot verify threshold {thresh} "
                          f"(no GRIB attrs, no aux dim) for {filepath} "
                          f"step={step}; trusting byte-range selection",
                          flush=True)
        arr = np.array(da.values).squeeze()
        # After threshold sel, arr should be 2-D. If it isn't, that means
        # there are still un-disambiguated dims — return None loudly rather
        # than picking [0] and serving wrong data silently.
        if arr.ndim > 2:
            print(f"[cfgrib] unresolved extra dims {da.dims} for "
                  f"{filepath} thresh={thresh}", flush=True)
            return None, None, None
        if arr.ndim != 2: return None, None, None
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        lons = np.where(lons > 180, lons - 360, lons)
        return np.nan_to_num(arr, nan=0.0), lats, lons

# =============================================================================
#  Plot manager
# =============================================================================
CITIES_DB = [
    # (lon, lat, name, rank)  — rank 1 always visible; rank 2 on regional
    # sectors (extent width < ~30°); curated for operational weather use.
    (-74.006, 40.713, 'New York',       1),
    (-118.244, 34.052, 'Los Angeles',   1),
    (-87.629, 41.878, 'Chicago',        1),
    (-95.369, 29.760, 'Houston',        1),
    (-112.074, 33.448, 'Phoenix',       1),
    (-75.165, 39.953, 'Philadelphia',   1),
    (-98.493, 29.424, 'San Antonio',    1),
    (-96.797, 32.776, 'Dallas',         1),
    (-117.161, 32.715, 'San Diego',     1),
    (-97.743, 30.267, 'Austin',         1),
    (-81.655, 30.332, 'Jacksonville',   1),
    (-80.843, 35.227, 'Charlotte',      1),
    (-122.419, 37.775, 'San Francisco', 1),
    (-122.332, 47.606, 'Seattle',       1),
    (-104.991, 39.739, 'Denver',        1),
    (-77.037, 38.907, 'Washington',     1),
    (-71.058, 42.360, 'Boston',         1),
    (-86.781, 36.163, 'Nashville',      1),
    (-122.676, 45.523, 'Portland',      1),
    (-97.516, 35.467, 'Oklahoma City',  1),
    (-115.139, 36.169, 'Las Vegas',     1),
    (-90.049, 35.149, 'Memphis',        1),
    (-84.387, 33.749, 'Atlanta',        1),
    (-80.192, 25.762, 'Miami',          1),
    (-93.265, 44.978, 'Minneapolis',    1),
    (-90.071, 29.951, 'New Orleans',    1),
    (-111.891, 40.760, 'Salt Lake City',1),
    (-106.486, 31.762, 'El Paso',       1),
    (-83.046, 42.331, 'Detroit',        1),
    # Rank 2 — regional reference cities
    (-86.158, 39.768, 'Indianapolis',   2),
    (-82.998, 39.961, 'Columbus',       2),
    (-87.906, 43.038, 'Milwaukee',      2),
    (-94.578, 39.099, 'Kansas City',    2),
    (-90.199, 38.627, 'St. Louis',      2),
    (-84.512, 39.103, 'Cincinnati',     2),
    (-79.996, 40.441, 'Pittsburgh',     2),
    (-81.694, 41.499, 'Cleveland',      2),
    (-78.879, 42.886, 'Buffalo',        2),
    (-72.685, 41.764, 'Hartford',       2),
    (-77.436, 37.541, 'Richmond',       2),
    (-76.286, 36.851, 'Norfolk',        2),
    (-78.638, 35.779, 'Raleigh',        2),
    (-79.937, 32.776, 'Charleston',     2),
    (-82.458, 27.948, 'Tampa',          2),
    (-81.379, 28.538, 'Orlando',        2),
    (-86.802, 33.521, 'Birmingham',     2),
    (-88.040, 30.696, 'Mobile',         2),
    (-92.289, 34.746, 'Little Rock',    2),
    (-90.184, 32.298, 'Jackson',        2),
    (-93.751, 32.525, 'Shreveport',     2),
    (-95.993, 36.154, 'Tulsa',          2),
    (-97.336, 37.689, 'Wichita',        2),
    (-95.934, 41.257, 'Omaha',          2),
    (-93.625, 41.587, 'Des Moines',     2),
    (-96.711, 43.547, 'Sioux Falls',    2),
    (-100.778, 46.808, 'Bismarck',      2),
    (-108.501, 45.787, 'Billings',      2),
    (-104.821, 41.140, 'Cheyenne',      2),
    (-116.203, 43.615, 'Boise',         2),
    (-117.426, 47.658, 'Spokane',       2),
    (-110.926, 32.222, 'Tucson',        2),
    (-106.650, 35.085, 'Albuquerque',   2),
    (-101.832, 35.222, 'Amarillo',      2),
    (-101.855, 33.578, 'Lubbock',       2),
    (-119.772, 36.748, 'Fresno',        2),
    (-121.494, 38.582, 'Sacramento',    2),
    (-119.792, 39.529, 'Reno',          2),
    (-97.143, 49.892, 'Winnipeg',       2),
]


# US county boundaries. cartopy's `cfeature` has no USCOUNTIES (that's a MetPy
# feature, which isn't installed on the Space) — use the Natural Earth
# admin_2_counties layer, which cartopy auto-downloads and caches. Only the
# 10m scale carries US counties (20m 404s). Cache one feature instance so its
# ~3,200 geometries are read once and reused across renders.
_US_COUNTIES_FEATURE = None
def _county_feature():
    global _US_COUNTIES_FEATURE
    if _US_COUNTIES_FEATURE is None:
        _US_COUNTIES_FEATURE = cfeature.NaturalEarthFeature(
            'cultural', 'admin_2_counties', '10m')
    return _US_COUNTIES_FEATURE


class PlotManager:
    FIG_W, FIG_H = 12.5, 8.6
    DPI = 110
    theme = THEMES['light']
    # Per-render visual toggles — set by the request layer before each call.
    # Defaults OFF so the basemap stays minimal (AG-WX-style); the frontend
    # exposes toggles that flip these on per-tile.
    show_counties = False
    show_cities   = False
    show_regions  = False    # WxWorks HU region boundaries baked into tile
    model_label   = "REFS"   # overridden to "HREF v3" by render.py for HREF tiles

    # --- Universal layout spec (all single-panel plots share this) ----------
    # Header band (text)
    HDR_Y1     = 0.955         # top line (REFS / Init)
    HDR_Y2     = 0.925         # subtitle / Valid
    HDR_LEFT_X = 0.035
    HDR_RIGHT_X = 0.965
    # Map area
    MAP_BOX    = [0.035, 0.135, 0.93, 0.77]   # [left, bottom, width, height]
    # Colorbar (centered horizontally)
    CBAR_BOX   = [0.16, 0.065, 0.68, 0.022]
    CBAR_LABEL_X = 0.855

    @staticmethod
    def _projection(region_key):
        r = REGIONS[region_key]
        lat_c = 0.5*(r['lat'][0] + r['lat'][1])
        return ccrs.LambertConformal(central_longitude=r.get('proj_lon', -97.5),
                                     central_latitude=lat_c,
                                     standard_parallels=(lat_c, lat_c))

    # --- Basemap cache --------------------------------------------------
    # Cartopy reprojects every feature on each render; USCOUNTIES at 20m is
    # ~3000 polygons. We rasterize the static parts once per (region, theme,
    # figsize, dpi) and blit them with ax.imshow on subsequent renders.
    # Two layers: bg (land/ocean/lakes, z=0) and fg (counties/states/borders/
    # coastline, z=10) so data (z≈3) layers between them — preserving the
    # original visual stacking.
    _BASEMAP_CACHE: dict = {}
    _BASEMAP_CACHE_MAX = 32

    @staticmethod
    def _theme_id(theme):
        return tuple(sorted((k, str(v)) for k, v in theme.items()))

    @staticmethod
    def _rasterize_axes(fig, ax):
        """Render fig with Agg and return (rgba_crop, xo, yo).

        rgba_crop: HxWx4 array, top-origin (row 0 is top of axes).
        (xo, yo): figure-pixel position of the crop's LOWER-LEFT corner —
        suitable for ``fig.figimage(rgba_crop, xo=xo, yo=yo, origin='upper')``
        so the crop is replayed at the exact same place on a fresh figure.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w = int(round(fig.bbox.width))
        h = int(round(fig.bbox.height))
        bb = ax.get_window_extent()
        x0 = max(0, int(round(bb.x0)))
        x1 = min(w, int(round(bb.x1)))
        y_bot = max(0, int(round(bb.y0)))
        y_top = min(h, int(round(bb.y1)))
        rgba = np.asarray(canvas.buffer_rgba())   # (H, W, 4), origin top
        crop = np.array(rgba[h - y_top: h - y_bot, x0:x1, :])
        return crop, x0, y_bot

    def _build_basemap_layers(self, region_key):
        r = REGIONS[region_key]
        t = self.theme
        proj = self._projection(region_key)
        extent = [r['lon'][0], r['lon'][1], r['lat'][0], r['lat'][1]]

        # --- Background: face + LAND/OCEAN/LAKES ---
        bg_fig = Figure(figsize=(self.FIG_W, self.FIG_H), dpi=self.DPI)
        bg_fig.patch.set_facecolor((0, 0, 0, 0))   # transparent figure bg
        bg_ax = bg_fig.add_axes(self.MAP_BOX, projection=proj)
        bg_ax.set_facecolor(t['fig_axes'])
        bg_ax.set_extent(extent, crs=ccrs.PlateCarree())
        bg_ax.add_feature(cfeature.LAND.with_scale('50m'),
                          facecolor=t['map_land'], zorder=1)
        bg_ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                          facecolor=t['map_ocean'], zorder=1)
        bg_ax.add_feature(cfeature.LAKES.with_scale('50m'),
                          facecolor=t['map_ocean'],
                          edgecolor=t['map_state'], linewidth=0.4, zorder=2)
        for s in bg_ax.spines.values():
            s.set_visible(False)
        bg_rgba, bg_xo, bg_yo = self._rasterize_axes(bg_fig, bg_ax)
        try:
            import matplotlib.pyplot as plt
            plt.close(bg_fig)
        except Exception:
            pass

        # --- Foreground: states / borders / coastline ---
        # Counties live OUTSIDE the cached basemap — they're slow to draw
        # (~3000 polygons) and now toggle-able per-render via show_counties.
        fg_fig = Figure(figsize=(self.FIG_W, self.FIG_H), dpi=self.DPI)
        fg_fig.patch.set_facecolor((0, 0, 0, 0))
        fg_ax = fg_fig.add_axes(self.MAP_BOX, projection=proj)
        fg_ax.set_facecolor((0, 0, 0, 0))
        fg_ax.set_extent(extent, crs=ccrs.PlateCarree())
        fg_ax.add_feature(cfeature.STATES.with_scale('50m'),
                          edgecolor=t['map_state'], linewidth=0.7, zorder=9)
        fg_ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                          edgecolor=t['map_border'], linewidth=1.0, zorder=9)
        fg_ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                          edgecolor=t['map_border'], linewidth=0.8, zorder=9)
        for s in fg_ax.spines.values():
            s.set_visible(False)
        fg_rgba, fg_xo, fg_yo = self._rasterize_axes(fg_fig, fg_ax)
        try:
            import matplotlib.pyplot as plt
            plt.close(fg_fig)
        except Exception:
            pass

        return (bg_rgba, bg_xo, bg_yo), (fg_rgba, fg_xo, fg_yo)

    # -------------------------------------------------------------------
    def _setup(self, region_key):
        r = REGIONS[region_key]
        t = self.theme
        proj = self._projection(region_key)
        fig = Figure(figsize=(self.FIG_W, self.FIG_H), dpi=self.DPI)
        fig.patch.set_facecolor(t['fig_face'])
        ax = fig.add_axes(self.MAP_BOX, projection=proj)
        ax.set_facecolor(t['fig_axes'])
        ax.set_extent([r['lon'][0], r['lon'][1], r['lat'][0], r['lat'][1]],
                      crs=ccrs.PlateCarree())

        # Custom (per-request) regions skip the cache: bbox varies per call
        # and we don't want unbounded growth from arbitrary user rectangles.
        use_cache = (region_key != '__custom__')
        cache_key = (region_key, self._theme_id(t),
                     self.FIG_W, self.FIG_H, self.DPI)
        layers = self._BASEMAP_CACHE.get(cache_key) if use_cache else None

        if layers is None and use_cache:
            try:
                layers = self._build_basemap_layers(region_key)
                if len(self._BASEMAP_CACHE) >= self._BASEMAP_CACHE_MAX:
                    # Evict oldest (FIFO)
                    self._BASEMAP_CACHE.pop(next(iter(self._BASEMAP_CACHE)))
                self._BASEMAP_CACHE[cache_key] = layers
            except Exception as e:
                print(f"[basemap] cache build failed for {region_key}: "
                      f"{type(e).__name__}: {e}", flush=True)
                layers = None

        # When either toggle is on, the fg figimage (states/borders/coastline)
        # would sit on top of any axes-level features at fig-z=5, hiding
        # counties (ax-z=8) and city labels (ax-z=11+). Use BG-only blit and
        # draw the fg features directly into the axes so they layer correctly
        # under counties/cities. Slow path (no cached layers) draws all
        # features into the axes regardless.
        use_fg_blit = (layers is not None
                       and not self.show_counties
                       and not self.show_cities)

        if layers is not None:
            (bg_rgba, bg_xo, bg_yo), (fg_rgba, fg_xo, fg_yo) = layers
            ax.set_facecolor((0, 0, 0, 0))
            fig.figimage(bg_rgba, xo=bg_xo, yo=bg_yo, origin='upper', zorder=-1)
            if use_fg_blit:
                fig.figimage(fg_rgba, xo=fg_xo, yo=fg_yo, origin='upper', zorder=5)

        if layers is None or not use_fg_blit:
            # Need fg features at axes level so counties/cities can layer
            # above them. (When `layers is None` we also need the bg.)
            if layers is None:
                ax.add_feature(cfeature.LAND.with_scale('50m'),
                               facecolor=t['map_land'], zorder=1)
                ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                               facecolor=t['map_ocean'], zorder=1)
                ax.add_feature(cfeature.LAKES.with_scale('50m'),
                               facecolor=t['map_ocean'],
                               edgecolor=t['map_state'], linewidth=0.4, zorder=2)
            # Counties go UNDER states so state borders read on top.
            if self.show_counties:
                try:
                    ax.add_feature(_county_feature(), facecolor='none',
                                   edgecolor=t['map_county'], linewidth=0.45,
                                   zorder=8)
                except Exception as e:
                    print(f"[counties] draw failed: {type(e).__name__}: {e}",
                          flush=True)
            ax.add_feature(cfeature.STATES.with_scale('50m'),
                           edgecolor=t['map_state'], linewidth=0.7, zorder=9)
            ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                           edgecolor=t['map_border'], linewidth=1.0, zorder=9)
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                           edgecolor=t['map_border'], linewidth=0.8, zorder=9)

        if self.show_cities:
            self._draw_cities(ax, region_key)

        if self.show_regions:
            self._draw_regions(ax)

        ax._refs_default_extent = [r['lon'][0], r['lon'][1], r['lat'][0], r['lat'][1]]
        return fig, ax

    # -------------------------------------------------------------------
    def _draw_cities(self, ax, region_key):
        """Plot a curated set of CONUS reference cities, rank-filtered by the
        current extent so a wide CONUS view shows only top-tier hubs while
        regional sectors fill in more cities."""
        import matplotlib.patheffects as pe
        if region_key not in REGIONS:
            return
        r = REGIONS[region_key]
        lon_min, lon_max = r['lon']
        lat_min, lat_max = r['lat']
        width = max(0.1, lon_max - lon_min)
        # Wide CONUS-like views: rank-1 only. Regional/zoomed: rank 1+2.
        max_rank = 1 if width >= 30.0 else 2
        t = self.theme
        # Theme-aware: subtle dot + readable label with a contrast halo so the
        # text stays legible over reflectivity / cape shading.
        dot_color   = t['map_border']
        text_color  = t['fg']
        halo_color  = t['fig_axes']    # = panel face (white in light, dark in dark)
        for lon, lat, name, rank in CITIES_DB:
            if rank > max_rank:
                continue
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            ax.scatter([lon], [lat], s=14, marker='o',
                       facecolor=dot_color, edgecolor=halo_color,
                       linewidth=0.6, zorder=11,
                       transform=ccrs.PlateCarree())
            txt = ax.text(lon + 0.18, lat + 0.08, name,
                          fontsize=8.0, fontweight='semibold',
                          color=text_color, zorder=12,
                          transform=ccrs.PlateCarree())
            txt.set_path_effects([
                pe.withStroke(linewidth=2.4, foreground=halo_color, alpha=0.95)
            ])

    def _draw_regions(self, ax):
        """Draw WxWorks HU region boundaries baked into the tile at zorder=10.

        Boundaries are drawn ABOVE forecast fill (z=3–8) but BELOW city markers
        (z=11) and city text (z=12), so labels always remain readable.
        """
        try:
            from app.regions_overlay import _load
        except ImportError:
            return
        import cartopy.crs as ccrs
        regions = _load()
        if not regions:
            return
        for reg in regions:
            coords = reg["coords"]
            if len(coords) < 2:
                continue
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            color = (reg["r"] / 255.0, reg["g"] / 255.0, reg["b"] / 255.0)
            # Close the polygon if not already
            if (lats[0], lons[0]) != (lats[-1], lons[-1]):
                lats = lats + [lats[0]]
                lons = lons + [lons[0]]
            ax.plot(lons, lats, color=color, linewidth=1.4, alpha=0.85,
                    transform=ccrs.PlateCarree(), zorder=10,
                    solid_capstyle="round")

    def _ink(self, light, dark):
        """Pick a semantically-colored contour line color for the active
        theme. Dark mode (sentinel: ``cnt_halo`` is set) returns a brightened
        variant so the line itself contrasts the dark map base; combined with
        the black casing from ``_halo`` it reads on both bright fills and the
        dark base. Light mode returns the original color unchanged."""
        return dark if self.theme.get('cnt_halo') else light

    def _halo(self, cs, labels=None, lw=3.4, label_lw=3.0):
        """Apply an opaque black casing behind contour lines + clabels.

        Dark mode draws white/near-white contour lines (see cnt_* theme
        keys). A thick, fully opaque BLACK casing is what makes them read
        over any field: the black outline contrasts bright fills (yellow
        vorticity, green QPF, white precip) while the white core contrasts
        the dark map base — one of the two always wins.

        Light mode sets ``cnt_halo = None``: dark lines on the light figure
        need no casing, so this is a no-op and light mode keeps its original
        clean appearance.
        """
        halo = self.theme.get('cnt_halo')
        if not halo:
            return                      # light mode: no casing (original look)
        import matplotlib.patheffects as pe
        eff = [pe.withStroke(linewidth=lw, foreground=halo, alpha=1.0)]
        try:
            for col in cs.collections:
                col.set_path_effects(eff)
        except AttributeError:
            try:
                cs.set_path_effects(eff)
            except Exception:
                pass
        if labels:
            lbl_eff = [pe.withStroke(linewidth=label_lw,
                                     foreground=halo, alpha=1.0)]
            for lbl in labels:
                try:
                    lbl.set_path_effects(lbl_eff)
                except Exception:
                    pass

    def _header(self, fig, prod, run_dt, fhr):
        valid_dt = run_dt + timedelta(hours=fhr)
        c = self.theme['fg']
        muted = self.theme.get('muted', c)
        # Big bold brand mark — slightly larger than before for an AG-WX-
        # style first impression.
        fig.text(self.HDR_LEFT_X, self.HDR_Y1, self.__class__.model_label,
                 ha='left', va='center', fontsize=17, fontweight='bold',
                 color=c, family='sans-serif')
        # Subtitle: product description line. Slightly muted so the bold
        # title carries the eye.
        fig.text(self.HDR_LEFT_X, self.HDR_Y2, prod['spc_title'],
                 ha='left', va='center', fontsize=11, color=muted)
        fig.text(self.HDR_RIGHT_X, self.HDR_Y1,
                 f"Init: {run_dt.strftime('%a %Y-%m-%d %H:%M')} UTC",
                 ha='right', va='center', fontsize=10.5, family='monospace', color=c)
        fig.text(self.HDR_RIGHT_X, self.HDR_Y2,
                 f"Valid: {valid_dt.strftime('%a %Y-%m-%d %H:%M')} UTC  [F{fhr:03d}]",
                 ha='right', va='center', fontsize=10.5, family='monospace', color=c)

    def _colorbar(self, fig, mappable, levels, label, extend='max', fmt='%g'):
        c = self.theme['fg']
        cax = fig.add_axes(self.CBAR_BOX)
        cb = fig.colorbar(mappable, cax=cax, orientation='horizontal',
                          ticks=levels, extend=extend)
        cb.ax.set_xticklabels([fmt % v if isinstance(v,(int,float)) else str(v)
                               for v in levels], color=c)
        cb.ax.tick_params(labelsize=9, length=3, pad=2, colors=c)
        cb.outline.set_edgecolor(c)
        # label sits centered vertically with the colorbar bar
        ybar = self.CBAR_BOX[1] + self.CBAR_BOX[3]/2.0
        fig.text(self.CBAR_LABEL_X, ybar, label,
                 ha='left', va='center', fontsize=10, color=c)
        return cb

    # -------------------------------------------------------------------
    def no_data(self, prod, region, run_dt, fhr, reason=""):
        """Render a placeholder tile (basemap + centered annotation) for
        the case where a product has no GRIB records at this fhour.

        Surfaces the reason to the user instead of returning a 404; gives
        operational context (e.g., "6-h accumulation needs F≥6, F003 is too
        early") rather than a blank failure."""
        import matplotlib.patheffects as pe
        fig, ax = self._setup(region)
        c = self.theme['fg']
        halo = self.theme.get('fig_axes', '#ffffff')
        line1 = f"No data for F{fhr:03d}"
        line2 = reason or "Product not available at this forecast hour"
        # Hint specific to accumulation products: the most common cause is
        # an n-hour product being asked at an fhour < n.
        step = prod.get('step') or (
            prod['step_from_fhr'](fhr) if 'step_from_fhr' in prod else None)
        if isinstance(step, str) and step.startswith('0-0'):
            line2 = (f"{prod['name']} is a multi-hour accumulation; "
                     f"earliest available frame is later in the run.")
        for y, txt, size, weight in [
            (0.54, line1, 22, 'bold'),
            (0.47, line2, 12, 'normal'),
        ]:
            t = fig.text(0.5, y, txt, ha='center', va='center',
                         fontsize=size, fontweight=weight, color=c, alpha=0.85)
            t.set_path_effects([pe.withStroke(linewidth=3, foreground=halo, alpha=0.9)])
        self._header(fig, prod, run_dt, fhr)
        return fig

    # -------------------------------------------------------------------
    def shaded(self, data, lats, lons, prod, region, run_dt, fhr,
               overlays=None):
        fig, ax = self._setup(region)
        cm, norm, lv = _CMAPS[prod['cmap']]()

        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        # Pre-display Gaussian smoothing for raw native-grid fields
        # (divergence, moisture convergence, etc.) that are visually
        # dominated by salt-and-pepper at full resolution. Opt-in per
        # product via `smooth_sigma`.
        sigma = prod.get('smooth_sigma')
        if HAS_SCIPY and sigma:
            data = gaussian_filter(np.asarray(data, dtype=float), sigma=sigma)

        # Most products mask values BELOW the colorbar minimum (no data
        # vs. genuinely-zero ambiguity). CIN inverts that. IR satellite +
        # divergence-style fields want to render every pixel — extreme
        # values get clipped to the under/over colors by BoundaryNorm.
        # Diverging fields (div / mconv) can also opt into hiding a
        # near-zero band via `mask_below_abs` so basemap shows through.
        if prod['cmap'] in ('ir', 'div', 'mconv'):
            m_abs = prod.get('mask_below_abs')
            if m_abs is not None:
                masked = np.ma.masked_where(np.abs(data) < m_abs, data)
            else:
                masked = data
        elif prod['cmap'] == 'cin':
            mask = data > lv[-1]
            masked = np.ma.masked_where(mask, data)
        else:
            mask = data < lv[0]
            masked = np.ma.masked_where(mask, data)

        ext = ('both' if prod['cmap'] in ('t2m','td2m','cin','ir','div','mconv')
               else 'max')
        cf = ax.pcolormesh(lons2d, lats2d, masked, cmap=cm, norm=norm,
                           transform=ccrs.PlateCarree(), zorder=3,
                           shading='nearest', antialiased=False)
        self._colorbar(fig, cf, lv, prod.get('units',''), extend=ext)

        if overlays:
            for ov in overlays:
                self._draw_overlay(ax, ov)

        self._header(fig, prod, run_dt, fhr)
        return fig

    def spc_prob_field(self, data, lats, lons, prod, region, run_dt, fhr):
        """Render an SPC HREF calibrated probability field (severe / thunder /
        lightning density) as filled categorical contours with labeled
        boundaries — matching the SPC experimental HREF page look, but in the
        app's theme / sector / basemap."""
        import cartopy.crs as ccrs
        fig, ax = self._setup(region)
        cm, norm, lv = _CMAPS[prod['cmap']]()
        levels = list(lv)

        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        d = np.asarray(data, dtype=float)
        # Light smoothing keeps the coarse ~40-km grid from looking blocky,
        # like the SPC product. Opt-out via smooth_sigma=0.
        sigma = prod.get('smooth_sigma', 0.7)
        if HAS_SCIPY and sigma:
            d = gaussian_filter(d, sigma=sigma)

        # Filled categorical bands; anything below the first level is masked
        # so the basemap shows through (no "0%" wash).
        masked = np.ma.masked_less(d, levels[0])
        cf = ax.contourf(lons2d, lats2d, masked, levels=levels,
                         cmap=cm, norm=norm, extend='neither',
                         transform=ccrs.PlateCarree(), zorder=3)

        # Boundary lines + inline value labels (the "2", "5", … on SPC plots).
        line_levels = levels[:-1]   # drop the sentinel 100 cap
        cs = ax.contour(lons2d, lats2d, d, levels=line_levels,
                        colors=self.theme.get('cnt_prob', '#ffffff'),
                        linewidths=0.8, transform=ccrs.PlateCarree(), zorder=6)
        lbls = ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
        self._halo(cs, lbls, lw=2.6, label_lw=2.6)

        self._colorbar(fig, cf, line_levels, prod.get('units', '%'),
                       extend='max')
        self._header(fig, prod, run_dt, fhr)
        return fig

    def _draw_overlay(self, ax, ov, zorder=6):
        """Draw a contour overlay. ``zorder`` defaults to 6 (sits between
        shaded data at zorder=3 and basemap features at 8-9), but the
        paintball recipe passes a higher value so contours stay above the
        per-member fills at zorder=3+i."""
        if not ov: return
        data = ov['data']; lats = ov['lats']; lons = ov['lons']
        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats
        d = data
        if HAS_SCIPY and ov.get('smooth'):
            d = gaussian_filter(d, sigma=ov['smooth'])
        if np.nanmax(d) < min(ov['levels']):
            return
        # Resolve contour color: fall back to theme's probability color so
        # legacy near-black values (#3a1f00, #222, etc.) become readable in
        # dark mode.  The _LEGACY_DARK set covers all historically used values.
        _LEGACY_DARK = {'#3a1f00', '#222222', '#222', '#1a1a1a',
                        '#2a1800', '#3a2a10',
                        '#000000', '#000', 'black', '#101010', '#1c1c1c'}
        ov_color = ov.get('colors', None)
        if isinstance(ov_color, dict):
            # Explicit per-theme pair: {'light': ..., 'dark': ...}. Dark
            # theme is the one that defines a contour casing color.
            ov_color = ov_color.get(
                'dark' if self.theme.get('cnt_halo') else 'light', '#ffffff')
        if ov_color is None or (
                isinstance(ov_color, str) and ov_color.lower() in _LEGACY_DARK):
            ov_color = self.theme.get('cnt_prob', '#3a1f00')
        cs = ax.contour(lons2d, lats2d, d, levels=ov['levels'],
                        colors=ov_color,
                        linewidths=ov.get('linewidths', 1.2),
                        linestyles=ov.get('linestyles', 'solid'),
                        transform=ccrs.PlateCarree(), zorder=zorder)
        # Casing behind every contour line + label so they stay legible over
        # any shaded field and any palette. Dark mode uses an opaque BLACK
        # casing (cnt_halo) — thick + solid so white lines read over bright
        # fills. Light mode falls back to a white casing (fig_axes), matching
        # the original behavior on the light figure background.
        import matplotlib.patheffects as pe
        _dark_halo = self.theme.get('cnt_halo')          # None in light mode
        halo_fg = _dark_halo or self.theme.get('fig_axes', '#ffffff')
        _line_lw  = 4.0 if _dark_halo else 2.6
        _lbl_lw   = 3.4 if _dark_halo else 2.2
        _halo_a   = 1.0 if _dark_halo else 0.85
        try:
            for col in cs.collections:
                col.set_path_effects([pe.withStroke(
                    linewidth=_line_lw, foreground=halo_fg, alpha=_halo_a)])
        except AttributeError:
            # mpl >= 3.8 removed .collections; contour objects support
            # set_path_effects directly.
            try:
                cs.set_path_effects([pe.withStroke(
                    linewidth=_line_lw, foreground=halo_fg, alpha=_halo_a)])
            except Exception:
                pass
        labels = ax.clabel(cs, inline=True, fontsize=8.5, fmt='%g')
        for lbl in labels:
            lbl.set_path_effects([pe.withStroke(
                linewidth=_lbl_lw, foreground=halo_fg, alpha=max(_halo_a, 0.95))])

    # -------------------------------------------------------------------
    def wind_level(self, u, v, gh, lats, lons, prod, region, run_dt, fhr,
                   t=None, rh=None, dzdt=None):
        fig, ax = self._setup(region)
        cm, norm, lv = _CMAPS[prod['cmap']]()
        spd_kt = np.sqrt(u**2 + v**2) * 1.94384
        hgt_dam = gh / 10.0

        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        masked = np.ma.masked_less(spd_kt, lv[0])
        cf = ax.pcolormesh(lons2d, lats2d, masked, cmap=cm, norm=norm,
                           transform=ccrs.PlateCarree(), zorder=3,
                           shading='nearest', antialiased=False)

        if HAS_SCIPY: hgt_dam = gaussian_filter(hgt_dam, sigma=1.5)
        level = prod['level']
        # Standard contour intervals per level
        ci = {250: 12, 500: 6, 700: 3, 850: 3, 925: 3}.get(level, 6)
        cmin = np.floor(np.nanmin(hgt_dam)/ci)*ci
        cmax = np.ceil(np.nanmax(hgt_dam)/ci)*ci
        clev = np.arange(cmin, cmax+ci, ci)
        cs = ax.contour(lons2d, lats2d, hgt_dam, levels=clev,
                        colors=self.theme.get('cnt_h', '#3a2a10'), linewidths=1.3,
                        transform=ccrs.PlateCarree(), zorder=6)
        lbls = ax.clabel(cs, inline=True, fontsize=9, fmt='%1.0f')
        self._halo(cs, lbls)

        # RH shading — 700 mb starts at 70%, 850/925 mb at 80%
        if rh is not None:
            rh_thresh = 80 if level in (850, 925) else 70
            ax.contourf(lons2d, lats2d, rh, levels=[rh_thresh, 90, 101],
                        colors=['#a8d8a8', '#3a9a3a'], alpha=0.45,
                        transform=ccrs.PlateCarree(), zorder=4)
        # Vertical motion (omega) contours for 700 mb — DZDT in m/s (positive = rising).
        if dzdt is not None:
            if HAS_SCIPY:
                _dz = gaussian_filter(dzdt, sigma=2.0)
            else:
                _dz = dzdt
            _pos_levs = [0.15, 0.3, 0.6, 1.0, 1.5]
            _neg_levs = [-1.5, -1.0, -0.6, -0.3, -0.15]
            _dz_max = np.nanmax(_dz)
            _dz_min = np.nanmin(_dz)
            if _dz_max >= _pos_levs[0]:
                _dzc = ax.contour(lons2d, lats2d, _dz, levels=_pos_levs,
                           colors=self._ink('#bb00bb', '#ff66ff'), linewidths=1.1,
                           linestyles='solid',
                           transform=ccrs.PlateCarree(), zorder=8)
                self._halo(_dzc, lw=3.0)
            if _dz_min <= _neg_levs[-1]:
                _dzc = ax.contour(lons2d, lats2d, _dz, levels=_neg_levs,
                           colors=self._ink('#cc5500', '#ffa83a'), linewidths=1.1,
                           linestyles='dashed',
                           transform=ccrs.PlateCarree(), zorder=8)
                self._halo(_dzc, lw=3.0)
        # Temperature contours for 850/925 mb
        if t is not None:
            tC = t - 273.15
            if HAS_SCIPY: tC = gaussian_filter(tC, sigma=1.5)
            tcs = ax.contour(lons2d, lats2d, tC,
                             levels=np.arange(-40, 45, 2),
                             colors=self._ink('#bf1a1a', '#ff7a7a'), linewidths=0.8,
                             linestyles='--',
                             transform=ccrs.PlateCarree(), zorder=5)
            tcs_lbls = ax.clabel(tcs, inline=True, fontsize=7.5, fmt='%g')
            self._halo(tcs, tcs_lbls, lw=2.6, label_lw=2.6)
            zero = ax.contour(lons2d, lats2d, tC, levels=[0],
                              colors=self._ink('#0033bf', '#5b9bff'), linewidths=1.5,
                              transform=ccrs.PlateCarree(), zorder=5)
            zero_lbls = ax.clabel(zero, inline=True, fontsize=8, fmt='%g')
            self._halo(zero, zero_lbls, lw=3.6, label_lw=3.2)

        # 250 mb divergence contours (divergence only; positive = solid red).
        # Computed from ∂u/∂x + ∂v/∂y on a regular lat/lon grid.
        # Units scaled to 10⁻⁵ s⁻¹.
        if level == 250:
            _R = 6371000.0
            _lat_r = np.deg2rad(lats2d)
            # grid spacing in radians – use finite diff on first row/col
            if lats.ndim == 1:
                _dlat = np.deg2rad(abs(lats[1] - lats[0]))
                _dlon = np.deg2rad(abs(lons[1] - lons[0]))
            else:
                _dlat = np.deg2rad(abs(lats2d[1, 0] - lats2d[0, 0]))
                _dlon = np.deg2rad(abs(lons2d[0, 1] - lons2d[0, 0]))
            _du_dx = np.gradient(u, _dlon, axis=1) / (_R * np.cos(_lat_r))
            _dv_dy = np.gradient(v, _dlat, axis=0) / _R
            div = (_du_dx + _dv_dy) * 1e5   # 10⁻⁵ s⁻¹
            if HAS_SCIPY:
                div = gaussian_filter(div, sigma=2.5)
            _div_levs_pos = [8, 16, 24, 32]
            if np.nanmax(div) >= _div_levs_pos[0]:
                _divc = ax.contour(lons2d, lats2d, div, levels=_div_levs_pos,
                           colors=self._ink('#cc2200', '#ff6a4a'), linewidths=1.1,
                           linestyles='solid',
                           transform=ccrs.PlateCarree(), zorder=8)
                self._halo(_divc, lw=3.0)

        skip = 45
        ax.barbs(lons2d[::skip, ::skip], lats2d[::skip, ::skip],
                 u[::skip,::skip]*1.94384, v[::skip,::skip]*1.94384,
                 transform=ccrs.PlateCarree(), length=5.2, linewidth=0.55,
                 zorder=7)

        self._colorbar(fig, cf, lv, 'kt', extend='max')
        self._header(fig, prod, run_dt, fhr)
        return fig

    # -------------------------------------------------------------------
    def vort_level(self, u, v, gh, lats, lons, prod, region, run_dt, fhr):
        """500 mb absolute vorticity (shaded) + heights + wind barbs."""
        fig, ax = self._setup(region)

        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        # Absolute vorticity: ζ_abs = (∂v/∂x − ∂u/∂y) + f
        _R = 6371000.0
        _lat_r = np.deg2rad(lats2d)
        _dlat = np.deg2rad(abs(lats[1] - lats[0]) if lats.ndim == 1
                           else abs(lats2d[1, 0] - lats2d[0, 0]))
        _dlon = np.deg2rad(abs(lons[1] - lons[0]) if lons.ndim == 1
                           else abs(lons2d[0, 1] - lons2d[0, 0]))
        _dv_dx = np.gradient(v, _dlon, axis=1) / (_R * np.cos(_lat_r))
        _du_dy = np.gradient(u, _dlat, axis=0) / _R
        _f = 2.0 * 7.2921e-5 * np.sin(_lat_r)
        abs_vort = (_dv_dx - _du_dy + _f) * 1e5   # ×10⁻⁵ s⁻¹

        if HAS_SCIPY:
            abs_vort = gaussian_filter(abs_vort, sigma=1.5)

        cm, norm, lv = _CMAPS['vort']()
        masked = np.ma.masked_less(abs_vort, lv[0])
        cf = ax.pcolormesh(lons2d, lats2d, masked, cmap=cm, norm=norm,
                           transform=ccrs.PlateCarree(), zorder=3,
                           shading='nearest', antialiased=False)

        # Height contours (6-dam interval for 500 mb)
        hgt_dam = gh / 10.0
        if HAS_SCIPY:
            hgt_dam = gaussian_filter(hgt_dam, sigma=1.5)
        ci = 6
        clev = np.arange(np.floor(np.nanmin(hgt_dam)/ci)*ci,
                         np.ceil(np.nanmax(hgt_dam)/ci)*ci + ci, ci)
        cs = ax.contour(lons2d, lats2d, hgt_dam, levels=clev,
                        colors=self.theme.get('cnt_h', '#1a1a1a'), linewidths=1.3,
                        transform=ccrs.PlateCarree(), zorder=6)
        lbls = ax.clabel(cs, inline=True, fontsize=9, fmt='%1.0f')
        self._halo(cs, lbls)

        skip = 45
        ax.barbs(lons2d[::skip, ::skip], lats2d[::skip, ::skip],
                 u[::skip, ::skip] * 1.94384, v[::skip, ::skip] * 1.94384,
                 transform=ccrs.PlateCarree(), length=5.2, linewidth=0.55,
                 zorder=7)

        self._colorbar(fig, cf, lv, '×10⁻⁵ s⁻¹', extend='max')
        self._header(fig, prod, run_dt, fhr)
        return fig

    # -------------------------------------------------------------------
    def ptype_mslp(self, crain, csnow, cfrzr, cicep, mslp_pa,
                   lats, lons, prod, region, run_dt, fhr):
        """Precipitation type (categorical shading) + MSLP contours."""
        fig, ax = self._setup(region)

        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        # Shade each P-type as a contourf layer with alpha.
        # Priority order (lowest→highest zorder): snow, sleet, fzra, rain.
        # Using contourf avoids the cartopy imshow projection issues.
        thresh = 0.20
        _ptype_specs = [
            (csnow, '#8888ff', 'Snow'),
            (cicep, '#00bef0', 'Sleet'),
            (cfrzr, '#ff4dda', 'FZRA'),
            (crain, '#1abf1a', 'Rain'),
        ]
        for _z, (_arr, _color, _label) in enumerate(_ptype_specs, start=4):
            if _arr is None or np.nanmax(_arr) < thresh:
                continue
            ax.contourf(lons2d, lats2d, _arr,
                        levels=[thresh, 1.01],
                        colors=[_color], alpha=0.72,
                        transform=ccrs.PlateCarree(), zorder=_z)

        # MSLP contours
        mslp = mslp_pa / 100.0
        if HAS_SCIPY:
            mslp = gaussian_filter(mslp, sigma=2.0)
        clev_mslp = np.arange(940, 1060, 2)
        cs = ax.contour(lons2d, lats2d, mslp, levels=clev_mslp,
                        colors=self.theme.get('cnt_p', '#1a1a1a'), linewidths=0.9,
                        transform=ccrs.PlateCarree(), zorder=8)
        lbls = ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
        self._halo(cs, lbls)

        # Legend patches
        legend_items = [
            Patch(color='#1abf1a', label='Rain'),
            Patch(color='#ff4dda', label='FZRA'),
            Patch(color='#00bef0', label='Sleet'),
            Patch(color='#8888ff', label='Snow'),
        ]
        ax.legend(handles=legend_items, loc='lower left', fontsize=8,
                  framealpha=0.7, ncol=4)

        self._header(fig, prod, run_dt, fhr)
        return fig

    # -------------------------------------------------------------------
    def mslp_thickness(self, mslp_pa, t_unused, t_unused2, gh_low, gh500,
                       lats, lons, prod, region, run_dt, fhr):
        """MSLP contoured with thickness overlay. The low-level height field
        is 850 mb (REFS does not publish 1000 mb). 850-500 thickness 408 dm
        is the standard rain/snow boundary at the 850 mb base."""
        fig, ax = self._setup(region)
        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        mslp = mslp_pa / 100.0
        if HAS_SCIPY: mslp = gaussian_filter(mslp, sigma=2.0)
        clev = np.arange(940, 1060, 2)
        cs = ax.contour(lons2d, lats2d, mslp, levels=clev,
                        colors=self.theme.get('cnt_p', '#222222'), linewidths=1.0,
                        transform=ccrs.PlateCarree(), zorder=6)
        lbls = ax.clabel(cs, inline=True, fontsize=8.5, fmt='%g')
        self._halo(cs, lbls)

        # 850-500 mb thickness (dam). 408 dm ≈ 0°C 850-mb temperature; the
        # cold/warm shading uses ±20 dm bands around that.
        if gh500 is not None and gh_low is not None:
            th = (gh500 - gh_low) / 10.0
            if HAS_SCIPY: th = gaussian_filter(th, sigma=2.0)
            tlev_cold = np.arange(360, 408, 4)
            tlev_warm = np.arange(412, 460, 4)
            _thc = ax.contour(lons2d, lats2d, th, levels=tlev_cold,
                       colors=self._ink('#1f55e8', '#6aa8ff'), linewidths=0.9,
                       linestyles='--',
                       transform=ccrs.PlateCarree(), zorder=5)
            self._halo(_thc, lw=2.8)
            _th408 = ax.contour(lons2d, lats2d, th, levels=[408],
                       colors=self._ink('#5a1fa8', '#c08bff'), linewidths=1.6,
                       transform=ccrs.PlateCarree(), zorder=5)
            self._halo(_th408, lw=3.6)
            _thw = ax.contour(lons2d, lats2d, th, levels=tlev_warm,
                       colors=self._ink('#bf1a1a', '#ff7a7a'), linewidths=0.9,
                       linestyles='--',
                       transform=ccrs.PlateCarree(), zorder=5)
            self._halo(_thw, lw=2.8)
        self._header(fig, prod, run_dt, fhr)
        return fig

    def clouds_lmh(self, lcdc, mcdc, hcdc, lats, lons, prod, region, run_dt, fhr):
        """Low (blue) + Mid (green) + High (red) cloud cover overlay,
        matching the SPC HREF 'Cloud Cover' panel style."""
        fig, ax = self._setup(region)
        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        # Three discrete colormaps with increasing alpha (transparent below 10%).
        lv = [10, 25, 40, 55, 70, 85, 100]
        def _alpha_cmap(rgb, name):
            base = np.array(rgb, dtype=float)
            cols = []
            for i in range(len(lv)-1):
                t = (i+1) / (len(lv)-1)
                # blend white -> color, alpha 0.25 -> 0.85
                rgba = (1-0.85*t) + 0.85*t*base
                cols.append((rgba[0], rgba[1], rgba[2], 0.25 + 0.60*t))
            cm = ListedColormap(cols, name=name)
            return cm, BoundaryNorm(lv, cm.N)

        cm_lo, n_lo = _alpha_cmap((0.18,0.30,0.85), 'cloud_lo')   # blue
        cm_mi, n_mi = _alpha_cmap((0.12,0.55,0.20), 'cloud_mi')   # green
        cm_hi, n_hi = _alpha_cmap((0.78,0.18,0.18), 'cloud_hi')   # red

        # High first, mid, then low on top (so low clouds dominate near sfc).
        cf_hi = ax.pcolormesh(lons2d, lats2d, np.ma.masked_less(hcdc, lv[0]),
                              cmap=cm_hi, norm=n_hi,
                              transform=ccrs.PlateCarree(), zorder=3,
                              shading='nearest')
        cf_mi = ax.pcolormesh(lons2d, lats2d, np.ma.masked_less(mcdc, lv[0]),
                              cmap=cm_mi, norm=n_mi,
                              transform=ccrs.PlateCarree(), zorder=4,
                              shading='nearest')
        cf_lo = ax.pcolormesh(lons2d, lats2d, np.ma.masked_less(lcdc, lv[0]),
                              cmap=cm_lo, norm=n_lo,
                              transform=ccrs.PlateCarree(), zorder=5,
                              shading='nearest')

        # Three small colorbars sharing the standard CBAR_BOX band.
        c = self.theme['fg']
        left, bot, width, height = self.CBAR_BOX
        # 3 bars: each one quarter-bar wide, with gutters
        bar_w = (width - 0.04) / 3.0
        bots  = bot
        ticks = [25, 50, 75, 100]
        for i, (cf, lab) in enumerate([(cf_lo, 'Low'),
                                       (cf_mi, 'Mid'),
                                       (cf_hi, 'High')]):
            x0 = left + i*(bar_w + 0.02)
            cax = fig.add_axes([x0, bots, bar_w, height])
            cb = fig.colorbar(cf, cax=cax, orientation='horizontal',
                              ticks=ticks, extend='neither')
            cb.ax.tick_params(labelsize=8, length=3, pad=2, colors=c)
            cb.ax.set_xticklabels([str(v) for v in ticks], color=c)
            cb.outline.set_edgecolor(c)
            fig.text(x0 + bar_w/2.0, bots + height + 0.015, lab,
                     ha='center', va='bottom', fontsize=9.5,
                     fontweight='bold', color=c)
        self._header(fig, prod, run_dt, fhr)
        return fig

    def paintball(self, mems, prod, region, run_dt, fhr, overlays=None):
        """Overlay each member's threshold exceedance in a distinct color.

        ``overlays`` (optional) — list of overlay dicts already loaded via
        :meth:`_overlay_data` or :meth:`_joint_overlay_data`. Drawn on top
        of the member pixels at a high zorder so contour lines stay
        readable over the paintball colors.
        """
        fig, ax = self._setup(region)
        colors = cmap_paintball()
        thresh = prod.get('thresh', 40)
        from matplotlib.patches import Patch
        legend = []
        for i, (mem, data, lats, lons) in enumerate(mems):
            c = colors[i % len(colors)]
            if lats.ndim == 1:
                lons2d, lats2d = np.meshgrid(lons, lats)
            else:
                lons2d, lats2d = lons, lats
            mask = np.where(data >= thresh, 1.0, np.nan)
            ax.pcolormesh(lons2d, lats2d, mask,
                          cmap=ListedColormap([c]),
                          transform=ccrs.PlateCarree(), zorder=3+i,
                          shading='nearest')
            legend.append(Patch(facecolor=c, label=f"m{mem:03d}"))
        # Contour overlays sit on top of all member fills so they remain
        # readable on busy frames. Members go up to zorder=3+i (i up to
        # N-1, typically 7), so contours need to draw at zorder>=10 to
        # stay above every member's pcolormesh.
        if overlays:
            for ov in overlays:
                self._draw_overlay(ax, ov, zorder=10)
        # Member legend in the standard colorbar band (centered horizontally)
        ybar = self.CBAR_BOX[1] + self.CBAR_BOX[3]/2.0
        fig.legend(handles=legend, loc='center',
                   bbox_to_anchor=(0.5, ybar), ncol=len(legend),
                   fontsize=10, frameon=False,
                   labelcolor=self.theme['fg'])
        self._header(fig, prod, run_dt, fhr)
        return fig

    def stamps(self, mems, prod, region, run_dt, fhr):
        """Multi-panel grid showing each member side-by-side, packed inside
        the standard MAP_BOX so the figure matches single-panel layouts."""
        t = self.theme
        n = len(mems)
        ncols = 3 if n > 4 else 2
        nrows = (n + ncols - 1) // ncols
        fig = Figure(figsize=(self.FIG_W, self.FIG_H), dpi=self.DPI)
        fig.patch.set_facecolor(t['fig_face'])
        cm, norm, lv = _CMAPS[prod['cmap']]()
        r = REGIONS[region]
        proj = self._projection(region)

        # Pack the nrows x ncols subplots inside MAP_BOX with small gutters.
        left, bot, w, h = self.MAP_BOX
        gx, gy = 0.012, 0.030       # gutter (fraction of figure)
        cell_w = (w - (ncols-1)*gx) / ncols
        cell_h = (h - (nrows-1)*gy) / nrows
        cf = None
        # Unit conversion (e.g. hail m→in, gust m/s→kt). The colormap levels
        # are defined in display units; without this, raw SI values sit below
        # the first bin and every panel masks to blank.
        conv = prod.get('convert')
        # Optional display floor (display units). E.g. gust panels mask
        # below 20 kt — the ubiquitous 10-20 kt ambient gust field shades
        # nearly the whole domain and buries the convective signal.
        mask_lo = max(lv[0], prod.get('mask_below', lv[0]))
        for i, (mem, data, lats, lons) in enumerate(mems):
            if conv is not None:
                data = conv(data)
            row = i // ncols
            col = i %  ncols
            x0 = left + col*(cell_w + gx)
            y0 = bot + (nrows-1-row)*(cell_h + gy)
            ax = fig.add_axes([x0, y0, cell_w, cell_h], projection=proj)
            ax.set_facecolor(t['fig_axes'])
            ax.set_extent([r['lon'][0], r['lon'][1], r['lat'][0], r['lat'][1]],
                          crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND.with_scale('110m'),  facecolor=t['map_land'])
            ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor=t['map_ocean'])
            ax.add_feature(cfeature.STATES.with_scale('50m'),
                           edgecolor=t['map_state'], linewidth=0.4)
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                           edgecolor=t['map_border'], linewidth=0.4)
            if lats.ndim == 1:
                lons2d, lats2d = np.meshgrid(lons, lats)
            else:
                lons2d, lats2d = lons, lats
            masked = np.ma.masked_less(data, mask_lo)
            cf = ax.pcolormesh(lons2d, lats2d, masked, cmap=cm, norm=norm,
                               transform=ccrs.PlateCarree(), shading='nearest')
            ax.set_title(f"m{mem:03d}", fontsize=10, color=t['fg'], pad=2)
            ax._refs_default_extent = [r['lon'][0], r['lon'][1],
                                       r['lat'][0], r['lat'][1]]

        # Standard centered colorbar (same geometry as every other plot)
        if cf is not None:
            self._colorbar(fig, cf, lv, prod.get('units',''), extend='max')
        # Standard header (uses same Y positions as single-panel plots)
        self._header(fig, prod, run_dt, fhr)
        return fig

    def mean_spread(self, mean, spread, lats, lons, prod, region, run_dt, fhr):
        """Two modes:
          1. mean_contour=True  -> shaded SPREAD + contoured MEAN (synoptic style)
          2. otherwise          -> shaded MEAN + contoured SPREAD (CAPE style)"""
        fig, ax = self._setup(region)
        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        if prod.get('mean_contour'):
            # spread shaded
            cm, norm, lv = _CMAPS['sp_hgt' if prod['mean_var']=='gh_lvl'
                                  else 'sp_mslp']()
            sp = np.ma.masked_less(spread, lv[0])
            cf = ax.pcolormesh(lons2d, lats2d, sp, cmap=cm, norm=norm,
                               transform=ccrs.PlateCarree(), zorder=3,
                               shading='nearest')
            # mean contoured
            m = mean
            if HAS_SCIPY: m = gaussian_filter(m, sigma=1.5)
            ci = prod.get('mean_interval', 6)
            cmin = np.floor(np.nanmin(m)/ci)*ci
            cmax = np.ceil(np.nanmax(m)/ci)*ci
            clev = np.arange(cmin, cmax+ci, ci)
            cs = ax.contour(lons2d, lats2d, m, levels=clev,
                            colors=self.theme.get('cnt_h', '#222'), linewidths=1.1,
                            transform=ccrs.PlateCarree(), zorder=6)
            lbls = ax.clabel(cs, inline=True, fontsize=8.5, fmt='%g')
            self._halo(cs, lbls)
            self._colorbar(fig, cf, lv, f"spread ({prod.get('units','')})",
                           extend='max')
        else:
            # mean shaded
            cm, norm, lv = _CMAPS[prod['cmap']]()
            mm = np.ma.masked_less(mean, lv[0])
            cf = ax.pcolormesh(lons2d, lats2d, mm, cmap=cm, norm=norm,
                               transform=ccrs.PlateCarree(), zorder=3,
                               shading='nearest')
            # spread as dashed contours
            sp = spread
            if HAS_SCIPY: sp = gaussian_filter(sp, sigma=1.5)
            slev = prod.get('spread_levels', [250,500,750,1000])
            if np.nanmax(sp) >= slev[0]:
                cs = ax.contour(lons2d, lats2d, sp, levels=slev,
                                colors=self.theme.get('cnt_spd', '#222'),
                                linewidths=1.2,
                                linestyles='--',
                                transform=ccrs.PlateCarree(), zorder=6)
                lbls = ax.clabel(cs, inline=True, fontsize=8.5, fmt='%g')
                self._halo(cs, lbls, lw=3.2, label_lw=3.0)
            self._colorbar(fig, cf, lv, prod.get('units',''), extend='max')

        self._header(fig, prod, run_dt, fhr)
        return fig

    def combo_sfc(self, base, mslp_pa, u10, v10, lats, lons, prod, region,
                  run_dt, fhr):
        """Shaded surface field (T or Td) with MSLP isobars and 10-m barbs."""
        fig, ax = self._setup(region)
        cm, norm, lv = _CMAPS[prod['cmap']]()
        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        ext = 'both' if prod['cmap'] in ('t2m','td2m') else 'max'
        cf = ax.pcolormesh(lons2d, lats2d, base, cmap=cm, norm=norm,
                           transform=ccrs.PlateCarree(), zorder=3,
                           shading='nearest')

        if mslp_pa is not None:
            mslp = mslp_pa / 100.0
            if HAS_SCIPY: mslp = gaussian_filter(mslp, sigma=2.0)
            clev = np.arange(940, 1060, 2)
            cs = ax.contour(lons2d, lats2d, mslp, levels=clev,
                            colors=self.theme.get('cnt_p', '#222222'), linewidths=0.9,
                            transform=ccrs.PlateCarree(), zorder=6)
            lbls = ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
            self._halo(cs, lbls)

        if u10 is not None and v10 is not None:
            skip = 30
            ax.barbs(lons2d[::skip, ::skip], lats2d[::skip, ::skip],
                     u10[::skip,::skip]*1.94384, v10[::skip,::skip]*1.94384,
                     transform=ccrs.PlateCarree(), length=5.0, linewidth=0.45,
                     color=self.theme.get('cnt_p', '#1a1a1a'), zorder=7)

        self._colorbar(fig, cf, lv, prod.get('units',''), extend=ext)
        self._header(fig, prod, run_dt, fhr)
        return fig

    def wind_10m(self, u, v, lats, lons, prod, region, run_dt, fhr):
        fig, ax = self._setup(region)
        cm, norm, lv = _CMAPS[prod['cmap']]()
        spd_kt = np.sqrt(u**2 + v**2) * 1.94384
        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats
        masked = np.ma.masked_less(spd_kt, lv[0])
        cf = ax.pcolormesh(lons2d, lats2d, masked, cmap=cm, norm=norm,
                           transform=ccrs.PlateCarree(), zorder=3,
                           shading='nearest', antialiased=False)
        skip = 45
        ax.barbs(lons2d[::skip, ::skip], lats2d[::skip, ::skip],
                 u[::skip,::skip]*1.94384, v[::skip,::skip]*1.94384,
                 transform=ccrs.PlateCarree(), length=5.0, linewidth=0.5,
                 zorder=7)
        self._colorbar(fig, cf, lv, 'kt', extend='max')
        self._header(fig, prod, run_dt, fhr)
        return fig

    def storm_motion(self, spd_kt, u, v, lats, lons, prod, region, run_dt, fhr):
        """Storm motion (0-6 km mean wind) — shaded speed in kt + decimated
        arrows showing direction. Arrows are scaled so they read as direction
        only; the colorbar conveys magnitude."""
        fig, ax = self._setup(region)
        cm, norm, lv = _CMAPS[prod['cmap']]()
        if lats.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats
        masked = np.ma.masked_less(spd_kt, lv[0])
        cf = ax.pcolormesh(lons2d, lats2d, masked, cmap=cm, norm=norm,
                           transform=ccrs.PlateCarree(), zorder=3,
                           shading='nearest', antialiased=False)
        # Decimate to ~30 arrows wide for legibility.
        ny, nx = u.shape
        skip = max(1, nx // 30)
        ax.quiver(lons2d[::skip, ::skip], lats2d[::skip, ::skip],
                  u[::skip, ::skip], v[::skip, ::skip],
                  transform=ccrs.PlateCarree(),
                  scale=420, width=0.0022, color='#0a0a0a',
                  alpha=0.85, zorder=7)
        self._colorbar(fig, cf, lv, 'kt', extend='max')
        self._header(fig, prod, run_dt, fhr)
        return fig

# colormap registry must be defined after fns
# --- SPC HREF calibrated guidance probability scales -----------------------
# Discrete categorical scales matching the SPC experimental HREF page.
def cmap_spc_tor():
    # Tornado / STP categorical severe-probability scale (%)
    cs = ['#008b00', '#8b4726', '#ffc800', '#e60000',
          '#ff00ff', '#912cee', '#add8e6']
    lv = [2, 5, 10, 15, 30, 45, 60, 100]
    return _cmap(cs, lv, 'spc_tor')

def cmap_spc_hailwind():
    # Hail / Wind categorical severe-probability scale (%)
    cs = ['#8b4726', '#ffc800', '#e60000', '#ff00ff', '#912cee']
    lv = [5, 15, 30, 45, 60, 100]
    return _cmap(cs, lv, 'spc_hailwind')

def cmap_spc_thunder():
    # Calibrated thunderstorm probability (%), green→red ramp
    cs = ['#bdebbd', '#74d174', '#27a327', '#fff36b', '#ffc24d',
          '#ff7a3c', '#ff3b3b', '#c81e1e', '#a020f0']
    lv = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    return _cmap(cs, lv, 'spc_thunder')

def cmap_spc_ltg():
    # Lightning-strike probability (%) for a flash-count threshold
    cs = ['#cfe8ff', '#7fb2ff', '#2f7bff', '#ffd24d',
          '#ff8c1a', '#ff3b3b', '#a020f0']
    lv = [5, 10, 20, 30, 50, 70, 90, 100]
    return _cmap(cs, lv, 'spc_ltg')


_CMAPS = {
    'refc': cmap_refc, 'uh': cmap_uh, 'cape': cmap_cape, 'mucape': cmap_mucape,
    'spc_tor': cmap_spc_tor, 'spc_hailwind': cmap_spc_hailwind,
    'spc_thunder': cmap_spc_thunder, 'spc_ltg': cmap_spc_ltg,
    'cin': cmap_cin, 'qpf': cmap_qpf, 'prob': cmap_prob,
    'w500': cmap_wind500, 'w250': cmap_wind250, 'wind500': cmap_wind500,
    'wind250': cmap_wind250, 'pwat': cmap_pwat, 'pwat_in': cmap_pwat_in,
    'retop_kft': cmap_retop_kft, 'srh': cmap_srh, 't2m': cmap_t2m,
    'td2m': cmap_td2m, 'snow': cmap_snow, 'clouds': cmap_clouds, 'vis': cmap_vis,
    'sp_cape': cmap_spread_cape, 'sp_hgt': cmap_spread_hgt,
    'sp_mslp': cmap_spread_mslp,
    'composite': cmap_composite, 'lapse': cmap_lapse_rate,
    'div': cmap_divergence,
    'vort': cmap_vorticity,
    'vil': cmap_vil, 'gust': cmap_gust, 'dcape': cmap_dcape, 'hail': cmap_hail,
    'mconv': cmap_mconv, 'smot': cmap_smotion,
    'ir': cmap_ir_satellite, 'wfire': cmap_wfirepot,
}

# =============================================================================
#  Plot orchestration -- ties products to processor + plot manager
# =============================================================================
class PlotJob:
    def __init__(self, processor, plot_manager):
        self.proc = processor
        self.pm = plot_manager

    def _overlay_data(self, ov, date_str, run, fhr, status_cb):
        try:
            f = self.proc.find_or_fetch(date_str, run, fhr, ov['ftype'], status_cb)
        except Exception as e:
            print(f"[overlay] cannot fetch {ov['ftype']}: {e}")
            return None
        step = ov.get('step') or (ov['step_from_fhr'](fhr) if 'step_from_fhr' in ov else None)
        data, lats, lons = self.proc.load_var(
            f, ov['var'], level=ov.get('level'), step=step,
            thresh=ov.get('thresh'))
        if data is None:
            print(
                f"[overlay] no GRIB match: ftype={ov['ftype']} var={ov['var']} "
                f"step={step} thresh={ov.get('thresh')} F{fhr:03d}"
            )
            return None
        # Also log when data exists but is entirely below the minimum contour
        # level (so the user sees no contours and may wonder why).
        try:
            import numpy as _np
            mx = float(_np.nanmax(data))
            lvls = ov.get('levels') or []
            if lvls and mx < min(lvls):
                print(
                    f"[overlay] data max {mx:.2f} below min level {min(lvls)} "
                    f"for {ov['var']} F{fhr:03d} -- no contours drawn"
                )
        except Exception:
            pass
        out = dict(ov); out.update(data=data, lats=lats, lons=lons)
        return out

    def render(self, pid, date_str, run, fhr, region, run_dt, status_cb):
        prod = PRODUCTS[pid]

        # ---- Recipe products -------------------------------------------
        recipe = prod.get('recipe')
        if recipe == 'wind_level':
            return self._wind_level(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'wind_700_rh':
            return self._wind_level(prod, date_str, run, fhr, region, run_dt,
                                    status_cb, with_rh=True, with_omega=True)
        if recipe == 'wind_850_temp':
            return self._wind_level(prod, date_str, run, fhr, region, run_dt,
                                    status_cb, with_temp=True, with_rh_from_dpt=True)
        if recipe == 'vort_level':
            return self._vort_level(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'ptype_mslp':
            return self._ptype_mslp(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'wind_10m':
            return self._wind_10m(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'mslp_thickness':
            return self._mslp(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'combo_sfc':
            return self._combo_sfc(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'mean_spread':
            return self._mean_spread(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'paintball':
            return self._paintball(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'stamps':
            return self._stamps(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'clouds_lmh':
            return self._clouds_lmh(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'heat_index':
            return self._heat_index(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'composite':
            return self._composite(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'member_mean':
            return self._member_mean(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'member_prob':
            return self._member_prob(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'storm_motion':
            return self._storm_motion(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'spc_prob':
            return self._spc_prob(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'prob_window':
            return self._prob_window(prod, date_str, run, fhr, region, run_dt, status_cb)
        if recipe == 'qpf_sum':
            return self._qpf_sum(prod, date_str, run, fhr, region, run_dt, status_cb)

        # ---- Standard shaded product -----------------------------------
        f = self.proc.find_or_fetch(date_str, run, fhr, prod['ftype'], status_cb)
        step = prod.get('step') or (prod['step_from_fhr'](fhr)
                                    if 'step_from_fhr' in prod else None)
        status_cb(f"Reading {prod['name']}...")
        data, lats, lons = self.proc.load_var(
            f, prod['var'], level=prod.get('level'), step=step,
            thresh=prod.get('thresh'), below=prod.get('prob_below', False))
        if data is None:
            status_cb(f"Variable not present: {prod['name']} F{fhr:03d}")
            return None
        if 'convert' in prod:
            data = prod['convert'](data)

        overlays = []
        for ov in prod.get('overlay', []) or []:
            od = self._overlay_data(ov, date_str, run, fhr, status_cb)
            if od is not None:
                overlays.append(od)
        return self.pm.shaded(data, lats, lons, prod, region, run_dt, fhr,
                              overlays=overlays)

    def _prob_window(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """N-hr neighborhood probability built from REFS hourly NP records:
        elementwise max of the hourly NPs across the window ending at fhr.

        Matches the SPC HREF "4-hr NP" display convention — REFS publishes
        hourly NPs only, and an hourly NP reads systematically lower than
        HREF's windowed product for the same scenario. Max-of-hourly-NPs is
        a (slightly low) approximation of the true windowed NP.
        """
        fhrs = self._window_fhrs(fhr, prod.get('window_h', 4))
        agg = None
        lats = lons = None
        status_cb(f"Reading {prod['name']}...")
        for w in fhrs:
            try:
                f = self.proc.find_or_fetch(date_str, run, w, prod['ftype'],
                                            status_cb)
            except Exception as e:
                print(f"[prob-window] F{w:03d}: {e}", flush=True)
                continue
            d, la, lo = self.proc.load_var(
                f, prod['var'], level=prod.get('level'),
                thresh=prod.get('thresh'),
                below=prod.get('prob_below', False))
            if d is None:
                continue
            agg = d.astype(float) if agg is None else np.fmax(agg, d)
            if lats is None:
                lats, lons = la, lo
        if agg is None:
            status_cb(f"No prob records: {prod['name']} F{fhr:03d}")
            return None
        overlays = []
        for ov in prod.get('overlay', []) or []:
            od = self._overlay_data(ov, date_str, run, fhr, status_cb)
            if od is not None:
                overlays.append(od)
        return self.pm.shaded(agg, lats, lons, prod, region, run_dt, fhr,
                              overlays=overlays)

    # ----- SPC HREF calibrated guidance ------------------------------------
    def _spc_prob(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        f = self.proc.find_or_fetch(date_str, run, fhr, prod['ftype'], status_cb)
        if f is None:
            status_cb(f"Unavailable: {prod['name']} F{fhr:03d}")
            return None
        status_cb(f"Reading {prod['name']}...")
        data, lats, lons = self.proc.load_var(
            f, prod['var'], thresh=prod.get('thresh'))
        if data is None:
            status_cb(f"Variable not present: {prod['name']} F{fhr:03d}")
            return None
        return self.pm.spc_prob_field(data, lats, lons, prod, region, run_dt, fhr)

    # ----- wind level helper -----------------------------------------------
    def _wind_level(self, prod, date_str, run, fhr, region, run_dt, status_cb,
                    with_rh=False, with_temp=False,
                    with_omega=False, with_rh_from_dpt=False):
        f = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        lvl = prod['level']
        status_cb(f"Reading {lvl}mb wind/heights...")
        u, lats, lons = self.proc.load_var(f, 'u_lvl', level=lvl)
        v, _, _       = self.proc.load_var(f, 'v_lvl', level=lvl)
        gh,_,_        = self.proc.load_var(f, 'gh_lvl', level=lvl)
        if u is None or v is None or gh is None:
            status_cb(f"Missing wind/height at {lvl}mb F{fhr:03d}")
            return None
        t = rh = dzdt = None
        if with_temp or with_rh_from_dpt:
            t,_,_ = self.proc.load_var(f, 't_lvl', level=lvl)
        if with_rh:
            rh,_,_ = self.proc.load_var(f, 'rh_lvl', level=lvl)
        elif with_rh_from_dpt:
            # REFS mean only has RH directly at 700 mb; derive it at 850/925 mb
            # from the dewpoint depression: RH = 100 * e(Td)/e(T).
            dpt,_,_ = self.proc.load_var(f, 'dpt_lvl', level=lvl)
            if dpt is not None and t is not None:
                _td_c = dpt - 273.15
                _t_c  = t   - 273.15
                _e    = np.exp(17.625 * _td_c / (243.04 + _td_c))
                _es   = np.exp(17.625 * _t_c  / (243.04 + _t_c ))
                rh    = np.clip(100.0 * _e / _es, 0.0, 100.0)
        if with_omega:
            dzdt,_,_ = self.proc.load_var(f, 'dzdt_lvl', level=lvl)
        return self.pm.wind_level(u, v, gh, lats, lons, prod, region, run_dt,
                                  fhr, t=t, rh=rh, dzdt=dzdt)

    def _vort_level(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        f = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        lvl = prod['level']
        status_cb(f"Reading {lvl}mb wind/heights for vorticity...")
        u, lats, lons = self.proc.load_var(f, 'u_lvl', level=lvl)
        v, _, _       = self.proc.load_var(f, 'v_lvl', level=lvl)
        gh,_,_        = self.proc.load_var(f, 'gh_lvl', level=lvl)
        if u is None or v is None or gh is None:
            status_cb(f"Missing wind/height at {lvl}mb F{fhr:03d}")
            return None
        return self.pm.vort_level(u, v, gh, lats, lons, prod, region, run_dt, fhr)

    def _ptype_mslp(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        f = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        status_cb("Reading precipitation type + MSLP...")
        crain, lats, lons = self.proc.load_var(f, 'crain')
        csnow, _, _       = self.proc.load_var(f, 'csnow')
        cfrzr, _, _       = self.proc.load_var(f, 'cfrzr')
        cicep, _, _       = self.proc.load_var(f, 'cicep')
        mslp,  _, _       = self.proc.load_var(f, 'mslp')
        if mslp is None:
            status_cb(f"Missing MSLP F{fhr:03d}"); return None
        # Fill missing P-type arrays with zeros so the renderer can continue
        _zero = np.zeros_like(mslp)
        crain = crain if crain is not None else _zero
        csnow = csnow if csnow is not None else _zero
        cfrzr = cfrzr if cfrzr is not None else _zero
        cicep = cicep if cicep is not None else _zero
        return self.pm.ptype_mslp(crain, csnow, cfrzr, cicep, mslp,
                                  lats, lons, prod, region, run_dt, fhr)

    def _wind_10m(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        f = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        u, lats, lons = self.proc.load_var(f, 'u_10m')
        v,_,_         = self.proc.load_var(f, 'v_10m')
        if u is None or v is None:
            status_cb(f"Missing 10-m wind F{fhr:03d}"); return None
        return self.pm.wind_10m(u, v, lats, lons, prod, region, run_dt, fhr)

    @staticmethod
    def _window_fhrs(fhr, window):
        """Hourly record fhrs covering the `window`-hour period ending at
        `fhr` (each member record at w spans (w-1, w])."""
        window = int(window or 1)
        return [w for w in range(fhr - window + 1, fhr + 1) if w >= 1] or [fhr]

    def _load_members(self, prod, date_str, run, fhr, status_cb, n=None):
        """Return list of (member_id, data2d, lats, lons) for available members.

        If prod['window_h'] = W > 1, each member's field is the elementwise
        max over the W hourly records ending at fhr. This reproduces the SPC
        HREF "4-hr max" convention — REFS only publishes 1-h records, and a
        1-h field paints dramatically less area than HREF's 4-h product.
        """
        n = n or N_MEMBERS
        product = prod.get('member_product', '2dfld')
        fhrs = self._window_fhrs(fhr, prod.get('window_h', 1))
        results = []
        for mem in range(1, n+1):
            agg = None
            lats = lons = None
            for w in fhrs:
                try:
                    f = self.proc.find_or_fetch_member(date_str, run, w, mem,
                                                       product, status_cb)
                except Exception as e:
                    print(f"[member] m{mem} F{w}: {e}")
                    continue
                data, la, lo = self.proc.load_var(f, prod['var'],
                    level=prod.get('level'), step=prod.get('step'))
                if data is None:
                    # `f` may be a byte-range partial cached for a *different*
                    # product, missing this var's record entirely. Append just
                    # that record and retry once (mtime change invalidates the
                    # negative dataset-cache entry).
                    heal_match = self._member_idx_match(prod, w)
                    healed = False
                    if heal_match is not None:
                        try:
                            healed = self.proc.ensure_member_record(
                                f, date_str, run, w, mem, heal_match, product)
                        except AttributeError:
                            pass   # processor without partial support (HREF)
                    if healed:
                        data, la, lo = self.proc.load_var(f, prod['var'],
                            level=prod.get('level'), step=prod.get('step'))
                if data is None:
                    continue
                agg = data if agg is None else np.fmax(agg, data)
                if lats is None:
                    lats, lons = la, lo
            if agg is not None:
                results.append((mem, agg, lats, lons))
        return results

    @staticmethod
    def _member_idx_match(prod, fhr):
        """wgrib2 .idx substring for this product's member-file record, or
        None when unmapped. Uses the member-aware mapping (hourly-max vars)."""
        try:
            from app.idx_match import member_match_for
        except ImportError:
            return None
        try:
            return member_match_for(prod['var'], fhr=fhr,
                                    level=prod.get('level'))
        except Exception:
            return None

    def _paintball(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        mems = self._load_members(prod, date_str, run, fhr, status_cb)
        if not mems:
            status_cb(f"No members loaded for paintball F{fhr:03d}"); return None
        overlays = []
        # Single-variable contour overlays (same shape as shaded products).
        for ov in prod.get('overlay', []) or []:
            od = self._overlay_data(ov, date_str, run, fhr, status_cb)
            if od is not None:
                overlays.append(od)
        # Joint-probability overlays — combine multiple (ftype, var, thresh)
        # prob fields via min/product. Approximates joint NHP without
        # paying for a per-member compute.
        for ovp in prod.get('joint_overlay', []) or []:
            od = self._joint_overlay_data(ovp, date_str, run, fhr, status_cb,
                                          window=prod.get('window_h', 1))
            if od is not None:
                overlays.append(od)
        return self.pm.paintball(mems, prod, region, run_dt, fhr,
                                 overlays=overlays)

    def _joint_overlay_data(self, ov, date_str, run, fhr, status_cb,
                            window=1):
        """Load multiple per-variable prob fields and combine into one
        joint-probability proxy for use as a contour overlay.

        Spec format:
            dict(specs=[(ftype, var, thresh), ...],
                 combine='min' | 'product',     # default 'min'
                 style='contour', levels=[...], colors=..., linewidths=...,
                 smooth=2.0)

        Why min/product (not true joint): REFS publishes per-variable
        neighborhood probabilities, not joint ones. Computing true joint
        from members costs N_members × N_vars partial fetches per fhr and
        is too expensive on this pool. ``min(P1, P2, ...)`` is an upper
        bound on joint NHP and tends to read well visually (vivid
        contours where every condition is at least moderately probable).
        ``product`` is an independence-assumption lower bound (smaller
        contours).
        """
        import numpy as _np
        specs = ov.get('specs') or []
        if not specs:
            return None
        # REFS publishes hourly NPs only. For windowed products (SPC's 4-hr
        # convention) take the elementwise max of the hourly NPs across the
        # window before combining — close approximation of the windowed NP.
        fhrs = self._window_fhrs(fhr, ov.get('window_h', window))
        fields = []
        lats = lons = None
        for ftype, var, thresh in specs:
            agg = None
            for w in fhrs:
                try:
                    f = self.proc.find_or_fetch(date_str, run, w, ftype,
                                                status_cb)
                except Exception as e:
                    print(f"[joint-overlay] F{w:03d} cannot fetch {ftype}: {e}",
                          flush=True)
                    continue
                d, la, lo = self.proc.load_var(f, var, thresh=thresh)
                if d is None:
                    print(f"[joint-overlay] F{w:03d} no record: ftype={ftype} "
                          f"var={var} thresh={thresh}", flush=True)
                    continue
                agg = d.astype(float) if agg is None else _np.fmax(agg, d)
                if lats is None:
                    lats, lons = la, lo
            if agg is None:
                return None
            fields.append(agg)
        mode = ov.get('combine', 'min')
        if mode == 'product':
            joint = fields[0].astype(float)
            for d in fields[1:]:
                joint = joint * d.astype(float) / 100.0
        else:
            joint = fields[0].astype(float)
            for d in fields[1:]:
                joint = _np.minimum(joint, d.astype(float))
        out = dict(ov); out.update(data=joint, lats=lats, lons=lons)
        return out

    def _stamps(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        mems = self._load_members(prod, date_str, run, fhr, status_cb)
        if not mems:
            status_cb(f"No members loaded for stamps F{fhr:03d}"); return None
        return self.pm.stamps(mems, prod, region, run_dt, fhr)

    def _clouds_lmh(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """Three-layer cloud cover overlay (low/mid/high)."""
        f = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        status_cb("Reading low/mid/high cloud cover...")
        lcdc, lats, lons = self.proc.load_var(f, 'lcdc')
        mcdc, _, _       = self.proc.load_var(f, 'mcdc')
        hcdc, _, _       = self.proc.load_var(f, 'hcdc')
        if lcdc is None or mcdc is None or hcdc is None:
            status_cb("Cloud layers missing"); return None
        return self.pm.clouds_lmh(lcdc, mcdc, hcdc, lats, lons, prod,
                                  region, run_dt, fhr)

    def _composite(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """Compute a derived field from several REFS-mean ingredients.

        Driven by ``prod['ingredients']`` (dict name → {ftype, var, [level,
        step, thresh]}) and ``prod['composite_fn']`` (string key into the
        module-level COMPOSITES registry). The compute function gets a dict
        of {name: 2-D ndarray} plus ``lats`` / ``lons`` kwargs and returns
        one 2-D field that pm.shaded plots.
        """
        pid_dbg = prod.get('name', '?')
        fn_key = prod.get('composite_fn')
        fn = COMPOSITES.get(fn_key)
        if fn is None:
            print(f"[composite] {pid_dbg} F{fhr:03d}: composite_fn "
                  f"'{fn_key}' not in COMPOSITES (have: "
                  f"{sorted(COMPOSITES.keys())})", flush=True)
            return None
        ingredients = prod.get('ingredients') or {}
        loaded: dict = {}
        lats = lons = None
        # Group ingredients by ftype so each file is opened once.
        by_ftype: dict[str, list] = {}
        for name, spec in ingredients.items():
            by_ftype.setdefault(spec['ftype'], []).append((name, spec))
        for ftype, items in by_ftype.items():
            try:
                f = self.proc.find_or_fetch(date_str, run, fhr, ftype, status_cb)
            except Exception as e:
                print(f"[composite] {pid_dbg} F{fhr:03d}: cannot fetch "
                      f"{ftype}: {type(e).__name__}: {e}", flush=True)
                return None
            for name, spec in items:
                data, la, lo = self.proc.load_var(
                    f, spec['var'], level=spec.get('level'),
                    step=spec.get('step'), thresh=spec.get('thresh'))
                if data is None:
                    print(f"[composite] {pid_dbg} F{fhr:03d}: missing "
                          f"ingredient '{name}' (var={spec['var']}, "
                          f"level={spec.get('level')}, ftype={ftype})",
                          flush=True)
                    return None
                loaded[name] = data
                if lats is None:
                    lats, lons = la, lo
        try:
            result = fn(loaded, lats=lats, lons=lons)
        except Exception as e:
            import traceback
            print(f"[composite] {pid_dbg} F{fhr:03d}: compute '{fn_key}' "
                  f"failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return None
        if result is None:
            print(f"[composite] {pid_dbg} F{fhr:03d}: '{fn_key}' returned None",
                  flush=True)
            return None
        # Computed contour overlays: each entry names another COMPOSITES fn
        # evaluated over the SAME ingredient dict, drawn through the standard
        # theme-aware _draw_overlay. This is what lets one product stack
        # e.g. PWAT shading + sfc-convergence + jet isotachs + divergence.
        overlays = []
        for cspec in prod.get('contour_fns', []) or []:
            cfn = COMPOSITES.get(cspec.get('fn'))
            if cfn is None:
                print(f"[composite] {pid_dbg}: contour fn "
                      f"'{cspec.get('fn')}' not registered", flush=True)
                continue
            try:
                cdata = cfn(loaded, lats=lats, lons=lons)
            except Exception as e:
                print(f"[composite] {pid_dbg} contour '{cspec.get('fn')}': "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            if cdata is None:
                continue
            ov = dict(cspec)
            ov.update(data=cdata, lats=lats, lons=lons)
            overlays.append(ov)
        return self.pm.shaded(result, lats, lons, prod, region, run_dt, fhr,
                              overlays=overlays)

    def _member_mean(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """Mean of a single variable across all available ensemble members.

        Requires the per-member partial GRIB files to already be on disk
        (the API layer's _prefetch_records handles the async byte-range
        fetch). For each member, loads the variable and stacks; np.nanmean
        over the member axis is the rendered field. Members that fail to
        load are simply skipped — having 3 of 5 still produces a sensible
        ensemble mean.
        """
        pid_dbg = prod.get('name', '?')
        var = prod['var']
        n = prod.get('n_members', N_MEMBERS)
        member_product = prod.get('member_product', '2dfld')
        # Resolve the GRIB stepRange string ('A-B' form) for accumulated /
        # averaged forecasts. Same logic as enspost products.
        step = prod.get('step')
        if step is None and 'step_from_fhr' in prod:
            try:
                step = prod['step_from_fhr'](fhr)
            except Exception:
                step = None
        arrs = []
        lats = lons = None
        for mem in range(1, n + 1):
            try:
                f = self.proc.find_or_fetch_member(date_str, run, fhr, mem,
                                                   member_product, status_cb)
            except Exception as e:
                print(f"[member_mean] {pid_dbg} F{fhr:03d} m{mem:03d}: "
                      f"fetch failed: {type(e).__name__}: {e}", flush=True)
                continue
            data, la, lo = self.proc.load_var(
                f, var, level=prod.get('level'), step=step)
            if data is None:
                print(f"[member_mean] {pid_dbg} F{fhr:03d} m{mem:03d}: "
                      f"load_var returned None for {var}", flush=True)
                continue
            if lats is None:
                lats, lons = la, lo
            arrs.append(data)
        if not arrs:
            print(f"[member_mean] {pid_dbg} F{fhr:03d}: no members loaded",
                  flush=True)
            return None
        stack = np.stack(arrs, axis=0)
        result = np.nanmean(stack, axis=0)
        if 'convert' in prod:
            result = prod['convert'](result)
        return self.pm.shaded(result, lats, lons, prod, region, run_dt, fhr)

    def _qpf_sum(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """N-hr QPF ensemble mean derived by summing consecutive 6-hr means.

        Uses prod keys:
          sum_period (int) : hours per individual period (default 6)
          n_steps    (int) : number of periods to sum (default 4 → 24 h total)
        Fetches mean-file APCP for each contributing fhr and sums them.
        """
        period = prod.get('sum_period', 6)
        n_steps = prod.get('n_steps', 4)
        total_h = period * n_steps
        if fhr < total_h or fhr % period != 0:
            status_cb(f"{prod['name']}: fhr {fhr} not valid "
                      f"(need multiple of {period} and ≥ {total_h})")
            return None
        total = None
        lats = lons = None
        for i in range(n_steps):
            fetch_fhr = fhr - period * (n_steps - 1 - i)
            step = f"{fetch_fhr - period}-{fetch_fhr}"
            status_cb(f"Summing {total_h}-h QPF: F{fetch_fhr:03d} ({step} h)...")
            try:
                f = self.proc.find_or_fetch(date_str, run, fetch_fhr, 'mean',
                                            status_cb)
            except Exception as e:
                status_cb(f"{prod['name']}: cannot fetch F{fetch_fhr:03d}: {e}")
                return None
            data, la, lo = self.proc.load_var(f, 'tp_sfc', step=step)
            if data is None:
                status_cb(f"{prod['name']}: missing APCP at "
                          f"F{fetch_fhr:03d} step {step}")
                return None
            if total is None:
                total = data.copy()
                lats, lons = la, lo
            else:
                total += data
        if 'convert' in prod:
            total = prod['convert'](total)
        else:
            total = total / 25.4   # mm → inches
        overlays = []
        for ov in prod.get('overlay', []) or []:
            od = self._overlay_data(ov, date_str, run, fhr, status_cb)
            if od is not None:
                overlays.append(od)
        return self.pm.shaded(total, lats, lons, prod, region, run_dt, fhr,
                              overlays=overlays)

    def _member_prob(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """Member-derived neighborhood probability of var > thresh.

        Counts the fraction of members whose value at each grid point
        exceeds the threshold, then renders as a standard prob field. Use
        when the enspost prob file doesn't publish what you want (e.g.,
        gust probabilities are absent from the enspost prob file but each
        member file carries an instantaneous GUST record).
        """
        pid_dbg = prod.get('name', '?')
        var = prod['var']
        thresh = prod.get('thresh')
        if thresh is None:
            print(f"[member_prob] {pid_dbg}: no threshold defined", flush=True)
            return None
        n = prod.get('n_members', N_MEMBERS)
        member_product = prod.get('member_product', '2dfld')
        step = prod.get('step')
        if step is None and 'step_from_fhr' in prod:
            try:
                step = prod['step_from_fhr'](fhr)
            except Exception:
                step = None
        arrs = []
        lats = lons = None
        for mem in range(1, n + 1):
            try:
                f = self.proc.find_or_fetch_member(date_str, run, fhr, mem,
                                                   member_product, status_cb)
            except Exception as e:
                print(f"[member_prob] {pid_dbg} F{fhr:03d} m{mem:03d}: "
                      f"fetch failed: {type(e).__name__}: {e}", flush=True)
                continue
            data, la, lo = self.proc.load_var(
                f, var, level=prod.get('level'), step=step)
            if data is None:
                print(f"[member_prob] {pid_dbg} F{fhr:03d} m{mem:03d}: "
                      f"load_var returned None for {var}", flush=True)
                continue
            if lats is None:
                lats, lons = la, lo
            arrs.append(data)
        if not arrs:
            print(f"[member_prob] {pid_dbg} F{fhr:03d}: no members loaded",
                  flush=True)
            return None
        below = bool(prod.get('prob_below'))
        nbrhd_km = prod.get('nbrhd_km')
        if nbrhd_km:
            # Neighborhood probability: fraction of members with ANY
            # exceedance within radius `nbrhd_km`, then lightly smoothed.
            # Without this, 5 members give only {0,20,40,60,80,100}% — a
            # blocky field; the neighborhood max + Gaussian makes it read
            # like SPC's smoothed P(>0.01") product. REFS/HREF CONUS grid
            # spacing is ~3 km.
            from scipy.ndimage import maximum_filter, gaussian_filter
            rad_pts = max(1, int(round(float(nbrhd_km) / 3.0)))
            size = 2 * rad_pts + 1
            acc = np.zeros_like(arrs[0], dtype=float)
            for a in arrs:
                exc = (a < thresh) if below else (a > thresh)
                acc += maximum_filter(exc.astype(float), size=size)
            result = 100.0 * acc / float(len(arrs))
            result = gaussian_filter(result, sigma=rad_pts / 2.0)
            result = np.clip(result, 0.0, 100.0)
        else:
            stack = np.stack(arrs, axis=0)
            # Count exceedance per grid point. Use prob_below for <-threshold
            # variants (matches the convention used elsewhere).
            if below:
                count = (stack < thresh).sum(axis=0)
            else:
                count = (stack > thresh).sum(axis=0)
            result = 100.0 * count.astype(float) / float(len(arrs))
        return self.pm.shaded(result, lats, lons, prod, region, run_dt, fhr)

    def _storm_motion(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """Storm motion (USTM/VSTM, 0-6 km): ensemble-mean speed shaded with
        ensemble-mean vector arrows overlaid (decimated for legibility).

        Same fetch pattern as _member_mean. Internally uses pm.shaded for the
        speed field, then quivers the (u, v) mean vectors over the top.
        """
        pid_dbg = prod.get('name', '?')
        member_product = prod.get('member_product', '2dfld')
        n = prod.get('n_members', N_MEMBERS)
        us, vs = [], []
        lats = lons = None
        for mem in range(1, n + 1):
            try:
                f = self.proc.find_or_fetch_member(date_str, run, fhr, mem,
                                                   member_product, status_cb)
            except Exception as e:
                print(f"[storm_motion] {pid_dbg} F{fhr:03d} m{mem:03d}: "
                      f"fetch failed: {e}", flush=True)
                continue
            u, la, lo = self.proc.load_var(f, 'ustm_6km')
            v, _, _   = self.proc.load_var(f, 'vstm_6km')
            if u is None or v is None:
                print(f"[storm_motion] {pid_dbg} F{fhr:03d} m{mem:03d}: "
                      f"missing USTM or VSTM", flush=True)
                continue
            if lats is None:
                lats, lons = la, lo
            us.append(u); vs.append(v)
        if not us:
            print(f"[storm_motion] {pid_dbg} F{fhr:03d}: no members loaded",
                  flush=True)
            return None
        u_mean = np.nanmean(np.stack(us, axis=0), axis=0)
        v_mean = np.nanmean(np.stack(vs, axis=0), axis=0)
        spd_kt = np.sqrt(u_mean**2 + v_mean**2) * 1.94384
        return self.pm.storm_motion(spd_kt, u_mean, v_mean, lats, lons,
                                    prod, region, run_dt, fhr)

    def _heat_index(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """Derived heat index from 2-m T and Td (NWS Rothfusz regression)."""
        f = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        status_cb("Reading T2m / Td2m for heat index...")
        t_k, lats, lons = self.proc.load_var(f, 't_2m')
        td_k, _, _      = self.proc.load_var(f, 'd_2m')
        if t_k is None or td_k is None:
            status_cb("Missing T or Td for heat index"); return None
        # Convert to Fahrenheit and Celsius
        t_c  = t_k  - 273.15
        td_c = td_k - 273.15
        t_f  = t_c  * 9.0/5.0 + 32.0
        # RH from T and Td via Magnus formula
        es_t  = 6.1094 * np.exp(17.625 * t_c  / (243.04 + t_c))
        es_td = 6.1094 * np.exp(17.625 * td_c / (243.04 + td_c))
        rh = 100.0 * np.clip(es_td / es_t, 0.0, 1.0)
        # NWS Rothfusz regression -- valid above ~80 degF
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
        # Below 80 degF the Rothfusz formula is unreliable; fall back to T.
        hi = np.where(t_f < 80.0, t_f, hi)
        return self.pm.shaded(hi, lats, lons, prod, region, run_dt, fhr)

    def _combo_sfc(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """Shaded base field + MSLP isobars + 10-m wind barbs."""
        f = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        status_cb(f"Reading {prod['name']}...")
        base, lats, lons = self.proc.load_var(f, prod['var'])
        if base is None:
            status_cb(f"Missing base field for {prod['name']}"); return None
        if 'convert' in prod: base = prod['convert'](base)
        mslp, _, _ = self.proc.load_var(f, 'mslp')
        u, _, _    = self.proc.load_var(f, 'u_10m')
        v, _, _    = self.proc.load_var(f, 'v_10m')
        return self.pm.combo_sfc(base, mslp, u, v, lats, lons,
                                 prod, region, run_dt, fhr)

    def _mean_spread(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        """Mean + spread combo plot.  Either shaded-mean + spread-contours, or
        shaded-spread + mean-contours (when prod['mean_contour']=True)."""
        f_mean = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        f_sprd = self.proc.find_or_fetch(date_str, run, fhr, 'sprd', status_cb)
        lvl = prod.get('level')
        status_cb(f"Reading mean+spread {prod['name']}...")
        mean, lats, lons = self.proc.load_var(f_mean, prod['mean_var'], level=lvl)
        sprd, _, _       = self.proc.load_var(f_sprd, prod['spread_var'], level=lvl)
        if mean is None or sprd is None:
            status_cb(f"Missing mean or spread for {prod['name']} F{fhr:03d}")
            return None
        if prod.get('mean_convert'):   mean = prod['mean_convert'](mean)
        if prod.get('spread_convert'): sprd = prod['spread_convert'](sprd)
        return self.pm.mean_spread(mean, sprd, lats, lons, prod, region,
                                   run_dt, fhr)

    def _mslp(self, prod, date_str, run, fhr, region, run_dt, status_cb):
        # REFS publishes HGT at 250/500/700/850/925 mb in the mean file —
        # NO 1000 mb. Switched to 850-500 mb thickness (an SPC/HRRR-standard
        # alternative); was previously trying to load gh1000 and silently
        # producing MSLP-only renders that lied about the product label.
        f = self.proc.find_or_fetch(date_str, run, fhr, 'mean', status_cb)
        mslp, lats, lons = self.proc.load_var(f, 'mslp')
        if mslp is None:
            status_cb(f"MSLP missing F{fhr:03d}"); return None
        gh500, _, _ = self.proc.load_var(f, 'gh_lvl', level=500)
        gh850, _, _ = self.proc.load_var(f, 'gh_lvl', level=850)
        return self.pm.mslp_thickness(mslp, None, None, gh850, gh500,
                                      lats, lons, prod, region, run_dt, fhr)


# =============================================================================
#  HREFv3 data processor
# =============================================================================

class HREFDataProcessor(REFSDataProcessor):
    """Download + extract HREFv3 GRIB2 variables from NOMADS HTTPS.

    Drop-in replacement for REFSDataProcessor: same load_var() / _pygrib_load()
    / _cfgrib_load() logic, different find_or_fetch() that resolves filenames
    and URLs for HREF instead of REFS.

    HREF file layout on NOMADS:
      https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/v3.1/
         href.YYYYMMDD/ensprod/href.tHHz.conus.{ftype}.fXX.grib2
    """

    def find_or_fetch(self, date_str, run, fhr, ftype, status_cb=None):
        fname = HREF_FNAME_T.format(run=run, ftype=ftype, fhr=fhr)
        for p in [self.local_dir / fname,
                  self.local_dir / date_str / f"{run:02d}" / fname]:
            if p.exists():
                return p
        target = self.local_dir / date_str / f"{run:02d}" / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (f"{HREF_BASE_URL}/"
               f"{HREF_PREFIX_T.format(date=date_str)}"
               f"{fname}")
        if status_cb:
            status_cb(f"Downloading {fname}...")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get('Content-Length', 0))
                got = 0
                with open(target, 'wb') as f:
                    for chunk in r.iter_content(1 << 17):
                        f.write(chunk)
                        got += len(chunk)
                        if status_cb and total:
                            pct = 100.0 * got / total
                            status_cb(f"Downloading {fname} ({pct:.0f}%)")
        except Exception as e:
            if target.exists():
                target.unlink()
            raise RuntimeError(f"Could not fetch HREF {fname}: {e}")
        return target

    def find_or_fetch_member(self, *args, **kwargs):
        raise NotImplementedError(
            "HREFv3 does not publish individual ensemble member files"
        )


SPC_POST_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/v2.1"


class SPCPostDataProcessor(REFSDataProcessor):
    """Download + extract SPC post-processed HREF calibrated guidance from
    NOMADS HTTPS (severe, thunder, lightning-density).

    These are small (≤25 KB) single-field GRIB2 files, so we download the
    whole file rather than byte-range. The product's ``ftype`` encodes the
    NOMADS subdirectory and the filename token, e.g.
        'severe|href_cal_gefs_tor_{run:02d}.4hr'
        'thunder|hrefct_4hr'
        'ltgdensity|hrefld_4hr'
    where ``{run:02d}`` is substituted with the cycle hour.

    File layout:
      {BASE}/spc_post.YYYYMMDD/{subdir}/spc_post.tHHz.{token}.fNNN.grib2
    """

    def _spc_parts(self, run, ftype, fhr):
        subdir, token = ftype.split('|', 1)
        token = token.format(run=run)
        fname = f"spc_post.t{run:02d}z.{token}.f{fhr:03d}.grib2"
        return subdir, fname

    def find_or_fetch(self, date_str, run, fhr, ftype, status_cb=None):
        subdir, fname = self._spc_parts(run, ftype, fhr)
        target = self.local_dir / "spc" / date_str / f"{run:02d}" / fname
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{SPC_POST_BASE_URL}/spc_post.{date_str}/{subdir}/{fname}"
        if status_cb:
            status_cb(f"Downloading {fname}...")
        try:
            with requests.get(url, timeout=45) as r:
                r.raise_for_status()
                target.write_bytes(r.content)
        except Exception as e:
            if target.exists():
                target.unlink()
            raise RuntimeError(f"Could not fetch SPC {fname}: {e}")
        return target

    def load_var(self, filepath, varkey, level=None, step=None,
                 thresh=None, below=False):
        """Load an SPC calibrated field via the low-level eccodes API.

        We use eccodes directly (not pygrib, not cfgrib-by-name) because:
          • HF Spaces ships no pygrib (its wheels segfault vs libeccodes), so a
            pygrib-only path returns nothing there;
          • SPC uses NCEP-local parameter tables, so ``shortName`` may resolve
            to 'unknown' on a table-less eccodes, defeating a name match.
        eccodes reads values + latitude/longitude arrays for ANY grid without
        parameter tables. These files hold a single probability message
        (severe / thunder) or — for lightning density — four messages keyed by
        upper-limit flash count (≥25/50/100/200), matched via the standard
        ``scaledValueOfUpperLimit`` template key.
        Returns (data2d, lats, lons) or (None, None, None)."""
        try:
            from eccodes import (codes_grib_new_from_file, codes_get,
                                 codes_get_values, codes_get_array,
                                 codes_release)
        except Exception as e:
            print(f"[spc eccodes] import failed: {e}", flush=True)
            return None, None, None
        chosen = None
        try:
            with open(filepath, 'rb') as fh:
                while True:
                    gid = codes_grib_new_from_file(fh)
                    if gid is None:
                        break
                    try:
                        ok = True
                        if thresh is not None:
                            try:
                                sv = codes_get(gid, 'scaledValueOfUpperLimit')
                                sf = codes_get(gid, 'scaleFactorOfUpperLimit')
                                v = sv / (10 ** sf) if sf else float(sv)
                                ok = abs(v - thresh) <= max(1e-3, abs(thresh) * 0.01)
                            except Exception:
                                ok = False
                        if ok and chosen is None:
                            ni = int(codes_get(gid, 'Ni'))
                            nj = int(codes_get(gid, 'Nj'))
                            vals = np.array(codes_get_values(gid), dtype=float)
                            lats = np.array(codes_get_array(gid, 'latitudes'), dtype=float)
                            lons = np.array(codes_get_array(gid, 'longitudes'), dtype=float)
                            chosen = (vals.reshape(nj, ni),
                                      lats.reshape(nj, ni),
                                      lons.reshape(nj, ni))
                    finally:
                        codes_release(gid)
                    if chosen is not None:
                        break
            if chosen is None:
                print(f"[spc eccodes] no message matched "
                      f"(var={varkey} thresh={thresh}) in {filepath}", flush=True)
                return None, None, None
            vals, lats, lons = chosen
            lons = np.where(lons > 180, lons - 360, lons)
            vals = np.nan_to_num(vals, nan=0.0)
            return vals, lats, lons
        except Exception as e:
            print(f"[spc eccodes] read failed: {type(e).__name__}: {e}", flush=True)
            return None, None, None

    def find_or_fetch_member(self, *args, **kwargs):
        raise NotImplementedError("SPC post products have no member files")
