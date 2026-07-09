"""Register extra REFS products into refs_core.PRODUCTS at startup.

Keeps refs_core.py unchanged. Each product re-uses refs_core's existing
overlay system; the byte-range fetch in app/needed_records.py covers the
new products automatically because they're standard shaded+overlay specs.

Composites added here ("creative mash-ups"):
  • REFC PMM + UH-prob swath contours        — severe-mode signature
  • REFC PMM + Lightning-prob contours       — convective-active areas
  • MLCAPE mean + 0-6 km shear contours      — supercell environment
  • SBCAPE mean + 0-3 km SRH contours        — tornado-favorable
  • PWAT mean + 850 mb wind contours         — heavy-rain LLJ setup

Plus Winter-tab fillers (REFS publishes ASNOW at multiple thresholds):
  • Prob 6-h snow > 1" / 3" / 6" / 12"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_PAREN_RE = re.compile(r"\s*\([^)]*\)")


def _shorten_title(t: str) -> str:
    """Strip parentheticals + everything after an em/double-dash, cap length.

    refs_core renders the figure title at a fixed width; long subtitles
    spill out and overlap the run/valid stamps in the top-right.
    """
    if not t:
        return t
    # Drop "(...)" segments
    t = _PAREN_RE.sub("", t)
    # Cut off at " — " / " -- "
    t = re.split(r"\s+[—\-]{1,2}\s+", t, maxsplit=1)[0]
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # Hard cap so even very long mean/shear/etc lines fit
    if len(t) > 90:
        t = t[:87].rstrip() + "…"
    return t


def register() -> None:
    import refs_core as core   # local import; refs_core may take a moment
    P = core.PRODUCTS
    # Bind _step_for_acc as a local once so every lambda below resolves
    # it via closure when invoked. Several existing products
    # (snow_6h_prob_*, snow_6h_pmmn, the new heavy-rain augmentation)
    # build step_from_fhr lambdas that referenced this name without
    # ever binding it — meaning the lambdas hit NameError when
    # refs_core called them. Caught only because the products are
    # rarely-used / out-of-season.
    _step_for_acc = core._step_for_acc  # noqa: F841 — used by lambdas below

    # ------------------------------------------------------------------
    #  CONVECTION / SEVERE composites
    # ------------------------------------------------------------------

    # 1) Severe-mode signature: REFC PMM shaded + UH 2-5 km prob >75 contours
    P["refc_uh_swath"] = dict(
        cat="Reflectivity (PMM Series)",
        name="REFC PMM + UH 2-5 km swath contours",
        ftype="pmmn", var="refc", cmap="refc", units="dBZ",
        overlay=[dict(
            ftype="prob", var="mxuphl_25", thresh=75,
            style="contour",
            levels=[10, 30, 50, 70, 90], smooth=2.0,
            colors="#222222",
            linewidths=[0.8, 1.2, 1.6, 2.0, 2.4],
        )],
        spc_title=("Composite reflectivity (dBZ; shaded, PMM), "
                   "neighborhood prob 2-5 km UH > 75 m^2/s^2 (contours) "
                   "— rotating-storm signature"),
    )

    # 2) REFC PMM + Lightning prob contours
    P["refc_ltng_combo"] = dict(
        cat="Reflectivity (PMM Series)",
        name="REFC PMM + Lightning prob contours",
        ftype="pmmn", var="refc", cmap="refc", units="dBZ",
        overlay=[dict(
            ftype="prob", var="ltng", thresh=0.08,
            style="contour",
            levels=[10, 30, 50, 70, 90], smooth=2.0,
            colors="#7a1ddc",
            linewidths=[0.8, 1.2, 1.6, 2.0, 2.4],
        )],
        spc_title=("Composite reflectivity (dBZ; shaded, PMM) with "
                   "lightning probability > 0.08 fl/km^2/min (contours) "
                   "— electrically-active convection"),
    )

    # 3) Supercell environment: MLCAPE shaded + 0-6 km bulk shear contours (kt)
    #    Shear is stored in m/s — convert thresholds: 30/40/50 kt = 15.4/20.6/25.7 m/s
    P["mlcape_shear_env"] = dict(
        cat="Thermodynamics",
        name="ML CAPE + 0-6 km Shear contours",
        ftype="mean", var="cape_ml", cmap="cape", units="J/kg",
        overlay=[dict(
            ftype="mean", var="vwsh_06km",
            style="contour",
            levels=[15.4, 20.6, 25.7],   # m/s ≈ 30 / 40 / 50 kt
            smooth=1.2,
            colors="#0a0a0a",
            linewidths=[1.0, 1.5, 2.2],
        )],
        spc_title=("Mixed-layer (90-0 mb) CAPE (J/kg; shaded) with 0-6 km "
                   "bulk shear contours at 30 / 40 / 50 kt — supercell "
                   "environment discriminator"),
    )

    # 4) Tornado-favorable: SBCAPE shaded + 0-3 km SRH contours
    P["sbcape_srh_env"] = dict(
        cat="Thermodynamics",
        name="SBCAPE + 0-3 km SRH contours",
        ftype="mean", var="cape_sfc", cmap="cape", units="J/kg",
        overlay=[dict(
            ftype="mean", var="srh_3km",
            style="contour",
            levels=[100, 200, 400],     # m^2/s^2
            smooth=1.2,
            colors="#8a1d8a",
            linewidths=[1.0, 1.6, 2.4],
        )],
        spc_title=("Surface CAPE (J/kg; shaded) with 0-3 km SRH contours at "
                   "100 / 200 / 400 m^2/s^2 — supercell/tornado-favorable areas"),
    )

    # 6) Elevated / nocturnal supercell env: MUCAPE shaded + 0-6 km Shear
    P["mucape_shear_env"] = dict(
        cat="Thermodynamics",
        name="MU CAPE + 0-6 km Shear contours",
        ftype="mean", var="cape_mu", cmap="cape", units="J/kg",
        overlay=[dict(
            ftype="mean", var="vwsh_06km",
            style="contour",
            levels=[15.4, 20.6, 25.7],   # m/s ≈ 30/40/50 kt
            smooth=1.2,
            colors="#0a0a0a",
            linewidths=[1.0, 1.5, 2.2],
        )],
        spc_title=("Most-unstable (180-0 mb) CAPE (J/kg; shaded) with 0-6 km "
                   "bulk shear at 30 / 40 / 50 kt — elevated/nocturnal "
                   "supercell environment"),
    )

    # 7) Dual-overlay severe composite: SBCAPE shaded with BOTH SRH and Shear
    #    Identifies areas favorable for both supercells AND tornadoes.
    P["sbcape_combo_severe"] = dict(
        cat="Thermodynamics",
        name="SBCAPE + 0-3 km SRH + 0-6 km Shear",
        ftype="mean", var="cape_sfc", cmap="cape", units="J/kg",
        overlay=[
            dict(
                ftype="mean", var="srh_3km",
                style="contour",
                levels=[150, 300],
                smooth=1.2,
                colors="#8a1d8a",          # purple = rotation
                linewidths=[1.4, 2.2],
            ),
            dict(
                ftype="mean", var="vwsh_06km",
                style="contour",
                levels=[20.6, 25.7],       # 40 / 50 kt
                smooth=1.2,
                colors="#0a0a0a",          # black = shear
                linewidths=[1.4, 2.2],
            ),
        ],
        spc_title=("Surface CAPE (J/kg; shaded) with 0-3 km SRH (purple, "
                   "150 / 300 m^2/s^2) AND 0-6 km bulk shear (black, "
                   "40 / 50 kt) — supercell + tornado-favorable areas"),
    )

    # 8) Heavy-rain potential where storms are active: PWAT + REFC contours
    P["pwat_refc_combo"] = dict(
        cat="Synoptic / Moisture",
        name="PWAT + REFC PMM contours",
        ftype="mean", var="pwat", cmap="pwat_in", units="in",
        convert=lambda x: x/25.4,
        overlay=[dict(
            ftype="pmmn", var="refc",
            style="contour",
            levels=[20, 35, 50],            # dBZ
            smooth=1.2,
            colors="#a0007a",
            linewidths=[0.9, 1.4, 2.0],
        )],
        spc_title=("Precipitable water (in; shaded, ens mean) with composite "
                   "reflectivity (PMM) contours at 20 / 35 / 50 dBZ — heavy-"
                   "rain risk where convection is active in a moist airmass"),
    )

    # 9) Convective vigor: REFC PMM + Prob Max Updraft > 20 m/s contours
    P["refc_maxuvv_combo"] = dict(
        cat="Reflectivity (PMM Series)",
        name="REFC PMM + Prob Max Updraft > 20 m/s contours",
        ftype="pmmn", var="refc", cmap="refc", units="dBZ",
        overlay=[dict(
            ftype="prob", var="maxuvv_lyr", thresh=20,
            style="contour",
            levels=[10, 30, 50, 70, 90], smooth=2.0,
            colors="#c81e1e",                # red = vigorous updrafts
            linewidths=[0.8, 1.2, 1.6, 2.0, 2.4],
        )],
        spc_title=("Composite reflectivity (dBZ; shaded, PMM) with "
                   "neighborhood prob of max vertical velocity > 20 m/s "
                   "(red contours) — vigorous-updraft convection"),
    )

    # 10) Storm depth signature: REFC PMM + Echo Top contours
    P["refc_echotop_combo"] = dict(
        cat="Reflectivity (PMM Series)",
        name="REFC PMM + Echo Top contours",
        ftype="pmmn", var="refc", cmap="refc", units="dBZ",
        overlay=[dict(
            ftype="pmmn", var="retop",
            style="contour",
            # Echo top stored in m → 30/40/50 kft = 9144/12192/15240 m
            levels=[9144, 12192, 15240],
            smooth=1.2,
            colors="#222222",
            linewidths=[1.0, 1.5, 2.2],
        )],
        spc_title=("Composite reflectivity (dBZ; shaded, PMM) with echo-top "
                   "contours at 30 / 40 / 50 kft — storm depth signature"),
    )

    # ------------------------------------------------------------------
    #  Earlier QPF / Flash-Flood additions removed (rollback notes)
    #
    #  An inventory of the real REFS bucket (run 12z 2026-05-25, prob
    #  and ffri files at fhrs 1/3/6/9/12/18/24/30/36/48) confirmed:
    #
    #    • PROB file DOES publish 3/6/12/24-h QPF prob records at the
    #      thresholds 0.5 / 1 / 2 / 3 in (12.7/25.4/50.8/76.2 mm), plus
    #      5 in (127) and 8 in (203.2) at the longer accumulations.
    #      They exist only at fhrs aligned to the accumulation length
    #      (e.g. 6-h prob at f06, f09, ..., f48). That's the right
    #      backing for the next batch of QPF-prob products.
    #
    #    • FFRI file does NOT publish any 3-h APCP records — only 6-h
    #      (steps 0-6, 3-9, 6-12, 12-18, ...). It also uses different
    #      threshold values from PROB: 1, 2, 5, 10, 25, 50, 100 mm (no
    #      75-mm record exists). That killed:
    #          ffri_qpf_3h_2in   (no 3-h APCP)
    #          ffri_qpf_3h_3in   (no 3-h APCP and no 75-mm thresh)
    #          ffri_qpf_6h_3in   (no 75-mm thresh)
    #
    #    • FFRI PPFFG records exist only at 1-h / 3-h / 6-h accumulation
    #      lengths. There is no 12-h or 24-h PPFFG. That killed:
    #          ppffg_12h         (no record)
    #          ppffg_24h         (no record)
    #
    #  All five products that were "kept pending verification" have been
    #  removed. New QPF-prob products with verified backing records will
    #  be added in a follow-up pass once the cfgrib disambiguation fix is
    #  validated end-to-end on the deployed Space.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  HREF-style heavy-rain augmentation of the 3-h QPF PMM product
    #
    #  Add a second probability-contour overlay (prob >3 in, red) on top
    #  of the existing >0.5 in (brown) overlay so the product matches
    #  the SPC HREF "NP[QPF>1] + NP[QPF>3]" visual the user referenced.
    #  Safe now that the cfgrib loader's threshold-tightening pass and
    #  thresh-aware DataSet cache key are in place — two overlays with
    #  the same (var, step) but different thresh are now correctly
    #  resolved to distinct records, where previously cfgrib silently
    #  served the same record to both.
    # ------------------------------------------------------------------

    # The 3-h QPF PMM + >1"/>3" prob-contour overlays now live directly in
    # refs_core.PRODUCTS['qpf_3h_pmmn_series'] (both thresholds are published
    # in the prob file at the 3-h window — verified against the live .idx).

    # ------------------------------------------------------------------
    #  WINTER probability sweep (REFS publishes ASNOW prob over multiple
    #  6-h windows at 1"/3"/6"/12" thresholds)
    # ------------------------------------------------------------------

    # refs_core uses meters internally and matches against scaled GRIB
    # values. Threshold strings observed in the idx:
    #   0.025  (1")    0.076  (3")    0.152  (6")    0.304  (12")

    # (All winter products removed 2026-06-11 per user: the snow_6h_prob_*
    #  sweep [1/3/6/12 in] and snow_6h_pmmn. Revert this commit to restore
    #  for the cold season.)

    # ------------------------------------------------------------------
    #  Phase G — Mean-derived composite parameters (SCP, STP, EHI, lapse
    #  rate, SHIP, storm-top divergence). All driven by the new
    #  recipe='composite' in refs_core.PlotJob, with the computation
    #  registered into core.COMPOSITES below.
    # ------------------------------------------------------------------
    import numpy as np

    def _scp(d, **_):
        # Effective shear normalization in SPC's SCP definition uses EBWD;
        # we approximate with the mean-file 0-6 km bulk shear (m/s). Ramp:
        # ≤10 m/s → 0; ≥20 m/s → 1; capped at 1.5.
        mucape = np.asarray(d['cape_mu'])
        srh3 = np.asarray(d['srh_3km'])
        shr = np.asarray(d['vwsh_06km'])
        shr_t = np.clip((shr - 10.0) / 10.0, 0.0, 1.5)
        return (mucape / 1000.0) * (srh3 / 50.0) * shr_t

    def _stp_fixed(d, **_):
        # Thompson et al. fixed-layer STP. SRH normally is 0-1 km; REFS mean
        # only carries 0-3 km, which we use here (so values run a bit
        # higher than SPC's effective-layer STP — interpret accordingly).
        mlcape = np.asarray(d['cape_ml'])
        mlcin  = np.asarray(d['cin_ml'])     # negative when capped
        srh3   = np.asarray(d['srh_3km'])
        shr    = np.asarray(d['vwsh_06km'])
        cape_t = mlcape / 1500.0
        srh_t  = srh3 / 150.0
        # Shear: 0 below 12.5 m/s, 1 at 20+, cap 1.5
        shr_t  = np.clip((shr - 12.5) / 7.5, 0.0, 1.5)
        # CIN: 1 if MLCIN ≥ -50, 0 if ≤ -200, linear in between
        cin_t  = np.clip((mlcin + 200.0) / 150.0, 0.0, 1.0)
        return cape_t * srh_t * shr_t * cin_t

    def _ehi_03(d, **_):
        sbcape = np.asarray(d['cape_sfc'])
        srh3 = np.asarray(d['srh_3km'])
        return (sbcape * srh3) / 160000.0

    def _lapse_75(d, **_):
        # 700-500 mb lapse rate in K/km. dT positive (cooler aloft) ⇒ steeper.
        t700 = np.asarray(d['t_700'])
        t500 = np.asarray(d['t_500'])
        gh700 = np.asarray(d['gh_700'])
        gh500 = np.asarray(d['gh_500'])
        dz_km = np.maximum((gh500 - gh700) / 1000.0, 0.1)
        return (t700 - t500) / dz_km

    def _ship(d, **_):
        # WPC SHIP formulation. Mixing ratio is approximated from 2-m
        # dewpoint + MSLP (good enough where boundary-layer parcels lift).
        mucape = np.asarray(d['cape_mu'])
        t500_c = np.asarray(d['t_500']) - 273.15
        t700_k = np.asarray(d['t_700'])
        t500_k = np.asarray(d['t_500'])
        gh700 = np.asarray(d['gh_700'])
        gh500 = np.asarray(d['gh_500'])
        dz_km = np.maximum((gh500 - gh700) / 1000.0, 0.1)
        lr75 = (t700_k - t500_k) / dz_km                # K/km
        shr = np.clip(np.asarray(d['vwsh_06km']), 7.0, 27.0)
        td_c = np.asarray(d['d_2m']) - 273.15
        e = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))   # hPa
        p_hpa = np.asarray(d['mslp']) / 100.0
        w = 0.622 * e / np.maximum(p_hpa - e, 1.0) * 1000.0  # g/kg
        ship = -1.0 * (mucape * w * lr75 * t500_c * shr) / 42_000_000.0
        # Mask physically-impossible negatives (cold mid-level + cool air aloft
        # can produce <0 from this scaling; SHIP is defined to be ≥ 0).
        return np.where(ship > 0, ship, 0.0)

    def _div_1e5(u, v, lats, lons):
        """Horizontal divergence of (u, v) in 10^-5 s^-1 — spherical-earth
        finite differences on the lat/lon grid."""
        u = np.asarray(u); v = np.asarray(v)
        if lats is None or lons is None:
            return None
        if lats.ndim == 1:
            lat_g, lon_g = np.meshgrid(lats, lons, indexing='ij')
        else:
            lat_g, lon_g = lats, lons
        R = 6_371_000.0
        deg2rad = np.pi / 180.0
        cos_lat = np.cos(lat_g * deg2rad)
        du_dj = np.gradient(u, axis=1)
        dv_di = np.gradient(v, axis=0)
        dlon = np.gradient(lon_g, axis=1) * deg2rad
        dlat = np.gradient(lat_g, axis=0) * deg2rad
        du_dx = du_dj / (R * cos_lat * dlon + 1e-12)
        dv_dy = dv_di / (R * dlat + 1e-12)
        return (du_dx + dv_dy) * 1e5

    def _stdiv_250(d, lats=None, lons=None, **_):
        # Storm-top divergence proxy: horizontal divergence of 250-mb wind.
        return _div_1e5(d['u_250'], d['v_250'], lats, lons)

    # ---- Moisture & Lift combo ingredients ------------------------------
    def _pwat_in(d, **_):
        return np.asarray(d['pwat']) / 25.4          # mm → inches

    def _conv_10m(d, lats=None, lons=None, **_):
        # Surface (10 m) CONVERGENCE = -divergence; positive where air
        # piles up along boundaries / outflow collisions.
        div = _div_1e5(d['u_10m'], d['v_10m'], lats, lons)
        return None if div is None else -div

    def _div_250(d, lats=None, lons=None, **_):
        return _div_1e5(d['u_250'], d['v_250'], lats, lons)

    def _isotach_250(d, **_):
        return np.hypot(np.asarray(d['u_250']),
                        np.asarray(d['v_250'])) * 1.94384   # m/s → kt

    # ---- 850-mb moisture transport ingredients ---------------------------
    def _mflux_850(d, **_):
        # Moisture flux magnitude = mixing ratio (from 850 Td) × wind speed,
        # in g/kg·m/s. The flash-flood / PRE "low-level feed" field.
        td_c = np.asarray(d['dpt_850']) - 273.15
        e = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))   # hPa
        w = 622.0 * e / np.maximum(850.0 - e, 1.0)          # g/kg
        return w * np.hypot(np.asarray(d['u_850']),
                            np.asarray(d['v_850']))

    def _isotach_850(d, **_):
        return np.hypot(np.asarray(d['u_850']),
                        np.asarray(d['v_850'])) * 1.94384   # kt

    # ---- MLCAPE inflow-combo ingredients ----------------------------------
    def _cape_ml_field(d, **_):
        return np.asarray(d['cape_ml'])

    def _shear06_kt(d, **_):
        return np.asarray(d['vwsh_06km']) * 1.94384

    def _cin_ml_field(d, **_):
        return np.asarray(d['cin_ml'])

    core.COMPOSITES.update({
        'scp':         _scp,
        'stp_fixed':   _stp_fixed,
        'ehi_03':      _ehi_03,
        'lapse_75':    _lapse_75,
        'ship':        _ship,
        'stdiv_250':   _stdiv_250,
        'pwat_in':     _pwat_in,
        'conv_10m':    _conv_10m,
        'div_250':     _div_250,
        'isotach_250': _isotach_250,
        'mflux_850':   _mflux_850,
        'isotach_850': _isotach_850,
        'cape_ml_field': _cape_ml_field,
        'shear06_kt':  _shear06_kt,
        'cin_ml_field': _cin_ml_field,
    })

    # Moisture-flux colormap: green (modest feed) → blue → purple (extreme
    # LLJ transport). 50 g/kg·m/s is "worth noticing", 300+ is PRE/atmos-
    # river territory.
    core._CMAPS['mflux850'] = lambda: core._cmap(
        ['#e8f5e2', '#b9e3b0', '#7ac87a', '#34a853',
         '#2b8cbe', '#1f5fa8', '#6a3d9a', '#3f0d66'],
        [50, 100, 150, 200, 250, 300, 400, 500, 650], 'mflux850')

    # The "moisture & lift on one map" chart: where is the deep moisture,
    # what's converging on it at the surface, and is the upper jet venting
    # it. Reads like a hand analysis: PWAT shading, solid magenta surface-
    # convergence contours, bold blue 250-mb isotachs (jet axis), dashed
    # orange 250-mb divergence (upper-level evacuation).
    P['moisture_lift_combo'] = dict(
        cat='Synoptic / Moisture',
        name='PWAT + Sfc Convergence + 250 mb Jet/Div',
        recipe='composite', composite_fn='pwat_in',
        cmap='pwat_in', units='in',
        ingredients={
            'pwat':  dict(ftype='mean', var='pwat'),
            'u_10m': dict(ftype='mean', var='u_10m'),
            'v_10m': dict(ftype='mean', var='v_10m'),
            'u_250': dict(ftype='mean', var='u_lvl', level=250),
            'v_250': dict(ftype='mean', var='v_lvl', level=250),
        },
        contour_fns=[
            # 10-m convergence: VERY heavy smoothing — native-grid divergence
            # is salt-and-pepper at 3 km, and sigma 6 still left confetti of
            # micro-contours. Sigma 12 (~36 km, SPC-mesoanalysis scale) +
            # a higher floor keeps only coherent boundaries.
            dict(fn='conv_10m', levels=[15, 30, 60], smooth=12.0,
                 colors=dict(light='#b0006a', dark='#ff5fc0'),
                 linewidths=[1.1, 1.5, 1.9]),
            dict(fn='div_250', levels=[6, 12, 24], smooth=12.0,
                 colors=dict(light='#c75300', dark='#ffb056'),
                 linestyles='dashed', linewidths=[1.0, 1.3, 1.6]),
            # Jet isotachs from 60 kt so the summertime subtropical jet
            # still draws (75-kt start left June maps empty).
            dict(fn='isotach_250', levels=[60, 90, 120, 150], smooth=4.0,
                 colors=dict(light='#005a9e', dark='#6fc4ff'),
                 linewidths=[1.6, 2.0, 2.4, 2.8]),
        ],
        spc_title=('PWAT (in, shaded) · 10-m convergence (magenta, 10⁻⁵ s⁻¹) '
                   '· 250-mb isotachs (blue, kt) · 250-mb divergence '
                   '(dashed orange, 10⁻⁵ s⁻¹)'),
    )

    # Low-level moisture feed: 850-mb moisture flux shaded, PWAT contours
    # (the column reservoir), 850-mb LLJ isotachs. Companion chart to
    # moisture_lift_combo — that one answers "where will it focus", this
    # one answers "how hard is the moisture being pumped in".
    P['mtransport_850'] = dict(
        cat='Synoptic / Moisture',
        name='850 mb Moisture Transport + PWAT + LLJ',
        recipe='composite', composite_fn='mflux_850',
        cmap='mflux850', units='g/kg·m/s',
        ingredients={
            'u_850':   dict(ftype='mean', var='u_lvl',   level=850),
            'v_850':   dict(ftype='mean', var='v_lvl',   level=850),
            'dpt_850': dict(ftype='mean', var='dpt_lvl', level=850),
            'pwat':    dict(ftype='mean', var='pwat'),
        },
        contour_fns=[
            dict(fn='pwat_in', levels=[1.0, 1.5, 2.0], smooth=4.0,
                 colors=dict(light='#1c5e20', dark='#7fe08a'),
                 linewidths=[1.1, 1.5, 1.9]),
            # σ=8: 850 winds are convectively contaminated at 3 km — light
            # smoothing left 30-kt speckles all over active convection.
            dict(fn='isotach_850', levels=[30, 50, 70], smooth=8.0,
                 colors=dict(light='#8a0000', dark='#ff7a6e'),
                 linestyles='dashed', linewidths=[1.2, 1.6, 2.0]),
        ],
        spc_title=('850-mb moisture flux (g/kg·m/s, shaded) · PWAT contours '
                   '(green, in) · 850-mb LLJ isotachs (dashed red, kt)'),
    )

    # "Where can storms initiate and rotate" on one map: MLCAPE shading,
    # surface convergence (initiation focus), 0-6 km shear isotachs
    # (organization), MLCIN (the cap) as dashed blue contours.
    P['mlcape_inflow_combo'] = dict(
        cat='Thermodynamics',
        name='MLCAPE + Sfc Convergence + 0-6 km Shear + CIN',
        recipe='composite', composite_fn='cape_ml_field',
        cmap='cape', units='J/kg',
        ingredients={
            'cape_ml':   dict(ftype='mean', var='cape_ml'),
            'cin_ml':    dict(ftype='mean', var='cin_ml'),
            'vwsh_06km': dict(ftype='mean', var='vwsh_06km'),
            'u_10m':     dict(ftype='mean', var='u_10m'),
            'v_10m':     dict(ftype='mean', var='v_10m'),
        },
        contour_fns=[
            dict(fn='conv_10m', levels=[15, 30, 60], smooth=12.0,
                 colors=dict(light='#b0006a', dark='#ff5fc0'),
                 linewidths=[1.1, 1.5, 1.9]),
            dict(fn='shear06_kt', levels=[30, 40, 50], smooth=4.0,
                 colors=dict(light='#005a9e', dark='#6fc4ff'),
                 linewidths=[1.3, 1.7, 2.1]),
            dict(fn='cin_ml_field', levels=[-100, -50], smooth=4.0,
                 colors=dict(light='#3949ab', dark='#9aa8ff'),
                 linestyles='dashed', linewidths=[1.4, 1.0]),
        ],
        spc_title=('MLCAPE (J/kg, shaded) · 10-m convergence (magenta) · '
                   '0-6 km shear (blue, kt) · MLCIN -50/-100 (dashed) — '
                   'initiation + organization on one map'),
    )

    P['scp_mean'] = dict(
        cat='Severe Probabilities',
        name='Supercell Composite Parameter (SCP)',
        recipe='composite', composite_fn='scp',
        cmap='composite', units='',
        ingredients={
            'cape_mu':   dict(ftype='mean', var='cape_mu'),
            'srh_3km':   dict(ftype='mean', var='srh_3km'),
            'vwsh_06km': dict(ftype='mean', var='vwsh_06km'),
        },
        spc_title='Supercell composite parameter from MUCAPE, 0-3 km SRH and 0-6 km shear',
    )

    P['stp_fixed_mean'] = dict(
        cat='Severe Probabilities',
        name='Significant Tornado Parameter (STP)',
        recipe='composite', composite_fn='stp_fixed',
        cmap='composite', units='',
        ingredients={
            'cape_ml':   dict(ftype='mean', var='cape_ml'),
            'cin_ml':    dict(ftype='mean', var='cin_ml'),
            'srh_3km':   dict(ftype='mean', var='srh_3km'),
            'vwsh_06km': dict(ftype='mean', var='vwsh_06km'),
        },
        spc_title='Fixed-layer significant tornado parameter from MLCAPE, MLCIN, 0-3 km SRH and 0-6 km shear',
    )

    P['ehi_03_mean'] = dict(
        cat='Severe Probabilities',
        name='0-3 km Energy Helicity Index (EHI)',
        recipe='composite', composite_fn='ehi_03',
        cmap='composite', units='',
        ingredients={
            'cape_sfc': dict(ftype='mean', var='cape_sfc'),
            'srh_3km':  dict(ftype='mean', var='srh_3km'),
        },
        spc_title='0-3 km energy helicity index from SBCAPE and 0-3 km SRH',
    )

    P['lapse_75_mean'] = dict(
        cat='Thermodynamics',
        name='700-500 mb Lapse Rate',
        recipe='composite', composite_fn='lapse_75',
        cmap='lapse', units='K/km',
        ingredients={
            't_700':  dict(ftype='mean', var='t_lvl',  level=700),
            't_500':  dict(ftype='mean', var='t_lvl',  level=500),
            'gh_700': dict(ftype='mean', var='gh_lvl', level=700),
            'gh_500': dict(ftype='mean', var='gh_lvl', level=500),
        },
        spc_title=('700-500 mb lapse rate (K/km, ens mean) — values > 7 '
                   'indicate steep mid-level lapse rates / hail potential'),
    )

    P['ship_mean'] = dict(
        cat='Severe Probabilities',
        name='Significant Hail Parameter (SHIP)',
        recipe='composite', composite_fn='ship',
        cmap='composite', units='',
        ingredients={
            'cape_mu':   dict(ftype='mean', var='cape_mu'),
            't_700':     dict(ftype='mean', var='t_lvl',  level=700),
            't_500':     dict(ftype='mean', var='t_lvl',  level=500),
            'gh_700':    dict(ftype='mean', var='gh_lvl', level=700),
            'gh_500':    dict(ftype='mean', var='gh_lvl', level=500),
            'vwsh_06km': dict(ftype='mean', var='vwsh_06km'),
            'd_2m':      dict(ftype='mean', var='d_2m'),
            'mslp':      dict(ftype='mean', var='mslp'),
        },
        spc_title='Significant hail parameter from MUCAPE, 700-500 mb lapse rate, 500 mb temp and shear',
    )

    P['stdiv_250_mean'] = dict(
        cat='Kinematics',
        name='250 mb Divergence',
        recipe='composite', composite_fn='stdiv_250',
        cmap='div', units='10⁻⁵ s⁻¹',
        # Native-grid divergence is dominated by 1-2 gridpoint noise; a
        # gentle Gaussian recovers the coherent synoptic / mesoscale signal
        # without losing storm-top features. mask_below_abs hides the noisy
        # near-zero band so basemap shows through.
        smooth_sigma=3.0,
        mask_below_abs=1.0,
        ingredients={
            'u_250': dict(ftype='mean', var='u_lvl', level=250),
            'v_250': dict(ftype='mean', var='v_lvl', level=250),
        },
        spc_title=('250 mb horizontal divergence (10⁻⁵ s⁻¹; smoothed) — '
                   'upper-level divergent flow / storm-top exhaust proxy'),
    )

    # ------------------------------------------------------------------
    #  Phase H — Member-derived 2dfld means + storm-motion vectors.
    #  These pull from the per-member ensemble GRIBs (rrfs_a/rrfsens/.../mNNN)
    #  via the new ensure_member_partial_cached byte-range fetcher. Each one
    #  averages over the 5 publicly-available members.
    # ------------------------------------------------------------------
    P['vil_member_mean'] = dict(
        cat='Storm Attributes',
        name='VIL mean',
        recipe='member_mean', member_product='2dfld', n_members=5,
        var='vil', cmap='vil', units='kg/m²',
        spc_title=('Vertically integrated liquid (kg/m², ens mean of 5 '
                   'members) — hail / severe-storm signature, > 30 = '
                   'significant'),
    )

    P['gust_member_mean'] = dict(
        cat='Storm Attributes',
        name='Surface Wind Gust mean',
        recipe='member_mean', member_product='2dfld', n_members=5,
        var='gust', cmap='gust', units='kt',
        convert=lambda x: x * 1.94384,    # m/s → kt
        spc_title=('Surface wind gust (kt, ens mean of 5 members) — '
                   'instantaneous gust forecast across the ensemble'),
    )

    P['dcape_member_mean'] = dict(
        cat='Thermodynamics',
        name='DCAPE mean',
        recipe='member_mean', member_product='2dfld', n_members=5,
        var='dcape', cmap='dcape', units='J/kg',
        spc_title=('Downdraft CAPE (400-0 mb, J/kg, ens mean of 5 members) '
                   '— wet-microburst / damaging-wind potential'),
    )

    P['mconv_member_mean'] = dict(
        cat='Synoptic / Moisture',
        name='Moisture Convergence mean',
        recipe='member_mean', member_product='2dfld', n_members=5,
        var='mconv', cmap='mconv', units='10⁻⁵ s⁻¹',
        convert=lambda x: x * 1e5,        # scale to plot units
        # mconv at native grid is extremely noisy; the strong sigma blends
        # member-to-member texture down to the storm-scale boundaries that
        # are actually meteorologically meaningful. The widened cmap bins
        # (±200) prevent end-color saturation; mask_below_abs hides the
        # incoherent near-zero band.
        smooth_sigma=4.0,
        mask_below_abs=5.0,
        spc_title=('Column-integrated moisture convergence (10⁻⁵ s⁻¹; '
                   'smoothed, ens mean) — pre-storm boundary / CI signal'),
    )

    P['storm_motion_mean'] = dict(
        cat='Kinematics',
        name='Storm Motion (0-6 km) + vectors',
        recipe='storm_motion', member_product='2dfld', n_members=5,
        cmap='smot', units='kt',
        spc_title=('Storm motion (0-6 km mean wind, kt; shaded speed + '
                   'decimated arrows) — ensemble mean of 5 members'),
    )

    # ------------------------------------------------------------------
    #  Phase J — Low-threshold 10 m wind signals.
    #
    #  refs_core already has 50-kt and 58-kt 10m-wind probability products,
    #  but those thresholds are tuned for SPC severe criteria and rarely
    #  light up the map in non-extreme regimes. SPC HREF publishes a 30-kt
    #  paintball + neighborhood-prob pair that's vastly more informative
    #  for routine severe-wind / gradient-wind events. Mirror that here.
    #
    #  IMPORTANT: thresholds must match wgrib2's idx string for the prob
    #  record. REFS publishes 10m-wind NHP at exactly these m/s values:
    #     10.3 (~20 kt), 15.4 (~30 kt), 18.01 (~35 kt),
    #     20.6 (~40 kt), 25.72 (~50 kt), 30.9 (~60 kt)
    #  Using a more-precise conversion (e.g. 15.43 for 30 kt) produces a
    #  byte-range match string the idx file doesn't contain → no record
    #  found → empty render. Always use the exact published value.
    # ------------------------------------------------------------------
    KT_30 = 15.4    # ≈ 30 kt  — matches REFS idx record exactly
    KT_40 = 20.6    # ≈ 40 kt
    KT_50 = 25.72   # = 50 kt
    # NOTE: wind10_prob_30kt removed 2026-05 (gust_prob_30mph covers it);
    # hourly wind10_prob_40kt removed 2026-06-11 per user — the 4-hr
    # windowed NP below is the operational version.

    # ------------------------------------------------------------------
    #  Phase L — 4-hr windowed neighborhood probabilities.
    #
    #  SPC HREF's NP pages are 4-hr products; REFS publishes hourly NPs
    #  only, which read systematically lower for the same scenario. The
    #  prob_window recipe takes the elementwise max of the hourly NPs
    #  across the 4 hours ending at the frame — direct visual parity with
    #  the SPC convention (slightly conservative vs a true windowed NP).
    # ------------------------------------------------------------------
    _4H = dict(cat='Severe Probabilities', recipe='prob_window',
               window_h=4, ftype='prob', cmap='prob', units='%')
    P['wind10_prob_30kt_4h'] = dict(
        _4H, name='Prob 4-hr max 10-m wind > 30 kt',
        var='si_10m', thresh=KT_30,
        spc_title=('4-hr neighborhood probability of 10 m wind > 30 kt '
                   '(max of hourly NPs) — SPC HREF NP[V>30] analog'),
    )
    P['wind10_prob_40kt_4h'] = dict(
        _4H, name='Prob 4-hr max 10-m wind > 40 kt',
        var='si_10m', thresh=KT_40,
        spc_title='4-hr neighborhood probability of 10 m wind > 40 kt (max of hourly NPs)',
    )
    P['wind10_prob_50kt_4h'] = dict(
        _4H, name='Prob 4-hr max 10-m wind > 50 kt',
        var='si_10m', thresh=KT_50,
        spc_title='4-hr neighborhood probability of 10 m wind > 50 kt (max of hourly NPs)',
    )
    P['uh25_prob_75_4h'] = dict(
        _4H, name='Prob 4-hr max UH 2-5 km > 75',
        var='mxuphl_25', thresh=75,
        spc_title='4-hr neighborhood probability of 2-5 km UH > 75 m²/s² (max of hourly NPs)',
    )
    P['uh25_prob_150_4h'] = dict(
        _4H, name='Prob 4-hr max UH 2-5 km > 150',
        var='mxuphl_25', thresh=150,
        spc_title='4-hr neighborhood probability of 2-5 km UH > 150 m²/s² (max of hourly NPs)',
    )
    P['refc_prob_40_4h'] = dict(
        _4H, name='Prob 4-hr max Comp. Refl. > 40 dBZ',
        var='refc', thresh=40,
        spc_title='4-hr neighborhood probability of composite reflectivity > 40 dBZ (max of hourly NPs)',
    )
    # Paintballs read the *member* file (raw wind speed m/s), so the exact
    # decimal of the threshold doesn't matter — numeric comparison, not
    # idx-string match. Keep them rounded the same way for consistency.
    # window_h=4 — SPC HREF's wind paintballs/NPs are *4-hr max* products.
    # REFS only publishes 1-h max member records and hourly NPs; the renderer
    # takes the elementwise max across the 4 hourly records ending at the
    # frame hour. Without this the plot paints a fraction of HREF's area and
    # reads misleadingly low.
    # (paintball_wind10_30kt removed 2026-06-11 — strict subset of the
    #  _nhp variant below, which adds the joint NHP contours.)
    P['paintball_wind10_50kt'] = dict(
        cat='Member Plots (RRFS_A)',
        name='Paintball: 4-hr max 10-m wind >50 kt',
        recipe='paintball', member_product='2dfld',
        var='si_10m', thresh=KT_50, window_h=4,
        spc_title='4-hr max 10 m wind speed > 50 kt (severe), ensemble paintball',
    )
    # SPC HREF-style combo: ensemble paintball of 10m wind >30 kt with a
    # joint neighborhood-probability contour overlay (wind >30 kt AND
    # refc >20 dBZ). True joint NHP would require per-member compute; we
    # approximate with min(P_wind, P_refc) — an upper bound that reads
    # well visually. Both joint-overlay thresholds correspond to records
    # confirmed present in the REFS prob.idx (WIND prob>15.4, REFC prob>20).
    P['paintball_wind10_30kt_nhp'] = dict(
        cat='Member Plots (RRFS_A)',
        name='Paintball: 4-hr max 10-m wind >30 kt + NH joint prob contours',
        recipe='paintball', member_product='2dfld',
        var='si_10m', thresh=KT_30, window_h=4,
        joint_overlay=[dict(
            specs=[('prob', 'si_10m', KT_30), ('prob', 'refc', 20)],
            combine='min',
            style='contour',
            levels=[10, 30, 50, 70, 90],
            colors='#ff2a2a',
            linewidths=[0.8, 1.2, 1.6, 2.0, 2.4],
            smooth=2.0,
        )],
        spc_title=('4-hr max 10 m wind > 30 kt paintball + neighborhood prob '
                   'wind >30 kt AND refc >20 dBZ (red contours)'),
    )

    # (wfirepot_member_mean removed 2026-06-11 per user — Fire Weather
    #  category retired.)

    # ------------------------------------------------------------------
    #  Phase K — 1-h, 3-h, and 24-h QPF ensemble mean plots.
    #
    #  Data verified against live S3 idx (refs.20260602 00z cycle):
    #    mean file carries APCP at every fhr (1-hr acc fcst) AND at
    #    fhr%3==0 (3-hr acc) AND fhr%6==0 (6-hr acc).  No single 24-hr
    #    APCP record exists, so qpf_24h_mean sums four 6-hr means via
    #    the new qpf_sum recipe in refs_core._qpf_sum.
    # ------------------------------------------------------------------
    P['qpf_1h_mean'] = dict(
        cat='QPF (Mean)', name='1-h QPF mean + prob >0.50"',
        ftype='mean', var='tp_sfc',
        step_from_fhr=lambda f: _step_for_acc(f, 1),
        cmap='qpf', units='in', convert=lambda x: x / 25.4,
        overlay=[dict(ftype='prob', var='tp_sfc',
                      step_from_fhr=lambda f: _step_for_acc(f, 1),
                      thresh=12.7,   # 0.5 in
                      style='contour',
                      levels=[10, 30, 50, 70, 90], smooth=2.0,
                      colors='#3a1f00',
                      linewidths=[0.8, 1.2, 1.6, 2.0, 2.4])],
        spc_title='1-hr QPF (in; ens mean shaded), neighborhood prob >0.50 in (contours)',
    )

    P['qpf_3h_mean'] = dict(
        cat='QPF (Mean)', name='3-h QPF mean + prob >1.00"',
        ftype='mean', var='tp_sfc',
        step_from_fhr=lambda f: _step_for_acc(f, 3),
        cmap='qpf', units='in', convert=lambda x: x / 25.4,
        overlay=[dict(ftype='prob', var='tp_sfc',
                      step_from_fhr=lambda f: _step_for_acc(f, 3),
                      thresh=25.4,   # 1.0 in
                      style='contour',
                      levels=[10, 30, 50, 70, 90], smooth=2.0,
                      colors='#3a1f00',
                      linewidths=[0.8, 1.2, 1.6, 2.0, 2.4])],
        spc_title='3-hr QPF (in; ens mean shaded), neighborhood prob >1.00 in (contours)',
    )

    P['qpf_24h_mean'] = dict(
        cat='QPF (Mean)', name='24-h QPF mean',
        recipe='qpf_sum', sum_period=6, n_steps=4,
        cmap='qpf', units='in',
        min_fhr=24, fhr_stride=6,
        overlay=[dict(ftype='prob', var='tp_sfc',
                      step_from_fhr=lambda f: _step_for_acc(f, 24),
                      thresh=50.8,   # 2.0 in
                      style='contour',
                      levels=[10, 30, 50, 70, 90], smooth=2.0,
                      colors='#3a1f00',
                      linewidths=[0.8, 1.2, 1.6, 2.0, 2.4])],
        spc_title='24-hr QPF (in; ens mean, sum of 4 × 6-hr periods), prob >2.00 in (contours)',
    )

    extra_keys = [
        # Phase D
        "refc_uh_swath", "refc_ltng_combo", "mlcape_shear_env",
        "sbcape_srh_env", "pwat_llj_combo",
        # Phase E (warm-season composites)
        "mucape_shear_env", "sbcape_combo_severe", "pwat_refc_combo",
        "refc_maxuvv_combo", "refc_echotop_combo",
        # Phase G (sounding-derived composite parameters)
        "scp_mean", "stp_fixed_mean", "ehi_03_mean",
        "lapse_75_mean", "ship_mean", "stdiv_250_mean",
        # Phase H (member-derived)
        "vil_member_mean", "gust_member_mean", "dcape_member_mean",
        "mconv_member_mean", "storm_motion_mean",
        # Phase J (low-threshold 10m wind signals)
        "paintball_wind10_50kt", "paintball_wind10_30kt_nhp",
        # Phase K (QPF ensemble mean: 1-h, 3-h, 24-h)
        "qpf_1h_mean", "qpf_3h_mean", "qpf_24h_mean",
        # Phase L+ (4-hr NPs and multi-layer composites)
        "wind10_prob_30kt_4h", "wind10_prob_40kt_4h", "wind10_prob_50kt_4h",
        "uh25_prob_75_4h", "uh25_prob_150_4h", "refc_prob_40_4h",
        "moisture_lift_combo", "mtransport_850", "mlcape_inflow_combo",
    ]
    # Phase G plumbing: tag each product with `min_fhr` so the frontend can
    # disable forecast-hour cells where the product can't possibly have data
    # (e.g. a 24-h QPF below F24). Detection: call the product's
    # ``step_from_fhr`` at a comfortably high fhr (30) and parse the
    # accumulation period out of the returned ``"A-B"`` string. Inst
    # products without an accumulation default to min_fhr=0.
    PROBE_FHR = 30
    for k, prod in P.items():
        if 'min_fhr' in prod and 'fhr_stride' in prod:
            continue
        fn = prod.get('step_from_fhr')
        if fn is None:
            prod.setdefault('min_fhr', 0)
            prod.setdefault('fhr_stride', 1)
            continue
        try:
            s = fn(PROBE_FHR)
        except Exception:
            s = None
        if isinstance(s, str) and '-' in s:
            try:
                a, b = (int(x) for x in s.split('-'))
                period = b - a
                # Accumulation products: REFS only publishes the requested
                # period at fhrs aligned to that period (e.g. a 6-h product
                # is valid at fhr ∈ {6, 12, 18, 24, ...}). Set fhr_stride
                # equal to the period so the frontend can gray out the
                # intervening hours rather than showing the no-data
                # placeholder on most of them.
                prod.setdefault('min_fhr', period)
                prod.setdefault('fhr_stride', max(1, period))
            except ValueError:
                prod.setdefault('min_fhr', 0)
                prod.setdefault('fhr_stride', 1)
        else:
            prod.setdefault('min_fhr', 0)
            prod.setdefault('fhr_stride', 1)

    # Phase F: strip parentheticals + truncate every spc_title so the
    # rendered figure header doesn't overflow into the right-side timestamps.
    trimmed = 0
    for k, prod in P.items():
        orig = prod.get("spc_title")
        if not orig:
            continue
        short = _shorten_title(orig)
        if short != orig:
            prod["spc_title"] = short
            trimmed += 1
    print(f"[extras] registered {len(extra_keys)} new products "
          f"(catalog now has {len(P)} total); "
          f"shortened {trimmed} spc_titles", flush=True)
