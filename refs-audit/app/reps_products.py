"""REPS v1 product registry.

Registers Environment Canada REPS products into refs_core.PRODUCTS as its
own "REPS" tab (see app/catalog.py's TAB_ORDER/CATEGORY_TO_TAB), reusing
the recipe='reps_mean' dispatch added to PlotJob.render() in refs_core.py.

v1 is deliberately scoped to ensemble-MEAN products only. REPS also
publishes "-Prob" files (e.g. TMP-Prob, WIND-Prob), but those turned out
to be multi-message percentile/threshold-probability bundles (9 messages
per file: 5 percentiles + 4 unlabelled probability messages) rather than
a single clean field -- decoding them correctly needs its own validation
pass, deferred to a later addition rather than guessed at here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import refs_core as core  # noqa: E402

_K_TO_F = lambda k: (k - 273.15) * 9 / 5 + 32   # noqa: E731
_MS_TO_KT = lambda ms: ms * 1.94384              # noqa: E731
_PA_TO_HPA = lambda pa: pa / 100.0               # noqa: E731
_M_TO_DAM = lambda m: m / 10.0                   # noqa: E731 -- geopotential height, decameters
_FRAC_TO_PCT = lambda f: f * 100.0               # noqa: E731 -- volumetric soil moisture 0-1 -> %
# Spread (std-dev) conversions are DELTAS, not absolute values -- a Kelvin
# std-dev converts to a Fahrenheit std-dev by scaling only (no +32/-273.15
# offset, which would be meaningless applied to a spread rather than a value).
_K_TO_F_DELTA = lambda k: k * 9.0 / 5.0           # noqa: E731
_MM_TO_IN = lambda mm: mm / 25.4                  # noqa: E731

REPS_MAX_FHOUR = 72
REPS_FHR_STEP = 3

PRODUCTS = {
    "reps_t2m_mean": dict(
        cat="REPS", name="2m Temperature (Mean)", recipe="reps_mean",
        reps_var="TMP", reps_level="AGL-2m", mslp_overlay=True,
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS 2m temperature — 21-member ensemble mean, MSLP contours",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_rh2m_mean": dict(
        cat="REPS", name="2m Relative Humidity (Mean)", recipe="reps_mean",
        reps_var="RH", reps_level="AGL-2m", mslp_overlay=True,
        cmap="rh", units="%",
        spc_title="REPS 2m relative humidity — 21-member ensemble mean, MSLP contours",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind10m_mean": dict(
        cat="REPS", name="10m Wind Speed (Mean)", recipe="reps_mean",
        reps_var="WIND", reps_level="AGL-10m", mslp_overlay=True,
        cmap="wind_sfc", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 10m wind speed — 21-member ensemble mean, MSLP contours",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_mslp_mean": dict(
        cat="REPS", name="MSLP (Mean)", recipe="reps_mean",
        reps_var="PRMSL", reps_level="MSL", mslp_overlay=True,
        cmap="mslp", units="hPa", convert=_PA_TO_HPA,
        spc_title="REPS mean sea-level pressure — 21-member ensemble mean, contoured",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    # ---- v1.5: pressure-level / soil / cloud additions ----------------
    "reps_hgt500_mean": dict(
        cat="REPS", name="500mb Heights (Mean)", recipe="reps_mean",
        reps_var="HGT", reps_level="ISBL-0500",
        cmap="hgt500", units="dam", convert=_M_TO_DAM,
        spc_title="REPS 500-mb geopotential height — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_temp850_mean": dict(
        cat="REPS", name="850mb Temperature (Mean)", recipe="reps_mean",
        reps_var="TMP", reps_level="ISBL-0850",
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS 850-mb temperature — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_rh700_mean": dict(
        cat="REPS", name="700mb Relative Humidity (Mean)", recipe="reps_mean",
        reps_var="RH", reps_level="ISBL-0700",
        cmap="rh", units="%",
        spc_title="REPS 700-mb relative humidity — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind500_mean": dict(
        cat="REPS", name="500mb Wind Speed (Mean)", recipe="reps_wind_level_mean",
        reps_level="ISBL-0500",
        cmap="wind500", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 500-mb wind speed — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_soilt_mean": dict(
        cat="REPS", name="Soil Temperature 10cm (Mean)", recipe="reps_mean",
        reps_var="TSOIL", reps_level="DBS-10cm",
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS 10-cm soil temperature — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_soilm_mean": dict(
        cat="REPS", name="Soil Moisture 10cm (Mean)", recipe="reps_mean",
        reps_var="VSOILM", reps_level="DBS-10cm",
        cmap="rh", units="%", convert=_FRAC_TO_PCT,
        spc_title="REPS 10-cm volumetric soil moisture — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_tcdc_mean": dict(
        cat="REPS", name="Total Cloud Cover (Mean)", recipe="reps_mean",
        reps_var="TCDC", reps_level="SFC",
        cmap="clouds", units="%",
        spc_title="REPS total cloud cover — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    # ---- v2: creative derived/spread/precip-type products -------------
    "reps_heat_index_mean": dict(
        cat="REPS Derived", name="Heat Index (Mean)", recipe="reps_heat_index_mean",
        cmap="t2m", units="degF", mslp_overlay=True,
        spc_title="REPS heat index — 21-member ensemble mean (per-member Rothfusz, then averaged), MSLP contours",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind_chill_mean": dict(
        cat="REPS Derived", name="Wind Chill (Mean)", recipe="reps_wind_chill_mean",
        cmap="t2m", units="degF", mslp_overlay=True,
        spc_title="REPS wind chill — 21-member ensemble mean (per-member NWS formula, then averaged), MSLP contours",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_freezing_level": dict(
        cat="REPS Derived", name="Freezing Level (0°C Height)", recipe="reps_freezing_level",
        cmap="frzlvl", units="m",
        spc_title="REPS freezing level — height of 0°C isotherm from ensemble-mean T/Z profile",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_ptype_dominant": dict(
        cat="REPS Precip Type", name="Dominant Precip Type", recipe="reps_ptype_dominant",
        cmap="ptype_dom", units="", mslp_overlay=True,
        cbar_tick_positions=[0.5, 1.5, 2.5, 3.5],
        cbar_tick_labels=["Rain", "Snow", "Ice Pellets", "Frz Rain"],
        spc_title="REPS dominant precipitation type — winner among ensemble-mean rain/snow/ice pellets/freezing rain accumulation, MSLP contours",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_ptype_thickness": dict(
        cat="REPS Precip Type", name="Precip Type + Thickness", recipe="reps_ptype_thickness",
        cmap="ptype_dom", units="",
        cbar_tick_positions=[0.5, 1.5, 2.5, 3.5],
        cbar_tick_labels=["Rain", "Snow", "Ice Pellets", "Frz Rain"],
        spc_title="REPS dominant precipitation type with 1000-500mb thickness contours "
                   "(540 dam highlighted, classic rain/snow-line context)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_spread_t2m": dict(
        cat="REPS Spread", name="2m Temperature Spread", recipe="reps_spread",
        reps_var="TMP", reps_level="AGL-2m",
        cmap="sptemp2m", units="degF", convert=_K_TO_F_DELTA,
        spc_title="REPS 2m temperature — ensemble spread (std dev across 21 members)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_spread_wind10m": dict(
        cat="REPS Spread", name="10m Wind Speed Spread", recipe="reps_spread",
        reps_var="WIND", reps_level="AGL-10m",
        cmap="spwind10m", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 10m wind speed — ensemble spread (std dev across 21 members)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    # ---- v3: rain/flooding/severe-proxy products -----------------------
    # APCP/SFCWRO are cumulative-since-init (confirmed via direct GRIB
    # inspection) -- windowed totals come from load_reps_windowed_mean's
    # differencing, so these are available every 3-hourly step once
    # fhr > window_h, not just at 6-hourly boundaries like REFS's own QPF.
    "reps_qpf_mean_6h": dict(
        cat="REPS Precipitation", name="6-Hour QPF (Mean)", recipe="reps_qpf_mean",
        reps_qpf_window_h=6,
        cmap="qpf", units="in", convert=_MM_TO_IN,
        spc_title="REPS 6-hour QPF — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, min_fhr=6, source="reps",
    ),
    "reps_qpf_mean_24h": dict(
        cat="REPS Precipitation", name="24-Hour QPF (Mean)", recipe="reps_qpf_mean",
        reps_qpf_window_h=24,
        cmap="qpf", units="in", convert=_MM_TO_IN,
        spc_title="REPS 24-hour QPF — 21-member ensemble mean",
        fhr_stride=REPS_FHR_STEP, min_fhr=24, source="reps",
    ),
    "reps_runoff_mean_6h": dict(
        cat="REPS Precipitation", name="6-Hour Surface Runoff (Mean)",
        recipe="reps_runoff_mean", reps_qpf_window_h=6,
        cmap="qpf", units="in", convert=_MM_TO_IN,
        spc_title="REPS 6-hour surface runoff — 21-member ensemble mean "
                   "(land-surface runoff, not just rainfall amount)",
        fhr_stride=REPS_FHR_STEP, min_fhr=6, source="reps",
    ),
    # Thresholds decoded directly from REPS's own TPRATE-Accum6h-Prob file
    # (genuine probability-of-exceedance messages, no ensemble math) --
    # only published at 6-hourly-boundary fhrs.
    "reps_qpf_prob_1in_6h": dict(
        cat="REPS Flooding", name="P(6-hr QPF > 1.0 in)", recipe="reps_qpf_prob",
        reps_qpf_window_h=6, reps_qpf_thresh_mm=25,
        cmap="prob", units="%",
        spc_title="REPS probability of 6-hour QPF exceeding 1.0 in — "
                   "decoded directly from REPS's own probability product",
        fhr_stride=6, min_fhr=6, source="reps",
    ),
    "reps_qpf_prob_2in_6h": dict(
        cat="REPS Flooding", name="P(6-hr QPF > 2.0 in)", recipe="reps_qpf_prob",
        reps_qpf_window_h=6, reps_qpf_thresh_mm=50,
        cmap="prob", units="%",
        spc_title="REPS probability of 6-hour QPF exceeding 2.0 in (flash-flood-relevant) — "
                   "decoded directly from REPS's own probability product",
        fhr_stride=6, min_fhr=6, source="reps",
    ),
    # Severe-proxy ingredients -- REPS has no CAPE/CIN/reflectivity/UH at
    # all (confirmed via the full variable-token inventory), so these are
    # honest substitutes built from what REPS actually publishes, not a
    # full instability parameter.
    "reps_shear_850_500_mean": dict(
        cat="REPS Severe Proxy", name="850-500mb Bulk Shear (Mean)",
        recipe="reps_shear_mean", reps_level_lo="ISBL-0850", reps_level_hi="ISBL-0500",
        cmap="shear", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 850-500mb bulk shear — 21-member ensemble mean "
                   "(per-member vector shear, then averaged)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_lapse_700_500_mean": dict(
        cat="REPS Severe Proxy", name="700-500mb Lapse Rate (Mean)",
        recipe="reps_lapse_rate_mean", reps_level_lo="ISBL-0700", reps_level_hi="ISBL-0500",
        cmap="lapse", units="K/km",
        spc_title="REPS 700-500mb lapse rate — from ensemble-mean T/height profiles",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind850_mean": dict(
        cat="REPS Severe Proxy", name="850mb Wind Speed (Mean)",
        recipe="reps_wind_level_mean", reps_level="ISBL-0850",
        cmap="wind_sfc", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 850-mb wind speed — 21-member ensemble mean "
                   "(low-level jet / moisture-transport indicator)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    # ---- v4: "reasonable worst case" extremes from the TMP/WIND/HEATX/
    # WCF-Prob bundles' percentile (PDT6) and derived-stat (PDT2) messages.
    # These files have NO threshold-exceedance messages (confirmed via
    # eccodes inspection -- unlike the precip-rate -Prob family above),
    # so this is percentile/mean/spread/min/max, not probability-of-X.
    "reps_heatidx_p90": dict(
        cat="REPS Extremes", name="Heat Index (90th Percentile)",
        recipe="reps_prob_bundle_stat",
        reps_var="HEATX-Prob", reps_level="AGL-2m",
        reps_pdt=6, reps_match_key="percentileValue", reps_match_val=90,
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS heat index — 90th percentile across 21 members "
                   "(reasonable worst-case heat, decoded directly from REPS's own product)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_heatidx_max": dict(
        cat="REPS Extremes", name="Heat Index (Ensemble Max)",
        recipe="reps_prob_bundle_stat",
        reps_var="HEATX-Prob", reps_level="AGL-2m",
        reps_pdt=2, reps_match_key="derivedForecast", reps_match_val=9,
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS heat index — maximum across all 21 members "
                   "(single hottest member at each point, decoded directly from REPS's own product)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wchill_p10": dict(
        cat="REPS Extremes", name="Wind Chill (10th Percentile)",
        recipe="reps_prob_bundle_stat",
        reps_var="WCF-Prob", reps_level="AGL-2m",
        reps_pdt=6, reps_match_key="percentileValue", reps_match_val=10,
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS wind chill — 10th percentile across 21 members "
                   "(reasonable worst-case cold, decoded directly from REPS's own product)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_t2m_p10": dict(
        cat="REPS Extremes", name="2m Temperature (10th Percentile)",
        recipe="reps_prob_bundle_stat",
        reps_var="TMP-Prob", reps_level="AGL-2m",
        reps_pdt=6, reps_match_key="percentileValue", reps_match_val=10,
        cmap="t2m", units="degF", convert=_K_TO_F,
        spc_title="REPS 2m temperature — 10th percentile across 21 members "
                   "(reasonable worst-case cold/frost risk, decoded directly from REPS's own product)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind10m_p90": dict(
        cat="REPS Extremes", name="10m Wind Speed (90th Percentile)",
        recipe="reps_prob_bundle_stat",
        reps_var="WIND-Prob", reps_level="AGL-10m",
        reps_pdt=6, reps_match_key="percentileValue", reps_match_val=90,
        cmap="wind_sfc", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 10m wind speed — 90th percentile across 21 members "
                   "(reasonable worst-case wind, decoded directly from REPS's own product)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_wind10m_max": dict(
        cat="REPS Extremes", name="10m Wind Speed (Ensemble Max)",
        recipe="reps_prob_bundle_stat",
        reps_var="WIND-Prob", reps_level="AGL-10m",
        reps_pdt=2, reps_match_key="derivedForecast", reps_match_val=9,
        cmap="wind_sfc", units="kt", convert=_MS_TO_KT,
        spc_title="REPS 10m wind speed — maximum across all 21 members "
                   "(single windiest member at each point, decoded directly from REPS's own product)",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    # ---- v5: synoptic-style plots matching REFS/HREF's own look -------
    # (shaded field + height contours + barbs). REPS's grid is rotated-
    # pole, so the divergence/vorticity/advection math is REPS-specific
    # (app/reps_core.py's _grid_metrics), not a reuse of REFS's own
    # uniform-grid _divergence helper.
    "reps_div250_synoptic": dict(
        cat="REPS Synoptic", name="250mb Wind + Heights + Divergence",
        recipe="reps_div250_synoptic", reps_level="ISBL-0250",
        cmap="wind250", units="kt",
        spc_title="REPS 250-mb wind speed (shaded), heights (dam), divergence contours "
                   "(x10-5/s), ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_vort500_synoptic": dict(
        cat="REPS Synoptic", name="500mb Absolute Vorticity + Heights",
        recipe="reps_vort500", reps_level="ISBL-0500",
        cmap="vort", units="x10-5/s",
        spc_title="REPS 500-mb absolute vorticity (shaded), heights (dam), wind barbs, ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
    "reps_temp_adv850": dict(
        cat="REPS Synoptic", name="850mb Temperature Advection",
        recipe="reps_temp_advection", reps_level="ISBL-0850",
        cmap="temp_adv", units="K/3h",
        spc_title="REPS 850-mb temperature advection (shaded, blue=cold/red=warm-air "
                   "advection), heights (dam), wind barbs, ensemble mean",
        fhr_stride=REPS_FHR_STEP, source="reps",
    ),
}


def register() -> None:
    for pid, prod in PRODUCTS.items():
        core.PRODUCTS[pid] = prod
