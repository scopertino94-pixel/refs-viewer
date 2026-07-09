/*
 * REFS-Viewer frontend.
 *
 * SPC-HREF-style layout: top toolbar (date / run / sector), tab strip of
 * product categories, F-hour timeline scrubber, central map, right-rail
 * product picker / palette / shortcuts.
 *
 * Backend produces a self-contained PNG per (date, run, pid, region, fhr).
 * The client just swaps <img src> between cached tiles.
 *
 * Hotkeys:  <  prev    >  next    p  play/pause    l  latest    t  theme
 */

const $ = (id) => document.getElementById(id);

const state = {
  build: "init",
  catalog: { tabs: {}, regions: [], palettes: [], themes: [], max_fhour: 60, href_max_fhour: 48, runs: [0,6,12,18] },
  cycles: [],                            // [{date, run, label, max_fhour}] from /api/cycles
  hrefCycles: [],                        // [{date, run, label, max_fhour}] from /api/href-cycles
  date: null, run: null,                 // currently-selected cycle
  sector: "CONUS",
  palette: "Default",
  tab: null,
  pid: null, pidName: "",
  fhr: 6,
  fmin: 0, fmax: 60,
  step: 3,                               // 1 / 3 / 6 hr stride for timeline + anim (default 3-hr)
  availableFhours: new Set(),
  loadedFhours: new Set(),
  productSource: "",            // "" for REFS/HREF products, "spc_post" for SPC guidance
  sparseAvail: false,           // true when availableFhours is an exact sparse set (SPC)
  followLatest: false,          // "Latest" mode: keep snapping to the newest posted fhr
  playing: false, playTimer: null,
  speed: 3,
  pendingLoad: 0,
  // Custom sectors: array of {key, name, bbox:[lonMin,latMin,lonMax,latMax]}
  customSectors: [],
  drawMode: false,
  // Basemap overlay toggles (persisted to localStorage + URL).
  showCounties: false,
  showCities: false,
  // Model selection: "refs" (default) or "href"
  model: "refs",
  // Compare mode state
  compareMode: false,
  compareStyle: "split",    // "split" | "swipe"
  compareType:  "href",     // "href"  | "run_prev"
  swipePct: 50,
  compareLoadedFhours: new Set(),
  paintballVar: "refc|40",
};

const SPEED_TO_MS = { 1: 1200, 2: 800, 3: 500, 4: 320, 5: 200 };
// Renders are serialized backend-side and cold renders take ~30-60s. Keep
// preload concurrency low so the user-visible frame isn't stuck behind
// the queue, and the HF gateway doesn't 504 long-waiting connections.
const PRELOAD_CONCURRENCY = 3;
// While the user is actively scrubbing/clicking we pause preload kickoff so
// interactive renders aren't queued behind background ones on the server.
const PRELOAD_INTERACTION_PAUSE_MS = 700;
let preloadGen = 0;
let lastInteractionTs = 0;
function noteInteraction() { lastInteractionTs = Date.now(); }
const FRAME = () => $("frame");

document.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("refs-theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  $("btn-theme").textContent = saved === "dark" ? "☾" : "☀";

  // First-visit ALPHA acknowledgement gate. Shown until the user accepts;
  // the choice persists in localStorage so it only appears once per browser.
  const alphaOverlay = $("alpha-overlay");
  if (alphaOverlay) {
    if (localStorage.getItem("refs-alpha-ack") !== "1") {
      alphaOverlay.classList.remove("hidden");
    }
    $("alpha-accept").addEventListener("click", () => {
      localStorage.setItem("refs-alpha-ack", "1");
      alphaOverlay.classList.add("hidden");
    });
  }

  // Right-rail collapse preference (default = expanded)
  const railCollapsed = localStorage.getItem("refs-rail-collapsed") === "1";
  applyRailCollapsed(railCollapsed);
  $("rail-toggle").addEventListener("click", () => {
    const cur = $("viewer").classList.contains("rail-collapsed");
    applyRailCollapsed(!cur);
  });

  init().catch(err => setStatus("Init failed: " + err));
});

function applyRailCollapsed(collapsed) {
  $("viewer").classList.toggle("rail-collapsed", collapsed);
  $("rail-toggle").textContent = collapsed ? "›" : "‹";
  $("rail-toggle").title = (collapsed ? "Show" : "Hide") + " right rail (\\)";
  localStorage.setItem("refs-rail-collapsed", collapsed ? "1" : "0");
  // Day-mark + play-head positioning depends on map width; recompute.
  requestAnimationFrame(() => {
    if (typeof positionDayMarks === "function") positionDayMarks();
  });
}

async function init() {
  setStatus("Loading catalog…");
  const [catalogResp, cyclesResp, versionResp] = await Promise.all([
    fetch("/api/catalog").then(r => r.json()),
    fetch("/api/cycles").then(r => r.json()),
    fetch("/api/version").then(r => r.json()).catch(() => ({build: "?"})),
  ]);
  state.catalog = catalogResp;
  state.favorites = loadFavorites();
  refreshFavoritesTab();
  state.cycles = cyclesResp;
  state.build = versionResp.build || "?";
  $("footer-version").textContent =
    `REFS v${versionResp.version || ""} · build ${state.build}`;

  // Kick off HREF cycle discovery in the background (non-blocking).
  fetch("/api/href-cycles").then(r => r.json())
    .then(c => { state.hrefCycles = c; })
    .catch(() => {});

  loadCustomSectors();          // from localStorage
  populateSectors();
  populatePalettes();
  buildTabs();

  // 1. Apply URL params if present (overrides defaults below)
  const urlState = readURLState();

  // Apply model from URL before anything else, so cycle selection below
  // picks from the right pool.
  if (urlState.model) state.model = urlState.model;
  $("model-select").value = state.model;
  applyModelClass();

  // 2. Cycle defaults — most recent cycle, unless URL overrides
  if (urlState.date && urlState.run !== undefined) {
    state.date = urlState.date;
    state.run = urlState.run;
  } else if (state.cycles.length) {
    state.date = state.cycles[0].date;
    state.run  = state.cycles[0].run;
  } else {
    const now = new Date();
    state.date = now.toISOString().slice(0,10).replace(/-/g, "");
    state.run = 12;
  }

  // 3. Tab + product defaults
  // First visit (no URL state) lands on the 3-h QPF PMM product in the
  // Precipitation tab rather than the first tab (SPC Guidance), which is
  // 00z/12z-only and can be empty depending on the cycle.
  const DEFAULT_TAB = "Precipitation";
  const DEFAULT_PID = "qpf_3h_pmmn_series";
  const tabNames = Object.keys(state.catalog.tabs);
  if (urlState.pid && findTabForPid(urlState.pid)) {
    state.pid = urlState.pid;
    state.tab = findTabForPid(urlState.pid);
  } else if (urlState.tab && state.catalog.tabs[urlState.tab]) {
    state.tab = urlState.tab;
    state.pid = firstPidInTab(state.tab);
  } else if (state.catalog.tabs[DEFAULT_TAB] && findTabForPid(DEFAULT_PID) === DEFAULT_TAB) {
    state.tab = DEFAULT_TAB;
    state.pid = DEFAULT_PID;
  } else {
    state.tab = tabNames[0];
    state.pid = firstPidInTab(state.tab);
  }
  state.pidName = nameForPid(state.pid);
  state.minFhr = minFhrForPid(state.pid);
  state.fhrStride = fhrStrideForPid(state.pid);
  state.productSource = sourceForPid(state.pid);

  // 4. Sector / palette / fhr / loop range / step from URL
  if (urlState.sector) state.sector = urlState.sector;
  if (urlState.palette) state.palette = urlState.palette;
  // Seed the prob/non-prob palette tracker from the initial product so the
  // first product click doesn't spuriously re-default the palette.
  state._lastWasProb = isProbPid(state.pid);
  if (urlState.fhr !== undefined) state.fhr = urlState.fhr;
  if (urlState.step !== undefined && [1,3,6].includes(urlState.step))
    state.step = urlState.step;
  state.fmin = urlState.fmin !== undefined ? urlState.fmin : 0;
  const defaultFmax = activeMaxFhour();
  state.fmax = urlState.fmax !== undefined
             ? Math.min(urlState.fmax, activeMaxFhour())
             : defaultFmax;

  // Basemap toggle defaults: URL > localStorage > false. Both are off by
  // default for the minimal AG-WX-style basemap.
  state.showCounties = urlState.showCounties !== undefined
    ? urlState.showCounties
    : localStorage.getItem("refs-counties") === "1";
  state.showCities = urlState.showCities !== undefined
    ? urlState.showCities
    : localStorage.getItem("refs-cities") === "1";

  // 5. Reflect into form controls
  $("sector").value = state.sector; _syncSectorBtn();
  // Guard against a stale palette from an old URL/localStorage that's no
  // longer offered (we trimmed the palette list) — fall back to Default.
  if (!state.catalog.palettes.includes(state.palette)) state.palette = "Default";
  $("palette").value = state.palette;
  $("step").value = String(state.step);
  updateEditSectorButton();

  paintTabs();
  buildProductList();
  paintTopbar();
  buildTimeline();

  wireEvents();
  // SPC guidance is the default landing tab; if the default product is an SPC
  // product, force HREF + a 00z/12z cycle before resolving availability.
  updateSpcUiLock();
  if (state.productSource === "spc_post") await enforceSpcConstraints();
  await refreshCycleStatus();
  writeURLState();
  await loadFrame();
  startPreloadAllHours();
}

// ---- URL state sync ------------------------------------------------------
// Mirrors current state into the address bar as ?p=&g=&rd=&rt=&f=&tab=&fmin=&fmax=
// so a tab/refresh restores the same view and links are shareable.

function readURLState() {
  const qs = new URLSearchParams(location.search);
  const out = {};
  const p = qs.get("p");           if (p) out.pid = p;
  const tab = qs.get("tab");       if (tab) out.tab = tab;
  const g = qs.get("g");           if (g) out.sector = g;
  const palette = qs.get("palette"); if (palette) out.palette = palette;
  const rd = qs.get("rd");         if (rd && /^\d{8}$/.test(rd)) out.date = rd;
  const rt = qs.get("rt");
  if (rt !== null && /^\d{1,2}$/.test(rt)) out.run = parseInt(rt, 10);
  const f = qs.get("f");
  if (f !== null && /^\d+$/.test(f)) out.fhr = parseInt(f, 10);
  const fmin = qs.get("fmin");
  if (fmin !== null && /^\d+$/.test(fmin)) out.fmin = parseInt(fmin, 10);
  const fmax = qs.get("fmax");
  if (fmax !== null && /^\d+$/.test(fmax)) out.fmax = parseInt(fmax, 10);
  const step = qs.get("step");
  if (step !== null && /^\d+$/.test(step)) out.step = parseInt(step, 10);
  const co = qs.get("co"); if (co !== null) out.showCounties = co === "1";
  const ci = qs.get("ci"); if (ci !== null) out.showCities   = ci === "1";
  const model = qs.get("model");
  if (model && ["refs","href"].includes(model)) out.model = model;
  return out;
}

let _urlWriteTimer = null;
function writeURLState() {
  // Debounce so dragging the play-head doesn't spam history.
  clearTimeout(_urlWriteTimer);
  _urlWriteTimer = setTimeout(() => {
    const qs = new URLSearchParams();
    if (state.pid)     qs.set("p", state.pid);
    if (state.tab)     qs.set("tab", state.tab);
    if (state.sector)  qs.set("g", state.sector);
    if (state.palette && state.palette !== "Default")
                       qs.set("palette", state.palette);
    if (state.date)    qs.set("rd", state.date);
    if (state.run !== null) qs.set("rt", String(state.run).padStart(2, "0"));
    qs.set("f", String(state.fhr));
    if (state.fmin !== 0)            qs.set("fmin", String(state.fmin));
    if (state.fmax !== activeMaxFhour()) qs.set("fmax", String(state.fmax));
    if (state.step !== 3)            qs.set("step", String(state.step));
    if (state.showCounties)          qs.set("co", "1");
    if (state.showCities)            qs.set("ci", "1");
    if (state.model && state.model !== "refs") qs.set("model", state.model);
    const url = `${location.pathname}?${qs}`;
    history.replaceState(null, "", url);
  }, 120);
}

// ----- Model helpers -------------------------------------------------------
function activeMaxFhour() {
  return state.model === "href"
    ? (state.catalog.href_max_fhour || 48)
    : state.catalog.max_fhour;
}

function applyModelClass() {
  document.body.classList.toggle("model-href", state.model === "href");
}

function hrefCycleExists(date, run) {
  return state.hrefCycles.some(c => c.date === date && c.run === run);
}

function findTabForPid(pid) {
  for (const [tab, cats] of Object.entries(state.catalog.tabs))
    for (const items of Object.values(cats))
      if (items.some(it => it.pid === pid)) return tab;
  return null;
}

// ----- Favorites ----------------------------------------------------------
const FAV_TAB = "★ Favorites";
const FAV_KEY = "refs_favorites";

function loadFavorites() {
  try { return new Set(JSON.parse(localStorage.getItem(FAV_KEY) || "[]")); }
  catch { return new Set(); }
}
function saveFavorites() {
  try { localStorage.setItem(FAV_KEY, JSON.stringify([...state.favorites])); }
  catch { /* private mode etc. */ }
}
function refreshFavoritesTab() {
  // Rebuild the synthetic Favorites tab from the (real) catalog tabs.
  // Always present and always FIRST in tab order; empty shows a hint.
  const items = [];
  for (const [tab, cats] of Object.entries(state.catalog.tabs)) {
    if (tab === FAV_TAB) continue;
    for (const arr of Object.values(cats))
      for (const it of arr)
        if (state.favorites.has(it.pid) && !items.some(x => x.pid === it.pid))
          items.push(it);
  }
  const rest = {};
  for (const [t, c] of Object.entries(state.catalog.tabs))
    if (t !== FAV_TAB) rest[t] = c;
  state.catalog.tabs = Object.assign(
    { [FAV_TAB]: items.length ? { "Pinned": items } : {} }, rest);
}
function toggleFavorite(pid) {
  if (state.favorites.has(pid)) state.favorites.delete(pid);
  else state.favorites.add(pid);
  saveFavorites();
  refreshFavoritesTab();
  buildProductList();
}

// ----- Helpers -----------------------------------------------------------
function firstPidInTab(tab) {
  const cats = state.catalog.tabs[tab] || {};
  for (const items of Object.values(cats)) if (items.length) return items[0].pid;
  return null;
}
function nameForPid(pid) {
  for (const cats of Object.values(state.catalog.tabs))
    for (const items of Object.values(cats))
      for (const it of items) if (it.pid === pid) return it.name;
  return pid;
}
function minFhrForPid(pid) {
  // Earliest fhr where this product can have data. Floored at 1: neither REFS
  // nor HREF publishes an F000 analysis file — forecasts start at F001 — so
  // F000 must never be offered (instantaneous fields whose catalog min_fhr is
  // 0 would otherwise show a phantom F000 cell that always fails to load).
  // Accumulation products (e.g. 24-h QPF) carry a larger min_fhr already.
  for (const cats of Object.values(state.catalog.tabs))
    for (const items of Object.values(cats))
      for (const it of items) if (it.pid === pid) return Math.max(1, it.min_fhr || 0);
  return 1;
}
function sourceForPid(pid) {
  // "spc_post" for SPC calibrated guidance, "" otherwise.
  for (const cats of Object.values(state.catalog.tabs))
    for (const items of Object.values(cats))
      for (const it of items) if (it.pid === pid) return it.source || "";
  return "";
}
function fhrStrideForPid(pid) {
  // REFS publishes n-hour accumulations only at fhrs that are multiples
  // of n. A 6-h QPF is valid at F=6,12,18,...; a 3-h QPF at F=3,6,9,...
  // Instant products = stride 1.
  for (const cats of Object.values(state.catalog.tabs))
    for (const items of Object.values(cats))
      for (const it of items) if (it.pid === pid) return it.fhr_stride || 1;
  return 1;
}
function isFhrValid(h) {
  // True if forecast hour h is one the user can actually load right now:
  //   - within the product's valid window (h ≥ min_fhr)
  //   - aligned to the product's accumulation stride
  //   - aligned to the current timeline step
  if (h < (state.minFhr || 0)) return false;
  const stride = state.fhrStride || 1;
  if (stride > 1 && (h % stride) !== 0) return false;
  if (state.step > 1 && (h % state.step) !== 0) return false;
  return true;
}
function spcTitleForPid(pid) {
  for (const cats of Object.values(state.catalog.tabs))
    for (const items of Object.values(cats))
      for (const it of items) if (it.pid === pid) return it.spc_title || it.name;
  return "";
}
function ftypeForPid(pid) {
  for (const cats of Object.values(state.catalog.tabs))
    for (const items of Object.values(cats))
      for (const it of items) if (it.pid === pid) return it.ftype;
  return "";
}
// A product is a "probability" product when its colorbar unit is percent.
// Used to auto-select the SPC Ramp palette for probs (Default otherwise).
function isProbPid(pid) {
  for (const cats of Object.values(state.catalog.tabs))
    for (const items of Object.values(cats))
      for (const it of items) if (it.pid === pid) return it.units === "%";
  return false;
}
function currentTheme() {
  return document.documentElement.getAttribute("data-theme") || "dark";
}

// ----- Topbar population --------------------------------------------------
const CUSTOM_ADD = "__add_custom__";

function populateSectors() {
  const sel = $("sector");
  sel.innerHTML = "";
  // ── Hidden native select (for event-dispatch compatibility) ──
  const presetGrp = document.createElement("optgroup");
  presetGrp.label = "Presets";
  for (const r of state.catalog.regions) {
    const o = document.createElement("option");
    o.value = r.key; o.textContent = r.name;
    presetGrp.appendChild(o);
  }
  sel.appendChild(presetGrp);
  if (state.customSectors.length) {
    const custGrp = document.createElement("optgroup");
    custGrp.label = "My Sectors";
    for (const cs of state.customSectors) {
      const o = document.createElement("option");
      o.value = cs.key; o.textContent = cs.name + "  ✎";
      custGrp.appendChild(o);
    }
    sel.appendChild(custGrp);
  }
  const actGrp = document.createElement("optgroup");
  actGrp.label = "Custom";
  const add = document.createElement("option");
  add.value = CUSTOM_ADD; add.textContent = "+ Draw custom sector…";
  actGrp.appendChild(add);
  sel.appendChild(actGrp);

  // ── Custom visible dropdown ──
  const menu = $("sector-menu");
  if (menu) {
    menu.innerHTML = "";
    // Presets group
    const presHdr = document.createElement("div");
    presHdr.className = "cs-group-lbl"; presHdr.textContent = "Presets";
    menu.appendChild(presHdr);
    for (const r of state.catalog.regions) {
      const item = document.createElement("div");
      item.className = "cs-item"; item.dataset.value = r.key;
      item.textContent = r.name; item.setAttribute("role", "option");
      menu.appendChild(item);
    }
    // Custom sectors group
    if (state.customSectors.length) {
      const sep = document.createElement("hr"); sep.className = "cs-sep";
      menu.appendChild(sep);
      const custHdr = document.createElement("div");
      custHdr.className = "cs-group-lbl"; custHdr.textContent = "My Sectors";
      menu.appendChild(custHdr);
      for (const cs of state.customSectors) {
        const item = document.createElement("div");
        item.className = "cs-item"; item.dataset.value = cs.key;
        item.textContent = cs.name + "  ✎"; item.setAttribute("role", "option");
        menu.appendChild(item);
      }
    }
    // Add-new
    const sep2 = document.createElement("hr"); sep2.className = "cs-sep";
    menu.appendChild(sep2);
    const addItem = document.createElement("div");
    addItem.className = "cs-item add-new"; addItem.dataset.value = CUSTOM_ADD;
    addItem.textContent = "+ Draw custom sector…"; addItem.setAttribute("role", "option");
    menu.appendChild(addItem);
  }

  // Restore selection if still valid
  const valid = new Set([
    ...state.catalog.regions.map(r => r.key),
    ...state.customSectors.map(c => c.key),
  ]);
  if (!valid.has(state.sector)) state.sector = "CONUS";
  sel.value = state.sector;
  _syncSectorBtn();
}

function _syncSectorBtn() {
  const lbl = $("sector-label");
  if (!lbl) return;
  // Find a display name for the current sector
  const r = state.catalog && state.catalog.regions.find(x => x.key === state.sector);
  const cs = findCustomSector(state.sector);
  lbl.textContent = cs ? cs.name : (r ? r.name : state.sector);
  // Update active highlight in menu
  const menu = $("sector-menu");
  if (!menu) return;
  for (const item of menu.querySelectorAll(".cs-item")) {
    item.classList.toggle("active", item.dataset.value === state.sector);
  }
}

function _openSectorMenu() {
  const menu = $("sector-menu");
  const btn  = $("sector-btn");
  if (!menu || !btn) return;
  menu.classList.remove("hidden");
  btn.setAttribute("aria-expanded", "true");
  // Scroll active item into view
  const active = menu.querySelector(".cs-item.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function _closeSectorMenu() {
  const menu = $("sector-menu");
  const btn  = $("sector-btn");
  if (!menu || !btn) return;
  menu.classList.add("hidden");
  btn.setAttribute("aria-expanded", "false");
}

function findCustomSector(key) {
  return state.customSectors.find(c => c.key === key);
}

function updateEditSectorButton() {
  const btn = $("btn-edit-sector");
  if (!btn) return;
  btn.disabled = !findCustomSector(state.sector);
}

function loadCustomSectors() {
  try {
    state.customSectors = JSON.parse(localStorage.getItem("refs-custom-sectors") || "[]");
  } catch { state.customSectors = []; }
}
function persistCustomSectors() {
  localStorage.setItem("refs-custom-sectors", JSON.stringify(state.customSectors));
}

function currentBbox() {
  const cs = findCustomSector(state.sector);
  if (cs) return cs.bbox;
  const r = state.catalog.regions.find(x => x.key === state.sector);
  if (!r) return null;
  return [r.lon[0], r.lat[0], r.lon[1], r.lat[1]];
}
function populatePalettes() {
  const sel = $("palette");
  sel.innerHTML = "";
  for (const p of state.catalog.palettes) {
    const o = document.createElement("option");
    o.value = p; o.textContent = p;
    sel.appendChild(o);
  }
  sel.value = state.palette;
}

function paintTopbar() {
  // Set date input + run dropdown to current state.
  const d = state.date;
  $("date").value = `${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)}`;
  $("run").value = String(state.run).padStart(2, "0");
}

// ----- Tabs ---------------------------------------------------------------
function buildTabs() {
  const wrap = $("tabs");
  wrap.innerHTML = "";
  for (const tab of Object.keys(state.catalog.tabs)) {
    const b = document.createElement("button");
    b.className = "tab"; b.textContent = tab; b.dataset.tab = tab;
    b.addEventListener("click", () => {
      // Clicking a tab just shows that tab's product menu — does NOT change
      // the currently-rendered product. The user picks a product themselves.
      state.tab = tab;
      paintTabs();
      buildProductList();
      writeURLState();
    });
    wrap.appendChild(b);
  }
}
function paintTabs() {
  for (const b of document.querySelectorAll(".tab"))
    b.classList.toggle("active", b.dataset.tab === state.tab);
}

// ----- Product list (right rail) -----------------------------------------
function buildProductList() {
  const root = $("product-list");
  root.innerHTML = "";
  const q = ($("prod-search").value || "").toLowerCase().trim();
  const cats = state.catalog.tabs[state.tab] || {};
  if (state.tab === FAV_TAB && !Object.keys(cats).length) {
    const hint = document.createElement("div");
    hint.className = "fav-hint";
    hint.textContent = "No pinned products yet — click the ☆ next to any " +
                       "product to pin it here.";
    root.appendChild(hint);
    return;
  }
  for (const [cat, items] of Object.entries(cats)) {
    const visible = items.filter(it =>
      !q || it.name.toLowerCase().includes(q) || cat.toLowerCase().includes(q));
    if (!visible.length) continue;
    const h = document.createElement("div");
    h.className = "prod-cat"; h.textContent = cat;
    root.appendChild(h);
    for (const it of visible) {
      const el = document.createElement("div");
      el.className = "prod-item";
      el.dataset.pid = it.pid;
      if (it.pid === state.pid) el.classList.add("active");
      const isFav = state.favorites && state.favorites.has(it.pid);
      el.innerHTML =
        `<span class="fav-star${isFav ? " faved" : ""}" ` +
        `title="${isFav ? "Unpin from" : "Pin to"} Favorites">` +
        `${isFav ? "★" : "☆"}</span>${it.name}` +
        `<span class="ftype-chip">${it.ftype}</span>`;
      el.title = it.spc_title || it.name;
      el.querySelector(".fav-star").addEventListener("click", (e) => {
        e.stopPropagation();
        toggleFavorite(it.pid);
      });
      el.addEventListener("click", async () => {
        const wasSpc = state.productSource === "spc_post";
        state.pid = it.pid;
        state.pidName = it.name;
        state.minFhr = Math.max(1, it.min_fhr || 0);   // F000 never exists
        state.fhrStride = it.fhr_stride || 1;
        state.productSource = it.source || "";
        // Auto-select palette by product class: SPC Ramp for probability
        // products, Default otherwise. Only re-applies when crossing the
        // prob/non-prob boundary, so a manual palette pick sticks while
        // browsing within the same class.
        const isProb = it.units === "%";
        if (isProb !== state._lastWasProb) {
          state.palette = isProb ? "SPC Ramp" : "Default";
          $("palette").value = state.palette;
          state._lastWasProb = isProb;
        }
        // If the current fhr is below the new product's min OR off-stride,
        // snap forward to the next stride-aligned hour ≥ min.
        const stride = Math.max(1, state.fhrStride);
        if (state.fhr < state.minFhr || (state.fhr % stride) !== 0) {
          let h = Math.max(state.minFhr, state.fhr);
          h = Math.ceil(h / stride) * stride;
          if (state.step > 1) h = Math.ceil(h / state.step) * state.step;
          state.fhr = Math.min(h, activeMaxFhour());
        }
        state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
        for (const x of document.querySelectorAll(".prod-item"))
          x.classList.toggle("active", x.dataset.pid === it.pid);
        // SPC products carry their own (sparse) availability; non-SPC products
        // leaving an SPC selection must restore the model cycle's availability.
        if (state.productSource === "spc_post") {
          updateSpcUiLock();
          await enforceSpcConstraints();   // force HREF + 00z/12z
          await applySpcAvailability();
        } else if (wasSpc) {
          state.sparseAvail = false;
          updateSpcUiLock();
          await refreshCycleStatus();
        }
        paintTimeline();
        paintMeta();
        writeURLState();
        loadFrame();
        startPreloadAllHours();
      });
      root.appendChild(el);
    }
  }
}

// ----- Timeline (F-hour strip) -------------------------------------------
function buildTimeline() {
  const hoursRow = $("tl-hours");
  const daysRow  = $("tl-days");
  hoursRow.innerHTML = "";
  daysRow.innerHTML  = "";

  const max = activeMaxFhour();
  const initUTC = parseCycleUTC(state.date, state.run);

  // Day markers + cells (stride by state.step)
  let lastDay = null;
  for (let h = 0; h <= max; h += state.step) {
    const cell = document.createElement("div");
    cell.className = "fhr-cell"; cell.dataset.h = h;
    cell.innerHTML = `<span>F${String(h).padStart(2,"0")}</span><span class="dot"></span>`;
    cell.addEventListener("click", () => {
      if (cell.classList.contains("missing")) return;
      if (cell.classList.contains("na")) return;
      state.followLatest = false;     // manual pick stops "follow latest"
      state.fhr = h;
      paintTimeline();
      paintMeta();
      writeURLState();
      loadFrame();
    });
    hoursRow.appendChild(cell);

    const valid = new Date(initUTC.getTime() + h * 3600000);
    const dayStr = valid.toUTCString().slice(0, 7);   // "Wed 20 "
    if (dayStr !== lastDay) {
      lastDay = dayStr;
      const m = document.createElement("div");
      m.className = "tl-day-mark";
      m.textContent = valid.toLocaleDateString("en-US",
        { weekday: "short", day: "2-digit", month: "2-digit", timeZone: "UTC" });
      // Position day marker relative to its hour cell. We do it after layout.
      m.dataset.h = h;
      // Click jumps to first available fhour of that day.
      m.addEventListener("click", () => {
        const startH = parseInt(m.dataset.h, 10);
        const have = state.availableFhours;
        for (let dh = 0; dh <= 23; dh++) {
          const cand = startH + dh;
          if (cand > activeMaxFhour()) break;
          if (!have.size || have.has(cand)) {
            state.fhr = cand;
            paintTimeline();
            paintMeta();
            writeURLState();
            loadFrame();
            return;
          }
        }
      });
      daysRow.appendChild(m);
    }
  }

  // After render, position day labels above their first hour cell
  requestAnimationFrame(positionDayMarks);
  paintTimeline();
}

function positionDayMarks() {
  const hoursRow = $("tl-hours");
  for (const m of document.querySelectorAll(".tl-day-mark")) {
    const h = parseInt(m.dataset.h, 10);
    const cell = hoursRow.querySelector(`.fhr-cell[data-h="${h}"]`);
    if (!cell) continue;
    const r1 = cell.getBoundingClientRect();
    const r0 = hoursRow.getBoundingClientRect();
    m.style.left = (r1.left - r0.left + 2) + "px";
  }
  positionPlayhead();
  positionLoopRange();
}
window.addEventListener("resize", positionDayMarks);

function paintTimeline() {
  const minFhr = state.minFhr || 0;
  const stride = Math.max(1, state.fhrStride || 1);
  for (const cell of document.querySelectorAll(".fhr-cell")) {
    const h = parseInt(cell.dataset.h, 10);
    // N/A if outside the product's valid window OR off the product's
    // accumulation stride (e.g. a 6-h QPF asked at F=07 doesn't exist).
    let na = h < minFhr || (stride > 1 && (h % stride) !== 0);
    const ok = state.availableFhours.has(h);
    const loaded = state.loadedFhours.has(h);
    // For SPC products, availableFhours is an EXACT set — any in-range hour
    // that isn't in it simply doesn't exist (e.g. the 24-h field is only
    // F036), so treat it as n/a rather than a "missing" gap/error.
    if (state.sparseAvail && !ok) na = true;
    cell.classList.toggle("active", h === state.fhr);
    cell.classList.toggle("na", na);
    cell.classList.toggle("available", !na && ok && !loaded);
    cell.classList.toggle("loaded", !na && loaded);
    cell.classList.toggle("missing",
      !na && !state.sparseAvail && state.availableFhours.size > 0 && !ok);
  }
  positionPlayhead();
  positionLoopRange();
  updateLoopInfo();
}

function cellRectForFhr(fhr) {
  const row = $("tl-hours");
  const cell = row.querySelector(`.fhr-cell[data-h="${fhr}"]`);
  if (!cell) return null;
  const rcell = cell.getBoundingClientRect();
  const rrow  = row.getBoundingClientRect();
  return {
    left: rcell.left - rrow.left,
    width: rcell.width,
    rowWidth: rrow.width,
  };
}

function positionPlayhead() {
  const ph = $("tl-playhead");
  if (!ph) return;
  const r = cellRectForFhr(state.fhr);
  if (!r) { ph.style.display = "none"; return; }
  ph.style.display = "block";
  ph.style.left = (r.left + r.width / 2 - 1) + "px";
}

function positionLoopRange() {
  const left  = $("tl-range-shade-left");
  const right = $("tl-range-shade-right");
  const hMin  = $("tl-range-handle-min");
  const hMax  = $("tl-range-handle-max");
  if (!left || !right || !hMin || !hMax) return;
  const max = activeMaxFhour();
  const customRange = (state.fmin > 0) || (state.fmax < max);
  if (!customRange) {
    left.style.display = right.style.display = "none";
    hMin.style.display = hMax.style.display = "none";
    return;
  }
  const rMin = cellRectForFhr(state.fmin);
  const rMax = cellRectForFhr(state.fmax);
  if (!rMin || !rMax) return;
  const startX = rMin.left;
  const endX   = rMax.left + rMax.width;
  left.style.display  = "block";
  right.style.display = "block";
  hMin.style.display  = "block";
  hMax.style.display  = "block";
  left.style.left  = "0";
  left.style.width = startX + "px";
  right.style.left = endX + "px";
  right.style.width = Math.max(0, rMin.rowWidth - endX) + "px";
  hMin.style.left = (startX - 4) + "px";
  hMax.style.left = (endX - 5) + "px";
}

function updateLoopInfo() {
  const info = $("tl-loop-info");
  if (!info) return;
  const max = activeMaxFhour();
  if (state.fmin === 0 && state.fmax === max) {
    info.textContent = "loop: full"; info.classList.remove("active");
  } else {
    info.textContent = `loop: F${String(state.fmin).padStart(2,"0")}–F${String(state.fmax).padStart(2,"0")}`;
    info.classList.add("active");
  }
}

function fhrFromClientX(clientX) {
  const row = $("tl-hours");
  const rrow = row.getBoundingClientRect();
  let best = 0, bestDist = Infinity;
  for (const cell of row.querySelectorAll(".fhr-cell")) {
    const rc = cell.getBoundingClientRect();
    const cx = rc.left + rc.width / 2;
    const d = Math.abs(cx - clientX);
    if (d < bestDist) { bestDist = d; best = parseInt(cell.dataset.h, 10); }
  }
  return best;
}

function setupLoopDrag() {
  const hMin = $("tl-range-handle-min");
  const hMax = $("tl-range-handle-max");
  if (!hMin || !hMax) return;
  let dragging = null;     // "min" | "max" | null
  const onDown = (which) => (e) => {
    e.preventDefault();
    dragging = which;
    document.body.style.userSelect = "none";
  };
  hMin.addEventListener("mousedown", onDown("min"));
  hMax.addEventListener("mousedown", onDown("max"));
  document.addEventListener("mousemove", e => {
    if (!dragging) return;
    const fhr = fhrFromClientX(e.clientX);
    if (dragging === "min")      state.fmin = Math.min(fhr, state.fmax - 1);
    else if (dragging === "max") state.fmax = Math.max(fhr, state.fmin + 1);
    positionLoopRange();
    updateLoopInfo();
  });
  document.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = null;
    document.body.style.userSelect = "";
    writeURLState();
  });
}

function resetLoopRange() {
  state.fmin = 0;
  state.fmax = activeMaxFhour();
  positionLoopRange();
  updateLoopInfo();
  writeURLState();
}

// ----- Meta strip --------------------------------------------------------
function paintMeta() {
  const init = parseCycleUTC(state.date, state.run);
  const valid = new Date(init.getTime() + state.fhr * 3600000);
  const fmtUTC = (d) => d.toUTCString().replace(" GMT", " UTC");
  const region = state.catalog.regions.find(r => r.key === state.sector);
  $("meta-info").innerHTML =
    `<span class="label">Product</span><span class="val prod">${state.pidName}</span>` +
    `<span class="sep">·</span>` +
    `<span class="label">Init</span><span class="val">${fmtUTC(init)}</span>` +
    `<span class="sep">·</span>` +
    `<span class="label">Valid</span><span class="val">${fmtUTC(valid)}</span>` +
    `<span class="sep">·</span>` +
    `<span class="label">F</span><span class="val">F${String(state.fhr).padStart(3,"0")}</span>` +
    `<span class="sep">·</span>` +
    `<span class="label">Sector</span><span class="val">${region ? region.name : state.sector}</span>` +
    `<span class="sep">·</span>` +
    `<span class="label">Palette</span><span class="val">${state.palette}</span>` +
    `<span class="sep">·</span>` +
    `<span class="model-chip model-chip-${state.model}">${state.model === "href" ? "HREF v3" : "REFS"}</span>`;
}

// ----- SPC calibrated-guidance availability ------------------------------
// SPC products have product-specific forecast-hour ranges that differ from
// the REFS/HREF cycle, so their timeline is driven by /api/spc-status rather
// than the model cycle-status. availableFhours becomes an exact sparse set.
async function applySpcAvailability() {
  state.sparseAvail = true;
  const pill = $("cycle-status-pill");
  pill.className = "pill"; pill.textContent = "SPC…";
  try {
    const r = await fetch(
      `/api/spc-status/${state.date}/${state.run}/${state.pid}`).then(x => x.json());
    state.availableFhours = new Set(r.available || []);
    const rz = String(state.run).padStart(2, "0");
    if (state.availableFhours.size) {
      pill.className = "pill complete"; pill.textContent = `SPC ${rz}Z`;
    } else {
      pill.className = "pill partial";
      pill.textContent = `no SPC data · ${rz}Z`;
    }
  } catch {
    state.availableFhours = new Set();
    pill.className = "pill partial"; pill.textContent = "SPC status?";
  }
  // Snap the current fhr onto the nearest available hour (≥ current if any).
  if (state.availableFhours.size && !state.availableFhours.has(state.fhr)) {
    const sorted = [...state.availableFhours].sort((a, b) => a - b);
    state.fhr = sorted.find(h => h >= state.fhr) ?? sorted[sorted.length - 1];
  }
  paintTimeline();
}

// SPC calibrated guidance only exists for the HREF model, 00z/12z cycles.
function isSpcActive() { return state.productSource === "spc_post"; }
function snapSpcRun(run) { return run >= 12 ? 12 : 0; }

function updateSpcUiLock() {
  const spc = isSpcActive();
  const ms = $("model-select");
  if (ms) {
    ms.disabled = spc;
    ms.title = spc ? "SPC guidance is HREF-only"
                   : "Switch between REFS and HREF to compare the two models";
  }
  const cb = $("btn-compare");
  if (cb) cb.disabled = spc;
  const runSel = $("run");
  if (runSel) for (const opt of runSel.options) {
    const rv = parseInt(opt.value, 10);
    opt.disabled = spc && rv !== 0 && rv !== 12;
  }
}

async function enforceSpcConstraints() {
  // Force HREF model.
  if (state.model !== "href") {
    state.model = "href";
    const ms = $("model-select"); if (ms) ms.value = "href";
    applyModelClass();
  }
  if (state.hrefCycles.length === 0) {
    try { state.hrefCycles = await fetch("/api/href-cycles").then(r => r.json()); }
    catch { /* leave empty; availability will report no data */ }
  }
  // Restrict run to 00z/12z, keeping the user's date where possible.
  let date = state.date;
  let run = (state.run === 0 || state.run === 12) ? state.run : snapSpcRun(state.run);
  if (!hrefCycleExists(date, run)) {
    const v = state.hrefCycles.find(c => c.run === 0 || c.run === 12);
    if (v) { date = v.date; run = v.run; }
  }
  state.date = date; state.run = run;
  state.fmax = Math.min(state.fmax, activeMaxFhour());
  paintTopbar();
  buildTimeline();
}

// ----- Cycle status (which fhours are published) -------------------------
async function refreshCycleStatus() {
  // SPC calibrated products bypass the model cycle-status entirely.
  if (state.productSource === "spc_post") {
    await applySpcAvailability();
    return;
  }
  const pill = $("cycle-status-pill");
  pill.className = "pill"; pill.textContent = "…";

  // HREF publishes all forecast hours at once — the cycle is either fully
  // up or not yet posted. No S3 scan needed; just check our known cycle list.
  // If the run isn't in our local list, re-fetch before concluding unposted
  // (the list is only fetched once on page load, so a new cycle that came
  // online after load would be missed until a re-fetch here).
  if (state.model === "href") {
    const maxFhr = state.catalog.href_max_fhour || 48;
    if (!hrefCycleExists(state.date, state.run)) {
      try {
        const fresh = await fetch("/api/href-cycles").then(r => r.json());
        state.hrefCycles = fresh;
      } catch (_) {}
    }
    const exists = hrefCycleExists(state.date, state.run);
    if (exists) {
      // HREF forecast products run F001..maxFhr — there is no F000 analysis
      // file (same as REFS), so build the set starting at 1.
      state.availableFhours = new Set(Array.from({length: maxFhr}, (_, i) => i + 1));
      pill.textContent = "Complete"; pill.classList.add("complete");
    } else {
      state.availableFhours = new Set();
      pill.textContent = "unposted"; pill.classList.add("partial");
    }
    paintTimeline();
    return;
  }

  try {
    const r = await fetch(`/api/cycle-status/${state.date}/${state.run}`)
                .then(r => r.json());
    const beforeCount = state.availableFhours.size;
    state.availableFhours = new Set(r.available);
    // "Live" = this cycle is the newest one AND it's still posting.
    const isLatest = state.cycles.length > 0 &&
                     state.cycles[0].date === state.date &&
                     state.cycles[0].run === state.run;
    if (r.complete) {
      pill.textContent = "Complete"; pill.classList.add("complete");
    } else if (isLatest && r.count > 0) {
      pill.textContent = `Run in Progress ${r.count}/${r.expected}`;
      pill.classList.add("live");
    } else if (r.count > 0) {
      pill.textContent = `${r.count}/${r.expected}`;
      pill.classList.add("partial");
    } else {
      pill.textContent = "unposted"; pill.classList.add("partial");
    }
    paintTimeline();
    // Newly-posted fhours? Re-trigger preload so they fill in.
    if (state.availableFhours.size > beforeCount) startPreloadAllHours();
  } catch (e) {
    pill.textContent = "status?";
  }
}

// ----- Tile URL + load ----------------------------------------------------
function _commonTileParams() {
  const qs = new URLSearchParams({
    palette: state.palette,
    theme: currentTheme(),
    v: state.build,
  });
  const cs = findCustomSector(state.sector);
  if (cs) {
    qs.set("bbox", cs.bbox.join(","));
    qs.set("sector_name", cs.name);
  } else {
    qs.set("region", state.sector);
  }
  if (state.showCounties) qs.set("counties", "1");
  if (state.showCities)   qs.set("cities", "1");
  return qs;
}

function tileUrl(fhr) {
  const qs = _commonTileParams();
  if (state.model && state.model !== "refs") qs.set("model", state.model);
  if (regionsEnabled) qs.set("regions", "1");
  return `/api/tile/${state.date}/${String(state.run).padStart(2,"0")}/${state.pid}/${fhr}.webp?${qs}`;
}

function gifUrl() {
  const qs = _commonTileParams();
  qs.set("fmin", state.fmin);
  qs.set("fmax", state.fmax);
  qs.set("step", state.step);
  qs.set("duration_ms", SPEED_TO_MS[state.speed] || 500);
  if (state.model && state.model !== "refs") qs.set("model", state.model);
  if (regionsEnabled) qs.set("regions", "1");
  return `/api/export/gif/${state.date}/${String(state.run).padStart(2,"0")}/${state.pid}.gif?${qs}`;
}

// ----- MRMS observed-reflectivity overlay ---------------------------------
// Pulls observed MRMS composite reflectivity from the IEM mtarchive at the
// forecast's valid time and renders line contours on top of the forecast
// tile. Hidden by default; the user toggles it on per session.
// MRMS display mode:
//   "off"      no overlay
//   "contours" line contours at 25/40/55 dBZ (cheap, see-through)
//   "filled"   full NWS palette translucent over REFS
//   "only"     full palette at 100% over a desaturated REFS (basemap ghost)
//
// Modes share the same /api/mrms backend; "filled" and "only" both fetch
// style=filled, "contours" fetches style=contours. The same cached tile is
// reused across multiple display modes, so flipping between filled<->only
// is instant after the first load.
let mrmsMode = "off";
let mrmsLoadToken = 0;
let mrmsProduct = "refc";   // refc | hsr | base
let mrmsLoadingTimer = null;

function mrmsBackendStyle(mode) {
  // Filled mode overlays REFS underneath (REFS provides basemap) →
  // transparent filled tile, no basemap baked in.
  // MRMS-only and Compare are geographically self-contained → tile
  // includes basemap layers identical to the REFS tile.
  if (mode === "only" || mode === "compare") return "filled_basemap";
  if (mode === "filled") return "filled";
  return "contours";
}

const MRMS_MODE_CYCLE = ["off", "contours", "filled", "only", "compare"];
function cycleMrmsMode() {
  const i = Math.max(0, MRMS_MODE_CYCLE.indexOf(mrmsMode));
  const next = MRMS_MODE_CYCLE[(i + 1) % MRMS_MODE_CYCLE.length];
  mrmsMode = next;
  const sel = $("mrms-mode");
  if (sel) sel.value = next;
  updateMrmsOverlay();
  setStatus(`MRMS overlay: ${({
    off:"off", contours:"contours", filled:"filled translucent",
    only:"MRMS only (swap)", compare:"side-by-side compare"
  })[next]}`);
}

function mrmsUrl(fhr, mode) {
  const qs = _commonTileParams();
  qs.set("product", mrmsProduct);
  qs.set("style", mrmsBackendStyle(mode));
  return `/api/mrms/${state.date}/${String(state.run).padStart(2,"0")}/${fhr}.webp?${qs}`;
}

// MRMS preload + progress pill, mirroring the REFS preload-pill model so
// the user sees concrete "loading X/Y" feedback while observation tiles
// stream in (these are slow — NOAA S3 + GRIB parse — and previously the
// only signal was a tiny pulsing "MRMS…" badge that's easy to miss).
//
// Preload is keyed by everything that invalidates the cache (mode,
// product, run, sector, theme); when the key changes we drop the
// loaded-set and kick a fresh sweep. Future frames are skipped (no obs
// exists yet) so the total reflects only fetchable hours.
const MRMS_PRELOAD_CONCURRENCY = 2;     // S3+GRIB is much slower than REFS tiles
let mrmsPreloadGen = 0;
let _mrmsPreloadKey = "";
// state.mrmsLoadedFhours is created lazily in startMrmsPreloadAllHours so
// it stays in sync with the rest of state.* lifecycle.

function _mrmsPreloadKeyNow() {
  const sec = (state.sectorBbox && state.sectorBbox.length === 4)
              ? state.sectorBbox.join(",")
              : (state.sectorName || state.region || "CONUS");
  return [mrmsMode, mrmsProduct, state.date,
          String(state.run).padStart(2,"0"), sec,
          state.theme || "dark"].join("|");
}

function setMrmsLoading(active) {
  // Per-frame load signal — kept around for callers that still flip it
  // when an individual MRMS request begins/finishes. The cumulative
  // preload pill (updateMrmsPill) is now the primary feedback, but
  // this still nudges the pill on when the user jumps to a frame
  // before preload reaches it.
  if (!active || mrmsMode === "off") return;
  const pill = $("mrms-pill");
  if (!pill) return;
  if (pill.style.display === "none" || !pill.textContent) {
    pill.style.display = "inline-flex";
    pill.textContent = "MRMS…";
  }
}

function updateMrmsPill(done, total) {
  const pill = $("mrms-pill");
  if (!pill) return;
  if (mrmsLoadingTimer) { clearTimeout(mrmsLoadingTimer); mrmsLoadingTimer = null; }
  if (mrmsMode === "off" || total === 0) {
    pill.style.display = "none";
    pill.textContent = "";
    return;
  }
  pill.style.display = "inline-flex";
  if (done >= total) {
    pill.textContent = `MRMS ${total}/${total} ✓`;
    // Flash the "done" state briefly so the user sees completion,
    // then hide. Cleared if a new preload starts in the meantime.
    mrmsLoadingTimer = setTimeout(() => {
      mrmsLoadingTimer = null;
      if (mrmsMode !== "off") {
        pill.style.display = "none";
        pill.textContent = "";
      }
    }, 1200);
  } else {
    pill.textContent = `MRMS ${done}/${total}`;
  }
}

function startMrmsPreloadAllHours() {
  const myGen = ++mrmsPreloadGen;
  if (mrmsMode === "off") {
    updateMrmsPill(0, 0);
    return;
  }
  if (!state.mrmsLoadedFhours) state.mrmsLoadedFhours = new Set();
  const minFhr = state.minFhr || 0;
  const step = Math.max(1, state.step || 1);
  const stride = Math.max(1, state.fhrStride || 1);
  const initMs = parseCycleUTC(state.date, String(state.run).padStart(2,"0")).getTime();
  const nowMs = Date.now();
  // Only past/present frames can have observations.
  const available = [...(state.availableFhours || [])]
    .filter(h => !state.mrmsLoadedFhours.has(h)
              && h >= minFhr
              && (stride === 1 || (h % stride) === 0)
              && (step   === 1 || (h % step)   === 0)
              && (initMs + h * 3600_000) <= nowMs)
    .sort((a, b) => Math.abs(a - state.fhr) - Math.abs(b - state.fhr));
  const visibleLoaded = [...state.mrmsLoadedFhours].filter(
    h => h >= minFhr && (step === 1 || (h % step) === 0)
  ).length;
  const total = available.length + visibleLoaded;
  let done = visibleLoaded;
  updateMrmsPill(done, total);
  if (!available.length) { updateMrmsPill(total, total); return; }

  let active = 0, i = 0;
  const next = () => {
    if (myGen !== mrmsPreloadGen || mrmsMode === "off") return;
    // Yield to the user's interactive renders, matching the REFS preload
    // backoff so MRMS preload doesn't starve the foreground frame.
    const sinceInteraction = Date.now() - lastInteractionTs;
    if (sinceInteraction < PRELOAD_INTERACTION_PAUSE_MS) {
      setTimeout(next, PRELOAD_INTERACTION_PAUSE_MS - sinceInteraction + 30);
      return;
    }
    while (active < MRMS_PRELOAD_CONCURRENCY && i < available.length) {
      const h = available[i++];
      active++;
      const img = new Image();
      const finish = () => {
        active--;
        if (myGen !== mrmsPreloadGen) return;
        state.mrmsLoadedFhours.add(h);
        done++;
        updateMrmsPill(done, total);
        next();
      };
      img.onload = finish;
      img.onerror = finish;
      img.src = mrmsUrl(h, mrmsMode);
    }
  };
  next();
}

function updateMrmsCaption() {
  const cap = $("mrms-caption");
  if (!cap) return;
  if (mrmsMode !== "only") {
    cap.classList.add("hidden");
    return;
  }
  // Approximate snapshot time = the forecast valid time rounded to 2-min.
  // We don't have the exact MRMS snap UTC client-side (that's a server
  // detail) — the *valid* minute is plenty for the caption.
  const initMs = parseCycleUTC(state.date, String(state.run).padStart(2,"0")).getTime();
  const validMs = initMs + state.fhr * 3600_000;
  const v = new Date(validMs);
  const hh = String(v.getUTCHours()).padStart(2,"0");
  const mm = String(v.getUTCMinutes()).padStart(2,"0");
  const dd = `${v.getUTCFullYear()}-${String(v.getUTCMonth()+1).padStart(2,"0")}-${String(v.getUTCDate()).padStart(2,"0")}`;
  const prodLabel = ({refc:"QC Composite",hsr:"SeamlessHSR",base:"QC Base Refl."})[mrmsProduct] || mrmsProduct;
  cap.textContent = `MRMS ${prodLabel} · ${dd} ${hh}:${mm} UTC`;
  cap.classList.remove("hidden");
}

function updateMrmsLegend() {
  const legend = $("mrms-legend");
  const contoursRow = $("mrms-legend-contours");
  const cbRow = $("mrms-legend-colorbar");
  if (!legend || !contoursRow || !cbRow) return;
  contoursRow.classList.remove("active");
  cbRow.classList.remove("active");
  if (mrmsMode === "off") {
    legend.classList.add("hidden");
    return;
  }
  if (mrmsMode === "contours") {
    contoursRow.classList.add("active");
  } else {
    // filled / only
    cbRow.classList.add("active");
  }
  legend.classList.remove("hidden");
}

function applyMrmsBodyClass() {
  document.body.classList.toggle("mrms-only",    mrmsMode === "only");
  document.body.classList.toggle("mrms-compare", mrmsMode === "compare");
}

function updateMrmsOverlay() {
  const el = $("mrms-overlay");
  if (!el) return;
  applyMrmsBodyClass();
  updateMrmsLegend();
  updateMrmsCaption();
  // Detect any cache-invalidating change (mode/product/date/run/sector/theme)
  // and kick a fresh MRMS preload. This is the single hook point — handlers
  // for those inputs already call updateMrmsOverlay so we catch them all.
  const key = _mrmsPreloadKeyNow();
  if (key !== _mrmsPreloadKey) {
    _mrmsPreloadKey = key;
    state.mrmsLoadedFhours = new Set();
    startMrmsPreloadAllHours();
  }
  el.classList.remove("mode-contours","mode-filled","mode-only");
  if (mrmsMode === "off") {
    el.classList.remove("visible");
    el.removeAttribute("src");
    updateMrmsPill(0, 0);
    return;
  }
  el.classList.add(`mode-${mrmsMode}`);
  const url = mrmsUrl(state.fhr, mrmsMode);
  const myToken = ++mrmsLoadToken;
  setMrmsLoading(true);
  const probe = new Image();
  probe.onload = () => {
    if (myToken !== mrmsLoadToken || mrmsMode === "off") return;
    el.src = url;
    el.classList.add("visible");
    setMrmsLoading(false);
  };
  probe.onerror = () => {
    if (myToken !== mrmsLoadToken) return;
    el.classList.remove("visible");
    setMrmsLoading(false);
  };
  probe.src = url;
}

// ----- LSR verification overlay --------------------------------------------
// Renders observed storm reports from IEM (tornado/hail/wind) on top of the
// forecast image at the matching valid time. Hidden by default; the user
// toggles it on per session. Future frames render an empty overlay since
// no obs exist yet.
let lsrEnabled = false;
let lsrWindow = 30;
let lsrLoadToken = 0;
// Flood / Flash Flood is a separate layer (different IEM type set) with
// its own independent toggle, so it stacks cleanly on top of severe LSRs.
let floodLsrEnabled = false;
let floodLsrLoadToken = 0;

function lsrUrl(fhr, types) {
  const qs = _commonTileParams();
  qs.set("window_min", String(lsrWindow));
  if (types) qs.set("types", types);
  return `/api/lsr/${state.date}/${String(state.run).padStart(2,"0")}/${fhr}.webp?${qs}`;
}

function updateLsrOverlay() {
  const el = $("lsr-overlay");
  const legend = $("lsr-legend");
  if (!el || !legend) return;
  if (!lsrEnabled) {
    el.classList.remove("visible");
    el.removeAttribute("src");
    legend.classList.add("hidden");
    return;
  }
  legend.classList.remove("hidden");
  // No `types` arg → server uses its DEFAULT_TYPES (severe only).
  const url = lsrUrl(state.fhr);
  const myToken = ++lsrLoadToken;
  const probe = new Image();
  probe.onload = () => {
    if (myToken !== lsrLoadToken || !lsrEnabled) return;
    el.src = url;
    el.classList.add("visible");
  };
  probe.onerror = () => {
    if (myToken !== lsrLoadToken) return;
    el.classList.remove("visible");
  };
  probe.src = url;
}

function updateFloodLsrOverlay() {
  const el = $("flood-lsr-overlay");
  const legend = $("flood-lsr-legend");
  if (!el || !legend) return;
  if (!floodLsrEnabled) {
    el.classList.remove("visible");
    el.removeAttribute("src");
    legend.classList.add("hidden");
    return;
  }
  legend.classList.remove("hidden");
  // Flood + Flash Flood single combined layer.
  const url = lsrUrl(state.fhr, "F,E");
  const myToken = ++floodLsrLoadToken;
  const probe = new Image();
  probe.onload = () => {
    if (myToken !== floodLsrLoadToken || !floodLsrEnabled) return;
    el.src = url;
    el.classList.add("visible");
  };
  probe.onerror = () => {
    if (myToken !== floodLsrLoadToken) return;
    el.classList.remove("visible");
  };
  probe.src = url;
}

// ----- WxWorks region-boundary overlay -------------------------------------
// Regions are baked into the tile server-side (zorder 10, below city labels
// at zorder 11-12).  Toggling adds/removes ?regions=1 from the tile URL so
// the server renders a distinct cached tile with boundaries drawn.
let regionsEnabled = false;

function updateRegionsOverlay() {
  // Regions are now baked into the forecast tile — just reload the frame so
  // the new tile URL (with or without &regions=1) takes effect.
  loadFrame();
}

// ----- REFS forecast contours on MRMS panel --------------------------------
// Transparent dashed-line tile drawn from the REFS comp-ref PMM field at
// 20/35/50 dBZ. Independent of the active product so it works even when
// the left panel shows UH, QPF, etc. The frontend positions it over the
// right half in compare mode; over the full image otherwise (so it can
// double as a "forecast contour overlay on your current view" toggle).
let refsContourEnabled = false;
let refsContourLoadToken = 0;

function refsContourUrl(fhr) {
  const qs = _commonTileParams();
  return `/api/refs_contours/${state.date}/${String(state.run).padStart(2,"0")}/${fhr}.webp?${qs}`;
}

function updateRefsContourOverlay() {
  const el = $("refs-contour-overlay");
  const legend = $("refs-contour-legend");
  if (!el || !legend) return;
  if (!refsContourEnabled) {
    el.classList.remove("visible");
    el.removeAttribute("src");
    legend.classList.add("hidden");
    return;
  }
  legend.classList.remove("hidden");
  const url = refsContourUrl(state.fhr);
  const myToken = ++refsContourLoadToken;
  const probe = new Image();
  probe.onload = () => {
    if (myToken !== refsContourLoadToken || !refsContourEnabled) return;
    el.src = url;
    el.classList.add("visible");
  };
  probe.onerror = () => {
    if (myToken !== refsContourLoadToken) return;
    el.classList.remove("visible");
  };
  probe.src = url;
}

// ----- Frame render progress -------------------------------------------
// A single tile is one <img> request and the time is dominated by the
// server render (not the download), so there are no native progress events
// to read. Instead we drive an *estimated* bar: it eases toward ~94% over the
// expected render time (a rolling average of recent renders) and snaps to
// 100% the instant the image actually loads. Cached frames (sub-50 ms) are
// excluded from the estimate so the bar stays calibrated to real renders.
const _renderDurations = [];
function _recordRenderMs(ms) {
  if (ms > 50 && ms < 30000) {
    _renderDurations.push(ms);
    if (_renderDurations.length > 8) _renderDurations.shift();
  }
}
function _estRenderMs() {
  if (!_renderDurations.length) return 1800;
  const avg = _renderDurations.reduce((a, b) => a + b, 0) / _renderDurations.length;
  return Math.max(500, Math.min(8000, avg));
}
function frameProgressStart(box) {
  if (!box) return;
  const fill = box.querySelector(".fl-fill");
  const pct  = box.querySelector(".fl-pct");
  if (!fill || !pct) return;
  if (box._fpRaf) cancelAnimationFrame(box._fpRaf);
  const tau = _estRenderMs() * 0.55;     // time constant of the easing curve
  const start = performance.now();
  box._fpStart = start;
  fill.style.transition = "none";
  fill.style.width = "0%";
  pct.textContent = "0%";
  const tick = () => {
    const t = performance.now() - start;
    let frac = 1 - Math.exp(-t / tau);   // asymptotic; never reaches 1 on its own
    if (frac > 0.94) frac = 0.94;
    fill.style.width = (frac * 100).toFixed(1) + "%";
    pct.textContent = Math.round(frac * 100) + "%";
    box._fpRaf = requestAnimationFrame(tick);
  };
  box._fpRaf = requestAnimationFrame(tick);
}
function frameProgressDone(box) {
  if (!box) return;
  if (box._fpRaf) { cancelAnimationFrame(box._fpRaf); box._fpRaf = null; }
  if (box._fpStart) { _recordRenderMs(performance.now() - box._fpStart); box._fpStart = 0; }
  const fill = box.querySelector(".fl-fill");
  const pct  = box.querySelector(".fl-pct");
  if (fill) { fill.style.transition = "width 0.12s ease"; fill.style.width = "100%"; }
  if (pct)  pct.textContent = "100%";
}
function frameProgressStop(box) {        // error / cancel: just stop animating
  if (!box) return;
  if (box._fpRaf) { cancelAnimationFrame(box._fpRaf); box._fpRaf = null; }
  box._fpStart = 0;
}

function loadFrame() {
  noteInteraction();
  paintMeta();
  // Refresh verification overlays whenever the underlying frame changes.
  updateMrmsOverlay();
  updateLsrOverlay();
  updateFloodLsrOverlay();
  updateRefsContourOverlay();
  updatePaintballOverlay();
  // Regions are baked into tileUrl() — no separate overlay call needed.
  const url = tileUrl(state.fhr);
  const fhour = state.fhr;
  const myToken = ++state.pendingLoad;
  setStatus(`Rendering ${state.pidName} F${String(fhour).padStart(3,"0")}…`);
  $("frame-loading").classList.remove("hidden");
  $("frame-error").classList.add("hidden");
  frameProgressStart($("frame-loading"));

  const img = new Image();
  img.onload = () => {
    if (myToken !== state.pendingLoad) return;
    FRAME().src = url;
    frameProgressDone($("frame-loading"));
    $("frame-loading").classList.add("hidden");
    state.loadedFhours.add(fhour);
    paintTimeline();
    setStatus(`${state.pidName} · F${String(fhour).padStart(3,"0")} · ${spcTitleForPid(state.pid)}`);
  };
  img.onerror = () => {
    if (myToken !== state.pendingLoad) return;
    frameProgressStop($("frame-loading"));
    $("frame-loading").classList.add("hidden");
    const err = $("frame-error");
    err.textContent = `Unavailable: ${state.pidName} F${String(fhour).padStart(3,"0")}`;
    err.classList.remove("hidden");
    setStatus(`Unavailable: ${state.pidName} F${String(fhour).padStart(3,"0")}`);
  };
  img.src = url;
  loadCompareFrame();
}

function startPreloadAllHours() {
  const myGen = ++preloadGen;
  // Whenever REFS preload restarts (typically because a cache-invalidating
  // input changed), the MRMS layer's loaded-set is also stale — drop it
  // and rerun the MRMS sweep so the pill counts match the new frame set.
  // Cheap when MRMS is off: startMrmsPreloadAllHours short-circuits.
  state.mrmsLoadedFhours = new Set();
  startMrmsPreloadAllHours();
  const minFhr = state.minFhr || 0;
  const step = Math.max(1, state.step || 1);
  const stride = Math.max(1, state.fhrStride || 1);
  // Filter to (a) posted, (b) not loaded, (c) within product's valid window,
  // (d) aligned to the product's accumulation stride, (e) aligned to the
  // current timeline stride. Sorted by distance from the current frame so
  // adjacent fhrs render first — feels responsive while scrubbing.
  const available = [...state.availableFhours]
    .filter(h => !state.loadedFhours.has(h)
                 && h >= minFhr
                 && (stride === 1 || (h % stride) === 0)
                 && (step === 1 || (h % step) === 0))
    .sort((a,b) => Math.abs(a - state.fhr) - Math.abs(b - state.fhr));
  // Loading pill should count only step-aligned, in-window hours so it
  // matches the visible timeline cells (otherwise the denominator looks
  // wrong after switching to step=3 or to a 24-h accum product).
  const visibleLoaded = [...state.loadedFhours].filter(
    h => h >= minFhr && (step === 1 || (h % step) === 0)
  ).length;
  const total = available.length + visibleLoaded;
  let done = visibleLoaded;
  updatePreloadPill(done, total);
  if (!available.length) { updatePreloadPill(total, total); return; }
  let active = 0, i = 0;
  const next = () => {
    if (myGen !== preloadGen) return;
    // If the user just clicked/scrubbed, defer kicking off NEW preloads for
    // a short window so the interactive render gets the server slot. Already-
    // in-flight preloads continue.
    const sinceInteraction = Date.now() - lastInteractionTs;
    if (sinceInteraction < PRELOAD_INTERACTION_PAUSE_MS) {
      setTimeout(next, PRELOAD_INTERACTION_PAUSE_MS - sinceInteraction + 30);
      return;
    }
    while (active < PRELOAD_CONCURRENCY && i < available.length) {
      const h = available[i++];
      active++;
      const img = new Image();
      const finish = () => {
        active--;
        if (myGen !== preloadGen) return;
        state.loadedFhours.add(h);
        done++;
        paintTimeline();
        updatePreloadPill(done, total);
        next();
      };
      img.onload = finish;
      img.onerror = finish;
      img.src = tileUrl(h);
    }
  };
  next();
  startComparePreloadAllHours();
}

function updatePreloadPill(done, total) {
  const pill = document.getElementById("preload-pill");
  if (!pill) return;
  if (total === 0 || done >= total) {
    pill.className = "pill";
    pill.textContent = "";
    pill.style.display = "none";
  } else {
    // Label/style by the active source so the pill matches what's loading.
    let label, cls;
    if (state.productSource === "spc_post") { label = "SPC"; cls = "href-loading"; }
    else if (state.model === "href")        { label = "HREF"; cls = "href-loading"; }
    else                                     { label = "REFS"; cls = "refs-loading"; }
    pill.className = "pill " + cls;
    pill.style.display = "";
    pill.textContent = `${label} ${done}/${total}`;
  }
}

// ----- Playback ----------------------------------------------------------
function togglePlay() {
  state.playing = !state.playing;
  $("btn-play").textContent = state.playing ? "⏸" : "▶";
  if (state.playing) {
    const tick = () => {
      step(+1);
      if (state.playing)
        state.playTimer = setTimeout(tick, SPEED_TO_MS[state.speed]);
    };
    tick();
  } else if (state.playTimer) {
    clearTimeout(state.playTimer);
    state.playTimer = null;
  }
}

function step(delta) {
  // Manual frame nav (incl. play) stops "follow latest" tracking.
  state.followLatest = false;
  // delta: ±1 in "cells", which translates to ±state.step hours.
  const stride = state.step * (delta > 0 ? 1 : -1);
  const have = state.availableFhours.size ? state.availableFhours : null;
  const minFhr = state.minFhr || 0;
  const fhrStride = Math.max(1, state.fhrStride || 1);
  const lo = state.fmin, hi = state.fmax;
  const span = hi - lo + 1;
  let h = state.fhr;
  if (h < lo || h > hi) h = lo;
  for (let tries = 0; tries < span + 2; tries++) {
    h = lo + (((h - lo) + stride + span) % span);
    if (h < minFhr) continue;          // n/a for this product
    if (fhrStride > 1 && (h % fhrStride) !== 0) continue;   // off-stride
    if (!have || have.has(h)) {
      state.fhr = h;
      paintTimeline();
      paintMeta();
      writeURLState();
      loadFrame();
      return;
    }
  }
}

function changeSpeed(delta) {
  state.speed = Math.max(1, Math.min(5, state.speed + delta));
  $("speed-label").textContent = String(state.speed);
}

// ----- Cycle switching ----------------------------------------------------
async function applyCycle(date, run) {
  state.date = date; state.run = run;
  state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
  state.availableFhours = new Set();
  paintTopbar();
  buildTimeline();
  writeURLState();
  await refreshCycleStatus();
  loadFrame();
  startPreloadAllHours();
}

// Step ±6 hours between REFS runs (00 / 06 / 12 / 18 UTC).
function stepRun(deltaHours) {
  state.followLatest = false;        // manual run step stops "follow latest"
  // SPC guidance only has 00z/12z cycles — step a full 12 h between them.
  if (isSpcActive()) deltaHours = deltaHours > 0 ? 12 : -12;
  const runs = state.catalog.runs || [0, 6, 12, 18];
  // Build a full hour timeline anchored on state.(date,run) and add delta.
  const init = parseCycleUTC(state.date, state.run);
  const next = new Date(init.getTime() + deltaHours * 3600000);
  const y = next.getUTCFullYear();
  const m = String(next.getUTCMonth() + 1).padStart(2, "0");
  const d = String(next.getUTCDate()).padStart(2, "0");
  const newDate = `${y}${m}${d}`;
  const newRun  = next.getUTCHours();
  if (!runs.includes(newRun)) return;          // safety: stick to defined cycles
  applyCycle(newDate, newRun);
}

// Newest posted forecast hour that is valid for the current product
// (respects min_fhr, accumulation stride, and the timeline step). null if none.
function latestAvailableFhr() {
  const cand = [...state.availableFhours].filter(isFhrValid);
  return cand.length ? Math.max(...cand) : null;
}

// Snap the view to the newest available forecast hour and render it.
function goToLatestFhr() {
  const h = latestAvailableFhr();
  if (h === null || h === state.fhr) return;
  state.fhr = h;
  paintTimeline();
  paintMeta();
  writeURLState();
  loadFrame();
}

async function jumpToLatest() {
  try {
    // Pick the newest cycle for the active model (REFS vs HREF differ).
    let cyc;
    if (state.model === "href") {
      if (state.hrefCycles.length === 0) {
        try { state.hrefCycles = await fetch("/api/href-cycles").then(r => r.json()); }
        catch { /* fall through */ }
      }
      cyc = state.hrefCycles[0];
    } else {
      cyc = await fetch("/api/latest").then(r => r.json());
    }
    if (!cyc) { setStatus("No recent run available."); return; }
    state.followLatest = true;        // keep tracking the newest fhr as it posts
    if (cyc.date !== state.date || cyc.run !== state.run) {
      await applyCycle(cyc.date, cyc.run);   // refreshes availability for the run
    } else {
      await refreshCycleStatus();
    }
    goToLatestFhr();
    setStatus(`Latest: ${cyc.label || (cyc.date + " " + String(cyc.run).padStart(2,"0") + "Z")} · F${String(state.fhr).padStart(3,"0")}`);
  } catch (e) {
    setStatus("No recent run available.");
  }
}

// ----- Compare mode (split and swipe) ------------------------------------

let comparePending = 0;

// --- Helpers for run-over-run ---
function getPriorRun() {
  const pool = state.model === "href" ? state.hrefCycles : state.cycles;
  const idx = pool.findIndex(c => c.date === state.date && c.run === state.run);
  if (idx > 0) return pool[idx - 1];
  if (idx === -1 && pool.length > 1) return pool[1];
  return pool.length > 1 ? pool[1] : null;
}

function runDiffHours(d1, r1, d2, r2) {
  const ms = (d, r) => Date.UTC(+d.slice(0,4), +d.slice(4,6)-1, +d.slice(6,8), r);
  return (ms(d1, r1) - ms(d2, r2)) / 3600000;
}

function compareTileUrl(fhr) {
  if (state.compareType === "run_prev") {
    const prior = getPriorRun();
    if (!prior) return null;
    const delta = runDiffHours(state.date, state.run, prior.date, prior.run);
    const fhrB = fhr + delta;
    if (fhrB < 0 || fhrB > activeMaxFhour()) return null;
    const qs = _commonTileParams();
    if (state.model !== "refs") qs.set("model", state.model);
    return `/api/tile/${prior.date}/${String(prior.run).padStart(2,"0")}/${state.pid}/${fhrB}.webp?${qs}`;
  }
  const qs = _commonTileParams();
  qs.set("model", "href");
  return `/api/tile/${state.date}/${String(state.run).padStart(2,"0")}/${state.pid}/${fhr}.webp?${qs}`;
}

function applyCompareStyle() {
  const isSwipe = state.compareMode && state.compareStyle === "swipe";
  const isSplit = state.compareMode && state.compareStyle === "split";
  document.body.classList.toggle("compare-swipe",  isSwipe);
  document.body.classList.toggle("compare-active", isSplit);

  const panel = $("compare-panel");
  if (panel) panel.classList.toggle("hidden", !isSplit);

  const styleBtn = $("btn-compare-style");
  if (styleBtn) {
    styleBtn.classList.toggle("hidden", !state.compareMode);
    styleBtn.textContent = state.compareStyle === "swipe" ? "⊞ Split" : "⇸ Swipe";
    styleBtn.classList.toggle("active", state.compareStyle === "swipe");
  }

  // Swipe labels
  const ll = $("swipe-label-left"), lr = $("swipe-label-right");
  if (ll) ll.classList.toggle("hidden", !isSwipe);
  if (lr) {
    lr.classList.toggle("hidden", !isSwipe);
    lr.textContent = _compareRightLabel();
  }

  // Right-panel header chip
  const chip = $("compare-right-chip");
  if (chip) {
    chip.textContent = _compareRightLabel();
    chip.className = "model-chip " + (state.compareType === "run_prev" ? "model-chip-refs" : "model-chip-href");
  }

  // Swipe pct CSS var
  const fs = $("frame-stack");
  if (fs) fs.style.setProperty("--swipe-pct", state.swipePct + "%");

  // Sync compare-type select
  const ct = $("compare-type");
  if (ct) ct.value = state.compareType;
}

function _compareRightLabel() {
  if (state.compareType === "run_prev") {
    const p = getPriorRun();
    return p ? `${p.date.slice(4,6)}/${p.date.slice(6,8)} ${String(p.run).padStart(2,"0")}z` : "Prior Run";
  }
  return "HREF v3";
}

function loadCompareFrame() {
  if (!state.compareMode) return;
  const fhr = state.fhr;
  const url = compareTileUrl(fhr);
  const isSwipe = state.compareStyle === "swipe";
  const targetEl = isSwipe ? $("frame-swipe") : $("frame-b");
  const loadingEl = isSwipe ? null : $("frame-loading-b");
  const errorEl   = isSwipe ? null : $("frame-error-b");
  if (!url) { if (targetEl) targetEl.src = ""; return; }
  const myToken = ++comparePending;
  if (loadingEl) { loadingEl.classList.remove("hidden"); frameProgressStart(loadingEl); }
  if (errorEl)   errorEl.classList.add("hidden");
  const img = new Image();
  img.onload = () => {
    if (myToken !== comparePending) return;
    if (targetEl) targetEl.src = url;
    if (loadingEl) { frameProgressDone(loadingEl); loadingEl.classList.add("hidden"); }
  };
  img.onerror = () => {
    if (myToken !== comparePending) return;
    if (loadingEl) { frameProgressStop(loadingEl); loadingEl.classList.add("hidden"); }
    if (errorEl) { errorEl.textContent = `Unavailable F${String(fhr).padStart(3,"0")}`; errorEl.classList.remove("hidden"); }
  };
  img.src = url;
}

function toggleCompare() {
  state.compareMode = !state.compareMode;
  const btn = $("btn-compare");
  const modelLabel = document.querySelector(".model-label");
  if (state.compareMode) {
    btn.classList.add("active");
    if (state.model !== "refs") {
      state.model = "refs";
      applyModelClass();
      const ms = document.querySelector("#model-select");
      if (ms) ms.value = "refs";
      loadFrame();
    } else {
      loadCompareFrame();
    }
    if (modelLabel) { modelLabel.style.opacity = "0.4"; modelLabel.querySelector("select").disabled = true; }
    applyCompareStyle();
    startComparePreloadAllHours();
  } else {
    btn.classList.remove("active");
    if (modelLabel) { modelLabel.style.opacity = ""; modelLabel.querySelector("select").disabled = false; }
    ++comparePending;
    const fw = $("frame-swipe"); if (fw) fw.src = "";
    state.compareLoadedFhours = new Set();
    ++comparePreloadGen;
    updateComparePreloadPill(0, 0);
    applyCompareStyle();
  }
}

function wireSwipeHandle() {
  const handle = $("swipe-handle");
  if (!handle) return;
  let dragging = false;
  const move = (clientX) => {
    const fs = $("frame-stack");
    if (!fs) return;
    const rect = fs.getBoundingClientRect();
    const pct = Math.max(5, Math.min(95, ((clientX - rect.left) / rect.width) * 100));
    state.swipePct = pct;
    fs.style.setProperty("--swipe-pct", pct + "%");
  };
  handle.addEventListener("mousedown", e => { if (state.compareMode && state.compareStyle === "swipe") { dragging = true; e.preventDefault(); } });
  document.addEventListener("mousemove", e => { if (dragging) move(e.clientX); });
  document.addEventListener("mouseup",   () => { dragging = false; });
  handle.addEventListener("touchstart",  e => { if (state.compareMode && state.compareStyle === "swipe") { dragging = true; e.preventDefault(); } }, {passive: false});
  document.addEventListener("touchmove", e => { if (dragging) move(e.touches[0].clientX); }, {passive: false});
  document.addEventListener("touchend",  () => { dragging = false; });
}

// ----- Paintball member-contour overlay ----------------------------------
let paintballLoadToken = 0;
let paintballEnabled   = false;

function paintballOverlayUrl(fhr) {
  const [pbVar, pbThresh] = (state.paintballVar || "refc|40").split("|");
  // v= busts the browser cache on deploys — overlays are served immutable,
  // so without it a client that ever cached a blank/buggy overlay keeps it.
  const qs = new URLSearchParams({ region: state.sector, theme: currentTheme(), var: pbVar, thresh: pbThresh, v: state.build });
  const cs = state.customSectors.find(s => s.key === state.sector);
  if (cs) { qs.set("bbox", cs.bbox.join(",")); qs.set("sector_name", cs.name || ""); }
  return `/api/paintball-overlay/${state.date}/${String(state.run).padStart(2,"0")}/${fhr}.webp?${qs}`;
}

function updatePaintballOverlay() {
  const el     = $("paintball-overlay");
  const legend = $("paintball-legend");
  if (!el) return;
  if (!paintballEnabled) {
    el.classList.remove("visible");
    if (legend) legend.classList.add("hidden");
    return;
  }
  if (legend) legend.classList.remove("hidden");
  const url = paintballOverlayUrl(state.fhr);
  const myToken = ++paintballLoadToken;
  const probe = new Image();
  probe.onload = () => {
    if (myToken !== paintballLoadToken || !paintballEnabled) return;
    el.src = url; el.classList.add("visible");
  };
  probe.onerror = () => { if (myToken !== paintballLoadToken) return; el.classList.remove("visible"); };
  probe.src = url;
}

// ----- Dual-model preload (HREF tiles in compare mode) -------------------
let comparePreloadGen = 0;

function updateComparePreloadPill(done, total) {
  const pill = $("compare-preload-pill");
  if (!pill) return;
  if (!state.compareMode || total === 0 || done >= total) {
    pill.style.display = "none"; pill.textContent = "";
  } else {
    pill.className = "pill href-loading";
    pill.style.display = "";
    pill.textContent = `HREF ${done}/${total}`;
  }
}

function startComparePreloadAllHours() {
  if (!state.compareMode || state.compareType !== "href") { updateComparePreloadPill(0,0); return; }
  const myGen = ++comparePreloadGen;
  const step = Math.max(1, state.step || 1);
  const hrefMax = state.catalog.href_max_fhour || 48;
  const available = [...state.availableFhours]
    .filter(h => !state.compareLoadedFhours.has(h) && h <= hrefMax && (step === 1 || (h % step) === 0))
    .sort((a, b) => Math.abs(a - state.fhr) - Math.abs(b - state.fhr));
  const visLoaded = [...state.compareLoadedFhours].filter(h => step === 1 || (h % step) === 0).length;
  const total = available.length + visLoaded;
  let done = visLoaded;
  updateComparePreloadPill(done, total);
  if (!available.length) { updateComparePreloadPill(total, total); return; }
  let active = 0, i = 0;
  const next = () => {
    if (myGen !== comparePreloadGen) return;
    const sinceInteraction = Date.now() - lastInteractionTs;
    if (sinceInteraction < PRELOAD_INTERACTION_PAUSE_MS) { setTimeout(next, PRELOAD_INTERACTION_PAUSE_MS - sinceInteraction + 30); return; }
    while (active < PRELOAD_CONCURRENCY && i < available.length) {
      const h = available[i++];
      active++;
      const img = new Image();
      const finish = () => {
        active--;
        if (myGen !== comparePreloadGen) return;
        state.compareLoadedFhours.add(h);
        done++;
        updateComparePreloadPill(done, total);
        next();
      };
      img.onload = finish; img.onerror = finish;
      img.src = compareTileUrl(h) || "";
    }
  };
  next();
}

// ----- Event wiring -------------------------------------------------------
function wireEvents() {
  $("date").addEventListener("change", e => {
    state.followLatest = false;       // manual date change stops "follow latest"
    const d = e.target.value.replace(/-/g,"");
    applyCycle(d, state.run);
  });
  $("run").addEventListener("change", e => {
    state.followLatest = false;       // manual cycle change stops "follow latest"
    let r = parseInt(e.target.value, 10);
    if (isSpcActive() && r !== 0 && r !== 12) r = snapSpcRun(r);  // SPC: 00z/12z only
    applyCycle(state.date, r);
  });
  $("sector").addEventListener("change", e => {
    const v = e.target.value;
    if (v === CUSTOM_ADD) {
      e.target.value = state.sector;     // revert visible value
      // Draw-on-map is the primary way to define a new sector — dragging a
      // box on the map yields a far better domain than typing lat/lon by hand.
      // The draw completion pre-fills the naming modal (with editable corners)
      // for a final tweak + save.
      startDrawMode();
      return;
    }
    state.sector = v;
    _syncSectorBtn();
    state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
    paintMeta();
    updateEditSectorButton();
    writeURLState();
    loadFrame();
    startPreloadAllHours();
  });

  // ── Custom sector dropdown open/close ──
  if ($("sector-btn")) {
    $("sector-btn").addEventListener("click", e => {
      e.preventDefault();
      e.stopPropagation();
      const menu = $("sector-menu");
      if (menu.classList.contains("hidden")) _openSectorMenu();
      else _closeSectorMenu();
    });
  }
  if ($("sector-menu")) {
    $("sector-menu").addEventListener("click", e => {
      const item = e.target.closest(".cs-item");
      if (!item) return;
      // The enclosing <label> auto-associates with #sector-btn (its first
      // labelable descendant), so a click on any item would otherwise fire a
      // synthetic click on the trigger button and reopen the just-closed menu.
      // Cancel that default action (and stop bubbling) so selection closes.
      e.preventDefault();
      e.stopPropagation();
      _closeSectorMenu();
      const v = item.dataset.value;
      // Dispatch through the hidden native select so existing logic runs
      $("sector").value = v;
      $("sector").dispatchEvent(new Event("change"));
    });
  }
  // Close menu when clicking outside
  document.addEventListener("click", e => {
    const wrap = $("sector-wrap");
    if (wrap && !wrap.contains(e.target)) _closeSectorMenu();
  });
  // Close on Escape
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") _closeSectorMenu();
  });

  $("btn-edit-sector").addEventListener("click", () => {
    const cs = findCustomSector(state.sector);
    if (cs) openCustomSectorModal(cs);
  });
  $("btn-draw-sector").addEventListener("click", startDrawMode);
  wireCustomSectorModal();
  $("step").addEventListener("change", e => {
    state.step = parseInt(e.target.value, 10) || 1;
    // Snap fhr to the new stride (respecting the product's accumulation stride)
    const stride = Math.max(state.step, state.fhrStride || 1);
    state.fhr = Math.round(state.fhr / stride) * stride;
    if (state.fhr < (state.minFhr || 0)) {
      let h = state.minFhr || 0;
      h = Math.ceil(h / stride) * stride;
      state.fhr = Math.min(h, activeMaxFhour());
    }
    if (state.fhr > activeMaxFhour()) state.fhr = activeMaxFhour();
    buildTimeline();
    paintMeta();
    writeURLState();
    // Re-trigger preload so the "Loading X/Y" pill reflects the new stride
    // and we stop pulling in fhours that aren't on the timeline anymore.
    startPreloadAllHours();
  });
  $("btn-share").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      showToast("Link copied to clipboard");
    } catch {
      showToast("Couldn't access clipboard — copy from address bar");
    }
  });
  $("btn-export").addEventListener("click", e => {
    e.stopPropagation();
    $("export-menu").classList.toggle("hidden");
  });
  $("btn-export-png").addEventListener("click", () => {
    $("export-menu").classList.add("hidden");
    downloadCurrentPng();
  });
  $("btn-export-gif").addEventListener("click", () => {
    $("export-menu").classList.add("hidden");
    downloadGif();
  });
  document.addEventListener("click", e => {
    const dd = $("export-dd");
    if (dd && !dd.contains(e.target)) $("export-menu").classList.add("hidden");
  });
  $("palette").addEventListener("change", e => {
    state.palette = e.target.value;
    state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
    paintMeta();
    writeURLState();
    loadFrame();
    startPreloadAllHours();
  });
  $("btn-run-prev").addEventListener("click", () => stepRun(-6));
  $("btn-run-next").addEventListener("click", () => stepRun(+6));
  $("btn-loop-reset").addEventListener("click", resetLoopRange);
  setupLoopDrag();
  $("btn-latest").addEventListener("click", jumpToLatest);
  $("model-select").addEventListener("change", async () => {
    const prev = state.model;
    state.model = $("model-select").value;
    applyModelClass();
    // Cap fmax to the new model's limit.
    state.fmax = Math.min(state.fmax, activeMaxFhour());
    if (state.fhr > activeMaxFhour()) state.fhr = activeMaxFhour();
    // If switching to HREF and we haven't fetched HREF cycles yet, do so now.
    if (state.model === "href" && state.hrefCycles.length === 0) {
      setStatus("Checking HREF cycles…");
      try {
        const cycles = await fetch("/api/href-cycles").then(r => r.json());
        state.hrefCycles = cycles;
        if (cycles.length > 0 && !hrefCycleExists(state.date, state.run)) {
          state.date = cycles[0].date;
          state.run  = cycles[0].run;
          paintTopbar();
        }
      } catch (e) { setStatus("HREF cycle fetch failed: " + e); }
    } else if (state.model === "href" && !hrefCycleExists(state.date, state.run)) {
      // Current cycle doesn't exist in HREF — jump to latest available HREF run.
      if (state.hrefCycles.length > 0) {
        state.date = state.hrefCycles[0].date;
        state.run  = state.hrefCycles[0].run;
        paintTopbar();
      }
    }
    state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
    state.availableFhours = new Set();
    buildTimeline();
    writeURLState();
    await refreshCycleStatus();
    await loadFrame();
    startPreloadAllHours();
  });
  $("btn-theme").addEventListener("click", () => {
    const cur = currentTheme();
    const nxt = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nxt);
    localStorage.setItem("refs-theme", nxt);
    $("btn-theme").textContent = nxt === "dark" ? "☾" : "☀";
    state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
    paintTimeline();
    loadFrame();
    startPreloadAllHours();
  });
  $("btn-prev").addEventListener("click", () => step(-1));
  $("btn-next").addEventListener("click", () => step(+1));
  $("btn-play").addEventListener("click", togglePlay);
  $("btn-slower").addEventListener("click", () => changeSpeed(-1));
  $("btn-faster").addEventListener("click", () => changeSpeed(+1));
  $("prod-search").addEventListener("input", buildProductList);

  // MRMS overlay mode picker. Off by default each session — verification
  // is an explicit action, and quietly auto-on hides the perf cost. We
  // also clear any legacy "refs-mrms-enabled" key so returning users
  // don't carry forward the old boolean state.
  localStorage.removeItem("refs-mrms-enabled");
  const mrmsModeSel = $("mrms-mode");
  if (mrmsModeSel) {
    mrmsModeSel.value = "off";
    mrmsMode = "off";
    mrmsModeSel.addEventListener("change", () => {
      const v = mrmsModeSel.value;
      mrmsMode = MRMS_MODE_CYCLE.includes(v) ? v : "off";
      updateMrmsOverlay();
    });
  }

  // MRMS product selector (composite / hsr / base) — persisted across sessions.
  const mrmsProdSel = $("mrms-product");
  if (mrmsProdSel) {
    const storedProd = localStorage.getItem("refs-mrms-product");
    if (storedProd && ["refc","hsr","base"].includes(storedProd)) {
      mrmsProduct = storedProd;
      mrmsProdSel.value = storedProd;
    }
    mrmsProdSel.addEventListener("change", () => {
      mrmsProduct = mrmsProdSel.value || "refc";
      localStorage.setItem("refs-mrms-product", mrmsProduct);
      updateMrmsOverlay();
    });
  }

  // LSR verification overlay
  const lsrToggle = $("toggle-lsrs");
  const floodLsrToggle = $("toggle-flood-lsrs");
  const lsrWinSel = $("lsr-window");
  if (lsrToggle) {
    // LSRs are an explicit verification action — start each session
    // off and let the user opt in. We clear any legacy on-state so
    // returning users don't carry a stuck "on" across the change.
    // (The window-size preference below still persists.)
    localStorage.removeItem("refs-lsr-enabled");
    lsrToggle.addEventListener("change", () => {
      lsrEnabled = !!lsrToggle.checked;
      updateLsrOverlay();
    });
  }
  if (floodLsrToggle) {
    // Same opt-in semantics as severe LSRs; independent of the severe toggle.
    localStorage.removeItem("refs-flood-lsr-enabled");
    floodLsrToggle.addEventListener("change", () => {
      floodLsrEnabled = !!floodLsrToggle.checked;
      updateFloodLsrOverlay();
    });
  }
  // WxWorks region boundary overlay.
  // Regions are baked into the tile (zorder 10, below city labels at 11-12),
  // so toggling just reloads the current frame with the updated tileUrl().
  const regionsToggle = $("toggle-regions");
  if (regionsToggle) {
    const stored = localStorage.getItem("refs-regions-enabled");
    if (stored === "1") {
      regionsEnabled = true;
      regionsToggle.checked = true;
      // loadFrame() called below at startup; tileUrl() will include &regions=1
    }
    regionsToggle.addEventListener("change", () => {
      regionsEnabled = !!regionsToggle.checked;
      localStorage.setItem("refs-regions-enabled", regionsEnabled ? "1" : "0");
      state.loadedFhours = new Set();   // invalidate frame cache
      loadFrame();                       // reload with new tileUrl()
    });
  }

  // REFS forecast contour overlay on MRMS panel.
  const refsContourToggle = $("toggle-refs-contours");
  if (refsContourToggle) {
    localStorage.removeItem("refs-contour-enabled");
    refsContourToggle.addEventListener("change", () => {
      refsContourEnabled = !!refsContourToggle.checked;
      updateRefsContourOverlay();
    });
  }
  if (lsrWinSel) {
    const storedWin = localStorage.getItem("refs-lsr-window");
    if (storedWin && /^\d+$/.test(storedWin)) {
      lsrWindow = parseInt(storedWin, 10);
      lsrWinSel.value = storedWin;
    }
    lsrWinSel.addEventListener("change", () => {
      lsrWindow = parseInt(lsrWinSel.value, 10) || 30;
      localStorage.setItem("refs-lsr-window", String(lsrWindow));
      updateLsrOverlay();
      updateFloodLsrOverlay();
    });
  }

  // Counties / Cities basemap toggles. Default OFF; persisted across
  // sessions via localStorage, also serializable via the share-link.
  // Toggling either invalidates loaded frames and triggers a refresh so
  // the user immediately sees the new basemap state.
  const countiesToggle = $("toggle-counties");
  const citiesToggle   = $("toggle-cities");
  if (countiesToggle) {
    countiesToggle.checked = state.showCounties;
    countiesToggle.addEventListener("change", () => {
      state.showCounties = !!countiesToggle.checked;
      localStorage.setItem("refs-counties", state.showCounties ? "1" : "0");
      state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
      writeURLState();
      loadFrame();
      startPreloadAllHours();
    });
  }
  if (citiesToggle) {
    citiesToggle.checked = state.showCities;
    citiesToggle.addEventListener("change", () => {
      state.showCities = !!citiesToggle.checked;
      localStorage.setItem("refs-cities", state.showCities ? "1" : "0");
      state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
      writeURLState();
      loadFrame();
      startPreloadAllHours();
    });
  }

  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.shiftKey && (e.key === "<" || e.key === ",")) { stepRun(-6); return; }
    if (e.shiftKey && (e.key === ">" || e.key === ".")) { stepRun(+6); return; }
    if (e.key === "ArrowRight" || e.key === ">" || e.key === ".") step(+1);
    else if (e.key === "ArrowLeft"  || e.key === "<" || e.key === ",") step(-1);
    else if (e.key === " " || e.key.toLowerCase() === "p") { e.preventDefault(); togglePlay(); }
    else if (e.key.toLowerCase() === "l") jumpToLatest();
    else if (e.key.toLowerCase() === "t") $("btn-theme").click();
    else if (e.key.toLowerCase() === "m") cycleMrmsMode();
    else if (e.key.toLowerCase() === "c") toggleCompare();
    else if (e.key === "\\") $("rail-toggle").click();
  });

  $("btn-compare").addEventListener("click", toggleCompare);

  // Compare type (HREF vs Prior Run) and swipe/split toggle
  const compareTypeEl = $("compare-type");
  if (compareTypeEl) {
    compareTypeEl.addEventListener("change", e => {
      state.compareType = e.target.value;
      state.compareLoadedFhours = new Set();
      ++comparePreloadGen;
      applyCompareStyle();
      if (state.compareMode) { loadCompareFrame(); startComparePreloadAllHours(); }
    });
  }
  const compareStyleBtn = $("btn-compare-style");
  if (compareStyleBtn) {
    compareStyleBtn.addEventListener("click", () => {
      state.compareStyle = state.compareStyle === "split" ? "swipe" : "split";
      applyCompareStyle();
      if (state.compareMode) loadCompareFrame();
    });
  }
  wireSwipeHandle();

  // Paintball overlay toggle + variable picker
  const togglePB = $("toggle-paintball");
  if (togglePB) {
    togglePB.addEventListener("change", e => {
      paintballEnabled = e.target.checked;
      const varRow = $("paintball-var-row");
      if (varRow) varRow.classList.toggle("visible", paintballEnabled);
      updatePaintballOverlay();
    });
  }
  const pbVarEl = $("paintball-var");
  if (pbVarEl) {
    pbVarEl.addEventListener("change", e => {
      state.paintballVar = e.target.value;
      if (paintballEnabled) { ++paintballLoadToken; updatePaintballOverlay(); }
    });
  }
}

// ----- Misc --------------------------------------------------------------
function setStatus(s) { $("status").textContent = s; }

function parseCycleUTC(yyyymmdd, hh) {
  const y = +yyyymmdd.slice(0,4),
        m = +yyyymmdd.slice(4,6) - 1,
        d = +yyyymmdd.slice(6,8);
  return new Date(Date.UTC(y, m, d, +hh, 0, 0));
}

// ---------- Toast --------------------------------------------------------
let _toastTimer = null;
function showToast(msg, ms = 2200) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.add("hidden"), ms);
}

// ---------- Custom sector modal ------------------------------------------
let _editingCsKey = null;

function openCustomSectorModal(existing = null) {
  _editingCsKey = existing ? existing.key : null;
  $("cs-name").value   = existing ? existing.name : "";
  $("cs-lonmin").value = existing ? existing.bbox[0] : "";
  $("cs-latmin").value = existing ? existing.bbox[1] : "";
  $("cs-lonmax").value = existing ? existing.bbox[2] : "";
  $("cs-latmax").value = existing ? existing.bbox[3] : "";
  $("cs-error").classList.add("hidden");
  $("cs-delete").classList.toggle("hidden", !existing);
  $("cs-modal").classList.remove("hidden");
  $("cs-name").focus();
}
function closeCustomSectorModal() { $("cs-modal").classList.add("hidden"); }

function wireCustomSectorModal() {
  $("cs-cancel").addEventListener("click", closeCustomSectorModal);
  $("cs-modal").addEventListener("click", e => {
    if (e.target.id === "cs-modal") closeCustomSectorModal();
  });
  $("cs-save").addEventListener("click", () => {
    const name = $("cs-name").value.trim();
    const lonMin = parseFloat($("cs-lonmin").value);
    const latMin = parseFloat($("cs-latmin").value);
    const lonMax = parseFloat($("cs-lonmax").value);
    const latMax = parseFloat($("cs-latmax").value);
    const err = (msg) => {
      const e = $("cs-error"); e.textContent = msg; e.classList.remove("hidden");
    };
    if (!name) return err("Name is required.");
    if ([lonMin, latMin, lonMax, latMax].some(v => Number.isNaN(v)))
      return err("All four corners must be numbers.");
    if (!(lonMin < lonMax)) return err("West must be less than East.");
    if (!(latMin < latMax)) return err("South must be less than North.");
    if (lonMin < -180 || lonMax > 180) return err("Longitudes must be in [-180, 180].");
    if (latMin < -90 || latMax > 90) return err("Latitudes must be in [-90, 90].");
    const key = _editingCsKey || ("custom-" + Math.random().toString(36).slice(2, 10));
    const sector = { key, name, bbox: [lonMin, latMin, lonMax, latMax] };
    const i = state.customSectors.findIndex(s => s.key === key);
    if (i >= 0) state.customSectors[i] = sector;
    else state.customSectors.push(sector);
    persistCustomSectors();
    populateSectors();
    state.sector = key;
    $("sector").value = key; _syncSectorBtn();
    state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
    updateEditSectorButton();
    closeCustomSectorModal();
    paintMeta();
    writeURLState();
    loadFrame();
    startPreloadAllHours();
    showToast(`Saved sector: ${name}`);
  });
  $("cs-delete").addEventListener("click", () => {
    if (!_editingCsKey) return;
    state.customSectors = state.customSectors.filter(s => s.key !== _editingCsKey);
    persistCustomSectors();
    if (state.sector === _editingCsKey) state.sector = "CONUS";
    populateSectors();
    $("sector").value = state.sector;
    state.loadedFhours = new Set(); state.compareLoadedFhours = new Set(); ++comparePreloadGen;
    updateEditSectorButton();
    closeCustomSectorModal();
    paintMeta();
    writeURLState();
    loadFrame();
    startPreloadAllHours();
  });
}

// ---------- Draw-on-map sector creation ----------------------------------
let _drawing = false, _drawStart = null, _drawRectEl = null;
let _drawJustEnded = 0;

function startDrawMode() {
  if (state.drawMode) return endDrawMode();
  state.drawMode = true;
  $("map-wrap").classList.add("draw-mode");
  $("draw-instructions").classList.remove("hidden");
  setStatus("Drag a rectangle on the map to define a new sector.");
  $("frame").addEventListener("mousedown", _drawMouseDown);
  document.addEventListener("keydown", _drawKey);
}
function endDrawMode() {
  state.drawMode = false;
  _drawing = false;
  $("map-wrap").classList.remove("draw-mode");
  $("draw-instructions").classList.add("hidden");
  $("frame").removeEventListener("mousedown", _drawMouseDown);
  document.removeEventListener("keydown", _drawKey);
  document.removeEventListener("mousemove", _drawMove);
  document.removeEventListener("mouseup", _drawUp);
  if (_drawRectEl) { _drawRectEl.remove(); _drawRectEl = null; }
}
function _drawKey(e) {
  if (e.key === "Escape") { setStatus("Sector draw cancelled."); endDrawMode(); }
}
function _drawMouseDown(e) {
  e.preventDefault();
  _drawing = true;
  _drawStart = { x: e.clientX, y: e.clientY };
  _drawRectEl = document.createElement("div");
  _drawRectEl.className = "draw-rect";
  document.body.appendChild(_drawRectEl);
  $("draw-instructions").classList.add("hidden");
  document.addEventListener("mousemove", _drawMove);
  document.addEventListener("mouseup", _drawUp);
}
function _drawMove(e) {
  if (!_drawing || !_drawRectEl) return;
  const x = Math.min(_drawStart.x, e.clientX);
  const y = Math.min(_drawStart.y, e.clientY);
  _drawRectEl.style.left = x + "px";
  _drawRectEl.style.top = y + "px";
  _drawRectEl.style.width = Math.abs(e.clientX - _drawStart.x) + "px";
  _drawRectEl.style.height = Math.abs(e.clientY - _drawStart.y) + "px";
}
async function _drawUp() {
  _drawing = false;
  document.removeEventListener("mousemove", _drawMove);
  document.removeEventListener("mouseup", _drawUp);
  const r = _drawRectEl ? _drawRectEl.getBoundingClientRect() : null;
  if (_drawRectEl) { _drawRectEl.remove(); _drawRectEl = null; }
  if (!r || r.width < 16 || r.height < 16) {
    setStatus("Cancelled (rectangle too small).");
    endDrawMode(); return;
  }
  const img = $("frame");
  // Convert the drawn corners to image fractions, then let the backend invert
  // them with the tile's REAL Lambert projection. Inverting Lambert in the
  // browser with a linear/equirectangular guess lands the box far off (a box
  // over NJ/PA/MD came out over the Great Lakes), so the projection-aware
  // inverse must happen server-side.
  const f1 = frameFractions(img, { clientX: r.left,  clientY: r.top });
  const f2 = frameFractions(img, { clientX: r.right, clientY: r.bottom });
  endDrawMode();
  _drawJustEnded = Date.now();
  if (!f1 || !f2) {
    setStatus("Draw a box inside the map area. Try again.");
    return;
  }
  setStatus("Mapping drawn area…");
  const qs = new URLSearchParams();
  qs.set("fx1", f1.fx.toFixed(5)); qs.set("fy1", f1.fy.toFixed(5));
  qs.set("fx2", f2.fx.toFixed(5)); qs.set("fy2", f2.fy.toFixed(5));
  probeSectorParams(qs);
  try {
    const res = await fetch(`/api/unproject?${qs.toString()}`).then(x => x.json());
    if (!res || !res.ok) {
      setStatus("Couldn't map the drawn area — try again.");
      return;
    }
    const bbox = [res.lon_min, res.lat_min, res.lon_max, res.lat_max];
    const nm = `Custom ${bbox[1].toFixed(0)}-${bbox[3].toFixed(0)}N, ` +
               `${Math.abs(bbox[0]).toFixed(0)}-${Math.abs(bbox[2]).toFixed(0)}W`;
    setStatus("");
    openCustomSectorModal({ key: null, name: nm, bbox });
  } catch {
    setStatus("Couldn't map the drawn area — try again.");
  }
}

// (Removed the client-side _pixelToLatLon equirectangular estimate — it could
// not account for the tiles' Lambert Conformal projection or the MAP_BOX axes
// inset, so drawn boxes landed far off. The draw flow now sends image
// fractions to /api/unproject for a projection-accurate inverse.)

// ---------- Export -------------------------------------------------------
function _triggerDownload(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

function downloadPngUrl(fhr) {
  const qs = _commonTileParams();
  if (regionsEnabled) qs.set("regions", "1");
  return `/api/download/${state.date}/${String(state.run).padStart(2,"0")}/${state.pid}/${fhr}.png?${qs}`;
}

function _loadImg(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = src;
  });
}

async function downloadCurrentPng() {
  const fname = `REFS_${state.date}_${String(state.run).padStart(2,"0")}z_` +
                `${state.pid}_${state.sector}_F${String(state.fhr).padStart(3,"0")}.png`;
  showToast("Downloading PNG…");
  try {
    // Bake any visible LSR overlays (severe + flood) into the exported image
    // so the download matches what's on screen. Both overlays are same-origin
    // and share the tile's coordinate box, so drawing them scaled to the base
    // tile's native resolution keeps markers pixel-aligned.
    const lsr = $("lsr-overlay"), flood = $("flood-lsr-overlay");
    const overlaySrcs = [lsr, flood]
      .filter(el => el && el.classList.contains("visible") && el.src && !el.src.endsWith("#"))
      .map(el => el.src);

    if (overlaySrcs.length === 0) {
      // Fast path — nothing to composite; download the tile directly.
      const blob = await fetch(downloadPngUrl(state.fhr)).then(r => r.blob());
      _triggerDownload(blob, fname);
      showToast("PNG saved");
      return;
    }

    const base = await _loadImg(downloadPngUrl(state.fhr));
    const canvas = document.createElement("canvas");
    canvas.width = base.naturalWidth;
    canvas.height = base.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(base, 0, 0);
    for (const src of overlaySrcs) {
      try {
        const ov = await _loadImg(src);
        ctx.drawImage(ov, 0, 0, canvas.width, canvas.height);
      } catch { /* skip an overlay that fails to load rather than abort */ }
    }
    canvas.toBlob(blob => {
      if (blob) { _triggerDownload(blob, fname); showToast("PNG saved (with LSRs)"); }
      else showToast("PNG download failed");
    }, "image/png");
  } catch {
    showToast("PNG download failed");
  }
}

async function downloadGif() {
  const fname = `REFS_${state.date}_${String(state.run).padStart(2,"0")}z_` +
                `${state.pid}_${state.sector}_loop.gif`;
  showToast("Building GIF… may take 30–90s on a cold cycle.", 5000);
  try {
    const r = await fetch(gifUrl());
    if (!r.ok) throw new Error("HTTP " + r.status);
    _triggerDownload(await r.blob(), fname);
    showToast("GIF saved");
  } catch (e) {
    showToast("GIF export failed: " + e.message);
  }
}

// Refresh cycle status every 60s (catches newly-posted F-hours mid-run).
setInterval(async () => {
  if (!state.date) return;
  if (state.followLatest && !state.playing) {
    // "Follow latest": pick up a newer run if one posted, refresh the current
    // cycle's available hours, then snap the view to the newest posted fhr.
    try {
      let cyc = null;
      if (state.model === "refs") {
        cyc = await fetch("/api/latest").then(r => r.json()).catch(() => null);
      } else if (state.hrefCycles.length) {
        cyc = state.hrefCycles[0];
      }
      if (cyc && (cyc.date !== state.date || cyc.run !== state.run)) {
        await applyCycle(cyc.date, cyc.run);
        state.followLatest = true;          // applyCycle path doesn't clear it
      } else {
        await refreshCycleStatus();
      }
      goToLatestFhr();
    } catch { /* transient; try again next tick */ }
    return;
  }
  refreshCycleStatus();
}, 60_000);

// ---------- Click-to-zoom lightbox ---------------------------------------
// Self-contained: only acts when #zoom-overlay is in the DOM. Safe to bolt
// on without touching the existing wireEvents().
(function wireZoomLightbox() {
  const frame   = document.getElementById("frame");
  const overlay = document.getElementById("zoom-overlay");
  const img     = document.getElementById("zoom-img");
  const closeBtn = document.getElementById("zoom-close");
  if (!frame || !overlay || !img || !closeBtn) return;

  function openZoom() {
    // Don't open while the user is mid-draw of a custom sector.
    if (state && state.drawMode) return;
    if (!frame.src) return;
    img.src = frame.src;
    overlay.classList.remove("hidden");
  }
  function closeZoom() {
    overlay.classList.add("hidden");
    img.src = "";
  }

  frame.addEventListener("click", openZoom);
  closeBtn.addEventListener("click", closeZoom);
  overlay.addEventListener("click", e => {
    if (e.target === overlay) closeZoom();
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !overlay.classList.contains("hidden")) closeZoom();
  });
})();

// ---------- Frame coordinate helper ----------------------------------------
// #frame uses object-fit:contain — find the letterboxed content box so the
// returned fractions refer to the actual image, not the element. Returns
// {fx, fy} (fy from top) or null when outside the image content.
function frameFractions(frame, e) {
  const r = frame.getBoundingClientRect();
  const iw = frame.naturalWidth, ih = frame.naturalHeight;
  if (!r.width || !r.height || !iw || !ih) return null;
  const scale = Math.min(r.width / iw, r.height / ih);
  const dw = iw * scale, dh = ih * scale;
  const ox = r.left + (r.width - dw) / 2;
  const oy = r.top  + (r.height - dh) / 2;
  const fx = (e.clientX - ox) / dw;
  const fy = (e.clientY - oy) / dh;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null;
  return { fx, fy };
}

function probeSectorParams(qs) {
  const cs = findCustomSector(state.sector);
  if (cs) {
    qs.set("bbox", cs.bbox.join(","));
    qs.set("sector_name", cs.name || "");
  } else {
    qs.set("region", state.sector);
  }
  if (state.model && state.model !== "refs") qs.set("model", state.model);
  return qs;
}

// ---------- Hover value probe ---------------------------------------------
// Debounced cursor readout: when the mouse pauses over the frame, fetch the
// field value at that point from /api/probe and show it in a small tooltip.
// Server returns value:null for unsupported products (paintballs/stamps) —
// the tooltip just stays hidden.
(function wireValueProbe() {
  const frame = document.getElementById("frame");
  const stack = document.getElementById("frame-stack");
  if (!frame || !stack) return;
  const tip = document.createElement("div");
  tip.id = "probe-tip";
  tip.className = "hidden";
  stack.appendChild(tip);

  let timer = null;
  let ctrl  = null;
  function hide() {
    tip.classList.add("hidden");
    if (timer) { clearTimeout(timer); timer = null; }
    if (ctrl)  { ctrl.abort(); ctrl = null; }
  }
  frame.addEventListener("mouseleave", hide);

  frame.addEventListener("mousemove", (e) => {
    if (state.playing || state.drawMode) { hide(); return; }
    const f = frameFractions(frame, e);
    if (!f) { hide(); return; }
    const px = e.clientX, py = e.clientY;
    if (timer) clearTimeout(timer);
    timer = setTimeout(async () => {
      const qs = probeSectorParams(new URLSearchParams(
        { fx: f.fx.toFixed(4), fy: f.fy.toFixed(4) }));
      if (ctrl) ctrl.abort();
      ctrl = new AbortController();
      try {
        const rs = await fetch(
          `/api/probe/${state.date}/${String(state.run).padStart(2,"0")}/${state.pid}/${state.fhr}?${qs}`,
          { signal: ctrl.signal });
        const j = await rs.json();
        if (j && j.ok && j.value !== null && j.value !== undefined) {
          tip.innerHTML =
            `${j.value}${j.units ? " " + j.units : ""}` +
            `<span class="probe-hint">⇧click → meteogram</span>`;
          const sr = stack.getBoundingClientRect();
          tip.style.left = `${px - sr.left + 16}px`;
          tip.style.top  = `${py - sr.top - 12}px`;
          tip.classList.remove("hidden");
        } else {
          tip.classList.add("hidden");
        }
      } catch { /* aborted or transient — keep quiet */ }
    }, 220);
  });
})();

// ---------- Click meteogram -------------------------------------------------
// Shift+click a map point → popup time series of the current product's value
// at that location across every valid forecast hour. Capture-phase listener
// + stopPropagation keeps the plain-click zoom lightbox from also firing.
(function wireMeteogram() {
  const frame = document.getElementById("frame");
  if (!frame) return;
  const overlay = document.createElement("div");
  overlay.id = "meteogram-overlay";
  overlay.className = "zoom-overlay hidden";
  overlay.innerHTML =
    `<div id="meteogram-card">
       <div id="meteogram-head">
         <span id="meteogram-title">Meteogram</span>
         <button id="meteogram-close" title="Close">×</button>
       </div>
       <div id="meteogram-body"></div>
     </div>`;
  document.body.appendChild(overlay);
  const body  = overlay.querySelector("#meteogram-body");
  const title = overlay.querySelector("#meteogram-title");
  function close() { overlay.classList.add("hidden"); body.innerHTML = ""; }
  overlay.querySelector("#meteogram-close").addEventListener("click", close);
  overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !overlay.classList.contains("hidden")) close();
  });

  function validLabel(fhr) {
    const init = parseCycleUTC(state.date, state.run);
    const v = new Date(init.getTime() + fhr * 3600_000);
    const days = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
    return `${days[v.getUTCDay()]} ${String(v.getUTCHours()).padStart(2,"0")}Z`;
  }

  function drawChart(series) {
    const pts = series.points.filter(p => p.value !== null && p.value !== undefined);
    if (!pts.length) {
      body.innerHTML = `<div class="meteogram-msg">No data at this point for any forecast hour.</div>`;
      return;
    }
    const W = 640, H = 300, L = 52, R = 14, T = 14, B = 40;
    const fmin = series.points[0].fhr, fmax = series.points[series.points.length - 1].fhr;
    let vmin = Math.min(...pts.map(p => p.value));
    let vmax = Math.max(...pts.map(p => p.value));
    if (vmin === vmax) { vmin -= 1; vmax += 1; }
    const pad = (vmax - vmin) * 0.08;
    vmin -= pad; vmax += pad;
    const X = f => L + (f - fmin) / Math.max(1, fmax - fmin) * (W - L - R);
    const Y = v => T + (1 - (v - vmin) / (vmax - vmin)) * (H - T - B);
    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    // y gridlines + labels (4 steps)
    for (let i = 0; i <= 4; i++) {
      const v = vmin + (vmax - vmin) * i / 4;
      const y = Y(v);
      svg += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" class="mg-grid"/>`;
      svg += `<text x="${L - 6}" y="${y + 4}" class="mg-ylab" text-anchor="end">${v.toFixed(Math.abs(vmax) < 10 ? 1 : 0)}</text>`;
    }
    // x ticks every 6 fhr
    for (let f = Math.ceil(fmin / 6) * 6; f <= fmax; f += 6) {
      const x = X(f);
      svg += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" class="mg-grid"/>`;
      svg += `<text x="${x}" y="${H - B + 14}" class="mg-xlab" text-anchor="middle">F${String(f).padStart(2,"0")}</text>`;
      svg += `<text x="${x}" y="${H - B + 28}" class="mg-xlab2" text-anchor="middle">${validLabel(f)}</text>`;
    }
    // line segments (break across null gaps)
    let path = "";
    let prevNull = true;
    for (const p of series.points) {
      if (p.value === null || p.value === undefined) { prevNull = true; continue; }
      path += `${prevNull ? "M" : "L"}${X(p.fhr).toFixed(1)},${Y(p.value).toFixed(1)} `;
      prevNull = false;
    }
    svg += `<path d="${path}" class="mg-line" fill="none"/>`;
    for (const p of pts) {
      svg += `<circle cx="${X(p.fhr).toFixed(1)}" cy="${Y(p.value).toFixed(1)}" r="2.6" class="mg-dot"><title>F${String(p.fhr).padStart(2,"0")} (${validLabel(p.fhr)}): ${p.value}${series.units ? " " + series.units : ""}</title></circle>`;
    }
    svg += `</svg>`;
    const latlon = `${Math.abs(series.lat).toFixed(2)}°${series.lat >= 0 ? "N" : "S"} ${Math.abs(series.lon).toFixed(2)}°${series.lon <= 0 ? "W" : "E"}`;
    body.innerHTML =
      `<div class="meteogram-sub">${series.name}${series.units ? " (" + series.units + ")" : ""} · ${latlon} · ${state.date} ${String(state.run).padStart(2,"0")}Z run</div>` + svg;
  }

  frame.addEventListener("click", async (e) => {
    if (!e.shiftKey) return;
    e.stopPropagation();
    e.preventDefault();
    if (state.drawMode) return;
    const f = frameFractions(frame, e);
    if (!f) return;
    title.textContent = `Meteogram — ${state.pidName || state.pid}`;
    body.innerHTML = `<div class="meteogram-msg">Sampling all forecast hours…<br>
      <span class="meteogram-msg-sub">first request on a cold cycle can take ~30-60 s</span></div>`;
    overlay.classList.remove("hidden");
    const qs = probeSectorParams(new URLSearchParams(
      { fx: f.fx.toFixed(4), fy: f.fy.toFixed(4) }));
    try {
      const rs = await fetch(
        `/api/meteogram/${state.date}/${String(state.run).padStart(2,"0")}/${state.pid}?${qs}`);
      const j = await rs.json();
      if (!j || !j.ok || !j.series) {
        body.innerHTML = `<div class="meteogram-msg">${(j && j.reason) ||
          "Meteogram not available for this product."}</div>`;
        return;
      }
      drawChart(j.series);
    } catch {
      body.innerHTML = `<div class="meteogram-msg">Failed to build the meteogram — try again.</div>`;
    }
  }, true);
})();
