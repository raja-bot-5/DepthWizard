// DepthWizard workstation UI. Talks only to the local API (same origin). No external requests.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
  return r.json();
});
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const CAL_NAMES = { M0_dem_only: "DEM only", M1_global_affine: "Model scaled to DEM",
  M2_dem_residual_cell: "DEM + model detail, per 30 m cell", M2_dem_residual_smooth: "DEM + model detail (smooth)",
  a_dem_only: "DEM only", b_robust_affine: "Model scaled to DEM", c_dem_plus_residual: "DEM + model detail, per 30 m cell",
  c2_dem_plus_smooth_residual: "DEM + model detail", none: "None" };
const fmt = (v, d = 2, unit = "") => (v === null || v === undefined || Number.isNaN(Number(v))) ? "–" :
  `${Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}${unit}`;
const short = (s, n = 40) => (s || "").length > n ? s.slice(0, n - 1) + "…" : (s || "");
// perceptually uniform colormaps only; stops sampled from matplotlib (same maps as the PNG previews)
const CMAPS = {
  viridis: ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"],
  cividis: ["#00224e", "#1a386f", "#434e6c", "#61656f", "#7d7c78", "#9b9476", "#bcae6c", "#dec958", "#fee838"],
  magma: ["#000004", "#1d1147", "#51127c", "#832681", "#b73779", "#e75263", "#fc8961", "#fec488", "#fcfdbf"],
  berlin: ["#9eb0ff", "#519fd3", "#286886", "#14303e", "#190c09", "#411201", "#7d341e", "#be6f63", "#ffadad"],
  gray: ["#000000", "#404040", "#808080", "#c0c0c0", "#ffffff"],
};
const ramp = (name) => `linear-gradient(90deg, ${(CMAPS[name] || CMAPS.viridis).join(", ")})`;
const STAGES = [["input", "Input & metadata"], ["tiling", "Tiling"], ["depth", "Monocular depth"], ["dem", "Reference DEM"],
  ["calibration", "Calibration"], ["dsm", "DSM"], ["uncertainty", "Uncertainty"], ["ndsm", "Ground / nDSM"],
  ["slope", "Slope"], ["mesh", "Mesh"]];
const STAGE_ORDER = Object.fromEntries(STAGES.map(([k], i) => [k, i]));

const state = { job: null, md: null, calib: null, terrain: null, inner: [], textures: {}, layers: [],
  overlay: "photo", tool: "inspect", measure: [], size: 1, exag: 1, view: "3d" };

// ------------------------------------------------------------------ views
function setView(v) {
  if (v === "layers" && $("tab-layers").disabled) return;
  state.view = v;
  for (const [key, sec] of [["process", "view-process"], ["layers", "view-layers"], ["3d", "view-3d"]]) {
    $(sec).hidden = key !== v;
    document.querySelector(`[data-view="${key}"]`).setAttribute("aria-selected", String(key === v));
  }
  if (v === "3d") resize();
  if (v === "layers") requestAnimationFrame(() => { if (!c2.fitted) fit2d(); else apply2d(); });
}
document.querySelectorAll("[data-view]").forEach((b) => b.onclick = () => setView(b.dataset.view));
addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea") || e.ctrlKey || e.metaKey || e.altKey || fly.isLocked) return;
  const v = { Digit1: "process", Digit2: "layers", Digit3: "3d" }[e.code];
  if (v) setView(v);
});

// ------------------------------------------------------------------ three.js scene
const viewport = $("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 1, 1e6);
scene.add(new THREE.HemisphereLight(0xdfe8f0, 0x3a3228, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.7);
sun.position.set(-1, 2, 1);          // light from the north-west, as on printed relief maps
scene.add(sun);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.maxPolarAngle = Math.PI / 2 - 0.04;   // never orbit under the surface (its underside is culled)
const fly = new PointerLockControls(camera, renderer.domElement);
const keys = new Set();
let nav = "orbit";

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  if (!w || !h) return;                        // hidden view: keep the last size
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = `${w}px`; renderer.domElement.style.height = `${h}px`;
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);
resize();

// ------------------------------------------------------------------ loading a job
const loader = new GLTFLoader();
const loadGLB = (url) => new Promise((res, rej) => loader.load(url, (g) => res(g.scene), undefined, rej));
const texLoader = new THREE.TextureLoader();
const loadTex = (url) => new Promise((res, rej) => texLoader.load(url, (t) => {
  t.flipY = false;                      // match glTF texture convention
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  res(t);
}, undefined, rej));

function clearTerrain() {
  if (!state.terrain) return;
  scene.remove(state.terrain);
  state.terrain.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); });
  state.terrain = null; state.inner = []; state.textures = {};
  clearMeasure();
}

async function showJob(id, { view } = {}) {
  const job = await api(`/jobs/${id}`);
  if (job.status !== "done") return;
  const md = job.metadata;
  const calib = await api(`/jobs/${id}/files/calibration.json`);
  state.job = id; state.md = md; state.calib = calib; state.jobInfo = job;
  clearTerrain();
  $("empty").hidden = true;
  const base = `/jobs/${id}/files/`;
  const [hi, lo] = await Promise.all([loadGLB(base + "mesh.glb"), loadGLB(base + "mesh_lod1.glb")]);

  // data frame: x east, y north (<= 0 going south), z up  ->  three.js: y up
  const lod = new THREE.LOD();
  const W = md.mesh.grid[1] * md.mesh.vertex_spacing[0];
  const H = md.mesh.grid[0] * md.mesh.vertex_spacing[1];
  state.size = Math.max(W, H);
  for (const [g, dist] of [[hi, 0], [lo, state.size * 1.6]]) {
    const frame = new THREE.Group();
    frame.rotation.x = -Math.PI / 2;
    frame.position.set(-W / 2, 0, -H / 2);   // centre the terrain on the LOD origin
    g.traverse((o) => { if (o.isMesh) { o.material.roughness = 0.95; o.material.metalness = 0; state.inner.push(o); } });
    frame.add(g);
    lod.addLevel(frame, dist);
  }
  state.terrain = lod;
  scene.add(lod);
  state.textures.photo = state.inner[0].material.map;
  applyExaggeration();

  const metric = md.input.georeferenced && md.product.kind === "metric DSM";
  document.querySelector('[data-overlay="slope"]').disabled = !md.files?.slope;
  document.querySelector('[data-overlay="diff"]').disabled = true;
  setOverlay("photo");
  renderPanels(job, metric);
  frameCamera();

  state.layers = await api(`${base}layers.json?t=${Date.now()}`).catch(() => []);
  $("tab-layers").disabled = !state.layers.length;
  buildFilmstrip();
  if (!live.active || live.job !== id) {
    const evs = (await api(`/jobs/${id}/events`).catch(() => ({ events: [] }))).events;
    renderPipelineFrom(evs.length ? evs : (md.stages || []), job);
  }
  if (view) setView(view);
}

function frameCamera() {
  // frame from the real 3D extent: relief can be a large fraction of the width (steep sites)
  const box = new THREE.Box3().setFromObject(state.terrain);
  const size = box.getSize(new THREE.Vector3());
  const target = box.getCenter(new THREE.Vector3());
  const s = Math.max(size.x, size.z, size.y);
  const end = new THREE.Vector3(target.x + s * 0.15, box.max.y + s * 0.45, target.z + s * 0.8);
  orbit.target.copy(target);
  camera.near = s / 2000; camera.far = s * 20; camera.updateProjectionMatrix();
  orbit.minDistance = s * 0.02; orbit.maxDistance = s * 6;
  if (reduceMotion) { camera.position.copy(end); camera.lookAt(target); orbit.update(); return; }
  // one orchestrated moment: descend from overhead onto the oblique view
  const start = new THREE.Vector3(target.x, box.max.y + s * 1.4, target.z + 0.01);
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 1400), e = 1 - Math.pow(1 - k, 3);
    camera.position.lerpVectors(start, end, e); camera.lookAt(target);
    if (k < 1) requestAnimationFrame(step); else orbit.update();
  };
  requestAnimationFrame(step);
}

// ------------------------------------------------------------------ rail panels
function dl(node, rows) {
  node.replaceChildren(...rows.filter(Boolean).flatMap(([k, v]) => [el("dt", "", k), el("dd", "", v)]));
}

function renderPanels(job, metric) {
  const { md, calib } = state;
  $("product").hidden = false;
  $("stamp").classList.toggle("relative", !metric);
  $("stamp-title").textContent = metric ? "Metric DSM" : "Relative surface";
  $("stamp-sub").textContent = metric
    ? `Heights in metres, ${md.product.vertical_datum.split(" (")[0]} datum, tied to ${short(md.product.calibration_dem?.split(" (")[0], 30)}.`
    : md.input.georef_issue || "No units. This image has no map coordinates, so heights show shape only and cannot be measured.";
  $("chip").hidden = false;
  $("chip").classList.toggle("relative", !metric);
  $("chip-text").textContent = metric ? `Metric · ${md.product.vertical_datum.split(" (")[0]}` : "Relative · no units";

  $("input-block").hidden = false;
  dl($("input-dl"), [
    ["File", short(job.original_name || job.input, 30)],
    ["Size", `${md.input.width.toLocaleString()} × ${md.input.height.toLocaleString()} px`],
    ["Bands", `${md.input.bands} · ${md.input.dtype}`],
    ["Map grid", md.input.georeferenced ? "Yes" : "No"],
    md.input.crs && ["CRS", short(md.input.crs, 26)],
    md.input.gsd_m && ["Pixel size", `${fmt(md.input.gsd_m[0], 2)} m`],
    ["Model input", short(md.input.model_input, 60)],
  ]);

  $("calib-block").hidden = false;
  const res = calib.residual_vs_dem_m || {};
  dl($("calib-dl"), metric ? [
    ["Method", CAL_NAMES[calib.method] || calib.method],
    ["Calibrated to", short(calib.dem_source?.split(" (")[0].replace(" v001", ""), 28)],
    ["Vertical datum", calib.dem_datum_conversion ? `EGM2008 ← ${calib.dem_datum_conversion.source_datum}` : "EGM2008"],
    calib.gate && ["Quality gate", calib.gate.passed ? "Passed" : "Failed"],
    calib.s !== undefined && ["Detail scale", fmt(calib.s, 1)],
    res.std !== undefined && ["Added detail, std", `${fmt(res.std, 2)} m`],
    ["Model", md.product.model],
  ] : [["Method", "None (relative output)"], ["Model", md.product.model]]);
  const warns = [...(md.input.warnings || []), ...(calib.warnings || []),
    ...(!calib.gate?.passed && calib.gate?.reasons ? calib.gate.reasons : [])];
  $("warnings").replaceChildren(...[...new Set(warns)].map((w) => el("li", "", w[0].toUpperCase() + w.slice(1).replace(/\.$/, "") + ".")));

  $("ref-block").hidden = !metric;
  dl($("ref-dl"), []);
  $("export-block").hidden = false;
  const base = `/jobs/${state.job}/files/`;
  const f = md.files || {};
  const links = [["DSM", "dsm.tif", "GeoTIFF · m"], ["nDSM", "ndsm.tif", "derived · m"], ["Uncertainty", "uncertainty.tif", "1σ · m"],
    ["Confidence", "height_confidence.tif", "derived heights"], ["Relative", "rdsm.tif", "GeoTIFF · unitless"],
    ["DEM on grid", "dem.tif", "EGM2008"], ["Slope", "slope.tif", "degrees"], ["Mesh", "mesh.glb", "glTF binary"],
    ["Calibration", "calibration.json", "report"], ["Metadata", "metadata.json", "provenance"]]
    .filter(([, file]) => Object.values(f).includes(file) || file === "metadata.json");
  const nodes = links.map(([label, file, sub]) => {
    const a = el("a"); a.href = base + file; a.download = file; a.append(label, el("small", "", sub)); return a;
  });
  const shot = el("button", "", "Screenshot"); shot.type = "button";
  shot.onclick = () => { const a = el("a"); a.href = renderer.domElement.toDataURL("image/png");
    a.download = `depthwizard_${state.job.slice(0, 8)}.png`; a.click(); };
  $("exports").replaceChildren(...nodes, shot);
  $("fieldbook").hidden = false;
  $("fb-title").textContent = "Point";
  dl($("fb-dl"), []);
  $("fb-conf").hidden = true;
  $("fb-hint").textContent = "Click the surface to read its height.";
}

// ------------------------------------------------------------------ processing view (real pipeline events)
const live = { active: false, job: null, es: null, t0: null, timer: null, chain: Promise.resolve(), last: -1, cards: {} };

function buildPipeline() {
  live.cards = {};
  $("pipeline").replaceChildren(...STAGES.map(([key, title]) => {
    const li = el("li", "st"); li.dataset.stage = key;
    const head = el("header");
    const h3 = el("h3", "", title);
    const secs = el("span", "secs");
    head.append(el("span", "n"), h3, secs, el("span", "dot"));
    const sum = el("p", "sum", "Waiting");
    const thumbs = el("div", "thumbs");
    li.append(head, sum, thumbs);
    live.cards[key] = { li, h3, secs, sum, thumbs };
    return li;
  }));
}

function summarize(stage, s = {}) {
  const pct = (x) => `${fmt(100 * x, 0)} %`;
  switch (stage) {
    case "input": return `${s.width?.toLocaleString()} × ${s.height?.toLocaleString()} px · ${s.bands} band ${s.dtype}` +
      (s.georeferenced ? ` · ${s.crs} · ${fmt(s.gsd_m?.[0], 2)} m` : " · no map grid") +
      (s.warnings?.length ? ` · ${s.warnings.length} warning${s.warnings.length > 1 ? "s" : ""}` : "");
    case "tiling": return `${s.n_tiles} tile${s.n_tiles > 1 ? "s" : ""} of ${s.tile} px, ${s.overlap} px overlap`;
    case "depth": return `${s.model} · ${(s.polarity || "").split(" (")[0]} · tiles: ${s.tile_align || "–"}`;
    case "dem": return `${short((s.source || "").split(" (")[0], 26)} · ${s.source_datum} → EGM2008 · coverage ${pct(s.coverage ?? 0)}` +
      (s.water_fraction ? ` · water ${pct(s.water_fraction)}` : "");
    case "calibration": return `${CAL_NAMES[s.method] || s.method} · gate ${s.gate_passed ? "passed" : "FAILED"}` +
      (s.scale !== undefined && s.scale !== null ? ` · scale ${fmt(s.scale, 1)}` : "") +
      (!s.gate_passed && s.gate_reasons?.length ? ` · ${s.gate_reasons[0]}` : "");
    case "dsm": return s.kind === "metric" ? `Metric · ${s.datum} · ${CAL_NAMES[s.method] || s.method}` : `Relative only · ${(s.reasons || [])[0] || ""}`;
    case "uncertainty": {
      const n = Object.entries(s).filter(([, v]) => typeof v === "number").slice(0, 2);
      return n.length ? n.map(([k, v]) => `${k.replace(/_/g, " ")} ${fmt(v, 2)} m`).join(" · ") : "1-σ estimate";
    }
    case "ndsm": { const c = s.height_confidence || {};
      return `low-confidence heights ${pct(c.low_fraction ?? 0)} · nDSM p95 ${fmt(s.ndsm_p95_m, 1)} m`; }
    case "slope": return `median ${fmt(s.median_deg, 1)}°`;
    case "mesh": return `${s.vertices?.toLocaleString()} vertices · ${s.faces?.toLocaleString()} faces · ${s.units}`;
    default: return "";
  }
}

// masks and overlays are transparent PNGs: show them over the photo, as they are meant to be read
function underlay(img, layer, job) {
  img.style.background = layer && (layer.kind === "overlay" || layer.kind === "class")
    ? `url("/jobs/${job}/files/layer_rgb.png") center / 100% 100% no-repeat` : "";
}

function thumb(src, caption, cls, layerId, layer, job) {
  const fig = el("figure");
  const img = el("img", cls || ""); img.src = src; img.alt = caption; img.loading = "lazy";
  underlay(img, layer, job);
  if (layerId) {
    const b = el("button"); b.title = `Open “${caption}” in Layers`; b.append(img);
    b.onclick = () => { if (!$("tab-layers").disabled) { setB(layerId); setView("layers"); } };
    fig.append(b);
  } else fig.append(img);
  fig.append(el("figcaption", "", caption));
  return fig;
}

async function applyEvent(ev, job) {
  if (ev.type === "done") {
    live.done = true;
    $("proc-clock").textContent = `${fmt(ev.seconds, 1)} s total`;
    $("proc-sub").textContent = `Finished · ${ev.product?.kind || ""}`;
    $("open-3d").hidden = false;
    return;
  }
  const c = live.cards[ev.stage]; if (!c) return;
  if (ev.title) c.h3.textContent = ev.title;
  if (ev.type === "stage_start") {
    c.li.className = "st running"; c.sum.textContent = "Running…";
    c.thumbs.replaceChildren(el("div", "ph wait"), el("div", "ph wait"));
    $("proc-sub").textContent = `Running: ${c.h3.textContent}`;
  } else if (ev.type === "stage_done") {
    c.li.className = "st done"; c.secs.textContent = `${fmt(ev.seconds, ev.seconds < 10 ? 2 : 1)} s`;
    c.sum.textContent = summarize(ev.stage, ev.summary);
    const base = `/jobs/${job}/files/`;
    let layers = [];
    if (ev.layers?.length) layers = await api(`${base}layers.json?t=${ev.seq}`).catch(() => []);
    const figs = (ev.layers || []).map((id) => layers.find((l) => l.id === id)).filter(Boolean)
      .map((l) => thumb(`${base}${l.preview}?v=${ev.seq}`, l.title, "", l.id, l, job));
    for (const p of ev.summary?.plots || []) {
      figs.push(thumb(`${base}${p}?v=${ev.seq}`, p.includes("scatter") ? "Fit: low-pass model vs DEM" : "Held-out residuals", "plot"));
    }
    c.thumbs.replaceChildren(...figs);
  } else if (ev.type === "stage_skipped") {
    c.li.className = "st skipped"; c.sum.textContent = ev.reason || "Skipped"; c.thumbs.replaceChildren();
  } else if (ev.type === "stage_failed") {
    c.li.className = "st failed"; c.sum.textContent = ev.error || "Failed"; c.thumbs.replaceChildren();
  }
  const settled = Object.values(live.cards).filter((x) => /done|skipped|failed/.test(x.li.className)).length;
  $("bar").style.width = `${Math.round(100 * settled / STAGES.length)}%`;
  $("progress-text").textContent = `${settled} of ${STAGES.length} stages · ${c.h3.textContent}`;
}

function renderPipelineFrom(events, job) {
  buildPipeline();
  live.done = false;
  $("proc-title").textContent = short(job.original_name || job.input, 60);
  $("proc-sub").textContent = events.length ? "Finished" : "No stage log for this job (it predates the event log).";
  $("proc-clock").textContent = "";
  $("open-3d").hidden = true;
  live.chain = events.reduce((p, ev) => p.then(() => applyEvent(ev, job.id)), Promise.resolve());
}

function startLive(id, name) {
  live.active = true; live.job = id; live.last = -1; live.done = false;
  $("tab-layers").disabled = true;           // layers of the previous job must not be shown for this one
  buildPipeline();
  $("proc-title").textContent = short(name, 60);
  $("proc-sub").textContent = "Queued";
  $("open-3d").hidden = true;
  live.t0 = performance.now();
  clearInterval(live.timer);
  live.timer = setInterval(() => { if (!live.done) $("proc-clock").textContent = `${fmt((performance.now() - live.t0) / 1000, 1)} s`; }, 200);
  setView("process");
  return new Promise((resolve, reject) => {
    const handle = (ev) => { live.last = Math.max(live.last, ev.seq); live.chain = live.chain.then(() => applyEvent(ev, id)); };
    const finish = async (status, error) => {
      clearInterval(live.timer); live.active = false;
      await live.chain;
      if (status === "done") resolve(); else reject(new Error(error || "job failed"));
    };
    const poll = async () => {        // fallback when EventSource is unavailable or drops
      for (;;) {
        const r = await api(`/jobs/${id}/events?after=${live.last}`);
        r.events.forEach(handle);
        if (r.status === "done" || r.status === "failed") return finish(r.status, r.error);
        await new Promise((ok) => setTimeout(ok, 700));
      }
    };
    if (!("EventSource" in window)) { poll().catch(reject); return; }
    const es = new EventSource(`/jobs/${id}/stream`);
    live.es = es;
    es.onmessage = (e) => handle(JSON.parse(e.data));
    es.addEventListener("end", (e) => { es.close(); const d = JSON.parse(e.data); finish(d.status, d.error); });
    es.onerror = () => { if (es.readyState === EventSource.CLOSED || live.active) { es.close(); poll().catch(reject); } };
  });
}
$("open-3d").onclick = () => setView("3d");

// ------------------------------------------------------------------ layers view: swipe compare + filmstrip
const c2 = { a: null, b: null, split: 0.5, s: 1, tx: 0, ty: 0, W0: 1024, H0: 1024, fitted: false };
const layerById = (id) => state.layers.find((l) => l.id === id);
const layerUrl = (l) => `/jobs/${state.job}/files/${l.preview}`;

function legendHTML(node, l) {
  node.replaceChildren();
  if (!l) return;
  const t = el("div", "t", l.title);
  const u = el("div", "u", [l.kind, l.units, l.datum].filter(Boolean).join(" · "));
  node.append(t, u);
  if (l.colormap && l.vmin !== null && l.vmin !== undefined) {
    const r = el("div", "ramp"); r.style.background = ramp(l.colormap); node.append(r);
    const unit = l.units && !/relative|class/.test(l.units) ? ` ${l.units === "degrees" ? "°" : l.units}` : "";
    const ends = el("div", "ends");
    ends.append(el("span", "", fmt(l.vmin, Math.abs(l.vmax - l.vmin) < 10 ? 2 : 1) + unit),
      el("span", "", fmt(l.vmax, Math.abs(l.vmax - l.vmin) < 10 ? 2 : 1) + unit));
    node.append(ends);
  }
  if (l.notes?.length) node.append(el("div", "u", short(l.notes[0], 70)));
}

function setA(id) { const l = layerById(id); if (!l) return; c2.a = id; $("img-a").src = layerUrl(l); underlay($("img-a"), l, state.job); $("tag-a").textContent = l.title; legendHTML($("legend-a"), l); markFilm(); }
function setB(id) { const l = layerById(id); if (!l) return; c2.b = id; $("img-b").src = layerUrl(l); underlay($("img-b"), l, state.job); $("tag-b").textContent = l.title; legendHTML($("legend-b"), l); markFilm(); }

function buildFilmstrip() {
  const L = [...state.layers].sort((x, y) => (STAGE_ORDER[x.stage] ?? 99) - (STAGE_ORDER[y.stage] ?? 99));
  $("filmstrip").replaceChildren(...L.map((l) => {
    const f = el("div", "film"); f.dataset.id = l.id; f.setAttribute("role", "listitem");
    const img = el("img"); img.src = layerUrl(l); img.alt = l.title; img.loading = "lazy"; img.title = `${l.title}: click to show on the right (B)`;
    underlay(img, l, state.job);
    img.onclick = () => setB(l.id);
    const ab = el("div", "ab");
    for (const side of ["A", "B"]) {
      const b = el("button", "", side); b.type = "button"; b.setAttribute("aria-pressed", "false");
      b.setAttribute("aria-label", `Show ${l.title} as ${side}`);
      b.onclick = () => (side === "A" ? setA : setB)(l.id);
      ab.append(b);
    }
    const kind = el("span", `kind ${l.kind}`, l.kind);
    f.append(img, ab, kind, el("div", "ft", l.title));
    return f;
  }));
  const md = state.md;
  const m = Math.max(md.input.width, md.input.height);
  c2.W0 = 1024 * md.input.width / m; c2.H0 = 1024 * md.input.height / m;
  for (const img of [$("img-a"), $("img-b")]) { img.style.width = `${c2.W0}px`; img.style.height = `${c2.H0}px`; }
  c2.fitted = false;
  setA(layerById("rgb") ? "rgb" : state.layers[0]?.id);
  setB(layerById("dsm") ? "dsm" : state.layers[1]?.id || state.layers[0]?.id);
  $("readout2d").hidden = true;
  if (state.view === "layers") fit2d();
}
function markFilm() {
  document.querySelectorAll(".film").forEach((f) => {
    f.classList.toggle("is-a", f.dataset.id === c2.a); f.classList.toggle("is-b", f.dataset.id === c2.b);
    const [ba, bb] = f.querySelectorAll(".ab button");
    ba.setAttribute("aria-pressed", String(f.dataset.id === c2.a)); bb.setAttribute("aria-pressed", String(f.dataset.id === c2.b));
  });
}
function fit2d() {
  const r = $("compare").getBoundingClientRect();
  if (!r.width) return;
  c2.s = Math.min(r.width / c2.W0, r.height / c2.H0) * 0.96;
  c2.tx = (r.width - c2.W0 * c2.s) / 2; c2.ty = (r.height - c2.H0 * c2.s) / 2;
  c2.fitted = true; apply2d();
}
function apply2d() {
  const t = `translate(${c2.tx}px, ${c2.ty}px) scale(${c2.s})`;
  $("img-a").style.transform = t; $("img-b").style.transform = t;
  const w = $("compare").clientWidth;
  $("pane-b").style.clipPath = `inset(0 0 0 ${c2.split * w}px)`;
  $("divider").style.left = `${c2.split * 100}%`;
  $("divider").setAttribute("aria-valuenow", String(Math.round(c2.split * 100)));
}
new ResizeObserver(() => { if (state.view === "layers" && c2.fitted) apply2d(); }).observe($("compare"));
$("reset-2d").onclick = fit2d;

{ // divider drag, pan, zoom, click-to-inspect
  const cmp = $("compare"), div = $("divider");
  let mode = null, start = null;
  div.addEventListener("pointerdown", (e) => { mode = "split"; div.setPointerCapture(e.pointerId); e.stopPropagation(); });
  div.addEventListener("pointermove", (e) => { if (mode !== "split") return;
    const r = cmp.getBoundingClientRect(); c2.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); apply2d(); });
  div.addEventListener("pointerup", () => { mode = null; });
  div.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: -0.02, ArrowRight: 0.02 }[e.key]; if (!d) return;
    c2.split = Math.min(1, Math.max(0, c2.split + d)); apply2d(); e.preventDefault(); });
  cmp.addEventListener("pointerdown", (e) => { if (e.target.closest("button, .divider")) return;
    mode = "pan"; start = [e.clientX, e.clientY, c2.tx, c2.ty]; cmp.setPointerCapture(e.pointerId); cmp.classList.add("dragging"); });
  cmp.addEventListener("pointermove", (e) => { if (mode !== "pan") return;
    c2.tx = start[2] + e.clientX - start[0]; c2.ty = start[3] + e.clientY - start[1]; apply2d(); });
  cmp.addEventListener("pointerup", (e) => {
    cmp.classList.remove("dragging");
    const wasPan = mode === "pan"; mode = null;
    if (wasPan && Math.hypot(e.clientX - start[0], e.clientY - start[1]) < 4) inspect2d(e);
  });
  cmp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = cmp.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const k = Math.exp(-e.deltaY * 0.0015), s = Math.min(40, Math.max(0.1, c2.s * k));
    c2.tx = x - (x - c2.tx) * s / c2.s; c2.ty = y - (y - c2.ty) * s / c2.s; c2.s = s; apply2d();
  }, { passive: false });
}

async function inspect2d(e) {
  const r = $("compare").getBoundingClientRect();
  const fx = (e.clientX - r.left - c2.tx) / (c2.W0 * c2.s), fy = (e.clientY - r.top - c2.ty) / (c2.H0 * c2.s);
  if (fx < 0 || fy < 0 || fx >= 1 || fy >= 1) return;
  const row = Math.floor(fy * state.md.input.height), col = Math.floor(fx * state.md.input.width);
  const box = $("readout2d"); box.hidden = false;
  try {
    const q = await api(`/jobs/${state.job}/point?row=${row}&col=${col}`);
    const metric = q.kind === "absolute";
    const d = el("dl");
    dl(d, [[metric ? "Elevation" : "Relative", metric ? `${fmt(q.value, 2)} m` : fmt(q.value, 3)],
      metric && ["Above ground", `${fmt(q.derived_height_above_ground, 2)} m · derived`],
      metric && q.derived_height_confidence && ["Confidence", q.derived_height_confidence.level],
      ["Pixel", `${row.toLocaleString()}, ${col.toLocaleString()}`]]);
    box.replaceChildren(el("div", "lab", "Pixel inspector"), d);
    if (q.derived_height_confidence?.level === "low") box.append(el("p", "hint", `Low confidence: ${q.derived_height_confidence.reasons.join("; ")}`));
  } catch (err) { box.replaceChildren(el("p", "hint", `Could not read that point: ${err.message}`)); }
}

// ------------------------------------------------------------------ 3D tools and overlays
function pressed(selector, attr, value) {
  document.querySelectorAll(selector).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset[attr] === value)));
}
document.querySelectorAll("[data-nav]").forEach((b) => b.onclick = () => {
  nav = b.dataset.nav; pressed("[data-nav]", "nav", nav);
  orbit.enabled = nav === "orbit";
  if (nav === "fly") { $("fb-hint").textContent = "Click the view to fly. W A S D move, Q E down and up, Shift is faster, Esc exits."; }
});
renderer.domElement.addEventListener("click", () => { if (nav === "fly" && !fly.isLocked) fly.lock(); });
addEventListener("keydown", (e) => keys.add(e.code));
addEventListener("keyup", (e) => keys.delete(e.code));

document.querySelectorAll("[data-tool]").forEach((b) => b.onclick = () => {
  state.tool = b.dataset.tool; pressed("[data-tool]", "tool", state.tool);
  clearMeasure();
  $("fb-title").textContent = state.tool === "measure" ? "Measure" : "Point";
  dl($("fb-dl"), []); $("fb-conf").hidden = true;
  $("fb-hint").textContent = state.tool === "measure" ? "Click two points on the surface." : "Click the surface to read its height.";
});

async function setOverlay(name) {
  if (!state.terrain) return;
  if (!state.textures[name]) {
    const base = `/jobs/${state.job}/files/`;
    if (name === "slope") state.textures.slope = await loadTex(base + "slope.png");
    if (name === "diff") state.textures.diff = await compositeDiff(base + "reference_diff.png");
  }
  state.overlay = name; pressed("[data-overlay]", "overlay", name);
  for (const m of state.inner) { m.material.map = state.textures[name]; m.material.needsUpdate = true; }
  renderLegend();
}
document.querySelectorAll("[data-overlay]").forEach((b) => b.onclick = () => !b.disabled && setOverlay(b.dataset.overlay));

async function compositeDiff(url) {
  // photo underneath, difference colours on top where the reference has data
  const photo = state.textures.photo.image;
  const diff = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url + `?t=${Date.now()}`; });
  const c = document.createElement("canvas"); c.width = photo.width; c.height = photo.height;
  const g = c.getContext("2d");
  g.drawImage(photo, 0, 0); g.globalAlpha = 0.9; g.drawImage(diff, 0, 0, c.width, c.height);
  const t = new THREE.CanvasTexture(c); t.flipY = false; t.colorSpace = THREE.SRGBColorSpace; return t;
}

function renderLegend() {
  const L = $("legend");
  if (state.overlay === "slope") {
    legendHTML(L, { title: "Slope", kind: "derived", units: "degrees", colormap: "viridis", vmin: 0, vmax: 60,
      notes: ["60° and steeper shown as the top colour"] });
    L.hidden = false;
  } else if (state.overlay === "diff" && state.refLimit) {
    legendHTML(L, { title: "DSM minus reference", kind: "derived", units: "m", colormap: "berlin",
      vmin: -state.refLimit, vmax: state.refLimit, notes: ["blue: DSM below reference · red: above"] });
    L.hidden = false;
  } else { L.hidden = true; }
}

$("exag").oninput = applyExaggeration;
function applyExaggeration() {
  state.exag = Number($("exag").value);
  $("exag-out").textContent = `${state.exag.toFixed(1)}×`;
  for (const m of state.inner) m.scale.z = state.exag;   // data z = up; display only
  updateMeasureGraphics();
}

// ------------------------------------------------------------------ picking and measuring
const ray = new THREE.Raycaster();
function pick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  const p = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(p, camera);
  const hits = ray.intersectObjects(state.inner.filter((m) => m.visible && m.parent?.parent?.visible !== false), false);
  return hits[0];
}

function toPixel(hit) {
  // local mesh coordinates (metres or pixels from the upper-left corner), unaffected by exaggeration
  const local = hit.object.worldToLocal(hit.point.clone());
  const sx = state.md.input.georeferenced ? state.md.input.gsd_m[0] : 1;
  const sy = state.md.input.georeferenced ? state.md.input.gsd_m[1] : 1;
  return { col: Math.floor(local.x / sx), row: Math.floor(-local.y / sy), lx: local.x, ly: local.y };
}

function showConfidence(conf) {
  const cb = $("fb-conf");
  cb.hidden = !conf;
  if (!conf) return;
  cb.className = `conf ${conf.level}`;
  const head = el("strong", "", conf.level === "low" ? "Low confidence height" : `Height confidence: ${conf.level}`);
  const why = el("span", "", conf.reasons.length ? conf.reasons.join("; ") : "No known local problem; still derived, not measured.");
  const me = conf.measured_error;
  const ev = el("span", "", me ? `Tested error at this level: RMSE ${fmt(me.rmse_m, 1)} m, bias ${fmt(me.bias_m, 1)} m (US test sites vs LiDAR).` : "");
  cb.replaceChildren(head, why, ev);
}

let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", async (e) => {
  if (!state.terrain || nav !== "orbit" || !downAt) return;
  if (Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;   // it was a drag
  const hit = pick(e); if (!hit) return;
  const px = toPixel(hit);
  let q;
  try { q = await api(`/jobs/${state.job}/point?row=${px.row}&col=${px.col}`); }
  catch (err) { $("fb-hint").textContent = `Could not read that point: ${err.message}`; return; }
  const metric = q.kind === "absolute";
  if (state.tool === "inspect") {
    $("fb-title").textContent = "Point";
    dl($("fb-dl"), [
      [metric ? "Elevation" : "Relative value", metric ? `${fmt(q.value, 2)} m` : fmt(q.value, 3)],
      metric && ["Datum", (q.vertical_datum || "").split(" (")[0]],
      metric && ["Ground estimate", `${fmt(q.ground_estimate, 2)} m · derived`],
      metric && ["Height above ground", `${fmt(q.derived_height_above_ground, 2)} m · derived`],
      ["Pixel", `row ${q.row.toLocaleString()}, col ${q.col.toLocaleString()}`],
      metric && ["Map x, y", `${fmt(q.x, 1)}, ${fmt(q.y, 1)}`],
    ]);
    $("fb-dl").querySelector("dd")?.classList.add("big");
    showConfidence(metric ? q.derived_height_confidence : null);
    $("fb-hint").textContent = metric
      ? "Derived values use an estimated ground and are not measured. No derived height is rated high confidence."
      : "Relative values have no units; they only compare heights within this image.";
    setMarkers([{ hit, px, q }]);
  } else {
    if (state.measure.length >= 2) state.measure = [];
    state.measure.push({ hit, px, q });
    setMarkers(state.measure);
    showMeasure(metric);
  }
});

function showMeasure(metric) {
  const m = state.measure;
  $("fb-title").textContent = "Measure";
  $("fb-conf").hidden = true;
  if (m.length < 2) { dl($("fb-dl"), [["Point A", metric ? `${fmt(m[0].q.value, 2)} m` : fmt(m[0].q.value, 3)]]);
    $("fb-hint").textContent = "Click a second point."; return; }
  const [a, b] = m;
  const horiz = Math.hypot(b.px.lx - a.px.lx, b.px.ly - a.px.ly);
  const dz = b.q.value - a.q.value;
  dl($("fb-dl"), metric ? [
    ["Height difference", `${dz >= 0 ? "+" : "−"}${fmt(Math.abs(dz), 2)} m`],
    ["Horizontal distance", `${fmt(horiz, 1)} m`],
    ["Slope", `${fmt(Math.atan2(Math.abs(dz), horiz) * 180 / Math.PI, 1)}°`],
    ["A, B elevation", `${fmt(a.q.value, 1)} m, ${fmt(b.q.value, 1)} m`],
  ] : [["Relative difference", fmt(dz, 3)], ["Distance", `${fmt(horiz, 0)} px`]]);
  $("fb-dl").querySelector("dd")?.classList.add("big");
  $("fb-hint").textContent = metric ? "True heights from the DSM; the exaggeration slider does not change them."
    : "No units: this image has no map coordinates.";
}

// markers + a cyan dimension line, redrawn when exaggeration changes
let markerGroup = null;
function setMarkers(points) { state.markers = points; updateMeasureGraphics(); }
function clearMeasure() { state.measure = []; state.markers = []; updateMeasureGraphics(); }
function updateMeasureGraphics() {
  if (markerGroup) { scene.remove(markerGroup); markerGroup.traverse((o) => o.geometry?.dispose()); markerGroup = null; }
  $("labels").replaceChildren();
  if (!state.markers?.length || !state.terrain) return;
  markerGroup = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: 0x3cc8dc, depthTest: false });
  const world = state.markers.map(({ hit }) => {
    const local = hit.object.worldToLocal(hit.point.clone());
    return hit.object.localToWorld(local.clone());          // re-evaluated under the current exaggeration
  });
  for (const w of world) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(state.size * 0.004, 16, 12), mat);
    s.position.copy(w); s.renderOrder = 10; markerGroup.add(s);
  }
  if (world.length === 2) {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(world),
      new THREE.LineBasicMaterial({ color: 0x3cc8dc, depthTest: false }));
    line.renderOrder = 10; markerGroup.add(line);
    state.labelAt = world[0].clone().lerp(world[1], 0.5);
  } else { state.labelAt = world[0]; }
  scene.add(markerGroup);
}
function drawLabel() {
  const layer = $("labels");
  if (!state.labelAt || !state.markers?.length) return;
  const v = state.labelAt.clone().project(camera);
  const r = renderer.domElement.getBoundingClientRect();
  let lab = layer.firstChild;
  if (!lab) { lab = el("div", "dim-label"); layer.appendChild(lab); }
  const first = $("fb-dl").querySelector("dd");
  lab.textContent = first ? first.textContent : "";
  lab.style.left = `${(v.x + 1) / 2 * r.width}px`; lab.style.top = `${(1 - v.y) / 2 * r.height}px`;
  lab.style.display = v.z < 1 ? "block" : "none";
}

// ------------------------------------------------------------------ reference comparison
$("ref-file").onchange = async (e) => {
  const f = e.target.files[0]; if (!f || !state.job) return;
  dl($("ref-dl"), [["Status", "Comparing…"]]);
  const fd = new FormData(); fd.append("file", f);
  try {
    const r = await api(`/jobs/${state.job}/reference`, { method: "POST", body: fd });
    const m = r.aggregated_to_reference_grid && typeof r.aggregated_to_reference_grid === "object"
      ? r.aggregated_to_reference_grid : r.on_dsm_grid;
    const where = m === r.on_dsm_grid ? "on the DSM grid" : "averaged to the reference grid";
    dl($("ref-dl"), [["Compared", where], ["Cells", m.n?.toLocaleString()], ["Bias", `${fmt(m.bias, 2)} m`],
      ["MAE", `${fmt(m.mae, 2)} m`], ["RMSE", `${fmt(m.rmse, 2)} m`], ["Correlation r", fmt(m.pearson_r, 3)],
      ["Datum check", r.datum_check.split(" ")[0].toLowerCase()]]);
    state.refLimit = r.diff_colour_limit_m;
    state.textures.diff = null;
    const b = document.querySelector('[data-overlay="diff"]'); b.disabled = false;
    setOverlay("diff");
    state.layers = await api(`/jobs/${state.job}/files/layers.json?t=${Date.now()}`).catch(() => state.layers);
    const keepA = c2.a; buildFilmstrip(); setA(keepA); if (layerById("diff")) setB("diff");
  } catch (err) { dl($("ref-dl"), [["Could not compare", err.message]]); }
  e.target.value = "";
};

// ------------------------------------------------------------------ upload + live job
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { $("file").files = e.dataTransfer.files; $("file").dispatchEvent(new Event("change")); });
$("file").onchange = () => { $("drop-text").textContent = $("file").files[0]?.name || "Drop a GeoTIFF, PNG or JPG"; };

$("upload").onsubmit = async (e) => {
  e.preventDefault();
  const f = $("file").files[0]; if (!f) return;
  $("error").hidden = true; $("go").disabled = true; $("progress").hidden = false;
  $("bar").style.width = "0"; $("progress-text").textContent = "Uploading";
  const fd = new FormData();
  fd.append("file", f); ["model", "calibration", "dem_source", "band_order"].forEach((k) => fd.append(k, $(k).value));
  try {
    const { id } = await api("/jobs", { method: "POST", body: fd });
    $("progress-text").textContent = "Queued";
    await startLive(id, f.name);
    await refreshJobs(id);
    await showJob(id);
    $("progress-text").textContent = "Done";
  } catch (err) { $("error").textContent = `The surface model could not be built: ${err.message}`; $("error").hidden = false; }
  finally { $("go").disabled = false; }
};

async function refreshJobs(select) {
  const jobs = (await api("/jobs")).filter((j) => j.status === "done").sort((a, b) => b.created.localeCompare(a.created));
  const s = $("jobs");
  s.replaceChildren(...(jobs.length ? jobs : [null]).map((j) => {
    const o = el("option");
    if (!j) { o.value = ""; o.textContent = "None yet"; return o; }
    o.value = j.id; o.textContent = `${short(j.name, 28)} · ${new Date(j.created).toLocaleString()}`; return o;
  }));
  if (select) s.value = select;
}
$("jobs").onchange = (e) => e.target.value && showJob(e.target.value);

// ------------------------------------------------------------------ render loop + frame timing
const clock = new THREE.Clock();
let frames = 0, acc = 0;
window.__dw = { fps: 0, state, camera, orbit, live, c2, setView };
function loop() {
  requestAnimationFrame(loop);
  const dt = clock.getDelta();
  if (state.view !== "3d") return;            // the 3D view is hidden: don't spend the GPU
  if (nav === "fly" && fly.isLocked) {
    const v = state.size * 0.25 * dt * (keys.has("ShiftLeft") ? 3 : 1);
    if (keys.has("KeyW")) fly.moveForward(v);
    if (keys.has("KeyS")) fly.moveForward(-v);
    if (keys.has("KeyD")) fly.moveRight(v);
    if (keys.has("KeyA")) fly.moveRight(-v);
    if (keys.has("KeyE")) camera.position.y += v;
    if (keys.has("KeyQ")) camera.position.y -= v;
  } else if (orbit.enabled) { orbit.update(); }
  renderer.render(scene, camera);
  drawLabel();
  frames++; acc += dt;
  if (acc >= 1) { window.__dw.fps = frames / acc; $("perf").textContent = `${Math.round(window.__dw.fps)} FPS`; frames = 0; acc = 0; }
}
loop();

// open the newest finished job, if any, or a job named in the URL (?job=<id>&view=process|layers|3d)
buildPipeline();
refreshJobs().then(() => {
  const params = new URLSearchParams(location.search);
  const want = params.get("job") || $("jobs").value;
  const view = params.get("view");
  if (want) { $("jobs").value = want; showJob(want, { view: view || "3d" }); }
  else setView("3d");
});
