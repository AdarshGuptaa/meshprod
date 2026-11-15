import * as THREE from "three";
import { PLYLoader } from "./vendor/PLYLoader.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

if (new URLSearchParams(location.search).has("static")) {
  document.body.classList.add("static");
}

const STAGES = ["decode", "features", "match", "verify", "tracks", "sparse", "dense", "fusion", "mesh"];

const DEFAULT_CONFIG = {
  fx: 895, fy: 895, cx: 640, cy: 360,
  detector: "sift", frame_step: 10, n_features: 500, match_window: 5, ratio: 0.8,
  geo_verify: true, geo_threshold: 3.0, geo_min_inliers: 15, wide_baseline: 0,
  sprt: true, refine_intrinsics: false, auto_config: true, remove_plane: false,
  dense_scale: 0.5, dense_planes: 96, dense_window: 5, dense_neighbors: 3,
  ncc_min: 0.4, lr_required: 1, cost: "ncc", sgm_paths: 4,
  voxel_resolution: 256, trunc_voxels: 3.0, volume_smoothing: 2,
  hole_fill_iterations: 0, smooth_iterations: 8, bilateral_iterations: 2,
  needle_aspect: 0.0, target_faces: 100000, min_component_faces: 200,
};

const SCHEMA = [
  {
    title: "Camera", icon: "i-camera", open: true,
    items: [
      { key: "fx", label: "fx", type: "number", min: 100, max: 4000, step: 0.5,
        help: "Focal length in x, in pixels. f_px = f_35mm / 36 × frame width (about 895 px for 1280×720 phone video)." },
      { key: "fy", label: "fy", type: "number", min: 100, max: 4000, step: 0.5,
        help: "Focal length in y. Equal to fx for square pixels." },
      { key: "cx", label: "cx", type: "number", min: 0, max: 4000, step: 0.5,
        help: "Principal point x. The image centre is the standard guess for a phone lens." },
      { key: "cy", label: "cy", type: "number", min: 0, max: 4000, step: 0.5,
        help: "Principal point y. The image centre is the standard guess for a phone lens." },
    ],
  },
  {
    title: "Matching", icon: "i-layers", open: false,
    items: [
      { key: "detector", label: "Detector", type: "select", options: ["sift", "orb"],
        help: "SIFT is scale- and rotation-invariant and the safer choice for phone orbits. ORB uses binary descriptors: much faster, weaker across large viewpoint changes." },
      { key: "frame_step", label: "Frame step", type: "range", min: 1, max: 30, step: 1,
        help: "Keep every Nth frame. Higher means larger baselines, faster runs and fewer overlapping matches. 10 suits 30–120 fps clips." },
      { key: "n_features", label: "Max features", type: "range", min: 100, max: 2000, step: 100,
        help: "Keypoints per frame. More gives denser tracks and better registration; matching cost grows linearly." },
      { key: "match_window", label: "Match window", type: "range", min: 1, max: 12, step: 1,
        help: "How many following frames each frame is matched against. Larger bridges fast motion but admits more outliers." },
      { key: "ratio", label: "Lowe ratio", type: "range", min: 0.6, max: 0.95, step: 0.01,
        help: "Ratio between the best and second-best descriptor. Lower is stricter: fewer false matches, fewer matches overall. 0.8 is standard." },
      { key: "geo_verify", label: "Epipolar check", type: "toggle",
        help: "RANSAC fundamental-matrix check per frame pair. Removes descriptor false positives at a small time cost." },
      { key: "sprt", label: "SPRT", type: "toggle",
        help: "Wald sequential test during PnP: rejects hopeless poses early. Same decision, less time." },
      { key: "refine_intrinsics", label: "Refine K", type: "toggle",
        help: "Fits a global focal scale plus radial/tangential distortion against the solve. Useful when intrinsics were guessed." },
      { key: "auto_config", label: "Auto-config", type: "toggle",
        help: "Measures frame texture and picks cost, window, NCC threshold and plane count. Overrides those fields; every choice is logged." },
    ],
  },
  {
    title: "Dense depth", icon: "i-chart", open: true,
    items: [
      { key: "cost", label: "Cost", type: "select", options: ["ncc", "census", "sgm"],
        help: "ncc = normalized cross-correlation. census = intensity-ordering comparison, better on gentle shading. sgm = census plus a smoothness prior that fills low-texture regions." },
      { key: "dense_scale", label: "Scale", type: "range", min: 0.25, max: 1, step: 0.05,
        help: "Depth-map resolution relative to the source. 0.5 is half resolution and roughly 4× faster." },
      { key: "dense_planes", label: "Planes", type: "range", min: 32, max: 160, step: 8,
        help: "Inverse-depth planes swept per pixel. More means finer depth resolution, linearly slower." },
      { key: "dense_window", label: "Window", type: "range", min: 3, max: 7, step: 2,
        help: "Photometric window side. Larger is more robust to noise but blurs depth edges." },
      { key: "dense_neighbors", label: "Neighbours", type: "range", min: 1, max: 5, step: 1,
        help: "Neighbouring views each view compares against. More is more robust, and slower." },
      { key: "ncc_min", label: "Min NCC", type: "range", min: 0.2, max: 0.7, step: 0.05,
        help: "Minimum similarity for a depth pixel. Lower keeps more pixels (more coverage, more noise)." },
      { key: "lr_required", label: "Agreements", type: "range", min: 1, max: 3, step: 1,
        help: "Neighbours that must independently agree. 2 is stricter and more accurate but halves coverage on noisy captures." },
      { key: "sgm_paths", label: "SGM paths", type: "range", min: 1, max: 8, step: 1,
        help: "Aggregation directions for SGM. Four cardinals are the default; eight adds diagonals (better smoothing, slower)." },
    ],
  },
  {
    title: "Fusion", icon: "i-cube", open: false,
    items: [
      { key: "voxel_resolution", label: "Resolution", type: "range", min: 64, max: 384, step: 16,
        help: "Grid points along the longest scene axis. Higher means a finer mesh, more memory and time." },
      { key: "trunc_voxels", label: "Truncation", type: "range", min: 1, max: 8, step: 0.5,
        help: "Truncation band in voxels. It must exceed the depth noise; too small and views disagree, tearing the surface." },
      { key: "volume_smoothing", label: "Field smooth", type: "range", min: 0, max: 4, step: 1,
        help: "Box-filter passes on the signed distance field. Removes voxel-scale pyramids before meshing." },
      { key: "hole_fill_iterations", label: "Hole fill", type: "range", min: 0, max: 8, step: 1,
        help: "Diffusion passes into unobserved voxels. Off by default: diffusion invents surface where no camera looked." },
      { key: "remove_plane", label: "Remove plane", type: "toggle",
        help: "RANSAC-removes the dominant plane (table, floor) from the dense cloud before fusion." },
    ],
  },
  {
    title: "Mesh", icon: "i-gear", open: false,
    items: [
      { key: "smooth_iterations", label: "Taubin", type: "range", min: 0, max: 16, step: 1,
        help: "Alternating Laplacian passes (λ/μ) that smooth without shrinking. Eight is a good default." },
      { key: "bilateral_iterations", label: "Bilateral", type: "range", min: 0, max: 4, step: 1,
        help: "Feature-preserving passes: flattens noise but keeps edges such as keycaps and blade bevels." },
      { key: "needle_aspect", label: "Needle", type: "range", min: 0, max: 30, step: 1,
        help: "Drop triangles above this longest/shortest edge ratio. Zero disables it; a strict value opens many edges on noisy clouds." },
      { key: "target_faces", label: "Faces", type: "range", min: 5000, max: 150000, step: 5000,
        help: "Quadric edge-collapse budget. Lower gives a simpler, lighter mesh." },
      { key: "min_component_faces", label: "Min patch", type: "range", min: 0, max: 2000, step: 50,
        help: "Drop connected surface patches smaller than this. Higher cleans floaters but may remove real surface." },
    ],
  },
];

for (const group of SCHEMA) {
  for (const item of group.items) {
    item.group = group.title;
    item.icon = group.icon;
  }
}

const PRESETS = {
  fast: { frame_step: 20, dense_scale: 0.35, dense_planes: 64, voxel_resolution: 192, target_faces: 20000 },
  balanced: {},
  quality: { frame_step: 5, dense_scale: 0.75, dense_planes: 128, voxel_resolution: 320, target_faces: 80000 },
  auto: { auto_config: true },
};

const ALTAI = [
  ["Human agency", "Every parameter is exposed and logged; a person decides when a mesh is usable. The pipeline never auto-accepts a result.", "good"],
  ["Robustness", "Deterministic reruns, depth-vs-sparse checks, a mesh-quality pack and named failures. Degradation is visible, not silent.", "good"],
  ["Privacy", "Bound to 127.0.0.1. No uploads, no telemetry; runs stay in web_runs/ on this machine.", "good"],
  ["Transparency", "Live logs, derived parameters, failure codes with hints, and exportable run configs.", "good"],
  ["Fairness", "Four captures evaluated (specular, textureless, textured, thin). Dark, transparent and deformable objects are still unmeasured.", "warn"],
  ["Environment", "CPU-only path, about a minute per scan; resource use is recorded in every run report.", "good"],
  ["Accountability", "Provenance hashes, measured claims, pinned dependency, MIT license and AI disclosure.", "good"],
];

const SEED_HISTORY = [
  { name: "swiss_knife", faces: 86971, depth_median: 4.0, patches: 33, views: 22 },
  { name: "lip_balm", faces: 19581, depth_median: 14.8, patches: 41, views: 66 },
  { name: "phone_stand", faces: 21925, depth_median: 9.1, patches: 20, views: 140 },
  { name: "keyboard", faces: 4880, depth_median: 4.5, patches: 4, views: 63 },
];

const GOLD = "#d8b25c";
const STEEL = "#7fa8c9";
const INK = "#ece4d2";

let config = { ...DEFAULT_CONFIG };
let stats = null;
let lastMeshKey = null;
let chartData = SEED_HISTORY;
let videoFile = null;
let selectedKey = null;
let viewerActive = true;

/* ---------- tabs ---------- */
const tabs = [...document.querySelectorAll(".tab")];
tabs.forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
function activateTab(name) {
  viewerActive = name === "studio";
  tabs.forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "slides") resizeSlides();
  history.replaceState(null, "", `#${name}`);
}

/* ---------- inventory slots ---------- */
function renderGroups() {
  const host = document.getElementById("config-groups");
  host.innerHTML = "";
  for (const group of SCHEMA) {
    const details = document.createElement("details");
    details.className = "group";
    details.open = group.open;
    const summary = document.createElement("summary");
    summary.textContent = group.title;
    details.appendChild(summary);
    const grid = document.createElement("div");
    grid.className = "slot-grid";
    for (const item of group.items) grid.appendChild(slotEl(item));
    details.appendChild(grid);
    host.appendChild(details);
  }
  if (selectedKey) {
    document.querySelectorAll(".slot").forEach((s) => s.classList.toggle("selected", s.dataset.key === selectedKey));
  }
}

function slotEl(item) {
  const btn = document.createElement("button");
  btn.className = "slot";
  btn.dataset.key = item.key;
  btn.type = "button";
  if (item.type === "select" || item.type === "toggle") btn.classList.add("no-bar");
  btn.innerHTML = `
    <span class="slot-top"><svg class="icon"><use href="#${item.icon}"/></svg><span class="slot-name">${item.label}</span></span>
    <span class="slot-value">${formatValue(config[item.key])}</span>
    <span class="slot-bar"><i></i></span>`;
  updateSlotEl(btn, item);
  const select = () => openDetails(item);
  btn.addEventListener("mouseenter", select);
  btn.addEventListener("focus", select);
  btn.addEventListener("click", select);
  return btn;
}

function updateSlotEl(btn, item) {
  const value = config[item.key];
  btn.querySelector(".slot-value").textContent = formatValue(value);
  const bar = btn.querySelector(".slot-bar i");
  if (item.type === "range" || item.type === "number") {
    const pct = ((Number(value) - item.min) / (item.max - item.min)) * 100;
    bar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  } else {
    bar.style.width = value ? "100%" : "0%";
  }
}

function updateSlot(item) {
  const btn = document.querySelector(`.slot[data-key="${item.key}"]`);
  if (btn) updateSlotEl(btn, item);
}

function formatValue(v) {
  if (typeof v !== "number") return String(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

function openDetails(item) {
  selectedKey = item.key;
  document.querySelectorAll(".slot").forEach((s) => s.classList.toggle("selected", s.dataset.key === item.key));
  const value = config[item.key];
  let control;
  if (item.type === "range") {
    control = `<input type="range" id="det-input" min="${item.min}" max="${item.max}" step="${item.step}" value="${value}">
      <div class="value-line"><span>drag to change</span><span id="det-value">${formatValue(value)}</span></div>`;
  } else if (item.type === "number") {
    control = `<input type="number" id="det-input" min="${item.min}" max="${item.max}" step="${item.step}" value="${value}">
      <div class="value-line"><span>value</span><span id="det-value">${formatValue(value)}</span></div>`;
  } else if (item.type === "select") {
    control = `<select id="det-input">${item.options.map((o) => `<option ${o === value ? "selected" : ""}>${o}</option>`).join("")}</select>`;
  } else {
    control = `<label class="switch"><input type="checkbox" id="det-input" ${value ? "checked" : ""}><span></span></label>`;
  }
  const host = document.getElementById("details");
  host.innerHTML = `
    <div class="det-head"><span class="det-name">${item.key}</span><span class="det-group">${item.group}</span></div>
    <p class="det-help">${item.help}</p>
    <div class="det-control">${control}</div>
    <div class="det-foot"><button class="ghost" id="det-default">Reset to default</button></div>`;
  const input = document.getElementById("det-input");
  input.addEventListener("input", () => {
    let next;
    if (item.type === "toggle") next = input.checked;
    else if (item.type === "select") next = input.value;
    else next = Number(input.value);
    config[item.key] = next;
    updateSlot(item);
    const label = document.getElementById("det-value");
    if (label) label.textContent = formatValue(next);
  });
  document.getElementById("det-default").addEventListener("click", () => {
    config[item.key] = DEFAULT_CONFIG[item.key];
    updateSlot(item);
    openDetails(item);
  });
}

document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-preset]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    Object.assign(config, PRESETS[btn.dataset.preset] || {});
    renderGroups();
    const item = SCHEMA.flatMap((g) => g.items).find((i) => i.key === selectedKey);
    if (item) openDetails(item);
  });
});

/* ---------- guide ---------- */
function renderGuide() {
  const host = document.getElementById("guide");
  host.innerHTML = "";
  for (const group of SCHEMA) {
    const block = document.createElement("div");
    block.className = "guide-group";
    const title = document.createElement("h3");
    title.textContent = group.title;
    block.appendChild(title);
    for (const item of group.items) {
      const wrap = document.createElement("div");
      wrap.className = "guide-item";
      const dt = document.createElement("dt");
      dt.textContent = item.key;
      const dd = document.createElement("dd");
      dd.textContent = item.help;
      wrap.append(dt, dd);
      block.appendChild(wrap);
    }
    host.appendChild(block);
  }
}

/* ---------- upload / run / poll ---------- */
const dropzone = document.getElementById("dropzone");
const videoInput = document.getElementById("video-input");
dropzone.addEventListener("click", () => videoInput.click());
videoInput.addEventListener("change", () => setVideo(videoInput.files[0]));
for (const evt of ["dragover", "dragenter"]) {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
}
for (const evt of ["dragleave", "drop"]) {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); });
}
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setVideo(file);
});
function setVideo(file) {
  if (!file) return;
  videoFile = file;
  document.getElementById("drop-title").textContent = file.name;
  document.getElementById("drop-sub").textContent = `${(file.size / 1048576).toFixed(1)} MB · ready to run`;
}

const runButton = document.getElementById("run-button");
runButton.addEventListener("click", async () => {
  if (!videoFile) {
    runButton.textContent = "Choose a video first";
    setTimeout(() => {
      runButton.innerHTML = '<svg class="icon"><use href="#i-play"/></svg>Run reconstruction';
    }, 1600);
    return;
  }
  const fd = new FormData();
  fd.append("video", videoFile, videoFile.name);
  fd.append("config", JSON.stringify(config));
  runButton.disabled = true;
  lastMeshKey = null;
  hideError();
  try {
    const res = await fetch("/api/run", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
  } catch (err) {
    appendLog(`run failed: ${err.message}`);
    runButton.disabled = false;
  }
});

async function poll() {
  try {
    applyStatus(await (await fetch("/api/status", { cache: "no-store" })).json());
  } catch {
    /* server restarting */
  }
  setTimeout(poll, 900);
}

function applyStatus(data) {
  const pill = document.getElementById("status-pill");
  pill.textContent = data.status;
  pill.className = `status-pill ${data.status}`;
  stats = data.stats || {};
  renderStages(data.stage, data.status, data.progress);
  renderLog(data.log);
  renderCards(stats);
  renderError(data.error);
  if (data.report) renderReport(data.report);
  runButton.disabled = data.status === "running";
  runButton.innerHTML = data.status === "running"
    ? '<svg class="icon"><use href="#i-play"/></svg>Reconstructing…'
    : '<svg class="icon"><use href="#i-play"/></svg>Run reconstruction';
  if (data.status === "done" && stats.faces) {
    const key = `${stats.faces}-${stats.seconds}`;
    if (key !== lastMeshKey) {
      lastMeshKey = key;
      loadMesh();
    }
    const link = document.getElementById("download-mesh");
    link.href = "/api/mesh/mesh";
    link.style.display = "inline-flex";
  }
}

function renderStages(current, status, progress) {
  const host = document.getElementById("stages");
  if (host.childElementCount !== STAGES.length) {
    host.innerHTML = "";
    for (const name of STAGES) {
      const el = document.createElement("div");
      el.className = "stage";
      el.innerHTML = `<span class="stage-name">${name}</span><span class="stage-bar"><i></i></span>`;
      host.appendChild(el);
    }
  }
  [...host.children].forEach((el, i) => {
    const index = i + 1;
    const done = status === "done" || index < current;
    const active = status === "running" && index === current;
    const bar = el.querySelector(".stage-bar i");
    let pct = done ? 100 : 0;
    if (active && progress && progress.stage === index && progress.total > 0) {
      pct = Math.min(100, (progress.done / progress.total) * 100);
    } else if (active) {
      pct = 6;
    }
    bar.style.width = `${pct}%`;
    el.classList.toggle("done", done);
    el.classList.toggle("active", active);
  });
}

let logLines = [];
function renderLog(lines) {
  if (!lines) return;
  logLines = lines.slice(-200);
  const log = document.getElementById("log");
  log.textContent = logLines.join("\n");
  log.scrollTop = log.scrollHeight;
}
function appendLog(line) {
  logLines.push(line);
  renderLog(logLines);
}

function renderCards(s) {
  const host = document.getElementById("metric-cards");
  const entries = [
    ["Views", s.views ?? "—"],
    ["Points", s.points ?? "—"],
    ["RMSE", s.rmse != null ? `${s.rmse.toFixed(2)} px` : "—"],
    ["Depth error", s.depth_median != null ? `${s.depth_median.toFixed(1)} %` : "—"],
    ["Faces", s.faces ?? "—"],
    ["Patches", s.patches ?? "—"],
    ["Euler", s.euler ?? "—"],
    ["Closed", s.closed == null ? "—" : s.closed ? "yes" : "no"],
    ["Cost", s.cost ?? "—"],
    ["Time", s.seconds != null ? `${s.seconds.toFixed(1)} s` : "—"],
  ];
  host.innerHTML = "";
  for (const [k, v] of entries) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    host.appendChild(card);
  }
}

function hideError() {
  document.getElementById("error-banner").hidden = true;
}
function renderError(err) {
  const banner = document.getElementById("error-banner");
  if (!err) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  document.getElementById("error-code").textContent = err.code || "error";
  document.getElementById("error-message").textContent =
    `${err.message || "pipeline failed"}${err.stage ? ` · stage ${err.stage}` : ""}`;
  const hints = document.getElementById("error-hints");
  hints.innerHTML = "";
  for (const hint of err.hints || []) {
    const li = document.createElement("li");
    li.textContent = hint;
    hints.appendChild(li);
  }
}

function renderReport(report) {
  const q = report.mesh_quality || {};
  const p = report.provenance || {};
  const r = report.resources || {};
  const self = report.self_assessment || {};
  const entries = [
    ["Self-assessment", self.confidence != null ? `${(self.confidence * 100).toFixed(0)} % (${self.verdict})` : "—"],
    ["Closed mesh", q.closed ? "yes" : "no"],
    ["Boundary edges", q.boundary_edge_ratio != null ? `${(q.boundary_edge_ratio * 100).toFixed(1)} %` : "—"],
    ["Degenerate", q.degenerate_faces ?? "—"],
    ["Needle faces", q.needle_fraction != null ? `${(q.needle_fraction * 100).toFixed(1)} %` : "—"],
    ["Peak RSS", r.peak_rss_kb ? `${(r.peak_rss_kb / 1024).toFixed(0)} MB` : "—"],
    ["Elapsed", report.elapsed_s != null ? `${report.elapsed_s.toFixed(1)} s` : "—"],
    ["Video hash", p.video_sha256 ? `${p.video_sha256.slice(0, 12)}…` : "—"],
    ["Version", p.version ?? "—"],
  ];
  const host = document.getElementById("report-cards");
  host.innerHTML = "";
  for (const [k, v] of entries) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    host.appendChild(card);
  }
  document.getElementById("report-json").textContent = JSON.stringify(report, null, 2);
}

async function loadDiagnostics() {
  const rows = await (await fetch("/api/diagnostics", { cache: "no-store" })).json();
  const host = document.getElementById("diagnostics");
  host.innerHTML = "";
  for (const row of rows) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.innerHTML = `<code>${row.code}</code>`;
    details.appendChild(summary);
    const description = document.createElement("p");
    description.textContent = row.description;
    details.appendChild(description);
    const list = document.createElement("ul");
    for (const hint of row.hints || []) {
      const li = document.createElement("li");
      li.textContent = hint;
      list.appendChild(li);
    }
    details.appendChild(list);
    host.appendChild(details);
  }
}

/* ---------- charts ---------- */
function drawHistory() {
  drawChart(document.getElementById("chart-faces"), chartData.map((r) => r.faces), "#5f93c0");
  drawChart(document.getElementById("chart-depth"), chartData.map((r) => r.depth_median), "#d8b25c");
}

async function loadHistory() {
  try {
    chartData = await (await fetch("/api/history", { cache: "no-store" })).json();
  } catch {
    chartData = SEED_HISTORY;
  }
  drawHistory();
}

function drawChart(svg, values, color) {
  const w = 420, h = 140, pad = 24;
  svg.innerHTML = "";
  if (!values.length) return;
  const max = Math.max(...values, 1);
  const barW = (w - pad * 2) / values.length;
  values.forEach((v, i) => {
    const bh = (v / max) * (h - pad * 2);
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", pad + i * barW + barW * 0.15);
    rect.setAttribute("y", h - pad - bh);
    rect.setAttribute("width", barW * 0.7);
    rect.setAttribute("height", Math.max(bh, 1));
    rect.setAttribute("rx", 2);
    rect.setAttribute("fill", color);
    rect.setAttribute("opacity", 0.85);
    svg.appendChild(rect);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", pad + i * barW + barW / 2);
    label.setAttribute("y", h - pad + 12);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "8");
    label.setAttribute("fill", "#8ea3b2");
    label.textContent = (chartData[i]?.name || "").slice(0, 9);
    svg.appendChild(label);
    const value = document.createElementNS("http://www.w3.org/2000/svg", "text");
    value.setAttribute("x", pad + i * barW + barW / 2);
    value.setAttribute("y", h - pad - bh - 4);
    value.setAttribute("text-anchor", "middle");
    value.setAttribute("font-size", "8");
    value.setAttribute("fill", "#ece4d2");
    value.textContent = v >= 10000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(v < 100 ? 1 : 0);
    svg.appendChild(value);
  });
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axis.setAttribute("x1", pad);
  axis.setAttribute("x2", w - pad);
  axis.setAttribute("y1", h - pad);
  axis.setAttribute("y2", h - pad);
  axis.setAttribute("stroke", "rgba(122,160,190,.25)");
  svg.appendChild(axis);
}

function renderAltai() {
  const host = document.getElementById("altai");
  host.innerHTML = "";
  for (const [title, text, status] of ALTAI) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h3>${title} <span class="badge ${status}">${status === "good" ? "met" : "needs work"}</span></h3><p>${text}</p>`;
    host.appendChild(card);
  }
}

document.getElementById("export-config").addEventListener("click", () => {
  document.getElementById("config-json").textContent = JSON.stringify(config, null, 2);
});

/* ---------- slides ---------- */
function svg(body, viewBox = "0 0 360 280") {
  return `<svg viewBox="${viewBox}" fill="none" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
}

const SLIDES = [
  {
    kicker: "01",
    title: "meshprod",
    body: `<p><strong>One monocular video in, one measured mesh out.</strong></p>
      <p>Sparse structure-from-motion, dense multi-view stereo, signed-distance fusion, surface extraction and simplification — with a trust report attached to every run.</p>`,
    chips: ["22–66 registered views", "20k–87k faces", "~1 min/scene", "deterministic"],
    figure: svg(`
      <g stroke="${GOLD}" stroke-width="1.6" opacity="0.95">
        <path d="M180 60 270 108 180 156 90 108Z"/><path d="M90 108v70l90 48 90-48v-70"/>
        <path d="M180 156v70"/>
      </g>
      <g fill="${STEEL}" opacity="0.9">
        <circle cx="180" cy="60" r="3"/><circle cx="270" cy="108" r="3"/><circle cx="90" cy="108" r="3"/>
        <circle cx="180" cy="226" r="3"/>
      </g>`),
  },
  {
    kicker: "02",
    title: "The problem",
    body: `<p>Structure-from-Motion gives camera poses and a few thousand points. A surface needs <strong>every pixel</strong> to vote on where space ends.</p>
      <p>Three physical conditions decide whether that works: the scene is rigid, the surface has texture, the light is diffuse. All three are measurable failures.</p>`,
    chips: ["3/82 views when the object moved", "panning registered 16/228", "specular blade: 33 patches"],
    figure: svg(`
      <g stroke="${STEEL}" stroke-width="1.2" opacity="0.8">
        <circle cx="90" cy="70" r="18"/><circle cx="270" cy="70" r="18"/>
      </g>
      <path d="M108 82 200 150 252 86" stroke="${GOLD}" stroke-width="1.4" stroke-dasharray="5 5"/>
      <g fill="${GOLD}">
        <circle cx="200" cy="150" r="5"/>
      </g>`),
  },
  {
    kicker: "03",
    title: "The pipeline",
    body: `<p>Nine stages, each with its own log, timing and failure code. Sparse stages are cheap; dense depth dominates the runtime.</p>`,
    chips: ["decode → features → match → verify → tracks", "sparse → dense → fusion → mesh"],
    figure: svg(`
      <path d="M30 140h300" stroke="${GOLD}" stroke-width="1.2" class="flow-line"/>
      <g font-size="9" fill="${INK}" text-anchor="middle">
        ${["decode","features","match","verify","tracks","sparse","dense","fusion","mesh"]
          .map((n, i) => `<circle cx="${34 + i * 36.5}" cy="140" r="9" fill="#1b2c3d" stroke="${GOLD}" stroke-width="1.2"/>
            <text x="${34 + i * 36.5}" y="168" fill="${STEEL}">${i + 1}</text>
            <text x="${34 + i * 36.5}" y="122" fill="${INK}" font-size="7.5">${n}</text>`)
          .join("")}
      </g>`, "0 0 360 210"),
  },
  {
    kicker: "04",
    title: "Decode and features",
    body: `<p>Frames are decoded with GStreamer and converted to gray. SIFT finds scale-space extrema with 128-float descriptors; ORB uses FAST corners and 256-bit binary descriptors.</p>
      <p><strong>Choice:</strong> SIFT for phone orbits, ORB when speed matters more than large viewpoint changes.</p>`,
    chips: ["23.7k keypoints / 63 frames", "500 features per frame", "SIFT"],
    figure: svg(`
      <g stroke="${STEEL}" stroke-width="1">${Array.from({length: 8}, (_, i) =>
        `<line x1="${40 + i * 36}" y1="50" x2="${40 + i * 36}" y2="230"/>`).join("")}
        ${Array.from({length: 5}, (_, i) =>
        `<line x1="40" y1="${50 + i * 44}" x2="292" y2="${50 + i * 44}"/>`).join("")}
      </g>
      <g fill="none" stroke="${GOLD}" stroke-width="1.4">${[[94,94],[203,138],[167,182],[238,94]].map(
        ([x,y]) => `<circle cx="${x}" cy="${y}" r="7"/><path d="M${x-10} ${y}h20M${x} ${y-10}v20"/>`).join("")}
      </g>`),
  },
  {
    kicker: "05",
    title: "Matching",
    body: `<p>Three filters, each catching what the last missed: Lowe's ratio rejects ambiguous descriptors, mutual nearest neighbours enforce one-to-one matches, and epipolar RANSAC throws away pairs that violate <strong>x₂ᵀFx₁ = 0</strong>.</p>`,
    chips: ["21,667 raw matches", "17,170 after RANSAC", "3,780 tracks"],
    figure: svg(`
      <g stroke="${STEEL}" stroke-width="1.2"><path d="M60 90l40 26-40 26-40-26Z"/><path d="M300 90l40 26-40 26-40-26Z"/></g>
      <g stroke="${GOLD}" stroke-width="1" opacity="0.85">
        <line x1="60" y1="116" x2="300" y2="116" stroke-dasharray="4 6"/>
        <line x1="60" y1="116" x2="300" y2="150"/><line x1="60" y1="116" x2="300" y2="82"/>
      </g>
      <circle cx="60" cy="116" r="4" fill="${GOLD}"/><circle cx="300" cy="116" r="4" fill="${GOLD}"/>`),
  },
  {
    kicker: "06",
    title: "Sparse SfM",
    body: `<p>An essential-matrix bootstrap fixes the world frame. Each view is registered by PnP inside RANSAC with SPRT early rejection; new points are triangulated and bundle adjustment minimises reprojection error.</p>
      <p><strong>Built-in honesty:</strong> the reported RMSE is consistency, not truth.</p>`,
    chips: ["RMSE 2.09 px", "22/63 views", "1,175 points", "up to scale"],
    figure: svg(`
      <g fill="${GOLD}">${Array.from({length: 26}, (_, i) =>
        `<circle cx="${110 + (i * 53) % 150}" cy="${90 + (i * 37) % 120}" r="2.2"/>`).join("")}</g>
      <g fill="none" stroke="${STEEL}" stroke-width="1.4">
        <path d="M60 60l30 18-30 18-30-18Z"/><path d="M300 60l30 18-30 18-30-18Z"/>
        <path d="M60 200l30 18-30 18-30-18Z"/><path d="M300 200l30 18-30 18-30-18Z"/>
        <path d="M60 78 300 78M60 78 300 218M60 218 300 78" stroke-dasharray="3 6" opacity="0.5"/>
      </g>`),
  },
  {
    kicker: "07",
    title: "Dense depth",
    body: `<p>For every registered view, planes of constant inverse depth are swept along the rays. A plane induces a homography; the best photo-consistency wins the pixel.</p>
      <p>Census compares orderings instead of intensities, and SGM adds a smoothness prior that fills low-texture regions.</p>`,
    chips: ["median 4.0 % error", "p90 29.4 %", "sub-pixel refined"],
    figure: svg(`
      <g stroke="${STEEL}" stroke-width="1" opacity="0.75">
        ${Array.from({length: 5}, (_, i) => `<path d="M40 ${70 + i * 34}h280"/>`).join("")}
      </g>
      <g stroke="${GOLD}" stroke-width="1.4">
        <path d="M180 220 90 70M180 220 180 70M180 220 270 70"/>
      </g>
      <circle cx="180" cy="220" r="4" fill="${GOLD}"/>
      <path d="M96 104h168" stroke="${GOLD}" stroke-width="2" opacity="0.8"/>`),
  },
  {
    kicker: "08",
    title: "Fusion",
    body: `<p>Each depth map votes: empty in front of the surface, solid behind it. The weighted average is a signed distance field whose zero crossing is the surface.</p>
      <p>Confidence and view angle weight the votes, so a highlight cannot poison the field.</p>`,
    chips: ["truncation follows depth noise", "451k observed voxels", "TSDF"],
    figure: svg(`
      <g stroke="${STEEL}" stroke-width="1">
        ${Array.from({length: 4}, (_, r) => Array.from({length: 4}, (_, c) =>
          `<path d="M${70 + c * 44 + r * 18} ${70 + r * 34}l44 0 0 34-44 0Z"/>`).join("")).join("")}
      </g>
      <g fill="${GOLD}" font-size="10" font-family="monospace">
        <text x="92" y="92">+</text><text x="136" y="92">+</text><text x="92" y="126">−</text><text x="136" y="126">−</text>
      </g>`),
  },
  {
    kicker: "09",
    title: "Surface and simplification",
    body: `<p>Each cube is split into six tetrahedra; the zero crossing becomes triangles with no lookup tables. Small patches are removed, Taubin and bilateral passes de-spike the surface, and quadric error metrics spend the face budget where the shape bends.</p>`,
    chips: ["101k raw → 87k faces", "33 patches", "Euler −97"],
    figure: svg(`
      <g stroke="${STEEL}" stroke-width="1.2" fill="none">
        <path d="M90 90h140v120H90Z"/><path d="M90 90l50 40 90-40M140 130v80"/>
      </g>
      <path d="M90 90 140 130 230 90 90 90Z" fill="rgba(226,200,138,.12)" stroke="${GOLD}" stroke-width="1.6"/>
      <circle cx="90" cy="90" r="3" fill="${GOLD}"/><circle cx="140" cy="130" r="3" fill="${GOLD}"/><circle cx="230" cy="90" r="3" fill="${GOLD}"/>`),
  },
  {
    kicker: "10",
    title: "Trust, measured",
    body: `<p>Every run writes a report: mesh quality, provenance hashes, resource use and a self-assessment. Failures carry a stable code and hints.</p>
      <p>Determinism is verified per platform; robustness is a curve, not a claim.</p>`,
    chips: ["sha256 provenance", "quality pack", "13 failure codes", "confidence 0–1"],
    figure: svg(`
      <g stroke="${STEEL}" stroke-width="2" fill="none">
        <circle cx="180" cy="140" r="60" opacity="0.35"/>
        <circle cx="180" cy="140" r="40" opacity="0.5"/>
      </g>
      <path d="M180 140 180 88" stroke="${GOLD}" stroke-width="2.4"/>
      <path d="M180 140 218 168" stroke="${GOLD}" stroke-width="2.4" opacity="0.8"/>
      <circle cx="180" cy="140" r="5" fill="${GOLD}"/>
      <text x="180" y="228" text-anchor="middle" fill="${STEEL}" font-size="10" font-family="monospace">confidence 0.61 · medium</text>`),
  },
  {
    kicker: "11",
    title: "What fails, and why",
    body: `<p>Specular metal breaks photo-consistency. Plain white surfaces give NCC nothing to match. Panning without parallax cannot triangulate. Partial orbits leave open shells.</p>
      <p>Each failure is detectable — the reports and the capture checklist tell you before the mesh does.</p>`,
    chips: ["specular", "textureless", "no parallax", "partial coverage"],
    figure: svg(`
      <g stroke="${STEEL}" stroke-width="1.2" fill="none">
        <rect x="50" y="80" width="70" height="90" rx="4"/>
        <rect x="150" y="80" width="70" height="90" rx="4" stroke-dasharray="4 6"/>
        <rect x="250" y="80" width="70" height="90" rx="4" stroke-dasharray="4 6"/>
      </g>
      <g fill="${GOLD}" font-size="9" font-family="monospace" text-anchor="middle">
        <text x="85" y="192">glossy</text><text x="185" y="192">textureless</text><text x="285" y="192">no parallax</text>
      </g>`),
  },
  {
    kicker: "12",
    title: "Credits",
    body: `<p><strong>Adarsh Gupta</strong> — semester project.</p>
      <p>Built on kornia-rs, three.js, Blender, Open3D and GStreamer. AI-assisted with OpenCode (DeepSeek V4.1 Flash); every number here comes from a measured run.</p>
      <p><a href="https://github.com/AdarshGuptaa/meshprod" target="_blank" rel="noreferrer" style="color:${GOLD}">github.com/AdarshGuptaa/meshprod</a> · MIT</p>`,
    chips: ["open source", "MIT", "reproducible"],
    figure: svg(`
      <g stroke="${GOLD}" stroke-width="1.4" fill="none">
        <circle cx="180" cy="130" r="52"/>
        <path d="M180 78v104M128 130h104" opacity="0.4"/>
      </g>
      <text x="180" y="136" text-anchor="middle" fill="${INK}" font-size="22" font-family="monospace">meshprod</text>`),
  },
];

let slideIndex = 0;
function renderSlides() {
  const stage = document.getElementById("slide-stage");
  stage.innerHTML = "";
  SLIDES.forEach((slide) => {
    const el = document.createElement("article");
    el.className = "slide";
    el.innerHTML = `
      <div class="s-copy">
        <div class="s-index">${slide.kicker}</div>
        <h2>${slide.title}</h2>
        <div class="s-body">${slide.body}</div>
        <div class="s-chips">${(slide.chips || []).map((c) => `<span class="chip">${c}</span>`).join("")}</div>
      </div>
      <div class="s-figure">${slide.figure || ""}</div>`;
    stage.appendChild(el);
  });
  const dots = document.getElementById("slide-dots");
  dots.innerHTML = "";
  SLIDES.forEach((_, i) => {
    const dot = document.createElement("button");
    dot.className = "dot";
    dot.setAttribute("aria-label", `Slide ${i + 1}`);
    dot.addEventListener("click", () => showSlide(i));
    dots.appendChild(dot);
  });
  showSlide(0);
}

function showSlide(index) {
  slideIndex = (index + SLIDES.length) % SLIDES.length;
  [...document.querySelectorAll(".slide")].forEach((el, i) => {
    el.classList.toggle("active", i === slideIndex);
    el.classList.toggle("past", i < slideIndex);
  });
  [...document.querySelectorAll(".dot")].forEach((d, i) => d.classList.toggle("active", i === slideIndex));
  document.getElementById("slide-counter").textContent = `${slideIndex + 1} / ${SLIDES.length}`;
  document.getElementById("slide-progress").style.width = `${((slideIndex + 1) / SLIDES.length) * 100}%`;
}

document.getElementById("slide-prev").addEventListener("click", () => showSlide(slideIndex - 1));
document.getElementById("slide-next").addEventListener("click", () => showSlide(slideIndex + 1));
document.getElementById("slide-present").addEventListener("click", async () => {
  activateTab("slides");
  document.body.classList.toggle("presenting");
  if (document.body.classList.contains("presenting")) {
    await document.documentElement.requestFullscreen?.().catch(() => {});
  } else {
    await document.exitFullscreen?.().catch(() => {});
  }
  resizeSlides();
});
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement) document.body.classList.remove("presenting");
  resizeViewer();
});
document.addEventListener("keydown", (e) => {
  const slidesActive = document.getElementById("tab-slides").classList.contains("active");
  if (slidesActive) {
    if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); showSlide(slideIndex + 1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); showSlide(slideIndex - 1); }
    if (e.key === "Escape" && document.body.classList.contains("presenting")) {
      document.body.classList.remove("presenting");
    }
  }
});
function resizeSlides() {
  const active = document.querySelector(".slide.active");
  if (active) {
    active.classList.remove("active");
    requestAnimationFrame(() => active.classList.add("active"));
  }
}

/* ---------- three.js viewer ---------- */
const viewerHost = document.getElementById("viewer");
const viewfinder = document.getElementById("viewfinder");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1115);
const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1e7);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewerHost.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.zoomSpeed = 1.35;
controls.panSpeed = 1.15;
controls.rotateSpeed = 0.9;
controls.screenSpacePanning = true;

scene.add(new THREE.HemisphereLight(0xfdfaf2, 0x1a2229, 1.05));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(3, 5, 4);
scene.add(key);
const fill = new THREE.DirectionalLight(0xbcd2e2, 0.65);
fill.position.set(-4, 2, -3);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xd8b25c, 0.55);
rim.position.set(0, -4, -5);
scene.add(rim);

let current = null;
let originalColors = null;
let camTween = null;
let camRadius = 1;
let camCenter = new THREE.Vector3();
const loader = new PLYLoader();

function loadMesh() {
  document.getElementById("viewer-empty").classList.remove("hidden");
  loader.load(
    `/api/mesh/mesh?t=${Date.now()}`,
    (geometry) => {
      geometry.computeVertexNormals();
      originalColors = geometry.attributes.color ? Float32Array.from(geometry.attributes.color.array) : null;
      if (current) {
        scene.remove(current);
        current.geometry.dispose();
        current.material.dispose();
      }
      current = new THREE.Mesh(geometry, makeMaterial("colour", geometry));
      scene.add(current);
      fitCamera(geometry);
      document.getElementById("viewer-empty").classList.add("hidden");
      controls.autoRotate = true;
      controls.autoRotateSpeed = 1.1;
      setTimeout(() => (controls.autoRotate = false), 3500);
    },
    undefined,
    () => appendLog("mesh load failed"),
  );
}

function fitCamera(geometry) {
  geometry.computeBoundingSphere();
  const sphere = geometry.boundingSphere;
  if (!sphere) return;
  const r = Math.max(sphere.radius, 1e-6);
  camRadius = r;
  camCenter = sphere.center.clone();
  const dir = new THREE.Vector3(1.15, 0.72, 1.35).normalize();
  camera.position.copy(camCenter).addScaledVector(dir, r * 1.35);
  camera.near = r / 500;
  camera.far = r * 120;
  camera.updateProjectionMatrix();
  controls.target.copy(camCenter);
  controls.minDistance = r * 0.25;
  controls.maxDistance = r * 8;
  camTween = null;
  controls.update();
}

function tweenCamera(position, target, ms = 420) {
  camTween = {
    p0: camera.position.clone(),
    p1: position.clone(),
    t0: controls.target.clone(),
    t1: target.clone(),
    start: performance.now(),
    ms,
  };
}

function orbitBy(dTheta, dPhi) {
  const offset = camera.position.clone().sub(controls.target);
  const spherical = new THREE.Spherical().setFromVector3(offset);
  spherical.theta += dTheta;
  spherical.phi = THREE.MathUtils.clamp(spherical.phi + dPhi, 0.08, Math.PI - 0.08);
  offset.setFromSpherical(spherical);
  camera.position.copy(controls.target).add(offset);
  camTween = null;
  controls.update();
}

function zoomBy(factor) {
  const offset = camera.position.clone().sub(controls.target);
  const distance = THREE.MathUtils.clamp(
    offset.length() * factor,
    controls.minDistance,
    controls.maxDistance,
  );
  offset.setLength(distance);
  camera.position.copy(controls.target).add(offset);
  camTween = null;
  controls.update();
}

function panBy(dx, dy) {
  const step = camRadius * 0.08;
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
  const delta = right.multiplyScalar(dx * step).add(up.multiplyScalar(dy * step));
  camera.position.add(delta);
  controls.target.add(delta);
  camTween = null;
  controls.update();
}

function setView(kind) {
  const dirs = {
    front: new THREE.Vector3(0, 0.08, 1),
    top: new THREE.Vector3(0, 1, 0.02),
    side: new THREE.Vector3(1, 0.08, 0),
  };
  const dir = dirs[kind];
  if (!dir) return;
  const position = camCenter.clone().addScaledVector(dir.normalize(), camRadius * 1.35);
  tweenCamera(position, camCenter);
}

const camActions = {
  "orbit-up": () => orbitBy(0, -0.16),
  "orbit-down": () => orbitBy(0, 0.16),
  "orbit-left": () => orbitBy(-0.2, 0),
  "orbit-right": () => orbitBy(0.2, 0),
  "zoom-in": () => zoomBy(0.8),
  "zoom-out": () => zoomBy(1.25),
  "pan-up": () => panBy(0, 1),
  "pan-down": () => panBy(0, -1),
  "pan-left": () => panBy(-1, 0),
  "pan-right": () => panBy(1, 0),
  reset: () => { if (current) fitCamera(current.geometry); },
  fit: () => { if (current) fitCamera(current.geometry); },
  front: () => setView("front"),
  top: () => setView("top"),
  side: () => setView("side"),
  auto: () => { controls.autoRotate = !controls.autoRotate; },
};

document.querySelectorAll("[data-cam]").forEach((btn) => {
  btn.addEventListener("click", () => camActions[btn.dataset.cam]?.());
});

document.getElementById("view-fullscreen").addEventListener("click", async () => {
  if (!document.fullscreenElement) {
    await viewfinder.requestFullscreen?.().catch(() => {});
  } else {
    await document.exitFullscreen?.().catch(() => {});
  }
  resizeViewer();
});

document.addEventListener("keydown", (e) => {
  if (!document.getElementById("tab-studio").classList.contains("active")) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const actions = {
    ArrowUp: "orbit-up", ArrowDown: "orbit-down", ArrowLeft: "orbit-left", ArrowRight: "orbit-right",
    "+": "zoom-in", "=": "zoom-in", "-": "zoom-out", "0": "reset", f: "fit", F: "fit",
  };
  const action = actions[e.key];
  if (action) {
    e.preventDefault();
    camActions[action]?.();
  }
});

function heightColors(geometry) {
  const pos = geometry.attributes.position;
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    min = Math.min(min, y);
    max = Math.max(max, y);
  }
  const span = Math.max(max - min, 1e-9);
  const colors = new Float32Array(pos.count * 3);
  const low = new THREE.Color(0x24405e);
  const high = new THREE.Color(0xf0d79a);
  for (let i = 0; i < pos.count; i++) {
    const t = (pos.getY(i) - min) / span;
    const color = low.clone().lerp(high, t);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  return colors;
}

function makeMaterial(style, geometry) {
  switch (style) {
    case "clay":
      return new THREE.MeshStandardMaterial({ color: 0xe9e1d0, roughness: 0.85, metalness: 0.02 });
    case "normals":
      return new THREE.MeshNormalMaterial();
    case "wireframe":
      return new THREE.MeshBasicMaterial({ wireframe: true, color: 0xd8b25c });
    case "xray":
      return new THREE.MeshStandardMaterial({
        color: 0x9db4bf, transparent: true, opacity: 0.22,
        side: THREE.DoubleSide, depthWrite: false, roughness: 0.3,
      });
    case "height": {
      geometry.setAttribute("color", new THREE.BufferAttribute(heightColors(geometry), 3));
      return new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7 });
    }
    default:
      return new THREE.MeshStandardMaterial({
        vertexColors: Boolean(geometry.attributes.color),
        color: 0xffffff,
        roughness: 0.72,
        metalness: 0.04,
      });
  }
}

document.querySelectorAll("[data-style]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-style]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    if (!current) return;
    const style = btn.dataset.style;
    if (style !== "height" && originalColors && current.geometry.attributes.color) {
      current.geometry.setAttribute("color", new THREE.BufferAttribute(Float32Array.from(originalColors), 3));
    }
    current.material.dispose();
    current.material = makeMaterial(style, current.geometry);
  });
});

controls.addEventListener("start", () => { controls.autoRotate = false; });

function resizeViewer() {
  const { clientWidth: w, clientHeight: h } = viewerHost;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resizeViewer).observe(viewerHost);

renderer.setAnimationLoop(() => {
  if (!viewerActive) return;
  if (camTween) {
    const t = Math.min(1, (performance.now() - camTween.start) / camTween.ms);
    const e = t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
    camera.position.lerpVectors(camTween.p0, camTween.p1, e);
    controls.target.lerpVectors(camTween.t0, camTween.t1, e);
    if (t >= 1) camTween = null;
  }
  controls.update();
  renderer.render(scene, camera);
});

/* ---------- boot ---------- */
function firstPaint() {
  renderGroups();
  renderGuide();
  renderAltai();
  renderSlides();
  chartData = SEED_HISTORY;
  drawHistory();
  resizeViewer();
  const initial = location.hash.replace("#", "");
  if (initial) activateTab(initial);
  document.getElementById("log").textContent = "waiting for a run…";
  const first = SCHEMA[0].items[0];
  openDetails(first);
}

async function syncWithServer() {
  try {
    config = await (await fetch("/api/defaults", { cache: "no-store" })).json();
    renderGroups();
    const item = SCHEMA.flatMap((g) => g.items).find((i) => i.key === selectedKey);
    if (item) openDetails(item);
  } catch {
    /* offline: local defaults already rendered */
  }
  await loadHistory();
  await loadDiagnostics();
  poll();
}

firstPaint();
syncWithServer();
