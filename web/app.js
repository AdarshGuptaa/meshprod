import * as THREE from "three";
import { PLYLoader } from "./vendor/PLYLoader.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const STAGES = [
  "decode", "features", "match", "verify", "tracks", "sparse", "dense", "fusion", "mesh",
];

const SCHEMA = [
  {
    title: "Matching",
    open: false,
    items: [
      { key: "detector", label: "Detector", type: "select", options: ["sift", "orb"] },
      { key: "frame_step", label: "Frame step", type: "range", min: 1, max: 30, step: 1 },
      { key: "n_features", label: "Max features", type: "range", min: 100, max: 2000, step: 100 },
      { key: "match_window", label: "Match window", type: "range", min: 1, max: 12, step: 1 },
      { key: "ratio", label: "Lowe ratio", type: "range", min: 0.6, max: 0.95, step: 0.01 },
      { key: "geo_verify", label: "Epipolar verification", type: "toggle" },
      { key: "sprt", label: "SPRT registration", type: "toggle" },
      { key: "refine_intrinsics", label: "Refine intrinsics", type: "toggle" },
      { key: "auto_config", label: "Auto-config from texture", type: "toggle" },
    ],
  },
  {
    title: "Dense depth",
    open: true,
    items: [
      { key: "cost", label: "Cost", type: "select", options: ["ncc", "census", "sgm"] },
      { key: "dense_scale", label: "Dense scale", type: "range", min: 0.25, max: 1, step: 0.05 },
      { key: "dense_planes", label: "Planes", type: "range", min: 32, max: 160, step: 8 },
      { key: "dense_window", label: "Window", type: "range", min: 3, max: 7, step: 2 },
      { key: "dense_neighbors", label: "Neighbours", type: "range", min: 1, max: 5, step: 1 },
      { key: "ncc_min", label: "Min NCC", type: "range", min: 0.2, max: 0.7, step: 0.05 },
      { key: "lr_required", label: "Agreements", type: "range", min: 1, max: 3, step: 1 },
      { key: "sgm_paths", label: "SGM paths", type: "range", min: 1, max: 8, step: 1 },
    ],
  },
  {
    title: "Fusion",
    open: false,
    items: [
      { key: "voxel_resolution", label: "Voxel resolution", type: "range", min: 64, max: 384, step: 16 },
      { key: "trunc_voxels", label: "Truncation", type: "range", min: 1, max: 8, step: 0.5 },
      { key: "volume_smoothing", label: "Field smoothing", type: "range", min: 0, max: 4, step: 1 },
      { key: "hole_fill_iterations", label: "Hole fill", type: "range", min: 0, max: 8, step: 1 },
      { key: "remove_plane", label: "Remove dominant plane", type: "toggle" },
    ],
  },
  {
    title: "Mesh",
    open: false,
    items: [
      { key: "smooth_iterations", label: "Taubin smoothing", type: "range", min: 0, max: 16, step: 1 },
      { key: "bilateral_iterations", label: "Bilateral smoothing", type: "range", min: 0, max: 4, step: 1 },
      { key: "needle_aspect", label: "Needle filter", type: "range", min: 0, max: 30, step: 1 },
      { key: "target_faces", label: "Target faces", type: "range", min: 5000, max: 150000, step: 5000 },
      { key: "min_component_faces", label: "Min patch", type: "range", min: 0, max: 2000, step: 50 },
    ],
  },
];

const PRESETS = {
  fast: { frame_step: 20, dense_scale: 0.35, dense_planes: 64, voxel_resolution: 192, target_faces: 20000 },
  balanced: {},
  quality: { frame_step: 5, dense_scale: 0.75, dense_planes: 128, voxel_resolution: 320, target_faces: 80000 },
  auto: { auto_config: true },
};

const ALTAI = [
  ["Human agency & oversight", "Every parameter is exposed; a human decides when a mesh is usable. The pipeline never auto-accepts a result.", "good"],
  ["Technical robustness", "Deterministic reruns, depth-vs-sparse checks and honest closedness reporting. Specular and textureless captures degrade visibly, not silently.", "good"],
  ["Privacy & data governance", "Everything runs on 127.0.0.1. No upload to any server, no telemetry, no network access after the build.", "good"],
  ["Transparency", "All logs and derived parameters are visible and exportable; assumptions (pinhole, up-to-scale, Lambertian) are documented.", "good"],
  ["Fairness & diversity", "Only three captures evaluated so far — specular, textureless and textured. More object categories are needed.", "warn"],
  ["Societal & environmental", "CPU-only path, ~1 minute per scan; modest energy footprint. Scanning objects can raise IP/consent questions.", "good"],
  ["Accountability", "Reproducible commands, measured claims, pinned dependency, MIT license and AI disclosure in the credits.", "good"],
];

let config = null;
let stats = null;
let lastMeshKey = null;
let chartData = [];

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
  });
});

/* ---------- config form ---------- */
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
    for (const item of group.items) {
      details.appendChild(item.type === "toggle" ? toggleRow(item) : rangeRow(item));
    }
    host.appendChild(details);
  }
}

function clampToStep(value, item) {
  if (item.step) return Math.round(value / item.step) * item.step;
  return value;
}

function rangeRow(item) {
  const row = document.createElement("div");
  row.className = "row";
  const label = document.createElement("label");
  label.textContent = item.label;
  const value = document.createElement("span");
  value.className = "value";
  const input = document.createElement("input");
  input.type = "range";
  input.min = item.min;
  input.max = item.max;
  input.step = item.step;
  input.value = config[item.key];
  value.textContent = formatValue(config[item.key]);
  input.addEventListener("input", () => {
    config[item.key] = Number(input.value);
    value.textContent = formatValue(config[item.key]);
  });
  row.append(label, input, value);
  return row;
}

function toggleRow(item) {
  const row = document.createElement("div");
  row.className = "row";
  const label = document.createElement("label");
  label.textContent = item.label;
  const sw = document.createElement("label");
  sw.className = "switch";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(config[item.key]);
  input.addEventListener("change", () => (config[item.key] = input.checked));
  const knob = document.createElement("span");
  sw.append(input, knob);
  row.append(label, sw);
  return row;
}

function formatValue(v) {
  if (typeof v !== "number") return String(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

/* ---------- presets ---------- */
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-preset]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const preset = PRESETS[btn.dataset.preset] || {};
    Object.assign(config, preset);
    renderGroups();
  });
});

/* ---------- intrinsics ---------- */
function bindIntrinsics() {
  for (const key of ["fx", "fy", "cx", "cy"]) {
    const input = document.getElementById(key);
    input.value = config[key];
    input.addEventListener("change", () => (config[key] = Number(input.value)));
  }
}

/* ---------- upload ---------- */
let videoFile = null;
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

/* ---------- run ---------- */
const runButton = document.getElementById("run-button");
runButton.addEventListener("click", async () => {
  if (!videoFile) {
    runButton.textContent = "Choose a video first";
    setTimeout(() => (runButton.textContent = "Run reconstruction"), 1600);
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

/* ---------- polling ---------- */
async function poll() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    const data = await res.json();
    applyStatus(data);
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

  renderStages(data.stage, data.status);
  renderLog(data.log);
  renderCards(stats);
  renderError(data.error);
  if (data.report) renderReport(data.report);
  runButton.disabled = data.status === "running";
  runButton.textContent = data.status === "running" ? "Reconstructing…" : "Run reconstruction";

  if (data.status === "done" && stats.faces) {
    const key = `${stats.faces}-${stats.seconds}`;
    if (key !== lastMeshKey) {
      lastMeshKey = key;
      loadMesh();
    }
    const link = document.getElementById("download-mesh");
    link.href = "/api/mesh/mesh";
    link.style.display = "inline-block";
  }
  if (data.status === "running") {
    document.getElementById("download-mesh").style.display = "none";
  }
}

function renderStages(current, status) {
  const host = document.getElementById("stages");
  if (host.childElementCount !== STAGES.length) {
    host.innerHTML = "";
    for (const name of STAGES) {
      const el = document.createElement("div");
      el.className = "stage";
      el.textContent = name;
      host.appendChild(el);
    }
  }
  [...host.children].forEach((el, i) => {
    const index = i + 1;
    el.classList.toggle("done", status === "done" || index < current);
    el.classList.toggle("active", status === "running" && index === current);
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

/* ---------- named errors ---------- */
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

/* ---------- run report (trust metrics) ---------- */
function renderReport(report) {
  const q = report.mesh_quality || {};
  const p = report.provenance || {};
  const r = report.resources || {};
  const self = report.self_assessment || {};
  const entries = [
    ["Self-assessment", self.confidence != null ? `${(self.confidence * 100).toFixed(0)} % (${self.verdict})` : "—"],
    ["Closed mesh", q.closed ? "yes" : "no"],
    ["Boundary edges", q.boundary_edge_ratio != null ? `${(q.boundary_edge_ratio * 100).toFixed(1)} %` : "—"],
    ["Degenerate faces", q.degenerate_faces ?? "—"],
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

/* ---------- failure catalogue ---------- */
async function loadDiagnostics() {
  const res = await fetch("/api/diagnostics", { cache: "no-store" });
  const rows = await res.json();
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
async function loadHistory() {
  const res = await fetch("/api/history", { cache: "no-store" });
  chartData = await res.json();
  drawChart(document.getElementById("chart-faces"), chartData.map((r) => r.faces), "#3c4a55");
  drawChart(document.getElementById("chart-depth"), chartData.map((r) => r.depth_median), "#7d8a5c");
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
    rect.setAttribute("rx", 4);
    rect.setAttribute("fill", color);
    rect.setAttribute("opacity", 0.85);
    svg.appendChild(rect);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", pad + i * barW + barW / 2);
    label.setAttribute("y", h - pad + 12);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "8");
    label.setAttribute("fill", "#5b646b");
    label.textContent = (chartData[i]?.name || "").slice(0, 9);
    svg.appendChild(label);
    const value = document.createElementNS("http://www.w3.org/2000/svg", "text");
    value.setAttribute("x", pad + i * barW + barW / 2);
    value.setAttribute("y", h - pad - bh - 4);
    value.setAttribute("text-anchor", "middle");
    value.setAttribute("font-size", "8");
    value.setAttribute("fill", "#23282c");
    value.textContent = v >= 10000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(v < 100 ? 1 : 0);
    svg.appendChild(value);
  });
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axis.setAttribute("x1", pad);
  axis.setAttribute("x2", w - pad);
  axis.setAttribute("y1", h - pad);
  axis.setAttribute("y2", h - pad);
  axis.setAttribute("stroke", "#e5ddca");
  svg.appendChild(axis);
}

/* ---------- ALTAI ---------- */
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

/* ---------- config export ---------- */
document.getElementById("export-config").addEventListener("click", () => {
  document.getElementById("config-json").textContent = JSON.stringify(config, null, 2);
});

/* ---------- three.js viewer ---------- */
const viewerHost = document.getElementById("viewer");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x20272c);
const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1e7);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewerHost.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xfdfaf2, 0x2a3339, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(3, 5, 4);
scene.add(key);
const fill = new THREE.DirectionalLight(0xcfd9de, 0.7);
fill.position.set(-4, 2, -3);
scene.add(fill);

let current = null;
let originalColors = null;
const loader = new PLYLoader();

function loadMesh() {
  document.getElementById("viewer-empty").classList.remove("hidden");
  loader.load(
    `/api/mesh/mesh?t=${Date.now()}`,
    (geometry) => {
      geometry.computeVertexNormals();
      const hasColor = Boolean(geometry.attributes.color);
      originalColors = hasColor ? Float32Array.from(geometry.attributes.color.array) : null;
      if (current) {
        scene.remove(current);
        current.geometry.dispose();
      }
      current = new THREE.Mesh(geometry, makeMaterial("colour", geometry));
      scene.add(current);
      fitCamera(geometry);
      document.getElementById("viewer-empty").classList.add("hidden");
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
  const c = sphere.center;
  camera.position.set(c.x + r * 1.4, c.y + r * 0.9, c.z + r * 1.6);
  camera.near = r / 200;
  camera.far = r * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(c);
  controls.update();
}

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
  const low = new THREE.Color(0x3c4a55);
  const high = new THREE.Color(0xe8dfc8);
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
      return new THREE.MeshNormalMaterial({ flatShading: false });
    case "wireframe":
      return new THREE.MeshBasicMaterial({ wireframe: true, color: 0xa9bcc6 });
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
      current.geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(Float32Array.from(originalColors), 3),
      );
    }
    current.material.dispose();
    current.material = makeMaterial(style, current.geometry);
  });
});

function resize() {
  const { clientWidth: w, clientHeight: h } = viewerHost;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewerHost);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

/* ---------- boot ---------- */
async function boot() {
  const res = await fetch("/api/defaults", { cache: "no-store" });
  config = await res.json();
  renderGroups();
  bindIntrinsics();
  renderAltai();
  resize();
  await loadHistory();
  await loadDiagnostics();
  poll();
}

boot();
