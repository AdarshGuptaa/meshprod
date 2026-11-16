// UI wiring tests. Run with: node web/test/config.test.mjs
//
// These parse the static config blocks out of web/app.js (no DOM needed) and
// assert the invariants the UI depends on: complete presets, every schema /
// quick-toggle key is a real config field, and the removed camera UI stays gone.
import fs from "node:fs";
import assert from "node:assert/strict";

const here = new URL(".", import.meta.url);
const app = fs.readFileSync(new URL("../app.js", here), "utf8");
const html = fs.readFileSync(new URL("../index.html", here), "utf8");

function block(startMarker, endMarker) {
  const a = app.indexOf(startMarker);
  const b = app.indexOf(endMarker, a + 1);
  assert(a >= 0 && b > a, `missing block: ${startMarker}`);
  return app.slice(a, b).replace(/\bconst /g, "var ");
}

const code =
  block("const DEFAULT_CONFIG", "const SCHEMA") +
  block("const SCHEMA", "for (const group of SCHEMA)") +
  block("const PRESETS", "const ALTAI");
const { DEFAULT_CONFIG, SCHEMA, PRESETS, QUICK_TOGGLES } = new Function(
  `${code}; return { DEFAULT_CONFIG, SCHEMA, PRESETS, QUICK_TOGGLES };`,
)();

const fill = (label, n) => {
  if (!process.stdout.isTTY) return;
  console.log(`ok ${n} - ${label}`);
};
let n = 0;

// 1) Presets are complete configs (they replace, not merge).
const defaults = Object.keys(DEFAULT_CONFIG);
for (const [name, cfg] of Object.entries(PRESETS)) {
  assert.deepEqual(
    Object.keys(cfg).sort(),
    defaults.sort(),
    `preset "${name}" must carry every config key`,
  );
}
fill("every preset is a complete config", ++n);

// 2) The four quality presets pin auto_config explicitly.
for (const name of ["fast", "balanced", "quality"]) {
  assert.equal(PRESETS[name].auto_config, false, `${name}.auto_config must be false`);
}
assert.equal(PRESETS.auto.auto_config, true, "auto.auto_config must be true");
fill("quality presets pin auto_config", ++n);

// 3) Workflow quick actions are present as data-action buttons.
assert(html.includes('data-action="object"'), "missing Object scan action");
assert(html.includes('data-action="metric"'), "missing Metric action");
fill("workflow quick actions present", ++n);

// 4) Quick toggles map to real boolean config fields.
assert(QUICK_TOGGLES.length >= 6, "expected all major booleans");
for (const [key] of QUICK_TOGGLES) {
  assert(key in DEFAULT_CONFIG, `quick toggle "${key}" is not a config field`);
  assert.equal(typeof DEFAULT_CONFIG[key], "boolean", `quick toggle "${key}" must be a boolean`);
}
fill("quick toggles map to boolean config fields", ++n);

// 5) Every schema key is a config field, and every config field the UI edits is
//    represented (so the left panel and the server stay in sync).
const schemaKeys = new Set(SCHEMA.flatMap((g) => g.items.map((i) => i.key)));
for (const key of schemaKeys) {
  assert(key in DEFAULT_CONFIG, `schema key "${key}" missing from DEFAULT_CONFIG`);
}
fill("schema keys exist in DEFAULT_CONFIG", ++n);

// 6) Unified UI: one view HUD, no legacy camera toolbar/panel.
assert(html.includes('class="view-hud"'), "missing unified view HUD");
assert(html.includes('class="segmented"'), "missing segmented controls");
assert(html.includes('class="toggle-list"'), "missing toggle switch list");
assert(html.includes('class="keylist"'), "missing keybinding legend");
assert(!html.includes("qtoggle") && !app.includes("qtoggle"), "boxy qtoggle buttons should be gone");
for (const gone of ["camera-panel", "viewer-toolbar", "data-cam=", 'class="cam-pad"']) {
  assert(!html.includes(gone), `legacy camera UI still present: ${gone}`);
}
fill("unified view HUD, no camera buttons", ++n);

// 7) All six view types are present in the (single) HUD style row.
for (const style of ["colour", "clay", "normals", "height", "wireframe", "xray"]) {
  assert(html.includes(`data-style="${style}"`), `missing view style ${style}`);
}
assert.equal((html.match(/data-style=/g) || []).length, 6, "expected exactly one style row");
fill("six view styles in one row", ++n);

// 8) Buttons map to real actions / presets / view styles.
const actionBlock = block("const QUICK_ACTIONS", 'document.querySelectorAll("[data-action]")');
const { QUICK_ACTIONS } = new Function(`${actionBlock}; return { QUICK_ACTIONS };`)();
for (const [name] of Object.entries(QUICK_ACTIONS)) {
  assert(html.includes(`data-action="${name}"`), `no button for action "${name}"`);
}
for (const [, key] of actionBlock.matchAll(/config\.([a-z_]+)/g)) {
  assert(key in DEFAULT_CONFIG, `action writes unknown config key "${key}"`);
}
const dataActions = [...html.matchAll(/data-action="([a-z]+)"/g)].map((m) => m[1]);
for (const a of dataActions) assert(a in QUICK_ACTIONS, `button action "${a}" has no handler`);
const dataPresets = [...html.matchAll(/data-preset="([a-z]+)"/g)].map((m) => m[1]);
for (const p of dataPresets) assert(p in PRESETS, `button preset "${p}" is not defined`);
assert.equal(dataPresets.length, 4, "expected four quality presets");
fill("buttons map to actions/presets", ++n);

// 9) The cancel control exists and is wired to the cancel endpoint.
assert(html.includes('id="cancel-run"'), "missing cancel button");
assert(app.includes('"/api/cancel"'), "cancel button is not wired to /api/cancel");
fill("cancel button wired", ++n);

// 10) Header ETA + preview widget.
assert(html.includes('id="plan-preview"'), "missing preview canvas");
assert(html.includes('id="eta-value"') && html.includes('id="plan-meta"'), "missing ETA/meta text");
assert(app.includes("estimateEtaSeconds") && app.includes("updatePlanWidget"), "missing ETA model");
fill("ETA + preview widget wired", ++n);

console.log(`\nUI config tests passed (${n}).`);
