#!/usr/bin/env node
/**
 * Bake measured scale into scene metadata.
 *
 * The measure tool writes a calibrated scale to a scene's scene_state.json
 * (reason "measured_scale"), but the read-only web viewer only reads
 * scene_meta.json. This copies a real measurement into every scene_meta.json of
 * the same video so the viewer reports absolute height/speed instead of the
 * generic 117.6 m/unit fallback. Idempotent; run before publishing.
 *
 *   node tools/apply_calibration.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const scenesDir = path.join(repoRoot, "scenes");

function readJson(p) {
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return null;
  }
}

// A scene_state.json only counts as a real calibration if it came from a
// measurement (two picked points on a known length), not a default/verify pass.
function measuredScale(sceneDir) {
  const st = readJson(path.join(sceneDir, "scene_state.json"));
  if (!st) return null;
  if (st.reason !== "measured_scale" || !st.measured_vggt_units) return null;
  const s = Number(st.scale_m_per_unit);
  if (!(s > 0)) return null;
  return {
    scale: s,
    real_length_m: st.real_length_m ?? null,
    measured_vggt_units: st.measured_vggt_units,
    updated_at: st.updated_at ?? null,
  };
}

let applied = 0;
if (fs.existsSync(scenesDir)) {
  for (const stem of fs.readdirSync(scenesDir, { withFileTypes: true })) {
    if (!stem.isDirectory()) continue;
    const stemDir = path.join(scenesDir, stem.name);
    const sceneDirs = fs
      .readdirSync(stemDir, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => path.join(stemDir, d.name));

    // The measurement is a property of the reconstruction, so any scene dir of
    // this video (base or variant) that has one applies to all of them.
    let cal = null;
    for (const d of sceneDirs) {
      cal = measuredScale(d);
      if (cal) break;
    }
    if (!cal) continue;

    for (const d of sceneDirs) {
      const metaPath = path.join(d, "viewer", "scene_meta.json");
      const meta = readJson(metaPath);
      if (!meta) continue;
      meta.default_scale_m_per_unit = cal.scale;
      meta.calibration = {
        scale_m_per_vggt_unit: cal.scale,
        real_length_m: cal.real_length_m,
        measured_vggt_units: cal.measured_vggt_units,
        source: "measured",
        measured_at: cal.updated_at,
      };
      fs.writeFileSync(metaPath, JSON.stringify(meta));
      applied += 1;
    }
    console.log(`calibrated ${stem.name}: ${cal.scale.toFixed(2)} m/unit (${cal.real_length_m ?? "?"} m ref)`);
  }
}
console.log(`done: baked calibration into ${applied} scene_meta.json file(s)`);
