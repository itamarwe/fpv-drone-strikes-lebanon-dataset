import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SCENE_BASE = (window.SCENE_BASE || location.pathname.replace(/\/[^/]*$/, "/")).replace(/\/?$/, "/");
const APP_BASE = (window.APP_BASE ?? "").replace(/\/$/, "");
const API_BASE = (window.API_BASE ?? "").replace(/\/$/, "");

function apiUrl(path) {
  if (!path || path.startsWith("http://") || path.startsWith("https://")) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalized}`;
}

function sceneAsset(relative) {
  if (!relative || relative.startsWith("http://") || relative.startsWith("https://") || relative.startsWith("/")) {
    return relative;
  }
  return `${SCENE_BASE}${relative}`;
}

const canvas = document.getElementById("viewer");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0c0d0f, 1);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0c0d0f, 3.2, 8.5);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.001, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;

    const ambient = new THREE.AmbientLight(0xffffff, 1.5);
    scene.add(ambient);

    const root = new THREE.Group();
    scene.add(root);

    const pathGroup = new THREE.Group();
    const frustumGroup = new THREE.Group();
    const axesGroup = new THREE.Group();
    const measureGroup = new THREE.Group();
    root.add(pathGroup, frustumGroup, axesGroup, measureGroup);

    const pointMaterial = new THREE.PointsMaterial({
      size: Number(document.getElementById("pointSize").value),
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.98
    });

    let meta;
    let pointsObject;
    let currentFrustum;
    let currentCameraMarker;
    let pathLine;
    let fullPathPositions;
    let frameIndex = 0;
    let activeScaleMPerUnit = null;
    let isMeasureMode = false;
    let measureDistanceUnits = null;
    let measurementPoints = [];
    let measurementMarkers = [];
    let measurementLine = null;
    let isPlaying = false;
    let playbackTimeS = 0;
    let repeatPlayback = true;
    const clock = new THREE.Clock();
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let defaultScaleMPerUnit = 117.6;
    const scaleStorageKey = `vggt-viewer-scale-v4:${SCENE_BASE}`;

    const title = document.getElementById("title");
    const metaEl = document.getElementById("meta");
    const sceneSelectTop = document.getElementById("sceneSelectTop");
    const saveSceneButton = document.getElementById("saveSceneButton");
    const saveStateText = document.getElementById("saveStateText");
    const progress = document.getElementById("pathProgress");
    const frameValue = document.getElementById("frameValue");
    const pointSize = document.getElementById("pointSize");
    const pointSizeValue = document.getElementById("pointSizeValue");
    const scaleInput = document.getElementById("scaleInput");
    const measureButton = document.getElementById("measureButton");
    const scaleDialog = document.getElementById("scaleDialog");
    const scaleMeasuredText = document.getElementById("scaleMeasuredText");
    const scaleResultText = document.getElementById("scaleResultText");
    const realLengthM = document.getElementById("realLengthM");
    const speedReadout = document.getElementById("speedReadout");
    const heightReadout = document.getElementById("heightReadout");
    const pathReadout = document.getElementById("pathReadout");
    const cameraViewPanel = document.getElementById("cameraViewPanel");
    const cameraViewImage = document.getElementById("cameraViewImage");
    const cameraViewCaption = document.getElementById("cameraViewCaption");
    const speedPlot = document.getElementById("speedPlot");
    const heightPlot = document.getElementById("heightPlot");
    const topPlot = document.getElementById("topPlot");
    const playButton = document.getElementById("playButton");
    const repeatButton = document.getElementById("repeatButton");
    let cameraViewMode = "overlay";

    function vec3(values) {
      return new THREE.Vector3(values[0], values[1], values[2]);
    }

    function makeLine(points, color, opacity = 1, width = 1) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, linewidth: width });
      return new THREE.Line(geometry, material);
    }

    function makeGradientPath(path) {
      const positions = new Float32Array(path.length * 3);
      const colors = new Float32Array(path.length * 3);
      const start = new THREE.Color(0x36e4ff);
      const end = new THREE.Color(0xff4d6d);
      for (let i = 0; i < path.length; i += 1) {
        const p = path[i].position;
        positions.set(p, i * 3);
        const color = start.clone().lerp(end, i / Math.max(1, path.length - 1));
        colors.set([color.r, color.g, color.b], i * 3);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      const material = new THREE.LineBasicMaterial({ vertexColors: true });
      fullPathPositions = positions;
      return new THREE.Line(geometry, material);
    }

    function makeSphere(position, radius, color) {
      const geometry = new THREE.SphereGeometry(radius, 16, 12);
      const material = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(position);
      return mesh;
    }

    function currentScale() {
      return activeScaleMPerUnit || defaultScaleMPerUnit || null;
    }

    function syncScaleInput() {
      const scale = currentScale();
      if (scaleInput && Number.isFinite(scale)) {
        scaleInput.value = scale.toFixed(3).replace(/\.?0+$/, "");
      }
    }

    async function loadSceneState() {
      if (!meta?.scene_state_url) return null;
      try {
        const response = await fetch(apiUrl(meta.scene_state_url), { cache: "no-store" });
        if (!response.ok) return null;
        return await response.json();
      } catch {
        return null;
      }
    }

    async function saveSceneState(reason = "scale_update") {
      if (!Number.isFinite(activeScaleMPerUnit)) return false;
      const payload = {
        scale_m_per_unit: activeScaleMPerUnit,
        reason,
        frame_index: frameIndex,
        measured_vggt_units: measureDistanceUnits,
        real_length_m: Number(realLengthM.value) || null
      };
      localStorage.setItem(scaleStorageKey, String(activeScaleMPerUnit));
      if (!meta?.scene_state_url) return false;
      try {
        const response = await fetch(apiUrl(meta.scene_state_url), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        return response.ok;
      } catch {
        // Local standalone viewer still works without a backend.
        return false;
      }
    }

    function showSaveState(text) {
      if (!saveStateText) return;
      saveStateText.textContent = text;
      if (text) {
        window.clearTimeout(showSaveState.timeout);
        showSaveState.timeout = window.setTimeout(() => { saveStateText.textContent = ""; }, 2400);
      }
    }

    async function saveCurrentScene(reason = "save_button") {
      if (!saveSceneButton) return;
      saveSceneButton.disabled = true;
      const savedToBackend = await saveSceneState(reason);
      showSaveState(savedToBackend ? "Saved" : "Saved local");
      saveSceneButton.disabled = false;
    }

    function applyScale(scale, reason = "scale_update", persist = true, syncInputField = true) {
      if (!Number.isFinite(scale) || scale <= 0) return;
      activeScaleMPerUnit = scale;
      localStorage.setItem(scaleStorageKey, String(activeScaleMPerUnit));
      if (syncInputField) syncScaleInput();
      refreshGroundGridScale();
      axesGroup.clear();
      buildAxes(currentSceneRadius());
      updateMetaText();
      updateCurrentCamera(frameIndex);
      updateScaleDialogText();
      if (persist) saveSceneState(reason);
    }

    function formatNumber(value, digits = 3) {
      return Number.isFinite(value) ? value.toFixed(digits) : "--";
    }

    function updateMetaText() {
      if (!meta) return;
      const gridText = meta.ground_grid
        ? ` | grid ${meta.ground_grid.minor_step_m}/${meta.ground_grid.major_step_m}m`
        : "";
      metaEl.textContent = `${meta.point_count.toLocaleString()} points | ${meta.path.length} poses @ ${sampleFps().toFixed(1)} fps${gridText}`;
    }

    function sceneOptionLabel(scene) {
      const parts = [];
      if (scene.date) parts.push(scene.date);
      if (scene.town) parts.push(scene.town);
      if (scene.segment_ids?.length) parts.push(scene.segment_ids.join("+"));
      const prefix = parts.length ? `${parts.join(" | ")} - ` : "";
      return `${prefix}${scene.description || scene.scene_id}`;
    }

    async function loadSceneOptions() {
      if (!sceneSelectTop) return;
      try {
        const response = await fetch(apiUrl("/api/scenes"), { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        const scenes = (payload.scenes || []).filter((scene) => scene.exists && scene.viewer_url);
        sceneSelectTop.innerHTML = "";
        if (!scenes.length) {
          sceneSelectTop.innerHTML = '<option value="">No saved scenes</option>';
          sceneSelectTop.disabled = true;
          return;
        }
        for (const scene of scenes) {
          const option = document.createElement("option");
          option.value = scene.viewer_url;
          option.dataset.sceneId = scene.scene_id;
          option.textContent = sceneOptionLabel(scene);
          if (meta?.scene_id && scene.scene_id === meta.scene_id) option.selected = true;
          sceneSelectTop.appendChild(option);
        }
        sceneSelectTop.disabled = false;
      } catch {
        sceneSelectTop.innerHTML = '<option value="">Scenes unavailable</option>';
        sceneSelectTop.disabled = true;
      }
    }

    function setPlaying(next) {
      isPlaying = !!next;
      if (isPlaying) syncPlaybackTimeFromFrame();
      playButton.innerHTML = isPlaying ? "&#10074;&#10074;" : "&#9654;";
      playButton.title = isPlaying ? "Pause" : "Play";
      playButton.setAttribute("aria-label", playButton.title);
    }

    function frameTime(frame) {
      if (Number.isFinite(frame?.sequence_time_s)) return frame.sequence_time_s;
      if (Number.isFinite(frame?.segment_time_s)) return frame.segment_time_s;
      return 0;
    }

    function sampleFps() {
      const fps = Number(meta?.sample_fps);
      if (Number.isFinite(fps) && fps > 0) return fps;
      const path = meta?.path;
      if (path?.length >= 2) {
        const dt = frameTime(path[1]) - frameTime(path[0]);
        if (dt > 1e-6) return 1 / dt;
      }
      return 10;
    }

    function playbackEndTime() {
      const path = meta?.path;
      if (!path?.length) return 0;
      return frameTime(path[path.length - 1]) + 1 / sampleFps();
    }

    function syncPlaybackTimeFromFrame() {
      const frame = meta?.path?.[frameIndex];
      if (!frame) return;
      playbackTimeS = frameTime(frame);
    }

    function frameIndexAtTime(timeS) {
      const path = meta.path;
      for (let i = path.length - 1; i >= 0; i -= 1) {
        if (frameTime(path[i]) <= timeS) return i;
      }
      return 0;
    }

    function refreshGroundGridScale() {
      const grid = meta?.ground_grid;
      const scale = currentScale();
      if (!grid || !scale || !grid.minor_step_m || !grid.major_step_m || !grid.size_m) return;
      grid.size_units = grid.fixed_size_units || grid.size_units;
      grid.minor_step_units = grid.minor_step_m / scale;
      grid.major_step_units = grid.major_step_m / scale;
      grid.size_m = grid.size_units * scale;
    }

    // Recenter (and resize) the ground grid under the reconstructed scene itself,
    // rather than under the camera flight path. The generator centers the grid on
    // the path start/end midpoint, which leaves the point cloud sitting off to one
    // side; here we shift the origin to the robust center of the points projected
    // onto the ground plane so the scene lands in the middle of the grid.
    function recenterGridOnScene(positions) {
      const grid = groundGrid();
      if (!grid || !grid.origin || !grid.u || !grid.v || !meta?.ground_grid) return;
      const count = positions.length / 3;
      if (count < 100) return;
      const origin = vec3(grid.origin);
      const u = vec3(grid.u).normalize();
      const v = vec3(grid.v).normalize();
      // Subsample so a robust (percentile) center stays cheap on 1M-point clouds.
      const stride = Math.max(1, Math.floor(count / 40000));
      const us = [];
      const vs = [];
      const p = new THREE.Vector3();
      for (let i = 0; i < count; i += stride) {
        p.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).sub(origin);
        us.push(p.dot(u));
        vs.push(p.dot(v));
      }
      if (us.length < 8) return;
      us.sort((a, b) => a - b);
      vs.sort((a, b) => a - b);
      const q = (arr, t) => arr[Math.min(arr.length - 1, Math.max(0, Math.round(t * (arr.length - 1))))];
      const uLo = q(us, 0.01);
      const uHi = q(us, 0.99);
      const vLo = q(vs, 0.01);
      const vHi = q(vs, 0.99);
      const newOrigin = origin
        .clone()
        .add(u.clone().multiplyScalar((uLo + uHi) / 2))
        .add(v.clone().multiplyScalar((vLo + vHi) / 2));
      meta.ground_grid.origin = newOrigin.toArray();
      // Size the grid to frame the scene around the new center (path may extend
      // beyond the grid edge, which is fine).
      const major = meta.ground_grid.major_step_units || grid.major_step_units || 1;
      const span = Math.max(uHi - uLo, vHi - vLo);
      const size = Math.max(Math.ceil((span * 1.25) / major) * major, major * 5);
      meta.ground_grid.size_units = size;
      meta.ground_grid.fixed_size_units = size;
    }

    function measurementSummary() {
      if (measurementPoints.length === 0) return "Measure mode on: click the first point";
      if (measurementPoints.length === 1) return "Measure mode on: click the second point";
      const scale = currentScale();
      const meters = scale ? ` | ${(measureDistanceUnits * scale).toFixed(2)}m at active scale` : "";
      return `Measured ${measureDistanceUnits.toFixed(5)} VGGT units${meters}`;
    }

    function updateMeasureStatus() {
      measureButton.title = isMeasureMode ? measurementSummary() : "Measure two points";
    }

    function setMeasureMode(enabled) {
      if (enabled) clearMeasurement();
      isMeasureMode = enabled;
      controls.enabled = !enabled;
      measureButton.classList.toggle("active", enabled);
      measureButton.textContent = "Measure";
      renderer.domElement.style.cursor = enabled ? "crosshair" : "";
      updateMeasureStatus();
    }

    function updateScaleDialogText() {
      if (measureDistanceUnits == null) {
        scaleMeasuredText.textContent = "Select two points to measure a VGGT distance.";
        scaleResultText.textContent = "Scale: -- m/unit";
        return;
      }
      scaleMeasuredText.textContent = `Measured VGGT distance: ${measureDistanceUnits.toFixed(6)} units`;
      const real = Number(realLengthM.value);
      if (Number.isFinite(real) && real > 0) {
        const scale = real / measureDistanceUnits;
        scaleResultText.textContent = `Scale: ${scale.toFixed(4)} m/unit`;
      } else {
        scaleResultText.textContent = "Scale: -- m/unit";
      }
    }

    function openScaleDialog() {
      updateScaleDialogText();
      if (typeof scaleDialog.showModal === "function" && !scaleDialog.open) {
        scaleDialog.showModal();
      } else {
        scaleDialog.setAttribute("open", "");
      }
      realLengthM.focus();
    }

    function clearMeasurement() {
      measurementPoints = [];
      measureDistanceUnits = null;
      for (const marker of measurementMarkers) marker.removeFromParent();
      measurementMarkers = [];
      if (measurementLine) {
        measurementLine.removeFromParent();
        measurementLine.geometry.dispose();
        measurementLine.material.dispose();
        measurementLine = null;
      }
      updateMeasureStatus();
      updateScaleDialogText();
    }

    function setMeasurementLine() {
      if (measurementLine) {
        measurementLine.removeFromParent();
        measurementLine.geometry.dispose();
        measurementLine.material.dispose();
      }
      measurementLine = makeLine(measurementPoints, 0xfff36a, 1);
      measureGroup.add(measurementLine);
    }

    function addMeasurementPoint(point) {
      if (measurementPoints.length >= 2) clearMeasurement();
      const picked = point.clone();
      measurementPoints.push(picked);
      const marker = makeSphere(picked, 0.004, measurementPoints.length === 1 ? 0xffffff : 0xfff36a);
      measurementMarkers.push(marker);
      measureGroup.add(marker);
      if (measurementPoints.length === 2) {
        measureDistanceUnits = measurementPoints[0].distanceTo(measurementPoints[1]);
        setMeasurementLine();
        updateMeasureStatus();
        openScaleDialog();
      } else {
        updateMeasureStatus();
      }
    }

    function pickPoint(event) {
      if (!isMeasureMode || !pointsObject) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      raycaster.params.Points.threshold = Math.max(pointMaterial.size * 2.5, 0.012);
      const intersections = raycaster.intersectObject(pointsObject, false);
      // intersectObject returns hits sorted by depth along the ray, so
      // intersections[0] is the point nearest the camera within the threshold
      // cylinder -- not the one nearest the cursor. That mismatch is what makes
      // the picked point land with a lateral offset from where you clicked.
      // Pick the hit with the smallest perpendicular distance to the ray instead.
      let best = null;
      for (const hit of intersections) {
        if (hit.index == null) continue;
        if (best === null || hit.distanceToRay < best.distanceToRay) {
          best = hit;
        }
      }
      if (best === null) {
        return;
      }
      const position = new THREE.Vector3().fromBufferAttribute(pointsObject.geometry.getAttribute("position"), best.index);
      addMeasurementPoint(pointsObject.localToWorld(position));
    }

    function groundGrid() {
      const grid = meta?.ground_grid;
      if (!grid) return null;
      if (!grid.horizontal || !grid.fitted_normal) return grid;
      const normal = vec3(grid.fitted_normal).normalize();
      const ref = Math.abs(normal.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      const u = new THREE.Vector3().crossVectors(ref, normal).normalize();
      const v = new THREE.Vector3().crossVectors(normal, u).normalize();
      return { ...grid, normal: grid.fitted_normal, u: u.toArray(), v: v.toArray() };
    }

    function sceneAlignmentQuaternion() {
      const grid = groundGrid();
      if (!grid) return null;
      const stored = meta.scene_alignment_quaternion;
      if (Array.isArray(stored) && stored.length === 4) {
        return new THREE.Quaternion(stored[0], stored[1], stored[2], stored[3]);
      }
      const normal = grid.fitted_normal || grid.normal;
      return alignmentQuaternion(normal, grid.u);
    }

    function alignmentQuaternion(fromNormal, fromU) {
      const n = vec3(fromNormal).normalize();
      const qAlign = new THREE.Quaternion().setFromUnitVectors(n, new THREE.Vector3(0, 1, 0));
      const uRot = vec3(fromU).clone().applyQuaternion(qAlign).normalize();
      const uXZ = new THREE.Vector3(uRot.x, 0, uRot.z);
      if (uXZ.lengthSq() < 1e-8) uXZ.set(1, 0, 0);
      else uXZ.normalize();
      const qTwist = new THREE.Quaternion().setFromUnitVectors(uXZ, new THREE.Vector3(1, 0, 0));
      return qTwist.multiply(qAlign);
    }

    function applySceneAlignment() {
      const q = sceneAlignmentQuaternion();
      if (!q) return;
      root.quaternion.copy(q);
    }

    function buildGroundGrid() {
      const grid = groundGrid();
      if (!grid) return;
      const origin = vec3(grid.origin);
      const u = vec3(grid.u).normalize();
      const v = vec3(grid.v).normalize();
      const half = grid.size_units / 2;
      const step = grid.minor_step_units;
      const majorEvery = Math.max(1, Math.round(grid.major_step_units / grid.minor_step_units));
      const lineCount = Math.ceil(half / step);
      for (let i = -lineCount; i <= lineCount; i += 1) {
        const offset = i * step;
        const isMajor = i % majorEvery === 0;
        const color = isMajor ? 0x4b6472 : 0x25313a;
        const opacity = isMajor ? 0.86 : 0.46;
        const a1 = origin.clone().add(u.clone().multiplyScalar(-half)).add(v.clone().multiplyScalar(offset));
        const a2 = origin.clone().add(u.clone().multiplyScalar(half)).add(v.clone().multiplyScalar(offset));
        const b1 = origin.clone().add(v.clone().multiplyScalar(-half)).add(u.clone().multiplyScalar(offset));
        const b2 = origin.clone().add(v.clone().multiplyScalar(half)).add(u.clone().multiplyScalar(offset));
        axesGroup.add(makeLine([a1, a2], color, opacity));
        axesGroup.add(makeLine([b1, b2], color, opacity));
      }
      const normal = vec3(grid.normal).normalize();
      axesGroup.add(makeLine([origin, origin.clone().add(normal.multiplyScalar(grid.minor_step_units * 1.5))], 0xffd166, 0.9));
    }

    function canvasContext(canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, width: rect.width, height: rect.height };
    }

    function drawPlotFrame(ctx, width, height, title, xLabel, yLabel) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(255,255,255,0.035)";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
      ctx.fillStyle = "#dce4eb";
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(title, 8, 15);
      ctx.fillStyle = "#8fa0ad";
      ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(xLabel, width - 70, height - 7);
      ctx.save();
      ctx.translate(10, height - 18);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    function makeScale(values, minPad = 0.08) {
      let min = Math.min(...values.filter(Number.isFinite));
      let max = Math.max(...values.filter(Number.isFinite));
      if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
      if (Math.abs(max - min) < 1e-9) {
        min -= 0.5;
        max += 0.5;
      }
      const pad = (max - min) * minPad;
      return { min: min - pad, max: max + pad };
    }

    function mapValue(value, min, max, outMin, outMax) {
      return outMin + (value - min) / (max - min) * (outMax - outMin);
    }

    function formatTick(value) {
      if (!Number.isFinite(value)) return "--";
      const abs = Math.abs(value);
      if (abs >= 100) return value.toFixed(0);
      if (abs >= 10) return value.toFixed(1);
      return value.toFixed(2);
    }

    function smoothSeries(values, radius = 4) {
      return values.map((value, index) => {
        let total = 0;
        let count = 0;
        const start = Math.max(0, index - radius);
        const end = Math.min(values.length, index + radius + 1);
        for (let i = start; i < end; i += 1) {
          if (Number.isFinite(values[i])) {
            total += values[i];
            count += 1;
          }
        }
        return count ? total / count : value;
      });
    }

    function compactSegmentLabel(frame) {
      if (frame.is_attack) return "attack";
      const raw = String(frame.segment_id || `seg${frame.segment_index || ""}` || "flight");
      const match = raw.match(/seg\d+$/i);
      return match ? match[0] : raw.length > 14 ? "flight" : raw;
    }

    function drawEndpointLabels(ctx, plot, width, height, leftLabel, rightLabel) {
      ctx.fillStyle = "#aeb8c3";
      ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(leftLabel, plot.left, height - 7);
      const rightWidth = ctx.measureText(rightLabel).width;
      const rightX = Math.max(plot.left + ctx.measureText(leftLabel).width + 10, plot.right - rightWidth);
      ctx.fillText(rightLabel, Math.min(rightX, width - rightWidth - 4), height - 7);
    }

    function drawLegend(ctx, plot, legend) {
      if (!legend?.length) return;
      ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
      let x = plot.left + 2;
      const y = plot.top - 7;
      for (const item of legend) {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = item.width || 2;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 12, y);
        ctx.stroke();
        x += 16;
        ctx.fillStyle = "#aeb8c3";
        ctx.fillText(item.label, x, y + 3);
        x += ctx.measureText(item.label).width + 12;
      }
    }

    function drawLineSeries(canvas, title, xLabel, yLabel, xs, seriesInput, currentX = null, options = {}) {
      const { ctx, width, height } = canvasContext(canvas);
      drawPlotFrame(ctx, width, height, title, xLabel, yLabel);
      const plot = { left: 34, right: width - 10, top: 22, bottom: height - 20 };
      const series = Array.isArray(seriesInput) && seriesInput[0]?.values
        ? seriesInput
        : [{ values: seriesInput, color: options.color || "#36e4ff", alpha: 1, width: 2 }];
      const yValues = series.flatMap((entry) => entry.values).filter(Number.isFinite);
      const xr = makeScale(options.xScaleValues || xs, options.xPad ?? 0.03);
      const yr = makeScale(yValues, options.yPad ?? 0.12);
      ctx.strokeStyle = "rgba(255,255,255,0.10)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 3; i += 1) {
        const y = plot.top + (plot.bottom - plot.top) * i / 3;
        ctx.beginPath();
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.right, y);
        ctx.stroke();
      }
      for (const entry of series) {
        ctx.save();
        ctx.strokeStyle = entry.color || "#36e4ff";
        ctx.globalAlpha = entry.alpha ?? 1;
        ctx.lineWidth = entry.width || 2;
        ctx.beginPath();
        let started = false;
        const n = Math.min(xs.length, entry.values.length);
        for (let i = 0; i < n; i += 1) {
          const xValue = xs[i];
          const yValue = entry.values[i];
          if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) {
            started = false;
            continue;
          }
          const x = mapValue(xValue, xr.min, xr.max, plot.left, plot.right);
          const y = mapValue(yValue, yr.min, yr.max, plot.bottom, plot.top);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
        ctx.restore();
      }
      if (Number.isFinite(currentX)) {
        const x = mapValue(currentX, xr.min, xr.max, plot.left, plot.right);
        ctx.strokeStyle = "#ffb000";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.bottom);
        ctx.stroke();
      }
      if (options.currentPoint && Number.isFinite(options.currentPoint.x) && Number.isFinite(options.currentPoint.y)) {
        const x = mapValue(options.currentPoint.x, xr.min, xr.max, plot.left, plot.right);
        const y = mapValue(options.currentPoint.y, yr.min, yr.max, plot.bottom, plot.top);
        ctx.fillStyle = options.currentPoint.color || "#ffb000";
        ctx.strokeStyle = "rgba(0,0,0,0.55)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, options.currentPoint.radius || 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.fillStyle = "#aeb8c3";
      ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(formatTick(yr.max), 6, plot.top + 4);
      ctx.fillText(formatTick(Math.max(0, yr.min)), 6, plot.bottom);
      drawEndpointLabels(
        ctx,
        plot,
        width,
        height,
        options.xLeftLabel ?? formatTick(Math.max(0, xr.min)),
        options.xRightLabel ?? formatTick(xr.max)
      );
      drawLegend(ctx, plot, options.legend);
    }

    function groundHeightUnits(position) {
      const grid = groundGrid();
      if (!grid) return 0;
      const normal = vec3(grid.normal);
      return normal.dot(position) + grid.d;
    }

    function drawTopView(scale) {
      const { ctx, width, height } = canvasContext(topPlot);
      drawPlotFrame(ctx, width, height, "top view", "m", "m");
      const grid = groundGrid();
      if (!grid) return;
      const origin = vec3(grid.origin);
      const u = vec3(grid.u).normalize();
      const v = vec3(grid.v).normalize();
      const coords = meta.path.map((frame) => {
        const p = vec3(frame.position).sub(origin);
        return [p.dot(u) * scale, p.dot(v) * scale];
      });
      const xs = coords.map((p) => p[0]);
      const ys = coords.map((p) => p[1]);
      const xr = makeScale(xs, 0.18);
      const yr = makeScale(ys, 0.18);
      const plot = { left: 34, right: width - 10, top: 22, bottom: height - 20 };
      const toCanvas = ([x, y]) => [
        mapValue(x, xr.min, xr.max, plot.left, plot.right),
        mapValue(y, yr.min, yr.max, plot.bottom, plot.top)
      ];
      ctx.strokeStyle = "#36e4ff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      coords.forEach((point, index) => {
        const [x, y] = toCanvas(point);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const markers = [
        { point: coords[0], color: "#36e4ff", r: 3.5 },
        { point: coords[coords.length - 1], color: "#ff4d6d", r: 4.5 },
        { point: coords[frameIndex], color: "#ffb000", r: 4.0 }
      ];
      for (const marker of markers) {
        const [x, y] = toCanvas(marker.point);
        ctx.fillStyle = marker.color;
        ctx.beginPath();
        ctx.arc(x, y, marker.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function computePathSeries(scale) {
      if (!meta?.path?.length) return null;
      const speedTimes = [];
      const speeds = [];
      const speedByFrame = new Array(meta.path.length).fill(null);
      for (let i = 1; i < meta.path.length; i += 1) {
        const prev = vec3(meta.path[i - 1].position);
        const next = vec3(meta.path[i].position);
        const prevFrame = meta.path[i - 1];
        const frame = meta.path[i];
        const t0 = Number.isFinite(prevFrame.sequence_time_s) ? prevFrame.sequence_time_s : prevFrame.segment_time_s;
        const t1 = Number.isFinite(frame.sequence_time_s) ? frame.sequence_time_s : frame.segment_time_s;
        const sameSegment = (prevFrame.segment_id || prevFrame.segment_index) === (frame.segment_id || frame.segment_index);
        const dt = Math.max(1e-6, t1 - t0);
        const speed = sameSegment ? prev.distanceTo(next) * scale / dt : null;
        speedTimes.push(t1);
        speeds.push(speed);
        speedByFrame[i] = speed;
      }
      const smoothedSpeeds = smoothSeries(speeds, 4);
      const smoothedSpeedByFrame = new Array(meta.path.length).fill(null);
      for (let i = 1; i < meta.path.length; i += 1) {
        smoothedSpeedByFrame[i] = smoothedSpeeds[i - 1] ?? null;
      }
      const distances = meta.path.map((frame) => frame.distance_to_end_units * scale);
      const heights = meta.path.map((frame) => Math.max(0, groundHeightUnits(vec3(frame.position)) * scale));
      const startDistance = distances[0] || 0;
      const progressToTarget = distances.map((distance) => startDistance - distance);
      return {
        speedTimes,
        speeds,
        speedByFrame,
        smoothedSpeeds,
        smoothedSpeedByFrame,
        distances,
        heights,
        startDistance,
        progressToTarget
      };
    }

    function updateTelemetry(series) {
      if (!meta?.path?.length || !series) return;
      const frame = meta.path[frameIndex];
      const raw = series.speedByFrame[frameIndex];
      const smooth = series.smoothedSpeedByFrame[frameIndex];
      const segment = compactSegmentLabel(frame);
      speedReadout.textContent = Number.isFinite(smooth)
        ? `${formatTick(smooth)} m/s smooth${Number.isFinite(raw) ? ` | ${formatTick(raw)} raw` : ""}`
        : "--";
      heightReadout.textContent = `${formatTick(series.heights[frameIndex])}m AGL | ${formatTick(series.distances[frameIndex])}m tgt`;
      pathReadout.textContent = `${frameIndex + 1}/${meta.path.length} | t=${frame.video_time_s.toFixed(1)}s | ${segment}`;
    }

    function updateCameraView() {
      const frame = meta?.path?.[frameIndex];
      if (!frame || !cameraViewImage) return;
      const key = `${cameraViewMode}_image`;
      const src = cameraViewMode === "actual"
        ? (frame.actual_image || frame.frame_image || "")
        : (frame[key] || "");
      for (const button of document.querySelectorAll("[data-view-mode]")) {
        button.classList.toggle("active", button.dataset.viewMode === cameraViewMode);
        const candidateKey = `${button.dataset.viewMode}_image`;
        const hasImage = button.dataset.viewMode === "actual"
          ? !!(frame.actual_image || frame.frame_image)
          : !!frame[candidateKey];
        button.disabled = !hasImage;
      }
      if (!src) {
        cameraViewPanel.style.display = "grid";
        cameraViewImage.removeAttribute("src");
        cameraViewImage.style.display = "none";
        cameraViewCaption.textContent = `${cameraViewMode}: no image for this frame`;
        return;
      }
      cameraViewPanel.style.display = "grid";
      cameraViewImage.style.display = "block";
      cameraViewImage.src = sceneAsset(src);
      cameraViewCaption.textContent = `${cameraViewMode} | ${frame.file || `frame ${frame.frame}`}`;
    }

    function drawPlots() {
      if (!meta || !speedPlot || !heightPlot || !topPlot) return;
      const scale = currentScale() || 1;
      const series = computePathSeries(scale);
      if (!series) return;
      const currentRemainingDistance = meta.path[frameIndex]?.distance_to_end_units * scale;
      const speedPointIndex = frameIndex > 0 ? frameIndex - 1 : -1;
      drawLineSeries(
        speedPlot,
        "speed",
        "time",
        "m/s",
        series.speedTimes,
        [
          { values: series.speeds, color: "#36e4ff", alpha: 0.30, width: 1.2 },
          { values: series.smoothedSpeeds, color: "#ffb000", alpha: 1, width: 2.1 }
        ],
        meta.path[frameIndex]?.sequence_time_s ?? meta.path[frameIndex]?.segment_time_s,
        {
          currentPoint: {
            x: series.speedTimes[speedPointIndex],
            y: series.smoothedSpeeds[speedPointIndex],
            color: "#ffb000"
          },
          legend: [
            { label: "raw", color: "#36e4ff", width: 1.2 },
            { label: "smooth", color: "#ffb000", width: 2.1 }
          ]
        }
      );
      drawLineSeries(
        heightPlot,
        "height",
        "to target",
        "m",
        series.progressToTarget,
        [{ values: series.heights, color: "#9dff57", alpha: 1, width: 2 }],
        Number.isFinite(currentRemainingDistance) ? series.startDistance - currentRemainingDistance : null,
        {
          xLeftLabel: `${formatTick(series.startDistance)}m`,
          xRightLabel: "0m",
          currentPoint: {
            x: series.progressToTarget[frameIndex],
            y: series.heights[frameIndex],
            color: "#ffb000"
          }
        }
      );
      drawTopView(scale);
      updateTelemetry(series);
      updateCameraView();
    }

    function currentSceneBox() {
      // Everything here is in root-LOCAL coords (path/grid are raw); fitView maps
      // the resulting center/eye to world via toWorld(). Use the points geometry's
      // LOCAL bounding box -- setFromObject() would return a WORLD box (alignment
      // rotation already applied), which fitView would then rotate a second time,
      // throwing the camera off-screen for scenes with a tilted ground.
      const box = new THREE.Box3();
      if (pointsObject?.geometry) {
        pointsObject.geometry.computeBoundingBox();
        if (pointsObject.geometry.boundingBox) box.copy(pointsObject.geometry.boundingBox);
      }
      if (meta?.path?.length) {
        for (const frame of meta.path) box.expandByPoint(vec3(frame.position));
      }
      const grid = groundGrid();
      if (grid) {
        const origin = vec3(grid.origin);
        const u = vec3(grid.u).normalize();
        const v = vec3(grid.v).normalize();
        const half = grid.size_units / 2;
        for (const su of [-1, 1]) {
          for (const sv of [-1, 1]) {
            box.expandByPoint(origin.clone().add(u.clone().multiplyScalar(su * half)).add(v.clone().multiplyScalar(sv * half)));
          }
        }
      }
      return box;
    }

    function currentSceneRadius() {
      const sphere = new THREE.Sphere();
      currentSceneBox().getBoundingSphere(sphere);
      return Math.max(0.2, sphere.radius || 0.2);
    }

    function createFrustum(frame, color = 0x36e4ff, scale = 0.045, opacity = 0.56) {
      const center = vec3(frame.position);
      const forward = vec3(frame.forward).normalize();
      const right = vec3(frame.right).normalize();
      const down = vec3(frame.down).normalize();
      const nearCenter = center.clone().add(forward.clone().multiplyScalar(scale));
      const halfW = scale * 0.58;
      const halfH = scale * 0.34;
      const corners = [
        nearCenter.clone().add(right.clone().multiplyScalar(-halfW)).add(down.clone().multiplyScalar(-halfH)),
        nearCenter.clone().add(right.clone().multiplyScalar(halfW)).add(down.clone().multiplyScalar(-halfH)),
        nearCenter.clone().add(right.clone().multiplyScalar(halfW)).add(down.clone().multiplyScalar(halfH)),
        nearCenter.clone().add(right.clone().multiplyScalar(-halfW)).add(down.clone().multiplyScalar(halfH)),
      ];
      const linePoints = [
        center, corners[0], corners[1], center, corners[2], corners[1],
        corners[2], corners[3], center, corners[0], corners[3]
      ];
      return makeLine(linePoints, color, opacity);
    }

    function toWorld(localPoint) {
      return localPoint.clone().applyMatrix4(root.matrixWorld);
    }

    function fitView() {
      root.updateMatrixWorld(true);
      const box = currentSceneBox();
      if (box.isEmpty()) return;
      const sphere = new THREE.Sphere();
      box.getBoundingSphere(sphere);
      const localCenter = sphere.center.clone();
      const radius = Math.max(0.2, sphere.radius);
      const viewDirection = new THREE.Vector3(0.88, 0.58, 1.38).normalize();
      const distance = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov) / 2)) * 1.32;
      const localEye = localCenter.clone().add(viewDirection.multiplyScalar(distance));
      applyCameraView(toWorld(localCenter), toWorld(localEye), radius);
    }

    function updateCurrentCamera(index) {
      frameIndex = Math.max(0, Math.min(meta.path.length - 1, index));
      const frame = meta.path[frameIndex];
      currentCameraMarker.position.copy(vec3(frame.position));
      if (currentFrustum) currentFrustum.removeFromParent();
      currentFrustum = createFrustum(frame, 0xffb000, 0.08, 1);
      pathGroup.add(currentFrustum);
      progress.value = String(frameIndex);
      frameValue.textContent = String(frame.frame);
      playbackTimeS = frameTime(frame);
      drawPlots();
    }

    function buildAxes(radius) {
      axesGroup.add(new THREE.AxesHelper(radius * 0.36));
      buildGroundGrid();
    }

    function applyCameraView(target, eye, radius) {
      const distance = eye.distanceTo(target);
      camera.position.copy(eye);
      camera.near = Math.max(0.0005, radius / 3000);
      camera.far = distance + radius * 10;
      camera.updateProjectionMatrix();
      controls.target.copy(target);
      controls.minDistance = radius * 0.05;
      controls.maxDistance = distance + radius * 8;
      controls.update();
      axesGroup.clear();
      buildAxes(radius);
      drawPlots();
    }

    async function loadBinary(url, TypedArray) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${url}: ${response.status}`);
      return new TypedArray(await response.arrayBuffer());
    }

    async function main() {
      meta = await (await fetch(sceneAsset("scene_meta.json"))).json();
      defaultScaleMPerUnit = Number(meta.default_scale_m_per_unit) > 0
        ? Number(meta.default_scale_m_per_unit)
        : 117.6;
      const sceneState = await loadSceneState();
      const storedScale = Number(localStorage.getItem(scaleStorageKey));
      const savedScale = Number(sceneState?.scale_m_per_unit);
      activeScaleMPerUnit = Number.isFinite(savedScale) && savedScale > 0
        ? savedScale
        : Number.isFinite(storedScale) && storedScale > 0
        ? storedScale
        : defaultScaleMPerUnit;
      syncScaleInput();
      refreshGroundGridScale();
      applySceneAlignment();
      title.textContent = meta.title;
      updateMetaText();
      loadSceneOptions();
      progress.max = String(meta.path.length - 1);

      const [positions, colors8] = await Promise.all([
        loadBinary(sceneAsset(meta.assets.positions), Float32Array),
        loadBinary(sceneAsset(meta.assets.colors), Uint8Array)
      ]);
      const colors = new Float32Array(colors8.length);
      for (let i = 0; i < colors8.length; i += 1) colors[i] = colors8[i] / 255;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      pointsObject = new THREE.Points(geometry, pointMaterial);
      root.add(pointsObject);
      recenterGridOnScene(positions);

      pathLine = makeGradientPath(meta.path);
      pathGroup.add(pathLine);
      const start = makeSphere(vec3(meta.path[0].position), 0.012, 0x36e4ff);
      const end = makeSphere(vec3(meta.path[meta.path.length - 1].position), 0.017, 0xff4d6d);
      pathGroup.add(start, end);

      for (let i = 0; i < meta.path.length; i += 10) {
        frustumGroup.add(createFrustum(meta.path[i], 0x36e4ff, 0.045, 0.48));
      }
      frustumGroup.add(createFrustum(meta.path[meta.path.length - 1], 0xff4d6d, 0.065, 0.82));
      currentCameraMarker = makeSphere(vec3(meta.path[0].position), 0.015, 0xffb000);
      pathGroup.add(currentCameraMarker);
      fitView();
      updateCurrentCamera(0);
      updateMeasureStatus();
    }

    pointSize.addEventListener("input", () => {
      const value = Number(pointSize.value);
      pointMaterial.size = value;
      pointSizeValue.textContent = value.toFixed(3);
    });
    progress.addEventListener("input", () => updateCurrentCamera(Number(progress.value)));
    document.getElementById("resetView").addEventListener("click", fitView);
    sceneSelectTop.addEventListener("change", () => {
      if (sceneSelectTop.value) window.location.href = sceneSelectTop.value;
    });
    saveSceneButton.addEventListener("click", () => saveCurrentScene("save_button"));
    playButton.addEventListener("click", () => setPlaying(!isPlaying));
    repeatButton.addEventListener("click", () => {
      repeatPlayback = !repeatPlayback;
      repeatButton.classList.toggle("active", repeatPlayback);
    });
    document.addEventListener("keydown", (event) => {
      if (event.code !== "Space") return;
      const tag = document.activeElement?.tagName;
      if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag)) return;
      event.preventDefault();
      setPlaying(!isPlaying);
    });
    document.getElementById("showFrustums").addEventListener("change", (event) => { frustumGroup.visible = event.target.checked; });
    document.getElementById("showPath").addEventListener("change", (event) => { pathGroup.visible = event.target.checked; });
    document.getElementById("showAxes").addEventListener("change", (event) => { axesGroup.visible = event.target.checked; });
    document.getElementById("showPoints").addEventListener("change", (event) => { pointsObject.visible = event.target.checked; });
    document.getElementById("showDrone").addEventListener("change", (event) => {
      if (currentCameraMarker) currentCameraMarker.visible = event.target.checked;
    });
    measureButton.addEventListener("click", () => setMeasureMode(!isMeasureMode));
    renderer.domElement.addEventListener("pointerdown", pickPoint);
    realLengthM.addEventListener("input", updateScaleDialogText);
    scaleInput.addEventListener("input", () => {
      const next = Number(scaleInput.value);
      if (!Number.isFinite(next) || next <= 0) return;
      applyScale(next, "typed_scale", false, false);
    });
    scaleInput.addEventListener("change", () => {
      const next = Number(scaleInput.value);
      if (!Number.isFinite(next) || next <= 0) {
        syncScaleInput();
        return;
      }
      applyScale(next, "typed_scale", true);
    });
    for (const button of document.querySelectorAll("[data-view-mode]")) {
      button.addEventListener("click", () => {
        cameraViewMode = button.dataset.viewMode;
        updateCameraView();
      });
    }
    cameraViewImage.addEventListener("error", () => {
      cameraViewImage.style.display = "none";
      cameraViewCaption.textContent = "Camera-view image is unavailable from this server.";
    });
    scaleDialog.addEventListener("close", () => setMeasureMode(false));
    document.getElementById("applyScale").addEventListener("click", () => {
      if (measureDistanceUnits == null) return;
      const real = Number(realLengthM.value);
      if (!Number.isFinite(real) || real <= 0) {
        scaleResultText.textContent = "Enter a positive real length.";
        return;
      }
      applyScale(real / measureDistanceUnits, "measured_scale", true);
      setMeasureMode(false);
      if (scaleDialog.open) scaleDialog.close();
    });

    function resize() {
      const width = window.innerWidth;
      const height = window.innerHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      drawPlots();
    }
    window.addEventListener("resize", resize);
    resize();

    function animate() {
      requestAnimationFrame(animate);
      if (meta && isPlaying) {
        playbackTimeS += clock.getDelta();
        const endTime = playbackEndTime();
        if (playbackTimeS >= endTime) {
          if (repeatPlayback) playbackTimeS = frameTime(meta.path[0]);
          else {
            playbackTimeS = endTime;
            setPlaying(false);
          }
        }
        const next = frameIndexAtTime(playbackTimeS);
        if (next !== frameIndex) updateCurrentCamera(next);
      } else {
        clock.getDelta();
      }
      controls.update();
      renderer.render(scene, camera);
    }
    main().catch((error) => {
      metaEl.textContent = error.stack || String(error);
      console.error(error);
    });
    animate();
