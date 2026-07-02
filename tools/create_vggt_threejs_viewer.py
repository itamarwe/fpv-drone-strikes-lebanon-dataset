#!/usr/bin/env python3
"""Create an interactive Three.js viewer for a VGGT reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VGGT Scene Viewer</title>
  <style>
    * {
      box-sizing: border-box;
    }
    :root {
      --hud-pad: 24px;
      --hud-pad-x2: 48px;
      --topbar-h: 52px;
    }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #0c0d0f;
      color: #e8edf2;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #viewer {
      position: fixed;
      inset: 0;
      display: block;
      width: 100vw;
      height: 100vh;
    }
    .topbar {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      gap: 10px;
      height: var(--topbar-h);
      padding: 8px 14px;
      background: rgba(10, 12, 15, 0.92);
      border-bottom: 1px solid rgba(255, 255, 255, 0.10);
      backdrop-filter: blur(12px);
    }
    .logo {
      color: #36e4ff;
      font-weight: 800;
      letter-spacing: 0;
      white-space: nowrap;
    }
    .top-nav {
      display: flex;
      gap: 4px;
    }
    .top-nav a {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 0 10px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      color: #aeb8c3;
      text-decoration: none;
      font-size: 13px;
    }
    .top-nav a.active {
      color: #061018;
      background: #36e4ff;
      border-color: #36e4ff;
      font-weight: 800;
    }
    .scene-picker {
      display: block;
      flex: 1;
      min-width: 0;
      max-width: 520px;
    }
    #sceneSelectTop {
      min-width: 0;
      width: 100%;
      min-height: 32px;
      padding: 0 8px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      color: #e8edf2;
      background: rgba(255, 255, 255, 0.08);
      font: 13px ui-sans-serif, system-ui, sans-serif;
      text-transform: none;
      letter-spacing: 0;
    }
    #saveStateText {
      min-width: 62px;
      color: #8fa0ad;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .hdr-btns {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-left: auto;
      flex-shrink: 0;
    }
    .panel {
      position: fixed;
      left: var(--hud-pad);
      right: var(--hud-pad);
      bottom: var(--hud-pad);
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 40px;
      color: #e8edf2;
      text-shadow: 0 1px 8px rgba(0, 0, 0, 0.78);
      pointer-events: none;
    }
    .panel > * {
      pointer-events: auto;
    }
    .scene-label {
      display: grid;
      gap: 2px;
      min-width: 190px;
      max-width: min(380px, 27vw);
    }
    .title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .title span {
      min-width: 0;
    }
    .title button {
      flex: 0 0 auto;
    }
    .meta {
      color: #aeb8c3;
      font-size: 12px;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .scene-label strong {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      letter-spacing: 0;
    }
    .row {
      display: grid;
      grid-template-columns: 112px minmax(0, 1fr) 42px;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }
    .strip-range {
      display: grid;
      grid-template-columns: auto 96px 44px;
      align-items: center;
      gap: 7px;
      min-height: 32px;
      padding: 0 8px;
      border-radius: 8px;
      background: rgba(10, 12, 15, 0.56);
      color: #d5dde5;
      font-size: 12px;
      backdrop-filter: blur(8px);
    }
    .path-range {
      flex: 1 1 280px;
      grid-template-columns: auto minmax(160px, 1fr) 40px;
      max-width: 520px;
    }
    .toggles {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .measure-status {
      display: none;
      min-height: 32px;
      padding: 8px 10px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      color: #cbd5df;
      background: rgba(255, 255, 255, 0.055);
      font-size: 12px;
      line-height: 1.35;
    }
    dialog.scale-dialog {
      width: min(360px, calc(100vw - 28px));
      padding: 0;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 8px;
      color: #e8edf2;
      background: #11151a;
      box-shadow: 0 22px 60px rgba(0, 0, 0, 0.5);
    }
    dialog.scale-dialog::backdrop {
      background: rgba(0, 0, 0, 0.36);
    }
    .scale-form {
      display: grid;
      gap: 10px;
      padding: 14px;
    }
    .scale-form h2 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }
    .scale-form p {
      margin: 0;
      color: #aeb8c3;
      font-size: 12px;
      line-height: 1.4;
    }
    .scale-form input[type="number"] {
      width: 100%;
      min-height: 34px;
      padding: 6px 8px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 8px;
      color: #e8edf2;
      background: rgba(255, 255, 255, 0.08);
      font: inherit;
    }
    .scale-dialog-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    label {
      color: #d5dde5;
      font-size: 12px;
    }
    input[type="range"] {
      width: 100%;
      accent-color: #36e4ff;
    }
    button, .toggle {
      min-height: 32px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      color: #e8edf2;
      background: rgba(255, 255, 255, 0.08);
      font: inherit;
    }
    button {
      cursor: pointer;
    }
    button:disabled {
      cursor: default;
      opacity: 0.45;
    }
    button.active {
      border-color: rgba(54, 228, 255, 0.72);
      background: rgba(54, 228, 255, 0.16);
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 0 9px;
    }
    .toggle input {
      accent-color: #36e4ff;
    }
    .plots-panel {
      position: fixed;
      top: calc(var(--topbar-h) + var(--hud-pad));
      right: var(--hud-pad);
      display: grid;
      gap: 7px;
      width: min(340px, calc(100vw - var(--hud-pad-x2)));
      padding: 9px;
      background: rgba(10, 12, 15, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
    }
    .telemetry {
      display: grid;
      gap: 7px;
      padding: 8px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.045);
      border: 1px solid rgba(255, 255, 255, 0.10);
    }
    .telemetry-row {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      color: #dce4eb;
      font-size: 12px;
      line-height: 1.25;
    }
    .telemetry-row span:first-child {
      color: #8fa0ad;
      font-size: 10px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .telemetry-value {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }
    .scale-control {
      display: flex;
      align-items: center;
      gap: 6px;
      min-height: 32px;
      padding: 0 7px;
      border-radius: 8px;
      background: rgba(10, 12, 15, 0.56);
      backdrop-filter: blur(8px);
      align-items: center;
      color: #dce4eb;
      font-size: 12px;
    }
    .scale-control label {
      color: #8fa0ad;
      font-size: 10px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    #scaleInput {
      width: 76px;
      min-height: 30px;
      padding: 5px 7px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      color: #e8edf2;
      background: rgba(255, 255, 255, 0.08);
      font: inherit;
      font-variant-numeric: tabular-nums;
    }
    .camera-view {
      position: fixed;
      left: var(--hud-pad);
      top: calc(var(--topbar-h) + var(--hud-pad));
      display: grid;
      gap: 7px;
      width: min(430px, calc(100vw - var(--hud-pad-x2)));
      padding: 8px;
      border-radius: 6px;
      background: rgba(10, 12, 15, 0.68);
      border: 1px solid rgba(255, 255, 255, 0.10);
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.26);
    }
    .camera-view-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: #dce4eb;
      font-size: 12px;
      font-weight: 700;
    }
    .camera-view-tabs {
      display: flex;
      gap: 4px;
    }
    .camera-view-tabs button {
      min-height: 24px;
      padding: 2px 7px;
      border-radius: 6px;
      color: #aeb8c3;
      font-size: 11px;
    }
    .camera-view-tabs button.active {
      color: #e8edf2;
    }
    #cameraViewImage {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.32);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    #cameraViewCaption {
      min-height: 14px;
      color: #8fa0ad;
      font-size: 11px;
      line-height: 1.25;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .plot-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: #dce4eb;
      font-size: 12px;
      font-weight: 700;
    }
    .plot-title strong {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #ffb000;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }
    .plots-panel canvas {
      display: block;
      width: 100%;
      height: 116px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.04);
    }
    .swatches {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: #aeb8c3;
      font-size: 12px;
    }
    .swatches span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .play-controls {
      display: flex;
      gap: 6px;
    }
    .icon-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      min-width: 34px;
      padding: 0;
      font-size: 15px;
      line-height: 1;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }
    @media (max-width: 680px) {
      :root {
        --hud-pad: 12px;
        --hud-pad-x2: 24px;
      }
      .panel {
        left: var(--hud-pad);
        right: var(--hud-pad);
        bottom: var(--hud-pad);
        flex-wrap: wrap;
        gap: 7px;
      }
      .scene-label {
        max-width: none;
        flex: 1 1 100%;
      }
      .plots-panel {
        top: calc(var(--topbar-h) + 268px);
        left: var(--hud-pad);
        right: var(--hud-pad);
        width: auto;
        grid-template-columns: 1fr;
        max-height: min(38vh, 330px);
        overflow: auto;
      }
      .camera-view {
        left: var(--hud-pad);
        top: calc(var(--topbar-h) + var(--hud-pad));
        width: min(330px, calc(100vw - var(--hud-pad-x2)));
      }
      .swatches {
        display: none;
      }
      .path-range {
        flex-basis: 100%;
        max-width: none;
      }
      .plots-panel canvas {
        height: 72px;
      }
    }
  </style>
  <script type="importmap">
    {
      "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
      }
    }
  </script>
</head>
<body>
  <header class="topbar">
    <div class="logo">FPV Video</div>
    <nav class="top-nav" aria-label="Workflow">
      <a href="/" target="_top">Annotate</a>
      <a class="active" href="/scenes/" target="_top">Scenes</a>
    </nav>
    <label class="scene-picker" for="sceneSelectTop">
      <select id="sceneSelectTop">
        <option value="">Loading scenes...</option>
      </select>
    </label>
    <div class="hdr-btns">
      <button id="saveSceneButton" type="button">Save</button>
      <span id="saveStateText"></span>
    </div>
  </header>
  <canvas id="viewer"></canvas>
  <section class="panel">
    <div class="scene-label">
      <strong id="title">VGGT Scene</strong>
      <span class="meta" id="meta"></span>
    </div>
    <button id="resetView" title="Reset view">Reset</button>
    <label class="strip-range" for="pointSize">Point
      <input id="pointSize" type="range" min="0.001" max="0.018" step="0.001" value="0.005" />
      <span id="pointSizeValue">0.005</span>
    </label>
    <label class="strip-range path-range" for="pathProgress">Path
      <input id="pathProgress" type="range" min="0" max="120" step="1" value="0" />
      <span id="frameValue">1</span>
    </label>
    <div class="play-controls" aria-label="Playback">
      <button id="playButton" class="icon-button" type="button" title="Play" aria-label="Play">&#9654;</button>
      <button id="repeatButton" class="icon-button active" type="button" title="Repeat" aria-label="Repeat">&#8635;</button>
    </div>
    <div class="toggles">
      <label class="toggle"><input id="showFrustums" type="checkbox" checked /> Frustums</label>
      <label class="toggle"><input id="showPath" type="checkbox" checked /> Path</label>
      <label class="toggle"><input id="showAxes" type="checkbox" checked /> Axes</label>
      <label class="toggle"><input id="showPoints" type="checkbox" checked /> Points</label>
    </div>
    <div class="scale-control">
      <label for="scaleInput">Scale</label>
      <input id="scaleInput" type="number" min="0.0001" step="0.1" value="117.6" />
      <span>m/unit</span>
      <button id="measureButton" type="button">Measure</button>
    </div>
    <div class="swatches">
      <span><i class="dot" style="background:#36e4ff"></i>camera path</span>
      <span><i class="dot" style="background:#ffb000"></i>current camera</span>
      <span><i class="dot" style="background:#ff4d6d"></i>attack end</span>
      <span><i class="dot" style="background:#4b6472"></i>ground grid</span>
    </div>
  </section>
  <section class="camera-view" id="cameraViewPanel">
    <div class="camera-view-header">
      <span>Frame</span>
      <div class="camera-view-tabs" aria-label="Camera view mode">
        <button id="viewActual" type="button" data-view-mode="actual">Actual</button>
        <button id="viewRender" type="button" data-view-mode="render">Render</button>
        <button id="viewOverlay" type="button" data-view-mode="overlay" class="active">Overlay</button>
      </div>
    </div>
    <img id="cameraViewImage" alt="Camera view for selected pose" />
    <div id="cameraViewCaption">No camera-view render available.</div>
  </section>
  <section class="plots-panel">
    <div class="plot-title">Speed vs. Time <strong id="speedReadout">--</strong></div>
    <canvas id="speedPlot"></canvas>
    <div class="plot-title">Height Above Ground <strong id="heightReadout">--</strong></div>
    <canvas id="heightPlot"></canvas>
    <div class="plot-title">Top View Path <strong id="pathReadout">--</strong></div>
    <canvas id="topPlot"></canvas>
  </section>
  <dialog class="scale-dialog" id="scaleDialog">
    <form class="scale-form" method="dialog">
      <h2>Measure Scale</h2>
      <p id="scaleMeasuredText">Select two points to measure a VGGT distance.</p>
      <label>
        Real length in meters
        <input id="realLengthM" type="number" min="0" step="0.01" placeholder="8.2" />
      </label>
      <p id="scaleResultText">Scale: -- m/unit</p>
      <div class="scale-dialog-actions">
        <button id="applyScale" type="button">Apply Scale</button>
        <button value="close">Close</button>
      </div>
    </form>
  </dialog>
  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";

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
    let currentCameraMarker;
    let currentFrustum;
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
    let repeatPlayback = true;
    const clock = new THREE.Clock();
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let defaultScaleMPerUnit = 117.6;
    const scaleStorageKey = `vggt-viewer-scale-v3:${location.pathname}`;

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
        const response = await fetch(meta.scene_state_url, { cache: "no-store" });
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
        const response = await fetch(meta.scene_state_url, {
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
      metaEl.textContent = `${meta.point_count.toLocaleString()} points | ${meta.path.length} poses${gridText}`;
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
        const response = await fetch("/api/scenes", { cache: "no-store" });
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
      playButton.innerHTML = isPlaying ? "&#10074;&#10074;" : "&#9654;";
      playButton.title = isPlaying ? "Pause" : "Play";
      playButton.setAttribute("aria-label", playButton.title);
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

    function buildGroundGrid() {
      const grid = meta.ground_grid;
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
      const grid = meta.ground_grid;
      if (!grid) return 0;
      const normal = vec3(grid.normal);
      return normal.dot(position) + grid.d;
    }

    function drawTopView(scale) {
      const { ctx, width, height } = canvasContext(topPlot);
      drawPlotFrame(ctx, width, height, "top view", "m", "m");
      const grid = meta.ground_grid;
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
      cameraViewImage.src = src;
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
      const box = new THREE.Box3();
      if (pointsObject) box.setFromObject(pointsObject);
      if (meta?.path?.length) {
        for (const frame of meta.path) box.expandByPoint(vec3(frame.position));
      }
      const grid = meta?.ground_grid;
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

    function updateCurrentCamera(index) {
      frameIndex = Math.max(0, Math.min(meta.path.length - 1, index));
      const frame = meta.path[frameIndex];
      const position = vec3(frame.position);
      currentCameraMarker.position.copy(position);
      if (currentFrustum) currentFrustum.removeFromParent();
      currentFrustum = createFrustum(frame, 0xffb000, 0.08, 1);
      pathGroup.add(currentFrustum);
      progress.value = String(frameIndex);
      frameValue.textContent = String(frame.frame);
      drawPlots();
    }

    function buildAxes(radius) {
      axesGroup.add(new THREE.AxesHelper(radius * 0.36));
      buildGroundGrid();
    }

    function fitView() {
      const box = currentSceneBox();
      if (box.isEmpty()) return;
      const sphere = new THREE.Sphere();
      box.getBoundingSphere(sphere);
      const center = sphere.center;
      const radius = Math.max(0.2, sphere.radius);
      const viewDirection = new THREE.Vector3(0.88, 0.58, 1.38).normalize();
      const distance = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov) / 2)) * 1.32;
      camera.position.copy(center).add(viewDirection.multiplyScalar(distance));
      camera.near = Math.max(0.0005, radius / 3000);
      camera.far = distance + radius * 8;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.minDistance = radius * 0.08;
      controls.maxDistance = distance + radius * 6;
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
      meta = await (await fetch("scene_meta.json")).json();
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
      title.textContent = meta.title;
      updateMetaText();
      loadSceneOptions();
      progress.max = String(meta.path.length - 1);

      const [positions, colors8] = await Promise.all([
        loadBinary(meta.assets.positions, Float32Array),
        loadBinary(meta.assets.colors, Uint8Array)
      ]);
      const colors = new Float32Array(colors8.length);
      for (let i = 0; i < colors8.length; i += 1) colors[i] = colors8[i] / 255;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      pointsObject = new THREE.Points(geometry, pointMaterial);
      root.add(pointsObject);

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
        const step = Math.max(1, Math.round(clock.getDelta() * 16));
        let next = frameIndex + step;
        if (next >= meta.path.length) {
          if (repeatPlayback) next = 0;
          else {
            next = meta.path.length - 1;
            setPlaying(false);
          }
        }
        updateCurrentCamera(next);
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
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path, help="VGGT reconstruction dir with point_cloud.npz, vggt_scene.glb, relative_path.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="VGGT Attack Segment 3")
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--calibration-npz", type=Path, default=None)
    parser.add_argument("--calibration-summary", type=Path, default=None)
    parser.add_argument("--default-scale-m-per-unit", type=float, default=117.6)
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--state-url", default="")
    return parser.parse_args()


def transform_points(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return trimesh.transformations.transform_points(np.asarray(vertices), transform)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


class VggtGlbCameras:
    def __init__(self, glb_path: Path):
        self.scene = trimesh.load(glb_path)
        self.transforms: dict[str, np.ndarray] = {}
        for node in self.scene.graph.nodes:
            try:
                transform, geom_name = self.scene.graph[node]
            except Exception:
                continue
            if geom_name is not None:
                self.transforms[geom_name] = np.asarray(transform)

    def camera_basis(self, frame_number: int, flip_y: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        name = f"geometry_{frame_number}"
        geom = self.scene.geometry[name]
        transform = self.transforms.get(name, np.eye(4))
        vertices = transform_points(geom.vertices, transform)
        counts = np.bincount(geom.faces.reshape(-1), minlength=len(vertices))
        center = vertices[int(np.argmax(counts))]
        corners = vertices[[0, 2, 3, 4]]
        forward = unit(corners.mean(axis=0) - center)
        right = unit(((corners[0] + corners[3]) * 0.5) - ((corners[1] + corners[2]) * 0.5))
        down = unit(((corners[0] + corners[1]) * 0.5) - ((corners[2] + corners[3]) * 0.5))
        right = unit(right - np.dot(right, forward) * forward)
        down = unit(down - np.dot(down, forward) * forward - np.dot(down, right) * right)
        if flip_y:
            down = -down
        return center, right, down, forward


def relative_asset_url(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_dir).replace(os.sep, "/")


def copy_viewer_asset(out_dir: Path, target: Path) -> str:
    asset_dir = out_dir / "camera_view_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    dest = asset_dir / target.name
    if target.resolve() != dest.resolve():
        shutil.copy2(target, dest)
    return relative_asset_url(out_dir, dest)


def load_path(video_dir: Path, cameras: VggtGlbCameras, out_dir: Path) -> list[dict[str, object]]:
    with (video_dir / "relative_path.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    path: list[dict[str, object]] = []
    centers: list[np.ndarray] = []
    for row in rows:
        frame = int(row["frame_index"])
        center, right, down, forward = cameras.camera_basis(frame)
        centers.append(center)
        frame_file = row.get("file") or ""
        frame_path = video_dir / "frames" / frame_file
        frame_stem = Path(frame_file).stem
        actual_view = video_dir / "camera_views" / f"{frame_stem}_actual.jpg"
        render_view = video_dir / "camera_views" / f"{frame_stem}_vggt_render.jpg"
        overlay_view = video_dir / "camera_views" / f"{frame_stem}_overlay.jpg"
        image_assets: dict[str, str] = {}
        if frame_path.exists():
            frame_url = copy_viewer_asset(out_dir, frame_path)
            image_assets["frame_image"] = frame_url
            image_assets["actual_image"] = frame_url
        if actual_view.exists():
            image_assets["actual_image"] = copy_viewer_asset(out_dir, actual_view)
        if render_view.exists():
            image_assets["render_image"] = copy_viewer_asset(out_dir, render_view)
        if overlay_view.exists():
            image_assets["overlay_image"] = copy_viewer_asset(out_dir, overlay_view)
        path.append(
            {
                "frame": frame,
                "file": row.get("file"),
                "video_file": row.get("video_file"),
                "segment_id": row.get("segment_id"),
                "segment_index": int(row["segment_index"]) if row.get("segment_index") else None,
                "is_attack": str(row.get("is_attack", "")).lower() == "true",
                "video_time_s": float(row["video_time_s"]),
                "segment_time_s": float(row["segment_time_s"]),
                "sequence_time_s": float(row.get("sequence_time_s") or row["segment_time_s"]),
                "position": center.astype(float).tolist(),
                "right": right.astype(float).tolist(),
                "down": down.astype(float).tolist(),
                "forward": forward.astype(float).tolist(),
                **image_assets,
            }
        )
    end = centers[-1]
    for row, center in zip(path, centers):
        row["distance_to_end_units"] = float(np.linalg.norm(center - end))
    return path


def load_calibration(npz_path: Path | None, summary_path: Path | None) -> dict[str, object] | None:
    if npz_path is None or not npz_path.exists():
        return None
    data = np.load(npz_path)
    calibration: dict[str, object] = {}
    if "axis_endpoints" in data.files:
        calibration["axis_endpoints"] = data["axis_endpoints"].astype(float).tolist()
    if "corners" in data.files:
        calibration["corners"] = data["corners"].astype(float).tolist()
    if "dims" in data.files:
        calibration["dims_vggt_units"] = data["dims"].astype(float).tolist()
    if summary_path is not None and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        calibration.update(
            {
                "label": "manual object calibration",
                "known_length_m": summary.get("known_length_m"),
                "length_vggt_units": summary.get("length_vggt_units"),
                "scale_m_per_vggt_unit": summary.get("scale_m_per_vggt_unit"),
                "box_dims_m_if_length_known": summary.get("box_dims_m_if_length_known"),
            }
        )
    return calibration or None


def estimate_ground_grid(
    points: np.ndarray,
    colors: np.ndarray,
    calibration: dict[str, object] | None,
    path: list[dict[str, object]],
    default_scale_m_per_unit: float | None = None,
) -> dict[str, object] | None:
    cols = colors.astype(float)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    groundish = (
        (r > 85)
        & (g > 70)
        & (b < 190)
        & (((r + g) / (b + 1)) > 1.7)
        & ~((g > r * 1.12) & (g > b * 1.35))
    )
    candidates = points[groundish].astype(np.float64)
    if len(candidates) < 2000:
        return None

    rng = np.random.default_rng(42)
    if len(candidates) > 80_000:
        candidates = candidates[rng.choice(len(candidates), 80_000, replace=False)]
    lo = np.percentile(candidates, 1, axis=0)
    hi = np.percentile(candidates, 99, axis=0)
    candidates = candidates[((candidates >= lo) & (candidates <= hi)).all(axis=1)]
    if len(candidates) < 2000:
        return None

    y_axis = np.array([0.0, 1.0, 0.0])
    threshold = 0.012
    subset = candidates[rng.choice(len(candidates), min(14_000, len(candidates)), replace=False)]
    best_count = -1
    best_model: tuple[np.ndarray, float] | None = None
    sample_idx = rng.choice(len(candidates), size=(4000, 3), replace=True)
    for tri in candidates[sample_idx]:
        a, b_point, c = tri
        normal = np.cross(b_point - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        if np.dot(normal, y_axis) < 0:
            normal = -normal
        if np.dot(normal, y_axis) < 0.28:
            continue
        d = -float(np.dot(normal, a))
        count = int((np.abs(subset @ normal + d) < threshold).sum())
        if count > best_count:
            best_count = count
            best_model = (normal, d)
    if best_model is None:
        return None

    normal, d = best_model
    for _ in range(3):
        distances = np.abs(candidates @ normal + d)
        inliers = candidates[distances < threshold]
        if len(inliers) < 1000:
            return None
        centroid = inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = vh[-1]
        if np.dot(normal, y_axis) < 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))

    distances = np.abs(candidates @ normal + d)
    inliers = candidates[distances < threshold]
    centroid = np.median(inliers, axis=0)
    origin = centroid - (float(np.dot(normal, centroid)) + d) * normal
    _, _, vh = np.linalg.svd(inliers - inliers.mean(axis=0), full_matrices=False)
    u = vh[0] - float(np.dot(vh[0], normal)) * normal
    u = unit(u)
    v = unit(np.cross(normal, u))

    scale = float(default_scale_m_per_unit or 0.0)
    if scale <= 0 and calibration:
        scale = float(calibration.get("scale_m_per_vggt_unit", 0.0))
    if scale > 0:
        minor_step_m = 2.0
        major_step_m = 8.0
        minor_step_units = minor_step_m / scale
        major_step_units = major_step_m / scale
    else:
        minor_step_units = 0.1
        major_step_units = minor_step_units * 4.0
        minor_step_m = None
        major_step_m = None

    if path:
        path_points = np.asarray([row["position"] for row in path], dtype=np.float64)
        start_end_midpoint = (path_points[0] + path_points[-1]) * 0.5
        origin = start_end_midpoint - (float(np.dot(normal, start_end_midpoint)) + d) * normal
        projected_path = np.column_stack([(path_points - origin) @ u, (path_points - origin) @ v])
        span_units = projected_path.max(axis=0) - projected_path.min(axis=0)
        size_units = float(np.ceil((max(span_units) + major_step_units * 2.0) / major_step_units) * major_step_units)
        size_units = max(size_units, float(major_step_units * 5.0))
    else:
        projected = np.column_stack([(points - origin) @ u, (points - origin) @ v])
        span_units = np.percentile(projected, 99, axis=0) - np.percentile(projected, 1, axis=0)
        size_units = float(np.ceil((max(span_units) * 1.25) / major_step_units) * major_step_units)
    size_m = float(size_units * scale) if scale > 0 else None

    return {
        "normal": normal.astype(float).tolist(),
        "d": float(d),
        "origin": origin.astype(float).tolist(),
        "u": u.astype(float).tolist(),
        "v": v.astype(float).tolist(),
        "inlier_count": int(len(inliers)),
        "candidate_count": int(len(candidates)),
        "threshold_units": threshold,
        "size_units": float(size_units),
        "fixed_size_units": float(size_units),
        "minor_step_units": float(minor_step_units),
        "major_step_units": float(major_step_units),
        "size_m": size_m,
        "minor_step_m": minor_step_m,
        "major_step_m": major_step_m,
        "path_span_units": span_units.astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cloud = np.load(args.video_dir / "point_cloud.npz")
    points = cloud["pts"].astype(np.float32)
    colors = cloud["cols"].astype(np.uint8)
    if len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    positions_path = args.out_dir / "points_positions.bin"
    colors_path = args.out_dir / "points_colors.bin"
    points.astype("<f4").tofile(positions_path)
    colors.tofile(colors_path)

    cameras = VggtGlbCameras(args.video_dir / "vggt_scene.glb")
    path = load_path(args.video_dir, cameras, args.out_dir)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    calibration = load_calibration(args.calibration_npz, args.calibration_summary)
    ground_grid = estimate_ground_grid(points, colors, calibration, path, args.default_scale_m_per_unit)

    meta = {
        "title": args.title,
        "source_label": args.video_dir.name,
        "video_dir": str(args.video_dir),
        "scene_id": args.scene_id,
        "scene_state_url": args.state_url or "",
        "default_scale_m_per_unit": args.default_scale_m_per_unit,
        "point_count": int(len(points)),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "assets": {
            "positions": positions_path.name,
            "colors": colors_path.name,
        },
        "path": path,
        "calibration": calibration,
        "ground_grid": ground_grid,
    }
    (args.out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out_dir / "index.html").write_text(HTML)
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    main()
