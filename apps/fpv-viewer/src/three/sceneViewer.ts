// Lean, read-only Three.js scene viewer with playback.
//
// Renders a reconstructed scene from the repo's viewer data format:
//   <sceneBase>/<scenePath>/viewer/scene_meta.json
//   + points_positions.bin (Float32 xyz) / points_colors.bin (Uint8 rgb)
// Applies the stored ground-alignment quaternion, draws a ground grid centered
// on the point cloud and the flight path, and animates a camera marker along
// the path (setTime, driven by the source video's clock). No editing, saving,
// measuring or scene switching — those live in the full tool.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type PathFrame = {
  position: number[];
  right?: number[];
  down?: number[];
  forward?: number[];
  video_time_s?: number;
  sequence_time_s?: number;
  distance_to_end_units?: number;
  frame_image?: string;
  actual_image?: string;
  render_image?: string;
  overlay_image?: string;
};

type SceneMeta = {
  point_count: number;
  path: PathFrame[];
  assets: { positions: string; colors: string };
  default_scale_m_per_unit?: number;
  calibration?: { scale_m_per_vggt_unit?: number } | null;
  ground_grid?: {
    origin: number[];
    u: number[];
    v: number[];
    normal: number[];
    fitted_normal?: number[];
    d: number;
    size_units: number;
    minor_step_units: number;
    major_step_units: number;
  } | null;
  scene_alignment_quaternion?: number[] | null;
  bbox_min?: number[];
  bbox_max?: number[];
};

export type TimelinePoint = {
  t: number; // flight (sequence) time in seconds — continuous, pauses removed
  heightM: number | null; // height above the fitted ground plane (m)
  speedRawMs: number | null; // per-frame speed (m/s)
  speedMs: number | null; // smoothed speed (m/s)
  distM: number | null; // distance to the strike point (m)
};

export type SceneTimeline = {
  t0: number;
  t1: number;
  avgSpeedMs: number | null;
  points: TimelinePoint[];
};

export type SceneFrame = {
  t: number; // flight (sequence) time
  actual: string | null;
  render: string | null;
  overlay: string | null;
};

const vec3 = (v: number[]) => new THREE.Vector3(v[0], v[1], v[2]);

export class ReadOnlySceneViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private root = new THREE.Group();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private pointsMaterial = new THREE.PointsMaterial({ size: 0.004, vertexColors: true });
  private pathGroup = new THREE.Group();
  private gridGroup = new THREE.Group();
  private markerGroup = new THREE.Group();
  private frustaGroup = new THREE.Group();
  private measureGroup = new THREE.Group();
  private marker: THREE.Mesh | null = null;
  private frustum: THREE.LineSegments | null = null;
  private pointsObject: THREE.Points | null = null;
  private meta: SceneMeta | null = null;
  private viewerBase: string | null = null;
  private radius = 1;
  private disposed = false;
  private animationHandle = 0;
  private measureMode = false;
  private measurePoints: THREE.Vector3[] = [];
  private onMeasure: ((meters: number | null, units: number | null) => void) | null = null;
  private raycaster = new THREE.Raycaster();
  private pointerNdc = new THREE.Vector2();

  constructor(private holder: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x050607);
    holder.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.001, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.root.add(this.pathGroup, this.gridGroup, this.markerGroup, this.frustaGroup, this.measureGroup);
    this.scene.add(this.root);
    this.renderer.domElement.addEventListener("pointerdown", this.pickPoint);
    this.resize();
    // The holder resizes with the responsive layout, not only the window.
    this.resizeObserver = new ResizeObserver(this.resize);
    this.resizeObserver.observe(holder);
    const loop = () => {
      if (this.disposed) return;
      this.animationHandle = requestAnimationFrame(loop);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  private resizeObserver: ResizeObserver;

  async load(viewerBase: string): Promise<{ pointCount: number; frames: number }> {
    const meta: SceneMeta = await fetch(`${viewerBase}/scene_meta.json`).then((r) => {
      if (!r.ok) throw new Error(`scene_meta.json: HTTP ${r.status}`);
      return r.json();
    });
    const [positionsBuf, colorsBuf] = await Promise.all([
      fetch(`${viewerBase}/${meta.assets.positions}`).then((r) => {
        if (!r.ok) throw new Error(`positions: HTTP ${r.status}`);
        return r.arrayBuffer();
      }),
      fetch(`${viewerBase}/${meta.assets.colors}`).then((r) => {
        if (!r.ok) throw new Error(`colors: HTTP ${r.status}`);
        return r.arrayBuffer();
      }),
    ]);
    if (this.disposed) return { pointCount: 0, frames: 0 };
    this.meta = meta;
    this.viewerBase = viewerBase;

    const positions = new Float32Array(positionsBuf);
    const colors8 = new Uint8Array(colorsBuf);
    const colors = new Float32Array(colors8.length);
    for (let i = 0; i < colors8.length; i += 1) colors[i] = colors8[i] / 255;

    // Ground alignment: rotate the whole root so the fitted ground is horizontal.
    const q = this.alignmentQuaternion(meta);
    if (q) this.root.quaternion.copy(q);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.pointsObject = new THREE.Points(geometry, this.pointsMaterial);
    this.root.add(this.pointsObject);

    this.fitView(meta, positions); // sets this.radius, used to size markers
    this.buildPath(meta);
    this.buildGrid(meta, positions);
    this.buildMarker(meta);
    this.buildFrusta(meta);
    const t = this.timeline();
    if (t) this.setTime(t.t0);
    return { pointCount: positions.length / 3, frames: meta.path?.length ?? 0 };
  }

  // -- scale ---------------------------------------------------------------

  private scaleMPerUnit(): number {
    const cal = this.meta?.calibration?.scale_m_per_vggt_unit;
    if (cal && cal > 0) return cal;
    const def = this.meta?.default_scale_m_per_unit;
    return def && def > 0 ? def : 117.6;
  }

  // -- playback ------------------------------------------------------------

  /** Height/speed/distance series against flight (sequence) time. */
  timeline(): SceneTimeline | null {
    const meta = this.meta;
    const path = meta?.path ?? [];
    if (!meta || path.length < 2) return null;
    const scale = this.scaleMPerUnit();
    const grid = meta.ground_grid;
    const normal = grid ? vec3(grid.fitted_normal ?? grid.normal).normalize() : null;
    const points: TimelinePoint[] = [];
    for (let i = 0; i < path.length; i += 1) {
      const f = path[i];
      const p = vec3(f.position);
      const t = f.sequence_time_s ?? i;
      const heightM =
        normal && grid ? Math.max(0, (normal.dot(p) + grid.d) * scale) : null;
      let speedRawMs: number | null = null;
      if (i > 0) {
        const prev = path[i - 1];
        const dt = (f.sequence_time_s ?? 0) - (prev.sequence_time_s ?? 0);
        if (dt > 1e-4) {
          speedRawMs = (p.distanceTo(vec3(prev.position)) * scale) / dt;
        }
      }
      const distM =
        typeof f.distance_to_end_units === "number" ? f.distance_to_end_units * scale : null;
      points.push({ t, heightM, speedRawMs, speedMs: speedRawMs, distM });
    }
    // Smooth speed: median-of-3 (kills sampling spikes) then a short moving
    // average — the amber "smooth" curve; raw stays as the teal backdrop.
    const median3: (number | null)[] = points.map((p, i) => {
      if (i === 0 || i === points.length - 1) return p.speedRawMs;
      const trio = [points[i - 1].speedRawMs, p.speedRawMs, points[i + 1].speedRawMs].filter(
        (v): v is number => v !== null,
      );
      return trio.length === 3 ? trio.slice().sort((a, b) => a - b)[1] : p.speedRawMs;
    });
    for (let i = 0; i < points.length; i += 1) {
      const window: number[] = [];
      for (let j = Math.max(0, i - 2); j <= Math.min(points.length - 1, i + 2); j += 1) {
        const v = median3[j];
        if (v !== null) window.push(v);
      }
      points[i].speedMs = window.length
        ? window.reduce((a, b) => a + b, 0) / window.length
        : null;
    }
    // Time-weighted average speed = total path length / total flight time.
    let dist = 0;
    for (let i = 1; i < path.length; i += 1) {
      dist += vec3(path[i].position).distanceTo(vec3(path[i - 1].position));
    }
    const totalT = (points[points.length - 1].t ?? 0) - (points[0].t ?? 0);
    const avgSpeedMs = totalT > 1e-4 ? (dist * scale) / totalT : null;
    return { t0: points[0].t, t1: points[points.length - 1].t, avgSpeedMs, points };
  }

  /** Per-frame VGGT camera-view images (actual / render / overlay). */
  frames(): SceneFrame[] {
    const path = this.meta?.path ?? [];
    const base = this.viewerBase;
    if (!base) return [];
    const url = (rel?: string) => (rel ? `${base}/${rel}` : null);
    return path.map((f, i) => ({
      t: f.sequence_time_s ?? i,
      actual: url(f.actual_image ?? f.frame_image),
      render: url(f.render_image),
      overlay: url(f.overlay_image),
    }));
  }

  /** Move the camera marker to flight (sequence) time t (interpolated). */
  setTime(t: number) {
    const path = this.meta?.path ?? [];
    if (path.length === 0 || !this.marker) return;
    const times = path.map((f, i) => f.sequence_time_s ?? i);
    let i = 0;
    while (i < times.length - 1 && times[i + 1] <= t) i += 1;
    const a = path[i];
    const b = path[Math.min(i + 1, path.length - 1)];
    const ta = times[i];
    const tb = times[Math.min(i + 1, path.length - 1)];
    const gap = tb - ta;
    // Sequence time is continuous (pauses removed), so always interpolate.
    const lerp = gap > 1e-6 ? Math.min(1, Math.max(0, (t - ta) / gap)) : 0;
    const pos = vec3(a.position).lerp(vec3(b.position), lerp);
    this.marker.position.copy(pos);
    if (this.frustum) {
      this.frustum.position.copy(pos);
      this.orientByFrame(this.frustum, lerp < 0.5 ? a : b);
    }
  }

  // -- visibility toggles ---------------------------------------------------

  setPathVisible(v: boolean) {
    this.pathGroup.visible = v;
  }

  setGridVisible(v: boolean) {
    this.gridGroup.visible = v;
  }

  setPointsVisible(v: boolean) {
    if (this.pointsObject) this.pointsObject.visible = v;
  }

  setFrustaVisible(v: boolean) {
    this.frustaGroup.visible = v;
  }

  // -- measure tool ----------------------------------------------------------

  /** Toggle two-point measuring; reports distance via the callback. */
  setMeasureMode(enabled: boolean, onMeasure?: (meters: number | null, units: number | null) => void) {
    this.measureMode = enabled;
    if (onMeasure) this.onMeasure = onMeasure;
    this.clearMeasurement();
  }

  clearMeasurement() {
    this.measurePoints = [];
    this.measureGroup.clear();
    this.onMeasure?.(null, null);
  }

  private pickPoint = (event: PointerEvent) => {
    if (!this.measureMode || !this.pointsObject) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    this.raycaster.params.Points.threshold = Math.max(this.pointsMaterial.size * 2.5, 0.012);
    const hits = this.raycaster.intersectObject(this.pointsObject, false);
    // Pick the hit closest to the ray (not the closest to the camera), so the
    // selected point lands under the cursor instead of with a lateral offset.
    let best: THREE.Intersection | null = null;
    for (const hit of hits) {
      if (hit.index == null || hit.distanceToRay == null) continue;
      if (best === null || hit.distanceToRay < (best.distanceToRay ?? Infinity)) best = hit;
    }
    if (best === null || best.index == null) return;
    const local = new THREE.Vector3().fromBufferAttribute(
      this.pointsObject.geometry.getAttribute("position") as THREE.BufferAttribute,
      best.index,
    );
    if (this.measurePoints.length >= 2) this.clearMeasurement();
    this.measurePoints.push(local.clone());
    const r = Math.max(this.radius / 220, 0.002);
    const markerMesh = new THREE.Mesh(
      new THREE.SphereGeometry(r, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    markerMesh.position.copy(local);
    this.measureGroup.add(markerMesh);
    if (this.measurePoints.length === 2) {
      const [a, b] = this.measurePoints;
      this.measureGroup.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([a, b]),
          new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }),
        ),
      );
      const units = a.distanceTo(b);
      this.onMeasure?.(units * this.scaleMPerUnit(), units);
    }
  };

  // -- construction ----------------------------------------------------------

  private alignmentQuaternion(meta: SceneMeta): THREE.Quaternion | null {
    const stored = meta.scene_alignment_quaternion;
    if (Array.isArray(stored) && stored.length === 4) {
      return new THREE.Quaternion(stored[0], stored[1], stored[2], stored[3]);
    }
    const grid = meta.ground_grid;
    if (!grid) return null;
    const n = vec3(grid.fitted_normal ?? grid.normal).normalize();
    const qAlign = new THREE.Quaternion().setFromUnitVectors(n, new THREE.Vector3(0, 1, 0));
    const uRot = vec3(grid.u).applyQuaternion(qAlign).normalize();
    const uXZ = new THREE.Vector3(uRot.x, 0, uRot.z);
    if (uXZ.lengthSq() < 1e-8) uXZ.set(1, 0, 0);
    else uXZ.normalize();
    const qTwist = new THREE.Quaternion().setFromUnitVectors(uXZ, new THREE.Vector3(1, 0, 0));
    return qTwist.multiply(qAlign);
  }

  private buildPath(meta: SceneMeta) {
    const path = meta.path ?? [];
    if (path.length < 2) return;
    const pts = path.map((f) => vec3(f.position));
    const geometry = new THREE.BufferGeometry().setFromPoints(pts);
    this.pathGroup.add(
      new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ color: 0x3291ff, transparent: true, opacity: 0.85 }),
      ),
    );
    const mkSphere = (p: THREE.Vector3, r: number, color: number) => {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(r, 16, 16),
        new THREE.MeshBasicMaterial({ color }),
      );
      s.position.copy(p);
      this.pathGroup.add(s);
    };
    mkSphere(pts[0], Math.max(this.radius / 110, 0.003), 0x3291ff);
    mkSphere(pts[pts.length - 1], Math.max(this.radius / 80, 0.004), 0xff4d6d);
  }

  // A small camera frustum wireframe (apex at origin, opening along -Z).
  private frustumWire(scaleMult: number, color: number, opacity: number): THREE.LineSegments {
    const w = (this.radius / 14) * scaleMult;
    const h = w * (368 / 720);
    const d = (this.radius / 9) * scaleMult;
    const corners = [
      new THREE.Vector3(-w, -h, -d),
      new THREE.Vector3(w, -h, -d),
      new THREE.Vector3(w, h, -d),
      new THREE.Vector3(-w, h, -d),
    ];
    const o = new THREE.Vector3(0, 0, 0);
    const verts: THREE.Vector3[] = [];
    for (let i = 0; i < 4; i += 1) {
      verts.push(o.clone(), corners[i].clone()); // rays
      verts.push(corners[i].clone(), corners[(i + 1) % 4].clone()); // rim
    }
    return new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(verts),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
    );
  }

  private orientByFrame(obj: THREE.Object3D, frame: PathFrame) {
    if (frame.right && frame.down && frame.forward) {
      const m = new THREE.Matrix4().makeBasis(
        vec3(frame.right).normalize(),
        vec3(frame.down).normalize().multiplyScalar(-1),
        vec3(frame.forward).normalize().multiplyScalar(-1),
      );
      obj.setRotationFromMatrix(m);
    }
  }

  private buildMarker(meta: SceneMeta) {
    if (!meta.path?.length) return;
    const r = Math.max(this.radius / 130, 0.004);
    this.marker = new THREE.Mesh(
      new THREE.SphereGeometry(r, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffb000 }),
    );
    this.markerGroup.add(this.marker);
    this.frustum = this.frustumWire(1, 0xffb000, 0.9);
    this.markerGroup.add(this.frustum);
  }

  // Static frusta along the path (every ~10th frame, final frame highlighted),
  // like the full tool's camera-frusta layer. Hidden by default.
  private buildFrusta(meta: SceneMeta) {
    const path = meta.path ?? [];
    for (let i = 0; i < path.length; i += 10) {
      const f = this.frustumWire(0.55, 0x3291ff, 0.4);
      f.position.copy(vec3(path[i].position));
      this.orientByFrame(f, path[i]);
      this.frustaGroup.add(f);
    }
    if (path.length) {
      const last = path[path.length - 1];
      const f = this.frustumWire(0.8, 0xff4d6d, 0.8);
      f.position.copy(vec3(last.position));
      this.orientByFrame(f, last);
      this.frustaGroup.add(f);
    }
    this.frustaGroup.visible = false;
  }

  // Grid centered under the point cloud (robust percentile center), matching
  // the full viewer's behaviour.
  private buildGrid(meta: SceneMeta, positions: Float32Array) {
    const grid = meta.ground_grid;
    if (!grid) return;
    const origin = vec3(grid.origin);
    const u = vec3(grid.u).normalize();
    const v = vec3(grid.v).normalize();
    const count = positions.length / 3;
    const stride = Math.max(1, Math.floor(count / 40000));
    const us: number[] = [];
    const vs: number[] = [];
    const p = new THREE.Vector3();
    for (let i = 0; i < count; i += stride) {
      p.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).sub(origin);
      us.push(p.dot(u));
      vs.push(p.dot(v));
    }
    us.sort((a, b) => a - b);
    vs.sort((a, b) => a - b);
    const qtile = (arr: number[], t: number) =>
      arr[Math.min(arr.length - 1, Math.max(0, Math.round(t * (arr.length - 1))))];
    const uc = (qtile(us, 0.01) + qtile(us, 0.99)) / 2;
    const vc = (qtile(vs, 0.01) + qtile(vs, 0.99)) / 2;
    const center = origin.clone().add(u.clone().multiplyScalar(uc)).add(v.clone().multiplyScalar(vc));
    const span = Math.max(qtile(us, 0.99) - qtile(us, 0.01), qtile(vs, 0.99) - qtile(vs, 0.01));
    const major = grid.major_step_units || 1;
    const half = Math.max(Math.ceil((span * 1.25) / major) * major, major * 5) / 2;
    const step = grid.minor_step_units || major / 4;
    const majorEvery = Math.max(1, Math.round(major / step));
    const lineCount = Math.ceil(half / step);
    const material = (isMajor: boolean) =>
      new THREE.LineBasicMaterial({
        color: isMajor ? 0x4b6472 : 0x25313a,
        transparent: true,
        opacity: isMajor ? 0.85 : 0.45,
      });
    for (let i = -lineCount; i <= lineCount; i += 1) {
      const off = i * step;
      const isMajor = i % majorEvery === 0;
      const a = new THREE.BufferGeometry().setFromPoints([
        center.clone().add(u.clone().multiplyScalar(-half)).add(v.clone().multiplyScalar(off)),
        center.clone().add(u.clone().multiplyScalar(half)).add(v.clone().multiplyScalar(off)),
      ]);
      const b = new THREE.BufferGeometry().setFromPoints([
        center.clone().add(v.clone().multiplyScalar(-half)).add(u.clone().multiplyScalar(off)),
        center.clone().add(v.clone().multiplyScalar(half)).add(u.clone().multiplyScalar(off)),
      ]);
      this.gridGroup.add(new THREE.Line(a, material(isMajor)), new THREE.Line(b, material(isMajor)));
    }
  }

  // Fit entirely in root-LOCAL coordinates, then map the eye/center to world
  // through the alignment quaternion (never rotate twice).
  private fitView(meta: SceneMeta, positions: Float32Array) {
    const box = new THREE.Box3();
    if (meta.bbox_min && meta.bbox_max) {
      box.set(vec3(meta.bbox_min), vec3(meta.bbox_max));
    } else {
      const p = new THREE.Vector3();
      for (let i = 0; i < positions.length; i += 3) {
        p.set(positions[i], positions[i + 1], positions[i + 2]);
        box.expandByPoint(p);
      }
    }
    for (const f of meta.path ?? []) box.expandByPoint(vec3(f.position));
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.05);
    this.radius = radius;
    const eyeLocal = center
      .clone()
      .add(new THREE.Vector3(radius * 0.9, radius * 0.75, radius * 0.9));
    const toWorld = (p: THREE.Vector3) => p.clone().applyQuaternion(this.root.quaternion);
    const worldCenter = toWorld(center);
    this.camera.position.copy(toWorld(eyeLocal));
    this.camera.near = Math.max(0.0005, radius / 3000);
    this.camera.far = radius * 30;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(worldCenter);
    this.controls.minDistance = radius * 0.05;
    this.controls.maxDistance = radius * 12;
    this.controls.update();
    this.pointsMaterial.size = radius / 260;
  }

  private resize = () => {
    const w = this.holder.clientWidth || 1;
    const h = this.holder.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  };

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.animationHandle);
    this.resizeObserver.disconnect();
    this.renderer.domElement.removeEventListener("pointerdown", this.pickPoint);
    this.controls.dispose();
    this.root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else if (mat) mat.dispose();
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
