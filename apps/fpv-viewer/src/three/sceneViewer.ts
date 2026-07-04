// Lean, read-only Three.js scene viewer.
//
// Renders a reconstructed scene from the repo's viewer data format:
//   <sceneBase>/<scenePath>/viewer/scene_meta.json
//   + points_positions.bin (Float32 xyz) / points_colors.bin (Uint8 rgb)
// Applies the stored ground-alignment quaternion, draws a ground grid centered
// on the point cloud, and shows the flight path. No editing, saving,
// measuring or scene switching — those live in the full tool.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type SceneMeta = {
  point_count: number;
  path: { position: number[]; video_time_s?: number }[];
  assets: { positions: string; colors: string };
  ground_grid?: {
    origin: number[];
    u: number[];
    v: number[];
    normal: number[];
    fitted_normal?: number[];
    size_units: number;
    minor_step_units: number;
    major_step_units: number;
  } | null;
  scene_alignment_quaternion?: number[] | null;
  bbox_min?: number[];
  bbox_max?: number[];
};

const vec3 = (v: number[]) => new THREE.Vector3(v[0], v[1], v[2]);

export class ReadOnlySceneViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private root = new THREE.Group();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private pointsMaterial = new THREE.PointsMaterial({ size: 0.004, vertexColors: true });
  private points: THREE.Points | null = null;
  private disposed = false;
  private animationHandle = 0;

  constructor(private holder: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x050607);
    holder.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.001, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(this.root);
    this.resize();
    window.addEventListener("resize", this.resize);
    const loop = () => {
      if (this.disposed) return;
      this.animationHandle = requestAnimationFrame(loop);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

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
    this.points = new THREE.Points(geometry, this.pointsMaterial);
    this.root.add(this.points);

    this.buildPath(meta);
    this.buildGrid(meta, positions);
    this.fitView(meta, positions);
    return { pointCount: positions.length / 3, frames: meta.path?.length ?? 0 };
  }

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
    const line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0x36e4ff, transparent: true, opacity: 0.85 }),
    );
    this.root.add(line);
    const mkSphere = (p: THREE.Vector3, r: number, color: number) => {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(r, 16, 16),
        new THREE.MeshBasicMaterial({ color }),
      );
      s.position.copy(p);
      this.root.add(s);
    };
    mkSphere(pts[0], 0.012, 0x36e4ff);
    mkSphere(pts[pts.length - 1], 0.017, 0xff4d6d);
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
      this.root.add(new THREE.Line(a, material(isMajor)), new THREE.Line(b, material(isMajor)));
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

  setPointScale(mult: number) {
    // mult is relative to the auto-computed base size
    const base = this.controls.maxDistance / 12 / 260;
    this.pointsMaterial.size = base * mult;
  }

  resetView() {
    // Re-fit is cheap to approximate: reuse controls target and zoom out.
    this.controls.reset();
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
    window.removeEventListener("resize", this.resize);
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
