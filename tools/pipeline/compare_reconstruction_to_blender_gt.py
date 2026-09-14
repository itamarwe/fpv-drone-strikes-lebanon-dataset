#!/usr/bin/env python3
"""Score a VGGT-Omega reconstruction of the synthetic flyover against Blender ground truth.

Inputs (scene directory produced by run_vggt_omega_direct_on_runpod.py):
  vggt_scene.glb                      VGGT cameras (per-frame frusta) and cloud
  point_cloud.npz                     pts (N,3) float32 in VGGT units, cols (N,3) uint8
  runpod_artifacts/predictions.npz    optional: depth (S,H,W[,1]), extrinsic (S,3,4) w2c, intrinsic (S,3,3)
  ground_truth/gt_cameras.json        Blender cameras, c2w_opencv, metres
  ground_truth/gt_surface_points.npz  surface sample of the Blender scene, metres
  ground_truth/gt_depth/NNNNN.npy     optional: per-frame Cycles depth (ray length, metres) at 800x600

Outputs comparison.json plus a few diagnostic PNGs in <scene>/gt_comparison/.

Scale: a Sim(3) fit on the 120 camera centres gives metres per VGGT unit and the
residual path-shape error. Distortion: (1) local Sim(3) scale over sliding windows
relative to the global scale, (2) nearest-surface distance of the aligned cloud to
the Blender surface, (3) per-pixel depth ratio against the fisheye ground truth
converted to camera-z. Everything is against exact synthetic truth, so the numbers
are absolute, unlike the published-VGGT agreement scores in the two-scene benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_3d_trajectory import umeyama, rigid_alignment  # noqa: E402
import trimesh  # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class VggtGlbCameras:
    """Per-frame camera frusta exported by the VGGT-Omega demo GLB (geometry_<frame>)."""

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

    def camera_basis(self, frame_number: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        name = f"geometry_{frame_number}"
        if name not in self.scene.geometry:
            raise KeyError(name)
        geom = self.scene.geometry[name]
        tf = self.transforms.get(name, np.eye(4))
        vertices = (tf[:3, :3] @ np.asarray(geom.vertices, dtype=np.float64).T).T + tf[:3, 3]
        counts = np.bincount(np.asarray(geom.faces).reshape(-1), minlength=len(vertices))
        center = vertices[int(np.argmax(counts))]
        corners = vertices[[0, 2, 3, 4]]
        forward = _unit(corners.mean(axis=0) - center)
        right = _unit(((corners[0] + corners[3]) * 0.5) - ((corners[1] + corners[2]) * 0.5))
        down = _unit(((corners[0] + corners[1]) * 0.5) - ((corners[2] + corners[3]) * 0.5))
        right = _unit(right - np.dot(right, forward) * forward)
        down = _unit(down - np.dot(down, forward) * forward - np.dot(down, right) * right)
        return center, right, down, forward


def read_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_benchmark_scene_viewers import read_ply
    return read_ply(path)


def load_method_outputs(method: str, run_dir: Path, frame_ids: list[int]):
    """Return (pred_c2w by GT frame id, points xyz, colours or None, source string). Profile index i maps to frame_ids[i]."""
    from fit_moge3_colmap_scale import qvec_to_rotation, read_images
    if method == "scal3r":
        rows = np.atleast_2d(np.loadtxt(run_dir / "mat.txt", dtype=np.float64))
        c2w = {frame_ids[i]: row.reshape(4, 4) for i, row in enumerate(rows) if i < len(frame_ids)}
        xyz, rgb = read_ply_xyz_rgb(run_dir / "points" / "whole.ply")
        return c2w, xyz, rgb, "Scal3R mat.txt (c2w) + points/whole.ply"
    if method == "glomap":
        model = run_dir / "sparse-txt"
        c2w = {}
        for image in read_images(model / "images.txt"):
            R = qvec_to_rotation(image.qvec); m = np.eye(4); m[:3, :3] = R.T; m[:3, 3] = -(R.T @ image.tvec)
            i = int(Path(image.name).stem.rsplit("_", 1)[1])
            if i < len(frame_ids):
                c2w[frame_ids[i]] = m
        xyz, rgb = [], []
        for line in (model / "points3D.txt").read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            q = line.split(); xyz.append([float(q[1]), float(q[2]), float(q[3])]); rgb.append([int(q[4]), int(q[5]), int(q[6])])
        return c2w, np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8), "GLOMAP sparse-txt images.txt (w2c) + points3D.txt"
    if method == "querysplat":
        payload = json.loads((run_dir / "predicted_input_cameras.json").read_text())
        c2w = {}
        for f in payload["frames"]:
            i = int(str(f["name"]).rsplit("_", 1)[1])
            if i < len(frame_ids):
                c2w[frame_ids[i]] = np.asarray(f["c2w"], dtype=np.float64)
        xyz, rgb = read_ply_xyz_rgb(run_dir / "vggt_depth_pointcloud.ply")
        return c2w, xyz, rgb, "QuerySplat predicted_input_cameras.json (c2w) + vggt_depth_pointcloud.ply"
    raise ValueError(method)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_dir", type=Path, help="scene directory holding ground_truth/ (and the VGGT-Omega outputs for --method vggt_omega)")
    p.add_argument("--method", choices=["vggt_omega", "scal3r", "glomap", "querysplat"], default="vggt_omega")
    p.add_argument("--run-dir", type=Path, help="benchmark attempt directory for scal3r / glomap / querysplat (feed_forward dir for querysplat)")
    p.add_argument("--label", default=None, help="output subdirectory name under gt_comparison/ (default: method)")
    p.add_argument("--claimed-scale", type=float, help="metres per reconstruction unit claimed by an automatic scale method (e.g. MoGe-3); scored against the Sim(3) truth")
    p.add_argument("--window", type=int, default=20, help="frames per local-scale window")
    p.add_argument("--cloud-sample", type=int, default=2_000_000)
    p.add_argument("--conf-quantile", type=float, default=0.0, help="drop VGGT points below this depth-conf quantile if available")
    return p.parse_args()


def summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0}
    return {"count": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def equisolid_camera_z_factor(res: tuple[int, int], f_mm: float, sensor_w_mm: float, sensor_h_mm: float) -> np.ndarray:
    """cos(theta) per pixel for an equisolid fisheye: converts ray length to camera-z."""
    w, h = res
    px_per_mm = w / sensor_w_mm
    xs = (np.arange(w) + 0.5 - w / 2) / px_per_mm
    ys = (np.arange(h) + 0.5 - h / 2) / px_per_mm
    xx, yy = np.meshgrid(xs, ys)
    r = np.sqrt(xx**2 + yy**2)
    theta = 2.0 * np.arcsin(np.clip(r / (2.0 * f_mm), -1.0, 1.0))
    return np.cos(theta)


def resize_nearest(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    oh, ow = out_hw
    ys = np.clip((np.arange(oh) + 0.5) * h / oh, 0, h - 1).astype(int)
    xs = np.clip((np.arange(ow) + 0.5) * w / ow, 0, w - 1).astype(int)
    return img[ys][:, xs]


def main() -> int:
    args = parse_args()
    sd = args.scene_dir
    gt_dir = sd / "ground_truth"
    label = args.label or args.method
    out_dir = sd / "gt_comparison" if args.method == "vggt_omega" and not args.label else sd / "gt_comparison" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"scene_dir": str(sd), "method": args.method, "run_dir": str(args.run_dir) if args.run_dir else None}

    gt = json.loads((gt_dir / "gt_cameras.json").read_text())
    gt_frames = gt["frames"]
    gt_c2w = {int(f["frame"]): np.asarray(f["c2w_opencv"], dtype=np.float64) for f in gt_frames}
    frame_ids = sorted(gt_c2w)

    # ---------------------------------------------------------------- predicted cameras
    pred_path = sd / "runpod_artifacts" / "predictions.npz"
    pred_c2w = {}
    camera_source = None
    method_pts = None
    z = None
    if args.method != "vggt_omega":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for this method")
        pred_c2w, method_xyz, method_rgb, camera_source = load_method_outputs(args.method, args.run_dir, frame_ids)
        method_pts = method_xyz
    elif pred_path.exists():
        z = np.load(pred_path, allow_pickle=True)
        if "extrinsic" in z.files:
            ext = np.asarray(z["extrinsic"], dtype=np.float64)  # (S,3,4) world-to-camera, OpenCV convention
            for i, fid in enumerate(frame_ids[: len(ext)]):
                w2c = np.eye(4); w2c[:3, :4] = ext[i]
                pred_c2w[fid] = np.linalg.inv(w2c)
            camera_source = "predictions.npz extrinsic (w2c, OpenCV)"
            report["vggt_intrinsics_first_frame"] = np.asarray(z["intrinsic"][0]).round(3).tolist() if "intrinsic" in z.files else None
    if not pred_c2w:
        cams = VggtGlbCameras(sd / "vggt_scene.glb")
        for fid in frame_ids:
            try:
                center, right, down, forward = cams.camera_basis(fid)
            except KeyError:
                continue
            m = np.eye(4); m[:3, 0] = right; m[:3, 1] = down; m[:3, 2] = forward; m[:3, 3] = center
            pred_c2w[fid] = m
        camera_source = "vggt_scene.glb frusta"
    report["camera_source"] = camera_source
    common = [f for f in frame_ids if f in pred_c2w]
    report["frames"] = {"ground_truth": len(frame_ids), "predicted": len(pred_c2w), "matched": len(common)}
    P = np.vstack([pred_c2w[f][:3, 3] for f in common])
    G = np.vstack([gt_c2w[f][:3, 3] for f in common])

    # ---------------------------------------------------------------- global Sim(3) => metric scale
    # Rotation from camera orientations (Procrustes on stacked rotations): well posed even on a
    # straight flight line, where centre-only Umeyama leaves the roll about the line free.
    # Scale from the RMS radius ratio of the centres (rotation independent); translation from the means.
    s_u, R_u, t_u = umeyama(P, G)
    M = sum(gt_c2w[f][:3, :3] @ pred_c2w[f][:3, :3].T for f in common)
    U_, _, Vt_ = np.linalg.svd(M); corr = np.eye(3); corr[2, 2] = np.sign(np.linalg.det(U_ @ Vt_))
    R = U_ @ corr @ Vt_
    Pc, Gc = P - P.mean(axis=0), G - G.mean(axis=0)
    s = float(np.sqrt(np.sum(Gc**2) / np.sum(Pc**2)))
    t = G.mean(axis=0) - s * (R @ P.mean(axis=0))
    P_al = (s * (R @ P.T)).T + t
    err = np.linalg.norm(P_al - G, axis=1)
    err_u = np.linalg.norm((s_u * (R_u @ P.T)).T + t_u - G, axis=1)
    report["alignment"] = {"rotation": "procrustes_on_camera_rotations", "scale": "rms_radius_ratio_of_centres",
                           "centres_only_umeyama": {"scale": float(s_u), "centre_rmse_m": float(np.sqrt(np.mean(err_u**2))),
                                                    "rotation_difference_deg": float(np.degrees(np.arccos(np.clip((np.trace(R_u.T @ R) - 1) / 2, -1, 1))))}}
    gt_path_len = float(np.sum(np.linalg.norm(np.diff(G, axis=0), axis=1)))
    pred_path_len = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    # Orientation agreement after alignment
    ang = []
    for f in common:
        Rp = R @ pred_c2w[f][:3, :3]
        Rg = gt_c2w[f][:3, :3]
        cosang = (np.trace(Rp.T @ Rg) - 1) / 2
        ang.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    loo = []
    for i in range(len(common)):
        m = np.arange(len(common)) != i
        loo.append(umeyama(P[m], G[m])[0])
    report["scale"] = {
        "metres_per_vggt_unit_sim3": float(s),
        "metres_per_vggt_unit_path_length_ratio": gt_path_len / pred_path_len if pred_path_len else None,
        "gt_path_length_m": gt_path_len,
        "predicted_path_length_units": pred_path_len,
        "leave_one_out_scale_cv": float(np.std(loo) / np.mean(loo)),
        "camera_centre_error_m_after_sim3": summary(err),
        "camera_centre_error_fraction_of_path_length": float(np.sqrt(np.mean(err**2)) / gt_path_len),
        "camera_orientation_error_deg": summary(np.asarray(ang)),
        "note": "Sim(3) on camera centres against exact synthetic poses; scale is the true metres-per-unit of the reconstruction.",
    }
    per_frame = [{"frame": int(f), "centre_error_m": float(e), "orientation_error_deg": float(a)} for f, e, a in zip(common, err, ang)]
    if args.claimed_scale:
        report["claimed_scale"] = {"claimed_metres_per_unit": args.claimed_scale, "true_metres_per_unit_sim3": float(s),
                                  "ratio_claimed_over_true": args.claimed_scale / s, "relative_error": args.claimed_scale / s - 1.0}

    # ---------------------------------------------------------------- local scale drift
    local = []
    w = args.window
    for start in range(0, len(common) - w + 1, max(1, w // 2)):
        idx = slice(start, start + w)
        sl, _, _ = umeyama(P[idx], G[idx])
        local.append({"frames": [int(common[start]), int(common[start + w - 1])], "scale_m_per_unit": float(sl), "relative_to_global": float(sl / s)})
    rel = np.array([l["relative_to_global"] for l in local])
    report["local_scale"] = {
        "window_frames": w,
        "windows": local,
        "relative_scale_summary": summary(rel),
        "relative_scale_range": [float(rel.min()), float(rel.max())] if rel.size else None,
        "note": "Local Sim(3) scale divided by the global scale; 1.0 everywhere means no scale drift along the flight.",
    }

    # ---------------------------------------------------------------- cloud vs surface
    cloud_source = "point_cloud.npz (GLB frame)"
    if method_pts is not None:
        pts = method_pts.astype(np.float64)
        cloud_source = camera_source
    elif camera_source and camera_source.startswith("predictions.npz") and "world_points_from_depth" in z.files:
        wp = np.asarray(z["world_points_from_depth"])  # (S,H,W,3) in the raw VGGT frame, same as extrinsic
        S_, H_, W_ = wp.shape[:3]
        conf = np.asarray(z["depth_conf"]).reshape(S_, H_, W_)
        keep = np.isfinite(wp).all(axis=-1) & (conf >= np.quantile(conf, 0.5))
        sky_dir = sd / "runpod_artifacts" / "sky_masks"
        sky_files = sorted(sky_dir.glob("*")) if sky_dir.exists() else []
        if len(sky_files) == S_:
            from PIL import Image
            for i, f in enumerate(sky_files):
                m = np.asarray(Image.open(f).convert("L"))
                keep[i] &= resize_nearest(m, (H_, W_)) >= 128  # VGGT-Omega sky masks are 255 on non-sky pixels
            cloud_source = "predictions.npz world_points_from_depth, conf >= median, sky masked"
        else:
            cloud_source = "predictions.npz world_points_from_depth, conf >= median"
        pts = wp[keep].astype(np.float64)
        report["cloud_points_kept_fraction"] = float(keep.mean())
    else:
        cloud = np.load(sd / "point_cloud.npz")
        pts = cloud["pts"].astype(np.float64)
    report["cloud_source"] = cloud_source
    if len(pts) > args.cloud_sample:
        keep = np.random.default_rng(3).choice(len(pts), size=args.cloud_sample, replace=False)
        pts = pts[keep]
    pts_al = (s * (R @ pts.T)).T + t
    gts = np.load(gt_dir / "gt_surface_points.npz")
    gxyz = gts["xyz"].astype(np.float64)
    tree = cKDTree(gxyz)
    d_sim3, _ = tree.query(pts_al, k=1, workers=-1)
    # Rigid refinement (ICP-lite, 5 iterations on inliers) to separate camera-based alignment error from shape error.
    pts_ref = pts_al.copy()
    for _ in range(5):
        d, nn = tree.query(pts_ref, k=1, workers=-1)
        inl = d < np.percentile(d, 70)
        Rr, tr = rigid_alignment(pts_ref[inl], gxyz[nn[inl]])
        pts_ref = (Rr @ pts_ref.T).T + tr
    d_ref, _ = tree.query(pts_ref, k=1, workers=-1)
    # Coverage: fraction of GT surface points (near the path) with a reconstructed point within 2 m.
    gsub = gxyz[np.random.default_rng(4).choice(len(gxyz), size=min(300_000, len(gxyz)), replace=False)]
    ptree = cKDTree(pts_ref)
    dcov, _ = ptree.query(gsub, k=1, workers=-1)
    report["cloud_vs_surface"] = {
        "cloud_points_scored": int(len(pts_al)),
        "gt_surface_points": int(len(gxyz)),
        "distance_to_surface_m_after_camera_sim3": summary(d_sim3),
        "distance_to_surface_m_after_rigid_refinement": summary(d_ref),
        "fraction_within_1m_after_refinement": float(np.mean(d_ref < 1.0)),
        "fraction_within_3m_after_refinement": float(np.mean(d_ref < 3.0)),
        "gt_surface_coverage_within_2m": float(np.mean(dcov < 2.0)),
        "gt_surface_coverage_within_5m": float(np.mean(dcov < 5.0)),
        "note": "Nearest-neighbour distance from aligned VGGT points to a dense sample of the Blender surface. Points on sky, "
                "far background or vegetation interiors inflate the tail; coverage is over the whole sampling box, "
                "which is larger than the visible area.",
    }
    # Distance vs range from the nearest camera: does error grow with depth?
    cam_tree = cKDTree(G)
    rng_to_cam, _ = cam_tree.query(pts_ref, k=1, workers=-1)
    bins = [0, 25, 50, 100, 200, 400, 1e9]
    by_range = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rng_to_cam >= lo) & (rng_to_cam < hi)
        if m.sum() > 100:
            by_range.append({"range_m": [lo, hi if hi < 1e8 else None], **summary(d_ref[m])})
    report["cloud_vs_surface"]["by_range_to_nearest_camera"] = by_range

    # ---------------------------------------------------------------- per-frame depth vs fisheye GT depth
    depth_dir = gt_dir / "gt_depth"
    if args.method == "vggt_omega" and pred_path.exists() and depth_dir.exists():
        z = np.load(pred_path, allow_pickle=True)
        keys = list(z.files)
        depth = z["depth"] if "depth" in keys else None
        if depth is not None:
            depth = np.asarray(depth)
            if depth.ndim == 4:
                depth = depth[..., 0]
            S, H, W = depth.shape
            intr = gt["intrinsics"]
            if intr.get("model") == "pinhole":
                # GT depth is already camera-z; the central-cone mask uses the pinhole ray angles.
                Wp, Hp = intr["resolution_px"]
                uu, vv = np.meshgrid(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5)
                rays = np.stack([(uu - intr["cx"]) / intr["fx"], (vv - intr["cy"]) / intr["fy"], np.ones_like(uu)], axis=-1)
                cos_theta = 1.0 / np.linalg.norm(rays, axis=-1)
                cosz = np.ones_like(cos_theta)
                cos_for_mask = cos_theta
            else:
                cosz = equisolid_camera_z_factor(tuple(intr["resolution_px"]), intr["focal_length_mm"], intr["sensor_width_mm"], intr["sensor_height_mm"])
                cos_for_mask = cosz
            cos_c = resize_nearest(cos_for_mask, (H, W))
            conf = np.asarray(z["depth_conf"])[..., 0] if "depth_conf" in keys and np.asarray(z["depth_conf"]).ndim == 4 else (np.asarray(z["depth_conf"]) if "depth_conf" in keys else None)
            per = []
            ratios_all = []
            for i, fid in enumerate(frame_ids[:S]):
                gd_path = depth_dir / f"{fid:05d}.npy"
                if not gd_path.exists():
                    continue
                gd = np.load(gd_path).astype(np.float64)
                gz = gd * cosz
                valid_gt = np.isfinite(gz) & (gz > 0.1) & (gz < 5000)
                # VGGT-Omega resizes the full frame to (H, W) without cropping (800x600 -> 592x448).
                gz_c = resize_nearest(np.where(valid_gt, gz, np.nan), (H, W))
                pd = depth[i].astype(np.float64) * s
                m = np.isfinite(gz_c) & (pd > 0)
                if conf is not None and args.conf_quantile > 0:
                    m &= conf[i] >= np.quantile(conf[i], args.conf_quantile)
                if m.sum() < 500:
                    continue
                ratio = pd[m] / gz_c[m]
                rel_err = np.abs(pd[m] - gz_c[m]) / gz_c[m]
                ratios_all.append(np.log(ratio))
                central = m & (cos_c > np.cos(np.radians(30.0)))
                central_ratio = float(np.median(pd[central] / gz_c[central])) if central.sum() > 200 else None
                per.append({"frame": int(fid), "valid_px": int(m.sum()), "median_ratio_pred_over_gt": float(np.median(ratio)),
                            "median_ratio_central_30deg": central_ratio,
                            "log_ratio_mad": float(np.median(np.abs(np.log(ratio) - np.median(np.log(ratio))))),
                            "abs_rel_error_median": float(np.median(rel_err)), "abs_rel_error_p90": float(np.percentile(rel_err, 90)),
                            "gt_depth_median_m": float(np.median(gz_c[m]))})
            if per:
                lr = np.concatenate(ratios_all)
                report["depth_vs_gt"] = {
                    "frames_scored": len(per), "pred_depth_shape": [S, H, W],
                    "global_median_ratio_pred_over_gt": float(np.exp(np.median(lr))),
                    "median_central_30deg_ratio_across_frames": float(np.median([p["median_ratio_central_30deg"] for p in per if p["median_ratio_central_30deg"]])),
                    "global_log_ratio_mad": float(np.median(np.abs(lr - np.median(lr)))),
                    "median_abs_rel_error_across_frames": float(np.median([p["abs_rel_error_median"] for p in per])),
                    "per_frame": per,
                    "note": "Predicted depth is scaled by the Sim(3) metres-per-unit and compared with fisheye GT ray length "
                            "converted to camera-z; VGGT assumes a pinhole camera so a radial pattern in the error is expected.",
                }
    report["per_frame_cameras"] = per_frame
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    # ---------------------------------------------------------------- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(G[:, 0], G[:, 1], "k-", label="ground truth")
        ax[0].plot(P_al[:, 0], P_al[:, 1], "r--", label="VGGT (Sim3)")
        ax[0].set_aspect("equal"); ax[0].set_title("Camera path, top view (m)"); ax[0].legend()
        ax[1].plot(common, err); ax[1].set_title("Camera centre error after Sim(3) (m)"); ax[1].set_xlabel("frame")
        xs = [np.mean(l["frames"]) for l in local]
        ax[2].plot(xs, rel, "o-"); ax[2].axhline(1, color="k", lw=0.8); ax[2].set_title("Local scale / global scale"); ax[2].set_xlabel("frame")
        fig.tight_layout(); fig.savefig(out_dir / "cameras_and_scale.png", dpi=120); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.clip(d_ref, 0, 20), bins=80); ax.set_xlabel("distance to Blender surface (m, clipped 20)"); ax.set_title("Aligned cloud vs surface")
        fig.tight_layout(); fig.savefig(out_dir / "surface_distance_hist.png", dpi=120); plt.close(fig)
        if "depth_vs_gt" in report:
            fig, ax = plt.subplots(figsize=(7, 4))
            pf = report["depth_vs_gt"]["per_frame"]
            ax.plot([p["frame"] for p in pf], [p["median_ratio_pred_over_gt"] for p in pf], "o-", label="median pred/gt depth")
            ax.plot([p["frame"] for p in pf], [p["median_ratio_central_30deg"] for p in pf], "^-", label="median pred/gt depth, central 30 deg")
            ax.plot([p["frame"] for p in pf], [p["abs_rel_error_median"] for p in pf], "s-", label="median |rel err|")
            ax.axhline(1, color="k", lw=0.8); ax.legend(); ax.set_xlabel("frame"); ax.set_title("Per-frame depth vs fisheye GT")
            fig.tight_layout(); fig.savefig(out_dir / "depth_vs_gt.png", dpi=120); plt.close(fig)
    except Exception as exc:  # plots are optional
        report["plot_error"] = str(exc)
        (out_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_frame_cameras",)}, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
