#!/usr/bin/env python3
"""Index the real-scene lens-test reconstructions for the scene comparison page."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "scenes" / "real_two_scene_lens_test"
C = ROOT / "benchmarks" / "synthetic_flyover" / "real_calibration"
entries = []
for scene in ("scene_a", "scene_b"):
    lt = json.loads((C / scene / "lens_test.json").read_text()) if (C / scene / "lens_test.json").exists() else {"runs": {}}
    for variant, label in (("original", "original frames"), ("pinhole", "undistorted (GLOMAP fisheye, centred principal point)"), ("pinhole2", "undistorted (COLMAP-refined radial model with principal point)"), ("pinhole3", "undistorted (4-term fisheye with refined principal point)"), ("pinhole4", "undistorted (single-focal fisheye with refined principal point)")):
        d = R / f"{scene}_{variant}"
        meta_path = d / "viewer" / "scene_meta.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text())
        wall = json.loads((d / "wall_height_measurement.json").read_text()) if (d / "wall_height_measurement.json").exists() else None
        run = lt["runs"].get({"original": "original", "pinhole": "pinhole_glomap", "pinhole2": "pinhole_colmap_pp", "pinhole3": "pinhole_fisheye4_pp", "pinhole4": "pinhole_fisheye1_pp"}[variant], lt["runs"].get(variant, {}))
        entries.append({
            "scene": scene, "method": f"vggt_omega_{variant}", "profile": "full", "input_views": len(m["path"]),
            "title": f"{scene}: VGGT-Omega, {label} ({len(m['path'])} frames)", "key": f"{scene}_{variant}",
            "viewer_url": f"/scenes/real_two_scene_lens_test/{scene}_{variant}/viewer/", "point_count": m["point_count"], "cameras": len(m["path"]),
            "attempt": "lens test 2026-09-12", "alignment": {"note": "own frame; scale from the published manual calibration"},
            "sim3_rmse_fraction_published_path_rms_radius": run.get("sim3_vs_colmap_path", {}).get("rmse_fraction_of_path_length"),
            "lens_test": {"fx_ratio": run.get("fx_ratio_vggt_over_reference"), "local_scale_std": run.get("local_scale_relative", {}).get("std"),
                          "local_scale_range": run.get("local_scale_relative", {}).get("range"),
                          "wall_m_per_unit": wall and wall.get("metres_per_unit_from_wall_profile"), "wall_over_path": wall and wall.get("wall_over_path_length")},
        })
out = R / "index.json"
out.write_text(json.dumps({"schema_version": 1, "entries": entries}, indent=2) + "\n")
print(f"wrote {out} ({len(entries)} entries)")
