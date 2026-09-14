from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "tools" / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from evaluate_3d_trajectory import load_prediction, rigid_alignment, trajectory_stats, umeyama  # noqa: E402
from evaluate_moge3_vggt_scale import evaluate  # noqa: E402
from fit_moge3_colmap_scale import ColmapImage, frame_votes, reference_scale_comparison  # noqa: E402
from fit_moge3_scal3r_scale import evaluate as evaluate_scal3r_scale  # noqa: E402
from fit_moge3_scal3r_scale import scal3r_image_transform  # noqa: E402
from summarize_colmap_diagnostics import numeric_summary, read_point_diagnostics  # noqa: E402
from summarize_scal3r_diagnostics import opencv_matrices  # noqa: E402
from metric_scale import (  # noqa: E402
    crop_depth_to_vggt_view,
    occupied_grid_cells,
    project_camera_z,
    robust_log_scale,
    vggt_supported_crop,
)


class MetricNumericsTests(unittest.TestCase):
    def test_robust_log_scale_rejects_multiplicative_outlier(self) -> None:
        values = np.array([2.9, 3.0, 3.1, 300.0, np.nan, -2.0])
        fit = robust_log_scale(values, max_log_deviation=0.2)
        self.assertAlmostEqual(fit.scale, 3.0)
        self.assertEqual(fit.kept_count, 3)
        self.assertEqual(fit.input_count, 6)

    def test_projection_uses_camera_z_and_nearest_surface(self) -> None:
        points = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
        depth, valid = project_camera_z(
            points,
            center=np.zeros(3),
            right=np.array([1.0, 0.0, 0.0]),
            down=np.array([0.0, 1.0, 0.0]),
            forward=np.array([0.0, 0.0, 1.0]),
            width=5,
            height=5,
            focal_x_px=2.0,
        )
        self.assertTrue(valid[2, 2])
        self.assertEqual(depth[2, 2], 2.0)
        self.assertEqual(depth[2, 4], 2.0)

    def test_full_and_crop_depth_layouts_are_explicit(self) -> None:
        crop = vggt_supported_crop(10, 4)
        self.assertEqual(crop, (1, 0, 8, 4))
        full = np.arange(40).reshape(4, 10)
        normalized, _, layout = crop_depth_to_vggt_view(full, np.ones_like(full), (10, 4), crop)
        self.assertEqual(layout, "full")
        np.testing.assert_array_equal(normalized, full[:, 1:9])
        with self.assertRaises(ValueError):
            crop_depth_to_vggt_view(np.zeros((3, 3)), np.ones((3, 3)), (10, 4), crop)

    def test_spatial_coverage_counts_cells_not_votes(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[0, 0] = mask[7, 7] = True
        self.assertEqual(occupied_grid_cells(mask, 2, 2), 2)


class PublishedBaselineEvaluatorTests(unittest.TestCase):
    def test_recovers_scale_and_measured_length_on_same_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            scene = root / "scene"
            (scene / "source" / "images").mkdir(parents=True)
            (scene / "baseline").mkdir()
            (scene / "profiles" / "common16").mkdir(parents=True)
            depths = root / "depths"
            depths.mkdir()
            Image.new("RGB", (10, 4)).save(scene / "source" / "images" / "f_000001.jpg")

            points = []
            for pixel_y in range(4):
                for pixel_x in range(1, 9):
                    z = 2.0
                    points.append([(pixel_x - 5.0) * z / 4.0, (pixel_y - 2.0) * z / 4.0, z])
            np.asarray(points, dtype="<f4").tofile(scene / "baseline" / "points_positions.bin")
            metadata = {
                "scene_id": "synthetic",
                "calibration": {
                    "scale_m_per_vggt_unit": 3.0,
                    "real_length_m": 1.5,
                    "measured_vggt_units": 0.5,
                    "source": "measured",
                },
                "path": [
                    {
                        "position": [0, 0, 0],
                        "right": [1, 0, 0],
                        "down": [0, 1, 0],
                        "forward": [0, 0, 1],
                        "segment_id": "seg01",
                    }
                ],
            }
            (scene / "source" / "scene_meta.json").write_text(json.dumps(metadata))
            with (scene / "profiles" / "common16" / "frames.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "profile_index",
                        "source_index",
                        "profile_file",
                        "source_file",
                        "segment_id",
                        "video_time_s",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "profile_index": 0,
                        "source_index": 0,
                        "profile_file": "frame_000000.jpg",
                        "source_file": "f_000001.jpg",
                        "segment_id": "seg01",
                        "video_time_s": 0.0,
                    }
                )
            np.savez_compressed(
                depths / "frame_000000.npz",
                depth=np.full((4, 10), 6.0, dtype=np.float32),
                mask=np.ones((4, 10), dtype=np.uint8),
                intrinsics=np.array([[0.4, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]),
            )
            args = Namespace(
                scene_dir=scene,
                profile="common16",
                depths=depths,
                output=root / "summary.json",
                votes_dir=None,
                focal_source="fixed",
                focal_px=4.0,
                depth_layout="auto",
                min_votes=20,
                min_spatial_cells=8,
                grid_rows=2,
                grid_columns=4,
                max_log_deviation=0.2,
                min_metric_depth_m=0.1,
                max_metric_depth_m=20.0,
                max_points=1000,
            )
            result = evaluate(args)
            self.assertAlmostEqual(result["fit"]["scale_m_per_vggt_unit"], 3.0)
            measured = result["withheld_measured_element_evaluation"]
            self.assertAlmostEqual(measured["predicted_length_m"], 1.5)
            self.assertAlmostEqual(measured["relative_error"], 0.0)
            self.assertTrue(result["coordinate_comparison_valid"])


class Scal3RScaleTests(unittest.TestCase):
    def test_pinned_660x280_preprocessing_contract(self) -> None:
        transform = scal3r_image_transform(660, 280)
        self.assertEqual((transform.resized_width, transform.resized_height), (518, 224))
        self.assertEqual((transform.crop_left, transform.crop_top), (0, 7))
        self.assertEqual((transform.output_width, transform.output_height), (518, 210))

    def test_pixel_aligned_depth_ratio_recovers_native_gauge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            images = root / "images"
            moge = root / "moge"
            scal3r = root / "scal3r"
            (scal3r / "depths").mkdir(parents=True)
            images.mkdir()
            moge.mkdir()
            Image.new("RGB", (10, 4)).save(images / "frame_000000.jpg")
            np.savez_compressed(
                moge / "frame_000000.npz",
                depth=np.full((4, 10), 6.0, dtype=np.float32),
                mask=np.ones((4, 10), dtype=np.uint8),
            )
            np.save(scal3r / "depths" / "000000.npy", np.full((2, 8), 2.0, dtype=np.float32))
            result = evaluate_scal3r_scale(
                Namespace(
                    images=images,
                    scal3r_result=scal3r,
                    moge_depths=moge,
                    output=root / "scale.json",
                    votes_dir=None,
                    proc_max_size=8,
                    proc_align_size=2,
                    min_votes=8,
                    min_spatial_cells=4,
                    grid_rows=2,
                    grid_columns=4,
                    max_log_deviation=0.2,
                    min_metric_depth_m=0.1,
                    max_metric_depth_m=20.0,
                )
            )
            self.assertAlmostEqual(result["scale_m_per_scal3r_unit"], 3.0)
            self.assertEqual(
                result["known_element_evaluation"]["status"],
                "unavailable_missing_corresponding_endpoints_in_scal3r_geometry",
            )

    def test_opencv_yaml_matrix_parser(self) -> None:
        text = """K_000001: !!opencv-matrix
  rows: 2
  cols: 2
  dt: d
  data: [1.0, 2.0, 3.0, 4.0]
"""
        np.testing.assert_array_equal(opencv_matrices(text, "K")["000001"], [[1, 2], [3, 4]])


class ColmapContractTests(unittest.TestCase):
    def test_cross_coordinate_scale_reference_is_not_scored(self) -> None:
        comparison = reference_scale_comparison(10.0, 100.0, "new-glomap", "published-vggt")
        self.assertEqual(comparison["status"], "not_evaluated_unverified_coordinate_system")
        self.assertIsNone(comparison["relative_error"])
        valid = reference_scale_comparison(90.0, 100.0, "published-vggt", "published-vggt")
        self.assertAlmostEqual(valid["relative_error"], 0.1)

    def test_depth_shape_must_match_colmap_camera_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            depth_path = Path(temp_name) / "depth.npz"
            np.savez_compressed(depth_path, depth=np.ones((2, 2)), mask=np.ones((2, 2)))
            image = ColmapImage("", "0 0 1", 1, np.array([1, 0, 0, 0]), np.zeros(3), 1, "x.jpg")
            with self.assertRaisesRegex(ValueError, "cannot be sampled safely"):
                frame_votes(image, {1: np.array([0, 0, 1])}, depth_path, expected_size=(3, 2))

    def test_point_diagnostics_parse_residual_and_track_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            points = Path(temp_name) / "points3D.txt"
            points.write_text(
                "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
                "1 0 0 1 1 2 3 0.5 10 2 11 3\n"
                "2 0 1 1 4 5 6 1.5 12 4\n"
            )
            errors, tracks = read_point_diagnostics(points)
            self.assertEqual(errors, [0.5, 1.5])
            self.assertEqual(tracks, [2.0, 1.0])
            self.assertAlmostEqual(numeric_summary(errors)["median"], 1.0)


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_vggt_camera_loader_validates_contract_and_uses_c2w_center(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "cameras.json"
            c2w = np.eye(4)
            c2w[:3, 3] = [1, 2, 3]
            w2c = np.linalg.inv(c2w)
            frame = {
                "index": 0,
                "name": "frame_000000",
                "c2w": c2w.tolist(),
                "w2c": w2c.tolist(),
                "cam_view": w2c.T.tolist(),
                "intrinsics": [10, 11, 5, 6],
                "K": [[10, 0, 5], [0, 11, 6], [0, 0, 1]],
            }
            path.write_text(json.dumps({"frames": [frame]}))
            np.testing.assert_array_equal(load_prediction(path, "vggt-cameras-json")[0], [1, 2, 3])

    def test_sim3_and_rigid_alignment_recover_known_transforms(self) -> None:
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        target = (2.5 * (rotation @ source.T)).T + np.array([4, -3, 7])
        scale, fitted_rotation, translation = umeyama(source, target)
        aligned = (scale * (fitted_rotation @ source.T)).T + translation
        self.assertAlmostEqual(scale, 2.5)
        np.testing.assert_allclose(aligned, target, atol=1e-12)
        rigid_rotation, rigid_translation = rigid_alignment(target, target + np.array([1, 2, 3]))
        rigid_aligned = (rigid_rotation @ target.T).T + rigid_translation
        np.testing.assert_allclose(rigid_aligned, target + np.array([1, 2, 3]), atol=1e-12)

    def test_trajectory_stats_exclude_segment_transition_and_use_time(self) -> None:
        points = np.array([[0, 0, 0], [2, 0, 0], [100, 0, 0], [104, 0, 0]], dtype=float)
        records = [
            {"profile_index": 0, "source_index": 0, "segment_id": "a", "sequence_time_s": 0.0},
            {"profile_index": 1, "source_index": 1, "segment_id": "a", "sequence_time_s": 2.0},
            {"profile_index": 2, "source_index": 2, "segment_id": "b", "sequence_time_s": 2.1},
            {"profile_index": 3, "source_index": 4, "segment_id": "b", "sequence_time_s": 4.1},
        ]
        stats = trajectory_stats(points, records)
        self.assertEqual(stats["segment_transitions_excluded"], 1)
        self.assertEqual(stats["within_segment_intervals"], 2)
        self.assertEqual(stats["source_index_gaps_gt_1"], 1)
        self.assertAlmostEqual(stats["speed_median"], 1.5)


if __name__ == "__main__":
    unittest.main()
