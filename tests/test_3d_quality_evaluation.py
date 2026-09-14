from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "tools" / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import evaluate_3d_visualization as visual  # noqa: E402
from quality_result_inventory import comparison_status  # noqa: E402
from quality_render_querysplat_evaluation import align_cameras, transform_c2w  # noqa: E402


class StrictVisualEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.camera_path = self.root / "cameras.json"
        self.camera_path.write_text('{"frames": [{"id": "camera-0"}]}\n')

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save_rgb(self, name: str, shape: tuple[int, int], value: int) -> Path:
        height, width = shape
        path = self.root / name
        Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(path)
        return path

    def record(self, reference: Path, render: Path, **changes) -> dict:
        record = {
            "frame_id": "evaluation:source_000001",
            "scene": "scene_a",
            "segment_id": "seg03",
            "view_role": "evaluation",
            "reference": reference.name,
            "render": render.name,
            "pixel_mapping": {"type": "identity", "size": [reference_image_width(reference), reference_image_height(reference)]},
            "camera": {
                "camera_id": "camera-0",
                "target_frame_id": "evaluation:source_000001",
                "artifact": self.camera_path.name,
                "record_id": "frames[0]",
                "pose_source": "training_geometry_localization",
                "intrinsics_source": "frozen_evaluation_intrinsics",
                "photometric_fit_target_rgb": False,
            },
        }
        record.update(changes)
        return record

    def evaluate(self, record: dict) -> dict:
        return visual.evaluate_record(
            record,
            self.root,
            set(),
            structural_similarity=None,
            lpips_scorer=None,
            index=0,
        )

    def test_camera_target_frame_mismatch_is_rejected(self) -> None:
        reference = self.save_rgb("reference.png", (16, 20), 100)
        render = self.save_rgb("render.png", (16, 20), 101)
        record = self.record(reference, render)
        record["camera"]["target_frame_id"] = "evaluation:wrong-frame"
        with self.assertRaisesRegex(ValueError, "does not match frame_id"):
            self.evaluate(record)

    def test_pixel_perfect_result_uses_json_null_not_infinity(self) -> None:
        reference = self.save_rgb("reference.png", (16, 20), 73)
        render = self.save_rgb("render.png", (16, 20), 73)
        manifest = {
            "schema_version": 1,
            "dataset_id": "quality-unit-test",
            "method_run_id": "perfect-test",
            "records": [self.record(reference, render)],
        }
        manifest_path = self.root / "pairs.json"
        output_path = self.root / "metrics.json"
        manifest_path.write_text(json.dumps(manifest))
        argv = [
            "evaluate_3d_visualization.py",
            "--pairs-manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(visual.main(), 0)
        raw = output_path.read_text()
        payload = json.loads(raw, parse_constant=lambda value: self.fail(f"invalid JSON constant: {value}"))
        frame = payload["images"][0]["full_frame"]
        self.assertIsNone(frame["psnr_db"])
        self.assertTrue(frame["perfect_match"])
        self.assertIsNone(payload["full_frame"]["mean_psnr_db"])
        self.assertEqual(payload["full_frame"]["perfect_match_count"], 1)

    def test_mismatched_identity_dimensions_are_rejected(self) -> None:
        reference = self.save_rgb("reference.png", (16, 20), 50)
        render = self.save_rgb("render.png", (15, 20), 50)
        with self.assertRaisesRegex(ValueError, "identity-mapped dimensions differ"):
            self.evaluate(self.record(reference, render))

    def test_nonempty_and_empty_masks_are_reported_without_nan(self) -> None:
        reference = self.save_rgb("reference.png", (16, 20), 0)
        render = self.save_rgb("render.png", (16, 20), 20)
        nonempty = np.zeros((16, 20), dtype=np.uint8)
        nonempty[2:6, 3:8] = 255
        Image.fromarray(nonempty).save(self.root / "valid.png")
        record = self.record(reference, render, valid_mask="valid.png")
        result = self.evaluate(record)
        self.assertEqual(result["masked"]["valid_pixels"], 20)
        self.assertAlmostEqual(result["masked"]["coverage_fraction"], 20 / 320)
        self.assertTrue(math.isfinite(result["masked"]["psnr_db"]))

        Image.fromarray(np.zeros((16, 20), dtype=np.uint8)).save(self.root / "empty.png")
        empty_result = self.evaluate(self.record(reference, render, valid_mask="empty.png"))
        self.assertEqual(empty_result["masked"]["valid_pixels"], 0)
        self.assertEqual(empty_result["masked"]["coverage_fraction"], 0.0)
        self.assertIsNone(empty_result["masked"]["psnr_db"])
        self.assertEqual(empty_result["masked"]["unavailable_reason"], "empty_valid_mask")
        json.dumps(empty_result, allow_nan=False)


class ResultInventoryTests(unittest.TestCase):
    def test_querysplat_metadata_without_geometry_is_not_reviewable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            run_json = root / "run.json"
            run_log = root / "run.log"
            run_json.write_text('{"status": "unrecovered"}\n')
            run_log.write_text("")
            status, warnings = comparison_status("querysplat", "tto", [run_json, run_log])
        self.assertEqual(status, "not_reviewable")
        self.assertIn("missing Gaussian PLY export", warnings)


def reference_image_width(path: Path) -> int:
    with Image.open(path) as image:
        return image.width


def reference_image_height(path: Path) -> int:
    with Image.open(path) as image:
        return image.height


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis.astype(np.float64) / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def camera(center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = center
    return matrix


class ProxyCameraAlignmentTests(unittest.TestCase):
    def test_sim3_recovers_orientation_scale_translation_and_low_residuals(self) -> None:
        global_rotation = axis_angle_rotation(np.array([0.3, -0.4, 0.8]), 0.47)
        scale = 2.75
        translation = np.array([4.0, -2.0, 1.5])
        centers = np.array(
            [
                [-2.0, 0.2, 0.0],
                [-1.0, -0.4, 0.5],
                [0.0, 0.8, -0.3],
                [1.0, -0.2, 0.9],
                [2.0, 0.5, -0.5],
                [3.0, -0.7, 0.2],
            ]
        )
        local_rotations = [
            axis_angle_rotation(np.array([0.2 + index, 1.0, 0.4]), 0.03 * index)
            for index in range(len(centers))
        ]
        joint = np.stack([camera(center, rotation) for center, rotation in zip(centers, local_rotations)])
        frozen = np.stack(
            [
                camera(scale * (global_rotation @ center) + translation, global_rotation @ rotation)
                for center, rotation in zip(centers, local_rotations)
            ]
        )

        estimated_scale, estimated_rotation, estimated_translation, diagnostics = align_cameras(joint, frozen)
        np.testing.assert_allclose(estimated_scale, scale, atol=1e-10)
        np.testing.assert_allclose(estimated_rotation, global_rotation, atol=1e-10)
        np.testing.assert_allclose(estimated_translation, translation, atol=1e-10)
        self.assertLess(diagnostics["center_rmse_fraction_of_path_extent"], 1e-10)
        self.assertLess(diagnostics["orientation_median_degrees"], 1e-6)

        probe = camera(np.array([4.0, 1.0, -0.5]), axis_angle_rotation(np.array([1.0, 0.0, 1.0]), 0.2))
        transformed = transform_c2w(probe, estimated_scale, estimated_rotation, estimated_translation)
        np.testing.assert_allclose(transformed[:3, 3], scale * (global_rotation @ probe[:3, 3]) + translation, atol=1e-10)
        np.testing.assert_allclose(transformed[:3, :3], global_rotation @ probe[:3, :3], atol=1e-10)

    def test_alignment_diagnostics_expose_inconsistent_common_cameras(self) -> None:
        centers = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1], [3, -1, 0], [4, 0, -1]], dtype=np.float64)
        joint = np.stack([camera(center, np.eye(3)) for center in centers])
        frozen = joint.copy()
        frozen[-1, :3, 3] += np.array([0.0, 4.0, 3.0])
        frozen[-1, :3, :3] = axis_angle_rotation(np.array([0.0, 0.0, 1.0]), math.radians(45))
        _, _, _, diagnostics = align_cameras(joint, frozen)
        self.assertGreater(diagnostics["center_rmse_fraction_of_path_extent"], 0.1)
        self.assertGreater(diagnostics["orientation_median_degrees"], 1.0)


if __name__ == "__main__":
    unittest.main()
