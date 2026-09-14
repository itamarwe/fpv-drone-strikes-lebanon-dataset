from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "tools" / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from fit_moge3_colmap_scale import frame_votes, read_images, read_points, robust_scale  # noqa: E402
from prepare_3d_pipeline_benchmark import evaluation_split, select_common, select_heldout  # noqa: E402


class FrameSelectionTests(unittest.TestCase):
    def test_dense_split_excludes_holdout_buffers_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            scene = Path(temp_name)
            image_dir = scene / "source" / "images"
            image_dir.mkdir(parents=True)
            entries = []
            for index in range(20):
                name = f"f_{index:06d}.png"
                value = 40 if index == 19 else 0 if index == 18 else index * 10
                Image.new("L", (24, 12), value).save(image_dir / name)
                entries.append({"segment_id": "one", "frame_image": name})
            heldout, dense, audit = evaluation_split(entries, scene, [0, 8, 16])
            self.assertEqual(heldout, [4, 12])
            self.assertTrue(set([3, 4, 5, 11, 12, 13, 18, 19]).isdisjoint(dense))
            self.assertTrue(set([0, 8, 16]).issubset(dense))
            self.assertEqual(audit["temporal_buffer_frames_each_side_same_segment"], 1)

    def test_common16_covers_both_segments(self) -> None:
        entries = [
            {"segment_id": "seg03", "frame_image": f"f_{index:06d}.jpg"}
            for index in range(83)
        ] + [
            {"segment_id": "seg04", "frame_image": f"f_{index:06d}.jpg"}
            for index in range(83, 125)
        ]
        selected = select_common(entries, 16)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len(set(selected)), 16)
        self.assertIn(0, selected)
        self.assertIn(124, selected)
        self.assertTrue(any(index < 83 for index in selected))
        self.assertTrue(any(index >= 83 for index in selected))
        heldout = select_heldout(entries, selected)
        self.assertTrue(heldout)
        self.assertTrue(set(heldout).isdisjoint(selected))


class MetricScaleTests(unittest.TestCase):
    def test_colmap_depth_votes_recover_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            model = root / "model"
            depths = root / "depths"
            model.mkdir()
            depths.mkdir()
            observations = []
            point_lines = []
            depth = np.full((5, 5), 6.0, dtype=np.float32)
            mask = np.ones((5, 5), dtype=np.uint8)
            point_id = 1
            for y_coord in range(5):
                for x_coord in range(5):
                    observations.extend([str(x_coord), str(y_coord), str(point_id)])
                    point_lines.append(f"{point_id} 0 0 2 255 255 255 0.1 1 {point_id - 1}")
                    point_id += 1
            (model / "images.txt").write_text(
                "# Image list\n1 1 0 0 0 0 0 0 1 frame_000000.jpg\n"
                + " ".join(observations)
                + "\n"
            )
            (model / "points3D.txt").write_text("# Points\n" + "\n".join(point_lines) + "\n")
            np.savez_compressed(depths / "frame_000000.npz", depth=depth, mask=mask, intrinsics=np.eye(3))
            image = read_images(model / "images.txt")[0]
            points = read_points(model / "points3D.txt")
            votes = frame_votes(image, points, depths / "frame_000000.npz")
            scale, kept = robust_scale(votes, 0.7)
            self.assertEqual(len(kept), 25)
            self.assertAlmostEqual(scale, 3.0)


if __name__ == "__main__":
    unittest.main()
