#!/usr/bin/env python3
from __future__ import annotations

import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("gzip_scene_bins.py")
SPEC = importlib.util.spec_from_file_location("gzip_scene_bins", SCRIPT)
assert SPEC and SPEC.loader
gzip_scene_bins = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gzip_scene_bins)


class GzipSceneBinsTests(unittest.TestCase):
    def test_scene_binary_filter(self) -> None:
        self.assertTrue(gzip_scene_bins.is_scene_binary("scenes/video/scene/viewer/points_positions.bin"))
        self.assertFalse(gzip_scene_bins.is_scene_binary("scenes/video/scene/point_cloud.npz"))
        self.assertFalse(gzip_scene_bins.is_scene_binary("other/viewer/points_positions.bin"))

    def test_gzip_is_deterministic_and_decodes_exactly(self) -> None:
        payload = (b"point-cloud-data\x00" * 1000) + bytes(range(256))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.bin"
            first = Path(tmp) / "first.gz"
            second = Path(tmp) / "second.gz"
            source.write_bytes(payload)
            gzip_scene_bins.gzip_file(source, first)
            gzip_scene_bins.gzip_file(source, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(gzip.decompress(first.read_bytes()), payload)


if __name__ == "__main__":
    unittest.main()
