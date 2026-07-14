#!/usr/bin/env python3
"""Focused tests for reconstruction experiment settings and scene metadata."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ROOT / "tools" / "pipeline"))
server = load_module("fpv_tool_server_test", ROOT / "tools" / "server" / "fpv_tool_server.py")
reconstruct = load_module("reconstruct_scenes_test", ROOT / "tools" / "pipeline" / "reconstruct_scenes.py")


class ScenePipelineConfigTests(unittest.TestCase):
    def test_clean_crop_is_explicit_260_by_280(self) -> None:
        crop = server.crop_config("central_clean")
        self.assertEqual(crop["reference_bbox_px"], {"x": 320, "y": 190, "width": 260, "height": 280})
        self.assertIn("iw*260/848", server.ffmpeg_filter("central_clean", 260, 24))
        self.assertIn("iw*320/848", server.ffmpeg_filter("central_clean", 260, 24))

    def test_direct_images_are_the_default_transport(self) -> None:
        state = SimpleNamespace(
            vggt_space="facebook/vggt-omega",
            vggt_backend="omega",
            default_scale=117.6,
        )
        config = server.reconstruction_config(state, {})
        self.assertEqual(config["vggt"]["upload_mode"], "images")
        self.assertIsNone(config["vggt"]["upload_video_fps"])

    def test_experiment_overrides_follow_preset_defaults(self) -> None:
        args = argparse.Namespace(preset="clean", tail_seconds=8.0, exclude_tail_seconds=2.0, frames=80)
        flags = reconstruct.reconstruction_flags(args)
        self.assertEqual(flags[flags.index("--tail-seconds") + 1], "8.0")
        self.assertEqual(flags[flags.index("--exclude-tail-seconds") + 1], "2.0")
        self.assertEqual(flags[flags.index("--adaptive-target") + 1], "80")

if __name__ == "__main__":
    unittest.main()
