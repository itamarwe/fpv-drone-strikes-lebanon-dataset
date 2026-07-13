#!/usr/bin/env python3
"""Focused regression tests for the resumable VGGT batch runner."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("run_vggt_batch_from_annotations.py")
SPEC = importlib.util.spec_from_file_location("run_vggt_batch", SCRIPT)
assert SPEC and SPEC.loader
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class BatchRunnerTests(unittest.TestCase):
    def test_checkpoint_replaces_previous_result_for_same_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenes" / ".batch.json"
            batch.write_results(path, {"scene-a": {"scene_id": "scene-a", "status": "error"}})
            rows = batch.load_results(path)
            rows["scene-a"] = {"scene_id": "scene-a", "status": "done"}
            rows["scene-b"] = {"scene_id": "scene-b", "status": "skipped_complete"}
            batch.write_results(path, rows)
            saved = {row["scene_id"]: row["status"] for row in json.loads(path.read_text())}
        self.assertEqual(saved, {"scene-a": "done", "scene-b": "skipped_complete"})

    def test_scene_paths_follow_configured_output_root(self) -> None:
        annotation = {"video_file": "2026-05-26_anti_drone_platform_biranit.mp4"}
        self.assertEqual(
            batch.scene_dir_for(Path("/tmp/output"), annotation, "scene-1"),
            Path("/tmp/output/scenes/2026-05-26_anti_drone_platform_biranit/scene-1"),
        )

    def test_relative_result_file_uses_output_root(self) -> None:
        with patch("sys.argv", ["batch", "--out-dir", "/tmp/output"]):
            args = batch.parse_args()
        self.assertEqual(args.results_file, Path("/tmp/output/scenes/.batch_results.json"))

    def test_retries_transient_server_failure(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"ok": true}'

        request = batch.urllib.request.Request("http://127.0.0.1:8766/api/health")
        with patch.object(batch.urllib.request, "urlopen", side_effect=[batch.urllib.error.URLError("offline"), Response()]) as open_mock:
            with patch.object(batch.time, "sleep"):
                self.assertEqual(batch.request_json(request, retries=1, timeout=1), {"ok": True})
        self.assertEqual(open_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
