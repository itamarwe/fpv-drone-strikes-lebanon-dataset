#!/usr/bin/env python3
"""Serve generated VGGT scene-viewer assets from this checkout.

This is intentionally small: it serves the generic scene viewer UI plus static
files under scenes/<video>/<scene>/viewer so large point buffers can be loaded
locally without the full annotator tool server.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
SCENES_ROOT = ROOT / "scenes"
VIEWER_RE = re.compile(r"^/scenes/(.+)/viewer(?:/index\.html)?/?$")
GROUNDING_VIEWER_RE = re.compile(r"^/scenes/(.+)/viewer_grounding(?:/index\.html)?/?$")
SELECTION_RE = re.compile(r"^/api/selection/([A-Za-z0-9_-]+)$")  # review picks for a batch: benchmarks/<batch>/review_selection.json
SELECTION_LOCK = threading.Lock()


def selection_path(batch: str) -> Path:
    return ensure_child(ROOT / "benchmarks", batch, "review_selection.json")


def read_selection(batch: str) -> dict:
    data = load_json(selection_path(batch))
    sel = data.get("selected") if isinstance(data.get("selected"), list) else []
    return {"batch": batch, "selected": sorted(str(v) for v in sel), "updated_utc": data.get("updated_utc")}


def write_selection(batch: str, selected: set[str]) -> dict:
    path = selection_path(batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"batch": batch, "selected": sorted(selected), "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2) + "\n"); tmp.replace(path)
    return payload


def ensure_child(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise PermissionError(str(path))
    return path


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(data)


class SceneViewerHandler(BaseHTTPRequestHandler):
    server_version = "FPVSceneViewer/0.1"

    @property
    def viewer_root(self) -> Path:
        return self.server.viewer_root  # type: ignore[attr-defined]

    @property
    def viewer_index(self) -> Path:
        return self.viewer_root / "index.html"

    @property
    def grounded_viewer_root(self) -> Path:
        return self.server.grounded_viewer_root  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        print("[viewer-server] " + fmt % args, flush=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        """Update a batch's review selection. Body: {"selected": [...]} replaces it; {"toggle": id, "on": bool} edits one."""
        path = unquote(urlparse(self.path).path)
        match = SELECTION_RE.match(path)
        if not match:
            write_json(self, {"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            batch = match.group(1)
            with SELECTION_LOCK:
                current = set(read_selection(batch)["selected"])
                if isinstance(body.get("selected"), list):
                    current = {str(v) for v in body["selected"]}
                if body.get("toggle"):
                    (current.add if body.get("on") else current.discard)(str(body["toggle"]))
                write_json(self, write_selection(batch, current))
        except Exception as exc:
            write_json(self, {"error": str(exc)}, status=400)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/scenes":
                self.serve_scene_list()
            elif SELECTION_RE.match(path):
                with SELECTION_LOCK:
                    write_json(self, read_selection(SELECTION_RE.match(path).group(1)))
            elif path in {"", "/"}:
                self.serve_text("Scene viewer server is running. Open /scenes/<video>/<scene>/viewer/.\n")
            elif path.startswith("/tools/scene_viewer/"):
                self.serve_file(ensure_child(self.viewer_root, path.removeprefix("/tools/scene_viewer/")))
            elif path.startswith("/tools/grounded_scene_viewer/"):
                self.serve_file(ensure_child(self.grounded_viewer_root, path.removeprefix("/tools/grounded_scene_viewer/")))
            elif GROUNDING_VIEWER_RE.match(path):
                self.serve_grounding_viewer(path)
            elif VIEWER_RE.match(path):
                self.serve_viewer(path)
            elif path.startswith("/scenes/"):
                self.serve_file(ensure_child(SCENES_ROOT, *path.strip("/").split("/")[1:]))
            else:
                self.serve_file(ensure_child(ROOT, path.lstrip("/")))
        except Exception as exc:
            write_json(self, {"error": str(exc)}, status=500)

    def serve_text(self, text: str) -> None:
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def serve_scene_list(self) -> None:
        scenes = []
        for meta_path in SCENES_ROOT.glob("*/*/viewer/scene_meta.json"):
            scene_dir = meta_path.parents[1]
            rel = scene_dir.relative_to(SCENES_ROOT).as_posix()
            meta = load_json(meta_path)
            title = str(meta.get("title") or scene_dir.name)
            scenes.append(
                {
                    "scene_id": str(meta.get("scene_id") or scene_dir.name),
                    "title": title,
                    "description": title,
                    "viewer_url": f"/scenes/{rel}/viewer/",
                    "exists": True,
                    "date": "",
                    "town": "",
                    "segment_ids": [],
                    "point_count": int(meta.get("point_count") or 0),
                }
            )
        scenes.sort(key=lambda scene: (scene.get("title") or scene.get("scene_id") or ""))
        write_json(self, {"scenes": scenes})

    def serve_viewer(self, path: str) -> None:
        match = VIEWER_RE.match(path)
        assert match is not None
        rel_scene = match.group(1)
        viewer_dir = ensure_child(SCENES_ROOT, rel_scene, "viewer")
        if not (viewer_dir / "scene_meta.json").exists():
            write_json(self, {"error": "not found"}, status=404)
            return
        scene_base = f"/scenes/{rel_scene}/viewer/"
        data = (
            self.viewer_index.read_text()
            .replace("__SCENE_BASE__", scene_base)
            .replace("__APP_BASE__", "")
            .replace("__API_BASE__", "")
            .encode()
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def serve_grounding_viewer(self, path: str) -> None:
        match = GROUNDING_VIEWER_RE.match(path)
        assert match is not None
        rel_scene = match.group(1)
        viewer_dir = ensure_child(SCENES_ROOT, rel_scene, "viewer_grounding")
        if not (viewer_dir / "grounded_scene_meta.json").exists():
            write_json(self, {"error": "not found"}, status=404)
            return
        scene_base = f"/scenes/{rel_scene}/viewer_grounding/"
        data = self.grounded_viewer_root.joinpath("index.html").read_text().replace("__SCENE_BASE__", scene_base).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def serve_file(self, target: Path) -> None:
        if target.is_dir():
            target = target / "index.html"
        if not target.exists() or target.is_dir():
            write_json(self, {"error": "not found"}, status=404)
            return
        size = target.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command == "HEAD":
            return
        with target.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--viewer-root",
        type=Path,
        default=(ROOT / "tools" / "scene_viewer") if (ROOT / "tools" / "scene_viewer").exists() else ROOT / "tools" / "apps" / "scene-viewer",
        help="Directory containing index.html, viewer.js, and viewer.css (default: tools/apps/scene-viewer).",
    )
    parser.add_argument(
        "--grounded-viewer-root",
        type=Path,
        default=ROOT / "tools" / "grounded_scene_viewer",
        help="Directory containing the grounded-scene viewer assets.",
    )
    args = parser.parse_args()
    viewer_root = args.viewer_root.resolve()
    if not (viewer_root / "index.html").exists():
        raise SystemExit(f"Missing scene viewer assets: {viewer_root}")
    grounded_viewer_root = args.grounded_viewer_root.resolve()
    if not (grounded_viewer_root / "index.html").exists():
        raise SystemExit(f"Missing grounded scene viewer assets: {grounded_viewer_root}")
    server = ThreadingHTTPServer((args.host, args.port), SceneViewerHandler)
    server.viewer_root = viewer_root  # type: ignore[attr-defined]
    server.grounded_viewer_root = grounded_viewer_root  # type: ignore[attr-defined]
    print(f"FPV scene viewer server: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
