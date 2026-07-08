#!/usr/bin/env python3
"""Local FPV tagging + VGGT scene tool server.

This deliberately stays small and dependency-light: the browser UI talks to this
server for filesystem and long-running reconstruction work, while the heavy
model steps reuse the existing scripts in this repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from asset_urls import asset_root_from_env, scene_data_base, scene_viewer_page_url
DEFAULT_OUT_DIR = ROOT
DEFAULT_VIDEO_CACHE = Path("/tmp/fpv-model-benchmark/videos")
GENERIC_VIEWER_INDEX = ROOT / "tools" / "scene_viewer" / "index.html"
SCENE_VIEWER_INDEX_RE = re.compile(r"^/scenes/(.+)/viewer(?:/index\.html)?/?$")
# Soft advisory only: above this many frames VGGT reconstruction gets slow/heavy.
# This is NOT a hard cap -- frames are capped only when max_vggt_frames > 0.
VGGT_FRAME_WARN_THRESHOLD = 125

# Local copy of the VGGT-Omega sky-segmentation model, used to run skyseg on the
# CLEAN frames client-side (before painting exclusion boxes), so the black boxes
# never bias the sky prediction.
SKYSEG_ONNX = os.environ.get("SKYSEG_ONNX", "/tmp/fpv-skyseg/skyseg.onnx")
_skyseg_session = None


def slugify(value: str) -> str:
    value = Path(value).stem if value else "video"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return value[:120] or "video"


def scene_id_for(video_file: str, segment_ids: list[str]) -> str:
    clean_segments = [slugify(s).replace("_", "") for s in segment_ids]
    suffix = "_".join(clean_segments) or "selected"
    return f"{slugify(video_file)}_{suffix}"


def scene_rel_dir_for(video_file: str, scene_id: str) -> str:
    return f"{slugify(video_file)}/{scene_id}"


def ensure_child(root: Path, *parts: str) -> Path:
    path = (root.joinpath(*parts)).resolve()
    root_resolved = root.resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise PermissionError(path)
    return path


def run_command(cmd: list[str], job: "Job", *, timeout: int | None = None) -> None:
    job.log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    started = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        job.log(line.rstrip())
        if timeout and time.time() - started > timeout:
            proc.kill()
            raise TimeoutError(f"command timed out after {timeout}s")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def set_job_state(state: "ToolState", job: "Job", *, status: str | None = None, step: str | None = None, error: str | None = None) -> None:
    if status is not None:
        job.status = status
    if step is not None:
        job.step = step
    if error is not None:
        job.error = error
    job.updated_at = time.time()
    state.save_job(job)


def update_scene_manifest(scene_dir: Path | None, fields: dict[str, object]) -> None:
    if scene_dir is None:
        return
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.exists():
        return
    manifest = read_json_if_exists(manifest_path)
    manifest.update(fields)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def body_bool(body: dict[str, object], key: str, default: bool = False) -> bool:
    value = body.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def crop_config(crop_preset: str) -> dict[str, object]:
    if crop_preset == "central_clean":
        return {
            "preset": crop_preset,
            "reference_size_px": {"width": 848, "height": 478},
            "reference_bbox_px": {"x": 120, "y": 190, "width": 660, "height": 280},
            "ffmpeg_crop_expr": (
                "crop=trunc(iw*660/848/2)*2:"
                "trunc(ih*280/478/2)*2:"
                "trunc(iw*120/848/2)*2:"
                "trunc(ih*190/478/2)*2"
            ),
        }
    return {"preset": crop_preset}


def reconstruction_config(
    state: "ToolState",
    body: dict[str, object],
    *,
    frame_count: int | None = None,
    candidate_frames: int | None = None,
    effective_sample_fps: float | None = None,
    ffmpeg_filter_expr: str | None = None,
    clahe: dict[str, object] | None = None,
    adaptive_fps: dict[str, object] | None = None,
    exclusion_masks: dict[str, object] | None = None,
) -> dict[str, object]:
    sample_fps = float(body.get("sample_fps", 0) or 0)
    max_vggt_frames = int(body.get("max_vggt_frames", 0) or 0)
    upload_video_fps = min(float(effective_sample_fps or sample_fps or 2.0), 2.0)
    vggt_space = str(body.get("vggt_space") or state.vggt_space)
    vggt_backend = str(body.get("vggt_backend") or state.vggt_backend)
    vggt_upload_mode = str(body.get("vggt_upload_mode", "video"))
    vggt_timeout = int(body.get("vggt_timeout", 900) or 900)
    conf_thres = float(body.get("vggt_conf_thres", 50.0) or 50.0)
    max_points_k = float(body.get("vggt_max_points_k", 1000.0) or 1000.0)
    mask_sky = body_bool(body, "vggt_mask_sky", False)
    # If exclusion masks or client-side skyseg paint pixels black, tell VGGT to
    # drop the black background.
    mask_black_bg = bool(body.get("exclusion_masks")) or body_bool(body, "client_sky_seg", False)
    max_frames_arg = max_vggt_frames or frame_count or 0
    crop_preset = str(body.get("crop_preset", "central_clean"))
    width = int(body.get("width", 960) or 0)

    return {
        "preprocess": {
            "sample_fps_requested": sample_fps,
            "sample_fps_effective": effective_sample_fps,
            "target_frames": int(body.get("target_frames", 36) or 0),
            "frames_used": frame_count,
            "candidate_frames": candidate_frames,
            "frame_window": str(body.get("frame_window", "all")),
            "max_vggt_frames": max_vggt_frames,
            "output_width_px": width,
            "crop": crop_config(crop_preset),
            "ffmpeg_filter": ffmpeg_filter_expr,
            "clahe": clahe,
            "adaptive_fps": adaptive_fps,
            "exclusion_masks": exclusion_masks,
        },
        "vggt": {
            "model": "facebook/VGGT-Omega" if vggt_backend == "omega" else "facebook/VGGT-1B",
            "space": vggt_space,
            "backend": vggt_backend,
            "upload_mode": vggt_upload_mode,
            "upload_video_fps": upload_video_fps,
            "max_frames_arg": max_frames_arg,
            "conf_thres": conf_thres,
            "max_points_k": max_points_k,
            "mask_sky": mask_sky,
            "mask_black_bg": mask_black_bg,
            "timeout_s": vggt_timeout,
            "refresh": body_bool(body, "refresh_vggt", True),
        },
        "camera_views": {
            "render": body_bool(body, "render_camera_views", True),
            "focal_px": float(body.get("focal_px", 812) or 812),
            "view": "full",
            "splat": int(body.get("splat", 1) or 1),
        },
        "scale": {
            "default_scale_m_per_unit": float(body.get("default_scale_m_per_unit", state.default_scale)),
        },
    }


@dataclass
class Job:
    id: str
    scene_id: str
    status: str = "queued"
    step: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    logs: list[str] = field(default_factory=list)
    error: str = ""
    viewer_url: str = ""
    video_file: str = ""
    scene_rel_dir: str = ""
    request: dict[str, object] = field(default_factory=dict)
    on_update: Callable[["Job"], None] | None = field(default=None, repr=False, compare=False)

    def log(self, message: str) -> None:
        self.logs.append(message)
        self.logs = self.logs[-400:]
        self.updated_at = time.time()
        if self.on_update:
            self.on_update(self)

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "scene_id": self.scene_id,
            "status": self.status,
            "step": self.step,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "viewer_url": self.viewer_url,
            "video_file": self.video_file,
            "scene_rel_dir": self.scene_rel_dir,
            "logs": self.logs,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "Job":
        logs = data.get("logs") if isinstance(data.get("logs"), list) else []
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            scene_id=str(data.get("scene_id") or ""),
            status=str(data.get("status") or "queued"),
            step=str(data.get("step") or "queued"),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            logs=[str(line) for line in logs][-400:],
            error=str(data.get("error") or ""),
            viewer_url=str(data.get("viewer_url") or ""),
            video_file=str(data.get("video_file") or ""),
            scene_rel_dir=str(data.get("scene_rel_dir") or ""),
            request=data.get("request") if isinstance(data.get("request"), dict) else {},
        )


class ToolState:
    def __init__(self, args: argparse.Namespace):
        self.out_dir = args.out_dir
        self.scenes_dir = args.out_dir / "scenes"
        self.jobs_dir = self.scenes_dir / ".jobs"
        self.annotations_dir = ROOT / "annotations"
        self.video_cache_dir = args.video_cache_dir
        self.python = args.python
        self.vggt_python = args.vggt_python
        self.vggt_space = args.vggt_space
        self.vggt_backend = args.vggt_backend
        self.default_scale = args.default_scale_m_per_unit
        self.asset_root = str(getattr(args, "asset_root", "") or "").strip().rstrip("/")
        self.app_base = str(getattr(args, "app_base", "") or "").strip().rstrip("/")
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.video_cache_dir.mkdir(parents=True, exist_ok=True)
        self.load_jobs()

    def job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{slugify(job_id)}.json"

    def attach_job(self, job: Job) -> Job:
        job.on_update = self.save_job
        return job

    def save_job(self, job: Job) -> None:
        payload = {**job.to_json(), "request": job.request}
        path = self.job_path(job.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)

    def load_jobs(self) -> None:
        for path in sorted(self.jobs_dir.glob("*.json")):
            job = Job.from_json(read_json_if_exists(path))
            self.attach_job(job)
            if job.status in {"queued", "running"}:
                job.status = "stale"
                job.step = "server_restarted"
                job.log("[job] server restarted before this job reported completion")
                if job.scene_rel_dir:
                    update_scene_manifest(
                        self.scenes_dir / job.scene_rel_dir,
                        {
                            "job_status": job.status,
                            "job_step": job.step,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                    )
            self.jobs[job.id] = job

    def register_job(self, job: Job) -> None:
        self.attach_job(job)
        self.jobs[job.id] = job
        self.save_job(job)

    def active_job_for_scene(self, scene_id: str) -> Job | None:
        active = [
            job
            for job in self.jobs.values()
            if job.scene_id == scene_id and job.status in {"queued", "running"}
        ]
        if not active:
            return None
        return max(active, key=lambda item: item.updated_at)


def read_json_body(handler: SimpleHTTPRequestHandler) -> dict[str, object]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


def write_json(handler: SimpleHTTPRequestHandler, payload: object, status: int = 200) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def write_error(handler: SimpleHTTPRequestHandler, message: str, status: int = 400) -> None:
    write_json(handler, {"error": message}, status=status)


def read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def scene_viewer_dir(scenes_root: Path, path: str) -> Path | None:
    match = SCENE_VIEWER_INDEX_RE.match(path)
    if not match:
        return None
    viewer_dir = ensure_child(scenes_root, match.group(1), "viewer")
    if not (viewer_dir / "scene_meta.json").exists():
        return None
    return viewer_dir


def scene_has_viewer(scene_dir: Path) -> bool:
    return (scene_dir / "viewer" / "scene_meta.json").exists()


def scene_viewer_url(scenes_root: Path, scene_dir: Path, app_base: str = "") -> str:
    rel_path = scene_dir.relative_to(scenes_root).as_posix()
    return scene_viewer_page_url(rel_path, app_base)


def read_generic_viewer_html(scene_base: str, app_base: str = "", api_base: str = "") -> bytes:
    html = GENERIC_VIEWER_INDEX.read_text()
    scene_base = scene_base if scene_base.endswith("/") else f"{scene_base}/"
    return (
        html.replace("__SCENE_BASE__", scene_base)
        .replace("__APP_BASE__", app_base)
        .replace("__API_BASE__", api_base)
        .encode()
    )


def scene_summary(scene_dir: Path, scenes_root: Path, app_base: str = "") -> dict[str, object] | None:
    manifest = read_json_if_exists(scene_dir / "scene_manifest.json")
    metadata = read_json_if_exists(scene_dir / "metadata.json")
    state = read_json_if_exists(scene_dir / "scene_state.json")
    has_viewer = scene_has_viewer(scene_dir)
    if not has_viewer and not manifest:
        return None
    scene_id = str(manifest.get("scene_id") or metadata.get("scene_id") or scene_dir.name)
    rel_path = scene_dir.relative_to(scenes_root).as_posix()

    selected_segments = manifest.get("selected_segments") or metadata.get("selected_segments") or []
    if not isinstance(selected_segments, list):
        selected_segments = []
    segment_ids = [
        str(seg.get("segment_id"))
        for seg in selected_segments
        if isinstance(seg, dict) and seg.get("segment_id")
    ]
    is_attack = any(bool(seg.get("is_attack")) for seg in selected_segments if isinstance(seg, dict))
    title = str(manifest.get("description") or metadata.get("description") or scene_id)
    if segment_ids:
        title = f"{title} ({', '.join(segment_ids)})"

    mtimes = [
        p.stat().st_mtime
        for p in [scene_dir, scene_dir / "viewer" / "scene_meta.json", scene_dir / "scene_state.json"]
        if p.exists()
    ]
    updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(mtimes))) if mtimes else ""
    return {
        "scene_id": scene_id,
        "title": title,
        "viewer_url": scene_viewer_url(scenes_root, scene_dir, app_base=app_base) if has_viewer else "",
        "exists": has_viewer,
        "video_file": manifest.get("video_file") or metadata.get("video_file") or "",
        "description": manifest.get("description") or metadata.get("description") or "",
        "date": manifest.get("date") or metadata.get("date") or "",
        "town": manifest.get("town") or metadata.get("town") or "",
        "segment_ids": segment_ids,
        "is_attack": is_attack,
        "frames": metadata.get("frames") or manifest.get("frames") or "",
        "sample_fps": manifest.get("sample_fps") or metadata.get("sample_fps") or "",
        "crop_preset": manifest.get("crop_preset") or metadata.get("crop_preset") or "",
        "target_frames": manifest.get("target_frames", metadata.get("target_frames", "")),
        "scale_m_per_unit": state.get("scale_m_per_unit") or manifest.get("default_scale_m_per_unit") or "",
        "scale_saved": bool(state.get("saved")),
        "job_id": manifest.get("job_id") or "",
        "job_status": manifest.get("job_status") or "",
        "job_step": manifest.get("job_step") or "",
        "error": manifest.get("error") or "",
        "created_at": manifest.get("created_at") or "",
        "updated_at": updated_at,
        "path": rel_path,
    }


def list_scene_summaries(scenes_root: Path, app_base: str = "") -> list[dict[str, object]]:
    scenes: list[dict[str, object]] = []
    if not scenes_root.exists():
        return []
    candidates = {p.parent for p in scenes_root.rglob("scene_manifest.json")}
    candidates.update(p.parent for p in scenes_root.rglob("viewer/scene_meta.json"))
    candidates.update(p.parent.parent for p in scenes_root.rglob("viewer/index.html"))
    for scene_dir in sorted(candidates, key=lambda p: p.relative_to(scenes_root).as_posix()):
        summary = scene_summary(scene_dir, scenes_root, app_base=app_base)
        if summary is not None:
            scenes.append(summary)
    scenes.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return scenes


def find_scene_dir(scenes_root: Path, scene_id: str) -> Path | None:
    direct = scenes_root / scene_id
    if direct.exists():
        return direct
    for summary in list_scene_summaries(scenes_root, app_base=""):
        if summary.get("scene_id") == scene_id:
            path = summary.get("path")
            if path:
                return ensure_child(scenes_root, str(path))
    return None


class FPVRequestHandler(SimpleHTTPRequestHandler):
    server_version = "FPVTool/0.1"

    @property
    def state(self) -> ToolState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[fpv-server] " + fmt % args + "\n")

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                write_json(self, {"ok": True, "out_dir": str(self.state.out_dir)})
            elif path == "/api/scenes":
                write_json(self, {"scenes": list_scene_summaries(self.state.scenes_dir, self.state.app_base)})
            elif path == "/api/video-status":
                self.handle_video_status()
            elif path == "/api/annotations":
                self.handle_annotations(parse_qs(parsed.query))
            elif path == "/api/scenes/status":
                self.handle_scene_status(parse_qs(parsed.query))
            elif path == "/api/jobs":
                self.handle_jobs(parse_qs(parsed.query))
            elif path.startswith("/api/jobs/"):
                self.handle_job(path)
            elif path.startswith("/api/scenes/") and path.endswith("/state"):
                self.handle_get_scene_state(path)
            else:
                self.serve_static(path)
        except Exception as exc:
            write_error(self, str(exc), status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/reconstruct":
                self.handle_reconstruct()
            elif path == "/api/annotations":
                self.handle_save_annotation()
            elif path == "/api/scenes/import":
                self.handle_import_scene()
            elif path.startswith("/api/scenes/") and path.endswith("/state"):
                self.handle_save_scene_state(path)
            else:
                write_error(self, "unknown endpoint", status=404)
        except Exception as exc:
            write_error(self, str(exc), status=500)

    def handle_scene_status(self, query: dict[str, list[str]]) -> None:
        video_file = query.get("video_file", [""])[0]
        segment_ids = [s for s in query.get("segments", [""])[0].split(",") if s]
        sid = scene_id_for(video_file, segment_ids)
        scene_dir = find_scene_dir(self.state.scenes_dir, sid)
        viewer_meta = scene_dir / "viewer" / "scene_meta.json" if scene_dir else None
        state_path = scene_dir / "scene_state.json" if scene_dir else None
        summary = scene_summary(scene_dir, self.state.scenes_dir, self.state.app_base) if scene_dir else None
        payload = {
            "scene_id": sid,
            "exists": bool(viewer_meta and viewer_meta.exists()),
            "viewer_url": str(summary.get("viewer_url")) if summary else "",
            "state": json.loads(state_path.read_text()) if state_path and state_path.exists() else {},
        }
        write_json(self, payload)

    def handle_annotations(self, query: dict[str, list[str]]) -> None:
        video_file = query.get("video_file", [""])[0]
        if not video_file:
            write_error(self, "video_file is required", status=400)
            return
        for path in sorted(self.state.annotations_dir.glob("*_annotations.json")):
            data = json.loads(path.read_text())
            if data.get("video_file") == video_file or str(data.get("video_url", "")).endswith("/" + video_file):
                write_json(self, {"found": True, "path": str(path), "annotation": data})
                return
        write_json(self, {"found": False, "annotation": None})

    def handle_save_annotation(self) -> None:
        body = read_json_body(self)
        video_file = str(body.get("video_file") or Path(str(body.get("video_url", ""))).name)
        if not video_file:
            write_error(self, "video_file is required", status=400)
            return
        segments = body.get("segments")
        if not isinstance(segments, list):
            write_error(self, "segments must be a list", status=400)
            return
        payload = {
            **body,
            "video_file": video_file,
            "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = self.state.annotations_dir / f"{slugify(video_file)}_annotations.json"
        path.write_text(json.dumps(payload, indent=2))
        write_json(self, {"saved": True, "path": str(path), "annotation": payload})

    def handle_import_scene(self) -> None:
        body = read_json_body(self)
        source_raw = str(body.get("source_path") or "").strip()
        if not source_raw:
            write_error(self, "source_path is required", status=400)
            return
        source = Path(source_raw).expanduser().resolve()
        if not source.exists() or not source.is_dir():
            write_error(self, f"folder not found: {source}", status=404)
            return
        if source.name == "viewer" and ((source / "scene_meta.json").exists() or (source / "index.html").exists()):
            source = source.parent
        is_scene_folder = (source / "viewer" / "scene_meta.json").exists()
        is_standalone_viewer = (source / "scene_meta.json").exists()
        if not is_scene_folder and not is_standalone_viewer:
            write_error(self, "folder must contain viewer/scene_meta.json or scene_meta.json", status=400)
            return

        manifest = read_json_if_exists(source / "scene_manifest.json")
        scene_meta = read_json_if_exists((source / "viewer" / "scene_meta.json") if is_scene_folder else (source / "scene_meta.json"))
        scene_id = str(manifest.get("scene_id") or scene_meta.get("scene_id") or source.name)
        video_file = str(manifest.get("video_file") or scene_meta.get("video_file") or "imported")
        dest = ensure_child(self.state.scenes_dir, scene_rel_dir_for(video_file, scene_id))
        if source == dest.resolve():
            summary = scene_summary(dest, self.state.scenes_dir, self.state.app_base)
            write_json(self, {"imported": False, "scene": summary})
            return
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if is_scene_folder:
            shutil.copytree(source, dest)
        else:
            (dest / "viewer").mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, dest / "viewer", dirs_exist_ok=True)
            (dest / "scene_manifest.json").write_text(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "video_file": video_file,
                        "description": scene_meta.get("title") or scene_id,
                        "imported_from": str(source),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=2,
                )
            )
        summary = scene_summary(dest, self.state.scenes_dir, self.state.app_base)
        write_json(self, {"imported": True, "path": str(dest), "scene": summary})

    def handle_video_status(self) -> None:
        videos: dict[str, dict[str, object]] = {}
        for path in sorted(self.state.annotations_dir.glob("*_annotations.json")):
            data = read_json_if_exists(path)
            video_file = str(data.get("video_file") or Path(str(data.get("video_url", ""))).name)
            if not video_file:
                continue
            videos.setdefault(video_file, {})
            videos[video_file].update({"annotated": True, "annotation_path": str(path)})
        for scene in list_scene_summaries(self.state.scenes_dir, self.state.app_base):
            if not scene.get("exists") or not scene.get("viewer_url"):
                continue
            video_file = str(scene.get("video_file") or "")
            if not video_file:
                continue
            videos.setdefault(video_file, {})
            current = videos[video_file]
            current["has_scene"] = True
            current["scene_count"] = int(current.get("scene_count", 0) or 0) + 1
            current["latest_scene_id"] = scene.get("scene_id")
            current["latest_scene_url"] = scene.get("viewer_url")
        for status in videos.values():
            status.setdefault("annotated", False)
            status.setdefault("has_scene", False)
        write_json(self, {"videos": videos})

    def handle_jobs(self, query: dict[str, list[str]]) -> None:
        scene_id = query.get("scene_id", [""])[0]
        video_file = query.get("video_file", [""])[0]
        status = query.get("status", [""])[0]
        with self.state.lock:
            jobs = list(self.state.jobs.values())
        if scene_id:
            jobs = [job for job in jobs if job.scene_id == scene_id]
        if video_file:
            jobs = [job for job in jobs if job.video_file == video_file]
        if status:
            statuses = {item.strip() for item in status.split(",") if item.strip()}
            jobs = [job for job in jobs if job.status in statuses]
        jobs.sort(key=lambda job: job.updated_at, reverse=True)
        write_json(self, {"jobs": [job.to_json() for job in jobs]})

    def handle_job(self, path: str) -> None:
        job_id = path.rstrip("/").split("/")[-1]
        with self.state.lock:
            job = self.state.jobs.get(job_id)
        if not job:
            saved = read_json_if_exists(self.state.job_path(job_id))
            if saved:
                job = self.state.attach_job(Job.from_json(saved))
                with self.state.lock:
                    self.state.jobs[job.id] = job
        if not job:
            write_error(self, "job not found", status=404)
            return
        write_json(self, job.to_json())

    def handle_get_scene_state(self, path: str) -> None:
        scene_id = unquote(path.split("/")[3])
        scene_dir = find_scene_dir(self.state.scenes_dir, scene_id) or ensure_child(self.state.scenes_dir, scene_id)
        state_path = scene_dir / "scene_state.json"
        if not state_path.exists():
            write_json(self, {"scale_m_per_unit": self.state.default_scale, "saved": False})
            return
        write_json(self, json.loads(state_path.read_text()))

    def handle_save_scene_state(self, path: str) -> None:
        scene_id = unquote(path.split("/")[3])
        body = read_json_body(self)
        scene_dir = find_scene_dir(self.state.scenes_dir, scene_id) or ensure_child(self.state.scenes_dir, scene_id)
        state_path = scene_dir / "scene_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(state_path.read_text()) if state_path.exists() else {}
        scale = float(body.get("scale_m_per_unit", 0) or 0)
        if scale <= 0:
            write_error(self, "scale_m_per_unit must be positive", status=400)
            return
        merged = {
            **existing,
            **body,
            "scale_m_per_unit": scale,
            "saved": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        state_path.write_text(json.dumps(merged, indent=2))
        write_json(self, merged)

    def handle_reconstruct(self) -> None:
        body = read_json_body(self)
        video_file = str(body.get("video_file") or Path(str(body.get("video_url", ""))).name)
        segments = body.get("segments") or []
        if not isinstance(segments, list) or not segments:
            write_error(self, "segments are required", status=400)
            return
        segment_ids = [str(seg.get("segment_id") or f"seg{i + 1:02d}") for i, seg in enumerate(segments) if isinstance(seg, dict)]
        scene_id = scene_id_for(video_file, segment_ids)
        scene_rel_dir = scene_rel_dir_for(video_file, scene_id)
        with self.state.lock:
            existing = self.state.active_job_for_scene(scene_id)
            if existing:
                write_json(self, {"job_id": existing.id, "scene_id": scene_id, "viewer_url": existing.viewer_url, "reused": True}, status=202)
                return
            job = Job(
                id=str(uuid.uuid4()),
                scene_id=scene_id,
                viewer_url=f"/scenes/{scene_rel_dir}/viewer/index.html",
                video_file=video_file,
                scene_rel_dir=scene_rel_dir,
                request=body,
            )
            self.state.register_job(job)
        thread = threading.Thread(target=reconstruct_scene, args=(self.state, job, body), daemon=True)
        thread.start()
        write_json(self, {"job_id": job.id, "scene_id": scene_id, "viewer_url": job.viewer_url}, status=202)

    def serve_generic_scene_viewer(self, viewer_dir: Path) -> None:
        rel_scene = viewer_dir.parent.relative_to(self.state.scenes_dir).as_posix()
        scene_base = scene_data_base(rel_scene, self.state.asset_root)
        data = read_generic_viewer_html(scene_base, self.state.app_base, "")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def serve_static(self, path: str) -> None:
        viewer_dir = scene_viewer_dir(self.state.scenes_dir, path)
        if viewer_dir is not None:
            self.serve_generic_scene_viewer(viewer_dir)
            return
        if path in {"", "/"}:
            target = ROOT / "tools" / "annotator.html"
        elif path in {"/scenes", "/scenes/"}:
            target = ROOT / "tools" / "scene_browser.html"
        elif path.startswith("/tools/"):
            target = ensure_child(ROOT, path.lstrip("/"))
        elif path.startswith("/annotations/"):
            target = ensure_child(ROOT, path.lstrip("/"))
        elif path.startswith("/scenes/"):
            parts = [unquote(p) for p in path.strip("/").split("/")]
            target = ensure_child(self.state.scenes_dir, *parts[1:])
        else:
            target = ensure_child(ROOT, path.lstrip("/"))
        if not target.exists() or target.is_dir():
            write_error(self, "not found", status=404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def download_video(video_url: str, video_file: str, cache_dir: Path, job: Job) -> Path:
    out = cache_dir / video_file
    if out.exists() and out.stat().st_size > 0:
        job.log(f"[video] using cached {out}")
        return out
    tmp = out.with_suffix(out.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    run_command(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--connect-timeout",
            "20",
            "--max-time",
            "900",
            "-o",
            str(tmp),
            video_url,
        ],
        job,
        timeout=930,
    )
    tmp.replace(out)
    return out


# VGGT-Omega preprocesses inputs to ~368x720 (HxW); this is the target aspect
# (W/H) for the "full_frame" crop, which keeps the whole frame minus the letterbox
# needed to hit that aspect (paired with exclusion masks instead of a tight crop).
MODEL_INPUT_ASPECT = 720.0 / 368.0


def ffmpeg_filter(crop_preset: str, width: int, sample_fps: float) -> str:
    parts = [f"fps={sample_fps:g}"]
    if crop_preset == "central_clean":
        parts.append(
            "crop=trunc(iw*660/848/2)*2:"
            "trunc(ih*280/478/2)*2:"
            "trunc(iw*120/848/2)*2:"
            "trunc(ih*190/478/2)*2"
        )
    elif crop_preset == "full_frame":
        # Central crop to the model's input aspect, keeping as much of the frame
        # as possible (crop only one dimension to hit MODEL_INPUT_ASPECT).
        parts.append(
            f"crop=trunc(min(iw\\,ih*{MODEL_INPUT_ASPECT:.6f})/2)*2:"
            f"trunc(min(ih\\,iw/{MODEL_INPUT_ASPECT:.6f})/2)*2"
        )
    if width:
        parts.append(f"scale={width}:-2")
    return ",".join(parts)


def evenly_sample(items: list[dict[str, object]], target: int) -> list[dict[str, object]]:
    if target <= 0 or len(items) <= target:
        return items
    indexes = [round(i * (len(items) - 1) / max(target - 1, 1)) for i in range(target)]
    return [items[i] for i in indexes]


def sequence_clahe_lut(frame_paths: list[Path], clip_limit: float, sample: int = 48):
    """Build ONE contrast-limited luma equalization LUT from the whole sequence.

    Uniform for the sequence (not per-frame) so brightening a dark clip does not
    flicker frame-to-frame. Returns a 256-entry uint8 LUT for the Y channel, or
    None if it can't be built.
    """
    import cv2
    import numpy as np

    hist = np.zeros(256, dtype=np.float64)
    step = max(1, len(frame_paths) // max(1, sample))
    used = 0
    for path in frame_paths[::step]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        hist += cv2.calcHist([y], [0], None, [256], [0, 256]).ravel()
        used += 1
    if used == 0 or hist.sum() == 0:
        return None
    total = hist.sum()
    clip = max(1.0, clip_limit) * total / 256.0  # contrast limit like CLAHE
    excess = float(np.maximum(hist - clip, 0.0).sum())
    hist = np.minimum(hist, clip) + excess / 256.0
    cdf = np.cumsum(hist)
    lo, hi = float(cdf.min()), float(cdf.max())
    if hi <= lo:
        return None
    lut = np.round((cdf - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return lut


def apply_luma_lut(path: Path, lut) -> None:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = lut[ycrcb[:, :, 0]]
    cv2.imwrite(str(path), cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR))


def select_adaptive_frames(candidates: list[dict[str, object]], target_frames: int, job: "Job"):
    """Motion-aware keyframing.

    Keeps more frames where inter-frame motion is high (the fast terminal approach,
    where consecutive-frame overlap drops) and fewer during slow cruise, so VGGT
    sees a roughly constant baseline between adjacent frames. Within each window it
    keeps the *sharpest* (least motion-blurred) frame. Real per-frame timestamps
    are preserved because we only choose a subset of the densely-sampled candidates.
    """
    n = len(candidates)
    if target_frames <= 0 or n <= target_frames:
        return candidates
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - cv2 optional
        job.log(f"[frames] adaptive fps unavailable ({exc}); falling back to even sampling")
        return evenly_sample(candidates, target_frames)

    width = 256
    grays: list = []
    sharp = np.zeros(n, dtype=np.float64)
    for i, item in enumerate(candidates):
        img = cv2.imread(str(item["src"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            grays.append(None)
            continue
        h, w = img.shape[:2]
        g = cv2.resize(img, (width, max(1, round(h * width / w))))
        grays.append(g)
        sharp[i] = cv2.Laplacian(g, cv2.CV_64F).var()

    motion = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        a, b = grays[i - 1], grays[i]
        if a is None or b is None or a.shape != b.shape:
            continue
        motion[i] = float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))

    # Partition candidates into exactly target_frames disjoint contiguous bins,
    # weighted by cumulative motion (bins are narrower in index where motion is
    # high => more keyframes through the fast approach). Each bin is guaranteed
    # non-empty, so we always emit exactly target_frames frames. Within each bin
    # keep the sharpest (least motion-blurred) frame.
    cum = np.cumsum(motion)
    total = float(cum[-1])
    edges = [0]
    for i in range(1, target_frames):
        e = int(np.searchsorted(cum, total * i / target_frames)) if total > 0 else round(i * n / target_frames)
        e = max(e, edges[-1] + 1)                 # at least one frame per bin
        e = min(e, n - (target_frames - i))        # leave room for remaining bins
        edges.append(e)
    edges.append(n)

    chosen: list[int] = []
    for i in range(target_frames):
        lo, hi = edges[i], edges[i + 1]
        chosen.append(lo + int(np.argmax(sharp[lo:hi])))
    job.log(f"[frames] adaptive: {len(chosen)} of {n} frames (motion-weighted bins, sharpest-in-bin)")
    return [candidates[i] for i in chosen]


def probe_dimensions(video_path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    w, h = proc.stdout.strip().split("x")[:2]
    return int(w), int(h)


def probe_source_fps(video_path: Path) -> float:
    """Average source frame rate (fps) from the container, or 0.0 if unknown."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate,r_frame_rate", "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        for line in proc.stdout.split():
            line = line.strip()
            if not line or line == "0/0":
                continue
            if "/" in line:
                num, den = line.split("/", 1)
                den_f = float(den)
                if den_f:
                    return float(num) / den_f
            else:
                return float(line)
    except Exception:  # pragma: no cover - ffprobe optional
        pass
    return 0.0


def crop_rect_norm(crop_preset: str, iw: int, ih: int) -> tuple[float, float, float, float]:
    """Crop rectangle in normalized original-frame coords (x0,y0,x1,y1)."""
    if crop_preset == "central_clean":
        return (120 / 848, 190 / 478, (120 + 660) / 848, (190 + 280) / 478)
    if crop_preset == "full_frame":
        aspect = MODEL_INPUT_ASPECT
        src = iw / max(1, ih)
        if src >= aspect:  # source wider -> crop width
            w = aspect / src
            return ((1 - w) / 2, 0.0, (1 + w) / 2, 1.0)
        h = src / aspect  # source taller -> crop height
        return (0.0, (1 - h) / 2, 1.0, (1 + h) / 2)
    return (0.0, 0.0, 1.0, 1.0)


def paint_exclusion_masks(path: Path, masks: list[dict[str, object]], crop_rect: tuple[float, float, float, float]) -> bool:
    """Fill excluded regions with black on an already-cropped/scaled frame.

    Masks are stored in normalized original-frame coords; they are mapped through
    the active crop rectangle onto the output frame (out-of-crop parts clip away).
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    if img is None:
        return False
    height, out_w = img.shape[0], img.shape[1]
    x0, y0, x1, y1 = crop_rect
    cw, ch = max(1e-6, x1 - x0), max(1e-6, y1 - y0)

    def to_px(nx: float, ny: float) -> tuple[int, int]:
        return (int(round((nx - x0) / cw * out_w)), int(round((ny - y0) / ch * height)))

    drew = False
    for m in masks:
        t = m.get("type")
        try:
            if t == "rect":
                cv2.rectangle(img, to_px(float(m["x"]), float(m["y"])),
                              to_px(float(m["x"]) + float(m["w"]), float(m["y"]) + float(m["h"])),
                              (0, 0, 0), thickness=-1)
                drew = True
            elif t == "ellipse":
                rx = max(1, int(round(float(m["rx"]) / cw * out_w)))
                ry = max(1, int(round(float(m["ry"]) / ch * height)))
                cv2.ellipse(img, to_px(float(m["cx"]), float(m["cy"])), (rx, ry), 0, 0, 360, (0, 0, 0), -1)
                drew = True
            elif t == "polygon":
                pts = np.array([to_px(float(px), float(py)) for px, py in m["points"]], dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(img, [pts], (0, 0, 0))
                    drew = True
        except (KeyError, TypeError, ValueError):
            continue
    if drew:
        cv2.imwrite(str(path), img)
    return drew


def get_skyseg_session():
    global _skyseg_session
    if _skyseg_session is None:
        import onnxruntime

        _skyseg_session = onnxruntime.InferenceSession(SKYSEG_ONNX, providers=["CPUExecutionProvider"])
    return _skyseg_session


def skyseg_sky_mask(img_bgr, session):
    """Replicates VGGT-Omega's skyseg (visual_util.run_skyseg / segment_sky).

    Returns a boolean mask where True == sky (result_map < 32), at the frame's
    resolution. Run on the CLEAN frame so painted boxes don't bias the prediction.
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    inp = cv2.resize(img_bgr, (320, 320))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp = (inp / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    inp = inp.transpose(2, 0, 1).reshape(1, 3, 320, 320).astype("float32")
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name
    res = np.array(session.run([oname], {iname: inp})).squeeze()
    lo, hi = float(res.min()), float(res.max())
    res = (res - lo) / (hi - lo) if hi > lo else np.zeros_like(res)
    res = cv2.resize((res * 255.0).astype("uint8"), (w, h))
    # VGGT-Omega's apply_sky_mask keeps result_map < 32 (ground) and removes the
    # rest, so sky == result_map >= 32 (verified: ~5%, all in the top strip).
    return res >= 32


def paint_sky_black(path: Path, session) -> float:
    """Paint sky pixels black on a frame; returns the fraction masked as sky."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return 0.0
    mask = skyseg_sky_mask(img, session)
    img[mask] = 0
    cv2.imwrite(str(path), img)
    return float(mask.mean())


def extract_frames(
    video_path: Path,
    scene_dir: Path,
    video_file: str,
    segments: list[dict[str, object]],
    target_frames: int,
    sample_fps: float,
    width: int,
    crop_preset: str,
    max_output_frames: int,
    frame_window: str,
    job: Job,
    clahe: dict[str, object] | None = None,
    adaptive: dict[str, object] | None = None,
    exclusion_masks: list[dict[str, object]] | None = None,
    client_sky_seg: bool = False,
    drop_black_luma: float = 0.0,
) -> dict[str, object]:
    frames_dir = scene_dir / "frames"
    tmp_root = scene_dir / "_extract_tmp"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    total_duration = sum(max(0.0, float(seg["end_s"]) - float(seg["start_s"])) for seg in segments)
    explicit_sample_fps = sample_fps > 0
    adaptive_enabled = bool(adaptive and adaptive.get("enabled"))
    if adaptive_enabled:
        # Sample densely, then keyframe by motion below (bypasses the 10fps clamp).
        sample_fps = max(1.0, float(adaptive.get("base_fps", 24) or 24))
        explicit_sample_fps = True
    elif sample_fps > 0:
        sample_fps = max(0.5, min(10.0, sample_fps))
    else:
        sample_fps = max(0.5, min(10.0, target_frames / max(total_duration, 0.1)))
    vf = ffmpeg_filter(crop_preset, width, sample_fps)
    source_fps = probe_source_fps(video_path)
    frame_dur = 1.0 / source_fps if source_fps > 0 else 0.0
    target_label = "all" if explicit_sample_fps else str(target_frames)
    if max_output_frames > 0:
        target_label = f"{target_label}; {frame_window} {max_output_frames}"
    job.log(f"[frames] {len(segments)} segment(s), target={target_label}, fps={sample_fps:.3f}, vf={vf}")

    candidates: list[dict[str, object]] = []
    sequence_offset_s = 0.0
    for seg_index, seg in enumerate(segments, start=1):
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])
        if end_s <= start_s:
            continue
        # Half-open interval [start_s, end_s). end_s is the timestamp of the next
        # annotation marker (e.g. a pause_start); its frame is the first frame of
        # the pause and must be excluded. Trim exactly one source frame so the last
        # kept frame is the one immediately before the next annotation.
        to_s = max(start_s + frame_dur * 0.5, end_s - frame_dur)
        tmp_dir = tmp_root / f"seg_{seg_index:02d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{start_s:.3f}",
                "-to",
                f"{to_s:.3f}",
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-q:v",
                "2",
                str(tmp_dir / "f_%06d.jpg"),
            ],
            job,
            timeout=900,
        )
        files = sorted(tmp_dir.glob("f_*.jpg"))
        for local_idx, src in enumerate(files):
            segment_time_s = local_idx / sample_fps
            candidates.append(
                {
                    "src": src,
                    "video_file": video_file,
                    "segment_id": seg.get("segment_id") or f"seg{seg_index:02d}",
                    "segment_index": seg_index,
                    "is_attack": bool(seg.get("is_attack")),
                    "video_time_s": start_s + local_idx / sample_fps,
                    "segment_time_s": segment_time_s,
                    "sequence_time_s": sequence_offset_s + segment_time_s,
                }
            )
        sequence_offset_s += end_s - start_s

    # Safety net: drop near-black frames (leftover cut/transition frames) before
    # selection so they can't be chosen as keyframes.
    if drop_black_luma > 0 and candidates:
        try:
            import cv2

            kept = []
            dropped = 0
            for item in candidates:
                img = cv2.imread(str(item["src"]))
                if img is not None and float(img.mean()) < drop_black_luma:
                    dropped += 1
                    continue
                kept.append(item)
            if dropped and kept:
                job.log(f"[frames] dropped {dropped} near-black frame(s)")
                candidates = kept
        except Exception:  # pragma: no cover - cv2 optional
            pass

    adaptive_applied = None
    if adaptive_enabled:
        adaptive_target = int(adaptive.get("target_frames") or target_frames or max_output_frames or VGGT_FRAME_WARN_THRESHOLD)
        selected = select_adaptive_frames(candidates, adaptive_target, job)
        adaptive_applied = {"enabled": True, "base_fps": sample_fps, "target_frames": adaptive_target, "metric": "motion+sharpness"}
        # Dense tail: additionally keep EVERY sampled frame in the last N
        # seconds of flight (the attack run), on top of the adaptive budget.
        # With base_fps at the source rate this is true full-fps coverage.
        tail_dense_s = float(adaptive.get("tail_dense_s", 0) or 0)
        if tail_dense_s > 0 and candidates:
            end_t = float(candidates[-1]["sequence_time_s"])
            dense = [c for c in candidates if float(c["sequence_time_s"]) >= end_t - tail_dense_s]
            seen_srcs = {item["src"] for item in selected}
            added = [c for c in dense if c["src"] not in seen_srcs]
            selected = sorted(selected + added, key=lambda c: (c["segment_index"], c["sequence_time_s"]))
            adaptive_applied["tail_dense_s"] = tail_dense_s
            adaptive_applied["tail_dense_added"] = len(added)
            job.log(
                f"[frames] dense tail: +{len(added)} frame(s) covering the last "
                f"{tail_dense_s:g}s at {sample_fps:g} fps (total {len(selected)})"
            )
    else:
        selected = candidates if explicit_sample_fps else evenly_sample(candidates, target_frames)
    if not adaptive_enabled and max_output_frames > 0 and len(selected) > max_output_frames:
        if frame_window == "first":
            selected = selected[:max_output_frames]
        elif frame_window == "last":
            selected = selected[-max_output_frames:]
        else:
            selected = evenly_sample(selected, max_output_frames)
        job.log(f"[frames] kept {len(selected)} frame(s) from {len(candidates)} candidate frame(s) using {frame_window} window")
    elif len(selected) > VGGT_FRAME_WARN_THRESHOLD:
        job.log(
            f"[frames] WARNING: {len(selected)} frames selected (> {VGGT_FRAME_WARN_THRESHOLD}); "
            f"VGGT reconstruction may be slow or memory-heavy. Pass max_vggt_frames "
            f"(e.g. with frame_window=last) to cap the sequence if needed."
        )
    # Sequence-uniform CLAHE: build one luma LUT from all selected frames, then
    # apply the same mapping to every frame (recovers shadow detail on dark clips).
    clahe_lut = None
    clahe_applied = None
    if clahe and clahe.get("enabled"):
        clip = float(clahe.get("clip_limit", 2.0) or 2.0)
        try:
            clahe_lut = sequence_clahe_lut([Path(item["src"]) for item in selected], clip)
        except Exception as exc:  # pragma: no cover - cv2 optional
            job.log(f"[frames] CLAHE unavailable ({exc}); skipping enhancement")
            clahe_lut = None
        if clahe_lut is not None:
            clahe_applied = {"enabled": True, "clip_limit": clip, "scope": "sequence_uniform"}
            job.log(f"[frames] applying sequence-uniform CLAHE (clip_limit={clip:g})")

    # Exclusion masks: paint fixed junk regions (logos, blur bands, propellers)
    # black so they carry no features; VGGT is then told to mask the black bg.
    mask_rect = None
    masks_applied = None
    if exclusion_masks:
        try:
            iw, ih = probe_dimensions(video_path)
            mask_rect = crop_rect_norm(crop_preset, iw, ih)
        except Exception as exc:  # pragma: no cover
            job.log(f"[frames] could not probe dimensions for masks ({exc}); skipping masks")
            exclusion_masks = None
        else:
            masks_applied = {"count": len(exclusion_masks), "crop_rect_norm": list(mask_rect)}
            job.log(f"[frames] painting {len(exclusion_masks)} exclusion mask(s) black")

    # Client-side skyseg: detect sky on the CLEAN frame (before boxes) and paint it
    # black, so the exclusion boxes never bias the sky prediction.
    sky_session = None
    sky_applied = None
    if client_sky_seg:
        try:
            sky_session = get_skyseg_session()
            sky_applied = {"enabled": True, "model": SKYSEG_ONNX, "order": "sky_then_boxes"}
            job.log("[frames] client-side skyseg on clean frames (sky_then_boxes)")
        except Exception as exc:  # pragma: no cover
            job.log(f"[frames] client-side skyseg unavailable ({exc}); skipping")
            sky_session = None

    out_rows: list[dict[str, object]] = []
    sky_fracs: list[float] = []
    for frame_index, item in enumerate(selected, start=1):
        dst = frames_dir / f"f_{frame_index:06d}.jpg"
        shutil.copy2(Path(item["src"]), dst)
        if clahe_lut is not None:
            apply_luma_lut(dst, clahe_lut)
        if sky_session is not None:
            sky_fracs.append(paint_sky_black(dst, sky_session))
        if exclusion_masks and mask_rect is not None:
            paint_exclusion_masks(dst, exclusion_masks, mask_rect)
        out_rows.append(
            {
                "frame_index": frame_index,
                "file": dst.name,
                "video_file": item["video_file"],
                "segment_id": item["segment_id"],
                "segment_index": item["segment_index"],
                "is_attack": str(bool(item["is_attack"])).lower(),
                "video_time_s": f"{float(item['video_time_s']):.3f}",
                "segment_time_s": f"{float(item['segment_time_s']):.3f}",
                "sequence_time_s": f"{float(item['sequence_time_s']):.3f}",
            }
        )
    if sky_applied is not None and sky_fracs:
        mean_sky = sum(sky_fracs) / len(sky_fracs)
        sky_applied["mean_sky_fraction"] = round(mean_sky, 4)
        job.log(f"[frames] skyseg painted mean {mean_sky * 100:.1f}% of each frame black")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    if not out_rows:
        raise RuntimeError("no frames extracted")
    with (scene_dir / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    input_video = scene_dir / "vggt_input.mp4"
    upload_video_fps = min(sample_fps, 2.0)
    run_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            f"{upload_video_fps:g}",
            "-i",
            str(frames_dir / "f_%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(input_video),
        ],
        job,
        timeout=900,
    )
    job.log(f"[frames] wrote {len(out_rows)} frame(s); encoded VGGT upload video at {upload_video_fps:g} fps")
    return {
        "frame_count": len(out_rows),
        "candidate_frames": len(candidates),
        "sample_fps_effective": sample_fps,
        "ffmpeg_filter": vf,
        "upload_video_fps": upload_video_fps,
        "clahe": clahe_applied,
        "adaptive_fps": adaptive_applied,
        "exclusion_masks": masks_applied,
        "client_sky_seg": sky_applied,
    }


def maybe_render_camera_views(state: ToolState, scene_dir: Path, frame_count: int, body: dict[str, object], job: Job) -> None:
    if not (scene_dir / "point_cloud.npz").exists():
        return
    if not body.get("render_camera_views", True):
        return
    samples = ",".join(str(i) for i in range(1, frame_count + 1))
    out_dir = scene_dir / "camera_views"
    try:
        run_command(
            [
                state.python,
                str(ROOT / "tools" / "render_vggt_reprojection_samples.py"),
                str(scene_dir),
                "--out-dir",
                str(out_dir),
                "--samples",
                samples,
                "--flip-y",
                "--focal-px",
                str(float(body.get("focal_px", 812))),
                "--view",
                "full",
                "--splat",
                str(int(body.get("splat", 1))),
            ],
            job,
            timeout=900,
        )
    except Exception as exc:
        job.log(f"[views] skipped camera-view renders: {exc}")


def reconstruct_scene(state: ToolState, job: Job, body: dict[str, object]) -> None:
    scene_dir: Path | None = None
    try:
        set_job_state(state, job, status="running", step="starting")
        video_url = str(body.get("video_url") or "")
        video_file = str(body.get("video_file") or Path(video_url).name)
        segments = [seg for seg in body.get("segments", []) if isinstance(seg, dict)]
        segment_ids = [str(seg.get("segment_id") or f"seg{i + 1:02d}") for i, seg in enumerate(segments)]
        scene_id = scene_id_for(video_file, segment_ids)
        scene_rel_dir = scene_rel_dir_for(video_file, scene_id)
        scene_dir = state.scenes_dir / scene_rel_dir
        scene_dir.mkdir(parents=True, exist_ok=True)
        job.scene_id = scene_id
        job.viewer_url = f"/scenes/{scene_rel_dir}/viewer/index.html"
        job.video_file = video_file
        job.scene_rel_dir = scene_rel_dir
        recon_subdir = f"scenes/{slugify(video_file)}"
        state.save_job(job)

        manifest = {
            "scene_id": scene_id,
            "job_id": job.id,
            "job_status": job.status,
            "job_step": job.step,
            "video_file": video_file,
            "video_url": video_url,
            "description": body.get("description", ""),
            "date": body.get("date", ""),
            "town": body.get("town", ""),
            "selected_segments": segments,
            "target_frames": int(body.get("target_frames", 36)),
            "sample_fps": float(body.get("sample_fps", 0) or 0),
            "crop_preset": body.get("crop_preset", "central_clean"),
            "max_vggt_frames": int(body.get("max_vggt_frames", 0) or 0),
            "frame_window": body.get("frame_window", "all"),
            "default_scale_m_per_unit": float(body.get("default_scale_m_per_unit", state.default_scale)),
            "model_config": reconstruction_config(state, body),
            "viewer_url": job.viewer_url,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2))
        (scene_dir / "annotation_snapshot.json").write_text(json.dumps(body.get("annotations", []), indent=2))

        set_job_state(state, job, step="download")
        update_scene_manifest(scene_dir, {"job_status": job.status, "job_step": job.step, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        video_path = download_video(video_url, video_file, state.video_cache_dir, job)

        set_job_state(state, job, step="extract_frames")
        update_scene_manifest(scene_dir, {"job_status": job.status, "job_step": job.step, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        frame_info = extract_frames(
            video_path=video_path,
            scene_dir=scene_dir,
            video_file=video_file,
            segments=segments,
            target_frames=int(body.get("target_frames", 36)),
            sample_fps=float(body.get("sample_fps", 0) or 0),
            width=int(body.get("width", 960)),
            crop_preset=str(body.get("crop_preset", "central_clean")),
            max_output_frames=int(body.get("max_vggt_frames", 0) or 0),
            frame_window=str(body.get("frame_window", "all")),
            job=job,
            clahe=body.get("clahe") if isinstance(body.get("clahe"), dict) else None,
            adaptive=body.get("adaptive_fps") if isinstance(body.get("adaptive_fps"), dict) else None,
            exclusion_masks=body.get("exclusion_masks") if isinstance(body.get("exclusion_masks"), list) else None,
            client_sky_seg=body_bool(body, "client_sky_seg", False),
        )
        frame_count = int(frame_info["frame_count"])
        manifest.update(
            {
                "frames": frame_count,
                "model_config": reconstruction_config(
                    state,
                    body,
                    frame_count=frame_count,
                    candidate_frames=int(frame_info["candidate_frames"]),
                    effective_sample_fps=float(frame_info["sample_fps_effective"]),
                    ffmpeg_filter_expr=str(frame_info["ffmpeg_filter"]),
                    clahe=frame_info.get("clahe"),
                    adaptive_fps=frame_info.get("adaptive_fps"),
                    exclusion_masks=frame_info.get("exclusion_masks"),
                ),
            }
        )
        (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2))
        (scene_dir / "metadata.json").write_text(json.dumps(manifest, indent=2))

        if not body.get("skip_vggt", False):
            set_job_state(state, job, step="run_vggt")
            update_scene_manifest(scene_dir, {"job_status": job.status, "job_step": job.step, "frames": frame_count, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            vggt_config = manifest["model_config"]["vggt"]  # type: ignore[index]
            vggt_cmd = [
                state.vggt_python,
                str(ROOT / "tools" / "flight_path_pipeline.py"),
                "--out-dir",
                str(state.out_dir),
                "run-vggt",
                "--recon-subdir",
                recon_subdir,
                "--space",
                str(vggt_config["space"]),
                "--backend",
                str(vggt_config["backend"]),
                "--video-id",
                scene_id,
                "--max-frames",
                str(int(vggt_config["max_frames_arg"])),
                "--conf-thres",
                str(float(vggt_config["conf_thres"])),
                "--max-points-k",
                str(float(vggt_config["max_points_k"])),
                "--vggt-timeout",
                str(int(vggt_config["timeout_s"])),
                "--upload-mode",
                str(vggt_config["upload_mode"]),
                "--video-sample-fps",
                str(float(vggt_config["upload_video_fps"])),
            ]
            if vggt_config["mask_sky"]:
                vggt_cmd.append("--mask-sky")
            if vggt_config.get("mask_black_bg"):
                vggt_cmd.append("--mask-black-bg")
            if vggt_config["refresh"]:
                vggt_cmd.append("--refresh")
            run_command(
                vggt_cmd,
                job,
                timeout=int(vggt_config["timeout_s"]) + 120,
            )

        set_job_state(state, job, step="extract_vggt")
        update_scene_manifest(scene_dir, {"job_status": job.status, "job_step": job.step, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        run_command(
            [
                state.python,
                str(ROOT / "tools" / "flight_path_pipeline.py"),
                "--out-dir",
                str(state.out_dir),
                "extract-vggt",
                "--recon-subdir",
                recon_subdir,
                "--video-id",
                scene_id,
                "--refresh",
            ],
            job,
            timeout=600,
        )

        set_job_state(state, job, step="camera_views")
        update_scene_manifest(scene_dir, {"job_status": job.status, "job_step": job.step, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        maybe_render_camera_views(state, scene_dir, frame_count, body, job)

        default_scale = float(body.get("default_scale_m_per_unit", state.default_scale))
        state_path = scene_dir / "scene_state.json"
        if not state_path.exists():
            state_path.write_text(
                json.dumps(
                    {
                        "scale_m_per_unit": default_scale,
                        "saved": False,
                        "reason": "default_scale",
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=2,
                )
            )

        set_job_state(state, job, step="viewer")
        update_scene_manifest(scene_dir, {"job_status": job.status, "job_step": job.step, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        run_command(
            [
                state.python,
                str(ROOT / "tools" / "create_vggt_threejs_viewer.py"),
                str(scene_dir),
                "--out-dir",
                str(scene_dir / "viewer"),
                "--title",
                f"{manifest['description'] or scene_id} - VGGT scene",
                "--default-scale-m-per-unit",
                str(default_scale),
                "--scene-id",
                scene_id,
                "--state-url",
                f"/api/scenes/{scene_id}/state",
            ],
            job,
            timeout=600,
        )
        set_job_state(state, job, status="done", step="done")
        update_scene_manifest(
            scene_dir,
            {
                "job_status": job.status,
                "job_step": job.step,
                "viewer_url": job.viewer_url,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        job.log(f"[done] {job.viewer_url}")
    except Exception as exc:
        set_job_state(state, job, status="error", error=str(exc))
        update_scene_manifest(
            scene_dir,
            {
                "job_status": job.status,
                "job_step": job.step,
                "error": str(exc),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        job.log(f"[error] {exc}")
    finally:
        job.updated_at = time.time()
        state.save_job(job)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--video-cache-dir", type=Path, default=DEFAULT_VIDEO_CACHE)
    parser.add_argument("--default-scale-m-per-unit", type=float, default=117.6)
    parser.add_argument("--python", default=os.environ.get("FPV_TOOL_PYTHON", sys.executable))
    parser.add_argument("--vggt-python", default=os.environ.get("VGGT_PYTHON", os.environ.get("FPV_TOOL_PYTHON", sys.executable)))
    parser.add_argument("--vggt-space", default=os.environ.get("VGGT_SPACE", "facebook/vggt-omega"))
    parser.add_argument("--vggt-backend", choices=["omega", "classic"], default=os.environ.get("VGGT_BACKEND", "omega"))
    parser.add_argument(
        "--asset-root",
        default=asset_root_from_env(),
        help="CloudFront/S3 root for scene data (default: local). Example: https://d2fioemadmrru3.cloudfront.net",
    )
    parser.add_argument(
        "--app-base",
        default=os.environ.get("FPV_APP_BASE", ""),
        help="Web app path prefix when embedded (e.g. /research/fpv-drone-strikes)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = ToolState(args)

    class Server(ThreadingHTTPServer):
        pass

    server = Server((args.host, args.port), FPVRequestHandler)
    server.state = state  # type: ignore[attr-defined]
    print(f"FPV tool server: http://{args.host}:{args.port}/")
    print(f"Scenes: {state.scenes_dir}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
