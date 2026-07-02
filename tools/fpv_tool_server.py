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
DEFAULT_OUT_DIR = ROOT
DEFAULT_VIDEO_CACHE = Path("/tmp/fpv-model-benchmark/videos")


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
        self.default_scale = args.default_scale_m_per_unit
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


def scene_summary(scene_dir: Path, scenes_root: Path) -> dict[str, object] | None:
    manifest = read_json_if_exists(scene_dir / "scene_manifest.json")
    metadata = read_json_if_exists(scene_dir / "metadata.json")
    state = read_json_if_exists(scene_dir / "scene_state.json")
    viewer = scene_dir / "viewer" / "index.html"
    if not viewer.exists() and not manifest:
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

    mtimes = [p.stat().st_mtime for p in [scene_dir, viewer, scene_dir / "scene_state.json"] if p.exists()]
    updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(mtimes))) if mtimes else ""
    return {
        "scene_id": scene_id,
        "title": title,
        "viewer_url": f"/scenes/{rel_path}/viewer/index.html" if viewer.exists() else "",
        "exists": viewer.exists(),
        "video_file": manifest.get("video_file") or metadata.get("video_file") or "",
        "description": manifest.get("description") or metadata.get("description") or "",
        "date": manifest.get("date") or metadata.get("date") or "",
        "town": manifest.get("town") or metadata.get("town") or "",
        "segment_ids": segment_ids,
        "is_attack": is_attack,
        "frames": metadata.get("frames") or manifest.get("frames") or "",
        "sample_fps": manifest.get("sample_fps") or metadata.get("sample_fps") or "",
        "crop_preset": manifest.get("crop_preset") or metadata.get("crop_preset") or "",
        "target_frames": manifest.get("target_frames") or metadata.get("target_frames") or "",
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


def list_scene_summaries(scenes_root: Path) -> list[dict[str, object]]:
    scenes: list[dict[str, object]] = []
    if not scenes_root.exists():
        return []
    candidates = {p.parent for p in scenes_root.rglob("scene_manifest.json")}
    candidates.update(p.parent.parent for p in scenes_root.rglob("viewer/index.html"))
    for scene_dir in sorted(candidates, key=lambda p: p.relative_to(scenes_root).as_posix()):
        summary = scene_summary(scene_dir, scenes_root)
        if summary is not None:
            scenes.append(summary)
    scenes.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return scenes


def find_scene_dir(scenes_root: Path, scene_id: str) -> Path | None:
    direct = scenes_root / scene_id
    if direct.exists():
        return direct
    for summary in list_scene_summaries(scenes_root):
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
                write_json(self, {"scenes": list_scene_summaries(self.state.scenes_dir)})
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
        viewer = scene_dir / "viewer" / "index.html" if scene_dir else None
        state_path = scene_dir / "scene_state.json" if scene_dir else None
        summary = scene_summary(scene_dir, self.state.scenes_dir) if scene_dir else None
        payload = {
            "scene_id": sid,
            "exists": bool(viewer and viewer.exists()),
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
        if source.name == "viewer" and (source / "index.html").exists():
            source = source.parent
        is_scene_folder = (source / "viewer" / "index.html").exists()
        is_standalone_viewer = (source / "index.html").exists() and (source / "scene_meta.json").exists()
        if not is_scene_folder and not is_standalone_viewer:
            write_error(self, "folder must contain viewer/index.html or index.html with scene_meta.json", status=400)
            return

        manifest = read_json_if_exists(source / "scene_manifest.json")
        scene_meta = read_json_if_exists((source / "viewer" / "scene_meta.json") if is_scene_folder else (source / "scene_meta.json"))
        scene_id = str(manifest.get("scene_id") or scene_meta.get("scene_id") or source.name)
        video_file = str(manifest.get("video_file") or scene_meta.get("video_file") or "imported")
        dest = ensure_child(self.state.scenes_dir, scene_rel_dir_for(video_file, scene_id))
        if source == dest.resolve():
            summary = scene_summary(dest, self.state.scenes_dir)
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
        summary = scene_summary(dest, self.state.scenes_dir)
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
        for scene in list_scene_summaries(self.state.scenes_dir):
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

    def serve_static(self, path: str) -> None:
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


def ffmpeg_filter(crop_preset: str, width: int, sample_fps: float) -> str:
    parts = [f"fps={sample_fps:g}"]
    if crop_preset == "central_clean":
        parts.append("crop=trunc(iw*0.84/2)*2:ih:trunc(iw*0.08/2)*2:0")
    if width:
        parts.append(f"scale={width}:-2")
    return ",".join(parts)


def evenly_sample(items: list[dict[str, object]], target: int) -> list[dict[str, object]]:
    if target <= 0 or len(items) <= target:
        return items
    indexes = [round(i * (len(items) - 1) / max(target - 1, 1)) for i in range(target)]
    return [items[i] for i in indexes]


def extract_frames(
    video_path: Path,
    scene_dir: Path,
    video_file: str,
    segments: list[dict[str, object]],
    target_frames: int,
    sample_fps: float,
    width: int,
    crop_preset: str,
    job: Job,
) -> int:
    frames_dir = scene_dir / "frames"
    tmp_root = scene_dir / "_extract_tmp"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    total_duration = sum(max(0.0, float(seg["end_s"]) - float(seg["start_s"])) for seg in segments)
    if sample_fps > 0:
        sample_fps = max(0.5, min(10.0, sample_fps))
    else:
        sample_fps = max(0.5, min(10.0, target_frames / max(total_duration, 0.1)))
    vf = ffmpeg_filter(crop_preset, width, sample_fps)
    job.log(f"[frames] {len(segments)} segment(s), target={target_frames}, fps={sample_fps:.3f}, vf={vf}")

    candidates: list[dict[str, object]] = []
    sequence_offset_s = 0.0
    for seg_index, seg in enumerate(segments, start=1):
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])
        if end_s <= start_s:
            continue
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
                f"{end_s:.3f}",
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

    selected = evenly_sample(candidates, target_frames)
    out_rows: list[dict[str, object]] = []
    for frame_index, item in enumerate(selected, start=1):
        dst = frames_dir / f"f_{frame_index:06d}.jpg"
        shutil.copy2(Path(item["src"]), dst)
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
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    if not out_rows:
        raise RuntimeError("no frames extracted")
    with (scene_dir / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    job.log(f"[frames] wrote {len(out_rows)} frame(s)")
    return len(out_rows)


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
            "default_scale_m_per_unit": float(body.get("default_scale_m_per_unit", state.default_scale)),
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
        frame_count = extract_frames(
            video_path=video_path,
            scene_dir=scene_dir,
            video_file=video_file,
            segments=segments,
            target_frames=int(body.get("target_frames", 36)),
            sample_fps=float(body.get("sample_fps", 0) or 0),
            width=int(body.get("width", 960)),
            crop_preset=str(body.get("crop_preset", "central_clean")),
            job=job,
        )
        (scene_dir / "metadata.json").write_text(json.dumps({**manifest, "frames": frame_count}, indent=2))

        if not body.get("skip_vggt", False):
            set_job_state(state, job, step="run_vggt")
            update_scene_manifest(scene_dir, {"job_status": job.status, "job_step": job.step, "frames": frame_count, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            run_command(
                [
                    state.vggt_python,
                    str(ROOT / "tools" / "flight_path_pipeline.py"),
                    "--out-dir",
                    str(state.out_dir),
                    "run-vggt",
                    "--recon-subdir",
                    recon_subdir,
                    "--video-id",
                    scene_id,
                    "--max-frames",
                    str(int(body.get("max_vggt_frames", frame_count))),
                    "--vggt-timeout",
                    str(int(body.get("vggt_timeout", 900))),
                    *(["--refresh"] if body.get("refresh_vggt", False) else []),
                ],
                job,
                timeout=int(body.get("vggt_timeout", 900)) + 120,
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
