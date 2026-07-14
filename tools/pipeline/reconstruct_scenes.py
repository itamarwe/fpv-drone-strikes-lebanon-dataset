#!/usr/bin/env python3
"""End-to-end 3D reconstruction pipeline for FPV attack scenes.

One command runs the whole thing so nobody has to orchestrate it by hand:

  1. make sure the local frame-extraction server (fpv_tool_server) is up,
  2. bring the RunPod Omega pod online (start + launch app.py + wait), see
     ``omega_pod.py``,
  3. run the reconstruction batch for the requested scenes with a named preset,
  4. optionally stop the pod again.

Scenes are given either as explicit annotation JSON paths or as name/date
queries that are resolved against ``annotations/``. When a query matches several
files, manually-authored annotations (no ``auto_generated`` flag) win; auto ones
are only used when that is all that exists.

Examples:
  # Redo a set of scenes with the "clean" preset and shut the pod down after:
  python tools/pipeline/reconstruct_scenes.py --preset clean --stop-pod \
      2026-05-26_anti_drone_platform_barashit 2026-05-01_strike_on_soldiers

  # Explicit annotation files, keep the pod up:
  python tools/pipeline/reconstruct_scenes.py --preset clean \
      annotations/2026-05-03_strike_on_surveillance_camera_annotations.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import omega_pod

ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = ROOT / "annotations"
BATCH_SCRIPT = ROOT / "tools" / "pipeline" / "run_vggt_batch_from_annotations.py"
SERVER_SCRIPT = ROOT / "tools" / "server" / "fpv_tool_server.py"

# The frame-extraction server needs cv2 / onnxruntime; point at whatever
# interpreter has them (override with FPV_SERVER_PYTHON).
SERVER_PYTHON = os.environ.get("FPV_SERVER_PYTHON", "/tmp/fpv-model-benchmark/venv/bin/python")
SERVER_URL = os.environ.get("FPV_SERVER_URL", "http://127.0.0.1:8766")

# Named reconstruction presets -> extra flags for run_vggt_batch_from_annotations.
# "clean" is the current baseline: explicit 260x280 clean-centre crop, no black
# masks, Omega's own sky mask, sequence CLAHE, adaptive 125-frame keyframing, and the 12-second
# interval ending one second before impact: [-13s, -1s].
PRESETS: dict[str, list[str]] = {
    "clean": [
        "--tail-seconds", "12",
        "--exclude-tail-seconds", "1",
        "--crop-preset", "central_clean",
        "--width", "260",
        "--no-masks",
        "--adaptive-fps",
        "--clahe",
        "--vggt-mask-sky",
    ],
    "full-frame-skyseg": [
        "--tail-seconds", "12",
        "--exclude-tail-seconds", "1",
        "--crop-preset", "full_frame",
        "--width", "720",
        "--adaptive-fps",
        "--clahe",
        "--client-sky-seg",
    ],
}


def reconstruction_flags(args: argparse.Namespace) -> list[str]:
    """Apply experiment overrides after the named preset defaults."""
    flags = list(PRESETS[args.preset])
    def set_value(name: str, value: object) -> None:
        if name in flags:
            flags[flags.index(name) + 1] = str(value)
        else:
            flags.extend([name, str(value)])

    if args.tail_seconds is not None:
        set_value("--tail-seconds", args.tail_seconds)
    if args.exclude_tail_seconds is not None:
        set_value("--exclude-tail-seconds", args.exclude_tail_seconds)
    if args.frames is not None:
        set_value("--adaptive-target", args.frames)
    return flags


def log(msg: str) -> None:
    print(f"[reconstruct] {msg}", flush=True)


def server_output_root(url: str) -> Path | None:
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        out_dir = payload.get("out_dir")
        return Path(out_dir).resolve() if payload.get("ok") and out_dir else None
    except Exception:
        return None


def ensure_local_server(url: str, start: bool, out_dir: Path) -> None:
    running_out_dir = server_output_root(url)
    if running_out_dir:
        expected_out_dir = out_dir.resolve()
        if running_out_dir != expected_out_dir:
            raise SystemExit(
                f"[reconstruct] local server at {url} writes to {running_out_dir}, but this batch uses "
                f"{expected_out_dir}. Restart it with --out-dir {expected_out_dir}, or use the matching --out-dir."
            )
        log(f"local server up at {url} with output root {running_out_dir}")
        return
    if not start:
        raise SystemExit(f"[reconstruct] local server not reachable at {url} (start it or drop --no-start-server)")
    if not Path(SERVER_PYTHON).exists():
        raise SystemExit(
            f"[reconstruct] local server not running and interpreter {SERVER_PYTHON} not found; "
            "set FPV_SERVER_PYTHON to a python that has the server deps (cv2, onnxruntime)"
        )
    host = url.split("//", 1)[-1]
    hostname, _, port = host.partition(":")
    port = port or "8766"
    log(f"starting local server: {SERVER_PYTHON} {SERVER_SCRIPT} --host {hostname} --port {port} --out-dir {out_dir}")
    logf = open("/tmp/fpv_tool_server.log", "ab")
    subprocess.Popen(
        [SERVER_PYTHON, str(SERVER_SCRIPT), "--host", hostname, "--port", port, "--out-dir", str(out_dir)],
        stdout=logf, stderr=logf, stdin=subprocess.DEVNULL, cwd=str(ROOT),
    )
    for _ in range(30):
        if server_output_root(url):
            log("local server ready")
            return
        time.sleep(1)
    raise SystemExit("[reconstruct] local server did not come up; see /tmp/fpv_tool_server.log")


def resolve_annotation(query: str) -> Path:
    """Resolve a scene name/date/path to a single annotation file."""
    p = Path(query)
    if p.suffix == ".json" and p.exists():
        return p.resolve()
    stem = query[: -len("_annotations.json")] if query.endswith("_annotations.json") else query
    matches = sorted(ANNOTATION_DIR.glob(f"*{stem}*_annotations.json"))
    if not matches:
        raise SystemExit(f"[reconstruct] no annotation file matches '{query}'")
    if len(matches) == 1:
        return matches[0]

    def is_manual(path: Path) -> bool:
        try:
            return not json.loads(path.read_text()).get("auto_generated")
        except Exception:
            return False

    manual = [m for m in matches if is_manual(m)]
    pool = manual or matches
    if len(pool) == 1:
        return pool[0]
    names = "\n  ".join(m.name for m in pool)
    raise SystemExit(f"[reconstruct] '{query}' is ambiguous, pass an explicit path. Candidates:\n  {names}")


def run_batch(annotations: list[Path], preset_flags: list[str], args: argparse.Namespace, space: str) -> int:
    results_file = args.results_file or (args.out_dir / "scenes" / f".batch_{args.preset}.json")
    cmd = [
        sys.executable, str(BATCH_SCRIPT),
        "--server", args.server,
        "--out-dir", str(args.out_dir),
        "--annotations", *[str(a) for a in annotations],
        "--vggt-space", space,
        "--vggt-backend", "omega",
        "--vggt-upload-mode", "images",
        "--omega-pod-id", args.pod_id,
        "--continue-on-error",
        "--poll-seconds", str(args.poll_seconds),
        "--vggt-timeout", str(args.vggt_timeout),
        "--results-file", str(results_file),
        *preset_flags,
    ]
    if args.use_pod:
        # Each completed Omega request leaves a large Gradio upload directory
        # behind. The batch is sequential, so cleaning here is safe and keeps a
        # long run below the pod's workspace limit.
        cmd.extend(["--omega-cleanup-every", "1"])
    if args.skip_existing:
        cmd.append("--skip-existing")
    log("running batch:\n  " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="+", help="annotation paths or scene name/date queries")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="clean")
    parser.add_argument("--server", default=SERVER_URL)
    parser.add_argument("--out-dir", type=Path, default=ROOT, help="shared scene output root for the server and batch")
    parser.add_argument("--pod-id", default=omega_pod.POD_ID)
    parser.add_argument("--stop-pod", action="store_true", help="stop the pod when the batch finishes")
    parser.add_argument("--skip-existing", action="store_true", help="skip scenes already reconstructed")
    parser.add_argument("--no-start-server", dest="start_server", action="store_false",
                        help="do not auto-start the local frame server")
    parser.add_argument("--no-pod", dest="use_pod", action="store_false",
                        help="assume Omega is already reachable; do not touch the pod")
    parser.add_argument("--ready-timeout", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--vggt-timeout", type=int, default=1800)
    parser.add_argument("--frames", type=int, help="override the preset's adaptive VGGT frame count")
    parser.add_argument("--tail-seconds", type=float, help="override how many seconds to keep before the excluded tail")
    parser.add_argument("--exclude-tail-seconds", type=float, help="override seconds omitted from the end of the selected flight")
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="resolve scenes and print the plan, then exit")
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.tail_seconds is not None and args.tail_seconds <= 0:
        parser.error("--tail-seconds must be greater than zero")
    if args.exclude_tail_seconds is not None and args.exclude_tail_seconds < 0:
        parser.error("--exclude-tail-seconds must be zero or greater")

    annotations = [resolve_annotation(s) for s in args.scenes]
    preset_flags = reconstruction_flags(args)
    log(f"preset={args.preset} flags={' '.join(preset_flags)} scenes:")
    for a in annotations:
        print(f"    {os.path.relpath(a, ROOT)}")
    if args.dry_run:
        return 0

    ensure_local_server(args.server, args.start_server, args.out_dir)

    if args.use_pod:
        space = omega_pod.up(args.pod_id, ready_timeout_s=args.ready_timeout)
    else:
        space = omega_pod.space_url(args.pod_id)
        log(f"skipping pod management; using {space}")

    try:
        rc = run_batch(annotations, preset_flags, args, space)
    finally:
        if args.use_pod and args.stop_pod:
            log("stopping pod")
            omega_pod.pod_stop(args.pod_id)
    return rc


if __name__ == "__main__":
    sys.exit(main())
