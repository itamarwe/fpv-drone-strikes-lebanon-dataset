#!/usr/bin/env python3
"""Deterministic end-to-end driver: re-run the starred scenes through the undistort-to-pinhole pipeline.

One command, resumable, no manual steps in between:

  python3 tools/pipeline/starred_run_batch.py --pod-id <id>            # existing pod
  python3 tools/pipeline/starred_run_batch.py --create-pod --hours 6   # create an A100 pod with cloud-enforced deadlines

Stages (every one checks its own marker and is skipped when already done):
  1. pod       wait for SSH (optionally create the pod with --stop-after/--terminate-after)
  2. setup     VGGT-Omega space + venv on the pod, HF token forwarded (tools/setup_runpod_vggt_omega_server.py)
  3. push      published frames of every scene -> /workspace/starred/<video_id>/images, calibration script
  4. calib     launch ONE background CPU job on the pod that self-calibrates every scene (COLMAP + GLOMAP)
  5. scenes    for each scene in order: wait for its calibration -> fetch cameras.txt ->
               undistort locally -> VGGT-Omega on the pod (slim artifacts) -> fetch -> viewer artifacts
  6. post      metrics, overlay viewers, before/after videos (tools/pipeline/starred_postprocess.py)
  7. pod-down  stop (default) and delete (--delete-pod-when-done) the pod

State: benchmarks/starred_undistort/batch_status.json and batch_log.txt. Scene data: scenes/starred_undistort/<video_id>/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import run_vggt_omega_direct_on_runpod as direct  # noqa: E402

SPEC = ROOT / "benchmarks" / "starred_undistort" / "starred_scenes.json"
STATE = ROOT / "benchmarks" / "starred_undistort" / "batch_status.json"
LOG = ROOT / "benchmarks" / "starred_undistort" / "batch_log.txt"
SCENES = ROOT / "scenes" / "starred_undistort"
REMOTE_ROOT = "/workspace/starred"
POD_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
GPU_ID = "NVIDIA A100-SXM4-80GB"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def log(msg: str) -> None:
    line = f"[{now_utc().strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"scenes": {}, "pod": {}}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE)


def ssh_run(ssh: dict, script: str, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run([*direct.ssh_base(ssh), "bash -s"], input=script, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"remote failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-2000:]}")
    return proc


def scp_from(ssh: dict, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([*direct.scp_base(ssh), f"root@{ssh['ip']}:{remote}", str(local)], check=True, capture_output=True)


def push_dir(ssh: dict, local_dir: Path, remote_dir: str) -> None:
    remote = f"mkdir -p {shlex.quote(remote_dir)} && tar -C {shlex.quote(remote_dir)} -xf - && find {shlex.quote(remote_dir)} -name '._*' -delete"
    tar = subprocess.Popen(["tar", "-C", str(local_dir), "-cf", "-", "."], stdout=subprocess.PIPE, env={**os.environ, "COPYFILE_DISABLE": "1"})
    proc = subprocess.run([*direct.ssh_base(ssh), remote], stdin=tar.stdout, text=True, capture_output=True)
    tar.wait()
    if proc.returncode != 0 or tar.returncode != 0:
        raise RuntimeError(f"push {local_dir} failed: {proc.stderr[-1000:]}")


# ---------------------------------------------------------------- stages

def stage_pod(args, state) -> dict:
    pod_id = args.pod_id or state.get("pod", {}).get("id")
    if not pod_id and args.create_pod:
        stop = (now_utc() + dt.timedelta(hours=args.hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        term = (now_utc() + dt.timedelta(hours=args.hours + 0.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        name = f"fpv-starred-undistort-{now_utc().strftime('%Y%m%d-%H%M')}"
        cmd = ["runpodctl", "pod", "create", "--name", name, "--image", POD_IMAGE, "--gpu-id", GPU_ID, "--gpu-count", "1",
               "--cloud-type", "SECURE", "--container-disk-in-gb", str(args.container_disk_gb), "--volume-in-gb", str(args.volume_gb),
               "--volume-mount-path", "/workspace", "--ports", "22/tcp", "--stop-after", stop, "--terminate-after", term, "-o", "json"]
        log(f"creating pod {name} (stop {stop}, terminate {term})")
        out = subprocess.run(cmd, text=True, capture_output=True)
        if out.returncode != 0:
            raise SystemExit(f"pod create failed: {out.stderr or out.stdout}")
        pod = json.loads(out.stdout)
        pod_id = pod["id"]
        state["pod"] = {"id": pod_id, "name": name, "created_utc": now_utc().isoformat(), "stop_after": stop, "terminate_after": term,
                        "cost_per_hr": pod.get("costPerHr")}
        save_state(state)
        log(f"pod {pod_id} created, costPerHr={pod.get('costPerHr')}")
    if not pod_id:
        raise SystemExit("need --pod-id or --create-pod")
    state.setdefault("pod", {})["id"] = pod_id
    save_state(state)
    ssh = direct.wait_for_ssh(pod_id, args.wait_ssh_s)
    log(f"pod {pod_id} ssh {ssh['ip']}:{ssh['port']}")
    return ssh


def stage_setup(args, state, ssh) -> None:
    probe = ssh_run(ssh, "test -x /workspace/vggt-omega-venv/bin/python && test -d /workspace/vggt-omega-space && echo READY || echo MISSING", check=False)
    if "READY" in probe.stdout and state.get("pod", {}).get("setup_done"):
        log("setup: already done")
        return
    log("setup: installing VGGT-Omega on the pod (forwards the local HF token)")
    r = subprocess.run([sys.executable, str(ROOT / "tools/setup_runpod_vggt_omega_server.py"), state["pod"]["id"], "--wait-url-s", "0"], text=True)
    if r.returncode != 0:
        raise SystemExit("setup failed")
    state["pod"]["setup_done"] = True
    save_state(state)


def stage_push(args, state, scenes, ssh) -> None:
    ssh_run(ssh, f"mkdir -p {REMOTE_ROOT}")
    for s in scenes:
        vid = s["video_id"]
        img = SCENES / vid / "published" / "images"
        n_local = len(list(img.glob("*.jpg")))
        n_remote = ssh_run(ssh, f"find {REMOTE_ROOT}/{vid}/images -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l", check=False).stdout.strip()
        if n_remote.isdigit() and int(n_remote) == n_local and n_local > 0:
            continue
        log(f"push: {vid} ({n_local} frames)")
        push_dir(ssh, img, f"{REMOTE_ROOT}/{vid}/images")
    for name in ("starred_calibrate_remote.sh", "benchmark_remote_job.py"):
        subprocess.run([*direct.scp_base(ssh), str(ROOT / "tools/pipeline" / name), f"root@{ssh['ip']}:{REMOTE_ROOT}/{name}"], check=True, capture_output=True)
    log("push: done")


def stage_calib_launch(args, state, scenes, ssh) -> None:
    st = ssh_run(ssh, f"cat {REMOTE_ROOT}/calib_all.status.json 2>/dev/null || echo NONE", check=False).stdout
    if "running" in st or "succeeded" in st:
        log("calib: job already running/finished on the pod")
        return
    ids = " ".join(shlex.quote(s["video_id"]) for s in scenes)
    script = (f"cd {REMOTE_ROOT} && (setsid nohup python3 benchmark_remote_job.py --status {REMOTE_ROOT}/calib_all.status.json -- "
              f"bash {REMOTE_ROOT}/starred_calibrate_remote.sh {ids} > {REMOTE_ROOT}/calib_all.log 2>&1 &) ; sleep 1; echo launched")
    ssh_run(ssh, script)
    log("calib: background job launched on the pod")


def wait_calib(ssh, vid: str, timeout_s: int) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        out = ssh_run(ssh, f"cat {REMOTE_ROOT}/{vid}/calib/status.json 2>/dev/null; echo; tail -c 300 {REMOTE_ROOT}/calib_all.status.json 2>/dev/null", check=False).stdout
        try:
            st = json.loads(out.strip().split("\n")[0])
        except Exception:
            st = {}
        if st.get("status") == "succeeded":
            return st
        if st.get("status") == "failed":
            return st
        if '"status": "failed"' in out or '"status": "succeeded"' in out.split("\n")[-1] and not st:
            # global job finished without producing this scene
            return {"status": "failed", "note": "calibration job ended before this scene"}
        msg = st.get("note", "pending")
        if msg != last:
            log(f"calib: {vid} {msg}")
            last = msg
        time.sleep(30)
    return {"status": "failed", "note": "timeout"}


def undistort_local(vid: str) -> Path:
    base = SCENES / vid
    cams = base / "calib" / "glomap_fisheye" / "cameras.txt"
    out = base / "pinhole" / "frames"
    marker = out / "undistortion.json"
    n_src = len(list((base / "published" / "images").glob("*.jpg")))
    if marker.exists() and json.loads(marker.read_text()).get("frames") == n_src:
        return out
    r = subprocess.run([sys.executable, str(ROOT / "tools/pipeline/undistort_real_scene.py"), "--cameras", str(cams), "--input", str(base / "published" / "images"), "--output", str(out)],
                       text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"undistort failed: {r.stderr[-800:]}")
    log(f"undistort: {vid} {r.stdout.strip()[-160:]}")
    return out


def run_scene(args, state, s, ssh) -> None:
    vid = s["video_id"]
    base = SCENES / vid
    rec = state["scenes"].setdefault(vid, {})
    if rec.get("status") == "done":
        return
    if rec.get("status") == "failed" and not args.retry_failed:
        return
    t0 = time.time()
    rec.update(status="running", started_utc=now_utc().isoformat())
    save_state(state)
    try:
        # calibration
        calib_dir = base / "calib" / "glomap_fisheye"
        if not (calib_dir / "cameras.txt").exists():
            st = wait_calib(ssh, vid, args.calib_timeout_s)
            if st.get("status") != "succeeded":
                raise RuntimeError(f"calibration failed: {st.get('note')}")
            for name in ("cameras.txt", "images.txt", "analyzer.txt"):
                scp_from(ssh, f"{REMOTE_ROOT}/{vid}/calib/glomap_fisheye/{name}", calib_dir / name)
            scp_from(ssh, f"{REMOTE_ROOT}/{vid}/calib/status.json", base / "calib" / "status.json")
            rec["calibration"] = st
            log(f"calib: {vid} registered {st.get('registered')}/{st.get('images')}")
        # undistort
        frames = undistort_local(vid)
        # VGGT-Omega on the pod
        pin = base / "pinhole"
        if not ((pin / "viewer" / "scene_meta.json").exists() and (pin / "runpod_artifacts" / "predictions.npz").exists()):
            cmd = [sys.executable, str(ROOT / "tools/run_vggt_omega_direct_on_runpod.py"), state["pod"]["id"], str(pin), "--frames-dir", str(frames),
                   "--remote-name", f"{vid}_pinhole", "--max-points-k", str(args.max_points_k), "--artifact-max-points", str(args.artifact_max_points),
                   "--default-scale-m-per-unit", str(s.get("published_scale_m_per_unit") or 117.6), "--title", f"{s['title']} (undistorted to pinhole)",
                   "--slim-predictions"]
            log(f"vggt: {vid} starting ({s['published_frames']} frames)")
            r = subprocess.run(cmd, text=True)
            if r.returncode != 0:
                raise RuntimeError("VGGT-Omega run failed")
            # free pod disk
            ssh_run(ssh, f"rm -rf /workspace/{vid}_pinhole_frames /workspace/vggt-omega-space/demo_outputs/{vid}_pinhole_direct", check=False)
        rec.update(status="done", finished_utc=now_utc().isoformat(), elapsed_s=round(time.time() - t0))
        log(f"scene: {vid} done in {rec['elapsed_s']}s")
    except Exception as exc:  # keep going with the next scene
        rec.update(status="failed", error=str(exc)[-500:], finished_utc=now_utc().isoformat(), elapsed_s=round(time.time() - t0))
        log(f"scene: {vid} FAILED: {exc}")
    save_state(state)


def stage_pod_down(args, state) -> None:
    pod_id = state.get("pod", {}).get("id")
    if not pod_id or args.keep_pod:
        return
    log(f"pod: stopping {pod_id}")
    subprocess.run(["runpodctl", "pod", "stop", pod_id], text=True, capture_output=True)
    if args.delete_pod_when_done:
        log(f"pod: deleting {pod_id}")
        subprocess.run(["runpodctl", "pod", "delete", pod_id], text=True, capture_output=True)
    state["pod"]["down_utc"] = now_utc().isoformat()
    save_state(state)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod-id", default="")
    ap.add_argument("--create-pod", action="store_true", help="create an A100 pod (paid) with cloud-enforced stop/terminate deadlines")
    ap.add_argument("--hours", type=float, default=6.0, help="pod --stop-after horizon when creating (terminate = +30 min)")
    ap.add_argument("--container-disk-gb", type=int, default=60)
    ap.add_argument("--volume-gb", type=int, default=60)
    ap.add_argument("--only", nargs="*", default=None, help="video ids (default: all starred scenes)")
    ap.add_argument("--max-scenes", type=int, default=0, help="stop after N scenes (0 = all)")
    ap.add_argument("--deadline-utc", default="", help="do not start a new scene after this ISO time (default: pod stop_after - 20 min)")
    ap.add_argument("--est-scene-min", type=float, default=9.0, help="estimated minutes per scene for the deadline guard")
    ap.add_argument("--calib-timeout-s", type=int, default=3600)
    ap.add_argument("--wait-ssh-s", type=int, default=900)
    ap.add_argument("--max-points-k", type=float, default=3000.0)
    ap.add_argument("--artifact-max-points", type=int, default=3_000_000)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--skip-post", action="store_true")
    ap.add_argument("--keep-pod", action="store_true", help="leave the pod running when done")
    ap.add_argument("--delete-pod-when-done", action="store_true")
    ap.add_argument("--stages", default="pod,setup,push,calib,scenes,post,pod-down")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    spec = json.loads(SPEC.read_text())
    scenes = [s for s in spec["scenes"] if not args.only or s["video_id"] in args.only]
    state = load_state()
    log(f"batch start: {len(scenes)} scenes, stages {sorted(stages)}")

    ssh = None
    if stages & {"pod", "setup", "push", "calib", "scenes"}:
        ssh = stage_pod(args, state)
    if "setup" in stages:
        stage_setup(args, state, ssh)
    if "push" in stages:
        stage_push(args, state, scenes, ssh)
    if "calib" in stages:
        stage_calib_launch(args, state, scenes, ssh)
    if "scenes" in stages:
        deadline = None
        if args.deadline_utc:
            deadline = dt.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        elif state.get("pod", {}).get("stop_after"):
            deadline = dt.datetime.fromisoformat(state["pod"]["stop_after"].replace("Z", "+00:00")) - dt.timedelta(minutes=20)
        done = 0
        for s in scenes:
            if args.max_scenes and done >= args.max_scenes:
                break
            if deadline and now_utc() + dt.timedelta(minutes=args.est_scene_min) > deadline:
                log(f"deadline guard: not starting {s['video_id']} (deadline {deadline.isoformat()})")
                break
            before = state["scenes"].get(s["video_id"], {}).get("status")
            run_scene(args, state, s, ssh)
            if before != "done" and state["scenes"][s["video_id"]].get("status") == "done":
                done += 1
        n_done = sum(1 for s in scenes if state["scenes"].get(s["video_id"], {}).get("status") == "done")
        n_fail = sum(1 for s in scenes if state["scenes"].get(s["video_id"], {}).get("status") == "failed")
        log(f"scenes: {n_done} done, {n_fail} failed, {len(scenes) - n_done - n_fail} pending")
    if "pod-down" in stages:
        stage_pod_down(args, state)
    if "post" in stages and not args.skip_post:
        done_ids = [s["video_id"] for s in scenes if state["scenes"].get(s["video_id"], {}).get("status") == "done"]
        if done_ids:
            log(f"post: {len(done_ids)} scenes")
            subprocess.run([sys.executable, str(ROOT / "tools/pipeline/starred_postprocess.py"), "--only", *done_ids], text=True)
    log("batch end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
