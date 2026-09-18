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
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = __import__("os").environ.get("FPV_UNDISTORT_BATCH", "starred_undistort")  # batch name: benchmarks/<BATCH>, scenes/<BATCH>, reports/<BATCH>
sys.path.insert(0, str(ROOT / "tools"))
import run_vggt_omega_direct_on_runpod as direct  # noqa: E402

SPEC = ROOT / "benchmarks" / BATCH / "starred_scenes.json"
STATE = ROOT / "benchmarks" / BATCH / "batch_status.json"
LOG = ROOT / "benchmarks" / BATCH / "batch_log.txt"
SCENES = ROOT / "scenes" / BATCH
REPORTS = ROOT / "reports" / BATCH
STATUS_HTML = REPORTS / "status.html"
_STATUS_LOCK = threading.Lock()
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
    # network volumes refuse chown: extract with --no-same-owner; strip macOS xattrs on the sending side
    remote = f"mkdir -p {shlex.quote(remote_dir)} && tar --no-same-owner --no-same-permissions --warning=no-unknown-keyword -C {shlex.quote(remote_dir)} -xf - && find {shlex.quote(remote_dir)} -name '._*' -delete"
    tar = subprocess.Popen(["tar", "--no-xattrs", "--no-mac-metadata", "-C", str(local_dir), "-cf", "-", "."], stdout=subprocess.PIPE, env={**os.environ, "COPYFILE_DISABLE": "1"})
    proc = subprocess.run([*direct.ssh_base(ssh), remote], stdin=tar.stdout, text=True, capture_output=True)
    tar.wait()
    if proc.returncode != 0 or tar.returncode != 0:
        raise RuntimeError(f"push {local_dir} failed: {proc.stderr[-1000:]}")


# ---------------------------------------------------------------- status page (regenerated deterministically from state + files)

def write_status_html(state: dict, scenes: list[dict]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pod = state.get("pod", {})
    counts = {"done": 0, "failed": 0, "running": 0, "pending": 0}
    rows = []
    for s in scenes:
        vid = s["video_id"]; rec = state["scenes"].get(vid, {}); st = rec.get("status", "pending"); counts[st] = counts.get(st, 0) + 1
        m_path = SCENES / vid / "metrics.json"; m = json.loads(m_path.read_text()) if m_path.exists() else {}
        b, a = m.get("before_vs_reference", {}), m.get("after_vs_reference", {})
        path_txt = f"{100 * b['rmse_fraction_of_path_length']:.2f}% &rarr; {100 * a['rmse_fraction_of_path_length']:.2f}%" if "rmse_fraction_of_path_length" in a and "rmse_fraction_of_path_length" in b else ""
        drift_txt = f"{b['local_scale_std']:.3f} &rarr; {a['local_scale_std']:.3f}" if "local_scale_std" in a and "local_scale_std" in b else ""
        fx_txt = f"{m['after_focal']['fx_ratio_vggt_over_reference']:.2f}x" if "after_focal" in m else ""
        cal = rec.get("calibration", {}); cal_txt = f"{cal.get('registered')}/{cal.get('images')}" if cal.get("registered") is not None else ""
        if not cal_txt and m.get("reference_registered"): cal_txt = f"{m['reference_registered']}/{m.get('frames_published', '')}"
        lens = m.get("calibration", {})
        lens_txt = f"f = {lens['fx_px']:.0f} px, HFOV {lens['hfov_deg']:.0f}&deg;<br><small>{lens['model']} k = {', '.join(f'{x:.3f}' for x in lens.get('distortion_params', [])[:4])}</small>" if lens else ""
        if "improved" in m:
            verdict = '<span class="ok">IMPROVED</span>' if m["improved"] else '<span class="no">not improved</span><br><small>' + "; ".join(m.get("gate_reasons", [])) + "</small>"
        else:
            verdict = ""
        before_img, after_img = REPORTS / vid / "before.jpg", REPORTS / vid / "after.jpg"
        imgs = "".join(f'<a href="{vid}/{n}.jpg"><img src="{vid}/{n}.jpg" alt="{n}"></a>' for n in ("before", "after") if (REPORTS / vid / f"{n}.jpg").exists())
        links = []
        if (REPORTS / vid / "transition.mp4").exists(): links.append(f'<a href="{vid}/transition.mp4">transition video</a>')
        if (REPORTS / vid / "overlay.mp4").exists(): links.append(f'<a href="{vid}/overlay.mp4">reprojection overlay video</a>')
        if (SCENES / vid / "overlay" / "index.html").exists(): links.append(f'<a href="/scenes/{BATCH}/{vid}/overlay/index.html">3D overlay viewer</a>')
        if (SCENES / vid / "published" / "viewer" / "scene_meta.json").exists(): links.append(f'<a href="/scenes/{BATCH}/{vid}/published/viewer/">camera view: published</a>')
        if any((SCENES / vid / "pinhole" / "viewer" / "camera_view_assets").glob("*_overlay.jpg")): links.append(f'<a href="/scenes/{BATCH}/{vid}/pinhole/viewer/">camera view: undistorted</a>')
        err = f'<div class="err">{rec.get("error", "")}</div>' if st == "failed" else ""
        el = f"{rec.get('elapsed_s', 0) // 60} min" if rec.get("elapsed_s") else ""
        rows.append(f'<tr class="{st}"><td><b>{s["title"]}</b><br><small>{vid} &middot; {s["published_frames"]} frames</small>{err}</td>'
                    f'<td class="st">{st}</td><td>{verdict}</td><td>{lens_txt}</td><td>{cal_txt}</td><td>{el}</td><td>{path_txt}</td><td>{drift_txt}</td><td>{fx_txt}</td>'
                    f'<td class="imgs">{imgs}<div>{" &middot; ".join(links)}</div></td></tr>')
    html = f"""<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Starred undistort re-run</title>
<style>body{{font:14px system-ui;background:#0c0d0f;color:#e8ebef;margin:20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px 10px;border-bottom:1px solid #2a3037;vertical-align:top;text-align:left}}
th{{color:#8b95a1;font-weight:600}}a{{color:#36e4ff}}img{{width:300px;border-radius:6px;margin:2px 6px 2px 0;border:1px solid #2a3037}}.imgs{{white-space:nowrap}}
.st{{font-weight:700;text-transform:uppercase}}tr.done .st{{color:#36e4ff}}tr.failed .st{{color:#ff4d6d}}tr.running .st{{color:#ffd60a}}tr.pending .st{{color:#8b95a1}}.err{{color:#ff4d6d;font-size:12px;margin-top:4px}}
.ok{{color:#36e4ff;font-weight:700}}.no{{color:#ffb000;font-weight:700}}.sum{{color:#8b95a1;margin-bottom:14px}}@media(max-width:900px){{img{{width:46vw}}td,th{{padding:6px 5px;font-size:12px}}}}small{{color:#8b95a1}}</style>
<meta name="viewport" content="width=device-width, initial-scale=1"><h1>{BATCH}: undistort-to-pinhole re-run</h1>
<div class="sum">updated {now_utc().strftime('%Y-%m-%d %H:%M:%S')} UTC &middot; pod {pod.get('id', '-')} (stop after {pod.get('stop_after', '-')}) &middot;
<b>{counts.get('done', 0)} done</b>, {counts.get('running', 0)} running, {counts.get('failed', 0)} failed, {counts.get('pending', 0)} pending &middot; page refreshes every minute</div>
<table><tr><th>Scene</th><th>Status</th><th>Result</th><th>Calibrated lens</th><th>SfM reg.</th><th>Time</th><th>Path disagreement vs SfM</th><th>Local scale std</th><th>Focal ratio (after)</th><th>Before / after</th></tr>
{''.join(rows)}</table>
<p class="sum">Path disagreement: Sim(3) residual of the VGGT camera path against the GLOMAP self-calibration path, fraction of its length. Local scale std: 20-frame window scale over global. Focal ratio: VGGT's focal estimate over the calibrated pinhole focal (1.0 = consistent). Before = published product, after = undistorted re-run.</p>
"""
    with _STATUS_LOCK:  # the refresh thread and the main loop both write this page
        tmp = STATUS_HTML.with_name(f"status.{threading.get_ident()}.tmp"); tmp.write_text(html); tmp.replace(STATUS_HTML)


class StatusPager:
    """Rewrites status.html every `interval` seconds in the background while scenes run."""
    def __init__(self, state, scenes, interval=60):
        self.state, self.scenes, self.interval = state, scenes, interval
        self.stop = threading.Event(); self.thread = threading.Thread(target=self.run, daemon=True)
    def run(self):
        while not self.stop.is_set():
            try: write_status_html(self.state, self.scenes)
            except Exception as exc: log(f"status page error: {exc}")
            self.stop.wait(self.interval)
    def __enter__(self): self.thread.start(); return self
    def __exit__(self, *a): self.stop.set(); self.thread.join(timeout=5); write_status_html(self.state, self.scenes)


class PostQueue:
    """Runs starred_postprocess.py for finished scenes in the background, one process at a time."""
    def __init__(self): self.proc = None; self.pending = []; self.launched = set()
    def add(self, vid): self.pending.append(vid); self.pump()
    def pump(self):
        if self.proc is not None and self.proc.poll() is None: return
        if self.proc is not None: log(f"post: background job finished (rc={self.proc.returncode})"); self.proc = None
        if not self.pending: return
        ids, self.pending = self.pending, []
        log(f"post: background job for {len(ids)} scene(s): {' '.join(ids)}")
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "tools/pipeline/starred_postprocess.py"), "--only", *ids], stdout=(REPORTS / "postprocess.log").open("a"), stderr=subprocess.STDOUT)
        self.launched.update(ids)
    def wait(self):
        while self.proc is not None or self.pending:
            self.pump()
            if self.proc is not None: self.proc.wait()
            self.pump()


# ---------------------------------------------------------------- stages

def stage_pod(args, state) -> dict:
    pod_id = args.pod_id or ("" if args.create_pod else state.get("pod", {}).get("id"))
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
    for name in ("starred_calibrate_remote.sh", "starred_transplant_matches.py", "benchmark_remote_job.py"):
        subprocess.run([*direct.scp_base(ssh), str(ROOT / "tools/pipeline" / name), f"root@{ssh['ip']}:{REMOTE_ROOT}/{name}"], check=True, capture_output=True)
    log("push: done")


def stage_calib_launch(args, state, scenes, ssh) -> None:
    """Launch --calib-jobs background calibration jobs on the pod, each over a contiguous chunk of scenes in run order.

    Calibration (CPU: SIFT matching + GLOMAP) takes ~10 min per scene on 8 threads, longer than the GPU inference,
    so several chunks run side by side on the pod's cores; the scene loop only waits for its own scene's status."""
    todo = [s for s in scenes if not (SCENES / s["video_id"] / "calib" / "glomap_fisheye" / "cameras.txt").exists()]
    if not todo:
        log("calib: every scene already has a local calibration; nothing to launch")
        return
    k = max(1, min(args.calib_jobs, len(todo)))
    n = len(todo); size = -(-n // k)
    chunks = [todo[i:i + size] for i in range(0, n, size)]
    launched = 0
    for j, chunk in enumerate(chunks):
        tag = "all" if j == 0 else f"chunk{j}"
        st = ssh_run(ssh, f"cat {REMOTE_ROOT}/calib_{tag}.status.json 2>/dev/null || echo NONE", check=False).stdout
        if '"status": "running"' in st or '"status": "succeeded"' in st:
            continue
        ids = " ".join(shlex.quote(s["video_id"]) for s in chunk)
        # GLOMAP/Ceres spawn one thread per core with no option to cap it; pin each chunk to its own core range
        # so k concurrent jobs do not oversubscribe the pod (3 x 128 threads made GLOMAP take 20-30 min per scene).
        ncpu = int(ssh_run(ssh, "nproc", check=False).stdout.strip() or "0")
        pin = ""
        if ncpu >= 8:
            per = max(4, (ncpu - 8) // k); lo = j * per; hi = min(ncpu - 9, lo + per - 1)  # leave 8 cores for the GPU job
            pin = f"taskset -c {lo}-{hi} "
        script = (f"cd {REMOTE_ROOT} && (CALIB_THREADS={args.calib_threads} setsid nohup {pin}python3 benchmark_remote_job.py --status {REMOTE_ROOT}/calib_{tag}.status.json -- "
                  f"bash {REMOTE_ROOT}/starred_calibrate_remote.sh {ids} > {REMOTE_ROOT}/calib_{tag}.log 2>&1 &) ; sleep 1; echo launched")
        ssh_run(ssh, script)
        launched += 1
    log(f"calib: {launched} background job(s) launched on the pod ({k} chunks of up to {size} scenes, {args.calib_threads} threads each)")


def wait_calib(ssh, vid: str, timeout_s: int) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        proc = ssh_run(ssh, f"cat {REMOTE_ROOT}/{vid}/calib/status.json 2>/dev/null || echo '{{}}'", check=False)
        if proc.returncode != 0:
            log(f"calib: {vid} ssh poll failed rc={proc.returncode}: {(proc.stderr or '')[-200:]}")
        out = proc.stdout
        try:
            st = json.loads(out.strip() or "{}")
        except Exception:
            st = {}
        if int(time.time() - t0) // 300 != int(time.time() - t0 - 30) // 300:
            log(f"calib: {vid} still waiting ({int(time.time() - t0) // 60} min), last status {st or 'none'}")
        if st.get("status") in ("succeeded", "failed"):
            return st
        if not st:
            # scene not started yet: has the global calibration job already ended?
            glob = ssh_run(ssh, f"cat {REMOTE_ROOT}/calib_all.status.json 2>/dev/null || echo '{{}}'", check=False).stdout
            try:
                gst = json.loads(glob.strip() or "{}").get("status")
            except Exception:
                gst = None
            if gst in ("succeeded", "failed") and not any(
                    '"status": "running"' in ssh_run(ssh, f"cat {REMOTE_ROOT}/calib_chunk{j}.status.json 2>/dev/null || echo NONE", check=False).stdout for j in range(1, 8)):
                tail = ssh_run(ssh, f"tail -3 {REMOTE_ROOT}/calib_all.log 2>/dev/null", check=False).stdout.strip().replace("\n", " | ")
                return {"status": "failed", "note": f"calibration job ended ({gst}) before this scene: {tail[-300:]}"}
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


def stage_harvest(args, state, scenes, ssh) -> None:
    """Fetch every finished calibration from the pod so nothing is lost when the pod is deleted (resume on a new pod)."""
    got = 0
    for s in scenes:
        vid = s["video_id"]; local = SCENES / vid / "calib" / "glomap_fisheye"
        if (local / "cameras.txt").exists():
            continue
        st = ssh_run(ssh, f"cat {REMOTE_ROOT}/{vid}/calib/status.json 2>/dev/null", check=False).stdout
        if '"succeeded"' not in st:
            continue
        try:
            for name in ("cameras.txt", "images.txt", "analyzer.txt"):
                scp_from(ssh, f"{REMOTE_ROOT}/{vid}/calib/glomap_fisheye/{name}", local / name)
            scp_from(ssh, f"{REMOTE_ROOT}/{vid}/calib/status.json", SCENES / vid / "calib" / "status.json")
            got += 1
        except Exception as exc:
            log(f"harvest: {vid} failed: {exc}")
    log(f"harvest: fetched {got} calibration(s) not yet used by a scene")


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
    # GLOMAP takes 10-15 min per scene and only keeps ~10 cores busy, while VGGT-Omega needs 3-4 min: calibration is the
    # critical path, so run as many chunks as the pod's cores allow (6 x 24 threads fits a 128+ core pod).
    ap.add_argument("--calib-jobs", type=int, default=6, help="parallel calibration jobs on the pod (contiguous chunks in run order)")
    ap.add_argument("--calib-threads", type=int, default=24, help="SIFT/matching threads per calibration job")
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
    write_status_html(state, scenes)

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
        REPORTS.mkdir(parents=True, exist_ok=True)
        post_q = PostQueue()
        with StatusPager(state, scenes):
            for s in scenes:
                if args.max_scenes and done >= args.max_scenes:
                    break
                if deadline and now_utc() + dt.timedelta(minutes=args.est_scene_min) > deadline:
                    log(f"deadline guard: not starting {s['video_id']} (deadline {deadline.isoformat()})")
                    break
                before = state["scenes"].get(s["video_id"], {}).get("status")
                run_scene(args, state, s, ssh)
                try:
                    write_status_html(state, scenes)
                except Exception as exc:
                    log(f"status page error: {exc}")
                if state["scenes"][s["video_id"]].get("status") == "done":
                    if before != "done":
                        done += 1
                    if not args.skip_post and not (SCENES / s["video_id"] / "metrics.json").exists():
                        post_q.add(s["video_id"])
                post_q.pump()
            n_done = sum(1 for s in scenes if state["scenes"].get(s["video_id"], {}).get("status") == "done")
            n_fail = sum(1 for s in scenes if state["scenes"].get(s["video_id"], {}).get("status") == "failed")
            log(f"scenes: {n_done} done, {n_fail} failed, {len(scenes) - n_done - n_fail} pending")
            if not args.skip_post:
                post_q.wait()
    if "pod-down" in stages:
        if ssh is not None:
            try:
                stage_harvest(args, state, scenes, ssh)
            except Exception as exc:
                log(f"harvest: {exc}")
        stage_pod_down(args, state)
    if "post" in stages and not args.skip_post:
        todo = [s["video_id"] for s in scenes if state["scenes"].get(s["video_id"], {}).get("status") == "done" and not (REPORTS / s["video_id"] / "transition.mp4").exists()]
        if todo:
            log(f"post: {len(todo)} scene(s) still to post-process")
            subprocess.run([sys.executable, str(ROOT / "tools/pipeline/starred_postprocess.py"), "--only", *todo], text=True)
        else:
            subprocess.run([sys.executable, str(ROOT / "tools/pipeline/starred_postprocess.py"), "--only", "__none__"], text=True, capture_output=True)  # refresh the index only
    write_status_html(state, scenes)
    log("batch end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
