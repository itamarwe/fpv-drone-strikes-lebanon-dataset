#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check, input=input_text)


def run_json(cmd: list[str]) -> Any:
    proc = run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit((proc.stderr or proc.stdout).strip())
    return json.loads(proc.stdout)


def pod_get(pod_id: str) -> dict[str, Any]:
    return run_json(["runpodctl", "pod", "get", pod_id, "-o", "json"])


def wait_for_ssh(pod_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        pod = pod_get(pod_id)
        ssh = pod.get("ssh") or {}
        if ssh.get("ip") and ssh.get("port"):
            return ssh
        last = ssh.get("error") or pod.get("desiredStatus") or "pending"
        print(f"[runpod] wait ssh={last}", flush=True)
        time.sleep(10)
    raise SystemExit(f"[runpod] ssh not ready within {timeout_s}s; last={last}")


def ssh_base(ssh: dict[str, Any]) -> list[str]:
    key_path = ssh.get("ssh_key", {}).get("path") or str(Path.home() / ".runpod/ssh/runpodctl-ssh-key")
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(ssh["port"]),
        f"root@{ssh['ip']}",
    ]


def scp_base(ssh: dict[str, Any]) -> list[str]:
    key_path = ssh.get("ssh_key", {}).get("path") or str(Path.home() / ".runpod/ssh/runpodctl-ssh-key")
    return [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-P",
        str(ssh["port"]),
    ]


def transfer_frames(frames_dir: Path, ssh: dict[str, Any], remote_frames_dir: str) -> None:
    if not frames_dir.is_dir():
        raise SystemExit(f"Missing frames dir: {frames_dir}")
    remote = (
        f"rm -rf {shlex.quote(remote_frames_dir)}; "
        f"mkdir -p {shlex.quote(remote_frames_dir)}; "
        f"tar -C {shlex.quote(remote_frames_dir)} -xf -; "
        f"find {shlex.quote(remote_frames_dir)} -maxdepth 1 -type f -name '._*' -delete; "
        f"find {shlex.quote(remote_frames_dir)} -maxdepth 1 -type f -name '*.jpg' | wc -l"
    )
    tar_proc = subprocess.Popen(
        ["tar", "-C", str(frames_dir), "-cf", "-", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "COPYFILE_DISABLE": "1"},
    )
    ssh_proc = subprocess.run([*ssh_base(ssh), remote], stdin=tar_proc.stdout, text=True, capture_output=True)
    if tar_proc.stdout:
        tar_proc.stdout.close()
    _, tar_err = tar_proc.communicate()
    if tar_proc.returncode != 0:
        raise SystemExit(tar_err.decode(errors="ignore"))
    if ssh_proc.returncode != 0:
        raise SystemExit((ssh_proc.stderr or ssh_proc.stdout).strip())
    print(f"[transfer] remote_jpg_frames={ssh_proc.stdout.strip()}", flush=True)


def run_remote_vggt(
    ssh: dict[str, Any],
    remote_frames_dir: str,
    remote_target_dir: str,
    max_points_k: float,
    force_inference: bool,
    slim_predictions: bool = False,
) -> None:
    slim_step = ""
    if slim_predictions:
        # Drop the derivable/bulky arrays before download: world points (recomputable from depth + K + pose),
        # the ViT tokens and the float image tensor (the input frames are kept locally). ~1.4 GB -> ~0.2 GB.
        slim_step = f"""PYTHONUNBUFFERED=1 python - <<'PY'
import numpy as np
from pathlib import Path
p = Path({remote_target_dir!r}) / "predictions.npz"
z = np.load(p, allow_pickle=True)
keep = {{k: z[k] for k in z.files if k not in ("world_points_from_depth", "camera_and_register_tokens", "images")}}
keep["depth"] = keep["depth"].astype(np.float32); keep["depth_conf"] = keep["depth_conf"].astype(np.float16)
keep["slim"] = np.array(["world_points_from_depth, camera_and_register_tokens and images removed on the pod; depth_conf stored as float16"])
np.savez(p.with_name("predictions_slim.npz"), **keep)
p.with_name("predictions_slim.npz").replace(p)
print("[direct] slim predictions " + str(p.stat().st_size), flush=True)
PY
"""
    remote_script = f"""set -euo pipefail
cd /workspace/vggt-omega-space
if [ -s /workspace/vggt_omega_server.pid ]; then
  old_pid="$(cat /workspace/vggt_omega_server.pid || true)"
  if [ -n "$old_pid" ]; then kill "$old_pid" 2>/dev/null || true; fi
fi
rm -rf {shlex.quote(remote_target_dir)}
mkdir -p {shlex.quote(remote_target_dir)}/images
find {shlex.quote(remote_frames_dir)} -maxdepth 1 -type f -name '*.jpg' -print0 | sort -z | xargs -0 -I{{}} cp {{}} {shlex.quote(remote_target_dir)}/images/
echo "[direct] frames=$(find {shlex.quote(remote_target_dir)}/images -maxdepth 1 -type f -name '*.jpg' | wc -l)"
. /workspace/vggt-omega-venv/bin/activate
export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil
p = hf_hub_download(repo_id="facebook/vggt-omega", repo_type="space", filename="skyseg.onnx")
shutil.copy2(p, "skyseg.onnx")
print("[direct] skyseg_size=" + str(Path("skyseg.onnx").stat().st_size), flush=True)
PY
# The demo's CPU stages (sky segmentation, export) size their thread pools to every core. On a big pod that is
# also running lens calibration this oversubscribes and turns a 3 min scene into 15-25 min: give inference its own
# block of cores (the top quarter) and size the pools to it.
NCPU="$(nproc)"; PIN=""
if [ "$NCPU" -ge 64 ]; then LO=$((NCPU * 3 / 4)); PIN="taskset -c $LO-$((NCPU - 1))"; export OMP_NUM_THREADS=$((NCPU - LO)) MKL_NUM_THREADS=$((NCPU - LO)) OPENBLAS_NUM_THREADS=$((NCPU - LO)); fi
echo "[direct] cpu pin: ${{PIN:-none}}"
PYTHONUNBUFFERED=1 $PIN python - <<'PY'
import json, time
from pathlib import Path
from app import gradio_demo, update_visualization

target_dir = {remote_target_dir!r}
max_points_k = {float(max_points_k)!r}
force_inference = {bool(force_inference)!r}
t0 = time.time()
if force_inference or not Path(target_dir, "predictions.npz").exists():
    print("[direct] start gradio_demo", flush=True)
    glb, log = gradio_demo(target_dir, 50.0, False, False, True, True, max_points_k)
else:
    print("[direct] start update_visualization", flush=True)
    glb, log = update_visualization(target_dir, 50.0, False, False, True, True, max_points_k)
summary = {{"glb": glb, "log": log, "elapsed_s": time.time() - t0, "size_bytes": Path(glb).stat().st_size}}
Path("/workspace/vggt_omega_direct_summary.json").write_text(json.dumps(summary, indent=2))
print("[direct] done " + json.dumps(summary), flush=True)
PY
{slim_step}"""
    proc = subprocess.run([*ssh_base(ssh), "bash -s"], input=remote_script, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def copy_results(
    scene_dir: Path,
    ssh: dict[str, Any],
    remote_target_dir: str,
    copy_remote_artifacts: bool,
) -> None:
    summary_proc = run([*ssh_base(ssh), "cat /workspace/vggt_omega_direct_summary.json"], check=False)
    if summary_proc.returncode != 0:
        raise SystemExit((summary_proc.stderr or summary_proc.stdout).strip())
    summary = json.loads(summary_proc.stdout)
    glb_remote = f"root@{ssh['ip']}:/workspace/vggt-omega-space/{summary['glb']}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    glb_local = scene_dir / "vggt_scene.glb"
    summary_local = scene_dir / "vggt_remote_summary.json"
    subprocess.run([*scp_base(ssh), glb_remote, str(glb_local)], check=True)
    summary_local.write_text(json.dumps(summary, indent=2))
    if copy_remote_artifacts:
        artifacts_dir = scene_dir / "runpod_artifacts"
        if artifacts_dir.exists():
            import shutil

            shutil.rmtree(artifacts_dir)
        artifacts_dir.mkdir(parents=True)
        remote_tar = "/workspace/vggt_omega_direct_artifacts.tar.gz"
        remote_script = f"""set -euo pipefail
cd /workspace/vggt-omega-space
tar -czf {shlex.quote(remote_tar)} -C {shlex.quote(remote_target_dir)} .
find {shlex.quote(remote_target_dir)} -type f -printf '%P\\t%s\\n' | sort > /workspace/vggt_omega_direct_artifacts_manifest.tsv
"""
        subprocess.run([*ssh_base(ssh), "bash -s"], input=remote_script, text=True, check=True)
        local_tar = artifacts_dir / "remote_target.tar.gz"
        remote_base = f"root@{ssh['ip']}:"
        subprocess.run([*scp_base(ssh), remote_base + remote_tar, str(local_tar)], check=True)
        subprocess.run(
            [*scp_base(ssh), remote_base + "/workspace/vggt_omega_direct_artifacts_manifest.tsv", str(artifacts_dir / "manifest.tsv")],
            check=True,
        )
        subprocess.run(["tar", "-xzf", str(local_tar), "-C", str(artifacts_dir)], check=True)
        local_tar.unlink()
        print(f"[copy] remote artifacts -> {artifacts_dir}", flush=True)
    print(f"[copy] {glb_local} bytes={glb_local.stat().st_size}", flush=True)


def build_artifacts(scene_dir: Path, title: str, scale: float, max_points: int) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/build_vggt_scene_artifacts.py"),
            str(scene_dir),
            "--title",
            title,
            "--default-scale-m-per-unit",
            str(scale),
            "--max-points",
            str(max_points),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT-Omega directly on a RunPod pod for a prepared scene directory.")
    parser.add_argument("pod_id")
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--remote-name", default="")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Flat JPEG directory to stage to the pod (default: <scene_dir>/frames).")
    parser.add_argument("--wait-ssh-s", type=int, default=600)
    parser.add_argument("--max-points-k", type=float, default=10000.0)
    parser.add_argument("--artifact-max-points", type=int, default=10_000_000)
    parser.add_argument("--default-scale-m-per-unit", type=float, default=76.0)
    parser.add_argument("--title", default="VGGT-Omega scene")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--no-copy-remote-artifacts", dest="copy_remote_artifacts", action="store_false")
    parser.add_argument("--slim-predictions", action="store_true", help="strip world points, tokens and image tensors from predictions.npz on the pod before download")
    parser.set_defaults(copy_remote_artifacts=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    frames_dir = (args.frames_dir or (scene_dir / "frames")).resolve()
    remote_name = args.remote_name or scene_dir.name
    remote_frames_dir = f"/workspace/{remote_name}_frames"
    remote_target_dir = f"demo_outputs/{remote_name}_direct"
    ssh = wait_for_ssh(args.pod_id, args.wait_ssh_s)
    if not args.skip_transfer:
        transfer_frames(frames_dir, ssh, remote_frames_dir)
    run_remote_vggt(ssh, remote_frames_dir, remote_target_dir, args.max_points_k, args.force_inference, args.slim_predictions)
    copy_results(scene_dir, ssh, remote_target_dir, args.copy_remote_artifacts)
    if not args.skip_artifacts:
        build_artifacts(scene_dir, args.title, args.default_scale_m_per_unit, args.artifact_max_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
