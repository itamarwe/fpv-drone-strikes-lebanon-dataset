#!/usr/bin/env python3
"""Manage the RunPod pod that hosts the VGGT-Omega gradio server.

This encapsulates the whole "bring Omega online" dance that previously had to be
done by hand: start the pod, wait for SSH, launch ``app.py`` (it does not
auto-start), and wait for the gradio endpoint to answer. Everything is
configurable through environment variables so no secrets live in the repo; the
defaults match the project's current pod.

Environment overrides:
  RUNPOD_POD_ID        pod id (default: the project's Omega pod)
  RUNPOD_SSH_KEY       path to the ssh private key (default: runpodctl's key)
  OMEGA_DIR            app dir on the pod (default: /workspace/vggt-omega)
  OMEGA_PYTHON         python on the pod (default: /workspace/vggt-venv/bin/python)
  OMEGA_CHECKPOINT     checkpoint path on the pod
  OMEGA_GRADIO_PORT    gradio port (default: 7860)

Usage:
  python tools/omega_pod.py up      # start pod + launch Omega, wait until ready
  python tools/omega_pod.py down    # stop the pod
  python tools/omega_pod.py status  # print pod + gradio state
  python tools/omega_pod.py url     # print the gradio space URL
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

POD_ID = os.environ.get("RUNPOD_POD_ID", "7i0jtqk99phk2j")
SSH_KEY = os.path.expanduser(os.environ.get("RUNPOD_SSH_KEY", "~/.runpod/ssh/runpodctl-ssh-key"))
OMEGA_DIR = os.environ.get("OMEGA_DIR", "/workspace/vggt-omega")
OMEGA_PYTHON = os.environ.get("OMEGA_PYTHON", "/workspace/vggt-venv/bin/python")
OMEGA_CHECKPOINT = os.environ.get(
    "OMEGA_CHECKPOINT", "/workspace/hf-checkpoints/vggt-omega/vggt_omega_1b_512.pt"
)
GRADIO_PORT = int(os.environ.get("OMEGA_GRADIO_PORT", "7860"))

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "BatchMode=yes",
]


def log(msg: str) -> None:
    print(f"[omega-pod] {msg}", flush=True)


def space_url(pod_id: str = POD_ID) -> str:
    return f"https://{pod_id}-{GRADIO_PORT}.proxy.runpod.net"


def _runpodctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["runpodctl", *args], capture_output=True, text=True)


def pod_start(pod_id: str = POD_ID) -> None:
    _runpodctl("pod", "start", pod_id)
    log(f"pod {pod_id} start requested")


def pod_stop(pod_id: str = POD_ID) -> None:
    _runpodctl("pod", "stop", pod_id)
    log(f"pod {pod_id} stop requested")


def ssh_info(pod_id: str = POD_ID) -> dict | None:
    """Return the current direct-SSH connection details, or None if unavailable.

    The proxy hostname (``<id>-<hash>@ssh.runpod.io``) can stop accepting the key
    after a restart, so we use the direct ip:port endpoint runpodctl reports.
    """
    res = _runpodctl("ssh", "info", pod_id)
    try:
        info = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    if not info.get("ip") or not info.get("port"):
        return None
    return info


def _ssh_base(info: dict) -> list[str]:
    key = (info.get("ssh_key") or {}).get("path") or SSH_KEY
    return ["ssh", *SSH_OPTS, "-i", os.path.expanduser(key), "-p", str(info["port"]), f"root@{info['ip']}"]


def ssh_run(info: dict, remote_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(_ssh_base(info) + [remote_cmd], capture_output=True, text=True, timeout=timeout)


def gradio_status(pod_id: str = POD_ID) -> int:
    req = urllib.request.Request(space_url(pod_id) + "/", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


def wait_ssh(pod_id: str, timeout_s: int) -> dict | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = ssh_info(pod_id)
        if info:
            try:
                res = ssh_run(info, "echo ok", timeout=20)
                if res.returncode == 0 and "ok" in res.stdout:
                    return info
            except subprocess.TimeoutExpired:
                pass
        time.sleep(10)
    return None


def launch_omega(info: dict) -> None:
    remote = (
        f"cd {OMEGA_DIR} && : > app.log && "
        f"VGGT_OMEGA_CHECKPOINT={OMEGA_CHECKPOINT} "
        f"setsid {OMEGA_PYTHON} -u app.py > app.log 2>&1 < /dev/null & echo launched"
    )
    ssh_run(info, remote, timeout=30)


def wait_gradio(pod_id: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if gradio_status(pod_id) == 200:
            return True
        time.sleep(15)
    return False


def up(pod_id: str = POD_ID, ready_timeout_s: int = 600) -> str:
    """Ensure the pod is running and Omega answers on the gradio port."""
    if gradio_status(pod_id) == 200:
        log(f"gradio already up at {space_url(pod_id)}")
        return space_url(pod_id)

    log(f"starting pod {pod_id} ...")
    pod_start(pod_id)

    log("waiting for SSH ...")
    info = wait_ssh(pod_id, timeout_s=ready_timeout_s)
    if not info:
        raise SystemExit("[omega-pod] SSH did not become reachable; check the pod in the RunPod console")
    log(f"SSH up ({info['ip']}:{info['port']})")

    if gradio_status(pod_id) == 200:
        log("gradio already serving")
        return space_url(pod_id)

    log("launching Omega app.py ...")
    launch_omega(info)

    log("waiting for gradio (model load) ...")
    if not wait_gradio(pod_id, timeout_s=ready_timeout_s):
        raise SystemExit("[omega-pod] gradio did not come up; check app.log on the pod")
    log(f"ready: {space_url(pod_id)}")
    return space_url(pod_id)


def status(pod_id: str = POD_ID) -> None:
    info = ssh_info(pod_id)
    ssh_state = f"{info['ip']}:{info['port']}" if info else "unreachable"
    log(f"pod={pod_id} ssh={ssh_state} gradio_http={gradio_status(pod_id)} url={space_url(pod_id)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["up", "down", "status", "url"])
    parser.add_argument("--pod-id", default=POD_ID)
    parser.add_argument("--ready-timeout", type=int, default=600, help="seconds to wait for SSH / gradio")
    args = parser.parse_args()

    if args.command == "up":
        print(up(args.pod_id, ready_timeout_s=args.ready_timeout))
    elif args.command == "down":
        pod_stop(args.pod_id)
    elif args.command == "status":
        status(args.pod_id)
    elif args.command == "url":
        print(space_url(args.pod_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
