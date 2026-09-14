#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


def read_local_hf_token() -> str:
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token.strip()
    for path in (
        Path.home() / ".cache/huggingface/token",
        Path.home() / ".huggingface/token",
    ):
        if path.exists():
            token = path.read_text().strip()
            if token:
                return token
    return ""


def run(cmd: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, input=input_text, text=True, capture_output=True, check=check)


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


def proxy_url(pod_id: str) -> str:
    return f"https://{pod_id}-7860.proxy.runpod.net"


def wait_for_url(url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:
            pass
        print("[runpod] waiting for gradio", flush=True)
        time.sleep(10)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install and launch the VGGT-Omega Gradio server on a RunPod pod.")
    parser.add_argument("pod_id")
    parser.add_argument("--wait-ssh-s", type=int, default=600)
    parser.add_argument("--wait-url-s", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_token = read_local_hf_token()
    if not hf_token:
        raise SystemExit(
            "Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or save a token in ~/.cache/huggingface/token before running this setup helper."
        )

    ssh = wait_for_ssh(args.pod_id, args.wait_ssh_s)
    key_path = Path(ssh.get("ssh_key", {}).get("path") or Path.home() / ".runpod/ssh/runpodctl-ssh-key")
    host = ssh["ip"]
    port = str(ssh["port"])

    base_ssh_cmd = [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        port,
        f"root@{host}",
    ]

    token_cmd = [
        *base_ssh_cmd,
        "mkdir -p /root/.cache/huggingface && umask 077 && cat > /root/.cache/huggingface/token",
    ]
    token_proc = run(token_cmd, input_text=hf_token, check=False)
    if token_proc.returncode != 0:
        sys.stderr.write(token_proc.stderr)
        return token_proc.returncode

    remote_script = r"""set -euo pipefail
export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
cd /workspace
if [ ! -d vggt-omega-space/.git ]; then
  echo "[setup] clone vggt-omega"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/spaces/facebook/vggt-omega vggt-omega-space >/tmp/vggt_clone.log 2>&1
fi
python -m venv --system-site-packages /workspace/vggt-omega-venv
. /workspace/vggt-omega-venv/bin/activate
python - <<'PY'
import torch
print(f"[setup] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
PY
echo "[setup] pip requirements"
pip install -q --upgrade pip
pip install -q -r /workspace/vggt-omega-space/requirements.txt
echo "[setup] launch gradio"
if [ -s /workspace/vggt_omega_server.pid ]; then
  old_pid="$(cat /workspace/vggt_omega_server.pid || true)"
  if [ -n "$old_pid" ]; then kill "$old_pid" 2>/dev/null || true; fi
fi
cd /workspace/vggt-omega-space
nohup /workspace/vggt-omega-venv/bin/python app.py > /workspace/vggt_omega_server.log 2>&1 < /dev/null &
echo $! > /workspace/vggt_omega_server.pid
echo "[setup] pid=$(cat /workspace/vggt_omega_server.pid)"
"""
    proc = run([*base_ssh_cmd, "bash -s"], input_text=remote_script, check=False)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    url = proxy_url(args.pod_id)
    print(f"[setup] url={url}", flush=True)
    if args.wait_url_s and not wait_for_url(url, args.wait_url_s):
        print("[setup] url not ready; check /workspace/vggt_omega_server.log on the pod", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
