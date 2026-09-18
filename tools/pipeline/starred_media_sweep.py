#!/usr/bin/env python3
"""Build the full media (transition + reprojection videos, camera-view overlays) for scenes the gate rejected.

The batch driver runs with FPV_UNDISTORT_GATE=1 and skips the expensive media for scenes that did not improve.
This sweep renders it for them anyway, so every scene can be inspected; the verdict and reasons in metrics.json
are unchanged. With --watch it keeps going until the batch is complete, picking up rejected scenes as they land.

  FPV_UNDISTORT_BATCH=all_undistort starred_media_sweep.py --watch
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = os.environ.get("FPV_UNDISTORT_BATCH", "starred_undistort")
SCENES = ROOT / "scenes" / BATCH
REPORTS = ROOT / "reports" / BATCH
STATE = ROOT / "benchmarks" / BATCH / "batch_status.json"
SPEC = ROOT / "benchmarks" / BATCH / "starred_scenes.json"


def todo() -> list[str]:
    out = []
    for m_path in sorted(SCENES.glob("*/metrics.json")):
        vid = m_path.parent.name
        try:
            m = json.loads(m_path.read_text())
        except Exception:
            continue  # being written right now; next pass
        if m.get("improved") is False and not (REPORTS / vid / "overlay.mp4").exists() and not (REPORTS / vid / "overlay.mp4.failed").exists():
            out.append(vid)
    return out


def batch_complete() -> bool:
    try:
        st = json.loads(STATE.read_text())["scenes"]; n = len(json.loads(SPEC.read_text())["scenes"])
    except Exception:
        return False
    return sum(1 for v in st.values() if v.get("status") in ("done", "failed")) >= n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true", help="keep sweeping until every scene of the batch is done or failed")
    args = ap.parse_args()
    env = {**os.environ, "FPV_UNDISTORT_BATCH": BATCH, "FPV_UNDISTORT_GATE": "0"}
    built = 0
    while True:
        ids = todo()
        for vid in ids:
            print(f"[sweep] {vid}", flush=True)
            r = subprocess.run([sys.executable, str(ROOT / "tools/pipeline/starred_postprocess.py"), "--only", vid], env=env, capture_output=True, text=True)
            ok = (REPORTS / vid / "overlay.mp4").exists()
            built += ok
            if not ok:
                print(f"[sweep] {vid} did not produce media: {(r.stdout + r.stderr)[-300:]}", flush=True)
                (REPORTS / vid).mkdir(parents=True, exist_ok=True)
                (REPORTS / vid / "overlay.mp4.failed").write_text((r.stdout + r.stderr)[-2000:])  # do not retry forever
        if not args.watch or (batch_complete() and not todo()):
            break
        if not ids:
            time.sleep(60)
    print(f"[sweep] built media for {built} scene(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
