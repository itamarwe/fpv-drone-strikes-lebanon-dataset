#!/usr/bin/env python3
"""Run both focal conditions on the frozen common16 inputs, sequentially."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workspace", type=Path, default=Path("/workspace/fpv-3d-benchmark"))
parser.add_argument("--python", type=Path, default=Path("/workspace/miniforge3/envs/moge3/bin/python"))
args = parser.parse_args()
bundle = args.workspace / "bundle"
scripts = bundle / "orchestration"
attempt = datetime.datetime.now(datetime.timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ")
environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
failures = []
for condition in ("estimated_focal", "diagnostic_fixed_focal"):
    for scene in ("scene_a", "scene_b"):
        output = args.workspace / "results" / scene / "common16" / "moge3_baseline" / attempt / condition
        output.mkdir(parents=True, exist_ok=True)
        record = {"method": "moge3_baseline", "scene": scene, "profile": "common16",
                  "mode": condition, "input_image_count": 16, "status": "running",
                  "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "limitations": ["Published camera intrinsics unavailable; projection-based scale sensitivity diagnostic, not independently surveyed accuracy."]}
        status_file = output / "run.json"
        status_file.write_text(json.dumps(record, indent=2) + "\n")
        inference = [str(args.python), "-u", str(scripts / "moge3_export_depths.py"),
                     "--images", str(bundle / "scenes" / scene / "profiles" / "common16" / "images"),
                     "--output", str(output / "depths"), "--checkpoint",
                     str(args.workspace / "repos" / "MoGe" / "checkpoints" / "moge-3-vitl" / "model.pt")]
        evaluation = [str(args.python), "-u", str(scripts / "evaluate_moge3_vggt_scale.py"),
                      "--scene-dir", str(bundle / "scenes" / scene), "--profile", "common16",
                      "--depths", str(output / "depths"), "--output", str(output / "metric_scale.json"),
                      "--depth-layout", "full"]
        if condition == "diagnostic_fixed_focal":
            inference += ["--fov-x", "44.23403727789238"]
            evaluation += ["--focal-source", "fixed", "--focal-px", "812"]
        else:
            evaluation += ["--focal-source", "moge"]
        record["commands"] = [inference, evaluation]
        started = time.monotonic()
        print(f"Starting {scene} {condition}: {output}", flush=True)
        try:
            with (output / "run.log").open("w") as log:
                for command in (inference, evaluation):
                    subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            record["status"] = "complete"
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            failures.append(f"{scene}/{condition}")
        record.update(elapsed_seconds=time.monotonic() - started,
                      ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        status_file.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Finished {scene} {condition}: {record['status']}", flush=True)
print(json.dumps({"attempt": attempt, "failures": failures}), flush=True)
sys.exit(bool(failures))
