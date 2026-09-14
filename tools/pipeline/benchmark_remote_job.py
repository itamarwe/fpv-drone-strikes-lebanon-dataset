#!/usr/bin/env python3
"""Run one remote benchmark command and persist its status even on failure."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--status", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    parser.error("command is required")
args.status.parent.mkdir(parents=True, exist_ok=True)
record = {"command": command, "pid": os.getpid(), "status": "running",
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}

def save():
    temporary = args.status.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(args.status)

save()
start = time.monotonic()
try:
    result = subprocess.run(command, check=False)
    record.update(status="succeeded" if result.returncode == 0 else "failed", returncode=result.returncode)
except Exception as exc:
    record.update(status="failed", error=str(exc), returncode=1)
record.update(elapsed_seconds=time.monotonic() - start,
              finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
raise SystemExit(record["returncode"])
