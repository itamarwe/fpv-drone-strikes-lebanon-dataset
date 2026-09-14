#!/usr/bin/env bash
# One-shot, unattended re-run of every starred scene through the undistort-to-pinhole pipeline.
#
#   tools/pipeline/run_starred_night.sh                 # creates an A100 pod (6 h stop / 6.5 h terminate), runs, stops+deletes it
#   POD_ID=xxxx tools/pipeline/run_starred_night.sh     # reuse an existing pod
#   HOURS=8 tools/pipeline/run_starred_night.sh         # longer cloud-enforced deadline
#
# Everything is resumable: re-running the same command skips finished stages/scenes.
# Log: benchmarks/starred_undistort/batch_log.txt   State: benchmarks/starred_undistort/batch_status.json
set -euo pipefail
cd "$(dirname "$0")/../.."
HOURS="${HOURS:-6}"
if [ -n "${POD_ID:-}" ]; then
  exec python3 tools/pipeline/starred_run_batch.py --pod-id "$POD_ID" --delete-pod-when-done "$@"
else
  exec python3 tools/pipeline/starred_run_batch.py --create-pod --hours "$HOURS" --delete-pod-when-done "$@"
fi
