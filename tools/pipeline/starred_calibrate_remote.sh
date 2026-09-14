#!/usr/bin/env bash
# Lens self-calibration for the starred scenes, run ON THE POD (CPU only, overlaps GPU inference).
#
# For each /workspace/starred/<video_id>/images/ directory:
#   COLMAP SIFT (single OPENCV_FISHEYE camera, CPU) -> exhaustive matching ->
#   GLOMAP global mapper (all intrinsics free, principal point at the frame centre) ->
#   TXT model in /workspace/starred/<video_id>/calib/glomap_fisheye/
# and a status JSON per scene in /workspace/starred/<video_id>/calib/status.json.
#
# COLMAP and GLOMAP come from conda-forge (Miniforge is bootstrapped once into /workspace/miniforge).
# Usage on the pod:  bash starred_calibrate_remote.sh [video_id ...]   (default: every scene dir)
set -uo pipefail
ROOT=/workspace/starred
CONDA_ROOT=/workspace/miniforge
export CONDA_PKGS_DIRS=/workspace/cache/conda-pkgs
THREADS="${CALIB_THREADS:-8}"

log() { echo "[calib $(date -u +%H:%M:%S)] $*"; }

ensure_tools() {
  if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    log "installing miniforge"
    curl -fsSL --retry 4 https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_ROOT" >/tmp/miniforge_install.log 2>&1
  fi
  export PATH="$CONDA_ROOT/bin:$PATH"
  if ! "$CONDA_ROOT/bin/conda" run -n sfm colmap -h >/dev/null 2>&1 || ! "$CONDA_ROOT/bin/conda" run -n sfm glomap -h >/dev/null 2>&1; then
    log "creating conda env sfm (colmap + glomap from conda-forge)"
    "$CONDA_ROOT/bin/conda" create -y -n sfm -c conda-forge colmap glomap >/tmp/sfm_env.log 2>&1 || { log "conda env creation failed, see /tmp/sfm_env.log"; tail -20 /tmp/sfm_env.log; exit 2; }
  fi
  COLMAP="$CONDA_ROOT/bin/conda run --no-capture-output -n sfm colmap"
  GLOMAP="$CONDA_ROOT/bin/conda run --no-capture-output -n sfm glomap"
  $COLMAP -h >/dev/null 2>&1 && $GLOMAP -h >/dev/null 2>&1 || { log "colmap/glomap not runnable"; exit 2; }
  log "tools ready"
}

write_status() {  # dir status registered images note
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, sys, datetime
d, st, reg, n, note = sys.argv[1:]
json.dump({"status": st, "registered": int(reg), "images": int(n), "note": note, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, open(f"{d}/status.json", "w"))
PY
}

calibrate_one() {
  local vid="$1"; local sd="$ROOT/$vid"; local img="$sd/images"; local cd="$sd/calib"; local out="$cd/glomap_fisheye"
  [ -d "$img" ] || { log "$vid: no images dir"; return; }
  if [ -s "$out/cameras.txt" ] && [ -s "$cd/status.json" ] && grep -q '"succeeded"' "$cd/status.json"; then log "$vid: already calibrated"; return; fi
  mkdir -p "$out"; rm -f "$cd/db.db" "$cd/db.db-wal" "$cd/db.db-shm"
  local n; n=$(find "$img" -maxdepth 1 -name '*.jpg' | wc -l)
  write_status "$cd" running 0 "$n" "features"
  local t0; t0=$(date +%s)
  if ! $COLMAP feature_extractor --database_path "$cd/db.db" --image_path "$img" --ImageReader.single_camera 1 \
        --ImageReader.camera_model OPENCV_FISHEYE --SiftExtraction.use_gpu 0 --SiftExtraction.max_num_features 16384 \
        --SiftExtraction.num_threads "$THREADS" > "$cd/log.txt" 2>&1; then
    write_status "$cd" failed 0 "$n" "feature_extractor failed"; return; fi
  # COLMAP >= 3.10 fisheye pairs need a focal prior on the camera row
  python3 - "$cd/db.db" <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1]); c.execute("UPDATE cameras SET prior_focal_length=1"); c.commit(); c.close()
PY
  write_status "$cd" running 0 "$n" "matching"
  if ! $COLMAP exhaustive_matcher --database_path "$cd/db.db" --SiftMatching.use_gpu 0 --SiftMatching.num_threads "$THREADS" >> "$cd/log.txt" 2>&1; then
    write_status "$cd" failed 0 "$n" "exhaustive_matcher failed"; return; fi
  write_status "$cd" running 0 "$n" "glomap"
  rm -rf "$cd/glomap_sparse"; mkdir -p "$cd/glomap_sparse"
  if ! $GLOMAP mapper --database_path "$cd/db.db" --image_path "$img" --output_path "$cd/glomap_sparse" >> "$cd/log.txt" 2>&1; then
    write_status "$cd" failed 0 "$n" "glomap mapper failed"; return; fi
  local model; model=$(ls -d "$cd"/glomap_sparse/*/ 2>/dev/null | head -1)
  [ -n "$model" ] || { write_status "$cd" failed 0 "$n" "glomap produced no model"; return; }
  $COLMAP model_converter --input_path "$model" --output_path "$out" --output_type TXT >> "$cd/log.txt" 2>&1
  $COLMAP model_analyzer --path "$model" > "$out/analyzer.txt" 2>&1
  local reg; reg=$(grep -c -vE '^#' "$out/images.txt" | awk '{print int($1/2)}')
  local dt=$(( $(date +%s) - t0 ))
  log "$vid: registered $reg / $n images in ${dt}s; $(grep -E 'Mean reprojection|Registered' "$out/analyzer.txt" | tr '\n' ' ')"
  write_status "$cd" succeeded "$reg" "$n" "glomap OPENCV_FISHEYE, ${dt}s"
}

ensure_tools
if [ "$#" -gt 0 ]; then LIST=("$@"); else mapfile -t LIST < <(ls "$ROOT" | while read -r d; do [ -d "$ROOT/$d/images" ] && echo "$d"; done); fi
for vid in "${LIST[@]}"; do calibrate_one "$vid"; done
log "all done"
