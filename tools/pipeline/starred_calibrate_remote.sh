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
WORK=${CALIB_WORK:-/root/calib_work}
CONDA_ROOT=/workspace/miniforge
export CONDA_PKGS_DIRS=/workspace/cache/conda-pkgs
THREADS="${CALIB_THREADS:-8}"
export QT_QPA_PLATFORM=offscreen

log() { echo "[calib $(date -u +%H:%M:%S)] $*"; }

ensure_tools() {
  if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    log "installing miniforge"
    curl -fsSL --retry 4 https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_ROOT" >/tmp/miniforge_install.log 2>&1
  fi
  export PATH="$CONDA_ROOT/bin:$PATH"
  # The CUDA variant of conda-forge colmap (4.0.4) is linked against an incompatible libfaiss and cannot start;
  # the CPU variant solves cleanly together with a matching glomap (colmap 3.11.1 + glomap 1.1.0 on 2026-09-14).
  if ! "$CONDA_ROOT/bin/conda" run -n sfm2 colmap -h >/dev/null 2>&1 || ! "$CONDA_ROOT/bin/conda" run -n sfm2 glomap -h >/dev/null 2>&1; then
    log "creating conda env sfm2 (cpu colmap + glomap from conda-forge)"
    "$CONDA_ROOT/bin/conda" create -y -n sfm2 -c conda-forge "colmap=*=cpu*" glomap >/tmp/sfm_env.log 2>&1 || { log "conda env creation failed, see /tmp/sfm_env.log"; tail -20 /tmp/sfm_env.log; exit 2; }
  fi
  COLMAP="$CONDA_ROOT/bin/conda run --no-capture-output -n sfm2 colmap"
  GLOMAP="$CONDA_ROOT/bin/conda run --no-capture-output -n sfm2 glomap"
  $COLMAP -h >/dev/null 2>&1 || { log "colmap not runnable: $($COLMAP -h 2>&1 | head -2)"; exit 2; }
  $GLOMAP -h >/dev/null 2>&1 || { log "glomap not runnable: $($GLOMAP -h 2>&1 | head -2)"; exit 2; }
  PYCOLMAP_PY=/workspace/vggt-omega-venv/bin/python
  [ -x "$PYCOLMAP_PY" ] || PYCOLMAP_PY=python3
  "$PYCOLMAP_PY" -c "import pycolmap" 2>/dev/null || "$PYCOLMAP_PY" -m pip install -q "pycolmap>=4.0" >/tmp/pycolmap_install.log 2>&1 || log "pycolmap install failed (fallback matcher unavailable)"
  log "tools ready"
}

write_status() {  # dir status registered images note   (dir = the volume-side calib dir the driver polls)
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, sys, datetime
d, st, reg, n, note = sys.argv[1:]
json.dump({"status": st, "registered": int(reg), "images": int(n), "note": note, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, open(f"{d}/status.json", "w"))
PY
}

calibrate_one() {
  local vid="$1"; local sd="$ROOT/$vid"; local img="$sd/images"; local final="$sd/calib"
  # SQLite on the /workspace network volume is 10-50x slower than local disk: do all database work under
  # $WORK (container disk) and copy the small text results to the volume at the end.
  local cd="$WORK/$vid"; local out="$cd/glomap_fisheye"
  [ -d "$img" ] || { log "$vid: no images dir"; return; }
  mkdir -p "$final"
  if [ -s "$final/glomap_fisheye/cameras.txt" ] && [ -s "$final/status.json" ] && grep -q '"succeeded"' "$final/status.json"; then log "$vid: already calibrated"; return; fi
  local n; n=$(find "$img" -maxdepth 1 -name '*.jpg' | wc -l)
  local t0; t0=$(date +%s)
  local resume=0
  if [ "${CALIB_RESUME:-0}" = "1" ] && [ -s "$cd/db_py.db" ]; then
    # CALIB_RESUME=1: keep an existing pycolmap database (matching is the expensive part), redo transplant + GLOMAP
    resume=1; log "$vid: resuming from existing pycolmap matches"
    rm -rf "$cd/db.db" "$cd/db.db-wal" "$cd/db.db-shm" "$cd/glomap_sparse"; mkdir -p "$out"
  else
    rm -rf "$cd"; mkdir -p "$out"
  fi
  local flann_ok=0
  if [ "$resume" = "0" ]; then
    write_status "$final" running 0 "$n" "features"
    if ! $COLMAP feature_extractor --database_path "$cd/db.db" --image_path "$img" --ImageReader.single_camera 1 \
          --ImageReader.camera_model OPENCV_FISHEYE --SiftExtraction.use_gpu 0 --SiftExtraction.max_num_features 16384 \
          --SiftExtraction.num_threads "$THREADS" > "$cd/log.txt" 2>&1; then
      cp "$cd/log.txt" "$final/log.txt" 2>/dev/null; write_status "$final" failed 0 "$n" "feature_extractor failed"; return; fi
    # COLMAP >= 3.10 fisheye pairs need a focal prior on the camera row; drop images with (almost) no keypoints.
    python3 - "$cd/db.db" <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1]); c.execute("UPDATE cameras SET prior_focal_length=1")
weak = [r[0] for r in c.execute("SELECT i.image_id FROM images i LEFT JOIN keypoints k ON k.image_id = i.image_id WHERE k.rows IS NULL OR k.rows < 64")]
for iid in weak:
    for tbl in ("keypoints", "descriptors", "images"):
        c.execute(f"DELETE FROM {tbl} WHERE image_id = ?", (iid,))
c.commit(); print(f"[calib] dropped {len(weak)} images with < 64 keypoints", flush=True); c.close()
PY
    write_status "$final" running 0 "$n" "matching"
    if $COLMAP exhaustive_matcher --database_path "$cd/db.db" --SiftMatching.use_gpu 0 --SiftMatching.num_threads "$THREADS" >> "$cd/log.txt" 2>&1; then
      flann_ok=1
    fi
  fi
  if [ "$flann_ok" = "0" ]; then
    # COLMAP 3.11's FLANN matcher segfaults deterministically on about half of these videos (block [1/3,2/3]).
    # Fallback: pycolmap (faiss matcher, no FLANN) computes features + matches in its own database, then the
    # keypoints/descriptors/matches/geometries are transplanted into a COLMAP 3.11 database that GLOMAP 1.1 can read.
    if [ "$resume" = "0" ]; then
      log "$vid: FLANN matcher failed, falling back to pycolmap matching + transplant"
      write_status "$final" running 0 "$n" "matching (pycolmap fallback)"
      rm -f "$cd/db.db" "$cd/db.db-wal" "$cd/db.db-shm" "$cd/db_py.db"*
      if ! "$PYCOLMAP_PY" - "$cd/db_py.db" "$img" "$THREADS" >> "$cd/log.txt" 2>&1 <<'PY'
import sqlite3, sys
import pycolmap
db, img, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
eo = pycolmap.FeatureExtractionOptions(); eo.use_gpu = False; eo.num_threads = threads; eo.sift.max_num_features = 16384
ro = pycolmap.ImageReaderOptions(); ro.camera_model = "OPENCV_FISHEYE"
pycolmap.extract_features(db, img, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=eo)
c = sqlite3.connect(db); c.execute("UPDATE cameras SET prior_focal_length=1"); c.commit(); c.close()
mo = pycolmap.FeatureMatchingOptions(); mo.use_gpu = False; mo.num_threads = threads
pycolmap.match_exhaustive(db, matching_options=mo)
print("[calib] pycolmap features + matches done", flush=True)
PY
      then cp "$cd/log.txt" "$final/log.txt" 2>/dev/null; write_status "$final" failed 0 "$n" "pycolmap fallback failed"; return; fi
    fi
    write_status "$final" running 0 "$n" "transplanting matches"
    # a fresh 3.11 schema with the same images (tiny feature set, replaced by the transplant)
    $COLMAP feature_extractor --database_path "$cd/db.db" --image_path "$img" --ImageReader.single_camera 1 \
        --ImageReader.camera_model OPENCV_FISHEYE --SiftExtraction.use_gpu 0 --SiftExtraction.max_num_features 256 \
        --SiftExtraction.num_threads "$THREADS" >> "$cd/log.txt" 2>&1
    python3 - "$cd/db.db" <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1]); c.execute("UPDATE cameras SET prior_focal_length=1"); c.commit(); c.close()
PY
    if ! "$PYCOLMAP_PY" "$ROOT/starred_transplant_matches.py" "$cd/db_py.db" "$cd/db.db" >> "$cd/log.txt" 2>&1; then
      cp "$cd/log.txt" "$final/log.txt" 2>/dev/null; write_status "$final" failed 0 "$n" "match transplant failed"; return; fi
  fi
  write_status "$final" running 0 "$n" "glomap"
  rm -rf "$cd/glomap_sparse"; mkdir -p "$cd/glomap_sparse"
  if ! $GLOMAP mapper --database_path "$cd/db.db" --image_path "$img" --output_path "$cd/glomap_sparse" >> "$cd/log.txt" 2>&1; then
    cp "$cd/log.txt" "$final/log.txt" 2>/dev/null; write_status "$final" failed 0 "$n" "glomap mapper failed"; return; fi
  local model; model=$(ls -d "$cd"/glomap_sparse/*/ 2>/dev/null | head -1)
  [ -n "$model" ] || { write_status "$final" failed 0 "$n" "glomap produced no model"; return; }
  $COLMAP model_converter --input_path "$model" --output_path "$out" --output_type TXT >> "$cd/log.txt" 2>&1
  $COLMAP model_analyzer --path "$model" > "$out/analyzer.txt" 2>&1
  local reg; reg=$(grep -c -vE '^#' "$out/images.txt" | awk '{print int($1/2)}')
  local dt=$(( $(date +%s) - t0 ))
  mkdir -p "$final/glomap_fisheye"; cp "$out"/cameras.txt "$out"/images.txt "$out"/analyzer.txt "$final/glomap_fisheye/"; cp "$cd/log.txt" "$final/log.txt" 2>/dev/null || true
  log "$vid: registered $reg / $n images in ${dt}s; $(grep -E 'Mean reprojection|Registered' "$out/analyzer.txt" | tr '\n' ' ')"
  write_status "$final" succeeded "$reg" "$n" "glomap OPENCV_FISHEYE$([ "$flann_ok" = "0" ] && echo " via pycolmap transplant"), ${dt}s"
}

ensure_tools
if [ "$#" -gt 0 ]; then LIST=("$@"); else mapfile -t LIST < <(ls "$ROOT" | while read -r d; do [ -d "$ROOT/$d/images" ] && echo "$d"; done); fi
for vid in "${LIST[@]}"; do calibrate_one "$vid"; done
log "all done"
