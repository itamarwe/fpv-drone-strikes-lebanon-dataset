#!/bin/zsh
# Queue: wait for the Sainte-Maxime run, then prepare the 98872949 flight (8-40 s):
#   1. static-overlay mask (HUD, logo, blur boxes) from per-pixel temporal variance
#   2. SAM 3 building detections on every 2nd frame
#   3. COLMAP self-calibration + SfM (SIMPLE_RADIAL_FISHEYE, masked), the setup that worked on Bint Jbeil
set -e
HERE=/Users/itamarwe/Documents/code/constellation-wt/geolocation/angular_scan
S=$HERE/scenes/flight_98872949
source ~/.venvs/roof-retrieval/bin/activate
cd $HERE
echo "[queue $(date +%T)] waiting for the Sainte-Maxime run to finish"
while pgrep -f "run_scene.py sainte_maxime" >/dev/null; do sleep 20; done
echo "[queue $(date +%T)] 1/3 overlay mask"
python3 - <<'EOF'
import cv2, glob, numpy as np
S = "scenes/flight_98872949"
fs = sorted(glob.glob(f"{S}/frames/*.jpg"))
st = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fs[::2]])
sd = st.std(0); static = (sd < 8).astype(np.uint8)
static = cv2.dilate(cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((15, 15), np.uint8))
mask = (1 - static) * 255                      # COLMAP: 255 = use, 0 = ignore
import os; os.makedirs(f"{S}/masks", exist_ok=True)
cv2.imwrite(f"{S}/overlay_mask.png", mask)
for f in fs:
    cv2.imwrite(f"{S}/masks/{os.path.basename(f)}.png", mask)
print(f"static overlay pixels: {static.mean():.1%}")
EOF
echo "[queue $(date +%T)] 2/3 SAM 3 on every 2nd frame"
mkdir -p $S/sam_frames
i=0
for f in $S/frames/*.jpg; do
  i=$((i+1)); [[ $((i % 2)) -eq 1 ]] || continue
  b=$(basename $f .jpg)
  [[ -f $S/sam/$b/segments.json ]] && continue
  python3 -u segment_tiled.py $f $S/sam/$b --tile 1000 --overlap 200 2>&1 | grep -E "buildings after|Error|Traceback" || true
done
echo "[queue $(date +%T)] 3/3 COLMAP"
mkdir -p $S/colmap
colmap feature_extractor --database_path $S/colmap/db.db --image_path $S/frames --ImageReader.camera_mask_path $S/overlay_mask.png \
  --ImageReader.camera_model SIMPLE_RADIAL_FISHEYE --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0 2>&1 | tail -2
colmap sequential_matcher --database_path $S/colmap/db.db --FeatureMatching.use_gpu 0 --SequentialMatching.overlap 12 2>&1 | tail -2
mkdir -p $S/colmap/sparse
colmap mapper --database_path $S/colmap/db.db --image_path $S/frames --output_path $S/colmap/sparse 2>&1 | grep -E "Registering|Elapsed|images" | tail -5
for m in $S/colmap/sparse/*; do colmap model_analyzer --path $m 2>&1 | grep -E "Registered images|Mean reprojection|Points" ; done
echo "[queue $(date +%T)] flight prep done"
